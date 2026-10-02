/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! xdg-desktop-portal client: one `RemoteDesktop` (or `ScreenCast`) session on the session bus
//! that hands out the PipeWire streams of the host's monitors and takes input events for its
//! seat. Compositors without the privileged wlroots protocols (GNOME, KDE) expose both only this
//! way, through their portal backend, so this is the rung host capture falls to when the registry
//! lacks capture or virtual-input globals.
//!
//! Every portal call that ends in a `Response` signal is made through [`PortalSession::request`],
//! which subscribes to the request object the portal will answer on before calling, as the
//! handle-token convention requires; a consent dialog the backend shows blocks that call until
//! the user answers. Input notifications are fire-and-forget method calls, so an event never
//! waits on the bus round trip. A restore token the backend hands back is kept under the
//! user's state directory, so a session reopened to change the cursor mode, and every later
//! start on this machine, needs no second consent; a backend that hands out none (KDE) shows
//! no dialog to an unsandboxed app in the first place.

use std::collections::HashMap;
use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicU32, Ordering};

use zbus::blocking::{Connection, Proxy};
use zbus::zvariant::{OwnedObjectPath, OwnedValue, Value};

const DESKTOP: &str = "org.freedesktop.portal.Desktop";
const DESKTOP_PATH: &str = "/org/freedesktop/portal/desktop";
const REQUEST_IFACE: &str = "org.freedesktop.portal.Request";
const SESSION_IFACE: &str = "org.freedesktop.portal.Session";
const SCREENCAST_IFACE: &str = "org.freedesktop.portal.ScreenCast";
const REMOTE_DESKTOP_IFACE: &str = "org.freedesktop.portal.RemoteDesktop";

pub const DEVICE_KEYBOARD: u32 = 1;
pub const DEVICE_POINTER: u32 = 2;
pub const CURSOR_HIDDEN: u32 = 1;
pub const CURSOR_EMBEDDED: u32 = 2;
pub const CURSOR_METADATA: u32 = 4;
const SOURCE_MONITOR: u32 = 1;
const PERSIST_UNTIL_REVOKED: u32 = 2;

type Options<'a> = HashMap<&'a str, Value<'a>>;

/// One monitor stream the portal started: its PipeWire node and, where the backend says,
/// its logical position and size in the compositor's coordinate space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortalStream {
    pub node_id: u32,
    pub position: Option<(i32, i32)>,
    pub size: Option<(i32, i32)>,
}

/// A started portal session: its streams, the devices it may drive, and the token that
/// restores it without a dialog.
pub struct PortalSession {
    conn: Connection,
    screencast: Proxy<'static>,
    remote: Option<Proxy<'static>>,
    session: OwnedObjectPath,
    pub streams: Vec<PortalStream>,
    pub devices: u32,
    pub cursor_mode: u32,
    pub restore_token: Option<String>,
    /// RemoteDesktop interface version, 0 when the session drives no input. `ConnectToEIS`
    /// exists from version 2.
    pub rd_version: u32,
    /// KDE's backend serves the session: it reads a smooth scroll's vertical axis negated.
    kde: bool,
}

fn opt_u32(o: &mut Options<'_>, key: &'static str, v: u32) {
    o.insert(key, Value::U32(v));
}

/// Where the portal's restore token lives between runs: `$XDG_STATE_HOME/pixelflux`, or
/// `~/.local/state/pixelflux` in its absence.
fn token_path() -> Option<std::path::PathBuf> {
    let base = match std::env::var_os("XDG_STATE_HOME") {
        Some(dir) if !dir.is_empty() => std::path::PathBuf::from(dir),
        _ => std::path::PathBuf::from(std::env::var_os("HOME")?).join(".local/state"),
    };
    Some(base.join("pixelflux/portal-restore-token"))
}

/// The restore token saved by an earlier session, if any.
pub fn saved_restore_token() -> Option<String> {
    let token = std::fs::read_to_string(token_path()?).ok()?;
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_string())
}

/// Keep a restore token for the next session (a token is single-use, so the newest replaces
/// the one that opened this session).
pub fn save_restore_token(token: &str) {
    let Some(path) = token_path() else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = std::fs::write(&path, token) {
        eprintln!(
            "[HostCapture] portal restore token not saved to {}: {e}",
            path.display()
        );
    }
}

fn value_i32(v: &Value<'_>) -> Option<i32> {
    match v {
        Value::I32(x) => Some(*x),
        Value::U32(x) => Some(*x as i32),
        Value::Value(inner) => value_i32(inner),
        _ => None,
    }
}

fn value_pair(v: &Value<'_>) -> Option<(i32, i32)> {
    match v {
        Value::Structure(s) if s.fields().len() == 2 => {
            Some((value_i32(&s.fields()[0])?, value_i32(&s.fields()[1])?))
        }
        Value::Value(inner) => value_pair(inner),
        _ => None,
    }
}

/// The `a(ua{sv})` stream list of a `Start` response.
fn parse_streams(v: &Value<'_>) -> Vec<PortalStream> {
    let mut out = Vec::new();
    let Value::Array(arr) = v else { return out };
    for item in arr.iter() {
        let Value::Structure(s) = item else { continue };
        let fields = s.fields();
        let Some(Value::U32(node_id)) = fields.first() else {
            continue;
        };
        let mut stream = PortalStream {
            node_id: *node_id,
            position: None,
            size: None,
        };
        if let Some(Value::Dict(d)) = fields.get(1) {
            for (k, val) in d.iter() {
                let Value::Str(k) = k else { continue };
                match k.as_str() {
                    "position" => stream.position = value_pair(val),
                    "size" => stream.size = value_pair(val),
                    _ => {}
                }
            }
        }
        out.push(stream);
    }
    out
}

impl PortalSession {
    /// Open and start a session. `devices` (`DEVICE_*` bits) selects what input the session
    /// drives — zero opens a plain ScreenCast session; `capture` asks for every monitor as its
    /// own stream with `cursor_mode` (`CURSOR_*`), downgraded to what the portal offers. The
    /// call returns once the portal (and any dialog it shows) has answered.
    pub fn open(
        capture: bool,
        devices: u32,
        cursor_mode: u32,
        restore_token: Option<&str>,
    ) -> Result<Self, String> {
        let conn = Connection::session().map_err(|e| format!("session bus: {e}"))?;
        let screencast = Proxy::new(&conn, DESKTOP, DESKTOP_PATH, SCREENCAST_IFACE)
            .map_err(|e| e.to_string())?;
        let remote = (devices != 0)
            .then(|| {
                Proxy::new(&conn, DESKTOP, DESKTOP_PATH, REMOTE_DESKTOP_IFACE)
                    .map_err(|e| e.to_string())
            })
            .transpose()?;
        let sc_version: u32 = screencast
            .get_property("version")
            .map_err(|e| format!("no ScreenCast portal: {e}"))?;
        let mut this = Self {
            conn,
            screencast,
            remote,
            session: OwnedObjectPath::default(),
            streams: Vec::new(),
            devices: 0,
            cursor_mode,
            restore_token: None,
            rd_version: 0,
            kde: false,
        };
        let opener = this
            .remote
            .clone()
            .unwrap_or_else(|| this.screencast.clone());
        let rd_version: u32 = match &this.remote {
            Some(rd) => rd
                .get_property("version")
                .map_err(|e| format!("no RemoteDesktop portal: {e}"))?,
            None => 0,
        };
        this.rd_version = rd_version;
        let mut options = Options::new();
        options.insert(
            "session_handle_token",
            Value::from(format!("pixelflux{}", std::process::id())),
        );
        let (_, results) = this.request("CreateSession", options, |o| {
            opener.call("CreateSession", &(o,))
        })?;
        let session = results
            .get("session_handle")
            .and_then(|v| match &**v {
                Value::Str(s) => OwnedObjectPath::try_from(s.as_str()).ok(),
                Value::ObjectPath(p) => Some(p.clone().into()),
                _ => None,
            })
            .ok_or("CreateSession returned no session handle")?;
        this.session = session;

        // Persistence rides SelectDevices on a remote-desktop session and SelectSources on a
        // plain screencast one; a backend without the option ignores it.
        let persist_on_sources = this.remote.is_none() && sc_version >= 4;
        let persist_on_devices = this.remote.is_some() && rd_version >= 2;
        if capture {
            let available: u32 = this
                .screencast
                .get_property("AvailableCursorModes")
                .unwrap_or(CURSOR_EMBEDDED);
            if available & cursor_mode == 0 {
                this.cursor_mode = [CURSOR_EMBEDDED, CURSOR_HIDDEN, CURSOR_METADATA]
                    .into_iter()
                    .find(|m| available & m != 0)
                    .unwrap_or(CURSOR_EMBEDDED);
            }
            let mut options = Options::new();
            opt_u32(&mut options, "types", SOURCE_MONITOR);
            options.insert("multiple", Value::Bool(true));
            if sc_version >= 2 {
                opt_u32(&mut options, "cursor_mode", this.cursor_mode);
            }
            if persist_on_sources {
                opt_u32(&mut options, "persist_mode", PERSIST_UNTIL_REVOKED);
                if let Some(t) = restore_token {
                    options.insert("restore_token", Value::from(t.to_string()));
                }
            }
            let (sc, session) = (this.screencast.clone(), this.session.clone());
            this.request("SelectSources", options, |o| {
                sc.call("SelectSources", &(&session, o))
            })?;
        }
        if let Some(rd) = this.remote.clone() {
            let available: u32 = rd
                .get_property("AvailableDeviceTypes")
                .unwrap_or(DEVICE_KEYBOARD | DEVICE_POINTER);
            let mut options = Options::new();
            opt_u32(&mut options, "types", devices & available);
            if persist_on_devices {
                opt_u32(&mut options, "persist_mode", PERSIST_UNTIL_REVOKED);
                if let Some(t) = restore_token {
                    options.insert("restore_token", Value::from(t.to_string()));
                }
            }
            let session = this.session.clone();
            this.request("SelectDevices", options, |o| {
                rd.call("SelectDevices", &(&session, o))
            })?;
        }
        let session = this.session.clone();
        let (_, results) = this.request("Start", Options::new(), |o| {
            opener.call("Start", &(&session, "", o))
        })?;
        if let Some(v) = results.get("streams") {
            this.streams = parse_streams(v);
        }
        if let Some(Value::U32(d)) = results.get("devices").map(|v| &**v) {
            this.devices = *d;
        }
        if let Some(Value::Str(t)) = results.get("restore_token").map(|v| &**v) {
            this.restore_token = Some(t.to_string());
            save_restore_token(t);
        }
        if capture && this.streams.is_empty() {
            return Err("the portal started no monitor stream".into());
        }
        // The backend has answered Start, so it runs by now; the frontend names none.
        this.kde = this.remote.is_some()
            && zbus::blocking::fdo::DBusProxy::new(&this.conn)
                .and_then(|bus| {
                    bus.name_has_owner(
                        "org.freedesktop.impl.portal.desktop.kde"
                            .try_into()
                            .map_err(zbus::Error::from)?,
                    )
                    .map_err(zbus::Error::from)
                })
                .unwrap_or(false);
        Ok(this)
    }

    /// Whether a ScreenCast portal answers on the session bus; the error says which step
    /// did not, since a bus nobody listens on and a portal without a backend read the same.
    pub fn probe() -> Result<(), String> {
        let conn = Connection::session().map_err(|e| format!("session bus: {e}"))?;
        let screencast = Proxy::new(&conn, DESKTOP, DESKTOP_PATH, SCREENCAST_IFACE)
            .map_err(|e| e.to_string())?;
        screencast
            .get_property::<u32>("version")
            .map(|_| ())
            .map_err(|e| format!("no ScreenCast portal: {e}"))
    }

    /// A method whose result arrives as a `Response` on a request object: subscribe to the
    /// request path the handle token names before the call `call` makes with the completed
    /// options, then wait for the answer.
    fn request(
        &self,
        method: &str,
        mut options: Options<'_>,
        call: impl FnOnce(&Options<'_>) -> zbus::Result<OwnedObjectPath>,
    ) -> Result<(u32, HashMap<String, OwnedValue>), String> {
        static COUNTER: AtomicU32 = AtomicU32::new(1);
        let token = format!(
            "pixelflux{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let sender = self
            .conn
            .unique_name()
            .ok_or("bus connection has no unique name")?
            .as_str()
            .to_string();
        let path = request_path(&sender, &token);
        options.insert("handle_token", Value::from(token.clone()));
        let request = Proxy::new(&self.conn, DESKTOP, path.as_str(), REQUEST_IFACE)
            .map_err(|e| e.to_string())?;
        let mut responses = request
            .receive_signal("Response")
            .map_err(|e| e.to_string())?;
        let handle = call(&options).map_err(|e| format!("{method}: {e}"))?;
        if handle.as_str() != path {
            // A portal predating handle tokens answers on the path it returned.
            let request = Proxy::new(&self.conn, DESKTOP, handle.as_str(), REQUEST_IFACE)
                .map_err(|e| e.to_string())?;
            responses = request
                .receive_signal("Response")
                .map_err(|e| e.to_string())?;
        }
        let msg = responses
            .next()
            .ok_or_else(|| format!("{method}: the request closed without a response"))?;
        let (code, results): (u32, HashMap<String, OwnedValue>) = msg
            .body()
            .deserialize()
            .map_err(|e| format!("{method}: {e}"))?;
        match code {
            0 => Ok((code, results)),
            1 => Err(format!("{method}: the user canceled the portal dialog")),
            _ => Err(format!("{method}: the portal refused (response {code})")),
        }
    }

    /// Whether this session can hand out an EIS socket: a RemoteDesktop session (input was
    /// granted) on a backend new enough to carry `ConnectToEIS`.
    pub fn eis_capable(&self) -> bool {
        self.remote.is_some() && self.devices != 0 && self.rd_version >= 2
    }

    /// The EIS socket for this session's seat. A backend that answers hands input over libei
    /// from here on and refuses the `Notify*` methods, so this is called once, at open, and
    /// only when the injector is committed to.
    pub fn connect_to_eis(&self) -> Result<OwnedFd, String> {
        let rd = self.remote.as_ref().ok_or("session drives no input")?;
        let fd: zbus::zvariant::OwnedFd = rd
            .call("ConnectToEIS", &(self.session.clone(), Options::new()))
            .map_err(|e| format!("ConnectToEIS: {e}"))?;
        Ok(fd.into())
    }

    /// The PipeWire connection the streams live on, opened by the portal with the node
    /// permissions of this session.
    pub fn open_pipewire_remote(&self) -> Result<OwnedFd, String> {
        let fd: zbus::zvariant::OwnedFd = self
            .screencast
            .call(
                "OpenPipeWireRemote",
                &(self.session.clone(), Options::new()),
            )
            .map_err(|e| format!("OpenPipeWireRemote: {e}"))?;
        Ok(fd.into())
    }

    fn notify<B>(&self, method: &str, args: &B)
    where
        B: serde::Serialize + zbus::zvariant::DynamicType,
    {
        if let Some(rd) = &self.remote {
            let _ = rd.call_noreply(method, args);
        }
    }

    /// Absolute pointer motion in the logical coordinate space of the stream `node`.
    pub fn pointer_motion_abs(&self, node: u32, x: f64, y: f64) {
        self.notify(
            "NotifyPointerMotionAbsolute",
            &(self.session.clone(), Options::new(), node, x, y),
        );
    }

    pub fn pointer_motion(&self, dx: f64, dy: f64) {
        self.notify(
            "NotifyPointerMotion",
            &(self.session.clone(), Options::new(), dx, dy),
        );
    }

    /// Button by evdev code.
    pub fn pointer_button(&self, button: i32, pressed: bool) {
        self.notify(
            "NotifyPointerButton",
            &(self.session.clone(), Options::new(), button, pressed as u32),
        );
    }

    /// Smooth scroll by logical pixels, positive down and right as on `wl_pointer`; `finish`
    /// closes the scroll series. KDE's backend negates the vertical axis it is handed and drops
    /// `finish` (xdg-desktop-portal-kde's `requestPointerAxis`, 5.27 through master), where
    /// GNOME's passes both on, so it is handed the axis negated.
    pub fn pointer_axis(&self, dx: f64, dy: f64, finish: bool) {
        let mut options = Options::new();
        options.insert("finish", Value::Bool(finish));
        let dy = if self.kde { -dy } else { dy };
        self.notify(
            "NotifyPointerAxis",
            &(self.session.clone(), options, dx, dy),
        );
    }

    /// Wheel steps: `axis` 0 vertical, 1 horizontal.
    pub fn pointer_axis_discrete(&self, axis: u32, steps: i32) {
        self.notify(
            "NotifyPointerAxisDiscrete",
            &(self.session.clone(), Options::new(), axis, steps),
        );
    }

    pub fn keysym(&self, keysym: i32, pressed: bool) {
        self.notify(
            "NotifyKeyboardKeysym",
            &(self.session.clone(), Options::new(), keysym, pressed as u32),
        );
    }

    /// Key by evdev keycode, in the host's own layout.
    pub fn keycode(&self, keycode: i32, pressed: bool) {
        self.notify(
            "NotifyKeyboardKeycode",
            &(
                self.session.clone(),
                Options::new(),
                keycode,
                pressed as u32,
            ),
        );
    }
}

impl Drop for PortalSession {
    fn drop(&mut self) {
        if let Ok(session) = Proxy::new(&self.conn, DESKTOP, self.session.as_str(), SESSION_IFACE) {
            let _ = session.call_noreply("Close", &());
        }
    }
}

/// The request object the portal answers a call from `sender` with `token` on.
fn request_path(sender: &str, token: &str) -> String {
    format!(
        "{DESKTOP_PATH}/request/{}/{token}",
        sender.trim_start_matches(':').replace('.', "_")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use zbus::zvariant::{Array, Dict, Structure, StructureBuilder};

    fn stream_value(node: u32, props: Vec<(&'static str, Value<'static>)>) -> Value<'static> {
        let mut dict = Dict::new(<&str>::SIGNATURE, Value::SIGNATURE);
        for (k, v) in props {
            dict.append(Value::from(k), Value::Value(Box::new(v)))
                .unwrap();
        }
        let st: Structure = StructureBuilder::new()
            .add_field(node)
            .append_field(Value::Dict(dict))
            .build()
            .unwrap();
        Value::Structure(st)
    }

    use zbus::zvariant::Type as _;

    /// A `Start` response's stream list parses into node ids with the optional geometry, as
    /// KDE (size only) and GNOME (position and size) send it.
    #[test]
    fn streams_parse_with_and_without_geometry() {
        let kde = stream_value(
            113,
            vec![
                ("size", Value::from((1280i32, 720i32))),
                ("source_type", Value::U32(1)),
            ],
        );
        let gnome = stream_value(
            42,
            vec![
                ("position", Value::from((1920i32, 0i32))),
                ("size", Value::from((1280i32, 720i32))),
            ],
        );
        let mut arr = Array::new(&kde.value_signature().clone());
        arr.append(kde).unwrap();
        arr.append(gnome).unwrap();
        let streams = parse_streams(&Value::Array(arr));
        assert_eq!(
            streams,
            vec![
                PortalStream {
                    node_id: 113,
                    position: None,
                    size: Some((1280, 720))
                },
                PortalStream {
                    node_id: 42,
                    position: Some((1920, 0)),
                    size: Some((1280, 720))
                },
            ]
        );
    }

    /// The request path follows the handle-token convention: the sender's unique name with
    /// its colon dropped and dots turned to underscores, then the token.
    #[test]
    fn request_path_follows_the_handle_token_convention() {
        assert_eq!(
            request_path(":1.42", "pixelflux7_3"),
            "/org/freedesktop/portal/desktop/request/1_42/pixelflux7_3"
        );
    }
}
