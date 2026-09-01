//! `zwlr_output_management_v1` client: scales and arranges the screens of a
//! nested session compositor, in-process rather than through a `wlr-randr` fork.
//!
//! Applications draw larger when the compositor they are on scales its own
//! output. Scaling pixelflux's capture output instead shrinks the logical size
//! the session is handed, which upscales the desktop rather than enlarging its
//! interface, so a DPI change for a nested session lands here.
//!
//! A session that opens one screen per capture output (a wlroots compositor
//! under `WLR_WL_OUTPUTS`) arranges them by its own rule — side by side in the
//! order they were opened — and that layout, not the capture one, is what
//! places its windows and carries its pointer between screens. So the capture
//! arrangement is mirrored into it (`set_screen_layout`), read back
//! (`list_screens`), and the screens no capture output drives yet are held at a
//! token size (`hold_spare_screens`) so the desktop does not lay out across a
//! screen nobody sees. Screens are addressed by position in name order
//! (WL-1, WL-2, ...), the order the session opened them in.
//!
//! Compositors without the protocol (KWin offers `kde_output_management_v2`)
//! report [`ScaleOutcome::Unsupported`] and position nothing; their scale comes
//! from the capture output, which they follow. Blocking, off the compositor
//! thread, with every round-trip deadline-bounded.

use std::os::unix::net::UnixStream;
use std::time::Instant;

use wayland_client::protocol::wl_registry;
use wayland_client::{delegate_noop, Connection, Dispatch, EventQueue, QueueHandle};
use wayland_protocols_wlr::output_management::v1::client::{
    zwlr_output_configuration_head_v1::ZwlrOutputConfigurationHeadV1,
    zwlr_output_configuration_v1::{self, ZwlrOutputConfigurationV1},
    zwlr_output_head_v1::{self, ZwlrOutputHeadV1},
    zwlr_output_manager_v1::{self, ZwlrOutputManagerV1},
    zwlr_output_mode_v1::{self, ZwlrOutputModeV1},
};

use crate::wayland::wlclient::{bounded_roundtrip, impl_sync_callback, SyncState, IO_TIMEOUT};

/// What a scale request did, from the caller's point of view.
pub enum ScaleOutcome {
    Applied,
    /// The compositor manages no outputs for clients: scale it another way.
    Unsupported,
}

#[derive(Default)]
struct OutState {
    manager: Option<ZwlrOutputManagerV1>,
    /// Announced heads with their name, enabled state and layout position. The
    /// manager's order is its own; screens are addressed by name below.
    heads: Vec<(ZwlrOutputHeadV1, Option<String>, bool, (i32, i32))>,
    /// Size per announced mode object, and the mode each head currently holds:
    /// a head carries no size of its own, so the two are joined to report one.
    modes: Vec<(ZwlrOutputModeV1, (i32, i32))>,
    current: Vec<(ZwlrOutputHeadV1, ZwlrOutputModeV1)>,
    serial: Option<u32>,
    applied: Option<bool>,
    /// The refusal was a cancellation: the configuration went stale under a
    /// state change of the compositor's own, so the same plan on a fresh
    /// serial can still land.
    cancelled: bool,
    sync_done: bool,
}

impl SyncState for OutState {
    fn sync_done_mut(&mut self) -> &mut bool {
        &mut self.sync_done
    }
}
impl_sync_callback!(OutState);

impl Dispatch<wl_registry::WlRegistry, ()> for OutState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, version } = event {
            if interface == "zwlr_output_manager_v1" && state.manager.is_none() {
                state.manager = Some(registry.bind(name, version.min(4), qh, ()));
            }
        }
    }
}

impl Dispatch<ZwlrOutputManagerV1, ()> for OutState {
    fn event(
        state: &mut Self,
        _: &ZwlrOutputManagerV1,
        event: zwlr_output_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_output_manager_v1::Event::Head { head } => {
                state.heads.push((head, None, false, (0, 0)))
            }
            // Every configuration is built against the serial of the state it
            // was read from; a stale one is refused by the compositor.
            zwlr_output_manager_v1::Event::Done { serial } => state.serial = Some(serial),
            _ => {}
        }
    }

    wayland_client::event_created_child!(OutState, ZwlrOutputManagerV1, [
        zwlr_output_manager_v1::EVT_HEAD_OPCODE => (ZwlrOutputHeadV1, ()),
    ]);
}

impl Dispatch<ZwlrOutputHeadV1, ()> for OutState {
    fn event(
        state: &mut Self,
        head: &ZwlrOutputHeadV1,
        event: zwlr_output_head_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let Some(entry) = state.heads.iter_mut().find(|(h, _, _, _)| h == head) else {
            return;
        };
        match event {
            zwlr_output_head_v1::Event::Name { name } => entry.1 = Some(name),
            zwlr_output_head_v1::Event::Enabled { enabled } => entry.2 = enabled != 0,
            zwlr_output_head_v1::Event::Position { x, y } => entry.3 = (x, y),
            zwlr_output_head_v1::Event::CurrentMode { mode } => {
                let head = head.clone();
                state.current.retain(|(h, _)| *h != head);
                state.current.push((head, mode));
            }
            _ => {}
        }
    }

    wayland_client::event_created_child!(OutState, ZwlrOutputHeadV1, [
        zwlr_output_head_v1::EVT_MODE_OPCODE => (ZwlrOutputModeV1, ()),
    ]);
}

impl Dispatch<ZwlrOutputConfigurationV1, ()> for OutState {
    fn event(
        state: &mut Self,
        _: &ZwlrOutputConfigurationV1,
        event: zwlr_output_configuration_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_output_configuration_v1::Event::Succeeded => state.applied = Some(true),
            zwlr_output_configuration_v1::Event::Failed => state.applied = Some(false),
            zwlr_output_configuration_v1::Event::Cancelled => {
                state.cancelled = true;
                state.applied = Some(false);
            }
            _ => {}
        }
    }
}

impl Dispatch<ZwlrOutputModeV1, ()> for OutState {
    fn event(
        state: &mut Self,
        mode: &ZwlrOutputModeV1,
        event: zwlr_output_mode_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwlr_output_mode_v1::Event::Size { width, height } = event {
            let mode = mode.clone();
            state.modes.retain(|(m, _)| *m != mode);
            state.modes.push((mode, (width, height)));
        }
    }
}
delegate_noop!(OutState: ZwlrOutputConfigurationHeadV1);

/// Set the scale of the `index`-th screen of the compositor on `socket_path`,
/// leaving its mode and position alone. Blocking; call off the compositor's
/// calloop thread.
pub fn set_output_scale(
    socket_path: &str,
    index: usize,
    scale: f64,
) -> Result<ScaleOutcome, String> {
    if !(0.1..=16.0).contains(&scale) {
        return Err(format!("scale {scale} out of range"));
    }
    configure(socket_path, |heads| {
        let target = heads
            .get(index)
            .cloned()
            .ok_or_else(|| format!("no enabled screen at index {index}"))?;
        Ok(vec![(target, Plan { scale: Some(scale), ..Plan::default() })])
    })
    .map(|changed| if changed == 0 { ScaleOutcome::Unsupported } else { ScaleOutcome::Applied })
}

/// Give the `index`-th screen of the compositor on `socket_path` this mode and
/// scale in one configuration.
///
/// A session lays its desktop out once per applied configuration, so setting the
/// two separately leaves it briefly at a geometry that never exists — a screen
/// still carrying the pre-connect mode at the new scale is a fraction of its
/// final size, and a client that does not lay out again keeps that size.
pub fn set_screen_geometry(
    socket_path: &str,
    index: usize,
    size: (i32, i32),
    scale: f64,
) -> Result<ScaleOutcome, String> {
    if !(0.1..=16.0).contains(&scale) {
        return Err(format!("scale {scale} out of range"));
    }
    if size.0 <= 0 || size.1 <= 0 {
        return Err(format!("size {}x{} out of range", size.0, size.1));
    }
    configure(socket_path, move |heads| {
        let target = heads
            .get(index)
            .cloned()
            .ok_or_else(|| format!("no enabled screen at index {index}"))?;
        Ok(vec![(target, Plan { mode: Some(size), scale: Some(scale), ..Plan::default() })])
    })
    .map(|changed| if changed == 0 { ScaleOutcome::Unsupported } else { ScaleOutcome::Applied })
}

/// The session's enabled screens as `(name, x, y, width, height)`, in screen
/// order — what the compositor actually did with a layout and a mode, which is
/// not always what it was asked for. A head carries no size itself, so the size
/// is its current mode's; `(0, 0)` where it announced none. Empty when the
/// compositor manages no outputs for clients.
pub fn list_screens(socket_path: &str) -> Result<Vec<(String, i32, i32, i32, i32)>, String> {
    let stream =
        UnixStream::connect(socket_path).map_err(|e| format!("connect {socket_path}: {e}"))?;
    let conn = Connection::from_socket(stream).map_err(|e| format!("wayland setup: {e}"))?;
    let mut queue: EventQueue<OutState> = conn.new_event_queue();
    let qh = queue.handle();
    let _registry = conn.display().get_registry(&qh, ());
    let mut state = OutState::default();
    bounded_roundtrip(&conn, &mut queue, &mut state)?;
    let Some(manager) = state.manager.clone() else {
        return Ok(Vec::new());
    };
    bounded_roundtrip(&conn, &mut queue, &mut state)?;
    let size_of = |head: &ZwlrOutputHeadV1| -> (i32, i32) {
        state
            .current
            .iter()
            .find(|(h, _)| h == head)
            .and_then(|(_, mode)| state.modes.iter().find(|(m, _)| m == mode))
            .map(|(_, size)| *size)
            .unwrap_or((0, 0))
    };
    let mut screens: Vec<(String, i32, i32, i32, i32)> = state
        .heads
        .iter()
        .filter(|(_, _, on, _)| *on)
        .map(|(head, name, _, pos)| {
            let (w, h) = size_of(head);
            (name.clone().unwrap_or_default(), pos.0, pos.1, w, h)
        })
        .collect();
    screens.sort_by_key(|(name, _, _, _, _)| (trailing_number(name), name.clone()));
    manager.stop();
    let _ = queue.flush();
    Ok(screens)
}

/// Lay the session's screens out at `rects`, one `(x, y, width, height)` per
/// screen in screen order, so its own layout matches the arrangement the capture
/// outputs were placed in.
///
/// A session compositor arranges the screens it opens by its own rule — wlroots
/// puts them side by side in the order they appear — and that layout, not the
/// capture one, is what places windows and carries the pointer between screens.
/// Every screen is positioned in one configuration: applied one at a time, an
/// intermediate state overlaps two screens and the compositor reflows around it.
///
/// Returns how many screens were positioned; 0 when the compositor manages no
/// outputs for clients, which is where a session on KWin lands (it offers
/// `kde_output_management_v2` instead) and where the arrangement stays whatever
/// that compositor chose.
pub fn set_screen_layout(
    socket_path: &str,
    rects: Vec<(i32, i32, i32, i32)>,
) -> Result<usize, String> {
    if rects.iter().any(|(_, _, w, h)| *w <= 0 || *h <= 0) {
        return Err("a screen rectangle has a non-positive size".to_string());
    }
    configure(socket_path, move |heads| {
        Ok(heads
            .iter()
            .zip(rects.iter())
            .map(|(h, (x, y, w, hgt))| {
                (h.clone(), Plan { mode: Some((*w, *hgt)), position: Some((*x, *y)), ..Plan::default() })
            })
            .collect())
    })
}

/// Hold every screen past the first `keep` at `size`. A session compositor opens
/// the screens it was started with whether or not anything watches them, and one
/// held at a real screen's size stretches the session's coordinate space onto a
/// screen nobody sees. Returns how many were resized.
pub fn hold_spare_screens(
    socket_path: &str,
    keep: usize,
    size: (i32, i32),
) -> Result<usize, String> {
    configure(socket_path, move |heads| {
        Ok(heads
            .iter()
            .skip(keep)
            .cloned()
            .map(|h| (h, Plan { mode: Some(size), ..Plan::default() }))
            .collect())
    })
}

/// The number a screen's name ends in (WL-2 -> 2), or none, which sorts first.
fn trailing_number(name: &str) -> u32 {
    let digits: String = name.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
    digits.chars().rev().collect::<String>().parse().unwrap_or(0)
}

/// What a head is being asked to change; an unset field keeps its current value.
#[derive(Clone, Copy, Default)]
struct Plan {
    mode: Option<(i32, i32)>,
    scale: Option<f64>,
    position: Option<(i32, i32)>,
}

/// Why a configuration did not land.
enum ConfigErr {
    /// The compositor cancelled it, which says only that its own state moved
    /// while the configuration was alive. The serial is what went stale, not
    /// the plan.
    Stale,
    Other(String),
}

impl From<String> for ConfigErr {
    fn from(e: String) -> Self {
        ConfigErr::Other(e)
    }
}

/// Rebuilds a cancelled configuration this many times before giving up. A
/// screen arriving or leaving cancels whatever is in flight, and that is the
/// same moment a display's mode and scale are being applied.
const CONFIGURE_ATTEMPTS: usize = 4;

/// Apply `plan` to the compositor's enabled heads, retrying while the
/// compositor cancels. `plan` receives them in announcement order and answers
/// with the ones to change; heads it leaves out keep their configuration. It is
/// called once per attempt, against the heads of that attempt. 0 = the
/// compositor manages no outputs, or the plan asked for nothing.
fn configure<F>(socket_path: &str, plan: F) -> Result<usize, String>
where
    F: Fn(&[ZwlrOutputHeadV1]) -> Result<Vec<(ZwlrOutputHeadV1, Plan)>, String>,
{
    let mut stale = 0;
    for _ in 0..CONFIGURE_ATTEMPTS {
        match configure_once(socket_path, &plan) {
            Ok(applied) => return Ok(applied),
            Err(ConfigErr::Other(e)) => return Err(e),
            Err(ConfigErr::Stale) => stale += 1,
        }
    }
    Err(format!("the compositor cancelled the configuration {stale} times"))
}

/// One attempt: a connection of its own, the heads and the serial that
/// stamps them, and `plan` applied against that state.
fn configure_once<F>(socket_path: &str, plan: &F) -> Result<usize, ConfigErr>
where
    F: Fn(&[ZwlrOutputHeadV1]) -> Result<Vec<(ZwlrOutputHeadV1, Plan)>, String>,
{
    let stream =
        UnixStream::connect(socket_path).map_err(|e| format!("connect {socket_path}: {e}"))?;
    let conn = Connection::from_socket(stream).map_err(|e| format!("wayland setup: {e}"))?;
    let mut queue: EventQueue<OutState> = conn.new_event_queue();
    let qh = queue.handle();
    let _registry = conn.display().get_registry(&qh, ());
    let mut state = OutState::default();
    bounded_roundtrip(&conn, &mut queue, &mut state)?;
    let Some(manager) = state.manager.clone() else {
        return Ok(0);
    };
    // The heads and the serial that stamps them arrive after the bind.
    bounded_roundtrip(&conn, &mut queue, &mut state)?;
    let serial = state
        .serial
        .ok_or_else(|| "output manager sent no state serial".to_string())?;
    // Screens in the order their names put them (WL-1, WL-2, ...), which is the
    // order a session opens them in and so the order they map to displays; the
    // manager's own announcement order carries no such promise.
    let mut named: Vec<(ZwlrOutputHeadV1, String)> = state
        .heads
        .iter()
        .filter(|(_, _, on, _)| *on)
        .map(|(h, name, _, _)| (h.clone(), name.clone().unwrap_or_default()))
        .collect();
    named.sort_by_key(|(_, name)| (trailing_number(name), name.clone()));
    let enabled: Vec<ZwlrOutputHeadV1> = named.into_iter().map(|(h, _)| h).collect();
    let wanted = plan(&enabled)?;
    if wanted.is_empty() {
        manager.stop();
        let _ = queue.flush();
        return Ok(0);
    }

    // A configuration describes every head: one left out would be disabled.
    let config = manager.create_configuration(serial, &qh, ());
    for head in &enabled {
        let cfg_head = config.enable_head(head, &qh, ());
        if let Some((_, want)) = wanted.iter().find(|(h, _)| h == head) {
            if let Some((w, h)) = want.mode {
                cfg_head.set_custom_mode(w, h, 0);
            }
            if let Some(scale) = want.scale {
                cfg_head.set_scale(scale);
            }
            if let Some((x, y)) = want.position {
                cfg_head.set_position(x, y);
            }
        }
    }
    config.apply();
    queue.flush().map_err(|e| format!("flush configuration: {e}"))?;
    state.applied = None;
    let deadline = Instant::now() + IO_TIMEOUT;
    while state.applied.is_none() && Instant::now() < deadline {
        bounded_roundtrip(&conn, &mut queue, &mut state)?;
    }
    config.destroy();
    manager.stop();
    let _ = queue.flush();
    match state.applied {
        Some(true) => Ok(wanted.len()),
        _ if state.cancelled => Err(ConfigErr::Stale),
        _ => Err(ConfigErr::Other(
            "the compositor refused the configuration".to_string(),
        )),
    }
}
