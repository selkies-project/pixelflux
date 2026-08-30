//! `zwlr_output_management_v1` client: sets the output scale of a nested session
//! compositor, in-process rather than through a `wlr-randr` fork.
//!
//! Applications draw larger when the compositor they are on scales its own
//! output. Scaling pixelflux's capture output instead shrinks the logical size
//! the session is handed, which upscales the desktop rather than enlarging its
//! interface, so a DPI change for a nested session lands here.
//!
//! Compositors without the protocol (KWin offers `kde_output_management_v2`)
//! report [`ScaleOutcome::Unsupported`]; their scale comes from the capture
//! output, which they follow. Blocking, off the compositor thread, with every
//! round-trip deadline-bounded.

use std::os::unix::net::UnixStream;
use std::time::Instant;

use wayland_client::protocol::wl_registry;
use wayland_client::{delegate_noop, Connection, Dispatch, EventQueue, QueueHandle};
use wayland_protocols_wlr::output_management::v1::client::{
    zwlr_output_configuration_head_v1::ZwlrOutputConfigurationHeadV1,
    zwlr_output_configuration_v1::{self, ZwlrOutputConfigurationV1},
    zwlr_output_head_v1::{self, ZwlrOutputHeadV1},
    zwlr_output_manager_v1::{self, ZwlrOutputManagerV1},
    zwlr_output_mode_v1::ZwlrOutputModeV1,
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
    serial: Option<u32>,
    applied: Option<bool>,
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
            zwlr_output_configuration_v1::Event::Failed
            | zwlr_output_configuration_v1::Event::Cancelled => state.applied = Some(false),
            _ => {}
        }
    }
}

delegate_noop!(OutState: ignore ZwlrOutputModeV1);
delegate_noop!(OutState: ZwlrOutputConfigurationHeadV1);

/// Set the scale of the screen of the compositor on `socket_path`, leaving its
/// mode and position alone. Blocking; call off the compositor's calloop thread.
pub fn set_output_scale(socket_path: &str, scale: f64) -> Result<ScaleOutcome, String> {
    if !(0.1..=16.0).contains(&scale) {
        return Err(format!("scale {scale} out of range"));
    }
    configure(socket_path, |heads| {
        let target = heads.first().cloned().ok_or("the session has no enabled screen")?;
        Ok(vec![(target, Plan { scale: Some(scale), ..Plan::default() })])
    })
    .map(|changed| if changed == 0 { ScaleOutcome::Unsupported } else { ScaleOutcome::Applied })
}

/// Give the screen of the compositor on `socket_path` this mode and scale in one
/// configuration.
///
/// A session lays its desktop out once per applied configuration, so setting the
/// two separately leaves it briefly at a geometry that never exists — a screen
/// still carrying the pre-connect mode at the new scale is a fraction of its
/// final size, and a client that does not lay out again keeps that size.
pub fn set_screen_geometry(
    socket_path: &str,
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
        let target = heads.first().cloned().ok_or("the session has no enabled screen")?;
        Ok(vec![(target, Plan { mode: Some(size), scale: Some(scale), ..Plan::default() })])
    })
    .map(|changed| if changed == 0 { ScaleOutcome::Unsupported } else { ScaleOutcome::Applied })
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

/// Apply `plan` to the compositor's enabled heads. `plan` receives them in
/// announcement order and answers with the ones to change; heads it leaves out
/// keep their configuration. 0 = the compositor manages no outputs, or the plan
/// asked for nothing.
fn configure<F>(socket_path: &str, plan: F) -> Result<usize, String>
where
    F: FnOnce(&[ZwlrOutputHeadV1]) -> Result<Vec<(ZwlrOutputHeadV1, Plan)>, String>,
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
    let serial = state.serial.ok_or("output manager sent no state serial")?;
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
        _ => Err("the compositor refused the configuration".to_string()),
    }
}
