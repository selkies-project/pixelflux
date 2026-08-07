//! `zwlr_data_control_v1` CLIENT: the native clipboard bridge to a NESTED app
//! compositor (labwc/kwin running under pixelflux).
//!
//! Apps in a nested session use the inner compositor's selection, which
//! pixelflux's own clipboard machinery never sees. selkies bridges it over the
//! ScreenCapture ABI backed by this module instead of forking wl-copy/wl-paste
//! per operation: one-shot [`list_types`]/[`read`], a persistent-source [`write`]
//! (a detached thread serves paste requests until another client takes the
//! selection), and [`watch`] (a thread reporting selection changes to a Python
//! callback). Every compositor round-trip is deadline-bounded via
//! [`wlclient::bounded_roundtrip`].

use std::collections::HashMap;
use std::os::fd::AsFd;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pyo3::{Py, PyAny, Python};
use wayland_client::backend::ObjectId;
use wayland_client::protocol::{wl_registry, wl_seat};
use wayland_client::{delegate_noop, Connection, Dispatch, Proxy, QueueHandle};
use wayland_protocols_wlr::data_control::v1::client::{
    zwlr_data_control_device_v1::{self, ZwlrDataControlDeviceV1},
    zwlr_data_control_manager_v1::ZwlrDataControlManagerV1,
    zwlr_data_control_offer_v1::{self, ZwlrDataControlOfferV1},
    zwlr_data_control_source_v1::{self, ZwlrDataControlSourceV1},
};

use crate::wayland::wlclient::{
    bounded_roundtrip, impl_sync_callback, pipe_cloexec, read_fd_to_end, wait_readable,
    write_fd_all, SyncState, IO_TIMEOUT,
};

/// How often a background thread wakes from its socket poll to check its stop
/// flag, bounding unwatch/shutdown latency.
const STOP_POLL: Duration = Duration::from_millis(500);

#[derive(Default)]
struct DcState {
    seat: Option<wl_seat::WlSeat>,
    manager: Option<ZwlrDataControlManagerV1>,
    /// Advertised mimes per live offer.
    offer_mimes: HashMap<ObjectId, Vec<String>>,
    selection: Option<ZwlrDataControlOfferV1>,
    /// Set on every `selection` event (the watch loop's change edge).
    selection_changed: bool,
    /// Compositor told this device it is done (seat gone).
    finished: bool,
    /// The write path's source lost the selection to another client.
    cancelled: bool,
    /// Mime -> bytes served by the write path's source.
    serve: Vec<(String, Vec<u8>)>,
    sync_done: bool,
}

impl SyncState for DcState {
    fn sync_done_mut(&mut self) -> &mut bool {
        &mut self.sync_done
    }
}
impl_sync_callback!(DcState);

impl Dispatch<wl_registry::WlRegistry, ()> for DcState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, .. } = event {
            // Version 1 of each suffices: the seat is only an argument, and v1
            // data-control carries the regular selection this bridge needs.
            match interface.as_str() {
                "wl_seat" if state.seat.is_none() => {
                    state.seat = Some(registry.bind(name, 1, qh, ()));
                }
                "zwlr_data_control_manager_v1" if state.manager.is_none() => {
                    state.manager = Some(registry.bind(name, 1, qh, ()));
                }
                _ => {}
            }
        }
    }
}

delegate_noop!(DcState: ignore wl_seat::WlSeat);
delegate_noop!(DcState: ZwlrDataControlManagerV1);

impl Dispatch<ZwlrDataControlOfferV1, ()> for DcState {
    fn event(
        state: &mut Self,
        offer: &ZwlrDataControlOfferV1,
        event: zwlr_data_control_offer_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwlr_data_control_offer_v1::Event::Offer { mime_type } = event {
            state.offer_mimes.entry(offer.id()).or_default().push(mime_type);
        }
    }
}

impl Dispatch<ZwlrDataControlDeviceV1, ()> for DcState {
    fn event(
        state: &mut Self,
        _: &ZwlrDataControlDeviceV1,
        event: zwlr_data_control_device_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_data_control_device_v1::Event::DataOffer { id } => {
                state.offer_mimes.entry(id.id()).or_default();
            }
            zwlr_data_control_device_v1::Event::Selection { id } => {
                // Replaced offers are dead objects; drop their proxy and mimes so
                // a long-lived watch connection doesn't accumulate them.
                if let Some(old) = state.selection.take()
                    && id.as_ref().map(|o| o.id()) != Some(old.id()) {
                        state.offer_mimes.remove(&old.id());
                        old.destroy();
                    }
                state.selection = id;
                state.selection_changed = true;
            }
            zwlr_data_control_device_v1::Event::Finished => {
                state.finished = true;
            }
            _ => {}
        }
    }

    wayland_client::event_created_child!(DcState, ZwlrDataControlDeviceV1, [
        zwlr_data_control_device_v1::EVT_DATA_OFFER_OPCODE => (ZwlrDataControlOfferV1, ()),
    ]);
}

impl Dispatch<ZwlrDataControlSourceV1, ()> for DcState {
    fn event(
        state: &mut Self,
        _: &ZwlrDataControlSourceV1,
        event: zwlr_data_control_source_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_data_control_source_v1::Event::Send { mime_type, fd } => {
                if let Some((_, data)) = state.serve.iter().find(|(m, _)| *m == mime_type) {
                    let _ = write_fd_all(&fd, data, IO_TIMEOUT);
                }
                // fd drops here, closing the pipe so the paster sees EOF.
            }
            zwlr_data_control_source_v1::Event::Cancelled => {
                state.cancelled = true;
            }
            _ => {}
        }
    }
}

/// Connect to `socket_path` and return (connection, queue, state, device) with
/// the current selection already delivered.
fn open_device(
    socket_path: &str,
) -> Result<
    (Connection, wayland_client::EventQueue<DcState>, DcState, ZwlrDataControlDeviceV1),
    String,
> {
    let stream =
        UnixStream::connect(socket_path).map_err(|e| format!("connect {socket_path}: {e}"))?;
    let conn = Connection::from_socket(stream).map_err(|e| format!("wayland setup: {e}"))?;
    let mut queue = conn.new_event_queue();
    let qh = queue.handle();
    let _registry = conn.display().get_registry(&qh, ());
    let mut state = DcState::default();
    bounded_roundtrip(&conn, &mut queue, &mut state)?;
    let seat = state.seat.clone().ok_or("app compositor advertises no wl_seat")?;
    let manager = state
        .manager
        .clone()
        .ok_or("app compositor does not advertise zwlr_data_control_manager_v1")?;
    let device = manager.get_data_device(&seat, &qh, ());
    bounded_roundtrip(&conn, &mut queue, &mut state)?;
    Ok((conn, queue, state, device))
}

/// Mimes offered by the current selection (empty when nothing is copied).
pub(crate) fn list_types(socket_path: &str) -> Result<Vec<String>, String> {
    let (_conn, _queue, state, device) = open_device(socket_path)?;
    let out = state
        .selection
        .as_ref()
        .and_then(|o| state.offer_mimes.get(&o.id()).cloned())
        .unwrap_or_default();
    device.destroy();
    Ok(out)
}

/// The current selection's payload for `mime`, or None when there is no
/// selection or it does not offer that mime.
pub(crate) fn read(socket_path: &str, mime: &str) -> Result<Option<Vec<u8>>, String> {
    let (conn, mut queue, mut state, device) = open_device(socket_path)?;
    let Some(offer) = state.selection.clone() else {
        device.destroy();
        return Ok(None);
    };
    let offered = state.offer_mimes.get(&offer.id()).is_some_and(|m| m.iter().any(|x| x == mime));
    if !offered {
        device.destroy();
        return Ok(None);
    }
    let (rd, wr) = pipe_cloexec()?;
    offer.receive(mime.to_string(), wr.as_fd());
    queue.flush().map_err(|e| format!("flush: {e}"))?;
    drop(wr);
    // The source app writes into the pipe as it pleases; dispatch is not needed
    // for the bytes, only the fd read.
    let data = read_fd_to_end(&rd, IO_TIMEOUT)?;
    let _ = bounded_roundtrip(&conn, &mut queue, &mut state);
    device.destroy();
    Ok(Some(data))
}

/// Take the selection, serving `entries` (mime, bytes) to every paster from a
/// detached thread until another client takes the selection (or the compositor
/// goes away). Replacing a previous write is implicit: the compositor cancels
/// the old source when the new one takes the selection.
pub(crate) fn write(socket_path: &str, entries: Vec<(String, Vec<u8>)>) -> Result<(), String> {
    let path = socket_path.to_string();
    std::thread::Builder::new()
        .name("pf-dc-selection".into())
        .spawn(move || {
            if let Err(e) = serve_selection(&path, entries) {
                eprintln!("[Clipboard] app-compositor selection serve ended: {e}");
            }
        })
        .map_err(|e| format!("spawn: {e}"))?;
    Ok(())
}

fn serve_selection(socket_path: &str, entries: Vec<(String, Vec<u8>)>) -> Result<(), String> {
    let stream =
        UnixStream::connect(socket_path).map_err(|e| format!("connect {socket_path}: {e}"))?;
    let conn = Connection::from_socket(stream).map_err(|e| format!("wayland setup: {e}"))?;
    let mut queue = conn.new_event_queue();
    let qh = queue.handle();
    let _registry = conn.display().get_registry(&qh, ());
    let mut state = DcState::default();
    bounded_roundtrip(&conn, &mut queue, &mut state)?;
    let seat = state.seat.clone().ok_or("app compositor advertises no wl_seat")?;
    let manager = state
        .manager
        .clone()
        .ok_or("app compositor does not advertise zwlr_data_control_manager_v1")?;
    let device = manager.get_data_device(&seat, &qh, ());
    let source = manager.create_data_source(&qh, ());
    for (mime, _) in &entries {
        source.offer(mime.clone());
    }
    state.serve = entries;
    device.set_selection(Some(&source));
    bounded_roundtrip(&conn, &mut queue, &mut state)?;
    while !state.cancelled && !state.finished {
        // Sends are served inside dispatch; block until the compositor has
        // something (with a poll so a dead compositor can't pin the thread).
        queue.flush().map_err(|e| format!("flush: {e}"))?;
        if let Some(guard) = conn.prepare_read() {
            use std::os::fd::AsRawFd;
            if wait_readable(guard.connection_fd().as_raw_fd(), STOP_POLL)? {
                guard.read().map_err(|e| format!("read: {e}"))?;
            }
        }
        queue.dispatch_pending(&mut state).map_err(|e| format!("dispatch: {e}"))?;
    }
    source.destroy();
    device.destroy();
    let _ = queue.flush();
    Ok(())
}

/// Drop the selection (the compositor also cancels whatever source held it).
pub(crate) fn clear(socket_path: &str) -> Result<(), String> {
    let (conn, mut queue, mut state, device) = open_device(socket_path)?;
    device.set_selection(None);
    bounded_roundtrip(&conn, &mut queue, &mut state)?;
    device.destroy();
    Ok(())
}

struct WatchHandle {
    stop: Arc<AtomicBool>,
}

static WATCHERS: Mutex<Option<HashMap<String, WatchHandle>>> = Mutex::new(None);

/// Report every selection change on `socket_path` (including the one current at
/// start) to `callback(mimes: list[str])` from a background thread. A second
/// watch on the same socket replaces the first.
pub(crate) fn watch(socket_path: &str, callback: Py<PyAny>) -> Result<(), String> {
    let stop = Arc::new(AtomicBool::new(false));
    {
        let mut reg = WATCHERS.lock().unwrap();
        let map = reg.get_or_insert_with(HashMap::new);
        if let Some(old) = map.insert(socket_path.to_string(), WatchHandle { stop: stop.clone() })
        {
            old.stop.store(true, Ordering::Relaxed);
        }
    }
    let path = socket_path.to_string();
    std::thread::Builder::new()
        .name("pf-dc-watch".into())
        .spawn(move || {
            if let Err(e) = watch_loop(&path, callback, &stop) {
                eprintln!("[Clipboard] app-compositor watch ended: {e}");
            }
        })
        .map_err(|e| format!("spawn: {e}"))?;
    Ok(())
}

/// Stop every clipboard watch (process teardown sweep; watches are NOT tied to
/// captures, so the global stop helper must reach them explicitly).
pub(crate) fn unwatch_all() {
    let mut reg = WATCHERS.lock().unwrap();
    if let Some(map) = reg.as_mut() {
        for (_, handle) in map.drain() {
            handle.stop.store(true, Ordering::Relaxed);
        }
    }
}

/// Stop the watch on `socket_path` (no-op when none is running).
pub(crate) fn unwatch(socket_path: &str) {
    let mut reg = WATCHERS.lock().unwrap();
    if let Some(map) = reg.as_mut()
        && let Some(handle) = map.remove(socket_path) {
            handle.stop.store(true, Ordering::Relaxed);
        }
}

fn watch_loop(socket_path: &str, callback: Py<PyAny>, stop: &AtomicBool) -> Result<(), String> {
    let (conn, mut queue, mut state, device) = open_device(socket_path)?;
    while !stop.load(Ordering::Relaxed) && !state.finished {
        if state.selection_changed {
            state.selection_changed = false;
            let mimes = state
                .selection
                .as_ref()
                .and_then(|o| state.offer_mimes.get(&o.id()).cloned())
                .unwrap_or_default();
            if !mimes.is_empty() {
                if crate::PY_SHUTDOWN.load(Ordering::Relaxed) {
                    break;
                }
                Python::attach(|py| {
                    if let Err(e) = callback.call1(py, (mimes,)) {
                        e.print(py);
                    }
                });
            }
        }
        queue.flush().map_err(|e| format!("flush: {e}"))?;
        if let Some(guard) = conn.prepare_read() {
            use std::os::fd::AsRawFd;
            if wait_readable(guard.connection_fd().as_raw_fd(), STOP_POLL)? {
                guard.read().map_err(|e| format!("read: {e}"))?;
            }
        }
        queue.dispatch_pending(&mut state).map_err(|e| format!("dispatch: {e}"))?;
    }
    device.destroy();
    let _ = queue.flush();
    Ok(())
}
