//! `zwp_virtual_keyboard_v1` client for typing Unicode text into ANOTHER Wayland
//! compositor's socket.
//!
//! pixelflux is normally the compositor, but in a nested deployment (a labwc/kwin
//! session running as a client of pixelflux) the apps live on that inner
//! compositor's socket, and keys injected into pixelflux's own seat resolve
//! against pixelflux's keymap — an overlay the inner compositor never sees. So
//! text is typed here as a client of whichever compositor the apps live under —
//! by Computer-Use actions and by selkies over the `type_text_wayland` ABI —
//! reusing the seat's [`KeymapPolicy`] over a US base: base-reachable characters
//! press their ordinary keycodes, everything else is overlay-bound, one upload
//! per batch. One-shot (connect, type, disconnect): a call carries a whole commit
//! of text, and a fresh connection leaves no stale-socket state to manage.
//! Blocking, off the compositor thread, with every round-trip deadline-bounded so
//! a wedged compositor cannot hang the caller forever.

use std::collections::HashSet;
use std::os::fd::AsFd;
use std::os::unix::net::UnixStream;
use std::sync::OnceLock;
use std::time::Duration;

use wayland_client::protocol::{wl_registry, wl_seat};
use wayland_client::{delegate_noop, Connection, Dispatch, EventQueue, QueueHandle};
use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::{
    zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1,
    zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
};

use crate::wayland::keymap::{compile_rmlvo, keysym_for_char, KeymapPolicy};
use crate::wayland::wlclient::{bounded_roundtrip, impl_sync_callback, memfd_with, SyncState};

/// Overlay keycodes stay under the X11 255 ceiling so XWayland apps under the app
/// compositor can still receive them (the seat's own overlay sits above 255).
const OVERLAY_FIRST_XKB: u32 = 150;
const OVERLAY_LAST_XKB: u32 = 255;
const OVERLAY_SLOTS: usize = (OVERLAY_LAST_XKB - OVERLAY_FIRST_XKB + 1) as usize;
/// wl_keyboard / zwp_virtual_keyboard key events carry evdev codes (xkb - 8).
const EVDEV_OFFSET: u32 = 8;
const KEYMAP_FORMAT_XKB_V1: u32 = 1;

#[derive(Default)]
struct Globals {
    seat: Option<wl_seat::WlSeat>,
    manager: Option<ZwpVirtualKeyboardManagerV1>,
    sync_done: bool,
}

impl SyncState for Globals {
    fn sync_done_mut(&mut self) -> &mut bool {
        &mut self.sync_done
    }
}
impl_sync_callback!(Globals);

impl Dispatch<wl_registry::WlRegistry, ()> for Globals {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, .. } = event {
            // Version 1 of each suffices: the seat is only the manager argument.
            match interface.as_str() {
                "wl_seat" if state.seat.is_none() => {
                    state.seat = Some(registry.bind(name, 1, qh, ()));
                }
                "zwp_virtual_keyboard_manager_v1" if state.manager.is_none() => {
                    state.manager = Some(registry.bind(name, 1, qh, ()));
                }
                _ => {}
            }
        }
    }
}

delegate_noop!(Globals: ignore wl_seat::WlSeat);
delegate_noop!(Globals: ZwpVirtualKeyboardManagerV1);
delegate_noop!(Globals: ZwpVirtualKeyboardV1);

/// The US base keymap text, compiled once per process: selkies types per text
/// commit, and xkbcommon compilation is the expensive part of a call.
pub(crate) fn us_base_text() -> Option<&'static str> {
    static CACHE: OnceLock<Option<String>> = OnceLock::new();
    CACHE.get_or_init(|| compile_rmlvo("", "", "us", "", "")).as_deref()
}

fn upload_keymap(
    vk: &ZwpVirtualKeyboardV1,
    queue: &mut EventQueue<Globals>,
    text: &str,
) -> Result<(), String> {
    let mut data = text.as_bytes().to_vec();
    data.push(0); // compositors parse the mapping as a NUL-terminated string
    let fd = memfd_with(&data)?;
    vk.keymap(KEYMAP_FORMAT_XKB_V1, fd.as_fd(), data.len() as u32);
    queue.flush().map_err(|e| format!("flush keymap: {e}"))
}

/// One-shot: connect to `socket_path`, type `text` in order, disconnect. Codepoints
/// with no keysym are skipped. Blocking; call off the compositor's calloop thread.
pub fn type_text_to(socket_path: &str, text: &str) -> Result<(), String> {
    let stream =
        UnixStream::connect(socket_path).map_err(|e| format!("connect {socket_path}: {e}"))?;
    let conn = Connection::from_socket(stream).map_err(|e| format!("wayland setup: {e}"))?;
    let mut queue = conn.new_event_queue();
    let qh = queue.handle();
    let _registry = conn.display().get_registry(&qh, ());
    let mut state = Globals::default();
    bounded_roundtrip(&conn, &mut queue, &mut state)?;
    let seat = state.seat.take().ok_or("app compositor advertises no wl_seat")?;
    let manager = state
        .manager
        .take()
        .ok_or("app compositor does not advertise zwp_virtual_keyboard_manager_v1")?;
    let vk = manager.create_virtual_keyboard(&seat, &qh, ());
    // Surfaces an "unauthorized" bind error before the first keymap upload.
    bounded_roundtrip(&conn, &mut queue, &mut state)?;

    let mut policy = KeymapPolicy::with_overlay_range(OVERLAY_FIRST_XKB, OVERLAY_LAST_XKB);
    policy.rebuild_base(
        us_base_text().ok_or("us base keymap failed to compile")?.to_string(),
    );
    let syms: Vec<u32> = text.chars().map(keysym_for_char).filter(|&s| s != 0).collect();
    let none = HashSet::new();
    let mut uploaded = false;
    // Chunk bound: at most OVERLAY_SLOTS keysyms per bind call, so a batch can
    // never recycle a slot it assigned earlier in the same batch.
    for chunk in syms.chunks(OVERLAY_SLOTS) {
        let (keycodes, changed) = policy.bind_many_plain(chunk, &none);
        // The protocol requires a keymap before the first key event even when the
        // whole text resolves in the base.
        if changed || !uploaded {
            upload_keymap(&vk, &mut queue, &policy.keymap_text())?;
            bounded_roundtrip(&conn, &mut queue, &mut state)?;
            std::thread::sleep(Duration::from_millis(10));
            uploaded = true;
        }
        for &kc in &keycodes {
            if kc < EVDEV_OFFSET {
                continue; // unbindable keysym
            }
            for pressed in [1u32, 0] {
                vk.key(0, kc - EVDEV_OFFSET, pressed);
                queue.flush().map_err(|e| format!("flush key: {e}"))?;
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    }
    bounded_roundtrip(&conn, &mut queue, &mut state)?;
    vk.destroy();
    let _ = queue.flush();
    Ok(())
}
