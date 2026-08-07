//! Host-capture mode: pixelflux as a CLIENT of an external Wayland compositor
//! (e.g. labwc running `WLR_BACKENDS=headless`), inverting the nested topology.
//!
//! The host compositor owns the session — one seat, one selection, one screen
//! model — and pixelflux captures and injects as a privileged-protocol client:
//! frames via `zwlr_screencopy_v1` (v3 `copy_with_damage`, so idle screens cost
//! nothing), keyboard via a persistent `zwp_virtual_keyboard_v1` device carrying
//! selkies' own keymap text, pointer via `zwlr_virtual_pointer_v1`. Zero-copy is
//! preserved by allocating the capture buffers from pixelflux's OWN GBM device
//! (render node — no privileges): the compositor blits straight into the dmabufs
//! the encoder imports, so no CPU ever touches a frame. Without a GPU the same
//! loop degrades to wl_shm buffers feeding the CPU encode pool.
//!
//! Every host `wl_output` maps to one selkies display id, in registry order: a
//! control thread on the primary connection owns wlr-output-management and
//! applies the whole layout (per-output custom modes plus row positions) as one
//! atomic configuration, while each output gets its own capture connection and
//! thread. Input proxies are called from the calloop thread directly
//! (wayland-client proxies are thread-safe); pointer coordinates arrive in the
//! same union-layout space selkies uses, so the virtual-pointer extent is the
//! layout's bounding box. All capture-side waits poll a wake pipe next to the
//! Wayland socket, so a resize or teardown lands immediately even while the
//! thread is parked in a `copy_with_damage` wait on a static screen.

use std::fs::File;
use std::os::fd::{AsFd, AsRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use gbm::{BufferObjectFlags, Device as GbmDevice, Format as GbmFormat};
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::Buffer as _;
use smithay::utils::{Physical, Rectangle};
use wayland_client::protocol::{wl_output, wl_pointer, wl_registry, wl_seat, wl_shm, wl_shm_pool};
use wayland_client::{delegate_noop, Connection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum};
use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_buffer_params_v1::{self, ZwpLinuxBufferParamsV1},
    zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1,
};
use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::{
    zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1,
    zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
};
use wayland_protocols_wlr::screencopy::v1::client::{
    zwlr_screencopy_frame_v1::{self, ZwlrScreencopyFrameV1},
    zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
};
use wayland_protocols_wlr::output_management::v1::client::{
    zwlr_output_configuration_head_v1::ZwlrOutputConfigurationHeadV1,
    zwlr_output_configuration_v1::{self, ZwlrOutputConfigurationV1},
    zwlr_output_head_v1::{self, ZwlrOutputHeadV1},
    zwlr_output_manager_v1::{self, ZwlrOutputManagerV1},
    zwlr_output_mode_v1::ZwlrOutputModeV1,
};
use wayland_protocols_wlr::virtual_pointer::v1::client::{
    zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1,
    zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1,
};

use crate::wayland::wlclient::{
    bounded_roundtrip, drain_pipe, impl_sync_callback, memfd_with, socket_path, wait_readable2,
    wake_pipe, wake_write, SyncState,
};

const KEYMAP_FORMAT_XKB_V1: u32 = 1;
/// Frames in flight between a capture thread and the encode tick. Two slots:
/// one being blitted by the compositor while the previous one is encoded.
const SLOTS: usize = 2;

/// A frame the compositor finished blitting, ready for the encoder.
pub struct HostFrame {
    gen: u64,
    slot: usize,
    /// GPU path: the filled dmabuf (Arc-backed; the buffer itself stays owned by
    /// the capture thread and is reused once the slot is released).
    pub dmabuf: Option<Dmabuf>,
    /// Software path: the shm mapping itself — the consumer converts straight
    /// out of it (one copy total); the slot is not reused until release.
    pub cpu: Option<HostCpuFrame>,
    pub width: i32,
    pub height: i32,
    pub damage: Vec<Rectangle<i32, Physical>>,
}

/// Borrow-by-Arc view of a software frame still sitting in its shm mapping.
pub struct HostCpuFrame {
    map: Arc<memmap2::MmapMut>,
    stride: usize,
    format: u32,
}

impl HostCpuFrame {
    /// Convert the frame into tight BGRA rows in `dst` (sized `w*4*h`); the
    /// announced shm format decides the per-pixel conversion (compositors pick
    /// their renderer's preferred read format, e.g. 24-bit BGR on NVIDIA GLES).
    pub fn write_bgra(&self, w: i32, h: i32, dst: &mut [u8]) {
        let row = (w * 4) as usize;
        let (src_bpp, swap_rb) = match self.format {
            f if f == wl_shm::Format::Xbgr8888 as u32
                || f == wl_shm::Format::Abgr8888 as u32 => (4usize, true),
            f if f == wl_shm::Format::Bgr888 as u32 => (3, false),
            _ => (4, false), // xrgb/argb: already BGRA byte order
        };
        for y in 0..h as usize {
            let src_start = y * self.stride;
            let src_end = (src_start + src_bpp * w as usize).min(self.map.len());
            if src_start >= src_end || (y + 1) * row > dst.len() {
                break;
            }
            let src_row = &self.map[src_start..src_end];
            let dst_row = &mut dst[y * row..(y + 1) * row];
            match (src_bpp, swap_rb) {
                (4, false) => {
                    let n = row.min(src_row.len());
                    dst_row[..n].copy_from_slice(&src_row[..n]);
                }
                (4, true) => {
                    for (d, s) in dst_row.chunks_exact_mut(4).zip(src_row.chunks_exact(4)) {
                        d[0] = s[2];
                        d[1] = s[1];
                        d[2] = s[0];
                        d[3] = s[3];
                    }
                }
                _ => {
                    for (d, s) in dst_row.chunks_exact_mut(4).zip(src_row.chunks_exact(3)) {
                        d[0] = s[0];
                        d[1] = s[1];
                        d[2] = s[2];
                        d[3] = 0xff;
                    }
                }
            }
        }
    }
}

/// One display's slice of the union layout, as selkies configured it. Slots are
/// keyed by selkies display id ('display2' is id 2 — ids are sparse); active ids
/// map onto host outputs by ascending rank.
#[derive(Clone, Copy, Default)]
struct LayoutSlot {
    want: (i32, i32),
    pos: (i32, i32),
    active: bool,
}

enum ToHost {
    Start { width: i32, height: i32 },
    /// Slot return; `gen` guards against slots recycled by a renegotiation
    /// while the consumer still held the frame.
    Release { gen: u64, slot: usize },
    /// Park the capture (display stopped) without ending the thread.
    Idle,
}

enum CtrlMsg {
    Apply(Vec<LayoutSlot>),
}

// ---------------------------------------------------------------------------
// Control connection: seat + virtual input devices + wlr-output-management.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct CtrlState {
    seat: Option<wl_seat::WlSeat>,
    vk_mgr: Option<ZwpVirtualKeyboardManagerV1>,
    vptr_mgr: Option<ZwlrVirtualPointerManagerV1>,
    has_screencopy: bool,
    outputs: Vec<(wl_output::WlOutput, Option<String>)>,
    /// Registry indices in display order (natural name sort — registry
    /// announcement order is not guaranteed to match output creation order).
    order: Vec<usize>,
    output_mgr: Option<ZwlrOutputManagerV1>,
    heads: Vec<(ZwlrOutputHeadV1, Option<String>)>,
    om_serial: Option<u32>,
    cfg_result: Option<bool>,
    cfg_cancelled: bool,
    sync_done: bool,
}


/// Natural sort key for an output name: text prefix plus trailing number, so
/// HEADLESS-2 orders before HEADLESS-10. Unnamed outputs keep registry order
/// after every named one.
fn output_order_key(name: Option<&String>, registry_idx: usize) -> (bool, String, u64, usize) {
    match name {
        Some(n) => {
            let digits_at = n.rfind(|c: char| !c.is_ascii_digit()).map(|i| i + 1).unwrap_or(0);
            let num = n[digits_at..].parse::<u64>().unwrap_or(0);
            (false, n[..digits_at].to_string(), num, registry_idx)
        }
        None => (true, String::new(), 0, registry_idx),
    }
}

impl SyncState for CtrlState {
    fn sync_done_mut(&mut self) -> &mut bool {
        &mut self.sync_done
    }
}
impl_sync_callback!(CtrlState);

impl Dispatch<wl_registry::WlRegistry, ()> for CtrlState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, version } = event {
            match interface.as_str() {
                "wl_seat" if state.seat.is_none() => {
                    state.seat = Some(registry.bind(name, 1, qh, ()))
                }
                "zwp_virtual_keyboard_manager_v1" => {
                    state.vk_mgr = Some(registry.bind(name, 1, qh, ()))
                }
                "zwlr_virtual_pointer_manager_v1" => {
                    state.vptr_mgr = Some(registry.bind(name, version.min(2), qh, ()))
                }
                "zwlr_screencopy_manager_v1" if version >= 3 => state.has_screencopy = true,
                "wl_output" => {
                    let idx = state.outputs.len();
                    let out = registry.bind(name, version.min(4), qh, idx);
                    state.outputs.push((out, None));
                }
                "zwlr_output_manager_v1" => {
                    state.output_mgr = Some(registry.bind(name, 1, qh, ()))
                }
                _ => {}
            }
        }
    }
}

macro_rules! impl_output_name {
    ($t:ty) => {
        impl Dispatch<wl_output::WlOutput, usize> for $t {
            fn event(
                state: &mut Self,
                _: &wl_output::WlOutput,
                event: wl_output::Event,
                idx: &usize,
                _: &Connection,
                _: &QueueHandle<Self>,
            ) {
                if let wl_output::Event::Name { name } = event {
                    if let Some(o) = state.outputs.get_mut(*idx) {
                        o.1 = Some(name);
                    }
                }
            }
        }
    };
}
impl_output_name!(CtrlState);

delegate_noop!(CtrlState: ignore wl_seat::WlSeat);
delegate_noop!(CtrlState: ZwpVirtualKeyboardManagerV1);
delegate_noop!(CtrlState: ZwpVirtualKeyboardV1);
delegate_noop!(CtrlState: ZwlrVirtualPointerManagerV1);
delegate_noop!(CtrlState: ZwlrVirtualPointerV1);
delegate_noop!(CtrlState: ignore ZwlrOutputModeV1);
delegate_noop!(CtrlState: ZwlrOutputConfigurationHeadV1);

impl Dispatch<ZwlrOutputManagerV1, ()> for CtrlState {
    fn event(
        state: &mut Self,
        _: &ZwlrOutputManagerV1,
        event: zwlr_output_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_output_manager_v1::Event::Head { head } => state.heads.push((head, None)),
            zwlr_output_manager_v1::Event::Done { serial } => state.om_serial = Some(serial),
            _ => {}
        }
    }

    wayland_client::event_created_child!(CtrlState, ZwlrOutputManagerV1, [
        zwlr_output_manager_v1::EVT_HEAD_OPCODE => (ZwlrOutputHeadV1, ()),
    ]);
}

impl Dispatch<ZwlrOutputHeadV1, ()> for CtrlState {
    fn event(
        state: &mut Self,
        head: &ZwlrOutputHeadV1,
        event: zwlr_output_head_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_output_head_v1::Event::Name { name } => {
                if let Some(h) = state.heads.iter_mut().find(|(h, _)| h.id() == head.id()) {
                    h.1 = Some(name);
                }
            }
            zwlr_output_head_v1::Event::Finished => {
                state.heads.retain(|(h, _)| h.id() != head.id());
            }
            _ => {}
        }
    }

    wayland_client::event_created_child!(CtrlState, ZwlrOutputHeadV1, [
        zwlr_output_head_v1::EVT_MODE_OPCODE => (ZwlrOutputModeV1, ()),
    ]);
}

impl Dispatch<ZwlrOutputConfigurationV1, ()> for CtrlState {
    fn event(
        state: &mut Self,
        _: &ZwlrOutputConfigurationV1,
        event: zwlr_output_configuration_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_output_configuration_v1::Event::Succeeded => state.cfg_result = Some(true),
            zwlr_output_configuration_v1::Event::Failed => state.cfg_result = Some(false),
            zwlr_output_configuration_v1::Event::Cancelled => state.cfg_cancelled = true,
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Capture connections: one per host output (screencopy + buffer allocation).
// ---------------------------------------------------------------------------

#[derive(Default)]
struct CaptureState {
    shm: Option<wl_shm::WlShm>,
    dmabuf: Option<ZwpLinuxDmabufV1>,
    screencopy: Option<ZwlrScreencopyManagerV1>,
    outputs: Vec<(wl_output::WlOutput, Option<String>)>,
    // Per-frame capture negotiation/results.
    announce_dmabuf: Option<(u32, i32, i32)>, // fourcc, w, h
    announce_shm: Option<(u32, i32, i32, i32)>, // format, w, h, stride
    buffer_done: bool,
    damage: Vec<Rectangle<i32, Physical>>,
    ready: bool,
    failed: bool,
    sync_done: bool,
}


impl CaptureState {
    fn reset_frame(&mut self) {
        self.announce_dmabuf = None;
        self.announce_shm = None;
        self.buffer_done = false;
        self.damage.clear();
        self.ready = false;
        self.failed = false;
    }
}

impl SyncState for CaptureState {
    fn sync_done_mut(&mut self) -> &mut bool {
        &mut self.sync_done
    }
}
impl_sync_callback!(CaptureState);
impl_output_name!(CaptureState);

impl Dispatch<wl_registry::WlRegistry, ()> for CaptureState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, version } = event {
            match interface.as_str() {
                "wl_shm" => state.shm = Some(registry.bind(name, 1, qh, ())),
                "zwp_linux_dmabuf_v1" if version >= 3 => {
                    state.dmabuf = Some(registry.bind(name, 3, qh, ()))
                }
                "zwlr_screencopy_manager_v1" if version >= 3 => {
                    state.screencopy = Some(registry.bind(name, 3, qh, ()))
                }
                "wl_output" => {
                    let idx = state.outputs.len();
                    let out = registry.bind(name, version.min(4), qh, idx);
                    state.outputs.push((out, None));
                }
                _ => {}
            }
        }
    }
}

delegate_noop!(CaptureState: ignore wl_shm::WlShm);
delegate_noop!(CaptureState: ignore wl_shm_pool::WlShmPool);
delegate_noop!(CaptureState: ignore wayland_client::protocol::wl_buffer::WlBuffer);
delegate_noop!(CaptureState: ZwlrScreencopyManagerV1);

impl Dispatch<ZwpLinuxDmabufV1, ()> for CaptureState {
    fn event(
        _: &mut Self,
        _: &ZwpLinuxDmabufV1,
        _: <ZwpLinuxDmabufV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // format/modifier advertisements: the allocation follows the screencopy
        // frame's announcement instead.
    }
}

impl Dispatch<ZwpLinuxBufferParamsV1, ()> for CaptureState {
    fn event(
        _: &mut Self,
        _: &ZwpLinuxBufferParamsV1,
        _: zwp_linux_buffer_params_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // created/failed are only sent for non-immed creation.
    }
}

impl Dispatch<ZwlrScreencopyFrameV1, ()> for CaptureState {
    fn event(
        state: &mut Self,
        _: &ZwlrScreencopyFrameV1,
        event: zwlr_screencopy_frame_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_screencopy_frame_v1::Event::Buffer {
                format: WEnum::Value(f), width, height, stride,
            } => {
                state.announce_shm = Some((f as u32, width as i32, height as i32, stride as i32));
            }
            zwlr_screencopy_frame_v1::Event::LinuxDmabuf { format, width, height } => {
                state.announce_dmabuf = Some((format, width as i32, height as i32));
            }
            zwlr_screencopy_frame_v1::Event::BufferDone => state.buffer_done = true,
            zwlr_screencopy_frame_v1::Event::Damage { x, y, width, height } => {
                state.damage.push(Rectangle::new(
                    (x as i32, y as i32).into(),
                    (width as i32, height as i32).into(),
                ));
            }
            zwlr_screencopy_frame_v1::Event::Ready { .. } => state.ready = true,
            zwlr_screencopy_frame_v1::Event::Failed => state.failed = true,
            _ => {}
        }
    }
}

enum SlotBuffer {
    Gpu {
        _bo: gbm::BufferObject<()>,
        dmabuf: Dmabuf,
        wl: wayland_client::protocol::wl_buffer::WlBuffer,
    },
    Cpu {
        _pool: wl_shm_pool::WlShmPool,
        map: Arc<memmap2::MmapMut>,
        stride: i32,
        format: u32,
        wl: wayland_client::protocol::wl_buffer::WlBuffer,
    },
}

/// One host output's calloop-side handle: control channel, wake pipe, frames.
struct OutputHandle {
    to_thread: Sender<ToHost>,
    wake: OwnedFd,
    frames: Receiver<HostFrame>,
    /// Newest frame, kept (slot and all) until replaced so an IDR request on a
    /// static screen can re-encode current content like compositor mode does.
    retained: Mutex<Option<HostFrame>>,
    name: Option<String>,
}

impl OutputHandle {
    fn send(&self, msg: ToHost) {
        let _ = self.to_thread.send(msg);
        wake_write(self.wake.as_raw_fd());
    }
}

/// The calloop-side session: input proxies plus one capture handle per output.
pub struct HostSession {
    conn: Connection,
    vk: Option<ZwpVirtualKeyboardV1>,
    vptr: Option<ZwlrVirtualPointerV1>,
    ctrl_tx: Sender<CtrlMsg>,
    ctrl_wake: OwnedFd,
    outputs: Vec<OutputHandle>,
    layout: Mutex<std::collections::BTreeMap<u32, LayoutSlot>>,
    alive: Arc<AtomicBool>,
}

impl HostSession {
    /// Connect to `display`, bring up input devices, enumerate the host's
    /// outputs and spawn one capture thread per output (idle until
    /// [`start_capture`]) plus the layout-control thread. `gbm_path` (this
    /// process's render node) enables the zero-copy path.
    pub fn connect(display: &str, gbm_path: Option<std::path::PathBuf>) -> Result<Self, String> {
        let path = socket_path(display).ok_or("XDG_RUNTIME_DIR is unset")?;
        let stream = UnixStream::connect(&path).map_err(|e| format!("connect {path}: {e}"))?;
        let conn = Connection::from_socket(stream).map_err(|e| format!("wayland setup: {e}"))?;
        let mut queue = conn.new_event_queue();
        let qh = queue.handle();
        let _registry = conn.display().get_registry(&qh, ());
        let mut state = CtrlState::default();
        bounded_roundtrip(&conn, &mut queue, &mut state)?;

        let seat = state.seat.clone().ok_or("host compositor advertises no wl_seat")?;
        if !state.has_screencopy {
            return Err("host compositor lacks zwlr_screencopy_manager_v1 (v3)".into());
        }
        if state.outputs.is_empty() {
            return Err("host compositor has no wl_output".into());
        }

        let vk = match &state.vk_mgr {
            Some(mgr) => {
                let vk = mgr.create_virtual_keyboard(&seat, &qh, ());
                // A keymap must precede any key event; selkies replaces this with
                // its managed keymap through the ABI as soon as it starts.
                if let Some(text) = crate::wayland::vkclient::us_base_text() {
                    let mut data = text.as_bytes().to_vec();
                    data.push(0);
                    let fd = memfd_with(&data)?;
                    vk.keymap(KEYMAP_FORMAT_XKB_V1, fd.as_fd(), data.len() as u32);
                }
                Some(vk)
            }
            None => {
                eprintln!("[HostCapture] no zwp_virtual_keyboard_manager_v1: keyboard injection disabled.");
                None
            }
        };
        let vptr = match &state.vptr_mgr {
            Some(mgr) => Some(mgr.create_virtual_pointer(Some(&seat), &qh, ())),
            None => {
                eprintln!("[HostCapture] no zwlr_virtual_pointer_manager_v1: pointer injection disabled.");
                None
            }
        };
        // Second round-trip: wl_output v4 name events and output-management heads.
        bounded_roundtrip(&conn, &mut queue, &mut state)?;

        let alive = Arc::new(AtomicBool::new(true));
        let mut order: Vec<usize> = (0..state.outputs.len()).collect();
        order.sort_by_key(|&i| output_order_key(state.outputs[i].1.as_ref(), i));
        let names: Vec<Option<String>> =
            order.iter().map(|&i| state.outputs[i].1.clone()).collect();
        state.order = order;
        println!(
            "[HostCapture] host outputs: {}.",
            names
                .iter()
                .enumerate()
                .map(|(i, n)| format!("{i}={}", n.as_deref().unwrap_or("?")))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let mut outputs = Vec::new();
        for (i, name) in names.iter().cloned().enumerate() {
            let (wake_rd, wake_wr) = wake_pipe()?;
            let (to_thread, from_main) = std::sync::mpsc::channel::<ToHost>();
            let (frame_tx, frames) = std::sync::mpsc::channel::<HostFrame>();
            let display = display.to_string();
            let expect = name.clone();
            let gbm_path = gbm_path.clone();
            let alive = alive.clone();
            std::thread::Builder::new()
                .name(format!("pf-host-cap{i}"))
                .spawn(move || {
                    if let Err(e) =
                        capture_loop(&display, i, expect, gbm_path, from_main, wake_rd, frame_tx)
                    {
                        eprintln!("[HostCapture] output {i} capture ended: {e}");
                    }
                    if i == 0 {
                        alive.store(false, Ordering::Relaxed);
                    }
                })
                .map_err(|e| format!("spawn: {e}"))?;
            outputs.push(OutputHandle {
                to_thread,
                wake: wake_wr,
                frames,
                retained: Mutex::new(None),
                name,
            });
        }

        let (ctrl_tx, ctrl_rx) = std::sync::mpsc::channel::<CtrlMsg>();
        let (ctrl_wake_rd, ctrl_wake) = wake_pipe()?;
        {
            let conn = conn.clone();
            std::thread::Builder::new()
                .name("pf-host-ctrl".into())
                .spawn(move || control_loop(conn, queue, state, ctrl_rx, ctrl_wake_rd))
                .map_err(|e| format!("spawn: {e}"))?;
        }

        let layout = Mutex::new(std::collections::BTreeMap::new());
        Ok(Self { conn, vk, vptr, ctrl_tx, ctrl_wake, outputs, layout, alive })
    }

    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// The host output backing `display_id`: active display ids map onto host
    /// outputs by ascending rank (selkies ids are sparse — 'display2' is id 2).
    fn output_index_for(&self, display_id: u32) -> Option<usize> {
        let layout = self.layout.lock().unwrap();
        let rank = layout
            .iter()
            .filter(|(_, s)| s.active)
            .position(|(id, _)| *id == display_id)?;
        (rank < self.outputs.len()).then_some(rank)
    }

    /// True when an active capture on `display_id` has a host output behind it.
    pub fn has_output_for(&self, display_id: u32) -> bool {
        self.output_index_for(display_id).is_some()
    }

    /// Record where selkies laid out `display_id` (union coordinates); pushed
    /// to the host with the next capture (re)start.
    pub fn set_layout(&self, display_id: u32, x: i32, y: i32) {
        let mut layout = self.layout.lock().unwrap();
        layout.entry(display_id).or_default().pos = (x, y);
    }

    /// Aim `display_id` at `width`x`height`: assigns every active display its
    /// host output (by rank), asks the host for all modes and positions in one
    /// atomic configuration, and points the capture threads at their sizes;
    /// frames gate until the host applies them.
    pub fn start_capture(&self, display_id: u32, width: i32, height: i32) {
        let assignments = {
            let mut layout = self.layout.lock().unwrap();
            let slot = layout.entry(display_id).or_default();
            slot.want = (width, height);
            slot.active = true;
            let active: Vec<(u32, LayoutSlot)> = layout
                .iter()
                .filter(|(_, s)| s.active)
                .map(|(id, s)| (*id, *s))
                .collect();
            if active.iter().position(|(id, _)| *id == display_id).unwrap_or(usize::MAX)
                >= self.outputs.len()
            {
                eprintln!(
                    "[HostCapture] display {display_id} has no host output (host has {}); not captured.",
                    self.outputs.len()
                );
            }
            active
        };
        // Ranks may have shifted (a lower id joined): repoint every assigned
        // capture thread; a same-size Start is a no-op for an unaffected one.
        let mut by_output: Vec<LayoutSlot> = Vec::new();
        for (rank, (_, slot)) in assignments.iter().enumerate() {
            if rank >= self.outputs.len() {
                break;
            }
            self.outputs[rank].send(ToHost::Start { width: slot.want.0, height: slot.want.1 });
            by_output.push(*slot);
        }
        let _ = self.ctrl_tx.send(CtrlMsg::Apply(by_output));
        wake_write(self.ctrl_wake.as_raw_fd());
    }

    /// Park `display_id`'s capture (its slot leaves the pointer extent). The
    /// retained frame goes back to the capture pool — dropping it silently
    /// would wedge one of the two slots until the next renegotiation.
    pub fn idle_output(&self, display_id: u32) {
        let Some(idx) = self.output_index_for(display_id) else {
            self.layout.lock().unwrap().entry(display_id).or_default().active = false;
            return;
        };
        self.layout.lock().unwrap().entry(display_id).or_default().active = false;
        self.outputs[idx].send(ToHost::Idle);
        if let Some(old) = self.outputs[idx].retained.lock().unwrap().take() {
            self.outputs[idx].send(ToHost::Release { gen: old.gen, slot: old.slot });
        }
    }

    pub fn alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    /// Newest ready frame for `display_id`, releasing any staler ones straight
    /// back to the pool.
    pub fn try_take_frame(&self, display_id: u32) -> Option<HostFrame> {
        let handle = self.outputs.get(self.output_index_for(display_id)?)?;
        let mut newest: Option<HostFrame> = None;
        while let Ok(frame) = handle.frames.try_recv() {
            if let Some(stale) = newest.replace(frame) {
                handle.send(ToHost::Release { gen: stale.gen, slot: stale.slot });
            }
        }
        newest
    }

    /// Keep `frame` as `display_id`'s current content (releasing the one it
    /// replaces).
    pub fn retain_frame(&self, display_id: u32, frame: HostFrame) {
        let Some(handle) = self.output_index_for(display_id).and_then(|i| self.outputs.get(i))
        else {
            return;
        };
        let old = handle.retained.lock().unwrap().replace(frame);
        if let Some(old) = old {
            handle.send(ToHost::Release { gen: old.gen, slot: old.slot });
        }
    }

    /// Run `f` with `display_id`'s retained (current-content) frame, if any.
    pub fn with_retained<R>(
        &self,
        display_id: u32,
        f: impl FnOnce(Option<&HostFrame>) -> R,
    ) -> R {
        match self.output_index_for(display_id).and_then(|i| self.outputs.get(i)) {
            Some(handle) => f(handle.retained.lock().unwrap().as_ref()),
            None => f(None),
        }
    }

    /// Upload selkies' managed keymap to the virtual keyboard verbatim.
    pub fn set_keymap(&self, text: &str) {
        let Some(vk) = &self.vk else { return };
        let mut data = text.as_bytes().to_vec();
        data.push(0);
        match memfd_with(&data) {
            Ok(fd) => {
                vk.keymap(KEYMAP_FORMAT_XKB_V1, fd.as_fd(), data.len() as u32);
                let _ = self.conn.flush();
            }
            Err(e) => eprintln!("[HostCapture] keymap upload failed: {e}"),
        }
    }

    /// Key event in xkb numbering (evdev + 8), matching the seat injectors.
    pub fn key(&self, xkb_keycode: u32, pressed: bool) {
        let Some(vk) = &self.vk else { return };
        if xkb_keycode < 8 {
            return;
        }
        vk.key(0, xkb_keycode - 8, if pressed { 1 } else { 0 });
        let _ = self.conn.flush();
    }

    /// Union-layout bounding box of every active, output-backed display (the
    /// virtual-pointer extent, matching the space selkies injects in).
    fn extent(&self) -> (i32, i32) {
        let layout = self.layout.lock().unwrap();
        let mut w = 0;
        let mut h = 0;
        for (rank, (_, slot)) in layout.iter().filter(|(_, s)| s.active).enumerate() {
            if rank >= self.outputs.len() {
                break;
            }
            w = w.max(slot.pos.0 + slot.want.0);
            h = h.max(slot.pos.1 + slot.want.1);
        }
        (w, h)
    }

    pub fn pointer_motion_abs(&self, x: f64, y: f64) {
        let Some(vp) = &self.vptr else { return };
        let (w, h) = self.extent();
        if w <= 0 || h <= 0 {
            return;
        }
        let cx = x.clamp(0.0, (w - 1) as f64) as u32;
        let cy = y.clamp(0.0, (h - 1) as f64) as u32;
        vp.motion_absolute(0, cx, cy, w as u32, h as u32);
        vp.frame();
        let _ = self.conn.flush();
    }

    pub fn pointer_button(&self, btn: u32, pressed: bool) {
        let Some(vp) = &self.vptr else { return };
        vp.button(
            0,
            btn,
            if pressed { wl_pointer::ButtonState::Pressed } else { wl_pointer::ButtonState::Released },
        );
        vp.frame();
        let _ = self.conn.flush();
    }

    pub fn pointer_axis(&self, dx: f64, dy: f64) {
        let Some(vp) = &self.vptr else { return };
        if dy != 0.0 {
            vp.axis(0, wl_pointer::Axis::VerticalScroll, dy);
        }
        if dx != 0.0 {
            vp.axis(0, wl_pointer::Axis::HorizontalScroll, dx);
        }
        vp.frame();
        let _ = self.conn.flush();
    }
}

// Dropping the session drops every channel sender and wake-pipe write end; the
// capture and control threads observe the disconnect (poll sees POLLHUP) and
// exit on their own.

// ---------------------------------------------------------------------------
// Control thread: pumps the primary connection and applies layout requests.
// ---------------------------------------------------------------------------

fn control_loop(
    conn: Connection,
    mut queue: EventQueue<CtrlState>,
    mut state: CtrlState,
    ctrl_rx: Receiver<CtrlMsg>,
    wake_rd: OwnedFd,
) {
    let qh = queue.handle();
    loop {
        let mut pending: Option<Vec<LayoutSlot>> = None;
        loop {
            match ctrl_rx.try_recv() {
                Ok(CtrlMsg::Apply(slots)) => pending = Some(slots),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        if let Some(slots) = pending {
            apply_layout(&conn, &mut queue, &mut state, &qh, &slots);
        }
        if queue.dispatch_pending(&mut state).is_err() {
            return;
        }
        let _ = queue.flush();
        if let Some(guard) = conn.prepare_read() {
            match wait_readable2(
                guard.connection_fd().as_raw_fd(),
                wake_rd.as_raw_fd(),
                Some(Duration::from_secs(1)),
            ) {
                Ok((wl, wake)) => {
                    if wake {
                        drain_pipe(wake_rd.as_raw_fd());
                    }
                    if wl {
                        let _ = guard.read();
                    } else {
                        drop(guard);
                    }
                }
                Err(_) => return,
            }
        }
    }
}

/// Ask the host (wlr-output-management) to give every active output its wanted
/// mode and layout position in one atomic configuration. Retries across a
/// `cancelled` (stale serial); a refusal only warns — capture threads keep
/// gating on the size they were promised.
fn apply_layout(
    conn: &Connection,
    queue: &mut EventQueue<CtrlState>,
    state: &mut CtrlState,
    qh: &QueueHandle<CtrlState>,
    slots: &[LayoutSlot],
) {
    let Some(mgr) = state.output_mgr.clone() else {
        eprintln!("[HostCapture] host lacks zwlr_output_manager_v1; capture follows the host's own size.");
        return;
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    for _ in 0..3 {
        while state.om_serial.is_none() && Instant::now() < deadline {
            if !pump_ctrl(conn, queue, state) {
                return;
            }
        }
        let Some(serial) = state.om_serial else {
            eprintln!("[HostCapture] output-management serial never arrived; resize skipped.");
            return;
        };
        state.cfg_result = None;
        state.cfg_cancelled = false;
        let cfg = mgr.create_configuration(serial, qh, ());
        let mut any = false;
        for (i, slot) in slots.iter().enumerate() {
            if !slot.active {
                continue;
            }
            let want_name = state
                .order
                .get(i)
                .and_then(|&oi| state.outputs.get(oi))
                .and_then(|(_, n)| n.clone());
            let head = match want_name
                .as_ref()
                .and_then(|n| state.heads.iter().find(|(_, hn)| hn.as_ref() == Some(n)))
                .or_else(|| state.heads.get(i))
            {
                Some((h, _)) => h.clone(),
                None => {
                    eprintln!("[HostCapture] no output-management head for output {i}; not resized.");
                    continue;
                }
            };
            let cfg_head = cfg.enable_head(&head, qh, ());
            cfg_head.set_custom_mode(slot.want.0, slot.want.1, 0);
            cfg_head.set_position(slot.pos.0, slot.pos.1);
            any = true;
        }
        if !any {
            cfg.destroy();
            return;
        }
        cfg.apply();
        let _ = queue.flush();
        while state.cfg_result.is_none() && !state.cfg_cancelled && Instant::now() < deadline {
            if !pump_ctrl(conn, queue, state) {
                cfg.destroy();
                return;
            }
        }
        cfg.destroy();
        if state.cfg_cancelled {
            // Stale serial: the compositor re-announces its state with a fresh one.
            state.om_serial = None;
            continue;
        }
        if state.cfg_result != Some(true) {
            eprintln!("[HostCapture] host refused the layout; capture follows the host's own size.");
        }
        return;
    }
}

/// One bounded dispatch step for the control connection (false = connection died).
fn pump_ctrl(conn: &Connection, queue: &mut EventQueue<CtrlState>, state: &mut CtrlState) -> bool {
    if queue.dispatch_pending(state).is_err() {
        return false;
    }
    let _ = queue.flush();
    let Some(guard) = conn.prepare_read() else { return true };
    match crate::wayland::wlclient::wait_readable(
        guard.connection_fd().as_raw_fd(),
        Duration::from_millis(200),
    ) {
        Ok(readable) => {
            if readable {
                let _ = guard.read();
            } else {
                drop(guard);
            }
            true
        }
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Capture threads: one connection + screencopy loop per host output.
// ---------------------------------------------------------------------------

fn fourcc_to_gbm(fourcc: u32) -> GbmFormat {
    // XR24 / AR24; anything else falls back to ARGB (the encoder reads BGRA bytes
    // and ignores alpha).
    match fourcc {
        0x34325258 => GbmFormat::Xrgb8888,
        _ => GbmFormat::Argb8888,
    }
}

/// What a control-channel drain decided while a capture wait was in progress.
enum Ctl {
    None,
    Renegotiate,
    Idle,
    Dead,
}

enum Pump {
    Done,
    Control,
    Timeout,
}

/// Dispatch events until `done(state)` holds, waiting on the Wayland socket and
/// the wake pipe together so control messages interrupt any capture wait. The
/// predicate is re-checked after every dispatch, so a wait whose condition was
/// satisfied by already-queued events returns without ever blocking.
fn pump_until(
    conn: &Connection,
    queue: &mut EventQueue<CaptureState>,
    state: &mut CaptureState,
    wake_rd: RawFd,
    timeout: Option<Duration>,
    done: impl Fn(&CaptureState) -> bool,
) -> Result<Pump, String> {
    let deadline = timeout.map(|t| Instant::now() + t);
    loop {
        queue.dispatch_pending(state).map_err(|e| format!("dispatch: {e}"))?;
        if done(state) {
            return Ok(Pump::Done);
        }
        queue.flush().map_err(|e| format!("flush: {e}"))?;
        let remaining = match deadline {
            Some(d) => match d.checked_duration_since(Instant::now()) {
                Some(r) => Some(r),
                None => return Ok(Pump::Timeout),
            },
            None => None,
        };
        let Some(guard) = conn.prepare_read() else { continue };
        let (wl, wake) = wait_readable2(guard.connection_fd().as_raw_fd(), wake_rd, remaining)?;
        if wake {
            drop(guard);
            drain_pipe(wake_rd);
            return Ok(Pump::Control);
        }
        if wl {
            let _ = guard.read();
        } else {
            drop(guard);
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(Pump::Timeout);
            }
        }
    }
}

fn capture_loop(
    display: &str,
    index: usize,
    expect_name: Option<String>,
    gbm_path: Option<std::path::PathBuf>,
    from_main: Receiver<ToHost>,
    wake_rd: OwnedFd,
    frame_tx: Sender<HostFrame>,
) -> Result<(), String> {
    let path = socket_path(display).ok_or("XDG_RUNTIME_DIR is unset")?;
    let stream = UnixStream::connect(&path).map_err(|e| format!("connect {path}: {e}"))?;
    let conn = Connection::from_socket(stream).map_err(|e| format!("wayland setup: {e}"))?;
    let mut queue = conn.new_event_queue();
    let qh = queue.handle();
    let _registry = conn.display().get_registry(&qh, ());
    let mut state = CaptureState::default();
    bounded_roundtrip(&conn, &mut queue, &mut state)?;
    // Names (wl_output v4) arrive after the binds from the first round-trip.
    bounded_roundtrip(&conn, &mut queue, &mut state)?;

    let screencopy = state.screencopy.clone().ok_or("no zwlr_screencopy_manager_v1")?;
    // Select this thread's output by the name the control connection assigned
    // it; ordering fallback only when the host names nothing.
    let by_name = expect_name.as_ref().and_then(|expect| {
        state.outputs.iter().find(|(_, n)| n.as_ref() == Some(expect)).cloned()
    });
    let (output, _name) = match by_name {
        Some(o) => o,
        None => {
            let mut order: Vec<usize> = (0..state.outputs.len()).collect();
            order.sort_by_key(|&i| output_order_key(state.outputs[i].1.as_ref(), i));
            let oi = *order
                .get(index)
                .ok_or_else(|| format!("host has no wl_output {index}"))?;
            state.outputs[oi].clone()
        }
    };
    let gbm = gbm_path
        .as_ref()
        .and_then(|p| File::options().read(true).write(true).open(p).ok())
        .and_then(|f| GbmDevice::new(f).ok());
    let wake = wake_rd.as_raw_fd();

    let mut slots: Vec<Option<SlotBuffer>> = (0..SLOTS).map(|_| None).collect();
    let mut free: Vec<usize> = (0..SLOTS).collect();
    let mut gen: u64 = 0;
    let mut announced: Option<(i32, i32)> = None;
    let mut warned_mismatch = false;
    let mut consecutive_failures = 0u32;
    let mut want: Option<(i32, i32)> = None;

    'main: loop {
        // Drain control; block while idle or out of slots.
        loop {
            let blocking = want.is_none() || free.is_empty();
            let msg = if blocking {
                match from_main.recv() {
                    Ok(m) => m,
                    Err(_) => return Ok(()),
                }
            } else {
                match from_main.try_recv() {
                    Ok(m) => m,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return Ok(()),
                }
            };
            match msg {
                ToHost::Release { gen: g, slot } => {
                    if g == gen {
                        free.push(slot);
                    }
                }
                ToHost::Idle => want = None,
                ToHost::Start { width, height } => {
                    if want != Some((width, height)) {
                        warned_mismatch = false;
                    }
                    want = Some((width, height));
                }
            }
        }
        let (want_w, want_h) = match want {
            Some(w) => w,
            None => continue,
        };

        // One frame: negotiate, attach our buffer, wait for damage. Control
        // messages (resize, idle, teardown) interrupt any of the waits.
        state.reset_frame();
        let frame = screencopy.capture_output(1, &output, &qh, ());
        loop {
            match pump_until(&conn, &mut queue, &mut state, wake, None, |s| {
                s.buffer_done || s.failed
            })? {
                Pump::Done => break,
                Pump::Control => match drain_ctl(&from_main, gen, &mut free, &mut want) {
                    Ctl::None => {}
                    Ctl::Renegotiate | Ctl::Idle => {
                        frame.destroy();
                        continue 'main;
                    }
                    Ctl::Dead => {
                        frame.destroy();
                        return Ok(());
                    }
                },
                Pump::Timeout => {}
            }
        }
        if state.failed {
            frame.destroy();
            consecutive_failures += 1;
            if consecutive_failures == 3 {
                eprintln!("[HostCapture] output {index}: repeated screencopy failures; is the output alive?");
            }
            let _ = pump_until(
                &conn,
                &mut queue,
                &mut state,
                wake,
                Some(Duration::from_millis(100)),
                |_| false,
            );
            continue;
        }

        let (fw, fh) = state
            .announce_dmabuf
            .map(|(_, w, h)| (w, h))
            .or(state.announce_shm.map(|(_, w, h, _)| (w, h)))
            .ok_or("screencopy announced no buffer type")?;
        if announced != Some((fw, fh)) {
            announced = Some((fw, fh));
            eprintln!(
                "[HostCapture] output {index} negotiated: {fw}x{fh} gbm={} dmabuf_global={} dmabuf_announce={:?} shm={:?}",
                gbm.is_some(),
                state.dmabuf.is_some(),
                state.announce_dmabuf,
                state.announce_shm,
            );
            // Size change: drop stale buffers. The generation bump makes any
            // release still in flight for the old buffers a no-op.
            for s in slots.iter_mut() {
                *s = None;
            }
            free = (0..SLOTS).collect();
            gen += 1;
        }
        if (fw, fh) != (want_w, want_h) {
            // The encoder is configured for (want_w, want_h): hold frames until
            // the host applies the resize rather than encode mismatched buffers.
            frame.destroy();
            if !warned_mismatch {
                warned_mismatch = true;
                eprintln!(
                    "[HostCapture] output {index}: waiting for host {want_w}x{want_h} (currently {fw}x{fh})."
                );
            }
            let _ = pump_until(
                &conn,
                &mut queue,
                &mut state,
                wake,
                Some(Duration::from_millis(150)),
                |_| false,
            );
            continue;
        }
        warned_mismatch = false;

        let slot_idx = *free.last().unwrap();
        if slots[slot_idx].is_none() {
            slots[slot_idx] = Some(match (&gbm, &state.dmabuf, state.announce_dmabuf) {
                (Some(dev), Some(dmabuf_global), Some((fourcc, w, h))) => {
                    let bo = dev
                        .create_buffer_object::<()>(
                            w as u32,
                            h as u32,
                            fourcc_to_gbm(fourcc),
                            BufferObjectFlags::RENDERING,
                        )
                        .map_err(|e| format!("GBM allocation {w}x{h}: {e:?}"))?;
                    let dmabuf = crate::create_dmabuf_from_bo(&bo);
                    let params = dmabuf_global.create_params(&qh, ());
                    let modifier: u64 = dmabuf.format().modifier.into();
                    for (i, handle) in dmabuf.handles().enumerate() {
                        params.add(
                            handle,
                            i as u32,
                            dmabuf.offsets().nth(i).unwrap_or(0),
                            dmabuf.strides().nth(i).unwrap_or(0),
                            (modifier >> 32) as u32,
                            (modifier & 0xffff_ffff) as u32,
                        );
                    }
                    let wl = params.create_immed(w, h, fourcc, zwp_linux_buffer_params_v1::Flags::empty(), &qh, ());
                    params.destroy();
                    SlotBuffer::Gpu { _bo: bo, dmabuf, wl }
                }
                _ => {
                    let (format, w, h, stride) =
                        state.announce_shm.ok_or("no shm fallback announced")?;
                    let shm = state.shm.clone().ok_or("host compositor lacks wl_shm")?;
                    let size = (stride * h) as usize;
                    let fd = memfd_with(&vec![0u8; size])?;
                    let pool = shm.create_pool(fd.as_fd(), size as i32, &qh, ());
                    let wl = pool.create_buffer(
                        0,
                        w,
                        h,
                        stride,
                        WEnum::<wl_shm::Format>::from(format).into_result().unwrap_or(wl_shm::Format::Xrgb8888),
                        &qh,
                        (),
                    );
                    let file = File::from(fd);
                    let map = unsafe { memmap2::MmapMut::map_mut(&file) }
                        .map_err(|e| format!("shm map: {e}"))?;
                    SlotBuffer::Cpu { _pool: pool, map: Arc::new(map), stride, format, wl }
                }
            });
        }
        free.pop();

        {
            let slot = slots[slot_idx].as_ref().unwrap();
            let wl = match slot {
                SlotBuffer::Gpu { wl, .. } => wl,
                SlotBuffer::Cpu { wl, .. } => wl,
            };
            frame.copy_with_damage(wl);
        }
        queue.flush().map_err(|e| format!("flush: {e}"))?;
        let mut aborted = false;
        loop {
            match pump_until(&conn, &mut queue, &mut state, wake, None, |s| s.ready || s.failed)? {
                Pump::Done => break,
                Pump::Control => match drain_ctl(&from_main, gen, &mut free, &mut want) {
                    Ctl::None => {}
                    Ctl::Renegotiate | Ctl::Idle => {
                        aborted = true;
                        break;
                    }
                    Ctl::Dead => {
                        frame.destroy();
                        return Ok(());
                    }
                },
                Pump::Timeout => {}
            }
        }
        frame.destroy();
        if aborted {
            free.push(slot_idx);
            continue 'main;
        }
        if state.failed {
            free.push(slot_idx);
            consecutive_failures += 1;
            if consecutive_failures == 3 {
                eprintln!("[HostCapture] output {index}: repeated screencopy failures; is the output alive?");
            }
            let _ = pump_until(
                &conn,
                &mut queue,
                &mut state,
                wake,
                Some(Duration::from_millis(50)),
                |_| false,
            );
            continue;
        }
        consecutive_failures = 0;

        let damage = std::mem::take(&mut state.damage);
        let out = match slots[slot_idx].as_ref().unwrap() {
            SlotBuffer::Gpu { dmabuf, .. } => HostFrame {
                gen,
                slot: slot_idx,
                dmabuf: Some(dmabuf.clone()),
                cpu: None,
                width: fw,
                height: fh,
                damage,
            },
            SlotBuffer::Cpu { map, stride, format, .. } => HostFrame {
                gen,
                slot: slot_idx,
                dmabuf: None,
                cpu: Some(HostCpuFrame {
                    map: map.clone(),
                    stride: *stride as usize,
                    format: *format,
                }),
                width: fw,
                height: fh,
                damage,
            },
        };
        if frame_tx.send(out).is_err() {
            return Ok(());
        }
    }
}

/// Drain the control channel from inside a capture wait: releases are applied
/// (generation-checked), and the newest Start/Idle decides the wait's fate.
fn drain_ctl(
    rx: &Receiver<ToHost>,
    gen: u64,
    free: &mut Vec<usize>,
    want: &mut Option<(i32, i32)>,
) -> Ctl {
    let mut out = Ctl::None;
    loop {
        match rx.try_recv() {
            Ok(ToHost::Release { gen: g, slot }) => {
                if g == gen {
                    free.push(slot);
                }
            }
            Ok(ToHost::Start { width, height }) => {
                if *want != Some((width, height)) {
                    *want = Some((width, height));
                    out = Ctl::Renegotiate;
                }
            }
            Ok(ToHost::Idle) => {
                *want = None;
                out = Ctl::Idle;
            }
            Err(TryRecvError::Empty) => return out,
            Err(TryRecvError::Disconnected) => return Ctl::Dead,
        }
    }
}
