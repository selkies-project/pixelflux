/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Headless Smithay compositor that stands in for a real display server so ordinary Wayland
//! clients have somewhere to render — their composited output is exactly what the capture pipeline
//! reads back and H.264-encodes. There is no monitor, KMS, or libinput in this process, so
//! everything a desktop session normally receives from hardware — an output to map windows onto, a
//! seat to deliver input to, a clipboard to share — this frontend has to synthesize itself.
//!
//! This module owns `AppState`, the single context threaded through every Smithay protocol
//! handler, and implements those handlers: `wl_compositor` commit handling with the window
//! map/configure/focus state machine, seat/keyboard/pointer/touch routing through `FocusTarget`,
//! clipboard and primary-selection bridging to Python, and the xdg-shell / layer-shell /
//! xdg-activation / decoration / fractional-scale / dmabuf wiring. It also resolves cursor images
//! to PNG for the Python callback and provides the serial and monotonic-time helpers the input
//! path stamps onto events.

use std::borrow::Cow;
use std::fs::File;
use std::time::Instant;

use gbm::{BufferObject, Device as RawGbmDevice};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Mutex;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use smithay::backend::renderer::utils::RendererSurfaceState;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::{ Buffer, Fourcc};
use smithay::backend::renderer::damage::OutputDamageTracker;
use smithay::backend::renderer::{
    Bind, ExportMem, gles::GlesRenderer, pixman::PixmanRenderer, ImportDma,
};
use smithay::backend::allocator::Modifier;
use smithay::backend::drm::DrmNode;
use smithay::output::WeakOutput;
use smithay::wayland::image_capture_source::{
    ImageCaptureSource, ImageCaptureSourceHandler, ImageCaptureSourceState,
    OutputCaptureSourceHandler, OutputCaptureSourceState,
};
use smithay::wayland::image_copy_capture::{
    BufferConstraints, CaptureFailureReason, DmabufConstraints, Frame as CopyFrame,
    FrameRef as CopyFrameRef, ImageCopyCaptureHandler, ImageCopyCaptureState,
    Session as CopySession, SessionRef as CopySessionRef,
};
use smithay::{
    delegate_image_capture_source, delegate_image_copy_capture, delegate_output_capture_source,
};
use smithay::input::dnd::{DndFocus, Source};
use std::sync::Arc;
use crate::wayland::cursor::{Cursor, CursorJob};
use crate::wayland::keymap::KeymapPolicy;
use smithay::reexports::wayland_protocols_misc::zwp_virtual_keyboard_v1::server::{
    zwp_virtual_keyboard_manager_v1::{self, ZwpVirtualKeyboardManagerV1},
    zwp_virtual_keyboard_v1::{self, ZwpVirtualKeyboardV1},
};
use smithay::reexports::wayland_server::{DataInit, Dispatch, GlobalDispatch, New};
use smithay::wayland::viewporter::ViewporterState;
use smithay::delegate_viewporter;
use smithay::wayland::pointer_warp::{PointerWarpHandler, PointerWarpManager};
use smithay::reexports::wayland_server::protocol::wl_pointer::WlPointer;
use smithay::reexports::wayland_server::protocol::wl_shm;
use smithay::wayland::relative_pointer::RelativePointerManagerState;
use smithay::wayland::pointer_constraints::{PointerConstraintsHandler, PointerConstraintsState};
use smithay::input::pointer::PointerHandle;
use smithay::wayland::single_pixel_buffer::SinglePixelBufferState;
use smithay::delegate_single_pixel_buffer;
use smithay::desktop::{PopupKind, PopupManager};
use smithay::wayland::presentation::PresentationState;
use smithay::delegate_presentation;
use smithay::wayland::foreign_toplevel_list::{
    ForeignToplevelHandle, ForeignToplevelListHandler, ForeignToplevelListState,
};
use smithay::wayland::shell::xdg::decoration::{
    XdgDecorationHandler, XdgDecorationState,
};
use smithay::desktop::{layer_map_for_output, LayerSurface as DesktopLayerSurface};
use smithay::wayland::shell::wlr_layer::{
    WlrLayerShellHandler, WlrLayerShellState, Layer as WlrLayer, LayerSurface as WlrLayerSurface,
};
use smithay::delegate_layer_shell;
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode;
use smithay::{delegate_foreign_toplevel_list, delegate_xdg_decoration};
use smithay::wayland::selection::wlr_data_control::{DataControlHandler, DataControlState};
use smithay::wayland::selection::ext_data_control::{
    DataControlHandler as ExtDataControlHandler, DataControlState as ExtDataControlState,
};
use smithay::wayland::cursor_shape::CursorShapeManagerState;
use smithay::{delegate_cursor_shape, delegate_ext_data_control};
use smithay::delegate_data_control;
use smithay::wayland::xdg_activation::{
    XdgActivationHandler, XdgActivationState, XdgActivationToken, XdgActivationTokenData,
};
use smithay::delegate_xdg_activation;
use smithay::wayland::selection::primary_selection::{
    set_primary_focus, PrimarySelectionHandler, PrimarySelectionState,
};
use smithay::delegate_primary_selection;

use smithay::{
    delegate_compositor, delegate_data_device, delegate_dmabuf, delegate_fractional_scale,
    delegate_output, delegate_seat, delegate_shm,
    delegate_xdg_shell, delegate_relative_pointer, delegate_pointer_warp,
    delegate_pointer_constraints,
    desktop::{Space, Window},
    input::{
        keyboard::{KeyboardTarget, KeysymHandle, ModifiersState},
        pointer::{
            AxisFrame, ButtonEvent, CursorIcon, CursorImageAttributes, CursorImageStatus, GestureHoldBeginEvent,
            GestureHoldEndEvent, GesturePinchBeginEvent, GesturePinchEndEvent,
            GesturePinchUpdateEvent, GestureSwipeBeginEvent, GestureSwipeEndEvent,
            GestureSwipeUpdateEvent, MotionEvent, PointerTarget, RelativeMotionEvent,
        },
        touch::{DownEvent, OrientationEvent, ShapeEvent, TouchTarget, UpEvent},
        Seat, SeatHandler, SeatState,
    },
    output::Output,
    reexports::{
        wayland_protocols::xdg::shell::server::xdg_toplevel::State as XdgState,
        wayland_server::{
            backend::{ClientData, ClientId, DisconnectReason, GlobalId, ObjectId},
            protocol::{wl_buffer::WlBuffer, wl_surface::WlSurface},
            Client, DisplayHandle, Resource,
        },
    },
    utils::{Clock, IsAlive, Monotonic, Serial, Rectangle, Point, Logical},
    wayland::{
        buffer::BufferHandler,
        compositor::{
            with_states, BufferAssignment, CompositorClientState, CompositorHandler,
            CompositorState, SurfaceAttributes,
        },
        dmabuf::{DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier, get_dmabuf},
        fractional_scale::{FractionalScaleHandler, FractionalScaleManagerState},
        output::{OutputHandler, OutputManagerState},
        seat::WaylandFocus,
        selection::{
            data_device::{
                request_data_device_client_selection, set_data_device_focus, DataDeviceHandler,
                DataDeviceState, WaylandDndGrabHandler,
            },
            SelectionHandler, SelectionSource, SelectionTarget,
        },
        shell::xdg::{
            PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
            XdgToplevelSurfaceData,
        },
        shm::{with_buffer_contents, ShmHandler, ShmState, BufferAccessError},
    },
};

use crate::encoders::overlay::OverlayState;
use crate::encoders::vaapi::VaapiEncoder;
use crate::nvenc::NvencEncoder;
use crate::{RustCaptureSettings, StripeState};

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

static SERIAL_COUNTER: AtomicU32 = AtomicU32::new(1);

/// Hand out the next unique, monotonically increasing Wayland event serial.
///
/// Wayland tags each event with a serial so clients and Smithay can prove a request was caused
/// by a specific event (popup grab, selection change). Every injected input event draws a fresh
/// value from this process-wide atomic counter.
///
/// # Returns
///
/// A new [`Serial`] value.
pub fn next_serial() -> Serial {
    Serial::from(SERIAL_COUNTER.fetch_add(1, Ordering::SeqCst))
}

/// Millisecond timestamp for pointer / keyboard / touch events.
///
/// Samples `CLOCK_MONOTONIC` and wraps it to a `u32` millisecond count as required by the
/// Wayland protocol for input event timestamps.
///
/// # Returns
///
/// Monotonic time in milliseconds, wrapping at `u32::MAX`.
pub fn wayland_time() -> u32 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    (ts.tv_sec as u32).wrapping_mul(1000).wrapping_add((ts.tv_nsec as u32) / 1_000_000)
}

/// Microsecond timestamp for relative-pointer motion.
///
/// Samples `CLOCK_MONOTONIC` at microsecond resolution for the higher-resolution `u64` time
/// field used by the relative-pointer protocol.
///
/// # Returns
///
/// Monotonic time in microseconds, wrapping at `u64::MAX`.
pub fn wayland_utime() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    (ts.tv_sec as u64).wrapping_mul(1_000_000).wrapping_add((ts.tv_nsec as u64) / 1_000)
}

/// The one hardware H.264 encoder session backing a capture. Only a single GPU backend is
/// ever live for a given capture, and VA-API and NVENC expose entirely different session types, so
/// this enum is what lets the render and delivery code pass around "the hardware encoder" without
/// caring which vendor path actually produced the frames.
#[allow(clippy::large_enum_variant)]
pub enum GpuEncoder {
    Vaapi(VaapiEncoder),
    Nvenc(NvencEncoder),
}

/// One capture pipeline bound to one output (display id): its settings, encoder set,
/// frame pools, delivery thread, and per-stream bookkeeping. Exactly one capture may run per
/// output; all fields mirror the pipeline strategy documented on [`AppState`], instantiated
/// per display.
pub struct WlCapture {
    pub settings: RustCaptureSettings,
    /// The Python frame callback, shared with the delivery thread. Kept here so a
    /// reconfigure the calloop performs on its own (a host that kept a different mode)
    /// can respawn delivery with the same consumer; `None` for a recorder-owned capture.
    pub callback: Option<Arc<Py<PyAny>>>,
    /// Zero-copy GPU session (GLES render + same-GPU dmabuf encode), calloop-affine;
    /// `None` whenever a readback path is active for this display.
    pub video_encoder: Option<GpuEncoder>,
    pub vaapi_state: StripeState,
    pub recording_sink: Option<Arc<crate::recording_sink::RecordingSink>>,
    pub deliver_tx: Option<std::sync::mpsc::SyncSender<Vec<crate::encoders::software::EncodedStripe>>>,
    pub deliver_join: Option<std::thread::JoinHandle<()>>,
    /// Raised at teardown so the deliver thread stops calling into Python and
    /// only drains: its exit is then bounded by the one in-flight callback.
    pub deliver_discard: Option<Arc<AtomicBool>>,
    pub pending_hw_delivery: Option<Vec<crate::encoders::software::EncodedStripe>>,
    pub pending_hw_damage: bool,
    pub encode_pool: Option<Arc<crate::WlFramePool>>,
    pub encode_join: Option<std::thread::JoinHandle<Option<GpuEncoder>>>,
    pub encode_controls: Arc<crate::WlEncodeControls>,
    pub encode_stats: Arc<crate::WlEncodeStats>,
    pub pool_last_render: Vec<u64>,
    pub render_seq: u64,
    pub pool_content_gen: Vec<u64>,
    pub content_gen: u64,
    pub frame_counter: u16,
    pub pending_force_idr: bool,
    pub needs_full_render: bool,
    /// Last tick this capture actually rendered; paces per-display fps under the one
    /// shared render timer (which fires at the fastest active capture's rate).
    pub last_tick: Option<Instant>,
    /// Consecutive zero-copy encode failures; reset by any frame that encodes cleanly.
    /// Reaching `HW_ERROR_RECOVERY_THRESHOLD` triggers recovery.
    pub hw_error_streak: u32,
    /// Whether the zero-copy session has already been rebuilt without a clean frame since.
    /// Recovery rebuilds once and demotes to readback on the next streak, so a session that
    /// constructs but never encodes cannot loop rebuilding forever.
    pub hw_rebuilt: bool,
}

impl WlCapture {
    /// Arm a keyframe on whichever path is live, plus a full render so a static
    /// screen still produces the frame. Exactly ONE flag is set: the pool encode
    /// loop consumes the atomic, every other path consumes `pending_force_idr` —
    /// setting both would leave one armed forever (the host-mode idle gate polls
    /// them every tick, so a stuck flag disables idle skipping for the session).
    pub fn request_idr(&mut self) {
        if self.encode_pool.is_some() {
            self.encode_controls.force_idr.store(true, Ordering::Relaxed);
        } else {
            self.pending_force_idr = true;
        }
        self.needs_full_render = true;
    }
}

/// One virtual output and everything sized to it: the Smithay `Output` + its advertised
/// global, its layout position, the damage tracker, render targets, and the (at most one)
/// capture bound to it. `id` is the Python-facing display key; id 0 is the primary
/// HEADLESS-1 output, which is never destroyed.
pub struct OutputNode {
    pub id: u32,
    pub output: Output,
    /// The wl_output global this node published, and `None` for a view: a view
    /// captures a rectangle of another node's output rather than a screen of its
    /// own, so it publishes nothing and the session sees one screen for all the
    /// views over it. That is what keeps a pointer grab -- a window drag -- inside
    /// one surface as it crosses from one display to the next.
    pub global: Option<GlobalId>,
    /// The node whose output this one is a view of; `None` when it owns its output.
    pub owner: Option<u32>,
    /// The rectangle size a view's damage tracker was built for, so a display that
    /// changes resolution rebuilds it rather than tracking damage for the old one.
    pub view_size: (i32, i32),
    /// The scale the view's tracker maps windows with; a screen's tracker follows its output.
    pub view_scale: f64,
    /// Layout offset: the output's logical position in the Space AND its physical offset
    /// for absolute input injection. The two coincide at scale 1; with mixed scales the
    /// caller must place outputs so neither the logical nor the physical rectangles
    /// overlap.
    pub pos: (i32, i32),
    pub damage_tracker: OutputDamageTracker,
    /// Host-side scratch target: pixman throttle path renders here, GLES screenshots read
    /// back here.
    pub frame_buffer: Vec<u8>,
    /// GPU render target for this output (GLES mode): the GBM BO and its dmabuf export.
    pub offscreen_buffer: Option<(BufferObject<()>, Dmabuf)>,
    /// Watermark overlay for THIS output: loaded at the output's scale, positioned (and,
    /// for the bouncing anchor, animated) against the output's own frame dimensions.
    pub overlay_state: OverlayState,
    pub capture: Option<WlCapture>,
    /// Monotonic count of frames reported presented through wp_presentation.
    pub frame_seq: u64,
    /// Whether the render target already holds a fully composited frame, so a
    /// capture-less render (screenshot, copy-capture client) may use buffer age
    /// instead of redrawing everything. Cleared whenever the target is replaced.
    pub target_seeded: bool,
    /// Deadline for this output's post-reconfigure content hold. While it is set, the
    /// display composites but does not publish: a resized or freshly created output paints
    /// its clear colour wherever a client has not yet answered the new size, and that grey
    /// must not reach the stream. Cleared by the first tick a client covers the output, and
    /// by the deadline for one that never does.
    pub content_hold_until: Option<Instant>,
}

/// A host-capture layout request in flight for one display: the epoch of the apply it
/// rode on, the size the capture was configured for, and the geometry readers parked
/// behind it, answered once the host has decided so they report the size actually
/// captured (the realized-geometry barrier).
pub struct PendingHostLayout {
    pub epoch: u64,
    pub want: (i32, i32),
    pub geometry_waiters: Vec<std::sync::mpsc::Sender<(i32, i32, f64)>>,
}

/// One ext-image-copy-capture session: an external client capturing one of this
/// compositor's outputs. A requested frame parks in `pending` until the render loop
/// has content for it, so delivery happens at most once per composited frame and an
/// unchanged screen holds the frame instead of duplicating it.
pub struct OutputCopySession {
    pub session: CopySession,
    pub output: WeakOutput,
    pub pending: Option<CopyFrame>,
    /// The first frame of a session ships without waiting for damage; only
    /// afterwards does an unchanged screen hold the frame.
    pub delivered_once: bool,
}

/// Every window placed on `display_id` and actually composited there. Keyed off the window's
/// own output tag rather than the Space's output map, which only catches up on the next
/// `Space::refresh`; a parked window carries the tag but is mapped clear of every output.
pub fn windows_on_output(space: &Space<Window>, display_id: u32) -> impl Iterator<Item = &Window> {
    space.elements().filter(move |w| {
        window_output_id(w) == display_id
            && !window_meta(w).map(|m| m.parked.load(Ordering::Relaxed)).unwrap_or(false)
    })
}

/// Whether a client on this display has COMMITTED content spanning the whole output. Under
/// forced fullscreen a window answers its configure at the output's logical size, so one
/// still carrying its pre-configure size reads as not covering and holds the display's frames.
pub fn output_content_covers(
    space: &Space<Window>,
    display_id: u32,
    logical_w: f64,
    logical_h: f64,
) -> bool {
    windows_on_output(space, display_id).any(|w| {
        let size = w.geometry().size;
        size.w as f64 >= logical_w && size.h as f64 >= logical_h
    })
}

impl OutputNode {
    /// The output's logical geometry in layout coordinates: origin plus mode/scale-derived
    /// logical size. A view's size is its own rectangle -- the mode belongs to the
    /// screen it is cut from.
    pub fn logical_geometry(&self) -> Option<Rectangle<i32, Logical>> {
        let scale = self.output.current_scale().fractional_scale();
        let (w, h) = if self.owner.is_some() {
            self.capture
                .as_ref()
                .map(|c| (c.settings.width, c.settings.height))
                .unwrap_or(self.view_size)
        } else {
            let mode = self.output.current_mode()?;
            (mode.size.w, mode.size.h)
        };
        Some(Rectangle::new(
            Point::from(self.pos),
            (
                (w as f64 / scale).round() as i32,
                (h as f64 / scale).round() as i32,
            )
                .into(),
        ))
    }
}

static NEXT_WINDOW_ID: AtomicU32 = AtomicU32::new(1);

/// Per-window bookkeeping carried in the window's user-data map: a stable numeric id the
/// Python side addresses the window by, the display id of the output it is placed on, and
/// whether that output has been chosen yet.
pub struct WindowMeta {
    pub id: u32,
    pub output: AtomicU32,
    /// Set once the first commit has picked the window's output. The choice cannot key off
    /// xdg's `initial_configure_sent`: a client that negotiates xdg-decoration (every
    /// wlroots-based nested compositor, GTK and Qt) is answered with a configure before it
    /// ever commits, which would leave every window on the pointer's output.
    pub placed: AtomicBool,
    /// Set while the window is mapped outside every output, tagged to the output it will
    /// take once one exists. See `park_window`.
    pub parked: AtomicBool,
}

/// Where a parked window is mapped: clear of every output, so it is composited into none of
/// them. Layout offsets are non-negative — the union layout Selkies computes re-anchors at
/// the origin — so far negative coordinates can never collide with a real output.
pub const PARKED_POS: (i32, i32) = (-(1 << 20), -(1 << 20));

/// The logical size a parked screen is held at. A nested session lays its desktop out across
/// every screen it has, including one waiting here: at the size of a real screen it would
/// double the session's coordinate space, sending anything a client centres on the desktop
/// (X11 applications place themselves) onto the screen nobody is watching. Small enough not
/// to move that centre, large enough to lay out on; `place_window_on_output` configures the
/// real size the moment the screen is given an output.
pub const PARKED_LOGICAL_SIZE: (i32, i32) = (320, 240);

/// The window's meta, inserted at `new_toplevel`; windows created before that (none in
/// practice) read as id 0 / primary output.
pub fn window_meta(window: &Window) -> Option<&WindowMeta> {
    window.user_data().get::<WindowMeta>()
}

/// The display id of the output this window is placed on (primary when untagged).
pub fn window_output_id(window: &Window) -> u32 {
    window_meta(window).map(|m| m.output.load(Ordering::Relaxed)).unwrap_or(0)
}

/// A queued computer-use screenshot as `(display id, reply)`; the reply carries the
/// encoded image or the reason it could not be produced.
pub type ScreenshotRequest = (u32, std::sync::mpsc::Sender<Result<Vec<u8>, String>>);

/// Central context threaded through every Smithay handler; owns the Wayland globals, the
/// GBM/EGL (or pixman) renderer state, and the capture/encode pipeline state.
///
/// A single `AppState` lives for the whole compositor thread and is mutated in place by the
/// calloop event sources (client dispatch, input injection, the capture timer). The frame
/// pipeline is instantiated PER DISPLAY: each `OutputNode` in `output_nodes` owns its output,
/// damage tracker, and render targets, and its optional `WlCapture` owns that display's
/// encoder set and delivery. The non-obvious `WlCapture`/`OutputNode` fields encode the
/// pipeline's threading and buffer strategy:
///
/// 1. **Render / encode targets** (per display):
///    - **`video_encoder`**: the zero-copy GPU session only (GLES render + same-GPU dmabuf
///      encode). Its EGL/dmabuf handles are calloop-affine, so this encoder runs inline on the
///      calloop thread; it is `None` whenever a readback path is active.
///    - **`encode_pool` / `encode_join`**: the readback capture||encode split. The calloop renders
///      and reads back into a pooled host buffer and publishes it, while a separate encode thread
///      owns the CPU / cross-GPU / pixman-HW encoders. `encode_join` returns that thread's hardware
///      session on shutdown so a restart can reconfigure it in place rather than rebuild it. Both
///      are `None` while a zero-copy GPU session runs.
///    - **`frame_buffer`**: scratch render target used only by the pixman memory-throttle path; the
///      normal readback path renders and reads back into the pooled buffers instead.
///    - **`pool_last_render` / `render_seq`**: buffer-age bookkeeping for the pixman path, which
///      renders directly into the pooled buffers (an age is "renders since this slot was last the
///      target"). The GLES path renders into one fixed offscreen buffer and never consults these.
///    - **`pool_content_gen` / `content_gen`**: staleness bookkeeping for the GLES readback path,
///      which skips the GPU readback on no-damage ticks. `content_gen` advances whenever a render
///      reports damage; a pooled buffer whose stamp lags it holds pre-damage pixels and must be
///      read back once before publishing, so the encoder's paint-over / burst / recovery sends
///      never ship stale content.
///
/// 2. **Delivery** (`deliver_tx` / `deliver_join`): encoded frames go to a dedicated delivery
///    thread over a capacity-1 rendezvous channel, mirroring the X11 single-slot FramePool
///    (non-dropping, ordered, at most one frame of blocking backpressure), so a slow GIL-holding
///    Python callback can never stall input/control dispatch on the calloop thread.
///
/// 3. **Keyframes** (`pending_force_idr`): set by an IDR request (client reconnect / decoder reset)
///    and consumed once on the next captured frame to force an immediate keyframe.
///
/// 4. **Clipboard** (`pending_clipboard_read`): stages the mime chosen in `new_selection`; the loop
///    drains it only after the dispatch that stores the new client source, so the read targets the
///    new selection rather than the previous one.
///
/// 5. **GPU selection** (`auto_gpu_selected`): records that automatic (not explicit) selection
///    picked `render_node_path`, so `StartCapture` aims the encoder at that same node unless a
///    device was chosen explicitly.
pub struct AppState {
    pub compositor_state: CompositorState,
    pub fractional_scale_state: FractionalScaleManagerState,
    pub viewporter_state: ViewporterState,
    pub presentation_state: PresentationState,
    pub shm_state: ShmState,
    pub single_pixel_buffer: SinglePixelBufferState,
    pub dmabuf_state: DmabufState,
    pub dmabuf_global: Option<DmabufGlobal>,
    pub ext_data_control_state: ExtDataControlState,
    pub cursor_shape_state: CursorShapeManagerState,
    pub image_capture_source_state: ImageCaptureSourceState,
    pub output_capture_source_state: OutputCaptureSourceState,
    pub image_copy_capture_state: ImageCopyCaptureState,
    pub copy_sessions: Vec<OutputCopySession>,
    #[allow(dead_code)]
    pub output_state: OutputManagerState,
    pub seat_state: SeatState<AppState>,
    pub shell_state: XdgShellState,
    pub layer_shell_state: WlrLayerShellState,
    pub space: Space<Window>,
    pub data_device_state: DataDeviceState,
    pub data_control_state: DataControlState,
    pub dh: DisplayHandle,
    #[allow(dead_code)]
    pub seat: Seat<AppState>,
    /// Every live output with its per-display render/capture state; index 0 is the
    /// primary (display id 0), which is never destroyed.
    pub output_nodes: Vec<OutputNode>,
    pub pending_windows: Vec<Window>,

    pub foreign_toplevel_list: ForeignToplevelListState,
    pub xdg_decoration_state: XdgDecorationState,
    pub xdg_activation_state: XdgActivationState,
    pub primary_selection_state: PrimarySelectionState,
    pub popups: PopupManager,

    pub gles_renderer: Option<GlesRenderer>,
    pub pixman_renderer: Option<PixmanRenderer>,

    pub gbm_device: Option<RawGbmDevice<File>>,

    /// Mirror of the PRIMARY display's capture settings (geometry fallbacks, computer-use
    /// info); per-display settings live on each capture.
    pub settings: RustCaptureSettings,
    /// A cursor callback is registered on the `wl-cursor` worker (which owns the actual
    /// `Py` object); tracked here so sprite resolution is skipped while nobody listens.
    pub cursor_callback_set: bool,
    /// Cursor delivery jobs to the `wl-cursor` worker (PNG encode + Python call off-thread).
    pub cursor_tx: std::sync::mpsc::Sender<CursorJob>,
    pub clipboard_callback: Option<Py<PyAny>>,
    pub pending_clipboard_read: Option<String>,
    /// Preferred mime of the current CLIENT-owned clipboard selection, recorded even while
    /// no callback is registered so `SetClipboardCallback` can re-stage a read of a copy made
    /// in the gap; `None` when the selection is cleared or compositor-owned.
    pub current_selection_mime: Option<String>,

    pub last_log_time: Instant,
    pub start_time: Instant,
    pub clock: Clock<Monotonic>,

    pub use_gpu: bool,

    pub cursor_helper: Cursor,

    /// Seat keymap owner: base layout plus batched overlay binds. Every seat keymap swap
    /// flows through this policy, so keymap identity has exactly one writer.
    pub keymap_policy: KeymapPolicy,
    /// Host-capture session when pixelflux captures an EXTERNAL compositor:
    /// frames arrive by screencopy and input routes to its virtual devices.
    pub host: Option<crate::wayland::host::HostSession>,
    /// Per display, the layout request the host has not answered yet. A capture start
    /// submits its mode and carries on at that size; the render tick polls the verdict
    /// and re-sizes the capture to the host's own mode when the host kept it. Emptied
    /// with the host session.
    pub host_layout_pending: std::collections::HashMap<u32, PendingHostLayout>,

    pub current_cursor_icon: Option<CursorImageStatus>,
    /// A surface-backed cursor set during the dispatch in progress and not delivered yet:
    /// clients commit the sprite's buffer after `set_cursor` in the same batch, so delivery
    /// waits for that commit, or for the end of the dispatch when none comes, rather than
    /// shipping the surface's previous sprite under the new hotspot.
    pub cursor_surface_pending: bool,
    pub cursor_buffer: Option<WlBuffer>,
    pub render_cursor_on_framebuffer: bool,
    pub pointer_warp_state: PointerWarpManager,
    pub relative_pointer_state: RelativePointerManagerState,
    pub pointer_constraints_state: PointerConstraintsState,
    pub render_node_path: String,
    pub auto_gpu_selected: bool,
    /// Computer-use screenshot request; served from that output's next render (the id
    /// was validated live when the request was queued).
    pub pending_screenshot: Option<ScreenshotRequest>,
    /// The command channel, drained in place (wakeups arrive on a separate ping channel) so
    /// the render tick can apply every queued command BEFORE starting a long render/encode —
    /// queued input is never starved behind the tick it arrived during.
    pub command_rx: Option<smithay::reexports::calloop::channel::Channel<crate::ThreadCommand>>,
    /// When the last input command landed: the no-capture tick speeds up right
    /// after input so apps pacing on frame callbacks respond promptly (they
    /// are the ones noticing the idle rate while nobody encodes).
    pub last_input_at: Option<Instant>,
    /// Whether the frame timer is armed at the long idle interval: input
    /// arriving then must not wait that deadline out, so the wake handler
    /// sends the idle frame callbacks itself (frame-paced via
    /// `last_idle_service_at`) instead of waiting for the timer.
    pub frame_idle_long: bool,
    /// When idle frame callbacks were last serviced, bounding the wake
    /// handler's servicing to frame pace under an input storm.
    pub last_idle_service_at: Option<Instant>,
    /// Deliver threads of stopped captures, joined without blocking the event
    /// loop: the timer tick reaps the finished ones, and the shutdown Barrier
    /// drains the rest so nothing that can attach to Python outlives it.
    pub deliver_reaper: Vec<std::thread::JoinHandle<()>>,
    /// Encode threads whose teardown-time harvest expired (wedged behind a
    /// consumer inside Python); reaped like the deliver threads, with the
    /// unclaimed encoder dropped here on the event loop — its usual home.
    pub encode_reaper: Vec<std::thread::JoinHandle<Option<crate::GpuEncoder>>>,
}

/// Pointer-constraints protocol wiring. The headless capture path never enforces a lock or
/// confinement region, so activation and cursor-position hints are accepted as no-ops; the global
/// still exists so clients may bind it without error.
impl PointerConstraintsHandler for AppState {
    fn new_constraint(&mut self, _surface: &WlSurface, _pointer: &PointerHandle<Self>) {}

    fn cursor_position_hint(
        &mut self,
        _surface: &WlSurface,
        _pointer: &PointerHandle<Self>,
        _location: Point<f64, Logical>,
    ) {}
}

/// Foreign-toplevel-list protocol: exposes the managed state so Smithay can advertise each
/// toplevel (title / app-id) to listing clients such as taskbars.
impl ForeignToplevelListHandler for AppState {
    fn foreign_toplevel_list_state(&mut self) -> &mut ForeignToplevelListState {
        &mut self.foreign_toplevel_list
    }
}

/// xdg-activation protocol: hands out activation tokens and, on redemption, raises the
/// target window.
///
/// `token_created` accepts every token. `request_activation` honors a token only while it is fresh
/// (issued less than 10 seconds ago) so a stale or replayed token cannot steal focus, then raises
/// the window whose surface matches to the top of the space.
impl XdgActivationHandler for AppState {
    fn activation_state(&mut self) -> &mut XdgActivationState {
        &mut self.xdg_activation_state
    }

    fn token_created(&mut self, _token: XdgActivationToken, _data: XdgActivationTokenData) -> bool {
        true
    }

    fn request_activation(
        &mut self,
        _token: XdgActivationToken,
        token_data: XdgActivationTokenData,
        surface: WlSurface,
    ) {
        if token_data.timestamp.elapsed().as_secs() < 10 {
            let window = self.space.elements().find(|w| w.wl_surface().as_deref() == Some(&surface)).cloned();
            if let Some(window) = window {
                self.space.raise_element(&window, true);
            }
        }
    }
}

/// Carry the X11-style primary selection so middle-click paste works between clients.
/// Exposing the managed state lets Smithay record which client currently owns the primary selection;
/// `focus_changed` then keeps that ownership tracking keyboard focus, so a middle-click pastes from
/// whichever client the user is actually working in.
impl PrimarySelectionHandler for AppState {
    fn primary_selection_state(&mut self) -> &mut PrimarySelectionState {
        &mut self.primary_selection_state
    }
}

/// Default every toplevel to server-side decorations so clients don't bake their own title
/// bars and borders into the captured image.
///
/// The frontend shows fullscreen application content with no window-manager chrome, so a client
/// left to draw client-side decorations would paint a stray title bar straight into the encoded
/// frame. Pinning the negotiation to `Mode::ServerSide` (in `new_decoration`, and again when a
/// client unsets its preference in `unset_mode`) hands decoration duty to the compositor — which
/// draws none — leaving clean, borderless output; `request_mode` still grants a mode a client
/// explicitly insists on. Each path acknowledges with a configure.
impl XdgDecorationHandler for AppState {
    fn new_decoration(&mut self, toplevel: ToplevelSurface) {
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(Mode::ServerSide);
        });
        toplevel.send_configure();
    }

    fn request_mode(&mut self, toplevel: ToplevelSurface, mode: Mode) {
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(mode);
        });
        toplevel.send_configure();
    }

    fn unset_mode(&mut self, toplevel: ToplevelSurface) {
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(Mode::ServerSide);
        });
        toplevel.send_configure();
    }
}

/// wlr-layer-shell protocol: places panels / overlays / backgrounds (layer surfaces).
///
/// `new_layer_surface` resolves the requested output (or the first output), configures the surface
/// to that output's full pixel size, and maps it into the output's layer map so the render loop
/// composites it in the correct z-order. `layer_destroyed` needs no action here.
impl WlrLayerShellHandler for AppState {
    fn shell_state(&mut self) -> &mut WlrLayerShellState {
        &mut self.layer_shell_state
    }

    fn new_layer_surface(
        &mut self,
        surface: WlrLayerSurface,
        output: Option<smithay::reexports::wayland_server::protocol::wl_output::WlOutput>,
        _layer: WlrLayer,
        namespace: String,
    ) {
        let smithay_output = if let Some(wlo) = output.as_ref() {
            self.output_nodes.iter().map(|n| &n.output).find(|o| o.owns(wlo))
        } else {
            self.primary_output()
        };

        if let Some(output) = smithay_output {
            let mode = output.current_mode().unwrap();
            
            surface.with_pending_state(|state| {
                state.size = Some(((mode.size.w as f64) as i32, (mode.size.h as f64) as i32).into());
            });
            surface.send_configure();

            let layer = DesktopLayerSurface::new(surface, namespace);
            let _ = layer_map_for_output(output).map_layer(&layer);
        }
    }

    fn layer_destroyed(&mut self, _surface: WlrLayerSurface) {}
}

/// Core `wl_compositor` protocol: per-surface commit handling plus the window
/// map/configure/focus state machine.
impl CompositorHandler for AppState {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }
    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ClientState>().unwrap().compositor_state
    }

    /// Fold a client's `wl_surface.commit` into compositor state and — the reason the window
    /// logic lives in this handler — walk a brand-new toplevel through the xdg-shell handshake it
    /// must finish before it may be shown.
    ///
    /// A window cannot just appear on its first commit: xdg-shell requires the compositor to send an
    /// initial configure (telling the client the size and state to draw at) and the client to ack it
    /// and commit a buffer matching that size before the surface counts as mapped. Mapping earlier
    /// would flash an unconfigured, wrongly-sized window into the captured frame. So a pending
    /// toplevel is carried across two commits instead of one, while everything else here is
    /// per-commit housekeeping that must run whether or not a map is in flight. On each commit, in
    /// order:
    ///
    /// 1. **Buffer intake**: `on_commit_buffer_handler` ingests the newly attached buffer.
    /// 2. **Layer relayout**: if the surface is a mapped layer surface, re-arrange the output's
    ///    layer map so geometry tracks the new content.
    /// 3. **Cursor refresh**: if the surface backs the current cursor, re-send its image so a client
    ///    animating its own cursor surface is reflected downstream.
    /// 4. **Foreign-toplevel metadata**: push the current title / app-id to the foreign-toplevel
    ///    handle for taskbar-style clients.
    /// 5. **Window on-commit**: forward the commit to the matching mapped `Window`.
    /// 6. **Two-phase map of a pending toplevel**:
    ///    - **First commit — no initial configure sent yet**: this commit is the client announcing
    ///      it wants to be shown, so compute the logical size from the output mode/scale (falling
    ///      back to the settings resolution), send a fullscreen + activated configure, and re-queue
    ///      the window to wait for the client's acked commit — nothing is mapped yet.
    ///    - **Acked commit — the client has drawn to that configure**: map the element at the origin,
    ///      refresh its cached bounding box *before* reading geometry (so the drift check sees
    ///      current geometry and doesn't fire a redundant configure), enter the output, and, only if
    ///      the client's geometry still drifts more than a pixel from the expected fullscreen size,
    ///      send one corrective configure. Finally give the new window keyboard focus so input lands
    ///      on it at once.
    fn commit(&mut self, surface: &WlSurface) {
        smithay::backend::renderer::utils::on_commit_buffer_handler::<Self>(surface);

        for node in &self.output_nodes {
            let mut layer_map = layer_map_for_output(&node.output);
            let found = layer_map.layers().any(|layer| layer.wl_surface() == surface);
            if found {
                layer_map.arrange();
                break;
            }
        }

        if let Some(CursorImageStatus::Surface(ref cursor_surface)) = self.current_cursor_icon
            && cursor_surface == surface {
                let status = CursorImageStatus::Surface(surface.clone());
                self.cursor_surface_pending = false;
                self.send_cursor_image(&status);
            }

        if let Some(handle) = with_states(surface, |states| states.data_map.get::<ForeignToplevelHandle>().cloned())
             && let Some(window) = self.space.elements().find(|w| w.wl_surface().as_deref() == Some(surface))
                 && let Some(_toplevel) = window.toplevel() {
                     let (title, app_id) = with_states(surface, |states| {
                        let attributes = states.data_map.get::<XdgToplevelSurfaceData>().unwrap().lock().unwrap();
                        (attributes.title.clone(), attributes.app_id.clone())
                     });
                     
                     handle.send_title(&title.unwrap_or_default());
                     handle.send_app_id(&app_id.unwrap_or_default());
                     handle.send_done();
                 }

        let mapped = self
            .space
            .elements()
            .find(|w| w.toplevel().map(|tl| tl.wl_surface() == surface).unwrap_or(false))
            .cloned();
        if let Some(window) = mapped {
            window.on_commit();
            // A null-buffer commit unmaps the toplevel (xdg-shell): purge it from the
            // space so it no longer lists or renders, and re-queue it so a client that
            // maps again goes back through the configure handshake.
            let has_buffer = smithay::backend::renderer::utils::with_renderer_surface_state(
                surface,
                |s| s.buffer().is_some(),
            )
            .unwrap_or(false);
            if !has_buffer {
                self.space.unmap_elem(&window);
                self.pending_windows.push(window);
                return;
            }
        }

        if let Some(idx) = self.pending_windows.iter().position(|w| {
            w.toplevel().map(|tl| tl.wl_surface() == surface).unwrap_or(false)
        }) {
            let window = self.pending_windows.remove(idx);
            let toplevel = window.toplevel().unwrap();

            // Whether this window still needs an output. Deliberately not xdg's
            // `initial_configure_sent`: the decoration handshake answers with a configure
            // before the client's first commit, which would skip the choice below for
            // every client that negotiates decorations.
            let needs_placement = window_meta(&window)
                .map(|meta| !meta.placed.swap(true, Ordering::Relaxed))
                .unwrap_or(false);

            if needs_placement {
                // A new toplevel opens fullscreened on the output the pointer is on
                // (primary when indeterminate); the choice is pinned on the window's
                // meta so the acked commit maps to the same output.
                let mut target_id = self.pointer_display();
                // A second fullscreen surface from a client that already owns one on
                // the target output is screen-like (a nested compositor opens one
                // host toplevel per screen): give it an empty output when one exists,
                // and park it otherwise rather than let it cover the screen already
                // there. `create_output` hands it the output it is waiting for.
                let mut park = false;
                if self.would_cover_screen(&window, target_id) {
                    // Screens only: a view is a capture over someone else's
                    // screen, and no window is ever tagged to one, so a view
                    // always reads as empty and would swallow the screen-like
                    // toplevel that `create_output` later needs parked.
                    let empty = self
                        .output_nodes
                        .iter()
                        .filter(|n| n.owner.is_none())
                        .map(|n| n.id)
                        .find(|oid| {
                            !self
                                .space
                                .elements()
                                .chain(self.pending_windows.iter())
                                .any(|w| window_output_id(w) == *oid)
                        });
                    match empty {
                        Some(oid) => target_id = oid,
                        None => park = true,
                    }
                }
                if let Some(meta) = window_meta(&window) {
                    meta.output.store(target_id, Ordering::Relaxed);
                    meta.parked.store(park, Ordering::Relaxed);
                }
                let (logical_width, logical_height) = if park {
                    PARKED_LOGICAL_SIZE
                } else if let Some(size) = self.logical_size_of(target_id) {
                    size
                } else {
                    let scale = self.settings.scale.max(0.1);
                    (
                        (self.settings.width as f64 / scale).round() as i32,
                        (self.settings.height as f64 / scale).round() as i32,
                    )
                };

                toplevel.with_pending_state(|state| {
                    state.states.set(XdgState::Activated);
                    state.states.set(XdgState::Fullscreen);
                    state.size = Some((logical_width, logical_height).into());
                });
                toplevel.send_configure();

                self.pending_windows.push(window);
            } else if smithay::backend::renderer::utils::with_renderer_surface_state(
                surface,
                |s| s.buffer().is_none(),
            )
            .unwrap_or(true)
            {
                // Configured but still buffer-less (a decoration-triggered configure, an
                // ack-only commit, or a remap after a null-buffer unmap — xdg
                // initial_configure_sent stays true there): keep waiting; mapping now would
                // list and hit-test a phantom window. Answer with the forced-fullscreen
                // configure so the client draws its first buffer at the right geometry.
                let tl = toplevel.clone();
                self.pending_windows.push(window);
                self.send_forced_fullscreen_configure(&tl);
            } else {
                let target_id = window_output_id(&window);
                let parked = window_meta(&window)
                    .map(|meta| meta.parked.load(Ordering::Relaxed))
                    .unwrap_or(false);
                let node_idx = self.node_idx_for_id(target_id).unwrap_or(0);
                let (target_output, pos) = {
                    let node = &self.output_nodes[node_idx];
                    (node.output.clone(), node.pos)
                };
                // A parked window enters no output and takes no focus: it is waiting for
                // an output of its own, and holds its size until one arrives.
                self.space.map_element(
                    window.clone(),
                    if parked { PARKED_POS } else { pos },
                    !parked,
                );
                window.on_commit();

                if !parked {
                    target_output.enter(surface);
                    let scale = target_output.current_scale().fractional_scale();
                    with_states(surface, |states| {
                        smithay::wayland::compositor::send_surface_state(
                            surface, states, scale.ceil() as i32, smithay::utils::Transform::Normal,
                        );
                        smithay::wayland::fractional_scale::with_fractional_scale(states, |fs| {
                            fs.set_preferred_scale(scale);
                        });
                    });

                    if let Some(geo_out) = self.output_nodes[node_idx].logical_geometry() {
                        let (expected_w, expected_h) = (geo_out.size.w, geo_out.size.h);
                        let geo = window.geometry();
                        if (geo.size.w - expected_w).abs() > 1
                            || (geo.size.h - expected_h).abs() > 1
                        {
                            toplevel.with_pending_state(|state| {
                                state.states.set(XdgState::Activated);
                                state.states.set(XdgState::Fullscreen);
                                state.size = Some((expected_w, expected_h).into());
                            });
                            toplevel.send_configure();
                        }
                    }

                    let serial = next_serial();
                    let target = FocusTarget::Window(window.clone());
                    if let Some(keyboard) = self.seat.get_keyboard() {
                        keyboard.set_focus(self, Some(target.clone()), serial);
                    }
                } else {
                    // A client that drew before answering the parked size (a nested
                    // compositor's spare screen starts at its backend's own default)
                    // is told again now that it has mapped, so the session lays out
                    // on a screen the size of the one it is waiting for.
                    let geo = window.geometry();
                    if (geo.size.w - PARKED_LOGICAL_SIZE.0).abs() > 1
                        || (geo.size.h - PARKED_LOGICAL_SIZE.1).abs() > 1
                    {
                        let tl = toplevel.clone();
                        self.send_forced_fullscreen_configure(&tl);
                    }
                }
            }
        }
    }
}


impl AppState {
    /// The primary (display id 0) output.
    pub(crate) fn primary_output(&self) -> Option<&Output> {
        self.output_nodes.first().map(|n| &n.output)
    }

    pub(crate) fn node_idx_for_id(&self, id: u32) -> Option<usize> {
        self.output_nodes.iter().position(|n| n.id == id)
    }

    /// The ids of the views onto `id`'s output.
    pub(crate) fn view_ids_of(&self, id: u32) -> Vec<u32> {
        self.output_nodes
            .iter()
            .filter(|n| n.owner == Some(id))
            .map(|n| n.id)
            .collect()
    }

    /// The node whose output `id` draws from: itself, or the node it views.
    pub(crate) fn screen_id_of(&self, id: u32) -> u32 {
        self.node_idx_for_id(id)
            .and_then(|i| self.output_nodes[i].owner)
            .unwrap_or(id)
    }

    /// Index of the node whose LOGICAL rect contains `p`.
    pub(crate) fn node_idx_under(&self, p: Point<f64, Logical>) -> Option<usize> {
        self.output_nodes.iter().position(|n| {
            n.logical_geometry()
                .map(|g| g.to_f64().contains(p))
                .unwrap_or(false)
        })
    }

    /// Map absolute PHYSICAL union-layout coordinates to a logical layout point: each
    /// output occupies the physical rectangle at its layout offset sized by its mode; the
    /// point is clamped into the nearest output when it falls outside all of them, so the
    /// pointer can never leave the layout.
    pub(crate) fn layout_physical_to_logical(&self, x: f64, y: f64) -> Point<f64, Logical> {
        let mut best: Option<(f64, Point<f64, Logical>)> = None;
        for node in self.output_nodes.iter().filter(|n| n.owner.is_none()) {
            let Some(mode) = node.output.current_mode() else { continue };
            let scale = node.output.current_scale().fractional_scale();
            let (px, py) = (node.pos.0 as f64, node.pos.1 as f64);
            let cx = x.max(px).min(px + mode.size.w as f64 - 1.0);
            let cy = y.max(py).min(py + mode.size.h as f64 - 1.0);
            let d2 = (x - cx).powi(2) + (y - cy).powi(2);
            let logical = Point::from((
                node.pos.0 as f64 + (cx - px) / scale,
                node.pos.1 as f64 + (cy - py) / scale,
            ));
            if best.as_ref().map(|(bd, _)| d2 < *bd).unwrap_or(true) {
                best = Some((d2, logical));
            }
        }
        best.map(|(_, p)| p).unwrap_or_else(|| (0.0, 0.0).into())
    }

    /// Clamp a logical layout point into the nearest output's logical rectangle.
    pub(crate) fn clamp_logical(&self, p: Point<f64, Logical>) -> Point<f64, Logical> {
        let mut best: Option<(f64, Point<f64, Logical>)> = None;
        for node in self.output_nodes.iter().filter(|n| n.owner.is_none()) {
            let Some(geo) = node.logical_geometry() else { continue };
            let scale = node.output.current_scale().fractional_scale();
            let g = geo.to_f64();
            let margin = 1.0 / scale.max(0.1);
            let cx = p.x.max(g.loc.x).min(g.loc.x + g.size.w - margin);
            let cy = p.y.max(g.loc.y).min(g.loc.y + g.size.h - margin);
            let d2 = (p.x - cx).powi(2) + (p.y - cy).powi(2);
            if best.as_ref().map(|(bd, _)| d2 < *bd).unwrap_or(true) {
                best = Some((d2, (cx, cy).into()));
            }
        }
        best.map(|(_, p)| p).unwrap_or(p)
    }

    /// Logical layout point -> physical union-layout coordinates (inverse of
    /// `layout_physical_to_logical` for in-bounds points; primary-relative otherwise).
    pub(crate) fn layout_logical_to_physical(&self, p: Point<f64, Logical>) -> (f64, f64) {
        let idx = self.node_idx_under(p).unwrap_or(0);
        let Some(node) = self.output_nodes.get(idx) else { return (p.x, p.y) };
        let scale = node.output.current_scale().fractional_scale();
        (
            node.pos.0 as f64 + (p.x - node.pos.0 as f64) * scale,
            node.pos.1 as f64 + (p.y - node.pos.1 as f64) * scale,
        )
    }

    /// Logical size of the given display's output.
    pub(crate) fn logical_size_of(&self, id: u32) -> Option<(i32, i32)> {
        let idx = self.node_idx_for_id(id)?;
        let geo = self.output_nodes[idx].logical_geometry()?;
        Some((geo.size.w, geo.size.h))
    }

    /// The display id under the pointer (primary when indeterminate).
    pub(crate) fn pointer_display(&self) -> u32 {
        self.seat
            .get_pointer()
            .map(|p| p.current_location())
            .and_then(|pos| self.node_idx_under(pos))
            .map(|idx| self.output_nodes[idx].id)
            .unwrap_or(0)
    }

    /// Whether `window` would land on top of a screen another window from the same client
    /// already occupies. A nested session opens one host toplevel per screen, and a second
    /// one covering the first would replace what the display shows rather than add to it.
    pub(crate) fn would_cover_screen(&self, window: &Window, id: u32) -> bool {
        let Some(client) = window.wl_surface().and_then(|s| s.client()) else { return false };
        self.space
            .elements()
            .chain(self.pending_windows.iter())
            .filter(|w| !std::ptr::eq(*w, window) && w.wl_surface() != window.wl_surface())
            .any(|w| {
                window_output_id(w) == id
                    && !window_meta(w).map(|m| m.parked.load(Ordering::Relaxed)).unwrap_or(false)
                    && w.wl_surface()
                        .and_then(|s| s.client())
                        .is_some_and(|c| c.id() == client.id())
            })
    }

    /// Map `window` clear of every output while keeping it tagged for `id`, so it holds its
    /// size and its place in the window list without being composited anywhere. A nested
    /// session's spare screens wait here until `create_output` gives them one.
    pub(crate) fn park_window(&mut self, window: &Window, id: u32) {
        let old_id = window_output_id(window);
        if let Some(meta) = window_meta(window) {
            meta.output.store(id, Ordering::Relaxed);
            meta.parked.store(true, Ordering::Relaxed);
        }
        if let (Some(surface), Some(idx)) = (window.wl_surface(), self.node_idx_for_id(old_id)) {
            self.output_nodes[idx].output.leave(&surface);
        }
        self.space.map_element(window.clone(), PARKED_POS, false);
        if let Some(toplevel) = window.toplevel() {
            let toplevel = toplevel.clone();
            self.send_forced_fullscreen_configure(&toplevel);
        }
    }

    /// Place `window` on output `id`: retag its meta, remap it at the output's layout
    /// origin, move output enter/leave, push the output's fractional scale, and send the
    /// forced-fullscreen configure at that output's logical size.
    pub(crate) fn place_window_on_output(&mut self, window: &Window, id: u32) -> bool {
        let Some(idx) = self.node_idx_for_id(id) else { return false };
        let old_id = window_output_id(window);
        let (new_output, pos) = {
            let node = &self.output_nodes[idx];
            (node.output.clone(), node.pos)
        };
        let old_output = self
            .node_idx_for_id(old_id)
            .map(|i| self.output_nodes[i].output.clone());
        if let Some(meta) = window_meta(window) {
            meta.output.store(id, Ordering::Relaxed);
            meta.parked.store(false, Ordering::Relaxed);
        }
        self.space.map_element(window.clone(), pos, true);
        if let Some(surface) = window.wl_surface() {
            if let Some(old) = old_output
                && old_id != id {
                    old.leave(&surface);
                }
            new_output.enter(&surface);
            let scale = new_output.current_scale().fractional_scale();
            with_states(&surface, |states| {
                smithay::wayland::compositor::send_surface_state(
                    &surface, states, scale.ceil() as i32, smithay::utils::Transform::Normal,
                );
                smithay::wayland::fractional_scale::with_fractional_scale(states, |fs| {
                    fs.set_preferred_scale(scale);
                });
            });
        }
        if let Some(toplevel) = window.toplevel() {
            let toplevel = toplevel.clone();
            self.send_forced_fullscreen_configure(&toplevel);
        }
        true
    }

    /// Drain a clipboard read staged by `new_selection` and hand `(mime, bytes)` to the
    /// Python callback off-thread.
    ///
    /// Runs from the loop *after* the dispatch that stored the new client source, so the request
    /// targets the current selection rather than the previous one. It clones the callback, opens a
    /// pipe, and asks the owning client source to write the chosen mime into the pipe's writer. A
    /// spawned reader thread then reads the response. The overall bound is by SIZE (64 MiB, then
    /// delivered truncated) so a hostile client cannot balloon memory; time only bounds
    /// INACTIVITY — a producer that keeps bytes flowing may take as long as it needs (a large
    /// transfer from a slow source still delivers), while one that goes silent for 10 s without
    /// closing its fd is dropped so each clipboard change cannot leak a pinned thread + pipe.
    /// The `PY_SHUTDOWN` checks keep this off a shutting-down interpreter.
    pub(crate) fn process_pending_clipboard_read(&mut self) {
        let Some(mime) = self.pending_clipboard_read.take() else { return };
        if crate::PY_SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let Some(cb) = self
            .clipboard_callback
            .as_ref()
            .map(|c| Python::attach(|py| c.clone_ref(py)))
        else {
            return;
        };
        let Ok((reader, writer)) = std::io::pipe() else { return };
        if request_data_device_client_selection::<AppState>(&self.seat, mime.clone(), writer.into())
            .is_err()
        {
            return;
        }
        std::thread::spawn(move || {
            use std::io::Read;
            use std::os::fd::AsRawFd;
            const CAP: usize = 64 * 1024 * 1024;
            const IDLE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);
            let mut last_data = Instant::now();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 65536];
            loop {
                let Some(remaining) = IDLE_DEADLINE.checked_sub(last_data.elapsed()) else {
                    return;
                };
                let mut pfd = libc::pollfd {
                    fd: reader.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                let timeout_ms = remaining.as_millis().min(i32::MAX as u128).max(1) as i32;
                let ready = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
                if ready < 0 {
                    if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    return;
                }
                if ready == 0 {
                    continue;
                }
                match (&reader).read(&mut chunk) {
                    Ok(0) => break,
                    Ok(n) => {
                        last_data = Instant::now();
                        let room = CAP - buf.len();
                        let take_n = n.min(room);
                        buf.extend_from_slice(&chunk[..take_n]);
                        if buf.len() == CAP {
                            break;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted
                        || e.kind() == std::io::ErrorKind::WouldBlock => continue,
                    Err(_) => return,
                }
            }
            if buf.is_empty() {
                return;
            }
            if crate::PY_SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            Python::attach(|py| {
                let bytes = PyBytes::new(py, &buf);
                let _ = cb.call1(py, (mime.as_str(), bytes));
            });
        });
    }


    /// Resolve a `CursorImageStatus` into a job for the `wl-cursor` worker, which does the
    /// PNG encode, caching, and the GIL-bound Python call off the calloop thread. Also re-invoked
    /// from the calloop command handlers to replay the retained cursor when a callback
    /// (re)registers or a capture restarts.
    ///
    /// Only the renderer/surface-affine work happens here:
    ///
    /// 1. **Named** / **Hidden**: forwarded as-is (the worker owns its own theme handle).
    /// 2. **Surface** (a client-supplied cursor sprite): ignore any surface without the
    ///    `cursor_image` role, read the hotspot, then read the backing buffer by one of two paths:
    ///    - **SHM**: hash only the sprite's sub-region — width/height/stride/offset/format plus the
    ///      pixel span — because many sprites share one pool and differ only by `offset`, so hashing
    ///      the whole pool would collide; ship the raw pool bytes plus descriptor to the worker.
    ///    - **dmabuf** (`NotManaged`): bind it to the GLES renderer, copy the framebuffer to
    ///      `Abgr8888`, map it back (calloop-affine), and ship the raw RGBA readback.
    ///
    ///    A sprite whose buffer could not be read is dropped by the worker, preserving the
    ///    consumer's last cursor instead of blanking it (only "hide" carries empty data).
    pub(crate) fn send_cursor_image(&mut self, image: &CursorImageStatus) {
        if !self.cursor_callback_set {
            return;
        }
        let job = match image {
            CursorImageStatus::Named(icon) => {
                self.cursor_buffer = None;
                CursorJob::Named { name: cursor_icon_to_str(icon) }
            }
            CursorImageStatus::Hidden => {
                self.cursor_buffer = None;
                CursorJob::Hide
            }
            CursorImageStatus::Surface(surface) => {
                let mut hot_x = 0;
                let mut hot_y = 0;
                let mut is_cursor_role = false;

                // The hotspot is set in logical surface coordinates; the consumer gets the
                // sprite in buffer pixels, so the hotspot ships in the same units.
                with_states(surface, |states| {
                    if states.role == Some("cursor_image") {
                        is_cursor_role = true;
                    }
                    let buffer_scale = states
                        .cached_state
                        .get::<SurfaceAttributes>()
                        .current()
                        .buffer_scale
                        .max(1);
                    if let Some(attributes) = states.data_map.get::<Mutex<CursorImageAttributes>>()
                        && let Ok(guard) = attributes.lock() {
                            hot_x = guard.hotspot.x * buffer_scale;
                            hot_y = guard.hotspot.y * buffer_scale;
                        }
                });

                if !is_cursor_role {
                    return;
                }

                let buffer_found = with_states(surface, |states| {
                    let mut attrs = states.cached_state.get::<SurfaceAttributes>();

                    if let Some(BufferAssignment::NewBuffer(b)) = &attrs.current().buffer {
                        return Some(b.clone());
                    }

                    if let Some(mutex) = states.data_map.get::<Mutex<RendererSurfaceState>>()
                        && let Ok(renderer_state) = mutex.try_lock()
                            && let Some(b) = renderer_state.buffer() {
                                let wl_buffer: &wayland_server::protocol::wl_buffer::WlBuffer = b;
                                return Some(wl_buffer.clone());
                            }
                    None
                });

                let Some(buffer) = buffer_found else { return };

                let shm_result = with_buffer_contents(&buffer, |ptr, len, spec| {
                    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                    let mut hasher = DefaultHasher::new();
                    spec.width.hash(&mut hasher);
                    spec.height.hash(&mut hasher);
                    spec.stride.hash(&mut hasher);
                    spec.offset.hash(&mut hasher);
                    spec.format.hash(&mut hasher);
                    let start = (spec.offset.max(0) as usize).min(len);
                    let span = (spec.stride.max(0) as usize)
                        .saturating_mul(spec.height.max(0) as usize);
                    let end = start.saturating_add(span).min(len);
                    slice[start..end].hash(&mut hasher);
                    let hash = hasher.finish();
                    (hash, spec.width, spec.height, spec.stride, spec.format, spec.offset, slice.to_vec())
                });

                let job = match shm_result {
                    Ok((hash, width, height, stride, format, buf_offset, raw_bytes)) => {
                        Some(CursorJob::Shm {
                            hash,
                            width,
                            height,
                            stride,
                            offset: buf_offset,
                            opaque: format == wl_shm::Format::Xrgb8888,
                            bytes: raw_bytes,
                            hot_x,
                            hot_y,
                        })
                    }
                    Err(BufferAccessError::NotManaged) => {
                        let mut gles_job = None;
                        let dmabuf_opt = get_dmabuf(&buffer).ok().cloned();
                        if let Some(mut dmabuf) = dmabuf_opt
                            && let Some(renderer) = self.gles_renderer.as_mut() {
                                let width = dmabuf.width() as i32;
                                let height = dmabuf.height() as i32;

                                match renderer.bind(&mut dmabuf) {
                                    Ok(frame) => {
                                        let rect = Rectangle::new((0, 0).into(), (width, height).into());
                                        match renderer.copy_framebuffer(&frame, rect, Fourcc::Abgr8888) {
                                            Ok(mapping) => match renderer.map_texture(&mapping) {
                                                Ok(data) => {
                                                    let mut hasher = DefaultHasher::new();
                                                    data.hash(&mut hasher);
                                                    gles_job = Some(CursorJob::Gles {
                                                        hash: hasher.finish(),
                                                        width,
                                                        height,
                                                        bytes: data.to_vec(),
                                                        hot_x,
                                                        hot_y,
                                                    });
                                                }
                                                Err(e) => eprintln!("Failed to map texture: {:?}", e),
                                            },
                                            Err(e) => eprintln!("Failed to copy framebuffer: {:?}", e),
                                        }
                                    }
                                    Err(e) => eprintln!("Failed to bind dmabuf to renderer: {:?}", e),
                                }
                            }
                        gles_job
                    }
                    Err(_) => None,
                };

                self.cursor_buffer = Some(buffer);
                let Some(job) = job else { return };
                job
            }
        };
        let _ = self.cursor_tx.send(job);
    }

    /// Deliver a surface-backed cursor whose dispatch ended without a commit of its surface
    /// (a client re-showing a sprite it attached earlier).
    pub(crate) fn flush_pending_cursor(&mut self) {
        if !self.cursor_surface_pending {
            return;
        }
        self.cursor_surface_pending = false;
        if let Some(icon) = self.current_cursor_icon.clone() {
            self.send_cursor_image(&icon);
        }
    }

    /// Re-apply the policy's keymap (base + overlays) to the seat keyboard, broadcasting to
    /// clients only when the content actually changed (smithay dedupes by content hash).
    pub(crate) fn apply_keymap_policy(&mut self) {
        let text = self.keymap_policy.keymap_text();
        self.apply_keymap_text(text);
    }

    /// Deliver `text` as the seat keymap (and the host virtual keyboard's, in
    /// host-capture mode). Empty text is ignored.
    pub(crate) fn apply_keymap_text(&mut self, text: String) {
        if text.is_empty() {
            return;
        }
        // Host-capture mode: the same managed keymap rides on the virtual
        // keyboard, so the host compositor translates injected keycodes with
        // selkies' keymap (overlay binds included) instead of its own.
        if let Some(host) = &self.host {
            host.set_keymap(&text);
        }
        if let Some(keyboard) = self.seat.get_keyboard()
            && let Err(e) = keyboard.set_keymap_from_string(self, text) {
                eprintln!("[Wayland] keymap swap failed: {e:?}");
            }
    }

    /// Resolve `keysyms` to `(keycode, level)` pairs, overlay-binding whatever the base
    /// cannot produce — at most ONE keymap swap for the whole batch, and never rebinding a
    /// keycode that is currently held down. Serves computer-use; the interactive input path
    /// resolves keysyms in selkies and injects plain keycodes.
    pub(crate) fn bind_keysyms(&mut self, keysyms: &[u32]) -> Vec<(u32, u32)> {
        let pressed: std::collections::HashSet<u32> = self
            .seat
            .get_keyboard()
            .map(|k| k.pressed_keys().iter().map(|c| c.raw()).collect())
            .unwrap_or_default();
        let (out, changed) = self.keymap_policy.bind_many(keysyms, &pressed);
        if changed {
            self.apply_keymap_policy();
        }
        out
    }

    /// `bind_keysyms` restricted to level-0 resolutions (see
    /// [`KeymapPolicy::bind_many_plain`]); used by the virtual-keyboard translation path, which
    /// cannot synthesize modifiers.
    pub(crate) fn bind_keysyms_plain(&mut self, keysyms: &[u32]) -> Vec<u32> {
        let pressed: std::collections::HashSet<u32> = self
            .seat
            .get_keyboard()
            .map(|k| k.pressed_keys().iter().map(|c| c.raw()).collect())
            .unwrap_or_default();
        let (out, changed) = self.keymap_policy.bind_many_plain(keysyms, &pressed);
        if changed {
            self.apply_keymap_policy();
        }
        out
    }

    /// Answer any client fullscreen/maximize (un)set request with the compositor's forced
    /// policy: every toplevel is Fullscreen+Activated at the CURRENT logical size of the
    /// output the window is placed on. The configure is always sent, so an app toggling
    /// fullscreen gets an explicit, current-geometry answer instead of silence or stale
    /// pending state.
    pub(crate) fn send_forced_fullscreen_configure(&mut self, toplevel: &ToplevelSurface) {
        let output_id = self
            .space
            .elements()
            .chain(self.pending_windows.iter())
            .find(|w| w.toplevel().map(|t| t == toplevel).unwrap_or(false))
            .map(window_output_id)
            .unwrap_or(0);
        let parked = self
            .space
            .elements()
            .chain(self.pending_windows.iter())
            .find(|w| w.toplevel().map(|t| t == toplevel).unwrap_or(false))
            .and_then(window_meta)
            .map(|meta| meta.parked.load(Ordering::Relaxed))
            .unwrap_or(false);
        let (logical_width, logical_height) = if parked {
            PARKED_LOGICAL_SIZE
        } else if let Some(size) = self.logical_size_of(output_id) {
            size
        } else {
            let scale = self.settings.scale.max(0.1);
            (
                (self.settings.width as f64 / scale).round() as i32,
                (self.settings.height as f64 / scale).round() as i32,
            )
        };
        toplevel.with_pending_state(|state| {
            state.states.set(XdgState::Fullscreen);
            state.states.set(XdgState::Activated);
            state.size = Some((logical_width, logical_height).into());
        });
        toplevel.send_configure();
    }
}

/// Clipboard mime types the bridge can hand to Python, most specific first; `new_selection`
/// picks the first of these that the client's source offers.
const CLIPBOARD_MIME_PREFERENCE: &[&str] = &[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp",
    "image/svg+xml",
    "image/svg",
    "text/plain;charset=utf-8",
    "UTF8_STRING",
    "text/plain",
    "STRING",
    "TEXT",
];

/// Selection (clipboard) bridge between Wayland clients and Python. `SelectionUserData` is
/// the Python-owned payload `(mime, bytes)` served to pasting clients when Python holds the
/// selection.
impl SelectionHandler for AppState {
    type SelectionUserData = std::sync::Arc<(String, Vec<u8>)>;

    /// A client took the clipboard: pick the best offered mime and stage it for the loop to
    /// read.
    ///
    /// Only client-owned clipboard (not primary) selections are relayed to Python. Among the
    /// source's offered mimes it chooses the most specific match from `CLIPBOARD_MIME_PREFERENCE`
    /// and records it in `pending_clipboard_read`. The read itself is deferred: the new source is
    /// stored only after this handler returns, so `process_pending_clipboard_read` runs
    /// post-dispatch and reads the new selection rather than the previous one.
    fn new_selection(
        &mut self,
        ty: SelectionTarget,
        source: Option<SelectionSource>,
        seat: Seat<Self>,
    ) {
        if ty != SelectionTarget::Clipboard {
            return;
        }
        let Some(source) = source else {
            self.current_selection_mime = None;
            return;
        };
        let mimes = source.mime_types();
        let mime = CLIPBOARD_MIME_PREFERENCE
            .iter()
            .find(|want| mimes.iter().any(|m| m == *want))
            .map(|s| s.to_string());
        // Recorded even with no callback armed, so SetClipboardCallback can re-stage a
        // read of a copy made while nobody was listening.
        self.current_selection_mime = mime.clone();
        if crate::PY_SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        if self.clipboard_callback.is_none() {
            return;
        }
        let Some(mime) = mime else { return };
        self.pending_clipboard_read = Some(mime);
        let _ = seat;
    }

    /// A client pastes the Python-owned selection: stream the stored bytes into the client's
    /// fd on a spawned thread, since the receiving pipe may backpressure. Bounded by an
    /// idle deadline: a client that never reads its paste fd must not pin the thread.
    fn send_selection(
        &mut self,
        ty: SelectionTarget,
        _mime_type: String,
        fd: std::os::fd::OwnedFd,
        _seat: Seat<Self>,
        user_data: &Self::SelectionUserData,
    ) {
        if ty != SelectionTarget::Clipboard && ty != SelectionTarget::Primary {
            return;
        }
        let payload = user_data.clone();
        std::thread::spawn(move || {
            let _ = crate::wayland::wlclient::write_fd_all(
                &fd,
                &payload.1,
                crate::wayland::wlclient::IO_TIMEOUT,
            );
        });
    }
}

/// `wl_data_device` protocol: exposes the managed state for drag-and-drop and clipboard
/// data transfers.
impl DataDeviceHandler for AppState {
    fn data_device_state(&mut self) -> &mut DataDeviceState {
        &mut self.data_device_state
    }
}
/// wlr-data-control protocol: exposes the managed state so privileged clients can read and
/// set selections.
impl DataControlHandler for AppState {
    fn data_control_state(&mut self) -> &mut DataControlState {
        &mut self.data_control_state
    }
}
/// No graphics tablets exist here (input is injected); the default no-op tool-image
/// callback is all cursor-shape's tablet half needs.
impl smithay::wayland::tablet_manager::TabletSeatHandler for AppState {}

/// ext-data-control: the standardized successor of wlr-data-control. Both globals
/// stay advertised, sharing the same selection state, so older clipboard managers
/// keep working while current ones use the ext protocol.
impl ExtDataControlHandler for AppState {
    fn data_control_state(&mut self) -> &mut ExtDataControlState {
        &mut self.ext_data_control_state
    }
}

/// Marker impl enabling Wayland drag-and-drop grabs with Smithay's default behavior.
impl WaylandDndGrabHandler for AppState {}
/// Buffer lifecycle hook: buffer destruction needs no bookkeeping here.
impl BufferHandler for AppState {
    fn buffer_destroyed(&mut self, _buffer: &WlBuffer) {}
}
/// `wl_shm` protocol: exposes the shared-memory buffer state.
impl ShmHandler for AppState {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}
/// Output protocol marker impl (no per-output callbacks are needed).
impl OutputHandler for AppState {}

/// Linux-dmabuf protocol: imports client dmabufs into the GLES renderer.
///
/// `dmabuf_imported` attempts the import into the GLES renderer and signals the client through the
/// notifier — success only when a GLES renderer exists and the import succeeds, otherwise failure
/// (the software / pixman path advertises no dmabuf global, so it always fails here).
impl DmabufHandler for AppState {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.dmabuf_state
    }

    fn dmabuf_imported(
        &mut self,
        _global: &DmabufGlobal,
        dmabuf: Dmabuf,
        notifier: ImportNotifier,
    ) {
        if let Some(renderer) = self.gles_renderer.as_mut() {
            if renderer.import_dmabuf(&dmabuf, None).is_ok() {
                let _ = notifier.successful::<AppState>();
            } else {
                notifier.failed();
            }
        } else {
            notifier.failed();
        }
    }
}

impl ImageCaptureSourceHandler for AppState {
    fn source_destroyed(&mut self, _source: ImageCaptureSource) {}
}

impl OutputCaptureSourceHandler for AppState {
    fn output_capture_source_state(&mut self) -> &mut OutputCaptureSourceState {
        &mut self.output_capture_source_state
    }

    fn output_source_created(&mut self, source: ImageCaptureSource, output: &Output) {
        source.user_data().insert_if_missing(|| output.downgrade());
    }
}

/// ext-image-copy-capture: external clients capturing this compositor's outputs.
///
/// A dmabuf-capable client gets the render formats of the GLES renderer and its frames
/// are filled with one GPU blit from the output's composited buffer; an shm client gets
/// a readback. Frames complete from the render loop, never from a stale backbuffer.
impl ImageCopyCaptureHandler for AppState {
    fn image_copy_capture_state(&mut self) -> &mut ImageCopyCaptureState {
        &mut self.image_copy_capture_state
    }

    fn capture_constraints(&mut self, source: &ImageCaptureSource) -> Option<BufferConstraints> {
        let output = source.user_data().get::<WeakOutput>()?.upgrade()?;
        output_capture_constraints(&output, self.gles_renderer.as_ref(), &self.render_node_path)
    }

    fn new_session(&mut self, session: CopySession) {
        let Some(output) = session.source().user_data().get::<WeakOutput>().cloned() else {
            session.stop();
            return;
        };
        match self.capture_constraints(&session.source()) {
            Some(constraints) => {
                session.update_constraints(constraints);
                self.copy_sessions.push(OutputCopySession {
                    session,
                    output,
                    pending: None,
                    delivered_once: false,
                });
            }
            None => session.stop(),
        }
    }

    fn frame(&mut self, session: &CopySessionRef, frame: CopyFrame) {
        match self.copy_sessions.iter_mut().find(|cs| cs.session == *session) {
            Some(cs) => cs.pending = Some(frame),
            None => frame.fail(CaptureFailureReason::Unknown),
        }
    }

    fn frame_aborted(&mut self, frame: CopyFrameRef) {
        for cs in self.copy_sessions.iter_mut() {
            if cs.pending.as_ref().is_some_and(|f| *f == frame) {
                cs.pending = None;
            }
        }
    }

    fn session_destroyed(&mut self, session: CopySessionRef) {
        self.copy_sessions.retain(|cs| cs.session != session);
    }
}

impl AppState {
    pub fn copy_frame_pending_for(&self, output: &Output) -> bool {
        self.copy_sessions
            .iter()
            .any(|cs| cs.pending.is_some() && cs.output.upgrade().as_ref() == Some(output))
    }
}

/// Buffer constraints for capturing `output`: its current mode's size, the shm formats
/// both render paths can serve, and — with a GLES renderer — the renderer's dmabuf
/// render formats so a client buffer can be blitted to on the GPU.
pub fn output_capture_constraints(
    output: &Output,
    renderer: Option<&GlesRenderer>,
    render_node_path: &str,
) -> Option<BufferConstraints> {
    let mode = output.current_mode()?;
    let dma = renderer.and_then(|renderer| {
        let node = DrmNode::from_path(render_node_path).ok()?;
        let mut formats: Vec<(Fourcc, Vec<Modifier>)> = Vec::new();
        for f in Bind::<Dmabuf>::supported_formats(renderer)? {
            match formats.iter_mut().find(|(code, _)| *code == f.code) {
                Some((_, mods)) => mods.push(f.modifier),
                None => formats.push((f.code, vec![f.modifier])),
            }
        }
        (!formats.is_empty()).then_some(DmabufConstraints { node, formats })
    });
    Some(BufferConstraints {
        size: (mode.size.w, mode.size.h).into(),
        shm: vec![wl_shm::Format::Argb8888, wl_shm::Format::Xrgb8888],
        dma,
    })
}

/// Fractional-scale protocol: tells a newly-bound surface the output's current fractional
/// scale so it renders at the right pixel density.
impl FractionalScaleHandler for AppState {
    fn new_fractional_scale(
        &mut self,
        surface: smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
    ) {
        if let Some(output) = self.primary_output() {
            let scale = output.current_scale().fractional_scale();
            with_states(&surface, |states| {
                smithay::wayland::compositor::send_surface_state(
                    &surface, states, scale.ceil() as i32, smithay::utils::Transform::Normal,
                );
                smithay::wayland::fractional_scale::with_fractional_scale(states, |fs| {
                    fs.set_preferred_scale(scale);
                });
            });
        }
    }
}

/// Input-event target for Smithay's seat handlers. Smithay requires a concrete type as the
/// "target" of a keyboard / pointer / touch event; `FocusTarget` bridges that to the concrete
/// Wayland surface behind a window, popup, or layer surface.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum FocusTarget {
    Window(Window),
    Popup(PopupKind),
    LayerSurface(DesktopLayerSurface),
}

/// Wrap a `Window` as a focus target.
impl From<Window> for FocusTarget {
    fn from(w: Window) -> Self { FocusTarget::Window(w) }
}

/// Wrap a popup as a focus target.
impl From<PopupKind> for FocusTarget {
    fn from(p: PopupKind) -> Self { FocusTarget::Popup(p) }
}

/// Wrap a layer surface as a focus target.
impl From<DesktopLayerSurface> for FocusTarget {
    fn from(l: DesktopLayerSurface) -> Self { FocusTarget::LayerSurface(l) }
}

/// Liveness of a focus target: true while its underlying window / popup / layer surface is
/// still alive, so dead targets are dropped from focus.
impl IsAlive for FocusTarget {
    fn alive(&self) -> bool {
        match self {
            FocusTarget::Window(w) => w.alive(),
            FocusTarget::Popup(p) => p.alive(),
            FocusTarget::LayerSurface(l) => l.alive(),
        }
    }
}

/// Expose the wrapped target's underlying `wl_surface` and same-client checks so Smithay
/// can route focus and selection ownership by client.
impl WaylandFocus for FocusTarget {
    fn wl_surface(&self) -> Option<Cow<'_, WlSurface>> {
        match self {
            FocusTarget::Window(w) => w.wl_surface(),
            FocusTarget::Popup(p) => Some(Cow::Borrowed(p.wl_surface())),
            FocusTarget::LayerSurface(l) => Some(Cow::Borrowed(l.wl_surface())),
        }
    }
    fn same_client_as(&self, object_id: &ObjectId) -> bool {
        match self {
            FocusTarget::Window(w) => w.same_client_as(object_id),
            FocusTarget::Popup(p) => p.wl_surface().id().same_client_as(object_id),
            FocusTarget::LayerSurface(l) => l.wl_surface().id().same_client_as(object_id),
        }
    }
}

/// Forward every keyboard event (enter / leave / key / modifiers) to the wrapped target's
/// underlying `wl_surface`, which carries Smithay's real keyboard-target implementation.
impl KeyboardTarget<AppState> for FocusTarget {
    fn enter(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        keys: Vec<KeysymHandle<'_>>,
        serial: Serial,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::keyboard::KeyboardTarget::enter(
                surface.as_ref(),
                seat,
                data,
                keys,
                serial,
            );
        }
    }
    fn leave(&self, seat: &Seat<AppState>, data: &mut AppState, serial: Serial) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::keyboard::KeyboardTarget::leave(surface.as_ref(), seat, data, serial);
        }
    }
    fn key(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        key: KeysymHandle<'_>,
        state: smithay::backend::input::KeyState,
        serial: Serial,
        time: u32,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::keyboard::KeyboardTarget::key(
                surface.as_ref(),
                seat,
                data,
                key,
                state,
                serial,
                time,
            );
        }
    }
    fn modifiers(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        modifiers: ModifiersState,
        serial: Serial,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::keyboard::KeyboardTarget::modifiers(
                surface.as_ref(),
                seat,
                data,
                modifiers,
                serial,
            );
        }
    }
}

/// Forward drag-and-drop focus events (enter / motion / leave / drop) to the wrapped
/// target's underlying `wl_surface`, reusing the `WlSurface` offer-data type.
impl DndFocus<AppState> for FocusTarget {
    type OfferData<S: Source> = <WlSurface as DndFocus<AppState>>::OfferData<S>;

    fn enter<S: Source>(
        &self,
        data: &mut AppState,
        dh: &DisplayHandle,
        source: Arc<S>,
        seat: &Seat<AppState>,
        location: Point<f64, Logical>,
        serial: &Serial,
    ) -> Option<Self::OfferData<S>> {
        if let Some(surface) = self.wl_surface() {
            <WlSurface as DndFocus<AppState>>::enter(
                surface.as_ref(),
                data,
                dh,
                source,
                seat,
                location,
                serial,
            )
        } else {
            None
        }
    }

    fn motion<S: Source>(
        &self,
        data: &mut AppState,
        offer: Option<&mut Self::OfferData<S>>,
        seat: &Seat<AppState>,
        location: Point<f64, Logical>,
        time: u32,
    ) {
        if let Some(surface) = self.wl_surface() {
            <WlSurface as DndFocus<AppState>>::motion(
                surface.as_ref(),
                data,
                offer,
                seat,
                location,
                time,
            )
        }
    }

    fn leave<S: Source>(
        &self,
        data: &mut AppState,
        offer: Option<&mut Self::OfferData<S>>,
        seat: &Seat<AppState>,
    ) {
        if let Some(surface) = self.wl_surface() {
            <WlSurface as DndFocus<AppState>>::leave(surface.as_ref(), data, offer, seat)
        }
    }

    fn drop<S: Source>(
        &self,
        data: &mut AppState,
        offer: Option<&mut Self::OfferData<S>>,
        seat: &Seat<AppState>,
    ) {
        if let Some(surface) = self.wl_surface() {
            <WlSurface as DndFocus<AppState>>::drop(surface.as_ref(), data, offer, seat)
        }
    }
}

/// Forward every pointer event (motion, buttons, axis, and all swipe / pinch / hold gesture
/// phases) to the wrapped target's underlying `wl_surface`.
impl PointerTarget<AppState> for FocusTarget {
    fn enter(&self, seat: &Seat<AppState>, data: &mut AppState, event: &MotionEvent) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::enter(surface.as_ref(), seat, data, event);
        }
    }
    fn motion(&self, seat: &Seat<AppState>, data: &mut AppState, event: &MotionEvent) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::motion(surface.as_ref(), seat, data, event);
        }
    }
    fn relative_motion(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &RelativeMotionEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::relative_motion(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn button(&self, seat: &Seat<AppState>, data: &mut AppState, event: &ButtonEvent) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::button(surface.as_ref(), seat, data, event);
        }
    }
    fn axis(&self, seat: &Seat<AppState>, data: &mut AppState, frame: AxisFrame) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::axis(surface.as_ref(), seat, data, frame);
        }
    }
    fn frame(&self, seat: &Seat<AppState>, data: &mut AppState) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::frame(surface.as_ref(), seat, data);
        }
    }
    fn leave(&self, seat: &Seat<AppState>, data: &mut AppState, serial: Serial, time: u32) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::leave(
                surface.as_ref(),
                seat,
                data,
                serial,
                time,
            );
        }
    }
    fn gesture_swipe_begin(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GestureSwipeBeginEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_swipe_begin(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_swipe_update(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GestureSwipeUpdateEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_swipe_update(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_swipe_end(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GestureSwipeEndEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_swipe_end(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_pinch_begin(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GesturePinchBeginEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_pinch_begin(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_pinch_update(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GesturePinchUpdateEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_pinch_update(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_pinch_end(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GesturePinchEndEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_pinch_end(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_hold_begin(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GestureHoldBeginEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_hold_begin(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_hold_end(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GestureHoldEndEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_hold_end(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
}

/// Forward every touch event (down / up / motion / frame / cancel / shape / orientation) to
/// the wrapped target's underlying `wl_surface`.
impl TouchTarget<AppState> for FocusTarget {
    fn down(&self, seat: &Seat<AppState>, data: &mut AppState, event: &DownEvent, serial: Serial) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::down(surface.as_ref(), seat, data, event, serial);
        }
    }
    fn up(&self, seat: &Seat<AppState>, data: &mut AppState, event: &UpEvent, serial: Serial) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::up(surface.as_ref(), seat, data, event, serial);
        }
    }
    fn motion(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &smithay::input::touch::MotionEvent,
        serial: Serial,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::motion(
                surface.as_ref(),
                seat,
                data,
                event,
                serial,
            );
        }
    }
    fn frame(&self, seat: &Seat<AppState>, data: &mut AppState, serial: Serial) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::frame(surface.as_ref(), seat, data, serial);
        }
    }
    fn cancel(&self, seat: &Seat<AppState>, data: &mut AppState, serial: Serial) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::cancel(surface.as_ref(), seat, data, serial);
        }
    }
    fn shape(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &ShapeEvent,
        serial: Serial,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::shape(surface.as_ref(), seat, data, event, serial);
        }
    }
    fn orientation(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &OrientationEvent,
        serial: Serial,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::orientation(
                surface.as_ref(),
                seat,
                data,
                event,
                serial,
            );
        }
    }
}

/// Map a Smithay `CursorIcon` to its CSS cursor-name string, used both for themed-cursor
/// lookup and for the name handed to the Python cursor callback; unknown icons fall back to
/// `"default"`.
pub fn cursor_icon_to_str(icon: &CursorIcon) -> &'static str {
    match icon {
        CursorIcon::Default => "default",
        CursorIcon::ContextMenu => "context-menu",
        CursorIcon::Help => "help",
        CursorIcon::Pointer => "pointer",
        CursorIcon::Progress => "progress",
        CursorIcon::Wait => "wait",
        CursorIcon::Cell => "cell",
        CursorIcon::Crosshair => "crosshair",
        CursorIcon::Text => "text",
        CursorIcon::VerticalText => "vertical-text",
        CursorIcon::Alias => "alias",
        CursorIcon::Copy => "copy",
        CursorIcon::Move => "move",
        CursorIcon::NoDrop => "no-drop",
        CursorIcon::NotAllowed => "not-allowed",
        CursorIcon::Grab => "grab",
        CursorIcon::Grabbing => "grabbing",
        CursorIcon::AllScroll => "all-scroll",
        CursorIcon::ColResize => "col-resize",
        CursorIcon::RowResize => "row-resize",
        CursorIcon::NResize => "n-resize",
        CursorIcon::EResize => "e-resize",
        CursorIcon::SResize => "s-resize",
        CursorIcon::WResize => "w-resize",
        CursorIcon::NeResize => "ne-resize",
        CursorIcon::NwResize => "nw-resize",
        CursorIcon::SeResize => "se-resize",
        CursorIcon::SwResize => "sw-resize",
        CursorIcon::EwResize => "ew-resize",
        CursorIcon::NsResize => "ns-resize",
        CursorIcon::NeswResize => "nesw-resize",
        CursorIcon::NwseResize => "nwse-resize",
        CursorIcon::ZoomIn => "zoom-in",
        CursorIcon::ZoomOut => "zoom-out",
        _ => "default",
    }
}

/// Seat wiring: declares the keyboard / pointer / touch focus types and reacts to cursor
/// changes and keyboard-focus moves.
impl SeatHandler for AppState {
    type KeyboardFocus = FocusTarget;
    type PointerFocus = FocusTarget;
    type TouchFocus = FocusTarget;
    fn seat_state(&mut self) -> &mut SeatState<AppState> {
        &mut self.seat_state
    }

    /// A client requested a cursor change (named, hidden, or surface-backed): retain it as
    /// the current icon and forward it to the Python cursor callback. A surface-backed one
    /// is delivered by the surface's commit or by `flush_pending_cursor`, whichever is first.
    fn cursor_image(&mut self, _seat: &Seat<AppState>, image: CursorImageStatus) {
        self.current_cursor_icon = Some(image.clone());
        self.cursor_surface_pending = matches!(image, CursorImageStatus::Surface(_));
        if !self.cursor_surface_pending {
            self.send_cursor_image(&image);
        }
    }

    /// Keep BOTH selections' focus following keyboard focus: without the data-device half,
    /// the focused client never receives wl_data_offer events and Ctrl+V paste is a silent
    /// no-op even while the compositor-side selection is correct (primary covers only
    /// middle-click paste).
    fn focus_changed(&mut self, seat: &Seat<AppState>, focus: Option<&Self::KeyboardFocus>) {
        let dh = &self.dh;
        let client = focus
            .and_then(|t| t.wl_surface())
            .and_then(|s| dh.get_client(s.id()).ok());
        set_data_device_focus(dh, seat, client.clone());
        set_primary_focus(dh, seat, client);
    }
}

/// Pointer-warp protocol: lets a client teleport the pointer to a surface-local position
/// (games / remote-desktop style warps).
///
/// `warp_pointer` locates the requesting surface's origin in the global space, adds the requested
/// surface-local offset to get a global position, recomputes what element lies under it, and emits
/// a synthetic motion event so focus and enter/leave follow the warp.
impl PointerWarpHandler for AppState {
    fn warp_pointer(
        &mut self,
        surface: WlSurface,
        _pointer: WlPointer,
        pos: Point<f64, Logical>,
        serial: Serial,
    ) {
        let surface_origin = self.space.elements().find_map(|window| {
            if window.wl_surface().as_deref() == Some(&surface) {
                self.space.element_location(window)
            } else {
                None
            }
        });

        if let Some(origin) = surface_origin {
            let global_pos = origin.to_f64() + pos;
            let time = wayland_time();

            if let Some(pointer) = self.seat.get_pointer() {
                let under = self.space.element_under(global_pos).map(|(w, loc)| {
                    (FocusTarget::Window(w.clone()), loc.to_f64())
                });
                
                pointer.motion(
                    self,
                    under,
                    &MotionEvent {
                        location: global_pos,
                        serial, 
                        time,
                    },
                );
            }
        }
    }
}

/// xdg-shell protocol: toplevel and popup lifecycle.
impl XdgShellHandler for AppState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.shell_state
    }
    /// A new toplevel appears: wrap it in a `Window`, queue it for mapping, and register a
    /// foreign-toplevel handle (seeded with title / app-id) stored on the surface for later updates.
    /// The window is pinned to the pointer's output HERE — the first commit can't be relied on
    /// for that, because a decoration-negotiating client (foot) has its initial configure sent
    /// by `new_decoration` before it ever commits, skipping the pre-configure commit branch.
    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        let target_id = self.pointer_display();
        let window = Window::new_wayland_window(surface.clone());
        window.user_data().insert_if_missing_threadsafe(|| WindowMeta {
            id: NEXT_WINDOW_ID.fetch_add(1, Ordering::Relaxed),
            output: AtomicU32::new(target_id),
            placed: AtomicBool::new(false),
            parked: AtomicBool::new(false),
        });
        self.pending_windows.push(window);
        let (title, app_id) = with_states(surface.wl_surface(), |states| {
            let attributes = states.data_map.get::<XdgToplevelSurfaceData>().unwrap().lock().unwrap();
            (attributes.title.clone(), attributes.app_id.clone())
        });

        let handle = self.foreign_toplevel_list.new_toplevel::<AppState>(title.unwrap_or_default(), app_id.unwrap_or_default());
        
        with_states(surface.wl_surface(), |states| states.data_map.insert_if_missing(|| handle));
    }
    /// Register a new popup (menu, tooltip, combo-box list) with the `PopupManager` so it
    /// takes part in grab and dismissal handling, then send the initial configure xdg-shell requires
    /// before the client is allowed to draw it.
    fn new_popup(&mut self, surface: PopupSurface, _positioner: PositionerState) {
        if let Err(err) = self.popups.track_popup(PopupKind::Xdg(surface.clone())) {
            eprintln!("Failed to track popup: {:?}", err);
        }
        let _ = surface.send_configure();
    }
    /// Popup grab: find the popup's root surface and its window, then install a popup grab so
    /// dismissal and pointer routing behave correctly.
    fn grab(
        &mut self,
        surface: PopupSurface,
        _seat: smithay::reexports::wayland_server::protocol::wl_seat::WlSeat,
        serial: Serial,
    ) {
        let kind = PopupKind::Xdg(surface);
        if let Ok(root_surface) = smithay::desktop::find_popup_root_surface(&kind)
            && let Some(window) = self.space.elements().find(|w| w.wl_surface().as_deref() == Some(&root_surface)).cloned() {
                let _ = self.popups.grab_popup(FocusTarget::Window(window), kind, &self.seat, serial);
            }
    }
    /// Re-track a popup whose position changed (e.g. a submenu flipping sides to stay
    /// on-screen) so the `PopupManager` follows its new geometry, then echo the client's reposition
    /// token back to confirm the move took effect.
    fn reposition_request(
        &mut self,
        surface: PopupSurface,
        _positioner: PositionerState,
        token: u32,
    ) {
        if let Err(err) = self.popups.track_popup(PopupKind::Xdg(surface.clone())) {
            eprintln!("Failed to track popup: {:?}", err);
        }
        let _ = surface.send_repositioned(token);
    }
    /// Client fullscreen request: always granted at the compositor's forced-fullscreen
    /// geometry (the CURRENT logical size), so the toggle gets a definite answer. A request
    /// naming an output moves the window there, which is how a nested compositor puts each
    /// of its screens on a monitor of its choosing.
    fn fullscreen_request(
        &mut self,
        surface: ToplevelSurface,
        output: Option<smithay::reexports::wayland_server::protocol::wl_output::WlOutput>,
    ) {
        if let Some(id) = output
            .as_ref()
            .and_then(|wlo| self.output_nodes.iter().find(|n| n.output.owns(wlo)).map(|n| n.id))
        {
            let mapped = self
                .space
                .elements()
                .find(|w| w.toplevel().map(|tl| *tl == surface).unwrap_or(false))
                .cloned();
            if let Some(window) = mapped {
                if self.place_window_on_output(&window, id) {
                    return;
                }
            } else if let Some(meta) = self
                .pending_windows
                .iter()
                .find(|w| w.toplevel().map(|tl| *tl == surface).unwrap_or(false))
                .and_then(window_meta)
            {
                // Still waiting for its first buffer: record the output instead of mapping
                // it, so nothing renders or hit-tests before the client has drawn. The
                // configure below then carries that output's geometry.
                meta.output.store(id, Ordering::Relaxed);
                meta.placed.store(true, Ordering::Relaxed);
            }
        }
        self.send_forced_fullscreen_configure(&surface);
    }
    /// Client unfullscreen request: the forced-fullscreen policy stands, but the client
    /// still receives an explicit configure at the current geometry (the Smithay default sends
    /// NOTHING here, leaving the app waiting on a toggle that never answers).
    fn unfullscreen_request(&mut self, surface: ToplevelSurface) {
        self.send_forced_fullscreen_configure(&surface);
    }
    /// Maximize request: answered with the forced-fullscreen configure (same geometry).
    fn maximize_request(&mut self, surface: ToplevelSurface) {
        self.send_forced_fullscreen_configure(&surface);
    }
    /// Unmaximize request: explicit current-geometry configure, policy unchanged.
    fn unmaximize_request(&mut self, surface: ToplevelSurface) {
        self.send_forced_fullscreen_configure(&surface);
    }
    /// A toplevel closed: drop it from the pending-window queue so a window that never
    /// finished mapping can't linger there, unmap it from the space at once so `list_windows`
    /// and hit-testing never see a husk (the periodic `space.refresh` would only reap it
    /// later), and remove its foreign-toplevel handle so taskbar-style clients stop listing a
    /// window that is gone.
    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        if let Some(idx) = self.pending_windows.iter().position(|w| w.toplevel().map(|t| *t == surface).unwrap_or(false)) {
            self.pending_windows.remove(idx);
        }
        let mapped = self
            .space
            .elements()
            .find(|w| w.toplevel().map(|t| *t == surface).unwrap_or(false))
            .cloned();
        if let Some(window) = mapped {
            self.space.unmap_elem(&window);
        }
        if let Some(handle) = with_states(surface.wl_surface(), |states| states.data_map.get::<ForeignToplevelHandle>().cloned()) {
             self.foreign_toplevel_list.remove_toplevel(&handle);
        }
    }
}

/// In-house `zwp_virtual_keyboard_v1` implementation. Smithay's manager swaps the
/// client-visible seat keymap to the virtual keyboard's keymap on every VK event and never
/// restores it, leaving every client holding a foreign keymap (and killing the compositor's
/// overlay keycodes) after any VK use. Here VK key events are TRANSLATED instead: each keycode
/// resolves to its level-0 keysym under the VK client's own uploaded keymap, maps onto the seat
/// keymap (overlay-binding on demand, batched at keymap upload), and injects through the seat's
/// regular input path — the seat keymap identity never changes and modifier/pressed-key state
/// stays coherent with server-side injection. VK `modifiers` requests are ignored: applying a
/// foreign modifier mask would corrupt the seat's own tracked state, and the supported VK
/// client (selkies' wayland_typer) binds every keysym at level 0 and never sends them.
pub struct PfVirtualKeyboard {
    inner: Mutex<PfVkState>,
}

#[derive(Default)]
struct PfVkState {
    /// Level-0 keysym per xkb keycode of the client's uploaded keymap.
    syms: Option<std::collections::HashMap<u32, u32>>,
    /// VK xkb keycode -> injected seat keycode, so a release always matches its press even
    /// across policy rebinds.
    pressed: std::collections::HashMap<u32, u32>,
}

impl GlobalDispatch<ZwpVirtualKeyboardManagerV1, ()> for AppState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwpVirtualKeyboardManagerV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<ZwpVirtualKeyboardManagerV1, ()> for AppState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &ZwpVirtualKeyboardManagerV1,
        request: zwp_virtual_keyboard_manager_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        if let zwp_virtual_keyboard_manager_v1::Request::CreateVirtualKeyboard { seat: _, id } =
            request
        {
            data_init.init(id, PfVirtualKeyboard { inner: Mutex::new(PfVkState::default()) });
        }
    }
}

impl Dispatch<ZwpVirtualKeyboardV1, PfVirtualKeyboard> for AppState {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ZwpVirtualKeyboardV1,
        request: zwp_virtual_keyboard_v1::Request,
        data: &PfVirtualKeyboard,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        use smithay::input::keyboard::xkb;
        match request {
            zwp_virtual_keyboard_v1::Request::Keymap { format, fd, size } => {
                if format != 1 {
                    return;
                }
                let ctx = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
                let keymap = unsafe {
                    xkb::Keymap::new_from_fd(
                        &ctx,
                        fd,
                        size as usize,
                        xkb::KEYMAP_FORMAT_TEXT_V1,
                        xkb::KEYMAP_COMPILE_NO_FLAGS,
                    )
                };
                let Ok(Some(keymap)) = keymap else {
                    eprintln!("[Wayland] virtual-keyboard keymap failed to compile; ignoring.");
                    return;
                };
                let syms = crate::wayland::keymap::level0_syms(&keymap);
                // Pre-bind everything this keymap can type that the seat cannot, in ONE
                // seat keymap swap, so the following key events bind nothing.
                let missing: Vec<u32> = syms
                    .values()
                    .copied()
                    .filter(|&s| !state.keymap_policy.resolves_plain(s))
                    .collect();
                if !missing.is_empty() {
                    let _ = state.bind_keysyms_plain(&missing);
                }
                data.inner.lock().unwrap().syms = Some(syms);
            }
            zwp_virtual_keyboard_v1::Request::Key { time: _, key, state: key_state } => {
                let mut vk = data.inner.lock().unwrap();
                if vk.syms.is_none() {
                    drop(vk);
                    resource.post_error(
                        zwp_virtual_keyboard_v1::Error::NoKeymap,
                        "`key` sent before keymap.",
                    );
                    return;
                }
                let vk_kc = key.wrapping_add(8);
                let seat_kc = if key_state == 1 {
                    let sym = vk.syms.as_ref().and_then(|m| m.get(&vk_kc)).copied();
                    // Translate through the seat keymap; an untranslatable keycode
                    // passes through raw (base sections of both keymaps agree for
                    // ordinary pc keycodes).
                    let kc = match sym {
                        Some(sym) => {
                            let bound = state.bind_keysyms_plain(&[sym])[0];
                            if bound != 0 { bound } else { vk_kc }
                        }
                        None => vk_kc,
                    };
                    vk.pressed.insert(vk_kc, kc);
                    kc
                } else {
                    vk.pressed.remove(&vk_kc).unwrap_or(vk_kc)
                };
                drop(vk);
                let pressed = key_state == 1;
                if let Some(keyboard) = state.seat.get_keyboard() {
                    let keyboard = keyboard.clone();
                    let serial = next_serial();
                    let time = wayland_time();
                    keyboard.input(
                        state,
                        smithay::backend::input::Keycode::new(seat_kc),
                        if pressed {
                            smithay::backend::input::KeyState::Pressed
                        } else {
                            smithay::backend::input::KeyState::Released
                        },
                        serial,
                        time,
                        |_, _, _| smithay::input::keyboard::FilterResult::<()>::Forward,
                    );
                }
            }
            zwp_virtual_keyboard_v1::Request::Modifiers { .. } => {}
            zwp_virtual_keyboard_v1::Request::Destroy => {}
            _ => {}
        }
    }

    /// Release every seat key this virtual keyboard still holds, so a VK client that
    /// disconnects mid-press cannot leave keys logically stuck.
    fn destroyed(
        state: &mut Self,
        _client: ClientId,
        _resource: &ZwpVirtualKeyboardV1,
        data: &PfVirtualKeyboard,
    ) {
        let held: Vec<u32> = data.inner.lock().unwrap().pressed.drain().map(|(_, kc)| kc).collect();
        if held.is_empty() {
            return;
        }
        if let Some(keyboard) = state.seat.get_keyboard() {
            let keyboard = keyboard.clone();
            for kc in held {
                let serial = next_serial();
                let time = wayland_time();
                keyboard.input(
                    state,
                    smithay::backend::input::Keycode::new(kc),
                    smithay::backend::input::KeyState::Released,
                    serial,
                    time,
                    |_, _, _| smithay::input::keyboard::FilterResult::<()>::Forward,
                );
            }
        }
    }
}

/// Per-client data attached to every Wayland client connection; holds the compositor's
/// per-client surface state.
#[derive(Default)]
pub struct ClientState {
    pub compositor_state: CompositorClientState,
}
/// Client lifecycle hooks; connect and disconnect need no bookkeeping here.
impl ClientData for ClientState {
    fn initialized(&self, _client_id: ClientId) {}
    fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
}

delegate_compositor!(AppState);
delegate_shm!(AppState);
delegate_output!(AppState);
delegate_seat!(AppState);
delegate_xdg_shell!(AppState);
delegate_dmabuf!(AppState);
delegate_image_capture_source!(AppState);
delegate_output_capture_source!(AppState);
delegate_image_copy_capture!(AppState);
delegate_ext_data_control!(AppState);
delegate_cursor_shape!(AppState);
delegate_fractional_scale!(AppState);
delegate_data_device!(AppState);
delegate_data_control!(AppState);
delegate_pointer_warp!(AppState);
delegate_relative_pointer!(AppState);
delegate_pointer_constraints!(AppState);
delegate_foreign_toplevel_list!(AppState);
delegate_xdg_decoration!(AppState);
delegate_layer_shell!(AppState);
delegate_single_pixel_buffer!(AppState);
delegate_viewporter!(AppState);
delegate_presentation!(AppState);
delegate_xdg_activation!(AppState);
delegate_primary_selection!(AppState);

/// Row stride (bytes) of a tightly-mapped RGBA8 GPU readback, derived from the mapping
/// length rather than assuming `width*4`.
///
/// Dividing the buffer length by the height recovers a padded stride, so a padded readback cannot
/// skew the cursor image; the result never drops below one full `width*4` row, and a zero height
/// short-circuits to one row to avoid dividing by zero.
pub(crate) fn rgba_readback_stride(buf_len: usize, height: usize, width: usize) -> usize {
    let row = width.saturating_mul(4);
    if height == 0 {
        return row;
    }
    (buf_len / height).max(row)
}

#[cfg(test)]
mod stride_tests {
    use super::rgba_readback_stride;

    /// A tightly-packed readback yields exactly `width*4` stride.
    #[test]
    fn packed_readback_is_width_times_four() {
        assert_eq!(rgba_readback_stride(64 * 4 * 48, 48, 64), 64 * 4);
        assert_eq!(rgba_readback_stride(3 * 4 * 2, 2, 3), 12);
    }

    /// A padded readback (extra bytes per row) recovers the true padded stride from the
    /// buffer length.
    #[test]
    fn padded_readback_recovers_real_stride() {
        let (w, h, pad) = (3usize, 2usize, 4usize);
        let stride = w * 4 + pad;
        assert_eq!(rgba_readback_stride(stride * h, h, w), stride);
    }

    /// Extracting pixels with the recovered stride from a padded buffer reproduces the
    /// written values with no row-to-row skew.
    #[test]
    fn padded_extraction_has_no_skew() {
        let (w, h) = (3usize, 2usize);
        let stride = 16usize;
        let mut buf = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                let o = y * stride + x * 4;
                buf[o] = x as u8 + 1;
                buf[o + 1] = y as u8 + 1;
            }
        }
        let s = rgba_readback_stride(buf.len(), h, w);
        assert_eq!(s, stride);
        for y in 0..h {
            for x in 0..w {
                let o = y * s + x * 4;
                assert_eq!(buf[o], x as u8 + 1);
                assert_eq!(buf[o + 1], y as u8 + 1);
            }
        }
    }

    /// Zero height returns one full row instead of dividing by zero.
    #[test]
    fn zero_height_no_divide_by_zero() {
        assert_eq!(rgba_readback_stride(0, 0, 10), 40);
    }

    /// A buffer shorter than a single row still reports a full `width*4` row.
    #[test]
    fn truncated_buffer_keeps_full_row() {
        assert_eq!(rgba_readback_stride(10, 5, 64), 64 * 4);
    }
}
