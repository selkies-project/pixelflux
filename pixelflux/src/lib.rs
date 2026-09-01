/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/*
  ▘    ▜ ▐▘▜
▛▌▌▚▘█▌▐ ▜▘▐ ▌▌▚▘
▙▌▌▞▖▙▖▐▖▐ ▐▖▙▌▞▖
▌
*/

//! # pixelflux
//!
//! A high-performance screen capture and encoding pipeline exposed as a Python extension via
//! PyO3. It supports two independent backends — **X11** (XShm + XFixes) and **Wayland**
//! (a headless [Smithay](https://github.com/Smithay/smithay) compositor) — and a shared
//! encoding layer that dispatches to software (x264/JPEG striping, OpenH264) or hardware
//! (NVENC, VA-API) encoders based on the available GPU and operator settings.
//!
//! ## Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`encoders`] | Encoder backends: software x264/JPEG, OpenH264, NVENC, VA-API, watermark overlay |
//! | [`wayland`] | Headless Smithay compositor, cursor rendering |
//! | [`x11`] | X11/XShm capture loop, XFixes out-of-band cursor monitor, and stripe dispatch |
//! | [`pipeline`] | Frame-processing policy shared by both backends (send/QP/keyframe decisions) |
//! | [`recording_sink`] | Unix-socket H.264 fan-out for external recording |
//! | [`computer_use`] | HTTP API for AI-agent desktop control (screenshots, input injection) |
//! | [`nvgpufilter`] | Multi-GPU NVENC device filtering via ioctl |
//!
//! ## Data flow
//!
//! ```text
//! Python  ──►  CaptureSettings  ──►  X11 / Wayland backend
//!                                         │
//!                                    frame pixels
//!                                         │
//!                                    ┌────┴────┐
//!                                    │ Encoder │  (NVENC / VAAPI / x264 / OpenH264 / JPEG)
//!                                    └────┬────┘
//!                                         │
//!                                   EncodedStripe(s)
//!                                         │
//!                                    Python callback
//! ```

#![allow(dead_code)]

use std::fs::File;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use gbm::{BufferObject, BufferObjectFlags, Device as RawGbmDevice, Format as GbmFormat};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use yuv::{
    BufferStoreMut, YuvBiPlanarImageMut, YuvConversionMode, YuvRange, YuvStandardMatrix,
};

use smithay::wayland::single_pixel_buffer::SinglePixelBufferState;
use smithay::wayland::viewporter::ViewporterState;
use smithay::wayland::presentation::PresentationState;
use smithay::wayland::selection::wlr_data_control::DataControlState;
use smithay::{
    backend::{
        allocator::{
            dmabuf::{Dmabuf, DmabufFlags},
            gbm::GbmDevice,
            Fourcc, Modifier,
        },
        drm::DrmNode,
        egl::{EGLContext, EGLDisplay},
        input::{Axis, AxisSource, KeyState, Keycode},
        renderer::{
            damage::OutputDamageTracker,
            element::{
                memory::MemoryRenderBufferRenderElement,
                surface::WaylandSurfaceRenderElement,
                AsRenderElements, Wrap,
            },
            gles::GlesRenderer,
            pixman::PixmanRenderer,
            Bind, ImportAll, ImportEgl, ImportMem, Renderer,
        },
    },
    desktop::{space::SpaceRenderElements, Space},
    input::{
        keyboard::{FilterResult, XkbConfig},
        pointer::{AxisFrame, ButtonEvent, CursorImageStatus, MotionEvent, RelativeMotionEvent},
        SeatState,
    },
    output::{Mode as OutputMode, Output, PhysicalProperties, Scale as OutputScale, Subpixel},
    reexports::{
        calloop::{
            channel::Event as CalloopEvent, generic::Generic, timer::{TimeoutAction, Timer},
            EventLoop, Interest, Mode, PostAction,
        },
        pixman,
        wayland_server::{Display, DisplayHandle},
    },
    utils::{Clock, Physical, Point, Rectangle, Scale, Transform},
    wayland::{
        compositor::{with_states, CompositorState},
        dmabuf::{DmabufFeedbackBuilder, DmabufState},
        fractional_scale::FractionalScaleManagerState,
        output::OutputManagerState,
        selection::data_device::DataDeviceState,
        seat::WaylandFocus,
        shell::xdg::XdgShellState,
        shm::ShmState,
        socket::ListeningSocketSource,
        virtual_keyboard::VirtualKeyboardManagerState,
        pointer_warp::PointerWarpManager,
        relative_pointer::RelativePointerManagerState,
        pointer_constraints::PointerConstraintsState,
        foreign_toplevel_list::ForeignToplevelListState,
        shell::xdg::decoration::XdgDecorationState,
    },
    desktop::{layer_map_for_output, PopupManager},
    wayland::shell::wlr_layer::WlrLayerShellState,
    wayland::xdg_activation::XdgActivationState,
    wayland::selection::primary_selection::PrimarySelectionState,
};

pub mod encoders {
    /// NVIDIA NVENC hardware H.264 encoder loaded via runtime `libcuda` / `libnvidia-encode`.
    pub mod nvenc;
    /// Cisco OpenH264 software H.264 encoder (BSD-licensed, full-frame, bitrate-controlled).
    pub mod oh264;
    /// PNG watermark overlay composited onto frames before encoding.
    pub mod overlay;
    /// CPU-based x264 / JPEG striped encoder with per-stripe change detection.
    pub mod software;
    /// VA-API hardware H.264 encoder for Intel / AMD GPUs via FFmpeg.
    pub mod vaapi;

    /// Size the CBR VBV/HRD buffer so rate control has enough slack to hold quality steady
    /// without letting end-to-end latency drift upward.
    ///
    /// The size is expressed as a multiple of one frame's bit budget (`bitrate_bps / fps`) rather
    /// than a fixed byte count so a live bitrate or framerate change rescales the buffer with it,
    /// preserving the same latency behavior at every operating point.
    ///
    /// # Arguments
    ///
    /// * `bitrate_bps` - Target bitrate in bits per second.
    /// * `fps` - Target frames per second.
    /// * `keyframe_interval_s` - Seconds between scheduled keyframes; `<= 0` for infinite GOP.
    /// * `multiplier` - Explicit buffer multiplier; `<= 0` selects the policy default (1.5 on
    ///   infinite GOP, 3 when keyframe interval is active).
    ///
    /// # Returns
    ///
    /// VBV buffer size in bits, clamped to `[1, u32::MAX]`.
    pub fn vbv_bits(bitrate_bps: u32, fps: f64, keyframe_interval_s: f64, multiplier: f64) -> u32 {
        let frame_bits = bitrate_bps as f64 / fps.max(1.0);
        let mult = if multiplier > 0.0 {
            multiplier
        } else if keyframe_interval_s > 0.0 {
            3.0
        } else {
            1.5
        };
        (frame_bits * mult).round().max(1.0).min(u32::MAX as f64) as u32
    }
}

/// Headless Wayland compositor and cursor rendering.
pub mod wayland;
/// Unix-socket H.264 recording fan-out for external capture tools.
pub mod recording_sink;
/// HTTP server implementing the Anthropic Computer Use spec for AI agent desktop control.
pub mod computer_use;
/// Frame-processing policy shared by the X11 and Wayland backends.
pub mod pipeline;
/// X11/XShm capture loop, stripe dispatch, and per-stripe change detection.
pub mod x11;
/// Multi-GPU NVENC device filtering via kernel ioctl.
pub mod nvgpufilter;

pub use encoders::nvenc;
pub use encoders::software::StripeState;
pub use encoders::vaapi;

fn get_process_rss_bytes() -> usize {
    if let Ok(contents) = std::fs::read_to_string("/proc/self/statm") {
        if let Some(rss_pages) = contents.split_whitespace().nth(1) {
            if let Ok(pages) = rss_pages.parse::<usize>() {
                return pages * 4096;
            }
        }
    }
    0
}

fn get_shm_usage_bytes() -> u64 {
    let mut total_size = 0;
    if let Ok(entries) = std::fs::read_dir("/dev/shm") {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                total_size += metadata.len();
            }
        }
    }
    total_size
}

fn calculate_memory_threshold(width: i32, height: i32) -> usize {
    let frame_size = (width.max(0) as usize)
        .saturating_mul(height.max(0) as usize)
        .saturating_mul(4);
    let base_app_memory: usize = 300 * 1024 * 1024;
    let buffer_allowance = frame_size.saturating_mul(20);
    let min_threshold: usize = 1536 * 1024 * 1024;
    base_app_memory
        .saturating_add(buffer_allowance)
        .max(min_threshold)
}

use encoders::nvenc::NvencEncoder;
use encoders::overlay::OverlayState;
use encoders::software::MAX_STRIPE_CAPACITY;
use encoders::vaapi::VaapiEncoder;

use wayland::cursor::Cursor;
use wayland::frontend::{AppState, ClientState, FocusTarget, GpuEncoder, next_serial, wayland_time, wayland_utime};

smithay::backend::renderer::element::render_elements! {
    pub CompositionElements<R, E> where R: ImportAll + ImportMem;
    Space=SpaceRenderElements<R, E>,
    Window=Wrap<E>,
    Cursor=MemoryRenderBufferRenderElement<R>,
    Surface=WaylandSurfaceRenderElement<R>,
}

/// Export the offscreen GBM render target as a Dmabuf so the very same GPU pixels can be both
/// rendered into and encoded with no intervening copy — the linchpin of the zero-copy capture path.
/// The returned dmabuf is the one handle the GLES renderer binds as its framebuffer AND a hardware
/// encoder (NVENC through CUDA, or VAAPI) imports to read those pixels directly, which only works if
/// the buffer is described precisely enough (fd, stride, DRM modifier) for the importer to interpret
/// it. One ARGB8888 plane is all that is carried because the compositor's offscreen target is exactly
/// that single-plane format.
fn create_dmabuf_from_bo(bo: &BufferObject<()>) -> Dmabuf {
    let fd = bo.fd().expect("Failed to get FD from GBM BO");
    let modifier = bo.modifier();
    let stride = bo.stride();
    let width = bo.width();
    let height = bo.height();

    let drm_modifier = Modifier::from(Into::<u64>::into(modifier));

    let mut builder = Dmabuf::builder(
        (width as i32, height as i32),
        Fourcc::Argb8888,
        drm_modifier,
        DmabufFlags::empty(),
    );

    builder.add_plane(fd, 0, 0, stride);
    builder.build().expect("Failed to build Dmabuf from GBM BO")
}

/// The full set of capture + encode parameters the Python layer hands to the Rust backend.
///
/// A single value configures a capture session end to end: capture geometry and frame rate, the
/// output mode (striped JPEG/x264 vs full-frame H.264), the H.264 quality and rate-control knobs,
/// cursor and watermark options, the encode-device selection, and the optional recording socket.
/// It derives `PartialEq` so the backend can detect when a live setting actually changed, and
/// `Clone` so each capture pipeline can own its own copy.
#[derive(Clone, Debug, PartialEq)]
pub struct RustCaptureSettings {
    pub width: i32,
    pub height: i32,
    pub scale: f64,
    pub capture_x: i32,
    pub capture_y: i32,
    pub target_fps: f64,
    pub jpeg_quality: i32,
    pub paint_over_jpeg_quality: i32,
    pub use_paint_over_quality: bool,
    pub paint_over_trigger_frames: u32,
    pub damage_block_threshold: u32,
    pub damage_block_duration: u32,
    pub output_mode: i32,
    pub video_crf: i32,
    pub video_paintover_crf: i32,
    pub video_paintover_burst_frames: i32,
    pub video_fullcolor: bool,
    pub video_fullframe: bool,
    pub video_streaming_mode: bool,
    pub capture_cursor: bool,
    /// Longest cursor edge the out-of-band cursor callback delivers; larger images are
    /// downscaled (`<= 0` = uncapped). Compositing via `capture_cursor` is unaffected.
    pub cursor_size_cap: i32,
    pub watermark_path: String,
    pub watermark_location_enum: i32,
    pub encode_node_index: i32,
    pub use_cpu: bool,
    pub use_openh264: bool,
    pub debug_logging: bool,
    pub auto_adjust_screen_capture_size: bool,
    pub recording_socket: String,
    /// When true, encoders emit the raw payload without the per-stripe header byte block;
    /// stripe metadata is then carried only on the frame attributes.
    pub omit_stripe_headers: bool,
    pub video_cbr_mode: bool,
    pub video_bitrate_kbps: i32,
    /// CBR VBV/HRD size as a multiple of one frame's bit budget (bitrate/framerate), so it
    /// rescales with live bitrate/fps changes. `<= 0` selects the policy default: 1.5 on an
    /// infinite GOP, 3 when scheduled keyframes are enabled.
    pub video_vbv_multiplier: f64,
    /// Seconds between scheduled recovery keyframes; `<= 0` keeps the GOP infinite
    /// (IDRs only on demand: client join / reset, recording cadence).
    pub keyframe_interval_s: f64,
    /// Rate-controlled (CBR) QP clamp: `video_max_qp` bounds the quality FLOOR (screen text stays
    /// legible under motion at the cost of overshooting impossible targets), `video_min_qp` bounds
    /// bit WASTE on easy content. 0 keeps the encoder's own default; CRF/CQP modes pin their QP
    /// directly and ignore these.
    pub video_min_qp: i32,
    pub video_max_qp: i32,
}

/// The per-frame decision/quality knobs every encoder re-reads from the settings on each
/// tick, so they retune a running capture with no encoder re-init: x264 reconfigures, NVENC
/// CQP retargets, VAAPI re-opens only its codec ctx, JPEG is stateless. Applied on the
/// thread that owns the settings copy. Structural switches (encoder, chroma, RC mode,
/// device) still need a capture restart.
#[derive(Clone, Copy, Debug)]
pub struct LiveTunables {
    pub jpeg_quality: i32,
    pub paint_over_jpeg_quality: i32,
    pub use_paint_over_quality: bool,
    pub paint_over_trigger_frames: u32,
    pub video_crf: i32,
    pub video_paintover_crf: i32,
    pub video_paintover_burst_frames: i32,
    pub video_streaming_mode: bool,
    pub keyframe_interval_s: f64,
    pub capture_cursor: bool,
}

impl LiveTunables {
    /// Snapshot the live-tunable subset out of a full settings value.
    pub fn from_settings(s: &RustCaptureSettings) -> Self {
        Self {
            jpeg_quality: s.jpeg_quality,
            paint_over_jpeg_quality: s.paint_over_jpeg_quality,
            use_paint_over_quality: s.use_paint_over_quality,
            paint_over_trigger_frames: s.paint_over_trigger_frames,
            video_crf: s.video_crf,
            video_paintover_crf: s.video_paintover_crf,
            video_paintover_burst_frames: s.video_paintover_burst_frames,
            video_streaming_mode: s.video_streaming_mode,
            keyframe_interval_s: s.keyframe_interval_s,
            capture_cursor: s.capture_cursor,
        }
    }

    /// Write these live tunables back into a full settings value in place.
    pub fn apply_to(&self, s: &mut RustCaptureSettings) {
        s.jpeg_quality = self.jpeg_quality;
        s.paint_over_jpeg_quality = self.paint_over_jpeg_quality;
        s.use_paint_over_quality = self.use_paint_over_quality;
        s.paint_over_trigger_frames = self.paint_over_trigger_frames;
        s.video_crf = self.video_crf;
        s.video_paintover_crf = self.video_paintover_crf;
        s.video_paintover_burst_frames = self.video_paintover_burst_frames;
        s.video_streaming_mode = self.video_streaming_mode;
        s.keyframe_interval_s = self.keyframe_interval_s;
        s.capture_cursor = self.capture_cursor;
    }
}

impl Default for RustCaptureSettings {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 768,
            scale: 1.0,
            capture_x: 0,
            capture_y: 0,
            target_fps: 60.0,
            jpeg_quality: 75,
            paint_over_jpeg_quality: 95,
            use_paint_over_quality: true,
            paint_over_trigger_frames: 15,
            damage_block_threshold: 10,
            damage_block_duration: 30,
            output_mode: 0,
            video_crf: 25,
            video_paintover_crf: 18,
            video_paintover_burst_frames: 5,
            video_fullcolor: false,
            video_fullframe: false,
            video_streaming_mode: false,
            capture_cursor: false,
            cursor_size_cap: 32,
            watermark_path: String::new(),
            watermark_location_enum: 0,
            encode_node_index: -2,
            use_cpu: false,
            use_openh264: false,
            debug_logging: false,
            auto_adjust_screen_capture_size: false,
            recording_socket: String::new(),
            omit_stripe_headers: false,
            video_cbr_mode: false,
            video_bitrate_kbps: 4000,
            video_vbv_multiplier: 0.0,
            keyframe_interval_s: 0.0,
            video_min_qp: 0,
            video_max_qp: 0,
        }
    }
}

/// Marshal a Python settings object into the plain owned Rust value both capture backends run on.
///
/// Fields are read by attribute name (`getattr`), not by position, so a caller can pass any object
/// exposing the `CaptureSettings` attributes — including a subclass carrying extras. Newer/optional
/// fields fall back to a default when absent rather than erroring, so a caller built against an
/// older schema still starts. Both the Wayland and X11 entry points route through this one reader
/// to prevent drift.
///
/// # Arguments
///
/// * `settings` - A Python object exposing `CaptureSettings` attributes (`capture_width`,
///   `capture_height`, `target_fps`, `jpeg_quality`, `video_crf`, etc.).
///
/// # Returns
///
/// An owned [`RustCaptureSettings`] on success, or a Python exception if a required field is
/// missing or has the wrong type.
pub(crate) fn extract_settings(settings: &Bound<'_, PyAny>) -> PyResult<RustCaptureSettings> {
    let watermark_path_obj = settings.getattr("watermark_path")?;
    let watermark_path = if let Ok(s) = watermark_path_obj.extract::<String>() {
        s
    } else if let Ok(b) = watermark_path_obj.extract::<Vec<u8>>() {
        String::from_utf8_lossy(&b).into_owned()
    } else {
        String::new()
    };

    let scale = settings
        .getattr("scale")
        .ok()
        .and_then(|x| x.extract().ok())
        .unwrap_or(1.0);

    Ok(RustCaptureSettings {
        width: settings.getattr("capture_width")?.extract()?,
        height: settings.getattr("capture_height")?.extract()?,
        scale,
        capture_x: settings.getattr("capture_x")?.extract()?,
        capture_y: settings.getattr("capture_y")?.extract()?,
        target_fps: settings.getattr("target_fps")?.extract()?,
        jpeg_quality: settings.getattr("jpeg_quality")?.extract()?,
        paint_over_jpeg_quality: settings.getattr("paint_over_jpeg_quality")?.extract()?,
        use_paint_over_quality: settings.getattr("use_paint_over_quality")?.extract()?,
        paint_over_trigger_frames: settings.getattr("paint_over_trigger_frames")?.extract()?,
        damage_block_threshold: settings.getattr("damage_block_threshold")?.extract()?,
        damage_block_duration: settings.getattr("damage_block_duration")?.extract()?,
        output_mode: settings.getattr("output_mode")?.extract()?,
        video_crf: settings.getattr("video_crf")?.extract()?,
        video_paintover_crf: settings.getattr("video_paintover_crf")?.extract()?,
        video_paintover_burst_frames: settings.getattr("video_paintover_burst_frames")?.extract()?,
        video_fullcolor: settings.getattr("video_fullcolor")?.extract()?,
        video_fullframe: settings.getattr("video_fullframe")?.extract()?,
        video_streaming_mode: settings.getattr("video_streaming_mode")?.extract()?,
        capture_cursor: settings.getattr("capture_cursor")?.extract()?,
        cursor_size_cap: settings
            .getattr("cursor_size_cap")
            .ok()
            .and_then(|v| v.extract::<i32>().ok())
            .unwrap_or(32),
        watermark_path,
        watermark_location_enum: settings.getattr("watermark_location_enum")?.extract()?,
        encode_node_index: settings.getattr("encode_node_index")?.extract()?,
        use_cpu: settings.getattr("use_cpu")?.extract()?,
        use_openh264: settings
            .getattr("use_openh264")
            .ok()
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false),
        debug_logging: settings.getattr("debug_logging")?.extract()?,
        auto_adjust_screen_capture_size: settings
            .getattr("auto_adjust_screen_capture_size")
            .ok()
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false),
        recording_socket: settings
            .getattr("recording_socket")
            .ok()
            .and_then(|v| v.extract::<String>().ok())
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var("PIXELFLUX_RECORDING_SOCKET").ok().filter(|s| !s.is_empty()))
            .or_else(|| std::env::var("SELKIES_RECORDING_SOCKET").ok().filter(|s| !s.is_empty()))
            .unwrap_or_default(),
        omit_stripe_headers: settings
            .getattr("omit_stripe_headers")
            .ok()
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false),
        video_cbr_mode: settings.getattr("video_cbr_mode")?.extract()?,
        video_bitrate_kbps: settings.getattr("video_bitrate_kbps")?.extract()?,
        video_vbv_multiplier: settings
            .getattr("video_vbv_multiplier")
            .ok()
            .and_then(|v| v.extract::<f64>().ok())
            .unwrap_or(0.0),
        keyframe_interval_s: settings
            .getattr("keyframe_interval_s")
            .ok()
            .and_then(|v| v.extract::<f64>().ok())
            .unwrap_or(0.0),
        video_min_qp: settings
            .getattr("video_min_qp")
            .ok()
            .and_then(|v| v.extract::<i32>().ok())
            .unwrap_or(0),
        video_max_qp: settings
            .getattr("video_max_qp")
            .ok()
            .and_then(|v| v.extract::<i32>().ok())
            .unwrap_or(0),
    })
}

/// Control messages sent from the Python-facing methods to the capture thread.
///
/// Every interaction with a running capture crosses the thread boundary as one of these variants
/// over the command channel: starting and stopping, injecting keyboard / pointer input, swapping
/// the xkb keymap, serving the clipboard, changing live rate and per-frame tunables, and the
/// computer-use queries that read back the screen, cursor, and geometry.
pub enum ThreadCommand {
    StartCapture(Py<PyAny>, RustCaptureSettings),
    StopCapture,
    SetCursorCallback(Py<PyAny>),
    SetClipboardCallback(Py<PyAny>),
    /// Server-side clipboard offer: the compositor owns the selection and serves `data` as
    /// `mime` (plus text aliases) to pasting clients.
    SetClipboard { mime: String, data: Vec<u8> },
    KeyboardKey { scancode: u32, state: u32 },
    /// Swap the seat keyboard's xkb keymap (XKB_KEYMAP_FORMAT_TEXT_V1 text); the caller owns
    /// keysym-to-keycode policy and injects the keycodes it defined via `KeyboardKey`.
    SetKeymapString(String),
    /// Reply with the smithay keyboard's keymap as an XKB_KEYMAP_FORMAT_TEXT_V1 string so a
    /// consumer (selkies) can build its reverse keysym map from the IDENTICAL keymap.
    GetXkbKeymap { reply: std::sync::mpsc::Sender<String> },
    PointerMotion { x: f64, y: f64 },
    PointerRelativeMotion { dx: f64, dy: f64 },
    /// `btn` is an evdev `BTN_` code by contract (e.g. 272 = BTN_LEFT, 273 = BTN_RIGHT,
    /// 274 = BTN_MIDDLE, 0x113 = BTN_SIDE / 0x114 = BTN_EXTRA for back/forward) and is passed
    /// straight through to smithay's pointer.
    PointerButton { btn: u32, state: u32 },
    PointerAxis { x: f64, y: f64 },
    UpdateCursorConfig { render_on_framebuffer: bool },
    /// On-demand keyframe request (client reconnect / decoder reset): forces a send and an IDR
    /// even on a static screen.
    RequestIdr,
    /// Live rate-control change for the Wayland calloop thread (parity with the X11 `rate_dirty`
    /// path). Each field is `None` when that dimension is unchanged.
    UpdateRate { bitrate_kbps: Option<i32>, vbv_multiplier: Option<f64>, fps: Option<f64> },
    /// Live per-frame tunables (quality / paint-over / streaming / cursor), applied to the
    /// calloop's settings and mirrored to the readback encode thread — no capture or encoder
    /// restart.
    UpdateTunables(LiveTunables),
    CuScreenshot { resp: std::sync::mpsc::Sender<Vec<u8>> },
    CuCursorPosition { resp: std::sync::mpsc::Sender<(f64, f64)> },
    CuGetInfo { resp: std::sync::mpsc::Sender<(i32, i32, f64)> },
}

/// Read the kernel driver bound to a render node for encoder routing.
///
/// An `nvidia` driver name routes to NVENC; anything else routes to VA-API. The name is
/// lowercased for case-insensitive substring matching, and is empty when the node has no driver
/// link (treated as "no detectable GPU" by the selection logic).
///
/// # Arguments
///
/// * `card_index` - DRM card index (maps to `/sys/class/drm/renderD{128 + card_index}`).
///
/// # Returns
///
/// Lowercased driver name, or an empty string if the node has no driver link.
pub(crate) fn get_gpu_driver(card_index: i32) -> String {
    let path = format!("/sys/class/drm/renderD{}/device/driver", 128 + card_index);
    match std::fs::read_link(&path) {
        Ok(link_path) => link_path.to_string_lossy().to_lowercase(),
        Err(_) => String::new(),
    }
}

/// A DRM card's identity as the kernel reports it, read from the device's
/// sysfs `uevent` (DRIVER=, PCI_ID=, OF_COMPATIBLE_n=) with per-file fallbacks
/// (`vendor`, `modalias`, the `driver` symlink). uevent is uniform across buses
/// (PCI, platform/devicetree, USB) and readable in unprivileged containers.
struct CardIdentity {
    driver: String,
    pci_vendor: Option<u32>,
    compatibles: Vec<String>,
}

/// Read a DRM card's `CardIdentity` from sysfs.
///
/// The device's `uevent` file is the primary source (`DRIVER=`, `PCI_ID=`, `OF_COMPATIBLE_n=`)
/// because it is uniform across buses and readable in unprivileged containers. Each field has a
/// fallback for cards whose `uevent` omits it: the `vendor` file for the PCI vendor, `modalias`
/// (`of:...C<compatible>`) for the devicetree compatibles, and the `driver` symlink for the name.
fn read_card_identity(device: &std::path::Path) -> CardIdentity {
    let mut id = CardIdentity { driver: String::new(), pci_vendor: None, compatibles: Vec::new() };
    if let Ok(uevent) = std::fs::read_to_string(device.join("uevent")) {
        for line in uevent.lines() {
            if let Some(v) = line.strip_prefix("DRIVER=") {
                id.driver = v.trim().to_lowercase();
            } else if let Some(v) = line.strip_prefix("PCI_ID=") {
                id.pci_vendor = v.split(':').next().and_then(|h| u32::from_str_radix(h, 16).ok());
            } else if line.starts_with("OF_COMPATIBLE_") && !line.starts_with("OF_COMPATIBLE_N") {
                if let Some(v) = line.split_once('=').map(|x| x.1) {
                    id.compatibles.push(v.trim().to_lowercase());
                }
            }
        }
    }
    if id.pci_vendor.is_none() {
        id.pci_vendor = std::fs::read_to_string(device.join("vendor"))
            .ok()
            .and_then(|v| u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok());
    }
    if id.compatibles.is_empty() {
        if let Ok(modalias) = std::fs::read_to_string(device.join("modalias")) {
            let modalias = modalias.trim();
            if let Some(rest) = modalias.strip_prefix("of:") {
                id.compatibles
                    .extend(rest.split('C').skip(1).map(|c| c.to_lowercase()));
            }
        }
    }
    if id.driver.is_empty() {
        id.driver = std::fs::read_link(device.join("driver"))
            .map(|p| p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default())
            .unwrap_or_default();
    }
    id
}

/// Human vendor name -> PCI vendor IDs. The kernel has no such table (the
/// grouping is conventional), and pci.ids/hwdata is often absent in containers,
/// so this is the one mapping that must be embedded — kept to the names only.
const VENDOR_PCI_IDS: &[(&str, &[u32])] = &[
    ("nvidia", &[0x10de, 0x12d2]),
    ("amd", &[0x1002, 0x1022]),
    ("ati", &[0x1002]),
    ("intel", &[0x8086, 0x8087]),
    ("arm", &[0x13b5]),
    ("qualcomm", &[0x5143, 0x17cb]),
    ("broadcom", &[0x14e4]),
    ("apple", &[0x106b]),
    ("mediatek", &[0x14c3]),
    ("samsung", &[0x144d]),
    ("vmware", &[0x15ad]),
    ("microsoft", &[0x1414]),
    ("virtio", &[0x1af4]),
];

/// Human name -> devicetree vendor prefix, only where they differ (a token
/// equal to the prefix itself, e.g. "qcom" or "rockchip", matches directly).
const OF_PREFIX_ALIASES: &[(&str, &str)] = &[
    ("mali", "arm"),
    ("qualcomm", "qcom"),
    ("adreno", "qcom"),
    ("broadcom", "brcm"),
    ("videocore", "brcm"),
    ("imagination", "img"),
    ("powervr", "img"),
];

/// Does a card match the requested token? Accepted token forms, checked against
/// the identity the kernel itself reports: a kernel DRIVER name (exact, no
/// table), a raw PCI vendor ID ("0x10de"/"10de"), a devicetree vendor prefix
/// (literal first segment of any compatible), or a human vendor name resolved
/// through the small embedded alias maps above.
fn card_matches_token(token: &str, id: &CardIdentity) -> bool {
    if !id.driver.is_empty() && token == id.driver {
        return true;
    }
    if let Some(vid) = id.pci_vendor {
        if u32::from_str_radix(token.trim_start_matches("0x"), 16) == Ok(vid) {
            return true;
        }
        if let Some((_, ids)) = VENDOR_PCI_IDS.iter().find(|(n, _)| *n == token) {
            if ids.contains(&vid) {
                return true;
            }
        }
    }
    if !id.compatibles.is_empty() {
        let prefix = OF_PREFIX_ALIASES
            .iter()
            .find(|(n, _)| *n == token)
            .map(|(_, p)| *p)
            .unwrap_or(token);
        let want = format!("{prefix},");
        if id.compatibles.iter().any(|c| c.starts_with(&want)) {
            return true;
        }
    }
    false
}

/// Parse an auto-GPU request (the CaptureSettings `auto_gpu` field, which selkies
/// fills from --auto-gpu / SELKIES_AUTO_GPU). `None` = disabled; `Some(None)` =
/// pick the first GPU overall ("true"); `Some(Some(token))` = pick the first GPU
/// whose kernel identity matches the case-insensitive token (vendor name, kernel
/// driver name, devicetree vendor prefix, or raw PCI vendor id).
fn parse_auto_gpu(value: &str) -> Option<Option<String>> {
    let value = value.to_lowercase();
    match value.as_str() {
        "" | "false" | "0" | "off" | "no" => None,
        "true" | "1" | "on" | "yes" => Some(None),
        token => Some(Some(token.to_string())),
    }
}

/// Resolve a usable `/dev/dri/renderD*` node, optionally matching a vendor/driver token.
///
/// Cards under `/sys/class/drm` are walked in numeric order, skipping cards with no render node
/// (e.g. IPMI/VGA). When `/sys/class/drm` is unreadable (container without `/sys`), falls
/// through to scanning `/dev/dri` directly — that fallback has no device identity so a `token`
/// request cannot be satisfied there.
///
/// # Arguments
///
/// * `token` - Optional vendor/driver filter: a kernel driver name, PCI vendor ID (`"0x10de"`),
///   devicetree vendor prefix, or human vendor name (`"nvidia"`, `"intel"`). `None` picks the
///   first available node.
///
/// # Returns
///
/// A `/dev/dri/renderD*` path, or `None` if no matching node exists.
fn auto_select_render_node(token: Option<&str>) -> Option<String> {
    let mut cards: Vec<(u32, std::path::PathBuf)> = std::fs::read_dir("/sys/class/drm")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let num = e.file_name().into_string().ok()?.strip_prefix("card")?.parse::<u32>().ok()?;
            Some((num, e.path()))
        })
        .collect();
    cards.sort_by_key(|(n, _)| *n);
    for (_, path) in &cards {
        if let Some(t) = token {
            if !card_matches_token(t, &read_card_identity(&path.join("device"))) {
                continue;
            }
        }
        if let Ok(drm_entries) = std::fs::read_dir(path.join("device/drm")) {
            for de in drm_entries.flatten() {
                let name = de.file_name().into_string().unwrap_or_default();
                if name.starts_with("renderD") {
                    let dev = format!("/dev/dri/{}", name);
                    if std::path::Path::new(&dev).exists() {
                        return Some(dev);
                    }
                }
            }
        }
    }
    if token.is_some() {
        return None;
    }
    let mut nodes: Vec<String> = std::fs::read_dir("/dev/dri")
        .ok()?
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with("renderD"))
        .collect();
    nodes.sort();
    nodes.first().map(|n| format!("/dev/dri/{}", n))
}


/// One captured host-pixel frame in flight from the calloop (render/readback) to the Wayland
/// encode thread: the pixels plus the per-frame inputs of the encode dispatch (damage,
/// overlay animation); the IDR request travels separately via the controls atomic.
pub struct WlFrame {
    /// Pool slot id; travels with the buffer so recycle returns it to the right slot.
    id: usize,
    buf: Vec<u8>,
    frame_id: u16,
    damage: Vec<Rectangle<i32, Physical>>,
    is_animated: bool,
}

/// Interior state of `WlFramePool`: the free-buffer list plus the single publish slot.
struct WlPoolInner {
    free: Vec<(usize, Vec<u8>)>,
    slot: Option<WlFrame>,
}

/// Render->encode handoff for the Wayland readback paths, mirroring the X11 FramePool's
/// single-slot non-dropping design with one deliberate difference: the calloop thread is also
/// the compositor + input dispatcher, so it must NEVER block on the pool. `try_begin` hands
/// out a buffer only while the publish slot is empty, so `publish` cannot block, and a
/// saturated encoder throttles capture by SKIPPING ticks (compositor damage accumulates via
/// buffer age, so nothing is lost). Every published frame is encoded, in order: the H.264
/// reference chain stays contiguous exactly as on X11.
pub struct WlFramePool {
    inner: Mutex<WlPoolInner>,
    cv: Condvar,
    stop: AtomicBool,
}

impl WlFramePool {
    /// Pre-allocate all `n` capture buffers up front (each `buf_len` bytes, all initially
    /// free) so the steady-state render/encode loop hands buffers around without ever allocating on
    /// the hot path.
    fn new(n: usize, buf_len: usize) -> Self {
        Self {
            inner: Mutex::new(WlPoolInner {
                free: (0..n).map(|i| (i, vec![0u8; buf_len])).collect(),
                slot: None,
            }),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
        }
    }

    /// Calloop: reserve a buffer for the next render/readback, NON-blocking. None means the
    /// encoder is still behind (slot full or every buffer in flight) -- skip this tick.
    /// Because the calloop is the only producer, a successful reservation guarantees the
    /// following `publish` finds the slot empty (the consumer only ever drains it).
    fn try_begin(&self) -> Option<(usize, Vec<u8>)> {
        let mut g = self.inner.lock().unwrap();
        if g.slot.is_some() {
            return None;
        }
        g.free.pop()
    }

    /// Calloop: hand the filled buffer to the encode thread. Never blocks (see `try_begin`).
    fn publish(&self, frame: WlFrame) {
        let mut g = self.inner.lock().unwrap();
        debug_assert!(g.slot.is_none());
        g.slot = Some(frame);
        drop(g);
        self.cv.notify_all();
    }

    /// Calloop: return an unused reservation (render failed, readback skipped).
    fn cancel(&self, id: usize, buf: Vec<u8>) {
        self.inner.lock().unwrap().free.push((id, buf));
    }

    /// Encode: block until a frame is published (Some) or the pool shuts down (None). The
    /// wait is bounded, re-checking `stop` as defense-in-depth against a lost wakeup.
    fn take(&self) -> Option<WlFrame> {
        let mut g = self.inner.lock().unwrap();
        loop {
            if let Some(f) = g.slot.take() {
                return Some(f);
            }
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            let (gg, _) = self.cv.wait_timeout(g, Duration::from_millis(20)).unwrap();
            g = gg;
        }
    }

    /// Encode: return an encoded frame's buffer so the calloop can capture into it again.
    /// No notify: the only waiter (take) waits on the slot, and try_begin never blocks.
    fn recycle(&self, id: usize, buf: Vec<u8>) {
        self.inner.lock().unwrap().free.push((id, buf));
    }

    /// Store stop under the lock so take() can't check stop==false and then park after the
    /// notify already fired (lost wakeup); notify after unlocking.
    fn shutdown(&self) {
        let g = self.inner.lock().unwrap();
        self.stop.store(true, Ordering::Release);
        drop(g);
        self.cv.notify_all();
    }
}

/// Cross-thread controls for the Wayland encode thread, the X11 `Controls` scheme: the
/// UpdateRate handler stores the current values then flips `rate_dirty` with Release; the
/// encode thread swaps it with Acquire and re-reads the payload, never seeing it half-applied.
/// `force_idr` is swapped just before each encode, so an on-demand keyframe lands on the
/// frame ALREADY in flight instead of waiting one pipeline stage for the next publish.
pub struct WlEncodeControls {
    rate_dirty: AtomicBool,
    bitrate_kbps: AtomicI32,
    vbv_mult_milli: AtomicI32,
    fps_milli: AtomicU64,
    force_idr: AtomicBool,
    /// Pending per-frame tunables for the encode thread (mutex, not atomics: one struct, set
    /// rarely, read only when the dirty flag says so).
    tunables_dirty: AtomicBool,
    tunables: Mutex<Option<LiveTunables>>,
}

impl WlEncodeControls {
    fn new() -> Self {
        Self {
            rate_dirty: AtomicBool::new(false),
            bitrate_kbps: AtomicI32::new(0),
            vbv_mult_milli: AtomicI32::new(0),
            fps_milli: AtomicU64::new(0),
            force_idr: AtomicBool::new(false),
            tunables_dirty: AtomicBool::new(false),
            tunables: Mutex::new(None),
        }
    }
}

/// Two buffers: one being encoded while the calloop fills the other. try_begin gates on the
/// publish slot, so a deeper pool would only add latency (staler frames), never overlap.
const WL_POOL_SURFACES: usize = 2;

/// Shared capture stats: whichever thread owns the encoders counts frames/stripes and
/// composes `desc` + `n_stripes` (the encoder half of the 1 s debug log line); the calloop
/// log loads, prints and resets the counters.
pub struct WlEncodeStats {
    frames: AtomicU32,
    stripes: AtomicU32,
    n_stripes: AtomicU32,
    desc: Mutex<String>,
}

impl WlEncodeStats {
    fn new() -> Self {
        Self {
            frames: AtomicU32::new(0),
            stripes: AtomicU32::new(0),
            n_stripes: AtomicU32::new(1),
            desc: Mutex::new(String::new()),
        }
    }
}

/// Everything the Wayland encode thread needs, fixed for the life of one capture (a
/// StartCapture reconfigure tears the thread down and spawns a fresh one). Rate changes
/// flow through `controls`; damage and the IDR request arrive per-frame in `WlFrame`.
struct WlEncodeConfig {
    settings: RustCaptureSettings,
    /// GLES readback is RGBA; the pixman framebuffer is BGRA. Selects CSC + encoder input kind.
    use_gpu: bool,
    /// Attempt a HW (NVENC/VAAPI) readback session before falling back to the CPU encoders.
    try_gpu: bool,
    /// HW session handed back by the previous encode thread; reconfigured in place and
    /// reused when still compatible, sparing the stream a session rebuild.
    prior: Option<GpuEncoder>,
    recording_sink: Option<Arc<crate::recording_sink::RecordingSink>>,
    deliver_tx: std::sync::mpsc::SyncSender<Vec<EncodedStripe>>,
    controls: Arc<WlEncodeControls>,
    stats: Arc<WlEncodeStats>,
}

/// Build the readback-mode encoder set on the thread that will own and drive it.
///
/// The result is one of three shapes: a hardware `GpuEncoder` (NVENC or VAAPI) consuming host
/// NV12/YUV444 via `encode_raw`, the OpenH264 full-frame encoder, or `(None, None)` for the
/// striped x264/JPEG path (where `encode_cpu` builds its own per-stripe state). Selection follows
/// the settings and the effective encode device — an auto index below zero resolves to device 0:
/// OpenH264 when explicitly opted in, NVENC on an NVIDIA driver, VAAPI on any other GPU, except a
/// 4:4:4 full-color request which VAAPI cannot do reliably and so falls through to the CPU. When a
/// compatible NVENC session is handed over from the previous encode thread it is reconfigured in
/// place (milliseconds) rather than rebuilt, which would stall the stream. The EGL display is
/// always null here: readback mode never imports dmabufs.
fn build_readback_encoders(
    settings: &RustCaptureSettings,
    try_gpu: bool,
    prior: Option<GpuEncoder>,
) -> (Option<GpuEncoder>, Option<crate::encoders::oh264::Openh264Encoder>) {
    if settings.output_mode == 1 && settings.use_openh264 {
        match crate::encoders::oh264::Openh264Encoder::new(settings) {
            Some(e) => {
                println!("[Wayland] OpenH264 software encoder selected.");
                return (None, Some(e));
            }
            None => {
                eprintln!("[Wayland] OpenH264 init failed; falling back to x264 software.");
                return (None, None);
            }
        }
    }
    if !try_gpu {
        return (None, None);
    }
    let encode_driver = get_gpu_driver(settings.encode_node_index.max(0));
    println!(
        "[Wayland] Encode Node Index: {} | Driver: {}",
        settings.encode_node_index.max(0), encode_driver
    );
    if encode_driver.contains("nvidia") {
        if let Some(GpuEncoder::Nvenc(mut enc)) = prior {
            match enc.reconfigure_resolution(settings) {
                Ok(()) => {
                    println!("[Wayland] NVENC session reconfigured in place.");
                    return (Some(GpuEncoder::Nvenc(enc)), None);
                }
                Err(e) => eprintln!(
                    "[Wayland] NVENC in-place reconfigure unavailable ({e}); rebuilding."
                ),
            }
        }
        println!("[Wayland] Nvidia Encoder detected. Initializing NVENC...");
        match NvencEncoder::new(settings, std::ptr::null()) {
            Ok(e) => {
                println!("[Wayland] NVENC Encoder initialized successfully.");
                return (Some(GpuEncoder::Nvenc(e)), None);
            }
            Err(e) => eprintln!("[Wayland] Failed to init NVENC: {}. Falling back to CPU.", e),
        }
    } else {
        println!("[Wayland] Initializing Unified VAAPI Encoder...");
        if settings.video_fullcolor {
            println!("[Wayland] 4:4:4 Fullcolor requested. VAAPI does not support this profile reliably. Falling back to CPU.");
        } else {
            match VaapiEncoder::new(settings) {
                Ok(e) => {
                    println!("[Wayland] VAAPI Encoder initialized successfully.");
                    return (Some(GpuEncoder::Vaapi(e)), None);
                }
                Err(e) => eprintln!("[Wayland] Failed to init VAAPI: {}. Falling back to CPU.", e),
            }
        }
    }
    (None, None)
}

/// Encode-thread body for the Wayland readback paths: drain published frames and run the
/// full encode dispatch, owning the encoders for the life of the capture.
///
/// Owning the encoders on this one thread (created, driven, and dropped here) is what lets the
/// calloop overlap the next render/readback with this encode — the same capture‖encode split used
/// on X11, minus the renderer, which is genuinely calloop-affine (EGL/GBM/dmabuf and the pixman
/// targets). Each published frame is processed in order so the H.264 reference chain stays
/// contiguous:
///
/// 1. **Apply cross-thread changes**: live tunables and rate/VBV/fps updates are read here, on the
///    thread that owns the encoders (each `Acquire` swap pairs with the command handler's `Release`
///    store, so a payload is never seen half-applied). The IDR request is swapped as late as
///    possible, so a request that arrived while this frame was in flight is honored one pipeline
///    stage earlier than the next publish.
/// 2. **Dispatch by encoder**: a hardware session runs `decide_hw_fullframe`, then colorspace-
///    converts only the frames actually being encoded (RGBA vs BGRA source chosen by the renderer)
///    into the reused NV12/YUV444 buffer and calls `encode_raw`; OpenH264 encodes host pixels
///    directly; otherwise `encode_cpu` runs the striped x264/JPEG path with compositor damage. The
///    software H.264 path keeps an infinite GOP, forcing an IDR only on an explicit request or the
///    configured interval, and an explicit request also forces a full JPEG resend for joiners.
/// 3. **Recycle then deliver**: the capture buffer is recycled BEFORE delivery so a slow consumer
///    never pins one, then the stripes go to the delivery thread through a single-slot `send` whose
///    blocking is the backpressure that overlaps delivery with the next render + encode.
///
/// On exit the hardware session is handed back to the calloop so a restart can reuse it in place
/// when the new settings stay compatible (a plain `StopCapture` just drops it).
fn wayland_encode_loop(pool: &WlFramePool, cfg: WlEncodeConfig) -> Option<GpuEncoder> {
    crate::boost_thread_priority(-10);
    let mut settings = cfg.settings;
    let (mut video_encoder, mut openh264_encoder) =
        build_readback_encoders(&settings, cfg.try_gpu, cfg.prior);
    if cfg.try_gpu && video_encoder.is_none() && openh264_encoder.is_none() {
        println!("[Wayland] Decision: No GPU Encoder available -> Using CPU Software Encoding.");
    }
    let n_stripes = wayland_stripe_count(
        &settings,
        video_encoder.is_some() || openh264_encoder.is_some(),
    );
    cfg.stats.n_stripes.store(n_stripes as u32, Ordering::Relaxed);
    *cfg.stats.desc.lock().unwrap() = encoder_desc(
        &settings,
        video_encoder.as_ref(),
        openh264_encoder.is_some(),
        false,
    );
    log_stream_settings(&settings, n_stripes, video_encoder.as_ref(), openh264_encoder.is_some());

    let width = settings.width;
    let height = settings.height;
    let mut stripes: Vec<StripeState> = Vec::with_capacity(MAX_STRIPE_CAPACITY);
    let mut hw_state = StripeState::default();
    let mut nv12_buffer: Vec<u8> = if video_encoder.is_some() {
        let n = (width * height) as usize;
        vec![0u8; if settings.video_fullcolor { n * 3 } else { n * 3 / 2 }]
    } else {
        Vec::new()
    };

    while let Some(mut f) = pool.take() {
        if cfg.controls.tunables_dirty.swap(false, Ordering::Acquire) {
            if let Some(t) = cfg.controls.tunables.lock().unwrap().take() {
                t.apply_to(&mut settings);
            }
        }
        if cfg.controls.rate_dirty.swap(false, Ordering::Acquire) {
            settings.video_bitrate_kbps = cfg.controls.bitrate_kbps.load(Ordering::Relaxed);
            settings.video_vbv_multiplier =
                cfg.controls.vbv_mult_milli.load(Ordering::Relaxed) as f64 / 1000.0;
            let fps = (cfg.controls.fps_milli.load(Ordering::Relaxed) as f64) / 1000.0;
            if fps > 0.0 {
                settings.target_fps = fps;
            }
            match video_encoder.as_mut() {
                Some(GpuEncoder::Nvenc(enc)) => enc.reconfigure_rate(&settings),
                Some(GpuEncoder::Vaapi(enc)) => enc.reconfigure_rate(&settings),
                None => {}
            }
            if let Some(enc) = openh264_encoder.as_mut() {
                enc.reconfigure_rate(settings.video_bitrate_kbps, settings.target_fps);
            }
        }

        let requested_idr = cfg.controls.force_idr.swap(false, Ordering::Relaxed);

        let mut out: Vec<EncodedStripe> = Vec::new();
        if let Some(ref mut encoder) = video_encoder {
            let decision = crate::pipeline::decide_hw_fullframe(
                &mut hw_state,
                &settings,
                f.frame_id,
                !f.damage.is_empty(),
                f.is_animated,
                requested_idr,
            );
            if decision.send {
                let w = width as u32;
                let h = height as u32;
                if settings.video_fullcolor {
                    let y_size = (w * h) as usize;
                    let (y_plane, rest) = nv12_buffer.split_at_mut(y_size);
                    let (u_plane, v_plane) = rest.split_at_mut(y_size);
                    let mut planar_image = yuv::YuvPlanarImageMut {
                        y_plane: BufferStoreMut::Borrowed(y_plane),
                        y_stride: w,
                        u_plane: BufferStoreMut::Borrowed(u_plane),
                        u_stride: w,
                        v_plane: BufferStoreMut::Borrowed(v_plane),
                        v_stride: w,
                        width: w,
                        height: h,
                    };
                    let _ = if cfg.use_gpu {
                        yuv::rgba_to_yuv444(&mut planar_image, &f.buf, w * 4, YuvRange::Full, YuvStandardMatrix::Bt709, YuvConversionMode::Fast)
                    } else {
                        yuv::bgra_to_yuv444(&mut planar_image, &f.buf, w * 4, YuvRange::Full, YuvStandardMatrix::Bt709, YuvConversionMode::Fast)
                    };
                } else {
                    let y_size = (w * h) as usize;
                    let (y_plane, uv_plane) = nv12_buffer.split_at_mut(y_size);
                    let mut planar_image = YuvBiPlanarImageMut {
                        y_plane: BufferStoreMut::Borrowed(y_plane),
                        y_stride: w,
                        uv_plane: BufferStoreMut::Borrowed(uv_plane),
                        uv_stride: w,
                        width: w,
                        height: h,
                    };
                    let csc = if cfg.use_gpu {
                        yuv::rgba_to_yuv_nv12(&mut planar_image, &f.buf, w * 4, YuvRange::Limited, YuvStandardMatrix::Bt709, YuvConversionMode::Fast)
                    } else {
                        yuv::bgra_to_yuv_nv12(&mut planar_image, &f.buf, w * 4, YuvRange::Limited, YuvStandardMatrix::Bt709, YuvConversionMode::Fast)
                    };
                    if let Err(e) = csc {
                        eprintln!("[wl-encode] NV12 CSC failed: {e:?}");
                    }
                }
                let force_idr = decision.force_idr;
                let result = match encoder {
                    GpuEncoder::Nvenc(enc) => {
                        enc.encode_raw(&nv12_buffer, f.frame_id as u64, decision.target_qp, force_idr)
                    }
                    GpuEncoder::Vaapi(enc) => {
                        enc.encode_raw(&nv12_buffer, f.frame_id as u64, decision.target_qp, force_idr)
                    }
                };
                match result {
                    Ok(data) if !data.is_empty() => out.push(EncodedStripe {
                        data,
                        data_type: 2,
                        stripe_y_start: 0,
                        stripe_height: height,
                        frame_id: f.frame_id as i32,
                    }),
                    Ok(_) => {}
                    Err(e) => eprintln!("HW Encode Error: {}", e),
                }
            }
        } else if let Some(ref mut enc) = openh264_encoder {
            let decision = crate::pipeline::decide_hw_fullframe(
                &mut hw_state,
                &settings,
                f.frame_id,
                !f.damage.is_empty(),
                f.is_animated,
                requested_idr,
            );
            if decision.send {
                let force_idr = decision.force_idr;
                let stride = (width * 4) as usize;
                match enc.encode_host_argb(&f.buf, stride, f.frame_id as u64, force_idr, cfg.use_gpu) {
                    Ok(data) if !data.is_empty() => out.push(EncodedStripe {
                        data,
                        data_type: 2,
                        stripe_y_start: 0,
                        stripe_height: height,
                        frame_id: f.frame_id as i32,
                    }),
                    Ok(_) => {}
                    Err(e) => eprintln!("OpenH264 Encode Error: {}", e),
                }
            }
        } else {
            let mut damage = std::mem::take(&mut f.damage);
            if f.is_animated {
                damage.push(Rectangle::new((0, 0).into(), (width, height).into()));
            }
            let force_idr_all = requested_idr
                || (settings.output_mode == 1
                    && crate::pipeline::periodic_idr_due(&settings, f.frame_id));
            out = encoders::software::encode_cpu(
                &mut stripes,
                &f.buf,
                width,
                height,
                &damage,
                &settings,
                f.frame_id,
                cfg.use_gpu,
                false,
                force_idr_all,
            );
        }

        let WlFrame { id, buf, .. } = f;
        pool.recycle(id, buf);
        if !out.is_empty() {
            cfg.stats.frames.fetch_add(1, Ordering::Relaxed);
            cfg.stats.stripes.fetch_add(out.len() as u32, Ordering::Relaxed);
            if let Some(ref socket) = cfg.recording_sink {
                for stripe in &out {
                    let _ = socket.write_encoded_frame(stripe);
                }
                if socket.should_force_idr() {
                    cfg.controls.force_idr.store(true, Ordering::Relaxed);
                }
            }
            let _ = cfg.deliver_tx.send(out);
        }
    }
    if settings.debug_logging {
        println!(
            "[Wayland] Encode thread exiting (hw={}, openh264={}, stripes={}).",
            video_encoder.is_some(),
            openh264_encoder.is_some(),
            stripes.len()
        );
    }
    video_encoder
}

/// Compose the encoder half of the 1 s debug log line (backend, colorspace, frame mode) for
/// whichever thread owns the encoders.
fn encoder_desc(
    settings: &RustCaptureSettings,
    video_encoder: Option<&GpuEncoder>,
    openh264: bool,
    zero_copy: bool,
) -> String {
    if settings.output_mode == 0 {
        return format!("JPEG Q:{}", settings.jpeg_quality);
    }
    let copy_mode = if zero_copy { "ZeroCopy" } else { "Readback" };
    let backend = if openh264 {
        "OpenH264".to_string()
    } else {
        match video_encoder {
            Some(GpuEncoder::Nvenc(_)) => format!("NVENC ({})", copy_mode),
            Some(GpuEncoder::Vaapi(_)) => format!("VAAPI ({})", copy_mode),
            None => "CPU".to_string(),
        }
    };
    let is_444 = !openh264
        && match video_encoder {
            Some(GpuEncoder::Nvenc(_)) => settings.video_fullcolor,
            Some(_) => false,
            None => settings.video_fullcolor,
        };
    let cs_str = if is_444 { "CS_IN:I444" } else { "CS_IN:I420" };
    let range_str = if is_444 { "FR" } else { "LR" };
    let frame_str = if video_encoder.is_some() || openh264 || settings.video_fullframe {
        "FF"
    } else {
        "Striped"
    };
    format!("H264 ({}) {} {} {} CRF:{}", backend, cs_str, range_str, frame_str, settings.video_crf)
}

/// Decide how many horizontal stripes a frame is split into, which is really the choice of how
/// much encode parallelism to spend on it. A full-frame session (`fullframe_encoder`: a HW or
/// OpenH264 encoder, or a forced `video_fullframe`) is one contiguous H.264 stream and so is always a
/// SINGLE stripe; the striped CPU paths instead fan the frame out across cores — bounded by
/// `height/64` for H.264 so no stripe is shorter than a macroblock row, or by `height` for JPEG — so
/// a many-core host encodes a frame in parallel without over-subscribing a tiny one.
fn wayland_stripe_count(settings: &RustCaptureSettings, fullframe_encoder: bool) -> usize {
    let num_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if settings.output_mode == 1 {
        if fullframe_encoder || settings.video_fullframe {
            1
        } else {
            let min_h = 64;
            if settings.height < min_h {
                1
            } else {
                num_cores.min((settings.height / min_h) as usize).max(1)
            }
        }
    } else {
        num_cores.min(settings.height as usize).max(1)
    }
}

/// One-shot "Stream settings active" line, printed by the thread that owns the encoders
/// once the selection is final (calloop for zero-copy, encode thread for readback).
fn log_stream_settings(
    settings: &RustCaptureSettings,
    n_stripes: usize,
    video_encoder: Option<&GpuEncoder>,
    openh264: bool,
) {
    let mut log_msg = format!(
        "Stream settings active -> Res: {}x{} | FPS: {:.1} | Stripes: {}",
        settings.width, settings.height, settings.target_fps, n_stripes
    );

    if settings.output_mode == 0 {
        log_msg.push_str(&format!(" | Mode: JPEG | Quality: {}", settings.jpeg_quality));
        if settings.use_paint_over_quality {
            log_msg.push_str(&format!(
                " | PaintOver Q: {} (Trigger: {}f)",
                settings.paint_over_jpeg_quality, settings.paint_over_trigger_frames
            ));
        }
    } else {
        let encoder_type = if openh264 {
            "OpenH264"
        } else {
            match video_encoder {
                Some(GpuEncoder::Nvenc(_)) => "NVENC",
                Some(GpuEncoder::Vaapi(_)) => "VAAPI",
                None => "CPU",
            }
        };
        log_msg.push_str(&format!(" | Mode: H264 ({})", encoder_type));

        if video_encoder.is_some() || openh264 || settings.video_fullframe {
            log_msg.push_str(" FullFrame");
        } else {
            log_msg.push_str(" Striped");
        }

        if settings.video_streaming_mode {
            log_msg.push_str(" Streaming");
        }

        log_msg.push_str(&format!(" | CRF: {}", settings.video_crf));

        if settings.use_paint_over_quality {
            log_msg.push_str(&format!(
                " | PaintOver CRF: {} (Burst: {}f)",
                settings.video_paintover_crf, settings.video_paintover_burst_frames
            ));
        }

        let is_actually_444 = if openh264 {
            false
        } else {
            match video_encoder {
                Some(GpuEncoder::Nvenc(_)) => settings.video_fullcolor,
                Some(_) => false,
                None => settings.video_fullcolor,
            }
        };

        let range_str = if is_actually_444 {
            "I444 (Full Range)"
        } else {
            "I420 (Limited Range)"
        };
        log_msg.push_str(&format!(" | Colorspace: {}", range_str));
    }

    log_msg.push_str(&format!(
        " | Damage Thresh: {}f | Damage Dur: {}f",
        settings.damage_block_threshold, settings.damage_block_duration
    ));

    println!("{}", log_msg);
}

/// The main execution loop of the Wayland backend.
///
/// This function is the central nervous system of the backend. It runs on its own thread and owns
/// the entire lifecycle of the headless Wayland compositor:
///
/// 1. **Initialization**: builds the `calloop` event loop and the Wayland display, raises
///    libwayland's per-client buffer limit when the newer setter is available (resolved at runtime
///    so the module still loads against older libwayland), and brings up the rendering pipeline —
///    GBM/EGL hardware acceleration on the resolved DRM render node, falling back to software
///    rendering (Pixman) when no node is usable.
/// 2. **State management**: constructs and holds the `AppState` — the Wayland globals (compositor,
///    seat, SHM, shell, dmabuf, selections, and the rest), the single virtual `HEADLESS-1` output,
///    the frame buffer and encode pools, and the window / cursor / encode bookkeeping.
/// 3. **Event dispatch**:
///    - **Command channel**: control messages from the Python thread — start/stop, input injection,
///      keymap and clipboard operations, live rate / tunable changes, and the computer-use queries.
///    - **Wayland socket**: accepts client connections and drives the compositor protocol.
/// 4. **StartCapture reconfigure**: reprograms the virtual output's mode / scale / refresh, resizes
///    the framebuffer and offscreen GBM buffer, and fullscreens mapped toplevels. The encode device
///    is resolved here: an operator's explicit `encode_node_index` (-1 software, >= 0 a device)
///    always wins, and only the unset `-2` sentinel is filled from the auto-picked render node.
///    H.264 output masks the dimensions even, because 4:2:0 needs even width and height.
/// 5. **Encode-path choice + render loop**: only a same-GPU GLES session encodes zero-copy on this
///    calloop thread, because the dmabuf and its EGL context are calloop-affine; every readback
///    flavor (OpenH264, striped x264/JPEG, Pixman, or a cross-GPU hardware encoder) builds its
///    encoders on the dedicated encode thread instead. A high-frequency timer composites the mapped
///    windows plus cursor and watermark, applies the shared paint-over / recovery-IDR policy, and
///    delivers the encoded stripes back to Python through the frame callback. The zero-copy encode
///    waits the GL render fence first, so a hardware encoder reading the dmabuf through CUDA/VA
///    never maps a half-rasterized (torn) frame.
/// 6. **Thread lifecycle**: the Python frame callback runs on a dedicated delivery thread so its
///    GIL never stalls calloop input / control dispatch, and in readback mode the encoders run on
///    the `wl-encode` thread. On a restart or stop the encode thread is torn down before the
///    delivery thread — it feeds the delivery sender and must be gone first — and the retained
///    callbacks are dropped and gated by a process-shutdown flag so nothing fires into a finalizing
///    interpreter.
fn run_wayland_thread(
    command_rx: smithay::reexports::calloop::channel::Channel<ThreadCommand>,
    command_tx: smithay::reexports::calloop::channel::Sender<ThreadCommand>,
    initial_width: i32,
    initial_height: i32,
    explicit_dri_node: String,
    auto_gpu_selected: bool,
    cursor_size: i32,
) {
    let width: i32 = if initial_width > 0 { initial_width } else { 1024 };
    let height: i32 = if initial_height > 0 { initial_height } else { 768 };

    let mut event_loop = EventLoop::<AppState>::try_new().expect("Unable to create event_loop");
    let display: Display<AppState> = Display::new().unwrap();
    let dh: DisplayHandle = display.handle();
    unsafe {
        if let Ok(lib) = libloading::Library::new("libwayland-server.so.0") {
            if let Ok(set_max) = lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void, usize)>(
                b"wl_display_set_default_max_buffer_size\0",
            ) {
                set_max(
                    dh.backend_handle().display_ptr() as *mut std::ffi::c_void,
                    10 * 1024 * 1024,
                );
            }
            std::mem::forget(lib);
        }
    }

    let dri_node = explicit_dri_node;

    let mut use_gpu = !dri_node.is_empty();
    let render_node_path = dri_node.clone();

    let mut gles_renderer = None;
    let mut pixman_renderer = None;
    let mut offscreen_buffer: Option<(BufferObject<()>, Dmabuf)> = None;
    let mut dmabuf_global = None;
    let mut gbm_device_raw = None;
    let mut dmabuf_state = DmabufState::new();

    let mut gpu_success = false;
    if use_gpu {
        println!("[Wayland] Initializing GL Renderer using device: {}", dri_node);
        let init_res: Result<(), String> = (|| {
            let device_path = std::path::Path::new(&dri_node);
            let file = File::options().read(true).write(true).open(device_path)
                .map_err(|e| format!("Failed to open render device: {}", e))?;
            let file_for_alloc = file.try_clone()
                .map_err(|e| format!("Failed to clone file for GBM Allocator: {}", e))?;
            let gbm_allocator = RawGbmDevice::new(file_for_alloc)
                .map_err(|_| "Failed to create Raw GBM Device")?;
            let gbm = GbmDevice::new(file)
                .map_err(|_| "Failed to create GBM device")?;
            let egl = unsafe { EGLDisplay::new(gbm) }
                .map_err(|_| "Failed to create EGL display")?;
            let context = EGLContext::new(&egl)
                .map_err(|_| "Failed to create EGL context")?;
            let mut renderer = unsafe { GlesRenderer::new(context) }
                .map_err(|_| "Failed to init GlesRenderer")?;
            
            if let Err(e) = renderer.bind_wl_display(&dh) {
                println!("[Wayland] Warning: Failed to bind EGL to Wayland Display (Optional): {:?}", e);
            }

            let formats = Bind::<Dmabuf>::supported_formats(&renderer)
                .ok_or("Failed to query formats")?
                .into_iter()
                .collect::<Vec<_>>();

            let node = DrmNode::from_path(device_path)
                .map_err(|_| "Failed to create DrmNode")?;
            let dmabuf_default_feedback = DmabufFeedbackBuilder::new(node.dev_id(), formats.clone()).build();

            dmabuf_global = Some(if let Ok(default_feedback) = dmabuf_default_feedback {
                dmabuf_state.create_global_with_default_feedback::<AppState>(&dh, &default_feedback)
            } else {
                dmabuf_state.create_global::<AppState>(&dh, formats)
            });

            let bo = gbm_allocator.create_buffer_object(
                width as u32, height as u32, GbmFormat::Argb8888, BufferObjectFlags::RENDERING
            ).map_err(|_| "Failed to allocate GBM buffer")?;

            let dmabuf = create_dmabuf_from_bo(&bo);
            offscreen_buffer = Some((bo, dmabuf));
            gbm_device_raw = Some(gbm_allocator);
            gles_renderer = Some(renderer);
            Ok(())
        })();

        match init_res {
            Ok(_) => gpu_success = true,
            Err(e) => {
                println!("[Wayland] GPU Initialization failed: {}. Falling back to Software Renderer (Pixman).", e);
                use_gpu = false;
            }
        }
    }

    if !gpu_success {
        if !dri_node.is_empty() && !use_gpu {
        } else {
            println!("[Wayland] No render node. Initializing Software Renderer (Pixman).");
        }
        pixman_renderer = Some(PixmanRenderer::new().expect("Failed to init PixmanRenderer"));
        use_gpu = false;
    }

    let compositor_state = CompositorState::new::<AppState>(&dh);
    let fractional_scale_state = FractionalScaleManagerState::new::<AppState>(&dh);
    let shm_state = ShmState::new::<AppState>(&dh, vec![]);
    let output_state = OutputManagerState::new_with_xdg_output::<AppState>(&dh);
    let mut seat_state = SeatState::new();
    let shell_state = XdgShellState::new::<AppState>(&dh);
    let space = Space::default();
    let layer_shell_state = WlrLayerShellState::new::<AppState>(&dh);
    let data_device_state = DataDeviceState::new::<AppState>(&dh);
    let data_control_state = DataControlState::new::<AppState, _>(&dh, None, |_| true);
    let virtual_keyboard_state = VirtualKeyboardManagerState::new::<AppState, _>(&dh, |_client| true);
    let pointer_warp_state = PointerWarpManager::new::<AppState>(&dh);
    let relative_pointer_state = RelativePointerManagerState::new::<AppState>(&dh);
    let pointer_constraints_state = PointerConstraintsState::new::<AppState>(&dh);

    let foreign_toplevel_list = ForeignToplevelListState::new::<AppState>(&dh);
    let xdg_decoration_state = XdgDecorationState::new::<AppState>(&dh);
    let single_pixel_buffer = SinglePixelBufferState::new::<AppState>(&dh);
    let viewporter_state = ViewporterState::new::<AppState>(&dh);
    let presentation_state = PresentationState::new::<AppState>(&dh, 1);
    let xdg_activation_state = XdgActivationState::new::<AppState>(&dh);
    let primary_selection_state = PrimarySelectionState::new::<AppState>(&dh);
    let popups = PopupManager::default();

    let mut seat = seat_state.new_wl_seat(&dh, "seat0");
    seat.add_keyboard(XkbConfig::default(), 200, 25)
        .expect("Failed to init keyboard");
    seat.add_pointer();

    let mut state = AppState {
        compositor_state,
        fractional_scale_state,
        viewporter_state,
        presentation_state,
        shm_state,
        single_pixel_buffer,
        dmabuf_state,
        dmabuf_global,
        output_state,
        seat_state,
        shell_state,
        layer_shell_state,
        space,
        data_device_state,
        data_control_state,
        dh: dh.clone(),
        seat,
        virtual_keyboard_state,
        pointer_warp_state,
        relative_pointer_state,
        pointer_constraints_state,
        outputs: Vec::new(),
        pending_windows: Vec::new(),
        foreign_toplevel_list,
        xdg_decoration_state,
        xdg_activation_state,
        primary_selection_state,
        popups,
        frame_buffer: vec![0u8; (width.max(0) as usize) * (height.max(0) as usize) * 4],
        gles_renderer,
        pixman_renderer,
        gbm_device: gbm_device_raw,
        offscreen_buffer,
        is_capturing: false,
        settings: RustCaptureSettings {
            width,
            height,
            ..RustCaptureSettings::default()
        },
        callback: None,
        cursor_callback: None,
        clipboard_callback: None,
        pending_clipboard_read: None,
        last_log_time: Instant::now(),
        start_time: Instant::now(),
        clock: Clock::new(),
        frame_counter: 0,
        pending_force_idr: false,
        needs_full_render: true,
        use_gpu,
        video_encoder: None,
        vaapi_state: StripeState::default(),
        cursor_helper: Cursor::load(cursor_size),
        overlay_state: OverlayState::default(),
        current_cursor_icon: None,
        cursor_buffer: None,
        cursor_cache: std::collections::HashMap::new(),
        render_cursor_on_framebuffer: false,
        render_node_path,
        auto_gpu_selected,
        recording_sink: None,
        deliver_tx: None,
        deliver_join: None,
        pending_hw_delivery: None,
        pending_hw_damage: false,
        encode_pool: None,
        encode_join: None,
        encode_controls: Arc::new(WlEncodeControls::new()),
        encode_stats: Arc::new(WlEncodeStats::new()),
        pool_last_render: Vec::new(),
        render_seq: 0,
        pool_content_gen: Vec::new(),
        content_gen: 0,
        pending_screenshot: None,
    };

    let output = Output::new(
        "HEADLESS-1".into(),
        PhysicalProperties {
            size: (width, height).into(),
            subpixel: Subpixel::Unknown,
            make: "Pixelflux".into(),
            model: "Virtual".into(),
            serial_number: "001".into(),
        },
    );
    output.change_current_state(
        Some(OutputMode {
            size: (width, height).into(),
            refresh: 60_000,
        }),
        Some(Transform::Normal),
        Some(OutputScale::Fractional(1.0)),
        Some((0, 0).into()),
    );
    output.set_preferred(OutputMode {
        size: (width, height).into(),
        refresh: 60_000,
    });
    state.space.map_output(&output, (0, 0));
    state.outputs.push(output.clone());
    let _global = output.create_global::<AppState>(&dh);
    let mut damage_tracker = OutputDamageTracker::from_output(&output);

    event_loop
        .handle()
        .insert_source(command_rx, move |event, _, state| {
            match event {
                CalloopEvent::Msg(ThreadCommand::StartCapture(cb, mut settings)) => {
                    if state.auto_gpu_selected && settings.encode_node_index < -1 {
                        if let Some(idx_str) = state.render_node_path.strip_prefix("/dev/dri/renderD") {
                            if let Ok(idx) = idx_str.parse::<i32>() {
                                settings.encode_node_index = idx - 128;
                            }
                        }
                    }

                    if settings.output_mode == 1 {
                        settings.width &= !1;
                        settings.height &= !1;
                    }

                    state.recording_sink =
                        crate::recording_sink::RecordingSink::try_bind(&settings.recording_socket);

                    if let Some(output) = state.outputs.first() {
                        let current_mode = output.current_mode().unwrap();
                        let current_w = current_mode.size.w;
                        let current_h = current_mode.size.h;
                        let current_scale = output.current_scale().fractional_scale();
                        let current_refresh = current_mode.refresh;
                        let target_refresh = (settings.target_fps * 1000.0).round() as i32;

                        let scale = settings.scale.max(0.1);
                        let logical_width = (settings.width as f64 / scale).round() as i32;
                        let logical_height = (settings.height as f64 / scale).round() as i32;

                        if current_w != settings.width
                            || current_h != settings.height
                            || (current_scale - settings.scale).abs() > 0.001
                            || current_refresh != target_refresh
                        {
                            // Allocate the GPU backing for the new dimensions BEFORE
                            // committing anything: if the driver refuses (VRAM
                            // exhaustion, dimensions it will not back), the whole
                            // reconfigure is skipped and the previous mode + buffers
                            // stay live. A failed resize must degrade to "no resize",
                            // never panic the compositor thread — that would drop
                            // every Wayland client with no recovery short of a
                            // process restart. (The INITIAL allocation path already
                            // degrades gracefully, to the software renderer.)
                            let mut new_offscreen = None;
                            let mut gbm_resize_failed = false;
                            if state.use_gpu {
                                if let Some(gbm) = state.gbm_device.as_mut() {
                                    match gbm.create_buffer_object(
                                        settings.width as u32,
                                        settings.height as u32,
                                        GbmFormat::Argb8888,
                                        BufferObjectFlags::RENDERING,
                                    ) {
                                        Ok(bo) => {
                                            let dmabuf = create_dmabuf_from_bo(&bo);
                                            new_offscreen = Some((bo, dmabuf));
                                        }
                                        Err(e) => {
                                            eprintln!(
                                                "[Wayland] GBM buffer resize to {}x{} failed ({:?}); keeping previous output mode.",
                                                settings.width, settings.height, e
                                            );
                                            gbm_resize_failed = true;
                                        }
                                    }
                                }
                            }
                            if gbm_resize_failed {
                                // The mode commit below is skipped wholesale, so the
                                // rest of this StartCapture (encoder setup, stored
                                // settings) must also see the dimensions that are
                                // actually live — otherwise the encoder and the
                                // still-old buffers disagree on frame geometry.
                                settings.width = current_w;
                                settings.height = current_h;
                                settings.scale = current_scale;
                                settings.target_fps = current_refresh as f64 / 1000.0;
                            } else {
                                println!(
                                    "[Wayland] Configuring Output: {}x{} @ {:.2} FPS (Scale {:.2})",
                                    settings.width, settings.height, settings.target_fps, settings.scale
                                );
                                let new_mode = OutputMode {
                                    size: (settings.width, settings.height).into(),
                                    refresh: target_refresh,
                                };
                                output.change_current_state(
                                    Some(new_mode),
                                    Some(Transform::Normal),
                                    Some(OutputScale::Fractional(settings.scale)),
                                    Some((0, 0).into()),
                                );
                                output.set_preferred(new_mode);

                                let pixel_count =
                                    (settings.width.max(0) as usize) * (settings.height.max(0) as usize);
                                state.frame_buffer = vec![0u8; pixel_count * 4];

                                if let Some(off) = new_offscreen.take() {
                                    state.offscreen_buffer = Some(off);
                                }
                            }
                        }

                        for window in state.space.elements() {
                            if let Some(surface) = window.wl_surface() {
                                output.enter(&surface);
                                with_states(&surface, |states| {
                                    smithay::wayland::fractional_scale::with_fractional_scale(states, |fs| {
                                        fs.set_preferred_scale(scale);
                                    });
                                });
                            }
                            if let Some(toplevel) = window.toplevel() {
                                toplevel.with_pending_state(|state| {
                                    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel::State;
                                    state.states.set(State::Fullscreen);
                                    state.states.set(State::Activated);
                                    state.size = Some((logical_width, logical_height).into());
                                });
                                toplevel.send_configure();
                            }
                        }
                    }

                    let use_cpu_explicit = settings.use_cpu || settings.encode_node_index == -1;
                    let gpu_intent =
                        settings.output_mode == 1 && !settings.use_openh264 && !use_cpu_explicit;
                    if use_cpu_explicit && !(settings.output_mode == 1 && settings.use_openh264) {
                        println!("[Wayland] CPU encoding selected (use_cpu=true or encode_node_index=-1).");
                    }

                    let mut different_gpu = false;
                    if gpu_intent {
                        let encode_node_idx = settings.encode_node_index.max(0);
                        if !state.render_node_path.is_empty()
                            && !state.render_node_path.contains(&format!("renderD{}", 128 + encode_node_idx))
                        {
                            different_gpu = true;
                        }
                    }

                    if gpu_intent && state.use_gpu && !different_gpu {
                        let encode_driver = get_gpu_driver(settings.encode_node_index.max(0));
                        println!(
                            "[Wayland] Encode Node Index: {} | Driver: {}",
                            settings.encode_node_index.max(0), encode_driver
                        );

                        if encode_driver.contains("nvidia") {
                            let reused = match state.video_encoder.as_mut() {
                                Some(GpuEncoder::Nvenc(enc)) => {
                                    match enc.reconfigure_resolution(&settings) {
                                        Ok(()) => {
                                            println!("[Wayland] NVENC session reconfigured in place.");
                                            true
                                        }
                                        Err(e) => {
                                            eprintln!("[Wayland] NVENC in-place reconfigure unavailable ({e}); rebuilding.");
                                            false
                                        }
                                    }
                                }
                                _ => false,
                            };
                            if !reused {
                                state.video_encoder = None;
                                println!("[Wayland] Nvidia Encoder detected. Initializing NVENC...");
                                let egl_display = if let Some(renderer) = state.gles_renderer.as_ref() {
                                    renderer.egl_context().display().get_display_handle().handle
                                } else {
                                    std::ptr::null()
                                };

                                match NvencEncoder::new(&settings, egl_display) {
                                    Ok(encoder) => {
                                        state.video_encoder = Some(GpuEncoder::Nvenc(encoder));
                                        println!("[Wayland] NVENC Encoder initialized successfully.");
                                    }
                                    Err(e) => eprintln!(
                                        "[Wayland] Failed to init NVENC: {}. Falling back to CPU.",
                                        e
                                    ),
                                }
                            }
                        } else {
                            state.video_encoder = None;
                            println!("[Wayland] Initializing Unified VAAPI Encoder...");
                            if settings.video_fullcolor {
                                println!("[Wayland] 4:4:4 Fullcolor requested. VAAPI does not support this profile reliably. Falling back to CPU.");
                            } else {
                                match VaapiEncoder::new(&settings) {
                                    Ok(encoder) => {
                                        state.video_encoder = Some(GpuEncoder::Vaapi(encoder));
                                        println!(
                                            "[Wayland] VAAPI Encoder initialized successfully."
                                        );
                                    }
                                    Err(e) => eprintln!(
                                        "[Wayland] Failed to init VAAPI: {}. Falling back to CPU.",
                                        e
                                    ),
                                }
                            }
                        }
                    } else {
                        state.video_encoder = None;
                    }

                    if different_gpu {
                        println!("[Wayland] Decision: Rendering and Encoding GPUs differ -> Forcing Readback (CPU path for pixels).");
                    }
                    if state.video_encoder.is_none() {
                        println!("[Wayland] Decision: Readback path (encode thread) active.");
                    } else if !different_gpu {
                        println!("[Wayland] Decision: Zero-Copy path active.");
                    }

                    if state.recording_sink.is_some() && settings.output_mode == 0 {
                        eprintln!(
                            "[recording_sink] WARNING: recording_socket is set but output_mode is JPEG (0). \
                             The recording socket requires a single H.264 stream. Please set output_mode=1 \
                             on the Python CaptureSettings to produce a recordable output."
                        );
                    }

                    let watermark_output_scale = state
                        .outputs
                        .first()
                        .map(|o| o.current_scale().fractional_scale())
                        .unwrap_or(1.0);
                    state
                        .overlay_state
                        .load_watermark(&settings.watermark_path, watermark_output_scale);
                    state.callback = Some(cb);
                    state.is_capturing = true;
                    WAYLAND_CAPTURE_ALIVE.store(true, Ordering::Release);
                    state.render_cursor_on_framebuffer = settings.capture_cursor;
                    state.settings = settings.clone();
                    state.encode_stats.frames.store(0, Ordering::Relaxed);
                    state.encode_stats.stripes.store(0, Ordering::Relaxed);
                    state.last_log_time = Instant::now();
                    state.frame_counter = 0;
                    state.pending_force_idr = false;
                    state.vaapi_state = StripeState::default();
                    if state.cursor_callback.is_some() {
                        if let Some(icon) = state.current_cursor_icon.clone() {
                            state.send_cursor_image(&icon);
                        }
                    }
                    if let Some(p) = state.encode_pool.take() { p.shutdown(); }
                    let prior_readback_encoder = state
                        .encode_join
                        .take()
                        .and_then(|j| j.join().ok())
                        .flatten();
                    if let Some(tx) = state.deliver_tx.take() { drop(tx); }
                    if let Some(j) = state.deliver_join.take() { let _ = j.join(); }
                    state.pending_hw_delivery = None;
                    state.pending_hw_damage = false;
                    if let Some(cb) = state.callback.take() {
                        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<EncodedStripe>>(1);
                        let join = thread::spawn(move || {
                            crate::boost_thread_priority(-10);
                            while let Ok(stripes) = rx.recv() {
                                if PY_SHUTDOWN.load(Ordering::Relaxed) { continue; }
                                Python::attach(|py| {
                                    for s in stripes {
                                        match Py::new(py, StripeFrame::new_owned_meta(
                                            s.data, s.data_type, s.stripe_y_start,
                                            s.stripe_height, s.frame_id,
                                        )) {
                                            Ok(f) => { if let Err(e) = cb.call1(py, (f,)) { e.print(py); } }
                                            Err(e) => eprintln!("[wayland] frame alloc error: {e:?}"),
                                        }
                                    }
                                });
                            }
                        });
                        state.deliver_tx = Some(tx);
                        state.deliver_join = Some(join);
                    }

                    if state.video_encoder.is_none() {
                        if let Some(ref deliver_tx) = state.deliver_tx {
                            let pool = Arc::new(WlFramePool::new(
                                WL_POOL_SURFACES,
                                (settings.width.max(0) as usize) * (settings.height.max(0) as usize) * 4,
                            ));
                            state.pool_last_render = vec![0; WL_POOL_SURFACES];
                            state.render_seq = 0;
                            // u64::MAX marks every slot stale so each one is read back
                            // before its first publish, whatever the damage says.
                            state.pool_content_gen = vec![u64::MAX; WL_POOL_SURFACES];
                            state.content_gen = 0;
                            let c = &state.encode_controls;
                            c.bitrate_kbps.store(settings.video_bitrate_kbps, Ordering::Relaxed);
                            c.vbv_mult_milli.store(
                                (settings.video_vbv_multiplier * 1000.0).round() as i32,
                                Ordering::Relaxed,
                            );
                            c.fps_milli.store(
                                (settings.target_fps.max(1.0) * 1000.0) as u64,
                                Ordering::Relaxed,
                            );
                            c.rate_dirty.store(false, Ordering::Relaxed);
                            c.force_idr.store(false, Ordering::Relaxed);
                            // Also drop any leftover LiveTunables snapshot: UpdateTunables
                            // sets it even when no encode thread is consuming (zero-copy
                            // mode, or after StopCapture), and without this reset the new
                            // encode thread's first frame would apply that stale snapshot
                            // OVER the fresh StartCapture settings just established.
                            c.tunables_dirty.store(false, Ordering::Relaxed);
                            *c.tunables.lock().unwrap() = None;
                            let cfg = WlEncodeConfig {
                                settings: settings.clone(),
                                use_gpu: state.use_gpu,
                                try_gpu: gpu_intent && (!state.use_gpu || different_gpu),
                                prior: prior_readback_encoder,
                                recording_sink: state.recording_sink.clone(),
                                deliver_tx: deliver_tx.clone(),
                                controls: state.encode_controls.clone(),
                                stats: state.encode_stats.clone(),
                            };
                            let pool2 = pool.clone();
                            state.encode_join = Some(
                                thread::Builder::new()
                                    .name("wl-encode".into())
                                    .spawn(move || wayland_encode_loop(&pool2, cfg))
                                    .expect("failed to spawn wl-encode thread"),
                            );
                            state.encode_pool = Some(pool);
                        }
                    } else {
                        state.encode_stats.n_stripes.store(1, Ordering::Relaxed);
                        *state.encode_stats.desc.lock().unwrap() =
                            encoder_desc(&settings, state.video_encoder.as_ref(), false, true);
                        log_stream_settings(&settings, 1, state.video_encoder.as_ref(), false);
                    }
                    // Force the keyframe on the now-active path (as RequestIdr does),
                    // unconditionally: the damage tracker and offscreen buffer stay
                    // warm across StopCapture/StartCapture, so a restarted capture on
                    // a static screen otherwise produces no damage, no first frame,
                    // and no IDR in either the striped or the zero-copy path (a
                    // stop+start encoder switch froze until the next screen damage).
                    if state.encode_pool.is_some() {
                        state.encode_controls.force_idr.store(true, Ordering::Relaxed);
                    } else {
                        state.pending_force_idr = true;
                    }
                    state.needs_full_render = true;
                }
                CalloopEvent::Msg(ThreadCommand::StopCapture) => {
                    println!("[Wayland] Capture loop stopped.");
                    state.is_capturing = false;
                    WAYLAND_CAPTURE_ALIVE.store(false, Ordering::Release);
                    state.callback = None;
                    state.cursor_callback = None;
                    state.clipboard_callback = None;
                    state.video_encoder = None;
                    state.vaapi_state = StripeState::default();
                    if let Some(p) = state.encode_pool.take() { p.shutdown(); }
                    if let Some(j) = state.encode_join.take() { let _ = j.join(); }
                    state.recording_sink = None;
                    *state.encode_stats.desc.lock().unwrap() = String::new();
                    if let Some(tx) = state.deliver_tx.take() { drop(tx); }
                    if let Some(j) = state.deliver_join.take() { let _ = j.join(); }
                    state.pending_hw_delivery = None;
                    state.pending_hw_damage = false;
                }
                CalloopEvent::Msg(ThreadCommand::SetClipboardCallback(cb)) => {
                    state.clipboard_callback = Some(cb);
                }
                CalloopEvent::Msg(ThreadCommand::SetClipboard { mime, data }) => {
                    let mimes: Vec<String> = if mime.starts_with("text/plain") {
                        ["text/plain;charset=utf-8", "UTF8_STRING", "text/plain",
                         "STRING", "TEXT"].iter().map(|s| s.to_string()).collect()
                    } else {
                        vec![mime.clone()]
                    };
                    smithay::wayland::selection::data_device::set_data_device_selection(
                        &state.dh,
                        &state.seat.clone(),
                        mimes,
                        std::sync::Arc::new((mime, data)),
                    );
                }
                CalloopEvent::Msg(ThreadCommand::SetCursorCallback(cb)) => {
                    state.cursor_callback = Some(cb);
                    if let Some(icon) = state.current_cursor_icon.clone() {
                        state.send_cursor_image(&icon);
                    }
                }
                CalloopEvent::Msg(ThreadCommand::KeyboardKey { scancode, state: key_state_val }) => {
                    let key_state = if key_state_val > 0 { KeyState::Pressed } else { KeyState::Released };
                    let serial = next_serial();
                    let time = wayland_time();
                    if let Some(keyboard) = state.seat.get_keyboard() {
                        keyboard.input(state, Keycode::new(scancode), key_state, serial, time, |_, _, _| {
                            FilterResult::<()>::Forward
                        });
                    }
                }
                CalloopEvent::Msg(ThreadCommand::SetKeymapString(text)) => {
                    if let Some(keyboard) = state.seat.get_keyboard() {
                        if let Err(e) = keyboard.set_keymap_from_string(state, text) {
                            eprintln!("[Wayland] keymap swap failed: {e:?}");
                        }
                    }
                }
                CalloopEvent::Msg(ThreadCommand::GetXkbKeymap { reply }) => {
                    let mut keymap_str = String::new();
                    if let Some(keyboard) = state.seat.get_keyboard() {
                        keymap_str = keyboard.with_xkb_state(state, |context| {
                            match context.xkb().lock() {
                                Ok(guard) => {
                                    let keymap = unsafe { guard.keymap() };
                                    keymap.get_as_string(
                                        smithay::input::keyboard::xkb::KEYMAP_FORMAT_TEXT_V1,
                                    )
                                }
                                Err(_) => String::new(),
                            }
                        });
                    }
                    let _ = reply.send(keymap_str);
                }
                CalloopEvent::Msg(ThreadCommand::PointerMotion { x, y }) => {
                    let serial = next_serial();
                    let time = wayland_time();
                    let scale = state.settings.scale;
                    let logical_w = (state.settings.width as f64 / scale).floor();
                    let logical_h = (state.settings.height as f64 / scale).floor();
                    let logical_x = (x / scale).max(0.0).min(logical_w - 1.0);
                    let logical_y = (y / scale).max(0.0).min(logical_h - 1.0);

                    if let Some(pointer) = state.seat.get_pointer() {
                        let p = Point::<f64, smithay::utils::Logical>::from((logical_x, logical_y));
                        let mut under = None;

                        if let Some(output) = state.outputs.first() {
                            let layer_map = layer_map_for_output(output);
                            let pointer_pos = p.to_i32_round();

                            for layer in layer_map.layers().rev() {
                                let state_layer = layer.layer();
                                if state_layer == smithay::wayland::shell::wlr_layer::Layer::Overlay || state_layer == smithay::wayland::shell::wlr_layer::Layer::Top {
                                    if let Some(bbox) = layer_map.layer_geometry(layer) {
                                        if bbox.contains(pointer_pos) {
                                            under = Some((FocusTarget::LayerSurface(layer.clone()), bbox.loc.to_f64()));
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        if under.is_none() {
                            under = state.space.element_under(p).map(|(window, loc)| {
                                (FocusTarget::Window(window.clone()), loc.to_f64())
                            });
                        }

                        if under.is_none() {
                            if let Some(output) = state.outputs.first() {
                                let layer_map = layer_map_for_output(output);
                                let pointer_pos = p.to_i32_round();

                                for layer in layer_map.layers().rev() {
                                    let state_layer = layer.layer();
                                    if state_layer == smithay::wayland::shell::wlr_layer::Layer::Bottom || state_layer == smithay::wayland::shell::wlr_layer::Layer::Background {
                                        if let Some(bbox) = layer_map.layer_geometry(layer) {
                                            if bbox.contains(pointer_pos) {
                                                under = Some((FocusTarget::LayerSurface(layer.clone()), bbox.loc.to_f64()));
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        pointer.motion(state, under, &MotionEvent { location: p, serial, time });
                        pointer.frame(state);
                    }
                }
                CalloopEvent::Msg(ThreadCommand::PointerRelativeMotion { dx, dy }) => {
                    let utime = wayland_utime();
                    let time = wayland_time();
                    let serial = next_serial();

                    if let Some(pointer) = state.seat.get_pointer() {
                        let current_pos = pointer.current_location();
                        
                        let scale = state.settings.scale;
                        let max_w = (state.settings.width as f64 / scale).floor() - 1.0;
                        let max_h = (state.settings.height as f64 / scale).floor() - 1.0;

                        let new_x = (current_pos.x + dx).max(0.0).min(max_w);
                        let new_y = (current_pos.y + dy).max(0.0).min(max_h);
                        
                        let new_pos = Point::<f64, smithay::utils::Logical>::from((new_x, new_y));

                        let under = state.space.element_under(new_pos).map(|(window, loc)| {
                            (FocusTarget::Window(window.clone()), loc.to_f64())
                        });

                        pointer.motion(
                            state, 
                            under.clone(), 
                            &MotionEvent { 
                                location: new_pos, 
                                serial, 
                                time 
                            }
                        );

                        let event = RelativeMotionEvent {
                            utime,
                            delta: (dx, dy).into(),
                            delta_unaccel: (dx, dy).into(),
                        };
                        pointer.relative_motion(state, under, &event);

                        pointer.frame(state);
                    }
                }
                CalloopEvent::Msg(ThreadCommand::PointerButton { btn, state: btn_state_val }) => {
                    let serial = next_serial();
                    let time = wayland_time();
                    let button_state = if btn_state_val > 0 { smithay::backend::input::ButtonState::Pressed } else { smithay::backend::input::ButtonState::Released };

                    if let Some(pointer) = state.seat.get_pointer() {
                        if button_state == smithay::backend::input::ButtonState::Pressed {
                            let pos = pointer.current_location();
                            let target_window = state.space.element_under(pos).map(|(w, _)| w.clone());

                            if let Some(window) = target_window {
                                state.space.raise_element(&window, true);
                                if let Some(keyboard) = state.seat.get_keyboard() {
                                    keyboard.set_focus(state, Some(FocusTarget::Window(window)), serial);
                                }
                            }
                        }
                        let button = btn;
                        pointer.button(state, &ButtonEvent { button, state: button_state, serial, time });
                        pointer.frame(state);
                    }
                }
                CalloopEvent::Msg(ThreadCommand::PointerAxis { x, y }) => {
                    let time = wayland_time();
                    
                    if let Some(pointer) = state.seat.get_pointer() {
                        const V120_MULTIPLIER: f64 = 12.0; 

                        let mut frame = AxisFrame::new(time).source(AxisSource::Wheel);

                        if x != 0.0 { 
                            frame = frame
                                .value(Axis::Horizontal, x)
                                .v120(Axis::Horizontal, (x * V120_MULTIPLIER) as i32);
                        }
                        
                        if y != 0.0 { 
                            frame = frame
                                .value(Axis::Vertical, y)
                                .v120(Axis::Vertical, (y * V120_MULTIPLIER) as i32);
                        }

                        if x != 0.0 || y != 0.0 {
                            pointer.axis(state, frame);
                            pointer.frame(state);
                        }
                    }
                }
                CalloopEvent::Msg(ThreadCommand::UpdateCursorConfig { render_on_framebuffer }) => {
                    state.render_cursor_on_framebuffer = render_on_framebuffer;
                }
                CalloopEvent::Msg(ThreadCommand::RequestIdr) => {
                    if state.encode_pool.is_some() {
                        state.encode_controls.force_idr.store(true, Ordering::Relaxed);
                    } else {
                        state.pending_force_idr = true;
                    }
                    state.needs_full_render = true;
                }
                CalloopEvent::Msg(ThreadCommand::UpdateRate { bitrate_kbps, vbv_multiplier, fps }) => {
                    if let Some(b) = bitrate_kbps { state.settings.video_bitrate_kbps = b; }
                    if let Some(v) = vbv_multiplier { state.settings.video_vbv_multiplier = v; }
                    if let Some(f) = fps { if f > 0.0 { state.settings.target_fps = f; } }
                    if let Some(GpuEncoder::Nvenc(enc)) = state.video_encoder.as_mut() {
                        enc.reconfigure_rate(&state.settings);
                    }
                    if let Some(GpuEncoder::Vaapi(enc)) = state.video_encoder.as_mut() {
                        enc.reconfigure_rate(&state.settings);
                    }
                    let c = &state.encode_controls;
                    c.bitrate_kbps.store(state.settings.video_bitrate_kbps, Ordering::Relaxed);
                    c.vbv_mult_milli.store(
                        (state.settings.video_vbv_multiplier * 1000.0).round() as i32,
                        Ordering::Relaxed,
                    );
                    c.fps_milli.store(
                        (state.settings.target_fps.max(1.0) * 1000.0) as u64,
                        Ordering::Relaxed,
                    );
                    c.rate_dirty.store(true, Ordering::Release);
                }
                CalloopEvent::Msg(ThreadCommand::UpdateTunables(t)) => {
                    t.apply_to(&mut state.settings);
                    state.render_cursor_on_framebuffer = t.capture_cursor;
                    *state.encode_controls.tunables.lock().unwrap() = Some(t);
                    state.encode_controls.tunables_dirty.store(true, Ordering::Release);
                }
                CalloopEvent::Msg(ThreadCommand::CuScreenshot { resp }) => {
                    state.pending_screenshot = Some(resp);
                }
                CalloopEvent::Msg(ThreadCommand::CuCursorPosition { resp }) => {
                    let pos = state.seat.get_pointer()
                        .map(|p| p.current_location())
                        .unwrap_or_else(|| (0.0f64, 0.0f64).into());
                    let scale = state.settings.scale;
                    let fb_x = pos.x * scale;
                    let fb_y = pos.y * scale;
                    let _ = resp.send((fb_x, fb_y));
                }
                CalloopEvent::Msg(ThreadCommand::CuGetInfo { resp }) => {
                    let _ = resp.send((
                        state.settings.width,
                        state.settings.height,
                        state.settings.scale,
                    ));
                }
                CalloopEvent::Closed => {}
            }
        })
        .unwrap();

    let source = ListeningSocketSource::new_auto().unwrap();
    let socket_name = source.socket_name().to_string_lossy().into_owned();
    println!("[Wayland] Socket listening on: {:?}", socket_name);
    std::env::set_var("WAYLAND_DISPLAY", &socket_name);

    event_loop
        .handle()
        .insert_source(source, |client_stream, _, state| {
            if let Err(err) = state
                .dh
                .insert_client(client_stream, Arc::new(ClientState::default()))
            {
                eprintln!("Error adding wayland client: {:?}", err);
            }
        })
        .expect("Failed to init wayland socket source");

    let timer = Timer::immediate();
    let mut is_memory_throttling = false;
    event_loop
        .handle()
        .insert_source(timer, move |_, _, state| {
            let loop_start_time = Instant::now();
            state.space.refresh();

            let current_rss = get_process_rss_bytes();
            let shm_usage = get_shm_usage_bytes();
            let memory_threshold = calculate_memory_threshold(state.settings.width, state.settings.height);

            if current_rss > memory_threshold || shm_usage > (4 * 1024 * 1024 * 1024) {
                if !is_memory_throttling {
                    is_memory_throttling = true;
                }
            } else if is_memory_throttling
                && current_rss < (memory_threshold as f64 * 0.75) as usize && shm_usage < (3 * 1024 * 1024 * 1024) {
                    is_memory_throttling = false;
                }

            let now = Instant::now();
            let elapsed = now.duration_since(state.last_log_time).as_secs_f64();
            if elapsed >= 1.0 {
                let frames = state.encode_stats.frames.swap(0, Ordering::Relaxed);
                let stripes = state.encode_stats.stripes.swap(0, Ordering::Relaxed);
                if state.is_capturing && state.settings.debug_logging {
                    let actual_fps = frames as f64 / elapsed;
                    let stripes_per_sec = stripes as f64 / elapsed;
                    let mode_str = state.encode_stats.desc.lock().unwrap().clone();
                    let n_stripes = state.encode_stats.n_stripes.load(Ordering::Relaxed);

                    let rss_mb = current_rss / 1024 / 1024;
                    let shm_mb = shm_usage / 1024 / 1024;
                    let throttle_warn = if is_memory_throttling { " [THROTTLED]" } else { "" };

                    println!("Res: {}x{} Mode: {} Stripes: {} EncFPS: {:.2} EncStripes/s: {:.2} Mem: {}MB SHM: {}MB{}",
                        state.settings.width, state.settings.height, mode_str, n_stripes, actual_fps, stripes_per_sec, rss_mb, shm_mb, throttle_warn);
                }
                state.last_log_time = now;
            }

            if !state.is_capturing && state.pending_screenshot.is_none() {
                // While capture is stopped the render tick early-outs
                // without finishing a Frame, so Smithay never drains its
                // deferred texture-destruction queue: GPU buffers imported
                // from nested clients that keep rendering accumulate and are
                // only released on the next start_capture(). Drain the cache
                // here so an idle session does not grow unbounded. On an
                // integrated GPU these buffers are system RAM, so the leak
                // surfaces as RSS growth and eventual OOM.
                // Upstream: linuxserver/pixelflux#26, smithay#1747.
                if let Some(renderer) = state.gles_renderer.as_mut() {
                    let _ = renderer.cleanup_texture_cache();
                }
                return TimeoutAction::ToDuration(Duration::from_millis(16));
            }

            let requested_idr = state.pending_force_idr;

            let mut pool_slot: Option<(usize, Vec<u8>)> = None;
            if let Some(ref pool) = state.encode_pool {
                if !is_memory_throttling {
                    pool_slot = pool.try_begin();
                    if pool_slot.is_none() {
                        return TimeoutAction::ToDuration(Duration::from_millis(1));
                    }
                }
            }

            state.overlay_state.update_position(
                state.settings.width,
                state.settings.height,
                state.settings.watermark_location_enum,
            );

            let mut render_success = false;
            let mut render_sync = None;
            let mut damage_rects: Vec<Rectangle<i32, Physical>> = Vec::new();
            let width = state.settings.width;
            let height = state.settings.height;

            if let Some(output) = state.outputs.first().cloned() {
                let render_age = if state.overlay_state.is_animated() || state.needs_full_render { 0 } else { 1 };
                if state.use_gpu {
                    if let Some(renderer) = state.gles_renderer.as_mut() {
                        if let Some((_bo, dmabuf)) = state.offscreen_buffer.as_mut() {
                            match renderer.bind(dmabuf) {
                                Ok(mut frame) => {
                                    let mut elements: Vec<CompositionElements<GlesRenderer, WaylandSurfaceRenderElement<GlesRenderer>>> = Vec::new();

                                    if state.render_cursor_on_framebuffer {
                                        if let Some(pointer) = state.seat.get_pointer() {
                                            let pos = pointer.current_location();
                                            let output_scale_val = output.current_scale().fractional_scale();
                                            let scale = Scale::from(output_scale_val);

                                            if let Some(CursorImageStatus::Named(icon)) = &state.current_cursor_icon {
                                                let name = wayland::frontend::cursor_icon_to_str(icon);
                                                let time = Duration::from_millis(state.clock.now().as_millis() as u64);
                                                if let Some(image) = state.cursor_helper.get_image_by_name(name, output_scale_val.round() as u32, time) {
                                                    if let Some(elem) = state.overlay_state.get_cursor_element(renderer, image, pos.to_i32_round()) {
                                                        elements.push(CompositionElements::Cursor(elem));
                                                    }
                                                }
                                            } else if let Some(CursorImageStatus::Surface(surface)) = &state.current_cursor_icon {
                                                 let phys_pos = pos.to_physical(scale);
                                                 let elem_result = with_states(surface, |states| {
                                                     WaylandSurfaceRenderElement::from_surface(renderer, surface, states, phys_pos, 1.0, smithay::backend::renderer::element::Kind::Cursor)
                                                 });
                                                 if let Ok(Some(cursor_elem)) = elem_result {
                                                     elements.push(CompositionElements::Surface(cursor_elem));
                                                 }
                                            } else if state.current_cursor_icon.is_none() {
                                                let time = Duration::from_millis(state.clock.now().as_millis() as u64);
                                                let image = state.cursor_helper.get_image(output_scale_val.round() as u32, time);
                                                if let Some(elem) = state.overlay_state.get_cursor_element(renderer, image, pos.to_i32_round()) {
                                                    elements.push(CompositionElements::Cursor(elem));
                                                }
                                            }
                                        }
                                    }

                                    if let Some(elem) = state.overlay_state.get_watermark_element(renderer) {
                                        elements.push(CompositionElements::Cursor(elem));
                                    }

                                    {
                                        let layer_map = layer_map_for_output(&output);

                                        let draw_layer = |renderer: &mut GlesRenderer, elements: &mut Vec<CompositionElements<GlesRenderer, WaylandSurfaceRenderElement<GlesRenderer>>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
                                            for surface in layer_map.layers().rev() {
                                                let current_layer = surface.layer();
                                                if current_layer == target_layer {
                                                    if let Some(geo) = layer_map.layer_geometry(surface) {
                                                        let elem = smithay::wayland::compositor::with_states(surface.wl_surface(), |states| {
                                                            WaylandSurfaceRenderElement::from_surface(
                                                                renderer, surface.wl_surface(), states,
                                                                geo.loc.to_physical_precise_round(1), 1.0,
                                                                smithay::backend::renderer::element::Kind::Unspecified
                                                            )
                                                        });
                                                        if let Ok(Some(e)) = elem {
                                                            elements.push(CompositionElements::Surface(e));
                                                        }
                                                    }
                                                }
                                            }
                                        };

                                        draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Overlay);
                                        draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Top);
                                    }

                                    for window in state.space.elements_for_output(&output).collect::<Vec<_>>().into_iter().rev() {
                                        let window_loc = state.space.element_location(window).unwrap_or_default();

                                        if let Some(surface) = window.wl_surface() {
                                            let popups = PopupManager::popups_for_surface(&surface);
                                            for (popup, location) in popups {
                                                let popup_surface = popup.wl_surface();
                                                let popup_pos = window_loc + location;
                                                let elem = smithay::wayland::compositor::with_states(popup_surface, |states| {
                                                    WaylandSurfaceRenderElement::from_surface(
                                                        renderer,
                                                        popup_surface,
                                                        states,
                                                        popup_pos.to_physical_precise_round(1),
                                                        1.0,
                                                        smithay::backend::renderer::element::Kind::Unspecified
                                                    )
                                                });
                                                if let Ok(Some(e)) = elem {
                                                    elements.push(CompositionElements::Surface(e));
                                                }
                                            }
                                        }

                                        elements.extend(window.render_elements(renderer, window_loc.to_physical_precise_round(1), Scale::from(1.0), 1.0).into_iter().map(CompositionElements::Space));
                                    }

                                    {
                                        let layer_map = layer_map_for_output(&output);

                                        let draw_layer = |renderer: &mut GlesRenderer, elements: &mut Vec<CompositionElements<GlesRenderer, WaylandSurfaceRenderElement<GlesRenderer>>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
                                            for surface in layer_map.layers() {
                                                let current_layer = surface.layer();
                                                if current_layer == target_layer {
                                                    if let Some(geo) = layer_map.layer_geometry(surface) {
                                                        let elem = smithay::wayland::compositor::with_states(surface.wl_surface(), |states| {
                                                            WaylandSurfaceRenderElement::from_surface(
                                                                renderer, surface.wl_surface(), states,
                                                                geo.loc.to_physical_precise_round(1), 1.0,
                                                                smithay::backend::renderer::element::Kind::Unspecified
                                                            )
                                                        });
                                                        if let Ok(Some(e)) = elem {
                                                            elements.push(CompositionElements::Surface(e));
                                                        }
                                                    }
                                                }
                                            }
                                        };

                                        draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Bottom);
                                        draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Background);
                                    }
                                    match damage_tracker.render_output(renderer, &mut frame, render_age, &elements, [0.1, 0.1, 0.1, 1.0]) {
                                        Ok(result) => {
                                            render_success = true;
                                            if let Some(damage) = result.damage {
                                                damage_rects = damage.clone();
                                            }
                                            render_sync = Some(result.sync);
                                            state.needs_full_render = false;
                                        },
                                        Err(e) => eprintln!("Render error: {:?}", e)
                                    }
                                    if !damage_rects.is_empty() {
                                        state.content_gen += 1;
                                    }
                                    if let Some((id, ref mut buf)) = pool_slot {
                                        // No-damage ticks skip the readback, so a pooled
                                        // buffer can lag the offscreen target whenever the
                                        // encoder held the other slot across a tick. Publishing
                                        // a lagging buffer would let a paint-over / burst /
                                        // recovery send re-encode pre-damage pixels; one
                                        // catch-up readback keeps every published buffer
                                        // current while idle readbacks stay skipped.
                                        if render_success && state.pool_content_gen[id] != state.content_gen {
                                            let _ = renderer.with_context(|gl| unsafe {
                                                gl.ReadPixels(
                                                    0,
                                                    0,
                                                    width,
                                                    height,
                                                    smithay::backend::renderer::gles::ffi::RGBA,
                                                    smithay::backend::renderer::gles::ffi::UNSIGNED_BYTE,
                                                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                                                );
                                            });
                                            state.pool_content_gen[id] = state.content_gen;
                                        }
                                    } else if state.pending_screenshot.is_some() {
                                        let _ = renderer.with_context(|gl| unsafe {
                                            gl.ReadPixels(
                                                0,
                                                0,
                                                width,
                                                height,
                                                smithay::backend::renderer::gles::ffi::RGBA,
                                                smithay::backend::renderer::gles::ffi::UNSIGNED_BYTE,
                                                state.frame_buffer.as_mut_ptr() as *mut std::ffi::c_void,
                                            );
                                        });
                                    }
                                },
                                Err(e) => eprintln!("Failed to bind buffer: {:?}", e)
                            }
                        }
                    }
                } else {
                    if let Some(renderer) = state.pixman_renderer.as_mut() {
                        let (ptr, buf_age) = match pool_slot {
                            Some((id, ref mut buf)) => {
                                let age = if state.pool_last_render[id] == 0 {
                                    0
                                } else {
                                    (state.render_seq + 1 - state.pool_last_render[id]) as usize
                                };
                                (buf.as_mut_ptr() as *mut u32, age)
                            }
                            None => (state.frame_buffer.as_mut_ptr() as *mut u32, 0),
                        };
                        let mut image = unsafe {
                            pixman::Image::from_raw_mut(pixman::FormatCode::A8R8G8B8, width as usize, height as usize, ptr, (width as usize) * 4, false).expect("Failed to create pixman image")
                        };
                                    match renderer.bind(&mut image) {
                                    Ok(mut frame) => {
                                        let mut elements: Vec<CompositionElements<PixmanRenderer, WaylandSurfaceRenderElement<PixmanRenderer>>> = Vec::new();

                                        if state.render_cursor_on_framebuffer {
                                            if let Some(pointer) = state.seat.get_pointer() {
                                                let pos = pointer.current_location();
                                                let output_scale_val = output.current_scale().fractional_scale();
                                                let scale = Scale::from(output_scale_val);

                                                if let Some(CursorImageStatus::Named(icon)) = &state.current_cursor_icon {
                                                    let name = wayland::frontend::cursor_icon_to_str(icon);
                                                    let time = Duration::from_millis(state.clock.now().as_millis() as u64);
                                                    if let Some(image) = state.cursor_helper.get_image_by_name(name, output_scale_val.round() as u32, time) {
                                                        if let Some(elem) = state.overlay_state.get_cursor_element(renderer, image, pos.to_i32_round()) {
                                                            elements.push(CompositionElements::Cursor(elem));
                                                        }
                                                    }
                                                } else if let Some(CursorImageStatus::Surface(surface)) = &state.current_cursor_icon {
                                                     let phys_pos = pos.to_physical(scale);
                                                     let elem_result = with_states(surface, |states| {
                                                         WaylandSurfaceRenderElement::from_surface(renderer, surface, states, phys_pos, 1.0, smithay::backend::renderer::element::Kind::Cursor)
                                                     });
                                                     if let Ok(Some(cursor_elem)) = elem_result {
                                                         elements.push(CompositionElements::Surface(cursor_elem));
                                                     }
                                                } else if state.current_cursor_icon.is_none() {
                                                    let time = Duration::from_millis(state.clock.now().as_millis() as u64);
                                                    let image = state.cursor_helper.get_image(output_scale_val.round() as u32, time);
                                                    if let Some(elem) = state.overlay_state.get_cursor_element(renderer, image, pos.to_i32_round()) {
                                                        elements.push(CompositionElements::Cursor(elem));
                                                    }
                                                }
                                            }
                                        }

                                        if let Some(elem) = state.overlay_state.get_watermark_element(renderer) {
                                            elements.push(CompositionElements::Cursor(elem));
                                        }

                                        {
                                            let layer_map = layer_map_for_output(&output);

                                            let draw_layer = |renderer: &mut PixmanRenderer, elements: &mut Vec<CompositionElements<PixmanRenderer, WaylandSurfaceRenderElement<PixmanRenderer>>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
                                                for surface in layer_map.layers() {
                                                    let current_layer = surface.layer();
                                                    if current_layer == target_layer {
                                                        if let Some(geo) = layer_map.layer_geometry(surface) {
                                                            let elem = smithay::wayland::compositor::with_states(surface.wl_surface(), |states| {
                                                                WaylandSurfaceRenderElement::from_surface(
                                                                    renderer, surface.wl_surface(), states,
                                                                    geo.loc.to_physical_precise_round(1), 1.0,
                                                                    smithay::backend::renderer::element::Kind::Unspecified
                                                                )
                                                            });
                                                            if let Ok(Some(e)) = elem {
                                                                elements.push(CompositionElements::Surface(e));
                                                            }
                                                        }
                                                    }
                                                }
                                            };

                                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Overlay);
                                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Top);
                                        }

                                        for window in state.space.elements_for_output(&output).collect::<Vec<_>>().into_iter().rev() {
                                            let loc = state.space.element_location(window).unwrap_or_default();
                                            
                                            if let Some(surface) = window.wl_surface() {
                                                let popups = PopupManager::popups_for_surface(&surface);
                                                for (popup, location) in popups {
                                                    let popup_surface = popup.wl_surface(); {
                                                        let popup_pos = loc + location;
                                                        let elem = smithay::wayland::compositor::with_states(popup_surface, |states| {
                                                            WaylandSurfaceRenderElement::from_surface(
                                                                renderer,
                                                                popup_surface,
                                                                states,
                                                                popup_pos.to_physical_precise_round(1),
                                                                1.0,
                                                                smithay::backend::renderer::element::Kind::Unspecified
                                                            )
                                                        });
                                                        if let Ok(Some(e)) = elem {
                                                            elements.push(CompositionElements::Surface(e));
                                                        }
                                                    }
                                                }
                                            }

                                            elements.extend(window.render_elements(renderer, loc.to_physical_precise_round(1), Scale::from(1.0), 1.0).into_iter().map(CompositionElements::Space));
                                        }

                                        {
                                            let layer_map = layer_map_for_output(&output);

                                            let draw_layer = |renderer: &mut PixmanRenderer, elements: &mut Vec<CompositionElements<PixmanRenderer, WaylandSurfaceRenderElement<PixmanRenderer>>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
                                                for surface in layer_map.layers() {
                                                    let current_layer = surface.layer();
                                                    if current_layer == target_layer {
                                                        if let Some(geo) = layer_map.layer_geometry(surface) {
                                                            let elem = smithay::wayland::compositor::with_states(surface.wl_surface(), |states| {
                                                                WaylandSurfaceRenderElement::from_surface(
                                                                    renderer, surface.wl_surface(), states,
                                                                    geo.loc.to_physical_precise_round(1), 1.0,
                                                                    smithay::backend::renderer::element::Kind::Unspecified
                                                                )
                                                            });
                                                            if let Ok(Some(e)) = elem {
                                                                elements.push(CompositionElements::Surface(e));
                                                            }
                                                        }
                                                    }
                                                }
                                            };

                                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Bottom);
                                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Background);
                                        }

                                let render_age = if state.overlay_state.is_animated() || state.needs_full_render { 0 } else { buf_age };
                                match damage_tracker.render_output(renderer, &mut frame, render_age, &elements, [0.1, 0.1, 0.1, 1.0]) {
                                    Ok(result) => {
                                        render_success = true;
                                        state.needs_full_render = false;
                                        if let Some(damage) = result.damage { damage_rects = damage.clone(); }
                                    },
                                    Err(e) => eprintln!("Render error: {:?}", e)
                                }
                                state.render_seq += 1;
                                if render_success {
                                    if let Some((id, _)) = pool_slot {
                                        state.pool_last_render[id] = state.render_seq;
                                    }
                                }
                            },
                            Err(e) => eprintln!("Failed to bind pixman image: {:?}", e)
                        }
                    }
                }

                if render_success {
                    let time = state.clock.now();
                    for window in state.space.elements() {
                        window.send_frame(&output, time, Some(Duration::ZERO), |_, _| Some(output.clone()));
                    }

                    if is_memory_throttling {
                        if state.encode_pool.is_none() {
                            state.frame_counter = state.frame_counter.wrapping_add(1);
                        }
                    } else if state.encode_pool.is_some() {
                        if state.pending_screenshot.is_some() {
                            if let Some((_, ref buf)) = pool_slot {
                                let n = buf.len().min(state.frame_buffer.len());
                                state.frame_buffer[..n].copy_from_slice(&buf[..n]);
                            }
                        }
                        if let Some((id, buf)) = pool_slot.take() {
                            let is_animated = state.overlay_state.is_animated();
                            state.encode_pool.as_ref().unwrap().publish(WlFrame {
                                id,
                                buf,
                                frame_id: state.frame_counter,
                                damage: std::mem::take(&mut damage_rects),
                                is_animated,
                            });
                            state.frame_counter = state.frame_counter.wrapping_add(1);
                        }
                    } else if let Some(ref mut encoder) = state.video_encoder {
                        // Deliver the parked frame (if any) first, WITHOUT blocking:
                        // this runs on the calloop thread, and a blocking send would
                        // freeze input/command/Wayland dispatch for as long as the
                        // Python consumer stalls (the readback path offloads its
                        // sends to the wl-encode thread for exactly this reason).
                        // While a frame stays parked, no new frame is encoded — an
                        // encoded frame joins the H.264 reference chain and can
                        // never be dropped — and the tick's damage is latched so
                        // the pause never loses a change.
                        let slot_free = match state.pending_hw_delivery.take() {
                            None => true,
                            Some(pending) => match state.deliver_tx.as_ref() {
                                None => true,
                                Some(tx) => match tx.try_send(pending) {
                                    Ok(()) => true,
                                    Err(std::sync::mpsc::TrySendError::Full(p)) => {
                                        state.pending_hw_delivery = Some(p);
                                        false
                                    }
                                    Err(std::sync::mpsc::TrySendError::Disconnected(_)) => true,
                                },
                            },
                        };
                        if !slot_free {
                            if !damage_rects.is_empty() {
                                state.pending_hw_damage = true;
                            }
                        } else {
                        let is_animated = state.overlay_state.is_animated();
                        let had_damage = !damage_rects.is_empty()
                            || std::mem::take(&mut state.pending_hw_damage);
                        let decision = crate::pipeline::decide_hw_fullframe(
                            &mut state.vaapi_state,
                            &state.settings,
                            state.frame_counter,
                            had_damage,
                            is_animated,
                            requested_idr,
                        );
                        let send_frame = decision.send;
                        let force_idr = decision.force_idr;
                        let target_qp = decision.target_qp;

                        if send_frame {
                            if let Some(sync) = render_sync.take() {
                                let _ = sync.wait();
                            }
                            let force_idr_for_recording = state
                                .recording_sink
                                .as_ref()
                                .map(|s| s.should_force_idr())
                                .unwrap_or(false);
                            let force_idr = force_idr || force_idr_for_recording;
                            let result = match encoder {
                                GpuEncoder::Nvenc(enc) => {
                                    if let Some((_, ref dmabuf)) = state.offscreen_buffer {
                                        enc.encode(dmabuf, state.frame_counter as u64, target_qp, force_idr)
                                    } else {
                                        Err("NVENC ZeroCopy requires offscreen buffer (GPU context)".to_string())
                                    }
                                },
                                GpuEncoder::Vaapi(enc) => {
                                    if let Some((_, ref dmabuf)) = state.offscreen_buffer {
                                        enc.encode_dmabuf(dmabuf, state.frame_counter as u64, target_qp, force_idr)
                                    } else {
                                        Err("Vaapi ZeroCopy requires offscreen buffer (GPU context)".to_string())
                                    }
                                }
                            };

                            if let Ok(data) = result {
                                if !data.is_empty() {
                                    state.encode_stats.frames.fetch_add(1, Ordering::Relaxed);
                                    state.encode_stats.stripes.fetch_add(1, Ordering::Relaxed);
                                    if let Some(ref tx) = state.deliver_tx {
                                        let stripes = vec![EncodedStripe {
                                            data, data_type: 2, stripe_y_start: 0,
                                            stripe_height: height, frame_id: state.frame_counter as i32,
                                        }];
                                        if let Some(ref socket) = state.recording_sink {
                                            for stripe in &stripes {
                                                let _ = socket.write_encoded_frame(stripe);
                                            }
                                        }
                                        // Non-blocking: a full slot parks the frame
                                        // (delivered ahead of any new encode above).
                                        match tx.try_send(stripes) {
                                            Ok(()) => {}
                                            Err(std::sync::mpsc::TrySendError::Full(s)) => {
                                                state.pending_hw_delivery = Some(s);
                                            }
                                            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {}
                                        }
                                    }
                                }
                            } else if let Err(e) = result {
                                eprintln!("HW Encode Error: {}", e);
                            }
                        }
                        state.pending_force_idr = false;
                        state.frame_counter = state.frame_counter.wrapping_add(1);
                        }
                    }
                    if let Some(resp) = state.pending_screenshot.take() {
                        if !state.frame_buffer.is_empty() {
                            let w = state.settings.width as u32;
                            let h = state.settings.height as u32;
                            let png = if state.use_gpu {
                                crate::computer_use::encode_png_rgba(&state.frame_buffer, w, h)
                            } else {
                                let mut rgba = state.frame_buffer.clone();
                                for px in rgba.chunks_exact_mut(4) {
                                    px.swap(0, 2);
                                }
                                crate::computer_use::encode_png_rgba(&rgba, w, h)
                            };
                            match png {
                                Ok(data) => { let _ = resp.send(data); }
                                Err(e) => { let _ = resp.send(Vec::new()); eprintln!("[ComputerUse] PNG encode error: {}", e); }
                            }
                        } else {
                            let _ = resp.send(Vec::new());
                        }
                    }
                }
            }
            if let Some((id, buf)) = pool_slot.take() {
                if let Some(ref pool) = state.encode_pool {
                    pool.cancel(id, buf);
                }
            }
            let work_elapsed = loop_start_time.elapsed();
            let fps = (if is_memory_throttling { 5.0 } else { state.settings.target_fps }).max(1.0);
            let target_frame_duration = Duration::from_secs_f64(1.0 / fps);
            let wait_duration = target_frame_duration.saturating_sub(work_elapsed);
            let final_wait = if wait_duration.as_millis() < 1 { Duration::from_millis(1) } else { wait_duration };
            TimeoutAction::ToDuration(final_wait)
        })
        .expect("Failed to init capture timer");

    event_loop
        .handle()
        .insert_source(Generic::new(display, Interest::READ, Mode::Level), |_, display, _state| {
            unsafe { display.get_mut().dispatch_clients(_state).unwrap(); }
            Ok(PostAction::Continue)
        })
        .unwrap();

    if let Ok(port_str) = std::env::var("PIXELFLUX_CU") {
        if let Ok(port) = port_str.parse::<u16>() {
            println!("[ComputerUse] Starting server on port {} (PIXELFLUX_CU={})", port, port);
            let cu_tx = command_tx.clone();
            thread::spawn(move || {
                crate::computer_use::run_cu_server(cu_tx, port);
            });
        } else {
            println!("[ComputerUse] Invalid PIXELFLUX_CU value: '{}' - expected a port number", port_str);
        }
    }

    event_loop.run(None, &mut state, |state| {
        state.process_pending_clipboard_read();
        state.dh.flush_clients().unwrap();
    }).unwrap();
}

/// Zero-copy encoded-frame handoff to Python. Owns the encoded `Vec<u8>` and
/// exposes it read-only via the buffer protocol, so `bytes(frame)` /
/// `memoryview(frame)` alias the Rust buffer instead of copying. Carries the
/// four stripe-metadata ints as Python attributes.
#[pyclass]
struct StripeFrame {
    data: Vec<u8>,
    #[pyo3(get, set)]
    data_type: i32,
    #[pyo3(get, set)]
    stripe_y_start: i32,
    #[pyo3(get, set)]
    stripe_height: i32,
    #[pyo3(get, set)]
    frame_id: i32,
}

impl StripeFrame {
    /// Hot-path constructor: MOVES the encoded buffer in (no copy) and carries stripe
    /// metadata as attributes, so the consumer can read it without parsing a header
    /// (required for omit_stripe_headers).
    fn new_owned_meta(data: Vec<u8>, data_type: i32, stripe_y_start: i32, stripe_height: i32, frame_id: i32) -> Self {
        Self { data, data_type, stripe_y_start, stripe_height, frame_id }
    }
}

#[pymethods]
impl StripeFrame {
    /// Symmetry / testability constructor: copies the bytes-like into the owned `Vec`.
    /// The hot path uses `new_owned_meta` (a move) instead.
    #[new]
    #[pyo3(signature = (data, data_type = 0, stripe_y_start = 0, stripe_height = 0, frame_id = 0))]
    fn new(data: Vec<u8>, data_type: i32, stripe_y_start: i32, stripe_height: i32, frame_id: i32) -> Self {
        Self { data, data_type, stripe_y_start, stripe_height, frame_id }
    }

    fn __len__(&self) -> usize {
        self.data.len()
    }

    /// Expose the owned bytes read-only through the Python buffer protocol.
    ///
    /// `PyBuffer_FillInfo` INCREFs `slf` into `view->obj`, pinning the `Vec` until every view is
    /// released, so memoryviews can outlive the Python `frame` handle.
    unsafe fn __getbuffer__(
        slf: PyRefMut<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        let r = pyo3::ffi::PyBuffer_FillInfo(
            view,
            slf.as_ptr(),
            slf.data.as_ptr() as *mut std::os::raw::c_void,
            slf.data.len() as pyo3::ffi::Py_ssize_t,
            1,
            flags,
        );
        if r != 0 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut pyo3::ffi::Py_buffer) {}
}

/// The Python handle to the one long-lived compositor thread. Because calloop, EGL/GBM, and
/// the Wayland display are all thread-affine and the compositor has to keep running to serve its
/// clients even between captures, the backend cannot be a passive object that starts work on demand:
/// constructing it spawns that thread and it stays up for the process lifetime. The struct itself is
/// nothing but the command-channel sender used to drive that thread across the boundary.
#[pyclass]
struct WaylandBackend {
    tx: smithay::reexports::calloop::channel::Sender<ThreadCommand>,
}

#[pymethods]
impl WaylandBackend {
    /// Construct the backend and spawn the long-lived compositor thread, handing it the
    /// strongest scheduling edge (nice -15) because it drives the calloop, input dispatch, the render
    /// loop, and — on the zero-copy path — the encode itself, all on this single thread, so any
    /// scheduling starvation here surfaces directly as dropped or late frames.
    #[new]
    #[pyo3(signature = (width, height, dri_node, auto_gpu_selected = false, cursor_size = -1))]
    fn new(
        width: i32,
        height: i32,
        dri_node: String,
        auto_gpu_selected: bool,
        cursor_size: i32,
    ) -> Self {
        let (tx, rx) = smithay::reexports::calloop::channel::channel();
        let cu_tx = tx.clone();
        thread::spawn(move || {
            crate::boost_thread_priority(-15);
            run_wayland_thread(rx, cu_tx, width, height, dri_node, auto_gpu_selected, cursor_size);
        });
        WaylandBackend { tx }
    }

    /// Begin a capture with the given frame callback and settings.
    ///
    /// Issuing the start also clears the interpreter-teardown gate: starting from Python proves the
    /// interpreter is live again after a manual atexit sweep.
    fn start_capture(&self, callback: Py<PyAny>, settings: &Bound<'_, PyAny>) -> PyResult<()> {
        let rust_settings = extract_settings(settings)?;

        PY_SHUTDOWN.store(false, Ordering::Relaxed);
        self.tx
            .send(ThreadCommand::StartCapture(callback, rust_settings))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to send start command: {}", e)))?;
        Ok(())
    }

    fn stop_capture(&self) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::StopCapture)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to send stop command: {}", e)))?;
        Ok(())
    }

    fn set_cursor_callback(&self, callback: Py<PyAny>) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::SetCursorCallback(callback))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set cursor callback: {}", e)))?;
        Ok(())
    }

    /// cb(mime: str, data: bytes) fires when a client app copies to the clipboard.
    fn set_clipboard_callback(&self, callback: Py<PyAny>) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::SetClipboardCallback(callback))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set clipboard callback: {}", e)))?;
        Ok(())
    }

    /// Compositor-side clipboard offer: serve `data` as `mime` to pasting clients.
    fn set_clipboard(&self, mime: String, data: Vec<u8>) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::SetClipboard { mime, data })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set clipboard: {}", e)))?;
        Ok(())
    }

    fn inject_key(&self, scancode: u32, state: u32) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::KeyboardKey { scancode, state })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject key: {}", e)))?;
        Ok(())
    }

    /// Swap the seat keyboard's xkb keymap (XKB_KEYMAP_FORMAT_TEXT_V1 text). The caller
    /// owns keysym-to-keycode policy: define keycodes here, then press them via
    /// `inject_key`. Ordered with key events on the one compositor channel.
    fn set_keymap_string(&self, text: String) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::SetKeymapString(text))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set keymap: {}", e)))?;
        Ok(())
    }

    /// Return the active xkb keymap as an XKB_KEYMAP_FORMAT_TEXT_V1 string so a consumer can
    /// build a reverse keysym->keycode map from the identical keymap.
    ///
    /// The GIL is released while awaiting the reply, because the Wayland thread can call back into
    /// Python and would otherwise deadlock; the wait is bounded so a stall cannot hang the caller,
    /// and an empty string is returned when the keymap cannot be read in time.
    fn get_xkb_keymap_string(&self, py: Python<'_>) -> PyResult<String> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<String>();
        self.tx
            .send(ThreadCommand::GetXkbKeymap { reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to request keymap: {}", e)))?;
        let result = py.detach(move || reply_rx.recv_timeout(Duration::from_secs(2)));
        match result {
            Ok(s) => Ok(s),
            Err(_) => Ok(String::new()),
        }
    }

    fn inject_mouse_move(&self, x: f64, y: f64) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::PointerMotion { x, y })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject motion: {}", e)))?;
        Ok(())
    }

    fn inject_relative_mouse_move(&self, dx: f64, dy: f64) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::PointerRelativeMotion { dx, dy })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject relative motion: {}", e)))?;
        Ok(())
    }

    fn inject_mouse_button(&self, btn: u32, state: u32) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::PointerButton { btn, state })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject button: {}", e)))?;
        Ok(())
    }

    fn inject_mouse_scroll(&self, x: f64, y: f64) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::PointerAxis { x, y })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject axis: {}", e)))?;
        Ok(())
    }

    fn set_cursor_rendering(&self, enabled: bool) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::UpdateCursorConfig { render_on_framebuffer: enabled })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set cursor config: {}", e)))?;
        Ok(())
    }

    /// Forces an IDR/keyframe on the next captured frame so a (re)connecting client
    /// or a decoder reset can resume immediately. With the default infinite GOP this
    /// is the only recovery path, so every consumer that can lose decoder state must
    /// call it. No-op cost on the JPEG/software path (keyframes are N/A).
    fn request_idr_frame(&self) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::RequestIdr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to request IDR: {}", e)))?;
        Ok(())
    }

    /// Apply a live bitrate (kbps) / VBV (kb) / framerate change to the running capture.
    fn update_rate(&self, bitrate_kbps: Option<i32>, vbv_multiplier: Option<f64>, fps: Option<f64>) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::UpdateRate { bitrate_kbps, vbv_multiplier, fps })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to update rate: {}", e)))?;
        Ok(())
    }
}

impl WaylandBackend {
    fn update_tunables(&self, t: LiveTunables) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::UpdateTunables(t))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to update tunables: {}", e)))?;
        Ok(())
    }
}

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use crate::encoders::software::EncodedStripe;

/// Let Python wrap already-encoded bytes back into a `StripeFrame`, for callers that produce or
/// replay stripe data outside a live capture (tests, re-sends to a late joiner). It copies the
/// buffer-like input (bytes/bytearray/memoryview) in because the frame owns its bytes; the capture
/// hot path instead uses `new_owned_meta` to MOVE the encoder's buffer with no copy, so this
/// copying constructor stays off that path.
#[pyfunction]
#[pyo3(signature = (data, data_type = 0, stripe_y_start = 0, stripe_height = 0, frame_id = 0))]
fn stripe_frame_from_buffer(
    data: Vec<u8>,
    data_type: i32,
    stripe_y_start: i32,
    stripe_height: i32,
    frame_id: i32,
) -> StripeFrame {
    StripeFrame::new_owned_meta(data, data_type, stripe_y_start, stripe_height, frame_id)
}

/// Capture configuration read by `start_capture` (each field by attribute name via
/// `extract_settings`, so the field names must match exactly). Declared `dict` so callers
/// can stash extra attributes not listed here.
#[pyclass(dict)]
struct CaptureSettings {
    #[pyo3(get, set)] capture_width: i32,
    #[pyo3(get, set)] capture_height: i32,
    #[pyo3(get, set)] scale: f64,
    #[pyo3(get, set)] capture_x: i32,
    #[pyo3(get, set)] capture_y: i32,
    #[pyo3(get, set)] target_fps: f64,
    #[pyo3(get, set)] jpeg_quality: i32,
    #[pyo3(get, set)] paint_over_jpeg_quality: i32,
    #[pyo3(get, set)] use_paint_over_quality: bool,
    #[pyo3(get, set)] paint_over_trigger_frames: i32,
    #[pyo3(get, set)] damage_block_threshold: i32,
    #[pyo3(get, set)] damage_block_duration: i32,
    #[pyo3(get, set)] output_mode: i32,
    #[pyo3(get, set)] video_crf: i32,
    #[pyo3(get, set)] video_paintover_crf: i32,
    #[pyo3(get, set)] video_paintover_burst_frames: i32,
    #[pyo3(get, set)] video_fullcolor: bool,
    #[pyo3(get, set)] video_fullframe: bool,
    #[pyo3(get, set)] video_streaming_mode: bool,
    #[pyo3(get, set)] capture_cursor: bool,
    #[pyo3(get, set)] watermark_path: Py<PyAny>,
    #[pyo3(get, set)] watermark_location_enum: i32,
    #[pyo3(get, set)] encode_node_index: i32,
    #[pyo3(get, set)] use_cpu: bool,
    #[pyo3(get, set)] use_openh264: bool,
    #[pyo3(get, set)] debug_logging: bool,
    #[pyo3(get, set)] video_cbr_mode: bool,
    #[pyo3(get, set)] video_bitrate_kbps: i32,
    #[pyo3(get, set)] video_vbv_multiplier: f64,
    #[pyo3(get, set)] keyframe_interval_s: f64,
    #[pyo3(get, set)] video_min_qp: i32,
    #[pyo3(get, set)] video_max_qp: i32,
    #[pyo3(get, set)] auto_adjust_screen_capture_size: bool,
    #[pyo3(get, set)] omit_stripe_headers: bool,
    #[pyo3(get, set)] encode_node_path: Py<PyAny>,
    /// Compositor render node (Wayland): an explicit path wins; empty with auto_gpu
    /// set lets the library pick one; empty without falls back to the encoder node.
    #[pyo3(get, set)] render_node_path: Py<PyAny>,
    /// Auto-GPU request: "" = off, "true" = first GPU, any other token = first GPU
    /// whose kernel identity matches (vendor name, driver name, DT prefix, PCI id).
    #[pyo3(get, set)] auto_gpu: Py<PyAny>,
    /// Backend choice: True/False force Wayland/X11; None follows WAYLAND_DISPLAY.
    #[pyo3(get, set)] use_wayland: Py<PyAny>,
    /// H.264 recording tap: a Unix socket path to bind, or empty for none.
    #[pyo3(get, set)] recording_socket: Py<PyAny>,
    /// Compositor cursor-theme size in pixels; <=0 keeps the theme default (24).
    #[pyo3(get, set)] cursor_size: i32,
    /// Longest cursor edge the X11 out-of-band cursor callback delivers; larger
    /// images are downscaled. <=0 disables the cap.
    #[pyo3(get, set)] cursor_size_cap: i32,
}

#[pymethods]
impl CaptureSettings {
    #[new]
    fn new(py: Python<'_>) -> Self {
        Self {
            capture_width: 1920, capture_height: 1080, scale: 1.0, capture_x: 0, capture_y: 0,
            target_fps: 60.0, jpeg_quality: 85, paint_over_jpeg_quality: 95,
            use_paint_over_quality: false, paint_over_trigger_frames: 10,
            damage_block_threshold: 15, damage_block_duration: 30, output_mode: 0,
            video_crf: 25, video_paintover_crf: 18, video_paintover_burst_frames: 5,
            video_fullcolor: false, video_fullframe: false, video_streaming_mode: false,
            capture_cursor: false, watermark_path: py.None(), watermark_location_enum: 0,
            encode_node_index: -2, use_cpu: false, use_openh264: false, debug_logging: false,
            video_cbr_mode: false, video_bitrate_kbps: 4000, video_vbv_multiplier: 0.0,
            keyframe_interval_s: 0.0,
            video_min_qp: 0, video_max_qp: 0,
            auto_adjust_screen_capture_size: false, omit_stripe_headers: false,
            encode_node_path: py.None(),
            render_node_path: py.None(), auto_gpu: py.None(), use_wayland: py.None(),
            recording_socket: py.None(), cursor_size: -1, cursor_size_cap: 32,
        }
    }
}

/// Process-wide Wayland backend: input and capture share ONE compositor (constructed lazily).
static WAYLAND_BACKEND: OnceLock<Mutex<Option<Py<WaylandBackend>>>> = OnceLock::new();
/// Cursor callback registered before the backend exists (selkies registers it pre-start);
/// applied when the backend is created, which is deferred to capture start so the real
/// render node (not a placeholder) reaches the compositor.
static PENDING_CURSOR_CALLBACK: Mutex<Option<Py<PyAny>>> = Mutex::new(None);
/// Interpreter-teardown gate, set by the atexit sweep: the detached compositor and delivery
/// threads must never attach to a finalizing interpreter (aborts the process pre-3.13).
/// Cleared by a fresh capture start (only a live interpreter can start one).
pub(crate) static PY_SHUTDOWN: AtomicBool = AtomicBool::new(false);
/// ScreenCapture id that owns the active shared Wayland capture (0 = none). Only the owner may
/// stop it, so an input-only or stale instance can't tear down a live capture.
static WAYLAND_OWNER: AtomicU64 = AtomicU64::new(0);
/// Real Wayland capture liveness (StartCapture -> true, StopCapture -> false), mirrored
/// from AppState.is_capturing so the Python-facing is_capturing() reports whether the
/// pipeline is actually running, not merely which ScreenCapture owns the backend.
static WAYLAND_CAPTURE_ALIVE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// Hands each `ScreenCapture` a unique, monotonic id — the token `WAYLAND_OWNER` compares to
/// decide which instance is allowed to stop the one shared compositor capture. Starts at 1 so 0 can
/// mean "no owner".
static NEXT_CAPTURE_ID: AtomicU64 = AtomicU64::new(1);
/// Registry of every live X11 capture's `Controls`, so the atexit sweep can flag them all to
/// stop before the interpreter finalizes even after the owning `ScreenCapture` Python handles have
/// been dropped — without a central registry those captures would have no reachable stop switch.
static LIVE_X11: OnceLock<Mutex<Vec<Arc<crate::x11::Controls>>>> = OnceLock::new();

fn live_x11() -> &'static Mutex<Vec<Arc<crate::x11::Controls>>> {
    LIVE_X11.get_or_init(|| Mutex::new(Vec::new()))
}

/// Convert premultiplied-alpha RGBA pixels to the straight alpha PNG expects. Every cursor
/// source feeding the callback stores premultiplied color (XFixes and Xcursor by format
/// definition, wl_shm/dmabuf by Wayland convention); encoding those values as straight
/// alpha double-darkens antialiased edges. Rounds as `(c * 255 + a/2) / a`, clamped —
/// selkies' python seed path mirrors this exact integer math so both sources hash a cursor
/// to the same content handle.
pub(crate) fn unpremultiply_rgba(image: &mut image::RgbaImage) {
    for p in image.pixels_mut() {
        let a = p.0[3] as u32;
        if a == 0 {
            p.0 = [0, 0, 0, 0];
        } else if a < 255 {
            for c in &mut p.0[..3] {
                *c = ((*c as u32 * 255 + a / 2) / a).min(255) as u8;
            }
        }
    }
}

/// Best-effort nice boost for the calling capture/encode/delivery thread. These threads
/// compete with the very workload being captured, so a scheduling edge keeps frame pacing
/// steady under load. Requires CAP_SYS_NICE (or root); otherwise EPERM and silently a no-op.
pub(crate) fn boost_thread_priority(nice: libc::c_int) {
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        let _ = libc::setpriority(libc::PRIO_PROCESS, tid, nice);
    }
}

/// Forward a live rate change to the shared Wayland backend (no-op if none is running).
fn wayland_update_rate(
    py: Python<'_>,
    bitrate_kbps: Option<i32>,
    vbv_multiplier: Option<f64>,
    fps: Option<f64>,
) {
    if let Some(slot) = WAYLAND_BACKEND.get() {
        if let Some(be) = slot.lock().unwrap().as_ref() {
            let _ = be.bind(py).borrow().update_rate(bitrate_kbps, vbv_multiplier, fps);
        }
    }
}

/// Forward live per-frame tunables to the shared Wayland backend (no-op if none is running).
fn wayland_update_tunables(py: Python<'_>, t: LiveTunables) {
    if let Some(slot) = WAYLAND_BACKEND.get() {
        if let Some(be) = slot.lock().unwrap().as_ref() {
            let _ = be.bind(py).borrow().update_tunables(t);
        }
    }
}

/// Get-or-create the singleton Wayland backend (idempotent: the first dimensions and render
/// node win, and a later capture just resizes).
///
/// Called from capture start (which knows the operator's real node) and from the import-time
/// bootstrap when the deployment opts into pixelflux-as-compositor. The render node is chosen once,
/// at creation, by precedence: an explicit `render_node_path`, then an `auto_gpu` pick, then the
/// encoder node (so a caller that sets only one node still renders on that GPU); empty selects the
/// software renderer.
fn ensure_wayland_backend(
    py: Python<'_>,
    width: i32,
    height: i32,
    explicit_node: String,
    auto_gpu: String,
    fallback_node: String,
    cursor_size: i32,
) -> PyResult<Py<WaylandBackend>> {
    let slot = WAYLAND_BACKEND.get_or_init(|| Mutex::new(None));
    let mut g = slot.lock().unwrap();
    if g.is_none() {
        let mut node = (!explicit_node.is_empty()).then_some(explicit_node);
        let mut auto_gpu_selected = false;
        if node.is_none() {
            if let Some(request) = parse_auto_gpu(&auto_gpu) {
                match auto_select_render_node(request.as_deref()) {
                    Some(picked) => {
                        println!("[Wayland] AUTO_GPU enabled. Selected: {}", picked);
                        node = Some(picked);
                        auto_gpu_selected = true;
                    }
                    None => {
                        if let Some(token) = request {
                            eprintln!("[pixelflux] AUTO_GPU={token}: no matching GPU found.");
                        }
                    }
                }
            }
        }
        let node = node.unwrap_or(fallback_node);
        let be = Py::new(
            py,
            WaylandBackend::new(width, height, node, auto_gpu_selected, cursor_size),
        )?;
        if let Some(cb) = PENDING_CURSOR_CALLBACK.lock().unwrap().take() {
            let _ = be.bind(py).borrow().set_cursor_callback(cb);
        }
        *g = Some(be);
    }
    Ok(g.as_ref().unwrap().clone_ref(py))
}

/// The live Wayland backend, if any — never creates one. The pre-capture entry points
/// (input injection, cursor/config setters) use this so they can't lock in a backend
/// with a placeholder render node.
fn wayland_backend_running(py: Python<'_>) -> Option<Py<WaylandBackend>> {
    let slot = WAYLAND_BACKEND.get()?;
    let g = slot.lock().unwrap();
    g.as_ref().map(|b| b.clone_ref(py))
}

/// Backend choice: an explicit `use_wayland` bool in the settings wins (selkies
/// forwards --wayland / SELKIES_WAYLAND there); when left unset (None), capture
/// goes through Wayland exactly when the session exposes a WAYLAND_DISPLAY.
fn want_wayland(settings: &Bound<'_, PyAny>) -> bool {
    if let Some(explicit) = settings
        .getattr("use_wayland")
        .ok()
        .and_then(|v| v.extract::<bool>().ok())
    {
        return explicit;
    }
    std::env::var("WAYLAND_DISPLAY").map(|v| !v.is_empty()).unwrap_or(false)
}

/// Mutable per-capture state behind `ScreenCapture`'s mutex: the active backend, the live
/// X11 controls and thread handle, and the capture / encode thread ids used to detect a re-entrant
/// stop.
struct ScState {
    /// 0 = idle, 1 = X11, 2 = Wayland.
    backend: u8,
    controls: Option<Arc<crate::x11::Controls>>,
    handle: Option<thread::JoinHandle<()>>,
    cap_thread_id: Option<thread::ThreadId>,
    /// The internal encode thread's id, so a re-entrant stop arriving on it is
    /// detected and doesn't try to join a chain that includes itself.
    encode_thread_id: Option<thread::ThreadId>,
    /// Handshake receiver kept when the bounded start-time wait for the encode thread id
    /// lapsed (slow X11 setup precedes the encode-thread spawn): the id is late-resolved
    /// from here on demand, so the re-entrant-stop guard still recognizes the encode
    /// thread — with `encode_thread_id` stuck at `None`, a stop from inside the delivery
    /// callback would join the capture thread, which joins the encode thread (the
    /// caller), a deadlock cycle. The id send strictly precedes any callback running on
    /// that thread, so a `try_recv` at stop time cannot miss it.
    encode_tid_rx: Option<std::sync::mpsc::Receiver<thread::ThreadId>>,
    /// Delivery thread: owns the GIL-bound Python callback so encode(N+1) never
    /// serializes behind deliver(N). Joined on stop; a re-entrant stop from
    /// inside the callback (which runs on this thread) must detach instead of
    /// self-joining.
    deliver_handle: Option<thread::JoinHandle<()>>,
    deliver_thread_id: Option<thread::ThreadId>,
}

/// Unified capture handle exposed to Python. Drives the X11 capture directly or delegates to the
/// shared Wayland backend, chosen at `start_capture` time. Exposes start_capture / stop_capture /
/// request_idr_frame / update_* / is_capturing, plus the Wayland input-injection methods.
#[pyclass]
struct ScreenCapture {
    id: u64,
    inner: Mutex<ScState>,
}

impl ScreenCapture {
    /// Stop this capture: signal the capture thread, drop the live controls, and join.
    ///
    /// The path forks on the backend. A **Wayland** capture only tells the shared compositor to
    /// stop when this instance still owns it — ownership is claimed-and-cleared atomically so a
    /// stale stop cannot tear down a capture another instance just started. An **X11** capture joins
    /// its capture thread (which also joins the encode thread) and then its deliver thread,
    /// releasing the GIL first because the deliver thread runs the Python callback and holding the
    /// GIL across the joins would deadlock. A re-entrant stop arriving on the capture, encode, or
    /// deliver thread cannot join itself, so it detaches and lets the threads exit on the stop flag.
    fn stop_internal(&self, py: Python<'_>) -> PyResult<()> {
        let (handle, deliver_handle, same_thread, backend, controls) = {
            let mut st = self.inner.lock().unwrap();
            if let Some(c) = &st.controls {
                c.stop.store(true, Ordering::Relaxed);
            }
            let cur = Some(thread::current().id());
            if st.encode_thread_id.is_none() {
                if let Some(rx) = st.encode_tid_rx.as_ref() {
                    if let Ok(id) = rx.try_recv() {
                        st.encode_thread_id = Some(id);
                    }
                }
            }
            let same = st.cap_thread_id == cur
                || st.encode_thread_id == cur
                || st.deliver_thread_id == cur;
            let controls = st.controls.take();
            let handle = st.handle.take();
            let deliver_handle = st.deliver_handle.take();
            let backend = st.backend;
            st.backend = 0;
            st.cap_thread_id = None;
            st.encode_thread_id = None;
            st.encode_tid_rx = None;
            st.deliver_thread_id = None;
            (handle, deliver_handle, same, backend, controls)
        };
        if let Some(c) = &controls {
            live_x11().lock().unwrap().retain(|x| !Arc::ptr_eq(x, c));
        }
        if backend == 1 {
            crate::x11::cursor::release(py);
        }
        if backend == 2 {
            if WAYLAND_OWNER
                .compare_exchange(self.id, 0, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                if let Some(slot) = WAYLAND_BACKEND.get() {
                    if let Some(be) = slot.lock().unwrap().as_ref() {
                        let _ = be.bind(py).borrow().stop_capture();
                    }
                }
            }
        } else {
            if same_thread {
                // Detach: the threads exit on the stop flag once the callback returns.
                drop(handle);
                drop(deliver_handle);
            } else {
                py.detach(|| {
                    if let Some(h) = handle {
                        let _ = h.join();
                    }
                    // The capture join above ends the encode thread, dropping the
                    // delivery sender; the deliver thread then drains its one
                    // queued frame and exits, so this join is bounded.
                    if let Some(h) = deliver_handle {
                        let _ = h.join();
                    }
                });
            }
        }
        Ok(())
    }
}

#[pymethods]
impl ScreenCapture {
    #[new]
    fn new() -> Self {
        Self {
            id: NEXT_CAPTURE_ID.fetch_add(1, Ordering::Relaxed),
            inner: Mutex::new(ScState {
                backend: 0,
                controls: None,
                handle: None,
                cap_thread_id: None,
                encode_thread_id: None,
                encode_tid_rx: None,
                deliver_handle: None,
                deliver_thread_id: None,
            }),
        }
    }

    /// Begin capture: `callback(frame)` is invoked per encoded stripe with a `StripeFrame`.
    ///
    /// The backend is chosen from the settings (`want_wayland`). A **Wayland** start delegates to
    /// the shared backend, distinguishing the compositor RENDER node from the ENCODER node and
    /// resolving an AUTO_GPU request; restarting this instance's own live Wayland capture skips the
    /// stop so the calloop can reconfigure the running session in place — keeping a compatible NVENC
    /// session alive — instead of destroying it and forcing a full rebuild. An **X11** start
    /// resolves AUTO_GPU to an encoder device when none was chosen explicitly, then spawns the
    /// capture thread (which internally spawns the encode+deliver thread). The per-frame delivery
    /// closure makes one GIL acquisition per frame with all stripes batched, and a failed start
    /// surfaces as a `PyErr` rather than a silent, forever-"capturing" state.
    fn start_capture(
        &self,
        py: Python<'_>,
        callback: Py<PyAny>,
        settings: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let live_wayland_restart = want_wayland(settings)
            && self.inner.lock().unwrap().backend == 2
            && WAYLAND_OWNER.load(Ordering::Relaxed) == self.id
            && WAYLAND_CAPTURE_ALIVE.load(Ordering::Acquire);
        if !live_wayland_restart {
            self.stop_internal(py)?;
        }
        let rs = extract_settings(settings)?;

        if want_wayland(settings) {
            let read_node = |attr: &str| -> Option<String> {
                settings.getattr(attr).ok().and_then(|o| {
                    o.extract::<String>()
                        .or_else(|_| {
                            o.extract::<Vec<u8>>()
                                .map(|b| String::from_utf8_lossy(&b).into_owned())
                        })
                        .ok()
                })
            };
            let cursor_size = settings
                .getattr("cursor_size")
                .ok()
                .and_then(|v| v.extract::<i32>().ok())
                .unwrap_or(-1);
            let be = ensure_wayland_backend(
                py,
                rs.width,
                rs.height,
                read_node("render_node_path").unwrap_or_default(),
                read_node("auto_gpu").unwrap_or_default(),
                read_node("encode_node_path").unwrap_or_default(),
                cursor_size,
            )?;
            be.bind(py).borrow().start_capture(callback, settings)?;
            WAYLAND_OWNER.store(self.id, Ordering::Relaxed);
            self.inner.lock().unwrap().backend = 2;
            return Ok(());
        }

        let mut rs = rs;
        if rs.encode_node_index < -1 {
            let auto_gpu = settings
                .getattr("auto_gpu")
                .ok()
                .and_then(|o| {
                    o.extract::<String>()
                        .or_else(|_| {
                            o.extract::<Vec<u8>>()
                                .map(|b| String::from_utf8_lossy(&b).into_owned())
                        })
                        .ok()
                })
                .unwrap_or_default();
            if let Some(request) = parse_auto_gpu(&auto_gpu) {
                if let Some(picked) = auto_select_render_node(request.as_deref()) {
                    if let Some(idx) = picked
                        .strip_prefix("/dev/dri/renderD")
                        .and_then(|s| s.parse::<i32>().ok())
                    {
                        println!("[x11] AUTO_GPU enabled. Selected: {picked}");
                        rs.encode_node_index = idx - 128;
                    }
                }
            }
        }

        println!(
            "[x11] Configuring Output: {}x{} @ {:.2} FPS (Encode Node: {})",
            rs.width, rs.height, rs.target_fps, rs.encode_node_index
        );

        let controls = Arc::new(crate::x11::Controls::new(&rs));
        let cursor_cap = rs.cursor_size_cap;
        live_x11().lock().unwrap().push(controls.clone());
        let c2 = controls.clone();
        let c3 = controls.clone();
        let cb = callback;
        let err_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let err_slot2 = err_slot.clone();

        // The Python callback is GIL-bound: run it on its own thread (as the
        // Wayland backend and pcmflux do) so encode(N+1) never serializes behind
        // deliver(N). Bounded at one in-flight frame: a slower consumer
        // backpressures the encoder by exactly one frame instead of growing a
        // queue, and no frame is ever dropped.
        let (deliver_tx, deliver_rx) = std::sync::mpsc::sync_channel::<Vec<EncodedStripe>>(1);
        let deliver_handle = thread::spawn(move || {
            crate::boost_thread_priority(-10);
            while let Ok(frame) = deliver_rx.recv() {
                if PY_SHUTDOWN.load(Ordering::Relaxed) {
                    continue;
                }
                Python::attach(|py| {
                    for s in frame {
                        match Py::new(
                            py,
                            StripeFrame::new_owned_meta(
                                s.data,
                                s.data_type,
                                s.stripe_y_start,
                                s.stripe_height,
                                s.frame_id,
                            ),
                        ) {
                            Ok(f) => {
                                if let Err(e) = cb.call1(py, (f,)) {
                                    e.print(py);
                                }
                            }
                            Err(e) => eprintln!("[x11] frame alloc error: {e:?}"),
                        }
                    }
                });
            }
        });
        let deliver_thread_id = deliver_handle.thread().id();

        let on_frame = move |frame: Vec<EncodedStripe>| {
            // Blocks only when the single slot is still occupied (consumer more
            // than one frame behind); a dropped receiver (stop) discards.
            let _ = deliver_tx.send(frame);
        };

        let (tid_tx, tid_rx) = std::sync::mpsc::channel();
        let (etid_tx, etid_rx) = std::sync::mpsc::channel();
        let handle = thread::spawn(move || {
            crate::boost_thread_priority(-15);
            let _ = tid_tx.send(thread::current().id());
            let res = crate::x11::run_capture(rs, c2, etid_tx, on_frame);
            c3.stop.store(true, Ordering::Release);
            if let Err(e) = res {
                let msg = e.to_string();
                eprintln!("[x11] capture error: {msg}");
                if let Ok(mut g) = err_slot2.lock() {
                    *g = Some(msg);
                }
            }
        });
        let (tid, etid_res, etid_rx) = py.detach(move || {
            let tid = tid_rx.recv().ok();
            let etid_res = etid_rx.recv_timeout(std::time::Duration::from_secs(2));
            (tid, etid_res, etid_rx)
        });
        let mut late_etid_rx = None;
        let etid = match etid_res {
            Ok(id) => Some(id),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                let _ = handle.join();
                live_x11().lock().unwrap().retain(|x| !Arc::ptr_eq(x, &controls));
                let msg = err_slot
                    .lock()
                    .ok()
                    .and_then(|g| g.clone())
                    .unwrap_or_else(|| "X11 capture thread exited during start".to_string());
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Slow pre-spawn X11 setup: the id will still arrive on this
                // channel. Keep the receiver so stop_internal can late-resolve
                // it — a None id would blind the re-entrant-stop guard.
                late_etid_rx = Some(etid_rx);
                None
            }
        };
        let mut st = self.inner.lock().unwrap();
        st.backend = 1;
        st.controls = Some(controls);
        st.handle = Some(handle);
        st.cap_thread_id = tid;
        st.encode_thread_id = etid;
        st.encode_tid_rx = late_etid_rx;
        st.deliver_handle = Some(deliver_handle);
        st.deliver_thread_id = Some(deliver_thread_id);
        drop(st);
        crate::x11::cursor::acquire(cursor_cap);
        Ok(())
    }

    fn stop_capture(&self, py: Python<'_>) -> PyResult<()> {
        self.stop_internal(py)
    }

    fn request_idr_frame(&self, py: Python<'_>) -> PyResult<()> {
        let (backend, controls) = {
            let st = self.inner.lock().unwrap();
            (st.backend, st.controls.clone())
        };
        match backend {
            1 => {
                if let Some(c) = controls {
                    c.force_idr.store(true, Ordering::Relaxed);
                }
            }
            2 => {
                if let Some(slot) = WAYLAND_BACKEND.get() {
                    if let Some(be) = slot.lock().unwrap().as_ref() {
                        let _ = be.bind(py).borrow().request_idr_frame();
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Apply a live target-bitrate (kbps) change to the running capture.
    ///
    /// On the X11 path the dirty flag is Release-published after the payload store, so the encode
    /// thread's Acquire read can never observe the flag set against a stale bitrate.
    fn update_video_bitrate(&self, py: Python<'_>, kbps: i32) -> PyResult<()> {
        let (backend, controls) = {
            let st = self.inner.lock().unwrap();
            (st.backend, st.controls.clone())
        };
        match backend {
            1 => {
                if let Some(c) = &controls {
                    c.bitrate_kbps.store(kbps, Ordering::Relaxed);
                    c.rate_dirty.store(true, Ordering::Release);
                }
            }
            2 => wayland_update_rate(py, Some(kbps), None, None),
            _ => {}
        }
        Ok(())
    }

    fn update_framerate(&self, py: Python<'_>, fps: f64) -> PyResult<()> {
        let (backend, controls) = {
            let st = self.inner.lock().unwrap();
            (st.backend, st.controls.clone())
        };
        match backend {
            1 => {
                if let Some(c) = &controls {
                    c.fps_milli.store((fps.max(1.0) * 1000.0) as u64, Ordering::Relaxed);
                    c.rate_dirty.store(true, Ordering::Release);
                }
            }
            2 => wayland_update_rate(py, None, None, Some(fps)),
            _ => {}
        }
        Ok(())
    }

    /// Live CBR VBV change, as a multiple of one frame's bit budget (<= 0 = policy default).
    fn update_vbv_multiplier(&self, py: Python<'_>, multiplier: f64) -> PyResult<()> {
        let (backend, controls) = {
            let st = self.inner.lock().unwrap();
            (st.backend, st.controls.clone())
        };
        match backend {
            1 => {
                if let Some(c) = &controls {
                    c.vbv_mult_milli
                        .store((multiplier * 1000.0).round() as i32, Ordering::Relaxed);
                    c.rate_dirty.store(true, Ordering::Release);
                }
            }
            2 => wayland_update_rate(py, None, Some(multiplier), None),
            _ => {}
        }
        Ok(())
    }

    /// Apply the live-tunable subset of `settings` (quality, paint-over, streaming mode,
    /// cursor overlay, keyframe interval) to the running capture -- no restart, no encoder
    /// re-init. Structural changes (encoder, chroma, RC mode, device) still need a restart.
    fn update_tunables(&self, py: Python<'_>, settings: &Bound<'_, PyAny>) -> PyResult<()> {
        let rs = extract_settings(settings)?;
        let t = LiveTunables::from_settings(&rs);
        let (backend, controls) = {
            let st = self.inner.lock().unwrap();
            (st.backend, st.controls.clone())
        };
        match backend {
            1 => {
                if let Some(c) = &controls {
                    c.capture_cursor.store(t.capture_cursor, Ordering::Relaxed);
                    *c.tunables.lock().unwrap() = Some(t);
                    c.tunables_dirty.store(true, Ordering::Release);
                }
                crate::x11::cursor::set_size_cap(rs.cursor_size_cap);
            }
            2 => wayland_update_tunables(py, t),
            _ => {}
        }
        Ok(())
    }

    /// Move/resize the live X11 capture region (root-relative). The capture loop drains
    /// in-flight frames, re-targets its surfaces, and the encoder follows in place where
    /// it can (NVENC reconfigure / stripe re-derive) -- no capture restart. `width`/
    /// `height` <= 0 mean "to the root edge". On Wayland the output IS the capture
    /// region: restart the capture with new dimensions instead (in-place there too).
    fn update_capture_region(&self, x: i32, y: i32, width: i32, height: i32) -> PyResult<()> {
        let controls = {
            let st = self.inner.lock().unwrap();
            if st.backend == 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "update_capture_region is X11-only; on Wayland restart the capture with new dimensions",
                ));
            }
            st.controls.clone()
        };
        if let Some(c) = &controls {
            *c.region.lock().unwrap() = (x.max(0), y.max(0), width, height);
            c.region_dirty.store(true, Ordering::Release);
        }
        Ok(())
    }

    #[getter]
    fn is_capturing(&self) -> bool {
        let st = self.inner.lock().unwrap();
        match st.backend {
            1 => st
                .controls
                .as_ref()
                .map(|c| !c.stop.load(Ordering::Relaxed))
                .unwrap_or(false),
            2 => WAYLAND_OWNER.load(Ordering::Relaxed) == self.id
                && WAYLAND_CAPTURE_ALIVE.load(Ordering::Acquire),
            _ => false,
        }
    }

    fn inject_key(&self, py: Python<'_>, scancode: u32, state: u32) -> PyResult<()> {
        wayland_backend_running(py).map_or(Ok(()), |be| be.bind(py).borrow().inject_key(scancode, state))
    }
    fn set_keymap_string(&self, py: Python<'_>, text: String) -> PyResult<()> {
        wayland_backend_running(py).map_or(Ok(()), |be| be.bind(py).borrow().set_keymap_string(text))
    }
    fn inject_mouse_move(&self, py: Python<'_>, x: f64, y: f64) -> PyResult<()> {
        wayland_backend_running(py).map_or(Ok(()), |be| be.bind(py).borrow().inject_mouse_move(x, y))
    }
    fn inject_relative_mouse_move(&self, py: Python<'_>, dx: f64, dy: f64) -> PyResult<()> {
        wayland_backend_running(py).map_or(Ok(()), |be| be.bind(py).borrow().inject_relative_mouse_move(dx, dy))
    }
    fn inject_mouse_button(&self, py: Python<'_>, btn: u32, state: u32) -> PyResult<()> {
        wayland_backend_running(py).map_or(Ok(()), |be| be.bind(py).borrow().inject_mouse_button(btn, state))
    }
    fn inject_mouse_scroll(&self, py: Python<'_>, x: f64, y: f64) -> PyResult<()> {
        wayland_backend_running(py).map_or(Ok(()), |be| be.bind(py).borrow().inject_mouse_scroll(x, y))
    }
    /// Toggle compositing the cursor into captured frames (the alternative to the
    /// out-of-band cursor callback): the X11 grab re-reads the flag per frame, Wayland
    /// forwards to the compositor.
    fn set_cursor_rendering(&self, py: Python<'_>, enabled: bool) -> PyResult<()> {
        let (backend, controls) = {
            let st = self.inner.lock().unwrap();
            (st.backend, st.controls.clone())
        };
        if backend == 1 {
            if let Some(c) = &controls {
                c.capture_cursor.store(enabled, Ordering::Relaxed);
            }
            return Ok(());
        }
        wayland_backend_running(py).map_or(Ok(()), |be| be.bind(py).borrow().set_cursor_rendering(enabled))
    }
    /// Register the client-copy cursor callback for whichever backend runs: the X11 cursor
    /// monitor reads it from its shared slot (re-delivering the current cursor to a late
    /// registration), and the Wayland backend takes it directly — or stashes it in
    /// `PENDING_CURSOR_CALLBACK`, applied by `ensure_wayland_backend` at creation; the
    /// backend slot lock is held across the check so a concurrent creation cannot miss the
    /// stash.
    fn set_cursor_callback(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<()> {
        crate::x11::cursor::set_callback(callback.clone_ref(py));
        let slot = WAYLAND_BACKEND.get_or_init(|| Mutex::new(None));
        let g = slot.lock().unwrap();
        match g.as_ref() {
            Some(be) => be.bind(py).borrow().set_cursor_callback(callback),
            None => {
                *PENDING_CURSOR_CALLBACK.lock().unwrap() = Some(callback);
                Ok(())
            }
        }
    }
    fn get_xkb_keymap_string(&self, py: Python<'_>) -> PyResult<String> {
        wayland_backend_running(py)
            .map_or(Ok(String::new()), |be| be.bind(py).borrow().get_xkb_keymap_string(py))
    }
    /// cb(mime: str, data: bytes) fires when a client app copies to the clipboard.
    fn set_clipboard_callback(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<()> {
        match wayland_backend_running(py) {
            Some(be) => be.bind(py).borrow().set_clipboard_callback(callback),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "wayland backend not running",
            )),
        }
    }
    /// Compositor-side clipboard offer: serve `data` as `mime` to pasting clients.
    fn set_clipboard(&self, py: Python<'_>, mime: String, data: Vec<u8>) -> PyResult<()> {
        match wayland_backend_running(py) {
            Some(be) => be.bind(py).borrow().set_clipboard(mime, data),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "wayland backend not running",
            )),
        }
    }
}

/// Best-effort teardown: flag the capture thread to exit without joining.
///
/// Joining would need the GIL (the thread calls back into Python), which `Drop` cannot safely take,
/// so this only sets the stop flag; the actual join is left to an explicit `stop_capture` or the
/// atexit sweep.
impl Drop for ScreenCapture {
    fn drop(&mut self) {
        if let Ok(st) = self.inner.lock() {
            if let Some(c) = &st.controls {
                c.stop.store(true, Ordering::Relaxed);
            }
        }
    }
}

/// Bring the Wayland compositor socket up before any capture so apps launched early can
/// connect (sets WAYLAND_DISPLAY for children of this process). Idempotent; the running
/// backend keeps its node/dimensions on later calls. `render_node` is an explicit
/// /dev/dri/renderD* path; `auto_gpu` is a truthy string or vendor/driver token; empty
/// values mean software rendering.
#[pyfunction]
#[pyo3(signature = (width = 0, height = 0, render_node = String::new(), auto_gpu = String::new(), cursor_size = -1))]
fn ensure_wayland_display(
    py: Python<'_>,
    width: i32,
    height: i32,
    render_node: String,
    auto_gpu: String,
    cursor_size: i32,
) -> PyResult<()> {
    ensure_wayland_backend(py, width, height, render_node, auto_gpu, String::new(), cursor_size)
        .map(|_| ())
}

/// Stop every live capture (registered with atexit) before interpreter finalization.
///
/// The interpreter-teardown gate is set first so no detached thread may attach to a finalizing
/// interpreter, and the cursor callbacks (the X11 monitor's and a never-applied Wayland stash)
/// are dropped while the GIL is held. Every X11 capture's stop flag is set, and a live Wayland
/// capture is stopped over the command channel (the compositor thread clears its callback and
/// encoder on `StopCapture`). A brief grace sleep lets the stops be observed before Python
/// finalizes.
#[pyfunction]
fn _stop_all_captures(py: Python<'_>) {
    PY_SHUTDOWN.store(true, Ordering::Relaxed);
    *PENDING_CURSOR_CALLBACK.lock().unwrap() = None;
    crate::x11::cursor::shutdown();
    if let Some(slot) = LIVE_X11.get() {
        for c in slot.lock().unwrap().iter() {
            c.stop.store(true, Ordering::Relaxed);
        }
    }
    if let Some(slot) = WAYLAND_BACKEND.get() {
        if let Some(be) = slot.lock().unwrap().as_ref() {
            let _ = be.bind(py).borrow().stop_capture();
        }
    }
    WAYLAND_OWNER.store(0, Ordering::Relaxed);
    WAYLAND_CAPTURE_ALIVE.store(false, Ordering::Relaxed);
    py.detach(|| std::thread::sleep(Duration::from_millis(50)));
}

/// The `pixelflux` Python module: registers the exported classes and functions, and hooks
/// `_stop_all_captures` into `atexit` so every live capture is stopped before interpreter shutdown.
#[pymodule]
fn pixelflux(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WaylandBackend>()?;
    m.add_class::<StripeFrame>()?;
    m.add_class::<CaptureSettings>()?;
    m.add_class::<ScreenCapture>()?;
    // Feature probe for consumers: this build delivers X11 cursors via set_cursor_callback.
    m.add("X11_CURSOR_CALLBACK", true)?;
    m.add_function(wrap_pyfunction!(stripe_frame_from_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(ensure_wayland_display, m)?)?;
    m.add_function(wrap_pyfunction!(_stop_all_captures, m)?)?;
    if let Ok(atexit) = m.py().import("atexit") {
        let _ = atexit.call_method1("register", (m.getattr("_stop_all_captures")?,));
    }


    Ok(())
}

#[cfg(test)]
mod wl_frame_pool_tests {
    //! Invariants under test: try_begin is non-blocking and hands out a buffer ONLY while the
    //! publish slot is empty (so publish can never block the calloop); take blocks until a
    //! frame or shutdown; recycle/cancel return buffers for reuse; every published frame is
    //! observed exactly once and in order (the H.264 reference chain depends on it).
    use super::*;

    fn frame(id: usize, buf: Vec<u8>, n: u16) -> WlFrame {
        WlFrame {
            id,
            buf,
            frame_id: n,
            damage: Vec::new(),
            is_animated: false,
        }
    }

    #[test]
    fn begin_gated_on_slot_and_free_list() {
        let p = WlFramePool::new(2, 16);
        let (a, abuf) = p.try_begin().expect("first buffer");
        let (b, bbuf) = p.try_begin().expect("second buffer");
        assert_ne!(a, b);
        assert!(p.try_begin().is_none(), "free list exhausted");
        p.publish(frame(a, abuf, 0));
        p.cancel(b, bbuf);
        assert!(p.try_begin().is_none(), "slot occupied blocks begin");
        let f = p.take().expect("published frame");
        assert_eq!(f.frame_id, 0);
        p.recycle(f.id, f.buf);
        assert!(p.try_begin().is_some(), "drained slot re-enables begin");
    }

    #[test]
    fn frames_flow_in_order_and_buffers_recycle() {
        let p = Arc::new(WlFramePool::new(2, 4));
        let p2 = p.clone();
        let consumer = thread::spawn(move || {
            let mut seen = Vec::new();
            while let Some(f) = p2.take() {
                seen.push(f.frame_id);
                p2.recycle(f.id, f.buf);
            }
            seen
        });
        let mut published = 0u16;
        while published < 50 {
            if let Some((id, buf)) = p.try_begin() {
                p.publish(frame(id, buf, published));
                published += 1;
            } else {
                thread::sleep(Duration::from_micros(50));
            }
        }
        thread::sleep(Duration::from_millis(50));
        p.shutdown();
        let seen = consumer.join().unwrap();
        assert_eq!(seen, (0..50).collect::<Vec<u16>>(), "every frame, in order");
    }

    #[test]
    fn shutdown_unblocks_take() {
        let p = Arc::new(WlFramePool::new(1, 4));
        let p2 = p.clone();
        let t = thread::spawn(move || p2.take());
        thread::sleep(Duration::from_millis(30));
        p.shutdown();
        assert!(t.join().unwrap().is_none(), "take returns None on shutdown");
    }

    #[test]
    fn cancel_returns_buffer_for_reuse() {
        let p = WlFramePool::new(1, 8);
        let (id, buf) = p.try_begin().expect("buffer");
        assert!(p.try_begin().is_none());
        p.cancel(id, buf);
        assert!(p.try_begin().is_some(), "cancelled reservation reusable");
    }
}

#[cfg(test)]
mod auto_gpu_token_tests {
    //! Invariant: AUTO_GPU tokens match a card's kernel-reported identity —
    //! driver name exactly (no table), raw PCI vendor ID, devicetree compatible
    //! prefix literally, or a human vendor name via the small embedded aliases —
    //! so users may pass either "amd" or "amdgpu" (etc.) interchangeably.
    use super::{card_matches_token, CardIdentity};

    fn pci(driver: &str, vendor: u32) -> CardIdentity {
        CardIdentity { driver: driver.into(), pci_vendor: Some(vendor), compatibles: vec![] }
    }

    fn dt(driver: &str, compatibles: &[&str]) -> CardIdentity {
        CardIdentity {
            driver: driver.into(),
            pci_vendor: None,
            compatibles: compatibles.iter().map(|c| c.to_string()).collect(),
        }
    }

    #[test]
    fn driver_names_match_without_any_table() {
        assert!(card_matches_token("amdgpu", &pci("amdgpu", 0x1002)));
        assert!(card_matches_token("panfrost", &dt("panfrost", &["rockchip,rk3399-mali"])));
        assert!(card_matches_token("nouveau", &pci("nouveau", 0x10de)));
        assert!(!card_matches_token("i915", &pci("amdgpu", 0x1002)));
    }

    #[test]
    fn vendor_names_and_raw_ids_match_pci_identity() {
        let nv = pci("nouveau", 0x10de);
        assert!(card_matches_token("nvidia", &nv));
        assert!(card_matches_token("0x10de", &nv));
        assert!(card_matches_token("10de", &nv));
        assert!(!card_matches_token("amd", &nv));
        assert!(card_matches_token("ati", &pci("radeon", 0x1002)));
    }

    #[test]
    fn devicetree_prefixes_match_literally_and_via_aliases() {
        let mali = dt("panfrost", &["rockchip,rk3399-mali", "arm,mali-t860"]);
        assert!(card_matches_token("rockchip", &mali));
        assert!(card_matches_token("arm", &mali));
        assert!(card_matches_token("mali", &mali));
        let adreno = dt("msm", &["qcom,adreno-630", "qcom,adreno"]);
        assert!(card_matches_token("qcom", &adreno));
        assert!(card_matches_token("qualcomm", &adreno));
        assert!(card_matches_token("adreno", &adreno));
        assert!(!card_matches_token("brcm", &adreno));
        assert!(card_matches_token("videocore", &dt("v3d", &["brcm,bcm2711-v3d"])));
    }

    #[test]
    fn missing_identity_fields_never_false_match() {
        let bare = CardIdentity { driver: String::new(), pci_vendor: None, compatibles: vec![] };
        for t in ["nvidia", "amdgpu", "0x10de", "qcom"] {
            assert!(!card_matches_token(t, &bare));
        }
    }
}
