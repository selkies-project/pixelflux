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
//! encoding layer that dispatches to software (striped JPEG, and H.264 through the build's
//! software encoder: libx264 with the `gpl` feature, OpenH264 without) or hardware (NVENC,
//! VA-API) encoders based on the available GPU and operator settings.
//!
//! ## Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`encoders`] | Encoder backends: software H.264 (libx264 or OpenH264) / JPEG, NVENC, VA-API, watermark overlay |
//! | [`wayland`] | Headless Smithay compositor, cursor rendering |
//! | [`x11`] | X11/XShm capture loop, XFixes out-of-band cursor monitor, and stripe dispatch |
//! | [`pipeline`] | Frame-processing policy shared by both backends (send/QP/keyframe decisions) |
//! | [`recording_sink`] | Unix-socket H.264 fan-out for external recording |
//! | [`recorder`] | Built-in MP4 recorder (fMP4 muxer + Python/env/REST control surfaces) |
//! | [`computer_use`] | HTTP API for AI-agent desktop control (screenshots, input injection) |
//! | [`nvgpufilter`] | Multi-GPU NVENC device filtering via ioctl |
//! | [`webcam`] | Virtual camera: client webcam uplink decoded into a V4L2 device (interposer ring, v4l2loopback, PipeWire node) |
//!
//! ## Data flow
//!
//! ```text
//! Python  ──►  CaptureSettings  ──►  X11 / Wayland backend
//!                                         │
//!                                    frame pixels
//!                                         │
//!                                    ┌────┴────┐
//!                                    │ Encoder │  (NVENC / VAAPI / x264 or OpenH264 / JPEG)
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
use smithay::wayland::presentation::{PresentationState, Refresh};
use smithay::wayland::image_capture_source::{ImageCaptureSourceState, OutputCaptureSourceState};
use smithay::wayland::image_copy_capture::{CaptureFailureReason, ImageCopyCaptureState};
use smithay::desktop::utils::{send_frames_surface_tree, OutputPresentationFeedback};
use smithay::reexports::wayland_protocols::wp::presentation_time::server::wp_presentation_feedback;
use smithay::wayland::selection::wlr_data_control::DataControlState;
use smithay::wayland::selection::ext_data_control::DataControlState as ExtDataControlState;
use smithay::wayland::cursor_shape::CursorShapeManagerState;
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
                AsRenderElements, Element, RenderElement, Wrap,
            },
            gles::GlesRenderer,
            pixman::PixmanRenderer,
            sync::SyncPoint,
            Bind, ExportMem, Frame as _, ImportAll, ImportDma, ImportEgl, ImportMem,
            Renderer as _,
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
            generic::Generic, timer::{TimeoutAction, Timer},
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
    /// Cisco OpenH264 software H.264 encoder (BSD-licensed): the software encoder of a build
    /// without `gpl`, and always built for the test suite.
    #[cfg(any(feature = "openh264", test))]
    pub mod oh264;
    /// PNG watermark overlay composited onto frames before encoding.
    pub mod overlay;
    /// CPU-based striped H.264 (libx264 or OpenH264, by build) / JPEG encoder with per-stripe
    /// change detection.
    pub mod software;
    /// VA-API hardware H.264 encoder for Intel / AMD GPUs via FFmpeg.
    pub mod vaapi;

    #[cfg(not(any(feature = "gpl", feature = "openh264")))]
    compile_error!(
        "pixelflux needs a software H.264 encoder: enable the `gpl` feature (libx264, the default) or `openh264`."
    );

    /// The software H.264 encoder this build resolved to, fixed by the crate features: `"x264"`
    /// whenever `gpl` is on (libx264 wins even if `openh264` is also enabled), `"openh264"` for
    /// a GPL-free build. It is what the striped software path and the full-frame software fallback
    /// under NVENC/VA-API both encode with; exposed to Python as `pixelflux.SOFTWARE_H264_ENCODER`.
    #[cfg(feature = "gpl")]
    pub const SOFTWARE_H264_ENCODER: &str = "x264";
    #[cfg(not(feature = "gpl"))]
    pub const SOFTWARE_H264_ENCODER: &str = "openh264";

    /// Whether the build's software H.264 encoder carries a 4:4:4 (`video_fullcolor`) request:
    /// libx264 does (High 4:4:4, full range); OpenH264 is 4:2:0-only and encodes such a request
    /// 4:2:0. Every "is this software stream 4:4:4" decision reads it from here.
    pub const SOFTWARE_H264_FULLCOLOR: bool = cfg!(feature = "gpl");

    /// Damps visible quality "blinking": the number of consecutive frames a QP *increase* (a
    /// quality drop under sustained motion) must be requested before a fixed-QP encoder commits
    /// it. Moving the quantizer costs a codec re-open (VA-API CQP) or a full encoder rebuild
    /// (OpenH264), and either forces an IDR, so acting on every transient increase — and then
    /// reversing it as motion settles — would make the picture pulse. Quality *increases* (a
    /// lower QP, e.g. a paint-over refresh) apply at once and never wait. Shared so the two
    /// fixed-QP encoders cannot disagree about how long a drop must persist.
    pub(crate) const QP_HYSTERESIS_LIMIT: u32 = 60;

    /// Lowest H.264 level whose Annex-A Table A-1 limits admit a `width` x `height` stream at
    /// `fps`, as level_idc (41 = 4.1, 52 = 5.2, 62 = 6.2).
    ///
    /// A frame is charged two ways: **MaxFS**, its size in macroblocks, and **MaxMBPS**, that
    /// size times the frame rate. Advertising the lowest fitting level asks the least of a
    /// decoder, so clients that gate hardware decode on the level accept the widest range of
    /// streams. One shared ladder keeps two encoders from disagreeing on the same geometry;
    /// it starts at 4.1 and a caller needing more raises the floor itself. Above 6.2's limits
    /// there is no higher level, so it returns 62 as a best effort.
    pub(crate) fn min_h264_level(width: u32, height: u32, fps: u32) -> u32 {
        let mbs = (width as u64).div_ceil(16) * (height as u64).div_ceil(16);
        let mbps = mbs * fps.max(1) as u64;
        const LEVELS: [(u32, u64, u64); 8] = [
            (41, 8192, 245760),
            (42, 8704, 522240),
            (50, 22080, 589824),
            (51, 36864, 983040),
            (52, 36864, 2073600),
            (60, 139264, 4177920),
            (61, 139264, 8355840),
            (62, 139264, 16711680),
        ];
        for &(level, max_fs, max_mbps) in &LEVELS {
            if mbs <= max_fs && mbps <= max_mbps {
                return level;
            }
        }
        62
    }

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

    /// The `Colorspace:` field of a stream log line, from what the session negotiated rather than
    /// what was asked for: a hardware encoder can refuse 4:4:4, and only the software encoder
    /// carries it at full range. Shared so the X11 and Wayland logs describe an identical session
    /// identically.
    pub fn colorspace_desc(fullcolor: bool, software: bool) -> &'static str {
        match (fullcolor, software) {
            (true, true) => "I444 (Full Range)",
            (true, false) => "I444 (Limited Range)",
            _ => "I420 (Limited Range)",
        }
    }
}

/// Headless Wayland compositor and cursor rendering.
pub mod wayland;
/// Unix-socket H.264 recording fan-out for external capture tools.
pub mod recording_sink;
/// Built-in MP4 recorder: independent capture-to-file with Python/env/REST control.
pub mod recorder;
/// HTTP server implementing the Anthropic Computer Use spec for AI agent desktop control.
pub mod computer_use;
/// Frame-processing policy shared by the X11 and Wayland backends.
pub mod pipeline;
/// X11/XShm capture loop, stripe dispatch, and per-stripe change detection.
pub mod x11;
/// Multi-GPU NVENC device filtering via kernel ioctl.
pub mod nvgpufilter;

pub mod webcam;

pub use encoders::nvenc;
pub use encoders::software::StripeState;
pub use encoders::vaapi;

fn get_process_rss_bytes() -> usize {
    if let Ok(contents) = std::fs::read_to_string("/proc/self/statm")
        && let Some(rss_pages) = contents.split_whitespace().nth(1)
        && let Ok(pages) = rss_pages.parse::<usize>() {
                return pages * 4096;
            }
    0
}

/// Blocks actually allocated, not apparent size, so a sparse file is not reported as
/// memory the process took. Feeds the debug log line only.
fn shm_usage_in(dir: &str) -> u64 {
    use std::os::unix::fs::MetadataExt;
    let mut total_size = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                total_size += metadata.blocks() * 512;
            }
        }
    }
    total_size
}

fn get_shm_usage_bytes() -> u64 {
    shm_usage_in("/dev/shm")
}

use encoders::nvenc::NvencEncoder;
use encoders::overlay::OverlayState;
use encoders::software::MAX_STRIPE_CAPACITY;
use encoders::vaapi::VaapiEncoder;

use smithay::reexports::wayland_protocols_misc::zwp_virtual_keyboard_v1::server::zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1;

use wayland::cursor::{Cursor, CursorJob};
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
pub(crate) fn create_dmabuf_from_bo(bo: &BufferObject<()>) -> Dmabuf {
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
    pub debug_logging: bool,
    pub auto_adjust_screen_capture_size: bool,
    pub recording_socket: String,
    /// Wayland display of an EXTERNAL compositor to capture (host-capture mode);
    /// empty composites own clients as usual.
    pub wayland_host_display: String,
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
    /// (IDRs only on demand: client join / reset, recorder connect).
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
    pub cursor_size_cap: i32,
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
            cursor_size_cap: s.cursor_size_cap,
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
        s.cursor_size_cap = self.cursor_size_cap;
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
            debug_logging: false,
            auto_adjust_screen_capture_size: false,
            recording_socket: String::new(),
            wayland_host_display: String::new(),
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

    // These fields are public attributes any caller can set to anything an i32 or f64
    // holds. Clamping here keeps a dimension from sizing an absurd frame buffer, and a
    // quality outside turbojpeg's 1..=100 from failing set_quality on every stripe, which
    // would emit no JPEG at all.
    let sanitize_dim = |v: i32| -> i32 {
        if v <= 0 { 0 } else { v.min(MAX_CAPTURE_DIM) }
    };
    let sanitize_fps = |v: f64| -> f64 {
        if v.is_finite() && v > 0.0 { v.min(MAX_FPS) } else { DEFAULT_FPS }
    };
    let sanitize_scale = |v: f64| -> f64 {
        if v.is_finite() && v > 0.0 { v.min(MAX_SCALE) } else { 1.0 }
    };

    Ok(RustCaptureSettings {
        width: sanitize_dim(settings.getattr("capture_width")?.extract()?),
        height: sanitize_dim(settings.getattr("capture_height")?.extract()?),
        scale: sanitize_scale(scale),
        capture_x: settings.getattr("capture_x")?.extract()?,
        capture_y: settings.getattr("capture_y")?.extract()?,
        target_fps: sanitize_fps(settings.getattr("target_fps")?.extract()?),
        jpeg_quality: settings.getattr("jpeg_quality")?.extract::<i32>()?.clamp(1, 100),
        paint_over_jpeg_quality: settings.getattr("paint_over_jpeg_quality")?.extract::<i32>()?.clamp(1, 100),
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
            .unwrap_or_default(),
        wayland_host_display: settings
            .getattr("wayland_host_display")
            .ok()
            .and_then(|v| v.extract::<String>().ok())
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

/// One live output as `(id, x, y, width, height, scale, capturing)`. Sizes are physical
/// pixels and `(x, y)` is the layout offset.
pub type OutputDesc = (u32, i32, i32, i32, i32, f64, bool);

/// One mapped window as `(window_id, title, app_id, output_id, parked)`. A parked window
/// has no output yet — a nested session's spare screens.
pub type WindowDesc = (u32, String, String, u32, bool);

/// Control messages sent from the Python-facing methods to the capture thread.
///
/// Every interaction with a running capture crosses the thread boundary as one of these variants
/// over the command channel: starting and stopping, injecting keyboard / pointer input, swapping
/// the xkb keymap, serving the clipboard, changing live rate and per-frame tunables, and the
/// computer-use queries that read back the screen, cursor, and geometry.
pub enum ThreadCommand {
    /// Start (or in-place reconfigure) the capture bound to output `display_id`.
    /// `callback` is the Python per-frame delivery target; `None` starts an internal
    /// capture with no Python consumer (the built-in recorder taps the delivery layer).
    StartCapture { display_id: u32, callback: Option<Py<PyAny>>, settings: RustCaptureSettings },
    /// Stop the capture bound to output `display_id` (other displays keep running).
    StopCapture { display_id: u32 },
    /// Create an additional output: `WxH` physical pixels at fractional `scale`, mapped
    /// into the layout at offset `(x, y)`. Replies false when the id is taken/reserved or
    /// the GPU render target cannot be allocated.
    CreateOutput {
        id: u32,
        width: i32,
        height: i32,
        x: i32,
        y: i32,
        scale: f64,
        reply: std::sync::mpsc::Sender<bool>,
    },
    /// Destroy a secondary output: its capture ends, its windows relocate to the primary
    /// output. Replies false for the primary (id 0) or an unknown id.
    DestroyOutput { id: u32, reply: std::sync::mpsc::Sender<bool> },
    /// Remap an existing output (the primary included) to layout offset `(x, y)`: the
    /// Space mapping, the offsets used for absolute input injection and cursor
    /// compositing, and the windows placed on it all follow, and the output is damaged so
    /// the next frames render correctly. Replies false for an unknown id.
    RepositionOutput { id: u32, x: i32, y: i32, reply: std::sync::mpsc::Sender<bool> },
    /// Reply with every live output as `(id, x, y, width, height, scale, capturing)`.
    ListOutputs { reply: std::sync::mpsc::Sender<Vec<OutputDesc>> },
    /// Reply with how many displays this backend can back with real content: -1 when
    /// self-compositing (outputs are created on demand), the host compositor's output
    /// count in host-capture mode.
    OutputCapacity { reply: std::sync::mpsc::Sender<i64> },
    /// Move the window with the given id onto output `output_id` (fullscreened there).
    MoveWindowToOutput { window_id: u32, output_id: u32, reply: std::sync::mpsc::Sender<bool> },
    /// Reply with every mapped window as `(window_id, title, app_id, output_id)`.
    ListWindows { reply: std::sync::mpsc::Sender<Vec<WindowDesc>> },
    SetCursorCallback(Py<PyAny>),
    SetClipboardCallback(Py<PyAny>),
    /// Server-side clipboard offer: the compositor owns the selection and serves `data` as
    /// `mime` (plus text aliases) to pasting clients.
    SetClipboard { mime: String, data: Vec<u8> },
    KeyboardKey { scancode: u32, state: u32 },
    /// A whole ordered run of key events in one message. Typing a paste one event at a
    /// time costs a channel send and a calloop wake per event, which competes with the
    /// render loop on the same thread; the caller still decides the sequence.
    KeyboardKeys { events: Vec<(u32, u32)> },
    /// Set the seat's BASE keymap from a full XKB_KEYMAP_FORMAT_TEXT_V1 string. The
    /// compositor's keymap policy rebuilds on top: overlay binds are re-spliced onto the new
    /// base (same keycodes) and the combined keymap is applied in one swap.
    SetKeymapString(String),
    /// Set the seat's BASE layout from RMLVO names (empty strings = xkbcommon defaults);
    /// replies whether compilation succeeded. Overlay binds rebuild on top as for
    /// `SetKeymapString`.
    SetXkbLayout {
        rules: String,
        model: String,
        layout: String,
        variant: String,
        options: String,
        reply: std::sync::mpsc::Sender<bool>,
    },
    /// Resolve keysyms to `(keycode, level)` against the seat keymap, overlay-binding every
    /// keysym the base cannot produce — ONE keymap swap for the whole batch, and a keycode that
    /// is currently pressed is never recycled. `(0, 0)` marks an unbindable keysym. Serves
    /// computer-use only; selkies resolves its own keysyms and injects plain keycodes.
    BindKeysyms {
        keysyms: Vec<u32>,
        reply: std::sync::mpsc::Sender<Vec<(u32, u32)>>,
    },
    /// Bind explicit `(keycode, keysym)` pairs onto the CURRENT base keymap and deliver it,
    /// in one swap. The caller decides the assignment; this only assembles and applies, so the
    /// base is neither re-sent nor recompiled and its reverse map is not rebuilt. Ordered with
    /// key events on the one command channel, so the keys that need the new binds cannot
    /// overtake it — no reply is awaited, and the caller's loop is never blocked on the swap.
    SetKeymapOverlay { binds: Vec<(u32, u32)> },
    /// Debug/verification readback: currently pressed xkb keycodes plus the modifier state
    /// bitmask (1 ctrl, 2 shift, 4 alt, 8 logo, 16 caps, 32 num, 64 altgr, 128 level5).
    GetKeyboardState {
        reply: std::sync::mpsc::Sender<(Vec<u32>, u32)>,
    },
    /// Reply with the smithay keyboard's keymap as an XKB_KEYMAP_FORMAT_TEXT_V1 string so a
    /// consumer (selkies) can build its reverse keysym map from the IDENTICAL keymap.
    GetXkbKeymap { reply: std::sync::mpsc::Sender<String> },
    /// Ack once every previously queued command has been fully processed (the channel is
    /// FIFO). The atexit sweep sends StopCapture + Barrier and waits, so the interpreter never
    /// exits while the calloop thread is still mid-teardown (an NVENC/CUDA session drop racing
    /// process exit segfaults).
    Barrier { reply: std::sync::mpsc::Sender<()> },
    PointerMotion { x: f64, y: f64 },
    PointerRelativeMotion { dx: f64, dy: f64 },
    /// `btn` is an evdev `BTN_` code by contract (e.g. 272 = BTN_LEFT, 273 = BTN_RIGHT,
    /// 274 = BTN_MIDDLE, 0x113 = BTN_SIDE / 0x114 = BTN_EXTRA for back/forward) and is passed
    /// straight through to smithay's pointer.
    PointerButton { btn: u32, state: u32 },
    PointerAxis { x: f64, y: f64 },
    UpdateCursorConfig { render_on_framebuffer: bool },
    /// Recreate the cursor theme handles at a new pixel size — the calloop's compositing
    /// helper (the burned-in cursor) and, through its job channel, the `wl-cursor` worker's
    /// (named-cursor PNG delivery). Replies false for a non-positive size.
    SetCursorSize { size: i32, reply: std::sync::mpsc::Sender<bool> },
    /// On-demand keyframe request (client reconnect / decoder reset) for one display's
    /// capture: forces a send and an IDR even on a static screen.
    RequestIdr { display_id: u32 },
    /// Live rate-control change for one display's capture (parity with the X11 `rate_dirty`
    /// path). Each field is `None` when that dimension is unchanged.
    UpdateRate {
        display_id: u32,
        bitrate_kbps: Option<i32>,
        vbv_multiplier: Option<f64>,
        fps: Option<f64>,
    },
    /// Live per-frame tunables (quality / paint-over / streaming / cursor) for one
    /// display's capture, mirrored to its readback encode thread — no restart.
    UpdateTunables { display_id: u32, tunables: LiveTunables },
    /// One-shot PNG of one output's next rendered frame (0 = primary); an unknown
    /// display id replies with an error immediately.
    CuScreenshot { display_id: u32, resp: std::sync::mpsc::Sender<Result<Vec<u8>, String>> },
    CuCursorPosition { resp: std::sync::mpsc::Sender<(f64, f64)> },
    CuGetInfo { display_id: u32, resp: std::sync::mpsc::Sender<(i32, i32, f64)> },
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

/// Whether an encode node's driver name routes to NVENC: NVIDIA, or unknown (no readable
/// render-node sysfs entry, the usual shape of an NVIDIA container without /dev/dri), so the
/// NVENC attempt runs before the CPU fallback. Any other named driver routes to VA-API. Shared
/// by the X11 and Wayland paths so both pick the same encoder for the same node.
pub(crate) fn driver_selects_nvenc(encode_driver: &str) -> bool {
    encode_driver.is_empty() || encode_driver.contains("nvidia")
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
            } else if line.starts_with("OF_COMPATIBLE_") && !line.starts_with("OF_COMPATIBLE_N")
                && let Some(v) = line.split_once('=').map(|x| x.1) {
                    id.compatibles.push(v.trim().to_lowercase());
                }
        }
    }
    if id.pci_vendor.is_none() {
        id.pci_vendor = std::fs::read_to_string(device.join("vendor"))
            .ok()
            .and_then(|v| u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok());
    }
    if id.compatibles.is_empty()
        && let Ok(modalias) = std::fs::read_to_string(device.join("modalias")) {
            let modalias = modalias.trim();
            if let Some(rest) = modalias.strip_prefix("of:") {
                id.compatibles
                    .extend(rest.split('C').skip(1).map(|c| c.to_lowercase()));
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
        if let Some((_, ids)) = VENDOR_PCI_IDS.iter().find(|(n, _)| *n == token)
            && ids.contains(&vid) {
                return true;
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
        if let Some(t) = token
            && !card_matches_token(t, &read_card_identity(&path.join("device"))) {
                continue;
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
            let (gg, _) = self.cv.wait_timeout(g, WL_POOL_WAKE_QUANTUM).unwrap();
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
/// The encode thread's bounded idle wait in `WlFramePool::take`.
const WL_POOL_WAKE_QUANTUM: Duration = Duration::from_millis(20);
/// How long a reconfigured output holds its frames while its clients answer the new size.
/// It is measured from the start, which rebuilds the encode path before there is any frame
/// to hold: the first one arrives roughly 300 ms later, so a window shorter than that
/// expires before it can suppress anything and the hold does nothing at all. This one
/// outlasts the rebuild and a client's own repaint, and is the whole delay an output that
/// no client ever draws on pays before its content reaches the viewer regardless.
const WL_CONTENT_HOLD: Duration = Duration::from_millis(500);

/// Upper bounds for Python-supplied capture geometry: keeps a hostile or buggy setting from
/// turning into a multi-GB `vec![]` allocation that would abort the process. 16384 covers every
/// real display wall; the fps/scale bounds keep timing math finite and sane.
const MAX_CAPTURE_DIM: i32 = 16384;
const MAX_FPS: f64 = 1000.0;
/// How often frame callbacks go out while nothing is being captured. Fast enough that a
/// client blocked on one keeps making progress, slow enough that an unwatched session is
/// not drawing frames nobody asked for.
const IDLE_FRAME_INTERVAL: Duration = Duration::from_millis(250);
const DEFAULT_FPS: f64 = 60.0;
const MAX_SCALE: f64 = 8.0;
/// Wheel v120 units per unit of injected scroll value: selkies sends 10 per notch, and one
/// notch is 120, so both the seat's v120 and the host virtual pointer's discrete steps derive
/// from the same value.
pub(crate) const SCROLL_V120_PER_UNIT: f64 = 12.0;

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
    /// Output/display id this encode loop serves; keys the recorder's delivery-layer tap.
    display_id: u32,
    /// GLES readback is RGBA; the pixman framebuffer is BGRA. Selects CSC + encoder input kind.
    use_gpu: bool,
    /// Attempt a HW (NVENC/VAAPI) readback session before falling back to the CPU encoders.
    try_gpu: bool,
    /// HW session handed back by the previous encode thread; reconfigured in place and
    /// reused when still compatible, sparing the stream a session rebuild.
    prior: Option<GpuEncoder>,
    /// The outgoing encode thread of a capture being restarted, joined here — off the
    /// calloop — before this thread builds anything. Everything the predecessor did
    /// therefore precedes everything this thread does, and its readback hardware session is
    /// inherited rather than rebuilt.
    predecessor: Option<std::thread::JoinHandle<Option<GpuEncoder>>>,
    /// Weak, so the calloop's teardown owns the sink's lifetime: a strong handle held here
    /// could outlive the capture and unlink a successor's freshly bound socket path.
    recording_sink: Option<std::sync::Weak<crate::recording_sink::RecordingSink>>,
    deliver_tx: std::sync::mpsc::SyncSender<Vec<EncodedStripe>>,
    controls: Arc<WlEncodeControls>,
    stats: Arc<WlEncodeStats>,
}

/// Build the readback-mode hardware encoder on the thread that will own and drive it.
///
/// The result is a hardware `GpuEncoder` (NVENC or VAAPI) consuming host NV12/YUV444 via
/// `encode_raw`, or `None` for the software path — striped or full-frame JPEG/H.264 through the
/// build's software encoder (`SOFTWARE_H264_ENCODER`), where `encode_cpu` builds its own per-stripe
/// state. Selection follows the settings and the effective encode device — an auto index below
/// zero resolves to device 0: NVENC on an NVIDIA driver, VAAPI on any other GPU, except a 4:4:4
/// full-color request which VAAPI cannot do reliably and so falls through to the CPU. When a
/// compatible NVENC session is handed over from the previous encode thread it is reconfigured in
/// place (milliseconds) rather than rebuilt, which would stall the stream. The EGL display is
/// always null here: readback mode never imports dmabufs.
fn build_readback_encoders(
    settings: &RustCaptureSettings,
    try_gpu: bool,
    prior: Option<GpuEncoder>,
) -> Option<GpuEncoder> {
    if !try_gpu {
        return None;
    }
    let encode_driver = get_gpu_driver(settings.encode_node_index.max(0));
    println!(
        "[Wayland] Encode Node Index: {} | Driver: {}",
        settings.encode_node_index.max(0), encode_driver
    );
    if driver_selects_nvenc(&encode_driver) {
        if let Some(GpuEncoder::Nvenc(mut enc)) = prior {
            match enc.reconfigure_resolution(settings) {
                Ok(()) => {
                    println!("[Wayland] NVENC session reconfigured in place.");
                    return Some(GpuEncoder::Nvenc(enc));
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
                return Some(GpuEncoder::Nvenc(e));
            }
            Err(e) => eprintln!(
                "[Wayland] Failed to init NVENC: {}. Falling back to CPU ({}).",
                e,
                encoders::SOFTWARE_H264_ENCODER
            ),
        }
    } else {
        println!("[Wayland] Initializing Unified VAAPI Encoder...");
        match VaapiEncoder::new(settings) {
            Ok(e) => {
                println!(
                    "[Wayland] VAAPI Encoder initialized successfully ({}).",
                    if e.is_fullcolor() { "4:4:4" } else { "4:2:0" }
                );
                return Some(GpuEncoder::Vaapi(e));
            }
            Err(e) => eprintln!(
                "[Wayland] Failed to init VAAPI: {}. Falling back to CPU ({}).",
                e,
                encoders::SOFTWARE_H264_ENCODER
            ),
        }
    }
    None
}

/// Planar CSC target for a readback VA-API session: YUV444 needs three full planes, NV12 one
/// plus a half-height interleaved chroma plane. Empty when no such session consumes it — NVENC
/// takes the packed readback rows and converts in hardware.
fn hw_plane_buffer(width: i32, height: i32, fullcolor: bool, planar: bool) -> Vec<u8> {
    if !planar {
        return Vec::new();
    }
    let n = (width * height) as usize;
    vec![0u8; if fullcolor { n * 3 } else { n * 3 / 2 }]
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
/// 2. **Dispatch by encoder**: a hardware session runs `decide_hw_fullframe`, then hands only the
///    frames actually being encoded to the session — NVENC takes the packed rows as they are (RGBA
///    vs BGRA source chosen by the renderer) through `encode_cpu_packed` and converts in hardware,
///    VA-API gets them colorspace-converted into the reused NV12/YUV444 buffer and `encode_raw`;
///    otherwise `encode_cpu` runs the
///    software path with compositor damage — JPEG, or H.264 through the build's software encoder,
///    striped or full-frame. The software H.264 path keeps an infinite GOP, forcing an IDR only on
///    an explicit request or the configured interval, and an explicit request also forces a full
///    JPEG resend for joiners.
/// 3. **Recycle then deliver**: the capture buffer is recycled BEFORE delivery so a slow consumer
///    never pins one, then the stripes go to the delivery thread through a single-slot `send` whose
///    blocking is the backpressure that overlaps delivery with the next render + encode.
///
/// On exit the hardware session is handed back to the calloop so a restart can reuse it in place
/// when the new settings stay compatible (a plain `StopCapture` just drops it).
fn wayland_encode_loop(pool: &WlFramePool, cfg: WlEncodeConfig) -> Option<GpuEncoder> {
    crate::boost_thread_priority(-10);
    let mut settings = cfg.settings;
    let inherited = cfg.predecessor.and_then(|h| h.join().ok().flatten());
    let mut video_encoder =
        build_readback_encoders(&settings, cfg.try_gpu, cfg.prior.or(inherited));
    if cfg.try_gpu && video_encoder.is_none() {
        println!(
            "[Wayland] Decision: No GPU Encoder available -> Using CPU Software Encoding ({}).",
            encoders::SOFTWARE_H264_ENCODER
        );
    }
    let n_stripes = wayland_stripe_count(&settings, video_encoder.is_some());
    cfg.stats.n_stripes.store(n_stripes as u32, Ordering::Relaxed);
    *cfg.stats.desc.lock().unwrap() = encoder_desc(&settings, video_encoder.as_ref(), false);
    log_stream_settings(&settings, n_stripes, video_encoder.as_ref());

    let width = settings.width;
    let height = settings.height;
    let mut stripes: Vec<StripeState> = Vec::with_capacity(MAX_STRIPE_CAPACITY);
    // Smoothed number of stripes carrying the encode budget (see stripe_rate_control).
    let mut stripes_carrying: f32 = 1.0;
    let mut hw_state = StripeState::default();
    // The CSC target follows the session's own chroma, not the request: a VA-API device that
    // refused 4:4:4 encodes 4:2:0, and sizing this from the request would hand it a buffer laid
    // out for planes it does not read.
    let mut hw_fullcolor = encoder_fullcolor(video_encoder.as_ref(), &settings);
    let mut nv12_buffer: Vec<u8> = hw_plane_buffer(
        width,
        height,
        hw_fullcolor,
        matches!(video_encoder, Some(GpuEncoder::Vaapi(_))),
    );
    // Mid-stream recovery state for the readback hardware session, mirroring the zero-copy tick.
    let mut hw_error_streak: u32 = 0;
    let mut hw_rebuilt = false;

    while let Some(mut f) = pool.take() {
        if cfg.controls.tunables_dirty.swap(false, Ordering::Acquire)
            && let Some(t) = cfg.controls.tunables.lock().unwrap().take() {
                t.apply_to(&mut settings);
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
                Some(GpuEncoder::Vaapi(enc)) => {
                    if let Err(e) = enc.reconfigure_rate(&settings) {
                        // The failed re-open left no codec context: the next encode fails,
                        // and a full streak makes that failure run the recovery ladder at
                        // once instead of after a window of dead frames.
                        eprintln!("[wl-encode] VAAPI rate reconfigure failed: {e}");
                        hw_error_streak = HW_ERROR_RECOVERY_THRESHOLD - 1;
                    }
                }
                None => {}
            }
        }

        // A recorder connecting counts as a request, so the decision layer sends a
        // decodable frame even when the screen is static. A sink whose capture has been
        // torn down is already gone, and its last frames are not recorded.
        let recording_sink = cfg.recording_sink.as_ref().and_then(|w| w.upgrade());
        let requested_idr = cfg.controls.force_idr.swap(false, Ordering::Relaxed)
            || recording_sink.as_ref().is_some_and(|s| s.should_force_idr());

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
                let force_idr = decision.force_idr;
                let outcome = match encoder {
                    // NVENC takes the readback rows as they are — BGRA from the pixman
                    // framebuffer, RGBA from a GLES readback — through a pinned upload and
                    // its hardware CSC, the same hand-over the X11 capture uses, so no
                    // colour conversion runs on this thread.
                    GpuEncoder::Nvenc(enc) => enc.encode_cpu_packed(
                        &f.buf,
                        (w * 4) as usize,
                        cfg.use_gpu,
                        f.frame_id as u64,
                        decision.target_qp,
                        force_idr,
                    ),
                    GpuEncoder::Vaapi(enc) => {
                        // VA-API's raw entry point takes planar YUV, converted here with the
                        // matrix and range its session declares: BT.709, limited (the range
                        // its own convert targets as well). A failed conversion leaves the
                        // planes holding the previous frame, so the encode is skipped rather
                        // than submitting stale content, and the failure feeds the same
                        // recovery ladder as an encode failure: both leave the session
                        // delivering nothing, and only the rebuild or the demote below can get
                        // the stream moving again.
                        let matrix = YuvStandardMatrix::Bt709;
                        let range = YuvRange::Limited;
                        let y_size = (w * h) as usize;
                        let csc = if hw_fullcolor {
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
                            if cfg.use_gpu {
                                yuv::rgba_to_yuv444(&mut planar_image, &f.buf, w * 4, range, matrix, YuvConversionMode::Fast)
                            } else {
                                yuv::bgra_to_yuv444(&mut planar_image, &f.buf, w * 4, range, matrix, YuvConversionMode::Fast)
                            }
                        } else {
                            let (y_plane, uv_plane) = nv12_buffer.split_at_mut(y_size);
                            let mut planar_image = YuvBiPlanarImageMut {
                                y_plane: BufferStoreMut::Borrowed(y_plane),
                                y_stride: w,
                                uv_plane: BufferStoreMut::Borrowed(uv_plane),
                                uv_stride: w,
                                width: w,
                                height: h,
                            };
                            if cfg.use_gpu {
                                yuv::rgba_to_yuv_nv12(&mut planar_image, &f.buf, w * 4, range, matrix, YuvConversionMode::Fast)
                            } else {
                                yuv::bgra_to_yuv_nv12(&mut planar_image, &f.buf, w * 4, range, matrix, YuvConversionMode::Fast)
                            }
                        };
                        match csc {
                            Err(e) => Err(format!(
                                "{} CSC failed: {e:?}",
                                if hw_fullcolor { "YUV444" } else { "NV12" }
                            )),
                            Ok(()) => enc.encode_raw(
                                &nv12_buffer,
                                f.frame_id as u64,
                                decision.target_qp,
                                force_idr,
                            ),
                        }
                    }
                };
                match outcome {
                    Ok(data) => {
                        hw_error_streak = 0;
                        hw_rebuilt = false;
                        if !data.is_empty() {
                            out.push(EncodedStripe {
                                data: Arc::new(data),
                                data_type: 2,
                                stripe_y_start: 0,
                                stripe_height: height,
                                frame_id: f.frame_id as i32,
                            });
                        }
                    }
                    Err(e) => {
                        // One line per recovery window: a session failing at frame rate would
                        // otherwise write a line per frame for the life of the capture.
                        if hw_error_streak % HW_ERROR_RECOVERY_THRESHOLD == 0 {
                            eprintln!("[wl-encode] HW encode error: {e}");
                        }
                        hw_error_streak = hw_error_streak.saturating_add(1);
                        if hw_error_streak >= HW_ERROR_RECOVERY_THRESHOLD {
                            // The readback session persistently fails after having worked:
                            // rebuild it once with the startup selection, then demote to the
                            // software encoders. A session whose encodes keep failing still
                            // constructs, so the rebuild only counts as recovery until the
                            // next streak; otherwise the stream would rebuild in a loop.
                            hw_error_streak = 0;
                            let try_gpu = !hw_rebuilt;
                            if try_gpu {
                                eprintln!("[wl-encode] rebuilding readback HW encoder after repeated encode errors.");
                            } else {
                                eprintln!(
                                    "[wl-encode] readback HW encoder unrecoverable; demoting to software encoding ({}).",
                                    encoders::SOFTWARE_H264_ENCODER
                                );
                            }
                            // The broken session is released before its replacement is opened:
                            // the failure it recovers from is usually device memory pressure,
                            // and holding both at once is what would make the rebuild fail too.
                            drop(video_encoder.take());
                            video_encoder = build_readback_encoders(&settings, try_gpu, None);
                            hw_rebuilt = try_gpu && video_encoder.is_some();
                            hw_fullcolor = encoder_fullcolor(video_encoder.as_ref(), &settings);
                            nv12_buffer = hw_plane_buffer(
                                width,
                                height,
                                hw_fullcolor,
                                matches!(video_encoder, Some(GpuEncoder::Vaapi(_))),
                            );
                            cfg.controls.force_idr.store(true, Ordering::Relaxed);
                            let n = wayland_stripe_count(&settings, video_encoder.is_some());
                            cfg.stats.n_stripes.store(n as u32, Ordering::Relaxed);
                            *cfg.stats.desc.lock().unwrap() =
                                encoder_desc(&settings, video_encoder.as_ref(), false);
                            log_stream_settings(&settings, n, video_encoder.as_ref());
                        }
                    }
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
                &mut stripes_carrying,
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
        // An unserved request stays armed: on an infinite GOP an IDR lost to an encode
        // error or skip would never self-heal.
        if requested_idr && out.is_empty() {
            cfg.controls.force_idr.store(true, Ordering::Relaxed);
        }
        if !out.is_empty() {
            cfg.stats.frames.fetch_add(1, Ordering::Relaxed);
            cfg.stats.stripes.fetch_add(out.len() as u32, Ordering::Relaxed);
            if let Some(ref socket) = recording_sink {
                socket.write_frame(&out, settings.height);
            }
            crate::recorder::wayland_tap(cfg.display_id, &out);
            let _ = cfg.deliver_tx.send(out);
        }
    }
    if settings.debug_logging {
        println!(
            "[Wayland] Encode thread exiting (hw={}, stripes={}).",
            video_encoder.is_some(),
            stripes.len()
        );
    }
    video_encoder
}

/// The chroma format a session actually carries, which is not always the one requested: NVENC
/// takes 4:4:4 whenever asked, VA-API only when the driver and FFmpeg build both carry it, and the
/// software path (`None`) only when the build's software encoder does (`SOFTWARE_H264_FULLCOLOR`:
/// libx264 yes, OpenH264 no). Every consumer of "is this stream 4:4:4" — the readback buffer
/// sizing, the CSC branch, and both log lines — reads it from here so they cannot disagree.
fn encoder_fullcolor(video_encoder: Option<&GpuEncoder>, settings: &RustCaptureSettings) -> bool {
    match video_encoder {
        Some(GpuEncoder::Vaapi(enc)) => enc.is_fullcolor(),
        Some(GpuEncoder::Nvenc(_)) => settings.video_fullcolor,
        None => settings.video_fullcolor && encoders::SOFTWARE_H264_FULLCOLOR,
    }
}

/// Compose the encoder half of the 1 s debug log line (backend, colorspace, frame mode) for
/// whichever thread owns the encoders.
fn encoder_desc(
    settings: &RustCaptureSettings,
    video_encoder: Option<&GpuEncoder>,
    zero_copy: bool,
) -> String {
    if settings.output_mode == 0 {
        return format!("JPEG Q:{}", settings.jpeg_quality);
    }
    let copy_mode = if zero_copy { "ZeroCopy" } else { "Readback" };
    let backend = match video_encoder {
        Some(GpuEncoder::Nvenc(_)) => format!("NVENC ({})", copy_mode),
        Some(GpuEncoder::Vaapi(_)) => format!("VAAPI ({})", copy_mode),
        None => format!("CPU {}", encoders::SOFTWARE_H264_ENCODER),
    };
    let is_444 = encoder_fullcolor(video_encoder, settings);
    let cs_str = if is_444 { "CS_IN:I444" } else { "CS_IN:I420" };
    // Only the software encoder carries 4:4:4 at full range; NVENC's hardware CSC is
    // limited-range whatever the chroma format.
    let range_str = if is_444 && video_encoder.is_none() { "FR" } else { "LR" };
    let frame_str = if video_encoder.is_some() || settings.video_fullframe {
        "FF"
    } else {
        "Striped"
    };
    format!("H264 ({}) {} {} {} CRF:{}", backend, cs_str, range_str, frame_str, settings.video_crf)
}

/// How many horizontal stripes a frame is split into, for the settings line and the stats.
///
/// A full-frame session (`fullframe_encoder`: a HW encoder, or a forced `video_fullframe`) is
/// one contiguous H.264 stream and so a single stripe. Everything else asks the encoder's own
/// rule, so what is reported is what is encoded.
fn wayland_stripe_count(settings: &RustCaptureSettings, fullframe_encoder: bool) -> usize {
    crate::encoders::software::stripe_count(
        settings.height,
        settings.output_mode,
        fullframe_encoder || settings.video_fullframe,
    )
}

/// One-shot "Stream settings active" line, printed by the thread that owns the encoders
/// once the selection is final (calloop for zero-copy, encode thread for readback).
fn log_stream_settings(
    settings: &RustCaptureSettings,
    n_stripes: usize,
    video_encoder: Option<&GpuEncoder>,
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
        let encoder_type = match video_encoder {
            Some(GpuEncoder::Nvenc(_)) => "NVENC",
            Some(GpuEncoder::Vaapi(_)) => "VAAPI",
            None => encoders::SOFTWARE_H264_ENCODER,
        };
        log_msg.push_str(&format!(" | Mode: H264 ({})", encoder_type));

        if video_encoder.is_some() || settings.video_fullframe {
            log_msg.push_str(" FullFrame");
        } else {
            log_msg.push_str(" Striped");
        }

        if settings.video_streaming_mode {
            log_msg.push_str(" Streaming");
        }

        if settings.video_cbr_mode {
            log_msg.push_str(&format!(" | CBR {}", settings.video_bitrate_kbps));
        } else {
            log_msg.push_str(&format!(" | CRF: {}", settings.video_crf));
            if settings.video_bitrate_kbps > 0 {
                log_msg.push_str(&format!(" | VBV: {} kbps", settings.video_bitrate_kbps));
            }
        }

        if settings.use_paint_over_quality {
            log_msg.push_str(&format!(
                " | PaintOver CRF: {} (Burst: {}f)",
                settings.video_paintover_crf, settings.video_paintover_burst_frames
            ));
        }

        let is_actually_444 = encoder_fullcolor(video_encoder, settings);
        log_msg.push_str(&format!(
            " | Colorspace: {}",
            encoders::colorspace_desc(is_actually_444, video_encoder.is_none())
        ));
    }

    log_msg.push_str(&format!(
        " | Damage Thresh: {}f | Damage Dur: {}f",
        settings.damage_block_threshold, settings.damage_block_duration
    ));

    println!("{}", log_msg);
}

/// Tear down a capture's encode/delivery threads and pools, returning both join handles as
/// `(deliver, encode)` for the caller to place. NEITHER is joined here: this runs on the
/// calloop thread, which also dispatches Wayland clients and input, and either thread can be
/// parked behind a consumer still inside Python. A restart hands each handle to its
/// successor thread, which joins it off the event loop — the encode successor inheriting the
/// readback hardware session, the deliver successor keeping stripes in capture order across
/// the restart — and a plain stop hands both to the reaper. The zero-copy session (if any)
/// is left on the capture for the caller to reuse or drop.
fn teardown_capture(
    cap: &mut wayland::frontend::WlCapture,
) -> (
    Option<std::thread::JoinHandle<()>>,
    Option<std::thread::JoinHandle<Option<GpuEncoder>>>,
) {
    if let Some(p) = cap.encode_pool.take() {
        p.shutdown();
    }
    if let Some(flag) = cap.deliver_discard.take() {
        flag.store(true, Ordering::Relaxed);
    }
    let encode_join = cap.encode_join.take();
    if let Some(tx) = cap.deliver_tx.take() {
        drop(tx);
    }
    let join = cap.deliver_join.take();
    cap.pending_hw_delivery = None;
    cap.pending_hw_damage = false;
    // The last strong reference: dropping it here, before a following start binds its own
    // sink, is what keeps the outgoing sink's unlink from stripping the successor's socket
    // path. The encode thread holds only a Weak.
    cap.recording_sink = None;
    (join, encode_join)
}

/// Drop a host-capture session whose compositor connection has ended, stopping every
/// capture it fed: their displays leave the alive set, so `is_capturing` turns false and the
/// consumer rebuilds them (the next start reconnects), and no input is written to a dead
/// connection. The same signal X11 gives when its capture thread ends.
fn reap_dead_host(state: &mut AppState) {
    if !state.host.as_ref().is_some_and(|h| !h.alive()) {
        return;
    }
    eprintln!("[HostCapture] host compositor connection lost; captures stop until restarted.");
    let ids: Vec<u32> = state
        .output_nodes
        .iter()
        .filter(|n| n.capture.is_some())
        .map(|n| n.id)
        .collect();
    for id in ids {
        stop_capture_on_display(state, id);
        // stop_capture_on_display cleared this display's outcome; overwrite it with the
        // host-death reason so capture_state reports why the pipeline went down (input is
        // black-holed to a dead connection until a fresh start rebuilds against a new one).
        set_wayland_capture_err(
            id,
            Some("host compositor connection lost; capture stopped".to_string()),
        );
    }
    state.host = None;
    // Nothing will answer the requests still in flight: their geometry readers get
    // what the outputs show now.
    let pending: Vec<_> = state.host_layout_pending.drain().collect();
    for (id, p) in pending {
        answer_geometry_waiters(state, id, p.geometry_waiters);
    }
}

/// Stop the capture bound to `display_id`, leaving the output (and every other display's
/// capture) running.
fn stop_capture_on_display(state: &mut AppState, display_id: u32) {
    let Some(idx) = state.node_idx_for_id(display_id) else { return };
    if let Some(mut cap) = state.output_nodes[idx].capture.take() {
        println!("[Wayland] Capture loop stopped (display {display_id}).");
        cap.video_encoder = None;
        let (join, encode_join) = teardown_capture(&mut cap);
        state.deliver_reaper.extend(join);
        state.encode_reaper.extend(encode_join);
    }
    wayland_alive().lock().unwrap().remove(&display_id);
    set_wayland_capture_err(display_id, None);
    if let Some(p) = state.host_layout_pending.remove(&display_id) {
        answer_geometry_waiters(state, display_id, p.geometry_waiters);
    }
}

/// The geometry `get_realized_geometry` reports for `display_id`: the live capture's size
/// and scale, else the output's current mode, else zeros for an unknown display.
fn realized_geometry(state: &AppState, display_id: u32) -> (i32, i32, f64) {
    state
        .node_idx_for_id(display_id)
        .map(|idx| {
            let node = &state.output_nodes[idx];
            match node.capture.as_ref() {
                Some(c) => (c.settings.width, c.settings.height, c.settings.scale),
                None => node
                    .output
                    .current_mode()
                    .map(|m| {
                        (
                            m.size.w,
                            m.size.h,
                            node.output.current_scale().fractional_scale(),
                        )
                    })
                    .unwrap_or((0, 0, 0.0)),
            }
        })
        .unwrap_or((0, 0, 0.0))
}

/// Answer geometry readers parked behind a host layout request with what `display_id`
/// captures now.
fn answer_geometry_waiters(
    state: &AppState,
    display_id: u32,
    waiters: Vec<std::sync::mpsc::Sender<(i32, i32, f64)>>,
) {
    if waiters.is_empty() {
        return;
    }
    let info = realized_geometry(state, display_id);
    for w in waiters {
        let _ = w.send(info);
    }
}

/// What the host's verdict on a layout request means for a capture configured at `want`:
/// `Some(size)` when it has to be re-sized to the mode the host kept (the request was
/// declined and the host runs a different size), `None` when nothing changes — the host
/// applied the request, already runs that size, or its mode is unknown (the capture keeps
/// gating on the size it asked for).
fn host_layout_resolution(
    realized: bool,
    want: (i32, i32),
    current: Option<(i32, i32)>,
) -> Option<(i32, i32)> {
    if realized {
        return None;
    }
    current.filter(|&c| c != want)
}

/// Settle the host layout requests answered since the last tick. One the host applied
/// needs nothing: the capture was configured for that size and its frames flow as the
/// host switches. One the host kept its own mode against (declined, no layout
/// management, no answer by the deadline) re-sizes the capture to that mode through the
/// same in-place reconfigure a resize takes, so it never gates on a size the host will
/// not produce, and the refusal becomes the capture's caveat. Geometry readers parked
/// behind a request answer once it is settled, with the size actually captured.
fn reconcile_host_layouts(state: &mut AppState) {
    if state.host_layout_pending.is_empty() {
        return;
    }
    let ids: Vec<u32> = state.host_layout_pending.keys().copied().collect();
    for id in ids {
        // An earlier iteration's restart may have reaped the host (and every request).
        let Some(pending) = state.host_layout_pending.get(&id) else { continue };
        let verdict = match state.host.as_ref() {
            Some(host) => match host.layout_outcome(pending.epoch) {
                None => continue,
                Some(realized) => {
                    host_layout_resolution(realized, pending.want, host.current_output_size(id))
                }
            },
            None => None,
        };
        let Some(pending) = state.host_layout_pending.remove(&id) else { continue };
        if let Some((rw, rh)) = verdict {
            let (w, h) = pending.want;
            let restart = state
                .node_idx_for_id(id)
                .and_then(|idx| state.output_nodes[idx].capture.as_ref())
                .map(|cap| (cap.callback.clone(), cap.settings.clone()));
            if let Some((cb, mut settings)) = restart {
                settings.width = rw;
                settings.height = rh;
                // H.264 even-masks its dimensions, so an odd host mode is followed as closely
                // as the encoder can; asking again would only loop.
                let followed = if settings.output_mode == 1 { (rw & !1, rh & !1) } else { (rw, rh) };
                if followed != (w, h) {
                    eprintln!(
                        "[HostCapture] host kept {rw}x{rh} for display {id} ({w}x{h} declined); capturing at that size."
                    );
                    start_capture_on_display(state, id, cb, settings);
                    let refusal = format!("host kept {rw}x{rh} ({w}x{h} declined)");
                    let own = wayland_capture_err().lock().unwrap().get(&id).cloned();
                    set_wayland_capture_err(
                        id,
                        Some(match own {
                            Some(e) => format!("{refusal}; {e}"),
                            None => refusal,
                        }),
                    );
                }
            }
        }
        // Settled either way: the restart, if any, runs at the size the host has, so the
        // readers need not wait for the host to acknowledge it again.
        answer_geometry_waiters(state, id, pending.geometry_waiters);
    }
}

/// Start (or in-place reconfigure) the capture bound to output `display_id`: reprogram the
/// output's mode/scale/refresh, size the render targets, fullscreen the display's windows at
/// the new logical size, resolve the encode path (zero-copy vs readback), and spawn the
/// delivery (and readback-mode encode) threads. The single-display behavior of the former
/// global StartCapture is preserved exactly for display 0.
/// Bring up the readback encode path (pixman readback → pool → encode thread) for a
/// capture whose zero-copy session is absent or just died. Mirrors the start-capture
/// bootstrap: u64::MAX content generations mark every pool slot stale so each one is
/// read back before its first publish, whatever the damage says.
fn bootstrap_readback_pool(
    cap: &mut wayland::frontend::WlCapture,
    display_id: u32,
    use_gpu: bool,
    try_gpu: bool,
    prior: Option<GpuEncoder>,
    predecessor: Option<std::thread::JoinHandle<Option<GpuEncoder>>>,
) {
    let Some(deliver_tx) = cap.deliver_tx.clone() else {
        return;
    };
    let settings = cap.settings.clone();
    let pool = Arc::new(WlFramePool::new(
        WL_POOL_SURFACES,
        (settings.width.max(0) as usize) * (settings.height.max(0) as usize) * 4,
    ));
    cap.pool_last_render = vec![0; WL_POOL_SURFACES];
    cap.render_seq = 0;
    cap.pool_content_gen = vec![u64::MAX; WL_POOL_SURFACES];
    cap.content_gen = 0;
    let c = &cap.encode_controls;
    c.bitrate_kbps.store(settings.video_bitrate_kbps, Ordering::Relaxed);
    c.vbv_mult_milli.store(
        (settings.video_vbv_multiplier * 1000.0).round() as i32,
        Ordering::Relaxed,
    );
    c.fps_milli.store(
        (settings.target_fps.max(1.0) * 1000.0) as u64,
        Ordering::Relaxed,
    );
    let cfg = WlEncodeConfig {
        settings: settings.clone(),
        display_id,
        use_gpu,
        try_gpu,
        prior,
        predecessor,
        recording_sink: cap.recording_sink.as_ref().map(Arc::downgrade),
        deliver_tx,
        controls: cap.encode_controls.clone(),
        stats: cap.encode_stats.clone(),
    };
    let pool2 = pool.clone();
    cap.encode_join = Some(
        thread::Builder::new()
            .name(format!("wl-encode-{display_id}"))
            .spawn(move || wayland_encode_loop(&pool2, cfg))
            .expect("failed to spawn wl-encode thread"),
    );
    cap.encode_pool = Some(pool);
}

/// Rebuild a broken zero-copy hardware session with the startup construction (driver
/// match, EGL display hand-over, the same chroma negotiation). `None` means unrecoverable —
/// the caller demotes to readback.
fn rebuild_zerocopy_encoder(
    cap: &wayland::frontend::WlCapture,
    state: &mut AppState,
) -> Option<GpuEncoder> {
    let settings = &cap.settings;
    let encode_driver = get_gpu_driver(settings.encode_node_index.max(0));
    if driver_selects_nvenc(&encode_driver) {
        let egl_display = state
            .gles_renderer
            .as_ref()
            .map(|r| r.egl_context().display().get_display_handle().handle)
            .unwrap_or(std::ptr::null());
        NvencEncoder::new(settings, egl_display)
            .ok()
            .map(GpuEncoder::Nvenc)
    } else {
        VaapiEncoder::new(settings).ok().map(GpuEncoder::Vaapi)
    }
}

/// Consecutive encode failures before a hardware path recovers (~0.5s at 60fps): a hiccup
/// outlasts it, anything longer starts to look like a dead session. Shared by every hardware
/// encode path (Wayland zero-copy, Wayland readback, X11) so recovery timing matches.
pub(crate) const HW_ERROR_RECOVERY_THRESHOLD: u32 = 30;

fn start_capture_on_display(
    state: &mut AppState,
    display_id: u32,
    cb: Option<Arc<Py<PyAny>>>,
    mut settings: RustCaptureSettings,
) {
    use smithay::wayland::fractional_scale::with_fractional_scale;

    // The cursor worker outlives individual captures, so the starting settings have to
    // reach it here — same point X11 applies the cap — or it keeps the previous capture's.
    let _ = state.cursor_tx.send(CursorJob::SetSizeCap(settings.cursor_size_cap));

    // Fresh attempt: drop any prior outcome so a stale caveat cannot read as this start's.
    set_wayland_capture_err(display_id, None);

    let Some(node_idx) = state.node_idx_for_id(display_id) else {
        eprintln!("[Wayland] StartCapture: no output with display id {display_id}.");
        set_wayland_capture_err(
            display_id,
            Some(format!("no output with display id {display_id}")),
        );
        return;
    };
    let mut node = state.output_nodes.remove(node_idx);
    // Geometry readers parked behind a layout request this start supersedes: they ride
    // on to this start's request, or answer at its end if no host is involved.
    let mut geometry_waiters = state
        .host_layout_pending
        .remove(&display_id)
        .map(|p| p.geometry_waiters)
        .unwrap_or_default();

    if state.auto_gpu_selected && settings.encode_node_index < -1
        && let Some(idx_str) = state.render_node_path.strip_prefix("/dev/dri/renderD")
        && let Ok(idx) = idx_str.parse::<i32>() {
                settings.encode_node_index = idx - 128;
            }

    if settings.output_mode == 1 {
        settings.width &= !1;
        settings.height &= !1;
    }

    // Tear down this display's previous capture first; its hardware sessions are the
    // reuse candidates below (zero-copy inline, readback via the encode config).
    let mut prior_zero_copy: Option<GpuEncoder> = None;
    let mut prior_encode_join: Option<std::thread::JoinHandle<Option<GpuEncoder>>> = None;
    let mut prior_deliver_join: Option<std::thread::JoinHandle<()>> = None;
    if let Some(mut old) = node.capture.take() {
        prior_zero_copy = old.video_encoder.take();
        (prior_deliver_join, prior_encode_join) = teardown_capture(&mut old);
    }

    // Bind only after the old capture is gone: its sink unlinks the socket path in
    // Drop, which would strip a fresh bind's filesystem name and leave every later
    // recorder connect with ENOENT.
    let recording_sink = crate::recording_sink::RecordingSink::try_bind(&settings.recording_socket);

    // Host-capture mode: connect on first use. The display's mode is requested from
    // the host further down, without waiting for an answer; a host that keeps its own
    // mode (refusal, or no layout management — KWin) has this capture re-sized to it
    // when the verdict arrives (`reconcile_host_layouts`), the same way a failed GBM
    // resize falls back to the live mode.
    let host_capture = !settings.wayland_host_display.is_empty();
    reap_dead_host(state);
    if host_capture && state.host.is_none() {
        // Capture buffers come from the same render node the encoder imports
        // from, resolved via its live fd (the path string is not retained in
        // auto mode); each capture thread opens its own device handle.
        let gbm_path = if state.use_gpu {
            state.gbm_device.as_ref().and_then(|dev| {
                use std::os::fd::{AsFd as _, AsRawFd as _};
                let fd = dev.as_fd().as_raw_fd();
                std::fs::read_link(format!("/proc/self/fd/{fd}")).ok()
            })
        } else {
            None
        };
        match crate::wayland::host::HostSession::connect(&settings.wayland_host_display, gbm_path) {
            Ok(h) => {
                println!(
                    "[HostCapture] capturing host compositor '{}' ({} outputs).",
                    settings.wayland_host_display,
                    h.output_count()
                );
                state.host = Some(h);
            }
            Err(e) => {
                eprintln!(
                    "[HostCapture] connect '{}' failed: {e}",
                    settings.wayland_host_display
                );
                set_wayland_capture_err(
                    display_id,
                    Some(format!(
                        "host compositor '{}' connect failed ({e}); capturing locally",
                        settings.wayland_host_display
                    )),
                );
            }
        }
    }
    // A failed connect leaves this display compositing locally, so its frames follow the
    // local renderer again.
    let host_capture = host_capture && state.host.is_some();
    if host_capture && let Some(host) = &state.host {
        // The node's layout offset rides along so the host's heads mirror
        // selkies' union layout (input coordinates already assume it).
        host.set_layout(display_id, node.pos.0, node.pos.1);
    }

    {
        // Never panic the compositor thread: an output momentarily without a current
        // mode falls back to the requested geometry so the reconfigure below is a
        // no-op for size/refresh instead of unwrap-panicking.
        let target_refresh = (settings.target_fps * 1000.0).round() as i32;
        let (current_w, current_h, current_refresh) = match node.output.current_mode() {
            Some(m) => (m.size.w, m.size.h, m.refresh),
            None => (settings.width, settings.height, target_refresh),
        };
        let current_scale = node.output.current_scale().fractional_scale();

        if current_w != settings.width
            || current_h != settings.height
            || (current_scale - settings.scale).abs() > 0.001
            || current_refresh != target_refresh
        {
            // Allocate the GPU backing for the new dimensions BEFORE committing
            // anything: if the driver refuses (VRAM exhaustion, dimensions it will
            // not back), the whole reconfigure is skipped and the previous mode +
            // buffers stay live. A failed resize must degrade to "no resize", never
            // panic the compositor thread.
            let mut new_offscreen = None;
            let mut gbm_resize_failed = false;
            if state.use_gpu
                && let Some(gbm) = state.gbm_device.as_mut() {
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
            if gbm_resize_failed {
                // The mode commit below is skipped wholesale, so the rest of this
                // StartCapture (encoder setup, stored settings) must see the
                // dimensions actually live.
                set_wayland_capture_err(
                    display_id,
                    Some(format!(
                        "GPU buffer resize to {}x{} refused; kept {current_w}x{current_h}",
                        settings.width, settings.height
                    )),
                );
                settings.width = current_w;
                settings.height = current_h;
                settings.scale = current_scale;
                settings.target_fps = current_refresh as f64 / 1000.0;
            } else {
                println!(
                    "[Wayland] Configuring Output {} ({}): {}x{} @ {:.2} FPS (Scale {:.2})",
                    display_id, node.output.name(),
                    settings.width, settings.height, settings.target_fps, settings.scale
                );
                let new_mode = OutputMode {
                    size: (settings.width, settings.height).into(),
                    refresh: target_refresh,
                };
                node.output.change_current_state(
                    Some(new_mode),
                    Some(Transform::Normal),
                    Some(OutputScale::Fractional(settings.scale)),
                    Some(Point::from(node.pos)),
                );
                node.output.set_preferred(new_mode);
                // Capture clients allocate to our announced size; a stale size means
                // every frame they submit from now on fails buffer validation.
                for cs in state
                    .copy_sessions
                    .iter()
                    .filter(|cs| cs.output.upgrade().as_ref() == Some(&node.output))
                {
                    if let Some(c) = wayland::frontend::output_capture_constraints(
                        &node.output,
                        state.gles_renderer.as_ref(),
                        &state.render_node_path,
                    ) {
                        cs.session.update_constraints(c);
                    }
                }

                let pixel_count =
                    (settings.width.max(0) as usize) * (settings.height.max(0) as usize);
                node.frame_buffer = vec![0u8; pixel_count * 4];
                node.target_seeded = false;

                if let Some(off) = new_offscreen.take() {
                    node.offscreen_buffer = Some(off);
                }
            }
        }

        let scale = settings.scale.max(0.1);
        let logical_width = (settings.width as f64 / scale).round() as i32;
        let logical_height = (settings.height as f64 / scale).round() as i32;

        for window in state.space.elements() {
            if wayland::frontend::window_output_id(window) != display_id {
                continue;
            }
            // A parked screen is tagged for this display but composited on none: it
            // holds PARKED_LOGICAL_SIZE until `create_output` gives it one. Resizing it
            // here would double the session's coordinate space onto a screen nobody
            // watches, which is where a window that centres itself then lands.
            if wayland::frontend::window_meta(window)
                .is_some_and(|meta| meta.parked.load(Ordering::Relaxed))
            {
                continue;
            }
            if let Some(surface) = window.wl_surface() {
                node.output.enter(&surface);
                with_states(&surface, |states| {
                    smithay::wayland::compositor::send_surface_state(
                        &surface, states, scale.ceil() as i32, Transform::Normal,
                    );
                    with_fractional_scale(states, |fs| {
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
    let gpu_intent = settings.output_mode == 1 && !use_cpu_explicit;
    if use_cpu_explicit {
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

    let mut video_encoder: Option<GpuEncoder> = None;
    if gpu_intent && state.use_gpu && !different_gpu {
        let encode_driver = get_gpu_driver(settings.encode_node_index.max(0));
        println!(
            "[Wayland] Encode Node Index: {} | Driver: {}",
            settings.encode_node_index.max(0), encode_driver
        );

        if driver_selects_nvenc(&encode_driver) {
            let reused = match prior_zero_copy.as_mut() {
                Some(GpuEncoder::Nvenc(enc)) => match enc.reconfigure_resolution(&settings) {
                    Ok(()) => {
                        println!("[Wayland] NVENC session reconfigured in place.");
                        true
                    }
                    Err(e) => {
                        eprintln!("[Wayland] NVENC in-place reconfigure unavailable ({e}); rebuilding.");
                        false
                    }
                },
                _ => false,
            };
            if reused {
                video_encoder = prior_zero_copy.take();
            } else {
                prior_zero_copy = None;
                println!("[Wayland] Nvidia Encoder detected. Initializing NVENC...");
                let egl_display = if let Some(renderer) = state.gles_renderer.as_ref() {
                    renderer.egl_context().display().get_display_handle().handle
                } else {
                    std::ptr::null()
                };

                match NvencEncoder::new(&settings, egl_display) {
                    Ok(encoder) => {
                        video_encoder = Some(GpuEncoder::Nvenc(encoder));
                        println!("[Wayland] NVENC Encoder initialized successfully.");
                    }
                    Err(e) => {
                        eprintln!(
                            "[Wayland] Failed to init NVENC: {}. Falling back to CPU.",
                            e
                        );
                        set_wayland_capture_err(
                            display_id,
                            Some(format!("NVENC init failed ({e}); using CPU encode")),
                        );
                    }
                }
            }
        } else {
            prior_zero_copy = None;
            println!("[Wayland] Initializing Unified VAAPI Encoder...");
            match VaapiEncoder::new(&settings) {
                Ok(encoder) => {
                    println!(
                        "[Wayland] VAAPI Encoder initialized successfully ({}).",
                        if encoder.is_fullcolor() { "4:4:4" } else { "4:2:0" }
                    );
                    video_encoder = Some(GpuEncoder::Vaapi(encoder));
                }
                Err(e) => {
                    eprintln!(
                        "[Wayland] Failed to init VAAPI: {}. Falling back to CPU.",
                        e
                    );
                    set_wayland_capture_err(
                        display_id,
                        Some(format!("VAAPI init failed ({e}); using CPU encode")),
                    );
                }
            }
        }
    }
    drop(prior_zero_copy);

    if different_gpu {
        println!("[Wayland] Decision: Rendering and Encoding GPUs differ -> Forcing Readback (CPU path for pixels).");
    }
    if video_encoder.is_none() {
        println!("[Wayland] Decision: Readback path (encode thread) active.");
    } else if !different_gpu {
        println!("[Wayland] Decision: Zero-Copy path active.");
    }

    // Point this display's host capture thread at the size the encoder was just
    // configured for and ask the host for that mode. The compositor keeps running (CU,
    // clipboard callbacks, input fallbacks) but its renderer is bypassed. The buffer
    // type follows the consumer settled on above: dmabufs only for a zero-copy encoder,
    // shm frames for every CPU path (use_cpu, JPEG, readback). The host's
    // answer is not waited for here — this thread carries input and every other
    // display — but polled by the render tick, which re-sizes this capture should the
    // host keep a different mode; frames gate until the sizes agree.
    if host_capture && let Some(host) = &state.host {
        let epoch = host.start_capture(
            display_id,
            settings.width,
            settings.height,
            video_encoder.is_some(),
            settings.capture_cursor,
        );
        state.host_layout_pending.insert(
            display_id,
            wayland::frontend::PendingHostLayout {
                epoch,
                want: (settings.width, settings.height),
                geometry_waiters: std::mem::take(&mut geometry_waiters),
            },
        );
    }

    if recording_sink.is_some() && settings.output_mode == 0 {
        eprintln!(
            "[recording_sink] WARNING: recording_socket is set but output_mode is JPEG (0). \
             The recording socket requires a single H.264 stream. Please set output_mode=1 \
             on the Python CaptureSettings to produce a recordable output."
        );
    }

    // Every display's capture composites its own watermark, uploaded at this output's
    // scale and placed against this output's frame dimensions.
    let watermark_output_scale = node.output.current_scale().fractional_scale();
    node.overlay_state
        .load_watermark(&settings.watermark_path, watermark_output_scale);
    if display_id == 0 {
        state.settings = settings.clone();
        if state.cursor_callback_set
            && let Some(icon) = state.current_cursor_icon.clone() {
                state.send_cursor_image(&icon);
            }
    }
    state.render_cursor_on_framebuffer = settings.capture_cursor;

    let mut cap = wayland::frontend::WlCapture {
        settings: settings.clone(),
        callback: cb.clone(),
        video_encoder,
        vaapi_state: StripeState::default(),
        recording_sink,
        deliver_tx: None,
        deliver_join: None,
        deliver_discard: None,
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
        frame_counter: 0,
        pending_force_idr: false,
        needs_full_render: true,
        last_tick: None,
        hw_error_streak: 0,
        hw_rebuilt: false,
    };

    {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<EncodedStripe>>(1);
        let discard = Arc::new(AtomicBool::new(false));
        let thread_discard = discard.clone();
        let predecessor = prior_deliver_join;
        // With no Python callback (internal recorder-owned capture) the delivery thread
        // only drains the channel: the recorder already consumed the frames at the
        // delivery-layer tap, upstream of this per-consumer handoff.
        let join = thread::spawn(move || {
            // The predecessor capture's deliver thread finishes first, off the
            // event loop: encoded stripes reach Python in capture order across
            // a reconfigure, and a stale pre-teardown stripe can never land
            // after this capture's first frame.
            if let Some(handle) = predecessor {
                let _ = handle.join();
            }
            match cb {
                Some(cb) => {
                    crate::boost_thread_priority(-10);
                    while let Ok(stripes) = rx.recv() {
                        if thread_discard.load(Ordering::Relaxed)
                            || PY_SHUTDOWN.load(Ordering::Relaxed) { continue; }
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
                }
                None => while rx.recv().is_ok() {},
            }
        });
        cap.deliver_tx = Some(tx);
        cap.deliver_discard = Some(discard);
        cap.deliver_join = Some(join);
    }

    if cap.video_encoder.is_none() {
        bootstrap_readback_pool(
            &mut cap,
            display_id,
            // Host frames land in the pool as BGRA whatever the local renderer is; only
            // a GLES readback of our own compositing produces RGBA.
            state.use_gpu && !host_capture,
            gpu_intent && (!state.use_gpu || different_gpu),
            None,
            prior_encode_join.take(),
        );
    } else {
        cap.encode_stats.n_stripes.store(1, Ordering::Relaxed);
        *cap.encode_stats.desc.lock().unwrap() =
            encoder_desc(&settings, cap.video_encoder.as_ref(), true);
        log_stream_settings(&settings, 1, cap.video_encoder.as_ref());
    }
    // A zero-copy start has no successor encode thread to inherit the outgoing readback
    // session, so the outgoing thread is reaped instead.
    state.encode_reaper.extend(prior_encode_join);
    // Force the keyframe unconditionally: the damage tracker and offscreen buffer
    // stay warm across stop/start, so a restarted capture on a static screen
    // otherwise produces no damage, no first frame, and no IDR in either path.
    cap.request_idr();

    node.capture = Some(cap);
    // The start reprogrammed this output, and until a client answers at the new size the
    // compositor paints its clear colour over whatever the client does not cover — a
    // freshly created output, whose session window is still parked at a placeholder size,
    // is covered by none of it. Those frames are held rather than streamed as a blank
    // screen; the deadline releases an output no client ever draws on.
    node.content_hold_until = Some(Instant::now() + WL_CONTENT_HOLD);
    state.output_nodes.insert(node_idx, node);
    wayland_alive().lock().unwrap().insert(display_id);
    answer_geometry_waiters(state, display_id, geometry_waiters);
}

/// One output's render + capture tick: composite the elements overlapping this output
/// (positions made output-local by subtracting its layout origin), track damage, feed the
/// display's own encode path, and answer a pending screenshot on the primary. Returns true
/// when the tick was skipped because this display's encode pool was still busy (the caller
/// then retries shortly instead of waiting a full frame interval).
/// Stamp the watermark element onto `target` in place — no clear, so the
/// captured content underneath stays — returning the draw's sync point for the
/// encoder to wait on.
fn draw_host_watermark(
    renderer: &mut GlesRenderer,
    overlay: &crate::encoders::overlay::OverlayState,
    target: &mut Dmabuf,
    size: (i32, i32),
) -> Result<SyncPoint, String> {
    let elem = overlay
        .get_watermark_element(renderer)
        .ok_or("watermark element unavailable")?;
    let mut fb = renderer.bind(target).map_err(|e| format!("bind: {e:?}"))?;
    let mut frame = renderer
        .render(&mut fb, (size.0, size.1).into(), Transform::Normal)
        .map_err(|e| format!("render: {e:?}"))?;
    let dst = elem.geometry(1.0.into());
    let local = Rectangle::from_size(dst.size);
    elem.draw(&mut frame, elem.src(), dst, &[local], &[], None)
        .map_err(|e| format!("draw: {e:?}"))?;
    frame.finish().map_err(|e| format!("finish: {e:?}"))
}

/// Re-compose `src` (the retained host frame) plus the watermark into `target`:
/// the path a moving watermark needs, since re-drawing over the same retained
/// buffer would leave trails.
fn compose_host_watermark(
    renderer: &mut GlesRenderer,
    overlay: &crate::encoders::overlay::OverlayState,
    src: &Dmabuf,
    target: &mut Dmabuf,
    size: (i32, i32),
) -> Result<SyncPoint, String> {
    let elem = overlay
        .get_watermark_element(renderer)
        .ok_or("watermark element unavailable")?;
    let tex = renderer
        .import_dmabuf(src, None)
        .map_err(|e| format!("import: {e:?}"))?;
    let mut fb = renderer.bind(target).map_err(|e| format!("bind: {e:?}"))?;
    let full: Rectangle<i32, Physical> = Rectangle::from_size((size.0, size.1).into());
    let mut frame = renderer
        .render(&mut fb, (size.0, size.1).into(), Transform::Normal)
        .map_err(|e| format!("render: {e:?}"))?;
    frame
        .render_texture_from_to(
            &tex,
            Rectangle::from_size((size.0 as f64, size.1 as f64).into()),
            full,
            &[full],
            // Opaque: the capture format's undefined alpha must not blend, it
            // would leave the target's previous content (and stamped
            // watermarks) underneath.
            &[full],
            Transform::Normal,
            1.0,
            None,
            &[],
        )
        .map_err(|e| format!("texture: {e:?}"))?;
    let dst = elem.geometry(1.0.into());
    let local = Rectangle::from_size(dst.size);
    elem.draw(&mut frame, elem.src(), dst, &[local], &[], None)
        .map_err(|e| format!("draw: {e:?}"))?;
    frame.finish().map_err(|e| format!("finish: {e:?}"))
}

fn warn_once_host_watermark(e: &str) {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!("[HostCapture] watermark compositing failed: {e}");
    }
}

/// Copy full rows from `src` (tight `width*4` stride) into an Argb8888/Xrgb8888 shm
/// buffer, honoring the destination's own offset and stride.
fn copy_rows_into_shm(
    buffer: &smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer,
    src: &[u8],
    width: i32,
    height: i32,
) -> Result<(), String> {
    use smithay::wayland::shm::with_buffer_contents_mut;
    let src_stride = (width.max(0) as usize) * 4;
    if src.len() < src_stride * (height.max(0) as usize) {
        return Err("source smaller than advertised".into());
    }
    with_buffer_contents_mut(buffer, |ptr, len, spec| {
        if spec.width < width || spec.height < height {
            return Err("shm buffer smaller than the output".to_string());
        }
        let dst_stride = spec.stride as usize;
        let offset = spec.offset as usize;
        if dst_stride < src_stride || len < offset + dst_stride * (height as usize) {
            return Err("shm stride or length mismatch".to_string());
        }
        for y in 0..height as usize {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(y * src_stride),
                    ptr.add(offset + y * dst_stride),
                    src_stride,
                );
            }
        }
        Ok(())
    })
    .map_err(|e| format!("{e:?}"))?
}

/// Complete parked ext-image-copy-capture frames for this output from the frame the
/// render pass just produced. A dmabuf target is filled by one GPU blit from the
/// output's composited buffer; an shm target by one readback (GLES) or one row copy
/// (pixman). With no damage, a session that has already received content keeps its
/// frame parked, so a static screen costs capture clients nothing.
fn service_copy_frames(
    state: &mut AppState,
    node: &mut wayland::frontend::OutputNode,
    width: i32,
    height: i32,
    damage_rects: &[Rectangle<i32, Physical>],
) {
    use smithay::backend::renderer::{buffer_type, Blit, BufferType, ExportMem, TextureFilter};
    use smithay::utils::Buffer as BufferCoords;

    if state.copy_sessions.is_empty() {
        return;
    }
    let time = state.clock.now();
    for i in 0..state.copy_sessions.len() {
        {
            let cs = &state.copy_sessions[i];
            if cs.output.upgrade().as_ref() != Some(&node.output)
                || cs.pending.is_none()
                || (damage_rects.is_empty() && cs.delivered_once)
            {
                continue;
            }
        }
        let first = !state.copy_sessions[i].delivered_once;
        let frame = state.copy_sessions[i].pending.take().unwrap();
        let buffer = frame.buffer();
        let full = Rectangle::<i32, Physical>::new((0, 0).into(), (width, height).into());
        let result: Result<(), String> = match buffer_type(&buffer) {
            Some(BufferType::Dma) => (|| {
                let renderer = state.gles_renderer.as_mut().ok_or("no GLES renderer")?;
                let mut client = smithay::wayland::dmabuf::get_dmabuf(&buffer)
                    .map_err(|e| e.to_string())?
                    .clone();
                let (_bo, offscreen) = node
                    .offscreen_buffer
                    .as_mut()
                    .ok_or("no composited buffer")?;
                let src = renderer.bind(offscreen).map_err(|e| format!("{e:?}"))?;
                let mut dst = renderer.bind(&mut client).map_err(|e| format!("{e:?}"))?;
                // ready() promises readable contents, and implicit dmabuf sync cannot
                // be relied on cross-process on every driver: wait out the blit fence
                // (microseconds) before declaring the frame done.
                renderer
                    .blit(&src, &mut dst, full, full, TextureFilter::Linear)
                    .map_err(|e| format!("{e:?}"))?
                    .wait()
                    .map_err(|_| "blit fence interrupted".to_string())?;
                Ok(())
            })(),
            Some(BufferType::Shm) => (|| {
                if let Some(renderer) = state.gles_renderer.as_mut() {
                    let (_bo, offscreen) = node
                        .offscreen_buffer
                        .as_mut()
                        .ok_or("no composited buffer")?;
                    let fb = renderer.bind(offscreen).map_err(|e| format!("{e:?}"))?;
                    let mapping = renderer
                        .copy_framebuffer(
                            &fb,
                            Rectangle::new((0, 0).into(), (width, height).into()),
                            Fourcc::Argb8888,
                        )
                        .map_err(|e| format!("{e:?}"))?;
                    let data = renderer.map_texture(&mapping).map_err(|e| format!("{e:?}"))?;
                    copy_rows_into_shm(&buffer, data, width, height)
                } else {
                    copy_rows_into_shm(&buffer, &node.frame_buffer, width, height)
                }
            })(),
            _ => Err("unsupported buffer type".into()),
        };
        match result {
            Ok(()) => {
                let damage: Option<Vec<Rectangle<i32, BufferCoords>>> = if first {
                    None
                } else {
                    Some(
                        damage_rects
                            .iter()
                            .map(|r| {
                                Rectangle::new(
                                    (r.loc.x, r.loc.y).into(),
                                    (r.size.w, r.size.h).into(),
                                )
                            })
                            .collect(),
                    )
                };
                frame.success(Transform::Normal, damage, time);
                state.copy_sessions[i].delivered_once = true;
            }
            Err(e) => {
                eprintln!("[Wayland] copy-capture frame failed: {e}");
                frame.fail(CaptureFailureReason::Unknown);
            }
        }
    }
}

/// Answer the frame callbacks of a surface-backed cursor. Xwayland and libwayland-cursor
/// clients wait for the cursor surface's frame callback before attaching their next
/// sprite, so a cursor surface that never gets one stops updating after its first sprite.
fn send_cursor_frame(state: &AppState, output: &Output, time: impl Into<Duration>) {
    if let Some(CursorImageStatus::Surface(surface)) = &state.current_cursor_icon {
        send_frames_surface_tree(surface, output, time, Some(Duration::ZERO), |_, _| Some(output.clone()));
    }
}

/// Hotspot of a surface-backed cursor, in the surface's logical coordinates (zero when the
/// client set none). The sprite is placed at pointer - hotspot, like named and X11 cursors.
fn cursor_surface_hotspot(
    surface: &smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
) -> Point<i32, smithay::utils::Logical> {
    with_states(surface, |states| {
        states
            .data_map
            .get::<std::sync::Mutex<smithay::input::pointer::CursorImageAttributes>>()
            .and_then(|attrs| attrs.lock().ok().map(|guard| guard.hotspot))
            .unwrap_or_default()
    })
}

fn render_node_tick(
    state: &mut AppState,
    node: &mut wayland::frontend::OutputNode,
) -> bool {
    let take_screenshot = state
        .pending_screenshot
        .as_ref()
        .is_some_and(|(id, _)| *id == node.id);
    let copy_frame_wanted = state.copy_frame_pending_for(&node.output);
    if node.capture.is_none() && !take_screenshot && !copy_frame_wanted {
        return false;
    }

    // Per-display frame pacing under the one shared timer (which fires at the fastest
    // active capture's rate).
    if let Some(cap) = node.capture.as_ref()
        && !take_screenshot {
            let fps = cap.settings.target_fps.max(1.0);
            if let Some(last) = cap.last_tick
                && last.elapsed().as_secs_f64() < (1.0 / fps) * 0.9 {
                    return false;
                }
        }

    let output = node.output.clone();
    let origin: Point<i32, smithay::utils::Logical> = node.pos.into();
    let output_scale_val = output.current_scale().fractional_scale();
    let (width, height) = match node.capture.as_ref() {
        Some(c) => (c.settings.width, c.settings.height),
        None => output
            .current_mode()
            .map(|m| (m.size.w, m.size.h))
            .unwrap_or((0, 0)),
    };
    if width <= 0 || height <= 0 {
        return false;
    }
    if node.frame_buffer.len() < (width as usize) * (height as usize) * 4 {
        node.frame_buffer = vec![0u8; (width as usize) * (height as usize) * 4];
    }
    let logical_w = (width as f64 / output_scale_val).round();
    let logical_h = (height as f64 / output_scale_val).round();

    // A reconfigured or freshly created output composites its clear colour wherever a client
    // has not yet answered the new size; publishing is held until one covers the output so
    // that grey never reaches the stream. Compositing continues, so the frame callbacks the
    // waited-on clients redraw on keep flowing and the hold cannot deadlock on itself. Host
    // capture streams the host compositor's own frames and has no such gap.
    let hold_frame = match node.content_hold_until {
        Some(deadline)
            if state.host.is_none()
                && Instant::now() < deadline
                && !wayland::frontend::output_content_covers(
                    &state.space,
                    node.id,
                    logical_w,
                    logical_h,
                ) =>
        {
            true
        }
        Some(_) => {
            node.content_hold_until = None;
            false
        }
        None => false,
    };

    // A recorder connecting counts as an IDR request, kept armed across skipped ticks.
    if let Some(cap) = node.capture.as_mut()
        && cap
            .recording_sink
            .as_ref()
            .map(|s| s.should_force_idr())
            .unwrap_or(false)
        {
            cap.request_idr();
        }
    let requested_idr = node.capture.as_ref().map(|c| c.pending_force_idr).unwrap_or(false);
    // A client keyframe request lands on the hardware path's atomic (RequestIdr sets
    // it whenever an encode pool exists); host mode consults it — without consuming —
    // to decide whether a static screen must re-encode its retained frame.
    let hw_idr_pending = node
        .capture
        .as_ref()
        .map(|c| c.encode_controls.force_idr.load(Ordering::Relaxed))
        .unwrap_or(false);
    let want_idr_for_host = requested_idr || hw_idr_pending;

    let mut pool_slot: Option<(usize, Vec<u8>)> = None;
    if !hold_frame
        && let Some(cap) = node.capture.as_ref()
        && let Some(ref pool) = cap.encode_pool {
            pool_slot = pool.try_begin();
            if pool_slot.is_none() {
                return true;
            }
        }

    let loc_enum = node
        .capture
        .as_ref()
        .map(|c| c.settings.watermark_location_enum)
        .unwrap_or(state.settings.watermark_location_enum);
    node.overlay_state.update_position(width, height, loc_enum);

    if let Some(cap) = node.capture.as_mut() {
        cap.last_tick = Some(Instant::now());
    }

    // The cursor is composited only on the output the pointer is on, at that output's
    // scale; its position is output-local.
    let pointer_local: Option<Point<f64, smithay::utils::Logical>> = state
        .seat
        .get_pointer()
        .map(|p| p.current_location())
        .and_then(|pos| {
            let rect = Rectangle::<f64, smithay::utils::Logical>::new(
                origin.to_f64(),
                (logical_w, logical_h).into(),
            );
            if rect.contains(pos) {
                Some(pos - origin.to_f64())
            } else {
                None
            }
        });

    let mut render_success = false;
    let mut render_sync = None;
    let mut damage_rects: Vec<Rectangle<i32, Physical>> = Vec::new();
    let needs_full = node.capture.as_ref().map(|c| c.needs_full_render).unwrap_or(!node.target_seeded);

    // Host-capture mode: the host compositor already blitted this display's frame
    // into one of our buffers (screencopy); adopt it in place of compositing.
    let host_mode = state.host.as_ref().map(|h| h.has_output_for(node.id)).unwrap_or(false);
    if state.host.is_some() && !host_mode {
        // No host output backs this display (start_capture already warned):
        // produce nothing rather than the compositor's own empty content.
        if let Some((id, buf)) = pool_slot.take()
            && let Some(cap) = node.capture.as_ref()
            && let Some(ref pool) = cap.encode_pool {
                    pool.cancel(id, buf);
                }
        return false;
    }
    // Dmabuf handed to the GPU encoder in host mode (from the new or retained frame).
    let mut host_enc_dmabuf: Option<Dmabuf> = None;
    // Host software frames arrive BGRA, so anything reading this display's frame buffer
    // back has to know it is not the GLES readback's RGBA.
    let mut host_cpu_frame = false;
    if host_mode {
        const RETAINED_OK: u8 = 0;
        const RETAINED_NONE: u8 = 1;
        // The host produced the buffer type this display's consumer cannot take; each
        // direction has its own recovery below.
        const RETAINED_CPU_FRAME: u8 = 2;
        const RETAINED_GPU_FRAME: u8 = 3;
        let host_idx = node.id;
        let gpu_encoder = node
            .capture
            .as_ref()
            .map(|c| c.video_encoder.is_some())
            .unwrap_or(false);
        // The session steps out of `state` while frames are adopted so the
        // renderer can composite the watermark / serve screenshot readbacks.
        let host = state.host.take().unwrap();
        // Stale-geometry rejection: a frame captured before a mode change is
        // useless at the new size (the CPU path would blit old-pitch rows into
        // a new-size buffer, and HW encoders cannot take a mismatched dmabuf),
        // so it goes back to the pool instead of being retained or encoded.
        let expect = node
            .capture
            .as_ref()
            .map(|c| (c.settings.width, c.settings.height));
        let new_frame = match (host.try_take_frame(host_idx), expect) {
            (Some(f), Some((w, h))) if f.width != w || f.height != h => {
                host.release_frame(host_idx, f);
                None
            }
            (f, _) => f,
        };
        let have_new = new_frame.is_some();
        // Streaming mode wants a constant-rate stream (the client's decoder pipeline
        // is built for it), so re-encode the retained frame every tick like the
        // compositor path does. Outside streaming mode, stay damage-driven, waking
        // only for a pending IDR (a viewer opening its keyframe gate), a screenshot
        // request, or a bouncing watermark that must keep moving.
        let streaming = node
            .capture
            .as_ref()
            .map(|c| c.settings.video_streaming_mode)
            .unwrap_or(false);
        let wm_active = node.overlay_state.is_active();
        let wm_animated = wm_active && node.overlay_state.is_animated();
        if !have_new && !want_idr_for_host && !streaming && !take_screenshot && !wm_animated {
            if let Some((id, buf)) = pool_slot.take()
                && let Some(cap) = node.capture.as_ref()
                && let Some(ref pool) = cap.encode_pool {
                        pool.cancel(id, buf);
                    }
            state.host = Some(host);
            return false;
        }
        if let Some(f) = new_frame {
            host.retain_frame(host_idx, f);
        }
        // Consume the (new or prior) retained frame's content into the pool slot /
        // GPU dmabuf. The frame stays retained for the next IDR. The CPU path
        // blends the watermark in place; the GPU path composites it below. Damage
        // is the fresh blit's; a re-encode of retained content has none of its own.
        let mut wm_drawn = false;
        let outcome = host.with_retained(host_idx, |r| {
            let Some(f) = r else { return RETAINED_NONE };
            damage_rects = if have_new { f.damage.clone() } else { Vec::new() };
            if let Some(cpu) = f.cpu.as_ref() {
                if gpu_encoder {
                    return RETAINED_CPU_FRAME;
                }
                host_cpu_frame = true;
                if let Some((_, ref mut buf)) = pool_slot {
                    cpu.write_bgra(f.width, f.height, buf);
                    if wm_active {
                        node.overlay_state.blend_bgra(buf, (f.width as usize) * 4, f.width, f.height);
                        wm_drawn = true;
                    }
                }
                cpu.write_bgra(f.width, f.height, &mut node.frame_buffer);
                if wm_active {
                    node.overlay_state
                        .blend_bgra(&mut node.frame_buffer, (f.width as usize) * 4, f.width, f.height);
                }
            } else if let Some(dmabuf) = f.dmabuf.as_ref() {
                if !gpu_encoder {
                    return RETAINED_GPU_FRAME;
                }
                host_enc_dmabuf = Some(dmabuf.clone());
            }
            RETAINED_OK
        });
        // GPU path: composite the watermark and serve screenshot readbacks with
        // the renderer. A bouncing watermark re-composes retained content into
        // this display's offscreen target every tick (drawing in place would
        // trail); anchored watermarks are stamped once onto each fresh blit and
        // ride along with retained re-encodes.
        if let Some(src) = host_enc_dmabuf.clone() {
            if wm_active
                && let Some(renderer) = state.gles_renderer.as_mut() {
                    if wm_animated {
                        if let Some((_, target)) = node.offscreen_buffer.as_mut() {
                            match compose_host_watermark(
                                renderer,
                                &node.overlay_state,
                                &src,
                                target,
                                (width, height),
                            ) {
                                Ok(sync) => {
                                    render_sync = Some(sync);
                                    host_enc_dmabuf = Some(target.clone());
                                    wm_drawn = true;
                                }
                                Err(e) => warn_once_host_watermark(&e),
                            }
                        }
                    } else if have_new {
                        let mut target = src.clone();
                        match draw_host_watermark(
                            renderer,
                            &node.overlay_state,
                            &mut target,
                            (width, height),
                        ) {
                            Ok(sync) => {
                                render_sync = Some(sync);
                                wm_drawn = true;
                            }
                            Err(e) => warn_once_host_watermark(&e),
                        }
                    }
                }
            if take_screenshot
                && let Some(renderer) = state.gles_renderer.as_mut() {
                    let mut shot = host_enc_dmabuf.clone().unwrap_or(src);
                    match renderer.bind(&mut shot) {
                        Ok(fb) => {
                            let rect = Rectangle::new((0, 0).into(), (width, height).into());
                            match renderer.copy_framebuffer(&fb, rect, Fourcc::Abgr8888) {
                                Ok(mapping) => match renderer.map_texture(&mapping) {
                                    Ok(data) => {
                                        let n = data.len().min(node.frame_buffer.len());
                                        node.frame_buffer[..n].copy_from_slice(&data[..n]);
                                    }
                                    Err(e) => eprintln!("[HostCapture] screenshot map: {e:?}"),
                                },
                                Err(e) => eprintln!("[HostCapture] screenshot copy: {e:?}"),
                            }
                        }
                        Err(e) => eprintln!("[HostCapture] screenshot bind: {e:?}"),
                    };
                }
        }
        if wm_drawn
            && let Some(rect) = node.overlay_state.damage_rect(width, height) {
                damage_rects.push(rect);
            }
        state.host = Some(host);
        if outcome != RETAINED_OK {
            if let Some((id, buf)) = pool_slot.take()
                && let Some(cap) = node.capture.as_ref()
                && let Some(ref pool) = cap.encode_pool {
                        pool.cancel(id, buf);
                    }
            // A frame type this display's consumer cannot take is recovered from rather
            // than warned about: either side of the mismatch would otherwise stream
            // nothing for the life of the capture.
            if outcome == RETAINED_GPU_FRAME {
                // The host is blitting into dmabufs a CPU encoder cannot read; ask it
                // for shm frames. Frames resume once the capture thread reallocates,
                // and the request is idempotent, so repeating it per tick costs nothing.
                if let Some(h) = state.host.as_ref() {
                    h.set_buffer_type(node.id, false);
                }
            } else if outcome == RETAINED_CPU_FRAME
                && let Some(cap) = node.capture.as_mut() {
                    // The host hands out software frames only (no zwp_linux_dmabuf v3),
                    // so the zero-copy session has nothing to import: demote it to the
                    // readback path, which encodes those frames as they arrive.
                    eprintln!(
                        "[HostCapture] host delivers software frames; demoting the zero-copy encoder to readback encode."
                    );
                    cap.video_encoder = None;
                    let s = &cap.settings;
                    let try_gpu = s.output_mode == 1
                        && !(s.use_cpu || s.encode_node_index == -1);
                    bootstrap_readback_pool(cap, node.id, false, try_gpu, None, None);
                    cap.request_idr();
                    // The consumer is now a CPU one, so the host stops preferring GPU slots.
                    if let Some(h) = state.host.as_ref() {
                        h.set_buffer_type(node.id, false);
                    }
                }
            return false;
        }
        render_success = true;
    }

    if !host_mode && state.use_gpu {
        if let Some(renderer) = state.gles_renderer.as_mut() {
            let mut cap = node.capture.as_mut();
            if let Some((_bo, dmabuf)) = node.offscreen_buffer.as_mut() {
                let render_age = if node.overlay_state.is_animated() || needs_full { 0 } else { 1 };
                match renderer.bind(dmabuf) {
                    Ok(mut frame) => {
                        let mut elements: Vec<CompositionElements<GlesRenderer, WaylandSurfaceRenderElement<GlesRenderer>>> = Vec::new();

                        if state.render_cursor_on_framebuffer
                            && let Some(pos) = pointer_local {
                                let scale = Scale::from(output_scale_val);

                                if let Some(CursorImageStatus::Named(icon)) = &state.current_cursor_icon {
                                    let name = wayland::frontend::cursor_icon_to_str(icon);
                                    let time = Duration::from_millis(state.clock.now().as_millis() as u64);
                                    if let Some(image) = state.cursor_helper.get_image_by_name(name, output_scale_val.round() as u32, time)
                                        && let Some(elem) = node.overlay_state.get_cursor_element(renderer, image, pos, output_scale_val) {
                                            elements.push(CompositionElements::Cursor(elem));
                                        }
                                } else if let Some(CursorImageStatus::Surface(surface)) = &state.current_cursor_icon {
                                     let hot = cursor_surface_hotspot(surface).to_f64();
                                     let phys_pos = (pos - hot).to_physical(scale);
                                     let elem_result = with_states(surface, |states| {
                                         WaylandSurfaceRenderElement::from_surface(renderer, surface, states, phys_pos, 1.0, smithay::backend::renderer::element::Kind::Cursor)
                                     });
                                     if let Ok(Some(cursor_elem)) = elem_result {
                                         elements.push(CompositionElements::Surface(cursor_elem));
                                     }
                                } else if state.current_cursor_icon.is_none() {
                                    let time = Duration::from_millis(state.clock.now().as_millis() as u64);
                                    let image = state.cursor_helper.get_image(output_scale_val.round() as u32, time);
                                    if let Some(elem) = node.overlay_state.get_cursor_element(renderer, image, pos, output_scale_val) {
                                        elements.push(CompositionElements::Cursor(elem));
                                    }
                                }
                            }

                        if let Some(elem) = node.overlay_state.get_watermark_element(renderer) {
                            elements.push(CompositionElements::Cursor(elem));
                        }

                        {
                            let layer_map = layer_map_for_output(&output);

                            let draw_layer = |renderer: &mut GlesRenderer, elements: &mut Vec<CompositionElements<GlesRenderer, WaylandSurfaceRenderElement<GlesRenderer>>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
                                for surface in layer_map.layers().rev() {
                                    let current_layer = surface.layer();
                                    if current_layer == target_layer
                                        && let Some(geo) = layer_map.layer_geometry(surface) {
                                            let elem = smithay::wayland::compositor::with_states(surface.wl_surface(), |states| {
                                                WaylandSurfaceRenderElement::from_surface(
                                                    renderer, surface.wl_surface(), states,
                                                    geo.loc.to_physical_precise_round(output_scale_val), 1.0,
                                                    smithay::backend::renderer::element::Kind::Unspecified
                                                )
                                            });
                                            if let Ok(Some(e)) = elem {
                                                elements.push(CompositionElements::Surface(e));
                                            }
                                        }
                                }
                            };

                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Overlay);
                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Top);
                        }

                        for window in state.space.elements_for_output(&output).collect::<Vec<_>>().into_iter().rev() {
                            let window_loc = state.space.element_location(window).unwrap_or_default() - origin;

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
                                            popup_pos.to_physical_precise_round(output_scale_val),
                                            1.0,
                                            smithay::backend::renderer::element::Kind::Unspecified
                                        )
                                    });
                                    if let Ok(Some(e)) = elem {
                                        elements.push(CompositionElements::Surface(e));
                                    }
                                }
                            }

                            elements.extend(window.render_elements(renderer, window_loc.to_physical_precise_round(output_scale_val), Scale::from(output_scale_val), 1.0).into_iter().map(CompositionElements::Space));
                        }

                        {
                            let layer_map = layer_map_for_output(&output);

                            let draw_layer = |renderer: &mut GlesRenderer, elements: &mut Vec<CompositionElements<GlesRenderer, WaylandSurfaceRenderElement<GlesRenderer>>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
                                for surface in layer_map.layers().rev() {
                                    let current_layer = surface.layer();
                                    if current_layer == target_layer
                                        && let Some(geo) = layer_map.layer_geometry(surface) {
                                            let elem = smithay::wayland::compositor::with_states(surface.wl_surface(), |states| {
                                                WaylandSurfaceRenderElement::from_surface(
                                                    renderer, surface.wl_surface(), states,
                                                    geo.loc.to_physical_precise_round(output_scale_val), 1.0,
                                                    smithay::backend::renderer::element::Kind::Unspecified
                                                )
                                            });
                                            if let Ok(Some(e)) = elem {
                                                elements.push(CompositionElements::Surface(e));
                                            }
                                        }
                                }
                            };

                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Bottom);
                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Background);
                        }
                        match node.damage_tracker.render_output(renderer, &mut frame, render_age, &elements, [0.1, 0.1, 0.1, 1.0]) {
                            Ok(result) => {
                                render_success = true;
                                if let Some(damage) = result.damage {
                                    damage_rects = damage.clone();
                                }
                                render_sync = Some(result.sync);
                                if let Some(c) = cap.as_deref_mut() {
                                    c.needs_full_render = false;
                                }
                            },
                            Err(e) => eprintln!("Render error: {:?}", e)
                        }
                        if let Some(c) = cap {
                            if !damage_rects.is_empty() {
                                c.content_gen += 1;
                            }
                            if let Some((id, ref mut buf)) = pool_slot {
                                // No-damage ticks skip the readback, so a pooled buffer can
                                // lag the offscreen target whenever the encoder held the
                                // other slot across a tick; one catch-up readback keeps
                                // every published buffer current.
                                if render_success && c.pool_content_gen[id] != c.content_gen {
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
                                    c.pool_content_gen[id] = c.content_gen;
                                }
                            }
                        }
                        if pool_slot.is_none() && take_screenshot {
                            let _ = renderer.with_context(|gl| unsafe {
                                gl.ReadPixels(
                                    0,
                                    0,
                                    width,
                                    height,
                                    smithay::backend::renderer::gles::ffi::RGBA,
                                    smithay::backend::renderer::gles::ffi::UNSIGNED_BYTE,
                                    node.frame_buffer.as_mut_ptr() as *mut std::ffi::c_void,
                                );
                            });
                        }
                    },
                    Err(e) => eprintln!("Failed to bind buffer: {:?}", e)
                }
            }
        }
    } else if !host_mode
        && let Some(renderer) = state.pixman_renderer.as_mut() {
            let mut cap = node.capture.as_mut();
            let (ptr, buf_age) = match pool_slot {
                Some((id, ref mut buf)) => {
                    let age = cap
                        .as_ref()
                        .map(|c| {
                            if c.pool_last_render[id] == 0 {
                                0
                            } else {
                                (c.render_seq + 1 - c.pool_last_render[id]) as usize
                            }
                        })
                        .unwrap_or(0);
                    (buf.as_mut_ptr() as *mut u32, age)
                }
                None => (node.frame_buffer.as_mut_ptr() as *mut u32, 0),
            };
            let mut image = unsafe {
                pixman::Image::from_raw_mut(pixman::FormatCode::A8R8G8B8, width as usize, height as usize, ptr, (width as usize) * 4, false).expect("Failed to create pixman image")
            };
                        match renderer.bind(&mut image) {
                        Ok(mut frame) => {
                            let mut elements: Vec<CompositionElements<PixmanRenderer, WaylandSurfaceRenderElement<PixmanRenderer>>> = Vec::new();

                            if state.render_cursor_on_framebuffer
                                && let Some(pos) = pointer_local {
                                    let scale = Scale::from(output_scale_val);

                                    if let Some(CursorImageStatus::Named(icon)) = &state.current_cursor_icon {
                                        let name = wayland::frontend::cursor_icon_to_str(icon);
                                        let time = Duration::from_millis(state.clock.now().as_millis() as u64);
                                        if let Some(image) = state.cursor_helper.get_image_by_name(name, output_scale_val.round() as u32, time)
                                            && let Some(elem) = node.overlay_state.get_cursor_element(renderer, image, pos, output_scale_val) {
                                                elements.push(CompositionElements::Cursor(elem));
                                            }
                                    } else if let Some(CursorImageStatus::Surface(surface)) = &state.current_cursor_icon {
                                         let hot = cursor_surface_hotspot(surface).to_f64();
                                         let phys_pos = (pos - hot).to_physical(scale);
                                         let elem_result = with_states(surface, |states| {
                                             WaylandSurfaceRenderElement::from_surface(renderer, surface, states, phys_pos, 1.0, smithay::backend::renderer::element::Kind::Cursor)
                                         });
                                         if let Ok(Some(cursor_elem)) = elem_result {
                                             elements.push(CompositionElements::Surface(cursor_elem));
                                         }
                                    } else if state.current_cursor_icon.is_none() {
                                        let time = Duration::from_millis(state.clock.now().as_millis() as u64);
                                        let image = state.cursor_helper.get_image(output_scale_val.round() as u32, time);
                                        if let Some(elem) = node.overlay_state.get_cursor_element(renderer, image, pos, output_scale_val) {
                                            elements.push(CompositionElements::Cursor(elem));
                                        }
                                    }
                                }

                            if let Some(elem) = node.overlay_state.get_watermark_element(renderer) {
                                elements.push(CompositionElements::Cursor(elem));
                            }

                            {
                                let layer_map = layer_map_for_output(&output);

                                let draw_layer = |renderer: &mut PixmanRenderer, elements: &mut Vec<CompositionElements<PixmanRenderer, WaylandSurfaceRenderElement<PixmanRenderer>>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
                                    for surface in layer_map.layers().rev() {
                                        let current_layer = surface.layer();
                                        if current_layer == target_layer
                                            && let Some(geo) = layer_map.layer_geometry(surface) {
                                                let elem = smithay::wayland::compositor::with_states(surface.wl_surface(), |states| {
                                                    WaylandSurfaceRenderElement::from_surface(
                                                        renderer, surface.wl_surface(), states,
                                                        geo.loc.to_physical_precise_round(output_scale_val), 1.0,
                                                        smithay::backend::renderer::element::Kind::Unspecified
                                                    )
                                                });
                                                if let Ok(Some(e)) = elem {
                                                    elements.push(CompositionElements::Surface(e));
                                                }
                                            }
                                    }
                                };

                                draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Overlay);
                                draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Top);
                            }

                            for window in state.space.elements_for_output(&output).collect::<Vec<_>>().into_iter().rev() {
                                let loc = state.space.element_location(window).unwrap_or_default() - origin;

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
                                                    popup_pos.to_physical_precise_round(output_scale_val),
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

                                elements.extend(window.render_elements(renderer, loc.to_physical_precise_round(output_scale_val), Scale::from(output_scale_val), 1.0).into_iter().map(CompositionElements::Space));
                            }

                            {
                                let layer_map = layer_map_for_output(&output);

                                let draw_layer = |renderer: &mut PixmanRenderer, elements: &mut Vec<CompositionElements<PixmanRenderer, WaylandSurfaceRenderElement<PixmanRenderer>>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
                                    for surface in layer_map.layers().rev() {
                                        let current_layer = surface.layer();
                                        if current_layer == target_layer
                                            && let Some(geo) = layer_map.layer_geometry(surface) {
                                                let elem = smithay::wayland::compositor::with_states(surface.wl_surface(), |states| {
                                                    WaylandSurfaceRenderElement::from_surface(
                                                        renderer, surface.wl_surface(), states,
                                                        geo.loc.to_physical_precise_round(output_scale_val), 1.0,
                                                        smithay::backend::renderer::element::Kind::Unspecified
                                                    )
                                                });
                                                if let Ok(Some(e)) = elem {
                                                    elements.push(CompositionElements::Surface(e));
                                                }
                                            }
                                    }
                                };

                                draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Bottom);
                                draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Background);
                            }

                    let render_age = if node.overlay_state.is_animated() || needs_full { 0 } else { buf_age };
                    match node.damage_tracker.render_output(renderer, &mut frame, render_age, &elements, [0.1, 0.1, 0.1, 1.0]) {
                        Ok(result) => {
                            render_success = true;
                            if let Some(c) = cap.as_deref_mut() {
                                c.needs_full_render = false;
                            }
                            if let Some(damage) = result.damage { damage_rects = damage.clone(); }
                        },
                        Err(e) => eprintln!("Render error: {:?}", e)
                    }
                    if let Some(c) = cap {
                        c.render_seq += 1;
                        if render_success
                            && let Some((id, _)) = pool_slot {
                                c.pool_last_render[id] = c.render_seq;
                            }
                    }
                },
                Err(e) => eprintln!("Failed to bind pixman image: {:?}", e)
            }
        }

    if render_success {
        node.target_seeded = true;
        let time = state.clock.now();
        // The composited frame is what the capture consumes, so this render is also the
        // presentation moment for wp_presentation feedback.
        let mut feedback = OutputPresentationFeedback::new(&output);
        for window in state.space.elements_for_output(&output).cloned().collect::<Vec<_>>() {
            window.send_frame(&output, time, Some(Duration::ZERO), |_, _| Some(output.clone()));
            window.take_presentation_feedback(
                &mut feedback,
                |_, _| Some(output.clone()),
                |_, _| wp_presentation_feedback::Kind::empty(),
            );
        }
        // Panels, backgrounds and other layer-shell surfaces are composited from the
        // layer map rather than the space, so they need the callback separately: one
        // that never arrives leaves a client which draws on frame callbacks showing
        // whatever it painted first, for as long as the session lasts.
        for layer in layer_map_for_output(&output).layers() {
            layer.send_frame(&output, time, Some(Duration::ZERO), |_, _| Some(output.clone()));
            layer.take_presentation_feedback(
                &mut feedback,
                |_, _| Some(output.clone()),
                |_, _| wp_presentation_feedback::Kind::empty(),
            );
        }
        send_cursor_frame(state, &output, time);
        let refresh = match node.capture.as_ref() {
            Some(c) => Refresh::Fixed(Duration::from_secs_f64(1.0 / c.settings.target_fps.max(1.0))),
            None => Refresh::Unknown,
        };
        node.frame_seq += 1;
        feedback.presented(time, refresh, node.frame_seq, wp_presentation_feedback::Kind::Vsync);

        // Host mode renders nothing locally, so there is no composited buffer to
        // serve capture clients from; their frames stay parked.
        if !host_mode {
            service_copy_frames(state, node, width, height, &damage_rects);
        }

        if !hold_frame && let Some(cap) = node.capture.as_mut() {
            // A dead encode thread (panic, unexpected exit) cannot drain the pool;
            // every publish would park the slot and each tick would silently skip
            // while is_capturing still reports true. Rebuild the readback path in
            // place, reusing a cleanly handed-back encoder session if any.
            if cap.encode_join.as_ref().is_some_and(|j| j.is_finished()) {
                let prior = cap
                    .encode_join
                    .take()
                    .and_then(|j| j.join().ok().flatten());
                if let Some(pool) = cap.encode_pool.take() {
                    pool.shutdown();
                }
                let s = &cap.settings;
                let try_gpu = s.output_mode == 1
                    && !(s.use_cpu || s.encode_node_index == -1);
                eprintln!("[Wayland] encode thread died; rebuilding the readback path.");
                // Host frames reach the pool as BGRA; only our own GLES readback produces RGBA.
                bootstrap_readback_pool(cap, node.id, state.use_gpu && !host_mode, try_gpu, prior, None);
                cap.request_idr();
            }
            if cap.encode_pool.is_some() {
                if take_screenshot
                    && let Some((_, ref buf)) = pool_slot {
                        let n = buf.len().min(node.frame_buffer.len());
                        node.frame_buffer[..n].copy_from_slice(&buf[..n]);
                    }
                if let Some((id, buf)) = pool_slot.take() {
                    let is_animated = node.overlay_state.is_animated();
                    cap.encode_pool.as_ref().unwrap().publish(WlFrame {
                        id,
                        buf,
                        frame_id: cap.frame_counter,
                        damage: std::mem::take(&mut damage_rects),
                        is_animated,
                    });
                    cap.frame_counter = cap.frame_counter.wrapping_add(1);
                }
            } else if let Some(ref mut encoder) = cap.video_encoder {
                // Deliver the parked frame (if any) first, WITHOUT blocking: this runs
                // on the calloop thread, and a blocking send would freeze
                // input/command/Wayland dispatch for as long as the Python consumer
                // stalls. While a frame stays parked, no new frame is encoded — an
                // encoded frame joins the H.264 reference chain and can never be
                // dropped — and the tick's damage is latched so the pause never loses
                // a change.
                let slot_free = match cap.pending_hw_delivery.take() {
                    None => true,
                    Some(pending) => match cap.deliver_tx.as_ref() {
                        None => true,
                        Some(tx) => match tx.try_send(pending) {
                            Ok(()) => true,
                            Err(std::sync::mpsc::TrySendError::Full(p)) => {
                                cap.pending_hw_delivery = Some(p);
                                false
                            }
                            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => true,
                        },
                    },
                };
                if !slot_free {
                    if !damage_rects.is_empty() {
                        cap.pending_hw_damage = true;
                    }
                } else {
                let is_animated = node.overlay_state.is_animated();
                let had_damage = !damage_rects.is_empty()
                    || std::mem::take(&mut cap.pending_hw_damage);
                let decision = crate::pipeline::decide_hw_fullframe(
                    &mut cap.vaapi_state,
                    &cap.settings,
                    cap.frame_counter,
                    had_damage,
                    is_animated,
                    requested_idr,
                );
                let send_frame = decision.send;
                let force_idr = decision.force_idr;
                let target_qp = decision.target_qp;

                let mut frame_out = false;
                if send_frame {
                    if let Some(sync) = render_sync.take() {
                        let _ = sync.wait();
                    }
                    // Host-capture frames encode from the buffer the host blitted
                    // into; otherwise from this display's own composited buffer.
                    let enc_dmabuf: Option<Dmabuf> = host_enc_dmabuf
                        .clone()
                        .or_else(|| node.offscreen_buffer.as_ref().map(|(_, d)| d.clone()));
                    let result = match encoder {
                        GpuEncoder::Nvenc(enc) => {
                            if let Some(ref dmabuf) = enc_dmabuf {
                                enc.encode(dmabuf, cap.frame_counter as u64, target_qp, force_idr)
                            } else {
                                Err("NVENC ZeroCopy requires offscreen buffer (GPU context)".to_string())
                            }
                        },
                        GpuEncoder::Vaapi(enc) => {
                            if let Some(ref dmabuf) = enc_dmabuf {
                                enc.encode_dmabuf(dmabuf, cap.frame_counter as u64, target_qp, force_idr)
                            } else {
                                Err("Vaapi ZeroCopy requires offscreen buffer (GPU context)".to_string())
                            }
                        }
                    };

                    if let Ok(data) = result {
                        cap.hw_error_streak = 0;
                        cap.hw_rebuilt = false;
                        if !data.is_empty() {
                            frame_out = true;
                            cap.encode_stats.frames.fetch_add(1, Ordering::Relaxed);
                            cap.encode_stats.stripes.fetch_add(1, Ordering::Relaxed);
                            if let Some(ref tx) = cap.deliver_tx {
                                let stripes = vec![EncodedStripe {
                                    data: Arc::new(data), data_type: 2, stripe_y_start: 0,
                                    stripe_height: height, frame_id: cap.frame_counter as i32,
                                }];
                                if let Some(ref socket) = cap.recording_sink {
                                    socket.write_frame(&stripes, height);
                                }
                                crate::recorder::wayland_tap(node.id, &stripes);
                                // Non-blocking: a full slot parks the frame (delivered
                                // ahead of any new encode above).
                                match tx.try_send(stripes) {
                                    Ok(()) => {}
                                    Err(std::sync::mpsc::TrySendError::Full(s)) => {
                                        cap.pending_hw_delivery = Some(s);
                                    }
                                    Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {}
                                }
                            }
                        }
                    } else if let Err(e) = result {
                        eprintln!("HW Encode Error: {}", e);
                        cap.hw_error_streak = cap.hw_error_streak.saturating_add(1);
                        if cap.hw_error_streak == HW_ERROR_RECOVERY_THRESHOLD {
                            // The zero-copy session persistently fails after having
                            // worked (driver hiccup, CUDA pressure from a co-tenant):
                            // rebuild the session once, else demote to the readback
                            // path. Streaming black frames forever is not an option.
                            // A session whose encodes keep failing still constructs, so
                            // the rebuild only counts as recovery until the next streak;
                            // otherwise the stream would rebuild in a loop and never demote.
                            let rebuilt = if cap.hw_rebuilt {
                                None
                            } else {
                                // The broken session is released before its replacement is
                                // opened: the failure it recovers from is usually device
                                // memory pressure, and holding both at once is what would
                                // make the rebuild fail too.
                                drop(cap.video_encoder.take());
                                rebuild_zerocopy_encoder(cap, state)
                            };
                            match rebuilt {
                                Some(enc) => {
                                    cap.video_encoder = Some(enc);
                                    cap.pending_force_idr = true;
                                    cap.hw_rebuilt = true;
                                    eprintln!("[Wayland] zero-copy HW encoder rebuilt after repeated encode errors.");
                                }
                                None => {
                                    eprintln!("[Wayland] zero-copy HW encoder unrecoverable; demoting to readback encode.");
                                    cap.video_encoder = None;
                                    cap.hw_rebuilt = false;
                                    // Mirror the startup intent: readback still
                                    // tries the GPU unless the operator opted out.
                                    let s = &cap.settings;
                                    let try_gpu = s.output_mode == 1
                                        && !(s.use_cpu || s.encode_node_index == -1);
                                    // Host frames reach the pool as BGRA; only our own
                                    // GLES readback produces RGBA.
                                    bootstrap_readback_pool(
                                        cap, node.id, state.use_gpu && !host_mode, try_gpu, None,
                                        None,
                                    );
                                    // The host has to switch to buffers the readback path
                                    // can read back on the CPU.
                                    if host_mode
                                        && let Some(h) = state.host.as_ref() {
                                            h.set_buffer_type(node.id, false);
                                        }
                                }
                            }
                            cap.hw_error_streak = 0;
                        }
                    }
                }
                // An unserved request stays armed: on an infinite GOP an IDR lost to an
                // encode error would never self-heal.
                cap.pending_force_idr = requested_idr && !frame_out;
                cap.frame_counter = cap.frame_counter.wrapping_add(1);
                }
            }
        }
        if take_screenshot
            && let Some((_, resp)) = state.pending_screenshot.take() {
                if !node.frame_buffer.is_empty() {
                    let w = width as u32;
                    let h = height as u32;
                    // A host software frame was written BGRA into the frame buffer, so it
                    // needs the swap even when the local renderer is GLES.
                    let png = if state.use_gpu && !host_cpu_frame {
                        crate::computer_use::encode_png_rgba(&node.frame_buffer, w, h)
                    } else {
                        let mut rgba = node.frame_buffer.clone();
                        for px in rgba.chunks_exact_mut(4) {
                            px.swap(0, 2);
                        }
                        crate::computer_use::encode_png_rgba(&rgba, w, h)
                    };
                    match png {
                        Ok(data) => { let _ = resp.send(Ok(data)); }
                        Err(e) => {
                            let _ = resp.send(Err(format!("PNG encode error: {e}")));
                            eprintln!("[ComputerUse] PNG encode error: {}", e);
                        }
                    }
                } else {
                    let _ = resp.send(Err("Screenshot render produced no pixels".to_string()));
                }
            }
    }
    if let Some((id, buf)) = pool_slot.take()
        && let Some(cap) = node.capture.as_ref()
        && let Some(ref pool) = cap.encode_pool {
                pool.cancel(id, buf);
            }
    false
}

/// True when the rectangles `(x, y, w, h)` overlap: strict interior intersection, so
/// touching edges do not count and empty (non-positive-dimension) rectangles never
/// overlap anything. Arithmetic is widened so extreme coordinates cannot wrap.
fn rects_overlap(a: (i32, i32, i32, i32), b: (i32, i32, i32, i32)) -> bool {
    let (ax, ay, aw, ah) = (a.0 as i64, a.1 as i64, a.2 as i64, a.3 as i64);
    let (bx, by, bw, bh) = (b.0 as i64, b.1 as i64, b.2 as i64, b.3 as i64);
    aw > 0 && ah > 0 && bw > 0 && bh > 0
        && ax < bx + bw && bx < ax + aw
        && ay < by + bh && by < ay + ah
}

/// An overlapping output as `(id, flavor, rect)`, where flavor names which rectangle
/// pair matched and rect is `(x, y, width, height)`.
type OutputOverlap = (u32, &'static str, (i32, i32, i32, i32));

/// The first live output (excluding `skip_id`) whose rectangle overlaps a candidate
/// placement. Both rectangle flavors are checked — logical
/// (Space layout, scale-divided) and physical (mode pixels at the same origin) — because
/// input injection and cursor compositing key off the physical rects while window layout
/// keys off the logical ones, and neither may overlap.
fn find_output_overlap(
    nodes: &[wayland::frontend::OutputNode],
    skip_id: Option<u32>,
    logical: (i32, i32, i32, i32),
    physical: (i32, i32, i32, i32),
) -> Option<OutputOverlap> {
    for n in nodes {
        if Some(n.id) == skip_id {
            continue;
        }
        if let Some(geo) = n.logical_geometry() {
            let other = (geo.loc.x, geo.loc.y, geo.size.w, geo.size.h);
            if rects_overlap(logical, other) {
                return Some((n.id, "logical", other));
            }
        }
        if let Some(mode) = n.output.current_mode() {
            let other = (n.pos.0, n.pos.1, mode.size.w, mode.size.h);
            if rects_overlap(physical, other) {
                return Some((n.id, "physical", other));
            }
        }
    }
    None
}

/// Create an additional output mapped into the layout at `(x, y)`. Fails (false) on a
/// duplicate id, non-positive geometry/scale, a rectangle overlapping a live output, or a
/// GPU render-target allocation failure. Only Create/Reposition placements are validated:
/// a capture reconfigure (StartCapture on an existing output) resizes UNVALIDATED, so
/// keeping a multi-step relayout overlap-free at every step is the caller's ordering
/// responsibility.
fn create_output_on(
    state: &mut AppState,
    id: u32,
    width: i32,
    height: i32,
    x: i32,
    y: i32,
    scale: f64,
) -> bool {
    if state.node_idx_for_id(id).is_some() || width <= 0 || height <= 0 || scale <= 0.0 {
        return false;
    }
    // Host-capture mode: displays map onto host outputs by rank, so an output beyond
    // the host's count would exist but never receive a frame. Refuse it instead.
    if let Some(host) = state.host.as_ref() {
        let capacity = host.output_count();
        if state.output_nodes.len() >= capacity {
            eprintln!(
                "[Wayland] CreateOutput {id}: rejected, the host compositor has {capacity} output(s) and all are backing displays."
            );
            return false;
        }
    }
    let logical_size = (
        (width as f64 / scale).round() as i32,
        (height as f64 / scale).round() as i32,
    );
    if let Some((oid, flavor, other)) = find_output_overlap(
        &state.output_nodes,
        None,
        (x, y, logical_size.0, logical_size.1),
        (x, y, width, height),
    ) {
        eprintln!(
            "[Wayland] CreateOutput {id}: rejected, {flavor} rect {}x{}+{x}+{y} overlaps output {oid} at {}x{}+{}+{}.",
            if flavor == "logical" { logical_size.0 } else { width },
            if flavor == "logical" { logical_size.1 } else { height },
            other.2, other.3, other.0, other.1,
        );
        return false;
    }
    let output = Output::new(
        format!("HEADLESS-{}", id + 1),
        PhysicalProperties {
            size: (width, height).into(),
            subpixel: Subpixel::Unknown,
            make: "Pixelflux".into(),
            model: "Virtual".into(),
            serial_number: format!("{:03}", id + 1),
        },
    );
    let mode = OutputMode { size: (width, height).into(), refresh: 60_000 };
    output.change_current_state(
        Some(mode),
        Some(Transform::Normal),
        Some(OutputScale::Fractional(scale)),
        Some((x, y).into()),
    );
    output.set_preferred(mode);
    let mut offscreen = None;
    if state.use_gpu {
        let Some(gbm) = state.gbm_device.as_mut() else { return false };
        match gbm.create_buffer_object(
            width as u32,
            height as u32,
            GbmFormat::Argb8888,
            BufferObjectFlags::RENDERING,
        ) {
            Ok(bo) => {
                let dmabuf = create_dmabuf_from_bo(&bo);
                offscreen = Some((bo, dmabuf));
            }
            Err(e) => {
                eprintln!("[Wayland] CreateOutput {id}: GBM allocation {width}x{height} failed ({e:?}).");
                return false;
            }
        }
    }
    state.space.map_output(&output, (x, y));
    let global = output.create_global::<AppState>(&state.dh);
    let damage_tracker = OutputDamageTracker::from_output(&output);
    if let Some(host) = state.host.as_ref() {
        host.set_layout(id, x, y);
    }
    println!("[Wayland] Output {id} created: {width}x{height} @ ({x}, {y}) scale {scale:.2}.");
    state.output_nodes.push(wayland::frontend::OutputNode {
        id,
        output,
        global,
        pos: (x, y),
        damage_tracker,
        frame_buffer: vec![0u8; (width.max(0) as usize) * (height.max(0) as usize) * 4],
        offscreen_buffer: offscreen,
        overlay_state: OverlayState::default(),
        capture: None,
        frame_seq: 0,
        target_seeded: false,
        content_hold_until: None,
    });
    // A nested session opens one host toplevel per screen and the extras wait
    // parked until a display exists for them: hand the newest waiting window to
    // the new output. Windows stacked on an output rather than parked — anything
    // placed before this compositor started parking them — remain candidates.
    let mut counts: Vec<(u32, usize)> = Vec::new();
    for w in state.space.elements() {
        let oid = wayland::frontend::window_output_id(w);
        match counts.iter_mut().find(|(o, _)| *o == oid) {
            Some((_, c)) => *c += 1,
            None => counts.push((oid, 1)),
        }
    }
    let newest = |pred: &dyn Fn(&smithay::desktop::Window) -> bool| {
        state
            .space
            .elements()
            .filter(|w| pred(w))
            .max_by_key(|w| wayland::frontend::window_meta(w).map(|m| m.id).unwrap_or(0))
            .cloned()
    };
    let adopt = newest(&|w| {
        wayland::frontend::window_meta(w)
            .map(|m| m.parked.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(false)
    })
    .or_else(|| {
        newest(&|w| {
            let oid = wayland::frontend::window_output_id(w);
            counts.iter().any(|(o, c)| *o == oid && *c >= 2)
        })
    });
    if let Some(window) = adopt {
        state.place_window_on_output(&window, id);
        println!(
            "[Wayland] Output {id}: adopted waiting window {}.",
            wayland::frontend::window_meta(&window).map(|m| m.id).unwrap_or(0)
        );
    }
    true
}

/// Move an existing output (the primary included) to layout offset `(x, y)`. The output's
/// advertised position, its Space mapping, and the windows placed on it (mapped at the
/// output's origin — window positions are output-relative under forced fullscreen) all
/// follow, so absolute input injection and cursor compositing — both keyed off `node.pos` —
/// resolve against the new layout immediately. A destination overlapping another live
/// output is refused (false). As with `CreateOutput`, only the placement itself is
/// validated: a capture reconfigure (StartCapture on an existing output) resizes
/// UNVALIDATED, so keeping a multi-step relayout overlap-free at every step is the
/// caller's ordering responsibility.
fn reposition_output_on(state: &mut AppState, id: u32, x: i32, y: i32) -> bool {
    let Some(idx) = state.node_idx_for_id(id) else { return false };
    let output = state.output_nodes[idx].output.clone();
    if state.output_nodes[idx].pos == (x, y) {
        return true;
    }
    let logical_size = state.output_nodes[idx]
        .logical_geometry()
        .map(|g| (g.size.w, g.size.h))
        .unwrap_or((0, 0));
    let physical_size = output.current_mode().map(|m| (m.size.w, m.size.h)).unwrap_or((0, 0));
    if let Some((oid, flavor, other)) = find_output_overlap(
        &state.output_nodes,
        Some(id),
        (x, y, logical_size.0, logical_size.1),
        (x, y, physical_size.0, physical_size.1),
    ) {
        eprintln!(
            "[Wayland] RepositionOutput {id}: rejected, {flavor} rect {}x{}+{x}+{y} overlaps output {oid} at {}x{}+{}+{}.",
            if flavor == "logical" { logical_size.0 } else { physical_size.0 },
            if flavor == "logical" { logical_size.1 } else { physical_size.1 },
            other.2, other.3, other.0, other.1,
        );
        return false;
    }
    state.output_nodes[idx].pos = (x, y);
    if let Some(host) = state.host.as_ref() {
        host.set_layout(id, x, y);
    }
    output.change_current_state(None, None, None, Some((x, y).into()));
    state.space.map_output(&output, (x, y));
    let windows: Vec<smithay::desktop::Window> = state
        .space
        .elements()
        .filter(|w| wayland::frontend::window_output_id(w) == id)
        .cloned()
        .collect();
    for window in &windows {
        state.space.map_element(window.clone(), (x, y), false);
    }
    if let Some(cap) = state.output_nodes[idx].capture.as_mut() {
        cap.needs_full_render = true;
    }
    println!("[Wayland] Output {id} repositioned to ({x}, {y}).");
    true
}

/// Destroy a secondary output: end its capture, relocate its windows onto the primary
/// output, unmap it from the space, and retract its global. The primary (id 0) is refused.
fn destroy_output_on(state: &mut AppState, id: u32) -> bool {
    if id == 0 {
        return false;
    }
    let Some(_) = state.node_idx_for_id(id) else { return false };
    stop_capture_on_display(state, id);
    if let Some(host) = state.host.as_ref() {
        host.idle_output(id);
    }
    wayland_owners().lock().unwrap().remove(&id);
    // Relocate while the node is still registered so output leave/enter both resolve.
    let windows: Vec<smithay::desktop::Window> = state
        .space
        .elements()
        .filter(|w| wayland::frontend::window_output_id(w) == id)
        .cloned()
        .collect();
    for window in &windows {
        // The primary keeps whichever screen it already shows: a nested session's
        // second screen parks again rather than covering the first.
        if state.would_cover_screen(window, 0) {
            state.park_window(window, 0);
        } else {
            state.place_window_on_output(window, 0);
        }
    }
    for w in &state.pending_windows {
        if let Some(meta) = wayland::frontend::window_meta(w)
            && meta.output.load(Ordering::Relaxed) == id {
                meta.output.store(0, Ordering::Relaxed);
            }
    }
    let idx = state.node_idx_for_id(id).unwrap();
    let node = state.output_nodes.remove(idx);
    state.space.unmap_output(&node.output);
    state.dh.remove_global::<AppState>(node.global);
    println!(
        "[Wayland] Output {id} destroyed; {} window(s) relocated to primary.",
        windows.len()
    );
    true
}
/// Startup inputs for the compositor thread: its two calloop channels, the sender it hands
/// to computer-use, the initial geometry, and the GPU / cursor policy.
struct WaylandThreadConfig {
    command_rx: smithay::reexports::calloop::channel::Channel<ThreadCommand>,
    wake_rx: smithay::reexports::calloop::channel::Channel<()>,
    command_tx: smithay::reexports::calloop::channel::Sender<ThreadCommand>,
    initial_width: i32,
    initial_height: i32,
    explicit_dri_node: String,
    auto_gpu_selected: bool,
    cursor_size: i32,
}

/// Bring the hardware renderer up on a DRM render node: the GBM device, the EGL display and
/// context it backs, the GLES renderer, and the allocator the render targets come from.
///
/// The compositor and [`probe_wayland_gpu`] share this so the probe's answer is the
/// compositor's own. Errors name the step that failed, which is all a caller can act on.
fn gpu_render_init(
    device_path: &std::path::Path,
) -> Result<(RawGbmDevice<File>, GlesRenderer), String> {
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
    let renderer = unsafe { GlesRenderer::new(context) }
        .map_err(|_| "Failed to init GlesRenderer")?;
    Ok((gbm_allocator, renderer))
}

/// `GL_RENDERER` of a live renderer — the driver that actually answered, which is the only
/// thing that separates a GPU from Mesa's software fallback on the same node. Empty when the
/// context cannot be made current or the string is unavailable.
fn gl_renderer_name(renderer: &mut GlesRenderer) -> String {
    renderer
        .with_context(|gl| unsafe {
            let ptr = gl.GetString(smithay::backend::renderer::gles::ffi::RENDERER);
            if ptr.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(ptr as *const std::ffi::c_char)
                    .to_string_lossy()
                    .into_owned()
            }
        })
        .unwrap_or_default()
}

/// A GPU is exposed to this container or machine.
///
/// Only device nodes count. `/sys/class/drm` is the host's and lists cards a container may
/// have no access to, while `/dev` is the container's own; the NVIDIA character devices
/// answer for a driver stack that was given no DRM node at all.
fn gpu_exposed() -> bool {
    std::path::Path::new("/dev/nvidiactl").exists()
        || std::fs::read_dir("/dev/dri")
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.starts_with("renderD") || name.starts_with("card")
            })
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
///    seat, SHM, shell, dmabuf, selections, and the rest) plus the output registry: the primary
///    virtual `HEADLESS-1` output at layout (0, 0), extended at runtime by CreateOutput with
///    additional outputs at their layout offsets, each `OutputNode` owning its damage tracker,
///    render targets, and (at most one) capture pipeline.
/// 3. **Event dispatch**:
///    - **Command channel**: control messages from the Python thread — per-display start/stop,
///      output lifecycle (create/destroy/list/move-window), input injection routed across the
///      output layout, keymap and clipboard operations, live rate / tunable changes, and the
///      computer-use queries.
///    - **Wayland socket**: accepts client connections and drives the compositor protocol.
/// 4. **StartCapture reconfigure** (per display): reprograms that output's mode / scale / refresh,
///    resizes its framebuffer and offscreen GBM buffer, and fullscreens the toplevels placed on
///    it. The encode device is resolved here: an operator's explicit `encode_node_index`
///    (-1 software, >= 0 a device) always wins, and only the unset `-2` sentinel is filled from
///    the auto-picked render node. H.264 output masks the dimensions even, because 4:2:0 needs
///    even width and height.
/// 5. **Encode-path choice + render loop**: only a same-GPU GLES session encodes zero-copy on this
///    calloop thread, because the dmabuf and its EGL context are calloop-affine; every readback
///    flavor (striped software H.264/JPEG, Pixman, or a cross-GPU hardware encoder) builds its
///    encoders on that display's dedicated encode thread instead. A shared timer (paced at the
///    fastest active capture) renders each capturing output — its windows, popups and layers made
///    output-local, the cursor only on the pointer's output — applies the shared paint-over /
///    recovery-IDR policy per display, and delivers each display's encoded stripes through its own
///    frame callback. The zero-copy encode waits the GL render fence first, so a hardware encoder
///    reading the dmabuf through CUDA/VA never maps a half-rasterized (torn) frame.
/// 6. **Thread lifecycle**: the Python frame callback runs on a dedicated delivery thread so its
///    GIL never stalls calloop input / control dispatch, and in readback mode the encoders run on
///    the `wl-encode` thread. On a restart or stop the encode thread is torn down before the
///    delivery thread — it feeds the delivery sender and must be gone first — and the retained
///    callbacks are dropped and gated by a process-shutdown flag so nothing fires into a finalizing
///    interpreter.
fn run_wayland_thread(cfg: WaylandThreadConfig) {
    let WaylandThreadConfig {
        command_rx,
        wake_rx,
        command_tx,
        initial_width,
        initial_height,
        explicit_dri_node,
        auto_gpu_selected,
        cursor_size,
    } = cfg;
    let width: i32 = if initial_width > 0 { initial_width } else { 1024 };
    let height: i32 = if initial_height > 0 { initial_height } else { 768 };

    let mut event_loop = match EventLoop::<AppState>::try_new() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[Wayland] compositor thread aborting: event loop init failed: {e}");
            return;
        }
    };
    let display: Display<AppState> = match Display::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[Wayland] compositor thread aborting: display init failed: {e}");
            return;
        }
    };
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
            let (gbm_allocator, mut renderer) = gpu_render_init(device_path)?;

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
        if dri_node.is_empty() {
            println!("[Wayland] No render node. Initializing Software Renderer (Pixman).");
        }
        pixman_renderer = Some(PixmanRenderer::new().expect("Failed to init PixmanRenderer"));
        use_gpu = false;
    }

    let compositor_state = CompositorState::new_v6::<AppState>(&dh);
    let image_capture_source_state = ImageCaptureSourceState::new();
    let output_capture_source_state = OutputCaptureSourceState::new::<AppState>(&dh);
    let image_copy_capture_state = ImageCopyCaptureState::new::<AppState>(&dh);
    let fractional_scale_state = FractionalScaleManagerState::new::<AppState>(&dh);
    let shm_state = ShmState::new::<AppState>(&dh, vec![]);
    let output_state = OutputManagerState::new_with_xdg_output::<AppState>(&dh);
    let mut seat_state = SeatState::new();
    let shell_state = XdgShellState::new::<AppState>(&dh);
    let space = Space::default();
    let layer_shell_state = WlrLayerShellState::new::<AppState>(&dh);
    let data_device_state = DataDeviceState::new::<AppState>(&dh);
    let data_control_state = DataControlState::new::<AppState, _>(&dh, None, |_| true);
    let ext_data_control_state = ExtDataControlState::new::<AppState, _>(&dh, None, |_| true);
    let cursor_shape_state = CursorShapeManagerState::new::<AppState>(&dh);
    let _vk_global = dh.create_global::<AppState, ZwpVirtualKeyboardManagerV1, _>(1, ());
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
        ext_data_control_state,
        cursor_shape_state,
        image_capture_source_state,
        output_capture_source_state,
        image_copy_capture_state,
        copy_sessions: Vec::new(),
        output_state,
        seat_state,
        shell_state,
        layer_shell_state,
        space,
        data_device_state,
        data_control_state,
        dh: dh.clone(),
        seat,
        pointer_warp_state,
        relative_pointer_state,
        pointer_constraints_state,
        output_nodes: Vec::new(),
        pending_windows: Vec::new(),
        foreign_toplevel_list,
        xdg_decoration_state,
        xdg_activation_state,
        primary_selection_state,
        popups,
        gles_renderer,
        pixman_renderer,
        gbm_device: gbm_device_raw,
        settings: RustCaptureSettings {
            width,
            height,
            ..RustCaptureSettings::default()
        },
        cursor_callback_set: false,
        cursor_tx: wayland::cursor::spawn_cursor_worker(
            cursor_size,
            RustCaptureSettings::default().cursor_size_cap,
        ),
        clipboard_callback: None,
        pending_clipboard_read: None,
        current_selection_mime: None,
        last_log_time: Instant::now(),
        start_time: Instant::now(),
        clock: Clock::new(),
        use_gpu,
        cursor_helper: Cursor::load(cursor_size),
        keymap_policy: wayland::keymap::KeymapPolicy::empty(),
        host: None,
        host_layout_pending: std::collections::HashMap::new(),
        current_cursor_icon: None,
        cursor_surface_pending: false,
        cursor_buffer: None,
        render_cursor_on_framebuffer: false,
        render_node_path,
        auto_gpu_selected,
        pending_screenshot: None,
        command_rx: None,
        last_input_at: None,
        frame_idle_long: false,
        last_idle_service_at: None,
        deliver_reaper: Vec::new(),
        encode_reaper: Vec::new(),
    };
    // Seed the keymap policy with the seat's initial keymap so overlay binds splice onto
    // the exact text clients received.
    {
        let initial_keymap = if let Some(kb) = state.seat.get_keyboard() {
            kb.with_xkb_state(&mut state, |context| match context.xkb().lock() {
                Ok(guard) => {
                    let keymap = unsafe { guard.keymap() };
                    keymap.get_as_string(smithay::input::keyboard::xkb::KEYMAP_FORMAT_TEXT_V1)
                }
                Err(_) => String::new(),
            })
        } else {
            String::new()
        };
        state.keymap_policy.rebuild_base(initial_keymap);
    }

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
    let global = output.create_global::<AppState>(&dh);
    let damage_tracker = OutputDamageTracker::from_output(&output);
    state.output_nodes.push(wayland::frontend::OutputNode {
        id: 0,
        output,
        global,
        pos: (0, 0),
        damage_tracker,
        frame_buffer: vec![0u8; (width.max(0) as usize) * (height.max(0) as usize) * 4],
        offscreen_buffer,
        overlay_state: OverlayState::default(),
        capture: None,
        frame_seq: 0,
        target_seeded: false,
        content_hold_until: None,
    });

    /// Apply every queued control command in FIFO order. Sends wake the loop through the
    /// separate wake channel, and the render tick ALSO drains before starting its work, so
    /// queued input is applied ahead of a long render/encode instead of waiting it out.
    fn drain_thread_commands(state: &mut AppState) {
        let Some(rx) = state.command_rx.take() else { return };
        let mut had_input = false;
        while let Ok(cmd) = rx.try_recv() {
            had_input |= matches!(
                cmd,
                ThreadCommand::KeyboardKey { .. }
                    | ThreadCommand::KeyboardKeys { .. }
                    | ThreadCommand::PointerMotion { .. }
                    | ThreadCommand::PointerRelativeMotion { .. }
                    | ThreadCommand::PointerButton { .. }
                    | ThreadCommand::PointerAxis { .. }
            );
            handle_thread_command(state, cmd);
        }
        if had_input {
            state.last_input_at = Some(Instant::now());
        }
        state.command_rx = Some(rx);
    }

    /// One idle-tick service pass for committed clients while nothing captures:
    /// frame callbacks unblock vsynced clients, requested presentation feedback
    /// is discarded (nothing presents), and the renderer's import cache is
    /// pruned since no render will do it. Shared by the frame timer's idle
    /// branch and the input wake handler, which must not let a fresh keypress
    /// wait out an already-armed long idle deadline.
    fn send_idle_frame_callbacks(state: &mut AppState) {
        let time = state.clock.now();
        for node in &state.output_nodes {
            let mut feedback = OutputPresentationFeedback::new(&node.output);
            for window in state
                .space
                .elements_for_output(&node.output)
                .cloned()
                .collect::<Vec<_>>()
            {
                window.send_frame(&node.output, time, Some(Duration::ZERO), |_, _| {
                    Some(node.output.clone())
                });
                window.take_presentation_feedback(
                    &mut feedback,
                    |_, _| Some(node.output.clone()),
                    |_, _| wp_presentation_feedback::Kind::empty(),
                );
            }
            for layer in layer_map_for_output(&node.output).layers() {
                layer.send_frame(&node.output, time, Some(Duration::ZERO), |_, _| {
                    Some(node.output.clone())
                });
                layer.take_presentation_feedback(
                    &mut feedback,
                    |_, _| Some(node.output.clone()),
                    |_, _| wp_presentation_feedback::Kind::empty(),
                );
            }
            feedback.discarded();
        }
        if let Some(output) = state.primary_output() {
            send_cursor_frame(state, output, time);
        }
        if let Some(renderer) = state.gles_renderer.as_mut() {
            let _ = renderer.cleanup_texture_cache();
        }
    }

    fn handle_thread_command(state: &mut AppState, cmd: ThreadCommand) {
            match cmd {
                ThreadCommand::StartCapture { display_id, callback, settings } => {
                    start_capture_on_display(state, display_id, callback.map(Arc::new), settings);
                }
                ThreadCommand::StopCapture { display_id } => {
                    // Cursor and clipboard callbacks deliberately SURVIVE StopCapture:
                    // captures cycle on client disconnects and setting restarts, and a
                    // copy or cursor change during that gap must still reach Python.
                    // PY_SHUTDOWN gates every use against a finalizing interpreter.
                    stop_capture_on_display(state, display_id);
                }
                ThreadCommand::CreateOutput { id, width, height, x, y, scale, reply } => {
                    let _ = reply.send(create_output_on(state, id, width, height, x, y, scale));
                }
                ThreadCommand::DestroyOutput { id, reply } => {
                    let _ = reply.send(destroy_output_on(state, id));
                }
                ThreadCommand::OutputCapacity { reply } => {
                    let _ = reply
                        .send(state.host.as_ref().map_or(-1, |h| h.output_count() as i64));
                }
                ThreadCommand::RepositionOutput { id, x, y, reply } => {
                    let _ = reply.send(reposition_output_on(state, id, x, y));
                }
                ThreadCommand::ListOutputs { reply } => {
                    let list = state
                        .output_nodes
                        .iter()
                        .map(|n| {
                            let (w, h) = n
                                .output
                                .current_mode()
                                .map(|m| (m.size.w, m.size.h))
                                .unwrap_or((0, 0));
                            (
                                n.id,
                                n.pos.0,
                                n.pos.1,
                                w,
                                h,
                                n.output.current_scale().fractional_scale(),
                                n.capture.is_some(),
                            )
                        })
                        .collect();
                    let _ = reply.send(list);
                }
                ThreadCommand::MoveWindowToOutput { window_id, output_id, reply } => {
                    let window = state
                        .space
                        .elements()
                        .find(|w| {
                            wayland::frontend::window_meta(w)
                                .map(|m| m.id == window_id)
                                .unwrap_or(false)
                        })
                        .cloned();
                    let ok = match window {
                        Some(w) => state.place_window_on_output(&w, output_id),
                        None => false,
                    };
                    let _ = reply.send(ok);
                }
                ThreadCommand::ListWindows { reply } => {
                    use smithay::wayland::shell::xdg::XdgToplevelSurfaceData;
                    let mut list = Vec::new();
                    for window in state.space.elements() {
                        let Some(meta) = wayland::frontend::window_meta(window) else { continue };
                        let (title, app_id) = window
                            .toplevel()
                            .map(|tl| {
                                with_states(tl.wl_surface(), |states| {
                                    states
                                        .data_map
                                        .get::<XdgToplevelSurfaceData>()
                                        .map(|d| {
                                            let a = d.lock().unwrap();
                                            (
                                                a.title.clone().unwrap_or_default(),
                                                a.app_id.clone().unwrap_or_default(),
                                            )
                                        })
                                        .unwrap_or_default()
                                })
                            })
                            .unwrap_or_default();
                        list.push((
                            meta.id,
                            title,
                            app_id,
                            meta.output.load(Ordering::Relaxed),
                            meta.parked.load(Ordering::Relaxed),
                        ));
                    }
                    let _ = reply.send(list);
                }
                ThreadCommand::SetClipboardCallback(cb) => {
                    state.clipboard_callback = Some(cb);
                    // Re-stage a read of the CURRENT selection so a copy made before this
                    // callback was (re)armed is delivered rather than lost; the post-dispatch
                    // drain performs the read (a compositor-owned selection is skipped there).
                    if let Some(mime) = state.current_selection_mime.clone() {
                        state.pending_clipboard_read = Some(mime);
                    }
                }
                ThreadCommand::SetClipboard { mime, data } => {
                    let mimes: Vec<String> = if mime.starts_with("text/plain") {
                        ["text/plain;charset=utf-8", "UTF8_STRING", "text/plain",
                         "STRING", "TEXT"].iter().map(|s| s.to_string()).collect()
                    } else {
                        vec![mime.clone()]
                    };
                    let payload = std::sync::Arc::new((mime, data));
                    smithay::wayland::selection::data_device::set_data_device_selection(
                        &state.dh,
                        &state.seat.clone(),
                        mimes.clone(),
                        payload.clone(),
                    );
                    // Middle-click parity with the X11 clipboard bridge: the same
                    // offer backs the primary selection too.
                    smithay::wayland::selection::primary_selection::set_primary_selection(
                        &state.dh,
                        &state.seat.clone(),
                        mimes,
                        payload,
                    );
                    // The selection is compositor-owned now; a later SetClipboardCallback
                    // must not try to re-read a client source that no longer holds it.
                    state.current_selection_mime = None;
                }
                ThreadCommand::SetCursorCallback(cb) => {
                    let _ = state.cursor_tx.send(CursorJob::SetCallback(cb));
                    state.cursor_callback_set = true;
                    if let Some(icon) = state.current_cursor_icon.clone() {
                        state.send_cursor_image(&icon);
                    } else {
                        // No client has set a cursor yet. The render path treats
                        // None as the default theme cursor, so the first consumer
                        // must receive that same sprite instead of a blank pointer
                        // until the first client cursor event.
                        state.send_cursor_image(&CursorImageStatus::Named(Default::default()));
                    }
                }
                ThreadCommand::KeyboardKeys { events } => {
                    for (scancode, key_state_val) in events {
                        if let Some(host) = state.host.as_ref() {
                            host.key(scancode, key_state_val > 0);
                            continue;
                        }
                        let key_state = if key_state_val > 0 {
                            KeyState::Pressed
                        } else {
                            KeyState::Released
                        };
                        let serial = next_serial();
                        let time = wayland_time();
                        if let Some(keyboard) = state.seat.get_keyboard() {
                            keyboard.input(
                                state,
                                Keycode::new(scancode),
                                key_state,
                                serial,
                                time,
                                |_, _, _| FilterResult::<()>::Forward,
                            );
                        }
                    }
                }
                ThreadCommand::KeyboardKey { scancode, state: key_state_val } => {
                    if let Some(host) = state.host.as_ref() {
                        host.key(scancode, key_state_val > 0);
                        return;
                    }
                    let key_state = if key_state_val > 0 { KeyState::Pressed } else { KeyState::Released };
                    let serial = next_serial();
                    let time = wayland_time();
                    if let Some(keyboard) = state.seat.get_keyboard() {
                        keyboard.input(state, Keycode::new(scancode), key_state, serial, time, |_, _, _| {
                            FilterResult::<()>::Forward
                        });
                    }
                }
                ThreadCommand::SetKeymapString(text) => {
                    // rebuild_base rejects a string that will not compile without touching
                    // the policy, so the seat keymap survives a bad one either way.
                    if state.keymap_policy.rebuild_base(text) {
                        state.apply_keymap_policy();
                    } else {
                        eprintln!("[Wayland] set_keymap_string: keymap failed to compile; keeping current keymap.");
                    }
                }
                ThreadCommand::SetXkbLayout { rules, model, layout, variant, options, reply } => {
                    match crate::wayland::keymap::compile_rmlvo(&rules, &model, &layout, &variant, &options) {
                        Some(text) => {
                            state.keymap_policy.rebuild_base(text);
                            state.apply_keymap_policy();
                            let _ = reply.send(true);
                        }
                        None => {
                            eprintln!("[Wayland] set_xkb_layout: RMLVO ({rules:?}, {model:?}, {layout:?}, {variant:?}, {options:?}) failed to compile.");
                            let _ = reply.send(false);
                        }
                    }
                }
                ThreadCommand::BindKeysyms { keysyms, reply } => {
                    let _ = reply.send(state.bind_keysyms(&keysyms));
                }
                ThreadCommand::SetKeymapOverlay { binds } => {
                    if state.keymap_policy.has_base() {
                        state.keymap_policy.set_manual_overlay(&binds);
                        state.apply_keymap_policy();
                    } else {
                        eprintln!(
                            "[Wayland] set_keymap_overlay: no base keymap to splice onto."
                        );
                    }
                }
                ThreadCommand::GetKeyboardState { reply } => {
                    let (pressed, mods) = state
                        .seat
                        .get_keyboard()
                        .map(|kb| {
                            let pressed: Vec<u32> =
                                kb.pressed_keys().iter().map(|c| c.raw()).collect();
                            let m = kb.modifier_state();
                            let mask = (m.ctrl as u32)
                                | (m.shift as u32) << 1
                                | (m.alt as u32) << 2
                                | (m.logo as u32) << 3
                                | (m.caps_lock as u32) << 4
                                | (m.num_lock as u32) << 5
                                | (m.iso_level3_shift as u32) << 6
                                | (m.iso_level5_shift as u32) << 7;
                            (pressed, mask)
                        })
                        .unwrap_or_default();
                    let _ = reply.send((pressed, mods));
                }
                ThreadCommand::Barrier { reply } => {
                    // The shutdown path fences on this after its StopCaptures:
                    // joining the reaped threads before acknowledging means
                    // nothing that can attach to Python survives past the
                    // fence, so the interpreter never finalizes under a live
                    // callback. Bounded — discard flags are up and senders
                    // dropped, so each chain exits after at most its
                    // in-flight callback.
                    for join in state.encode_reaper.drain(..) {
                        let _ = join.join();
                    }
                    for join in state.deliver_reaper.drain(..) {
                        let _ = join.join();
                    }
                    let _ = reply.send(());
                }
                ThreadCommand::GetXkbKeymap { reply } => {
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
                ThreadCommand::PointerMotion { x, y } => {
                    if let Some(host) = state.host.as_ref() {
                        host.pointer_motion_abs(x, y);
                        return;
                    }
                    let serial = next_serial();
                    let time = wayland_time();
                    // (x, y) are physical union-layout coordinates: each output occupies
                    // the physical rectangle at its layout offset, and the point maps
                    // through the CONTAINING output's scale (clamped into the nearest
                    // output when outside all of them).
                    let p = state.layout_physical_to_logical(x, y);

                    if let Some(pointer) = state.seat.get_pointer() {
                        // Layer surfaces live on the output under the point; their
                        // geometry is output-local, so hit-test with the local point and
                        // report the global location.
                        let layer_hit = |state: &AppState, layers: &[smithay::wayland::shell::wlr_layer::Layer]| {
                            let idx = state.node_idx_under(p)?;
                            let node = &state.output_nodes[idx];
                            let origin = Point::<i32, smithay::utils::Logical>::from(node.pos);
                            let local = (p - origin.to_f64()).to_i32_round();
                            let layer_map = layer_map_for_output(&node.output);
                            for layer in layer_map.layers().rev() {
                                if layers.contains(&layer.layer())
                                    && let Some(bbox) = layer_map.layer_geometry(layer)
                                    && bbox.contains(local) {
                                            return Some((
                                                FocusTarget::LayerSurface(layer.clone()),
                                                (bbox.loc + origin).to_f64(),
                                            ));
                                        }
                            }
                            None
                        };

                        let mut under = layer_hit(state, &[
                            smithay::wayland::shell::wlr_layer::Layer::Overlay,
                            smithay::wayland::shell::wlr_layer::Layer::Top,
                        ]);

                        if under.is_none() {
                            under = state.space.element_under(p).map(|(window, loc)| {
                                (FocusTarget::Window(window.clone()), loc.to_f64())
                            });
                        }

                        if under.is_none() {
                            under = layer_hit(state, &[
                                smithay::wayland::shell::wlr_layer::Layer::Bottom,
                                smithay::wayland::shell::wlr_layer::Layer::Background,
                            ]);
                        }

                        pointer.motion(state, under, &MotionEvent { location: p, serial, time });
                        pointer.frame(state);
                    }
                }
                ThreadCommand::PointerRelativeMotion { dx, dy } => {
                    if let Some(host) = state.host.as_ref() {
                        host.pointer_motion_rel(dx, dy);
                        return;
                    }
                    let utime = wayland_utime();
                    let time = wayland_time();
                    let serial = next_serial();

                    if let Some(pointer) = state.seat.get_pointer() {
                        let current_pos = pointer.current_location();
                        let new_pos = state.clamp_logical(
                            (current_pos.x + dx, current_pos.y + dy).into(),
                        );

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
                ThreadCommand::PointerButton { btn, state: btn_state_val } => {
                    if let Some(host) = state.host.as_ref() {
                        host.pointer_button(btn, btn_state_val > 0);
                        return;
                    }
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
                ThreadCommand::PointerAxis { x, y } => {
                    if let Some(host) = state.host.as_ref() {
                        host.pointer_axis(x, y);
                        return;
                    }
                    let time = wayland_time();
                    
                    if let Some(pointer) = state.seat.get_pointer() {
                        let mut frame = AxisFrame::new(time).source(AxisSource::Wheel);

                        if x != 0.0 { 
                            frame = frame
                                .value(Axis::Horizontal, x)
                                .v120(Axis::Horizontal, (x * SCROLL_V120_PER_UNIT) as i32);
                        }
                        
                        if y != 0.0 { 
                            frame = frame
                                .value(Axis::Vertical, y)
                                .v120(Axis::Vertical, (y * SCROLL_V120_PER_UNIT) as i32);
                        }

                        if x != 0.0 || y != 0.0 {
                            pointer.axis(state, frame);
                            pointer.frame(state);
                        }
                    }
                }
                ThreadCommand::UpdateCursorConfig { render_on_framebuffer } => {
                    state.render_cursor_on_framebuffer = render_on_framebuffer;
                    if let Some(host) = state.host.as_ref() {
                        host.set_cursor_painting(render_on_framebuffer);
                    }
                }
                ThreadCommand::SetCursorSize { size, reply } => {
                    if size <= 0 {
                        let _ = reply.send(false);
                    } else {
                        state.cursor_helper = Cursor::load(size);
                        let _ = state.cursor_tx.send(CursorJob::SetSize(size));
                        // The burned-in cursor changed size; force a repaint everywhere so
                        // a static screen doesn't keep showing the old sprite.
                        for node in state.output_nodes.iter_mut() {
                            if let Some(cap) = node.capture.as_mut() {
                                cap.needs_full_render = true;
                            }
                        }
                        let _ = reply.send(true);
                    }
                }
                ThreadCommand::RequestIdr { display_id } => {
                    if let Some(idx) = state.node_idx_for_id(display_id)
                        && let Some(cap) = state.output_nodes[idx].capture.as_mut() {
                            cap.request_idr();
                        }
                }
                ThreadCommand::UpdateRate { display_id, bitrate_kbps, vbv_multiplier, fps } => {
                    if let Some(idx) = state.node_idx_for_id(display_id)
                        && let Some(cap) = state.output_nodes[idx].capture.as_mut() {
                            if let Some(b) = bitrate_kbps { cap.settings.video_bitrate_kbps = b; }
                            if let Some(v) = vbv_multiplier { cap.settings.video_vbv_multiplier = v; }
                            if let Some(f) = fps && f > 0.0 { cap.settings.target_fps = f; }
                            if let Some(GpuEncoder::Nvenc(enc)) = cap.video_encoder.as_mut() {
                                enc.reconfigure_rate(&cap.settings);
                            }
                            if let Some(GpuEncoder::Vaapi(enc)) = cap.video_encoder.as_mut()
                                && let Err(e) = enc.reconfigure_rate(&cap.settings) {
                                    // The failed re-open left no codec context: the next
                                    // tick's encode fails, and a full streak makes that
                                    // failure run the recovery ladder at once.
                                    eprintln!("[Wayland] VAAPI rate reconfigure failed: {e}");
                                    cap.hw_error_streak = HW_ERROR_RECOVERY_THRESHOLD - 1;
                                }
                            let c = &cap.encode_controls;
                            c.bitrate_kbps.store(cap.settings.video_bitrate_kbps, Ordering::Relaxed);
                            c.vbv_mult_milli.store(
                                (cap.settings.video_vbv_multiplier * 1000.0).round() as i32,
                                Ordering::Relaxed,
                            );
                            c.fps_milli.store(
                                (cap.settings.target_fps.max(1.0) * 1000.0) as u64,
                                Ordering::Relaxed,
                            );
                            c.rate_dirty.store(true, Ordering::Release);
                            if display_id == 0 {
                                state.settings.video_bitrate_kbps = cap.settings.video_bitrate_kbps;
                                state.settings.video_vbv_multiplier = cap.settings.video_vbv_multiplier;
                                state.settings.target_fps = cap.settings.target_fps;
                            }
                        }
                }
                ThreadCommand::UpdateTunables { display_id, tunables: t } => {
                    state.render_cursor_on_framebuffer = t.capture_cursor;
                    if let Some(host) = state.host.as_ref() {
                        host.set_cursor_painting(t.capture_cursor);
                    }
                    let _ = state.cursor_tx.send(CursorJob::SetSizeCap(t.cursor_size_cap));
                    if display_id == 0 {
                        t.apply_to(&mut state.settings);
                    }
                    if let Some(idx) = state.node_idx_for_id(display_id)
                        && let Some(cap) = state.output_nodes[idx].capture.as_mut() {
                            t.apply_to(&mut cap.settings);
                            *cap.encode_controls.tunables.lock().unwrap() = Some(t);
                            cap.encode_controls.tunables_dirty.store(true, Ordering::Release);
                        }
                }
                ThreadCommand::CuScreenshot { display_id, resp } => {
                    if state.node_idx_for_id(display_id).is_some() {
                        state.pending_screenshot = Some((display_id, resp));
                    } else {
                        let _ = resp.send(Err(format!("Unknown display: {display_id}")));
                    }
                }
                ThreadCommand::CuCursorPosition { resp } => {
                    let pos = state.seat.get_pointer()
                        .map(|p| p.current_location())
                        .unwrap_or_else(|| (0.0f64, 0.0f64).into());
                    let _ = resp.send(state.layout_logical_to_physical(pos));
                }
                ThreadCommand::CuGetInfo { display_id, resp } => {
                    // A host-capture start whose mode the host has not answered yet:
                    // the read parks until it has (the reply then carries the size
                    // actually captured), which is what makes it a barrier.
                    match state.host_layout_pending.get_mut(&display_id) {
                        Some(p) => p.geometry_waiters.push(resp),
                        None => {
                            let _ = resp.send(realized_geometry(state, display_id));
                        }
                    }
                }
            }
    }

    state.command_rx = Some(command_rx);
    event_loop
        .handle()
        .insert_source(wake_rx, |_, _, state| {
            drain_thread_commands(state);
            // Input landing while the frame timer sits on its long idle
            // deadline must not wait it out: service the frame callbacks now
            // (at most at frame pace) so the app's echo repaint starts with
            // the keypress; the timer's own next fire re-arms the short pace.
            if state.frame_idle_long
                && state
                    .last_input_at
                    .is_some_and(|t| t.elapsed() < Duration::from_millis(50))
                && state
                    .last_idle_service_at
                    .is_none_or(|t| t.elapsed() >= Duration::from_millis(16))
            {
                state.last_idle_service_at = Some(Instant::now());
                send_idle_frame_callbacks(state);
            }
        })
        .unwrap();

    let source = match ListeningSocketSource::new_auto() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[Wayland] compositor thread aborting: could not bind a wayland-N socket (XDG_RUNTIME_DIR unset/full?): {e}");
            return;
        }
    };
    let socket_name = source.socket_name().to_string_lossy().into_owned();
    println!("[Wayland] Socket listening on: {:?}", socket_name);
    // Writing the process environment races any concurrent getenv, which is why
    // it is unsafe: nothing here can stop a thread already running from reading
    // it. This one write happens during compositor bring-up, before any capture
    // or encoder thread exists, and the value is what children of this process
    // and the backend probe below need to find the compositor.
    unsafe { std::env::set_var("WAYLAND_DISPLAY", &socket_name) };
    publish_socket_name(&socket_name);

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
    event_loop
        .handle()
        .insert_source(timer, move |_, _, state| {
            // Apply queued commands (input above all) BEFORE the render/encode work: a
            // command that raced the timer wakeup would otherwise wait out the whole tick.
            reap_dead_host(state);
            drain_thread_commands(state);
            reconcile_host_layouts(state);
            // Deliver and deferred encode threads of stopped captures are
            // joined only once they report finished (a plain atomic load), so
            // the tick never waits on one; whatever remains at shutdown, the
            // Barrier drain joins.
            let mut i = 0;
            while i < state.deliver_reaper.len() {
                if state.deliver_reaper[i].is_finished() {
                    let _ = state.deliver_reaper.swap_remove(i).join();
                } else {
                    i += 1;
                }
            }
            let mut i = 0;
            while i < state.encode_reaper.len() {
                if state.encode_reaper[i].is_finished() {
                    let _ = state.encode_reaper.swap_remove(i).join();
                } else {
                    i += 1;
                }
            }
            let loop_start_time = Instant::now();
            state.space.refresh();

            let now = Instant::now();
            let elapsed = now.duration_since(state.last_log_time).as_secs_f64();
            if elapsed >= 1.0 {
                // Memory is read here rather than per tick: it is reported, not acted on,
                // and /proc plus a /dev/shm walk have no place in the render path.
                let mut mem: Option<(usize, u64)> = None;
                for node in &state.output_nodes {
                    let Some(cap) = node.capture.as_ref() else { continue };
                    let frames = cap.encode_stats.frames.swap(0, Ordering::Relaxed);
                    let stripes = cap.encode_stats.stripes.swap(0, Ordering::Relaxed);
                    if cap.settings.debug_logging {
                        let actual_fps = frames as f64 / elapsed;
                        let stripes_per_sec = stripes as f64 / elapsed;
                        let mode_str = cap.encode_stats.desc.lock().unwrap().clone();
                        let n_stripes = cap.encode_stats.n_stripes.load(Ordering::Relaxed);
                        let (current_rss, shm_usage) = *mem
                            .get_or_insert_with(|| (get_process_rss_bytes(), get_shm_usage_bytes()));

                        println!("Display: {} Res: {}x{} Mode: {} Stripes: {} EncFPS: {:.2} EncStripes/s: {:.2} Mem: {}MB SHM: {}MB",
                            node.id, cap.settings.width, cap.settings.height, mode_str, n_stripes, actual_fps, stripes_per_sec, current_rss / 1024 / 1024, shm_usage / 1024 / 1024);
                    }
                }
                state.last_log_time = now;
            }

            let any_capturing = state.output_nodes.iter().any(|n| n.capture.is_some());
            let any_copy_frame = state.copy_sessions.iter().any(|cs| cs.pending.is_some());
            if !any_capturing && state.pending_screenshot.is_none() && !any_copy_frame {
                // No render/encode work, but committed clients still need their
                // frame callbacks: a vsynced client (FIFO Vulkan present, games, a
                // nested compositor's own clients) otherwise blocks in its swap
                // until a viewer attaches — apps appear frozen whenever nobody is
                // watching.
                //
                // Slowly, though: at a viewing rate the clients draw at a viewing rate
                // too, and every frame they hand over is one that nothing renders,
                // encodes or looks at. Unblocking them is the whole purpose here, and
                // that costs a callback every so often rather than sixty a second.
                state.last_idle_service_at = Some(Instant::now());
                send_idle_frame_callbacks(state);
                // A parked capture session is an attached consumer that may request a
                // frame at any moment: tick at frame pace so that request waits at most
                // one frame, not a quarter second. Input gets the same window: apps
                // pacing on frame callbacks see their echo promptly after a keypress,
                // or the idle rate alone turns typing into a 250ms stutter. The wake
                // handler covers the first keypress landing while the long deadline
                // is already armed.
                let post_input = state
                    .last_input_at
                    .is_some_and(|t| t.elapsed() < Duration::from_secs(1));
                let idle = if !state.copy_sessions.is_empty() || post_input {
                    Duration::from_millis(16)
                } else {
                    IDLE_FRAME_INTERVAL
                };
                state.frame_idle_long = idle == IDLE_FRAME_INTERVAL;
                return TimeoutAction::ToDuration(idle);
            }
            state.frame_idle_long = false;

            // Render every output that needs it. The nodes are taken out of the state so
            // each per-output render can borrow the shared renderer/space alongside its
            // own damage tracker and buffers.
            let mut nodes = std::mem::take(&mut state.output_nodes);
            let mut any_pool_busy = false;
            for node in nodes.iter_mut() {
                if render_node_tick(state, node) {
                    any_pool_busy = true;
                }
            }
            state.output_nodes = nodes;

            if any_pool_busy {
                return TimeoutAction::ToDuration(Duration::from_millis(1));
            }
            let work_elapsed = loop_start_time.elapsed();
            let max_fps = state
                .output_nodes
                .iter()
                .filter_map(|n| n.capture.as_ref().map(|c| c.settings.target_fps))
                .fold(0.0f64, f64::max);
            let raw_fps = if max_fps > 0.0 { max_fps } else { state.settings.target_fps };
            // Settings are sanitized at the Python boundary; this final guard keeps a
            // non-finite value from ever reaching Duration::from_secs_f64 (panics on NaN).
            let fps = if raw_fps.is_finite() && raw_fps > 0.0 { raw_fps.min(MAX_FPS) } else { DEFAULT_FPS };
            let target_frame_duration = Duration::from_secs_f64(1.0 / fps);
            let wait_duration = target_frame_duration.saturating_sub(work_elapsed);
            let final_wait = if wait_duration.as_millis() < 1 { Duration::from_millis(1) } else { wait_duration };
            TimeoutAction::ToDuration(final_wait)
        })
        .expect("Failed to init capture timer");

    event_loop
        .handle()
        .insert_source(Generic::new(display, Interest::READ, Mode::Level), |_, display, _state| {
            // A single misbehaving client must not take the compositor (and every other
            // session) down with it.
            if let Err(e) = unsafe { display.get_mut().dispatch_clients(_state) } {
                eprintln!("[Wayland] client dispatch error: {e:?}");
            }
            Ok(PostAction::Continue)
        })
        .unwrap();

    crate::computer_use::register_wayland_backend(command_tx.clone());
    crate::computer_use::spawn_cu_from_env();

    let _ = event_loop.run(None, &mut state, |state| {
        state.process_pending_clipboard_read();
        state.flush_pending_cursor();
        let _ = state.dh.flush_clients();
    });
}

/// Zero-copy encoded-frame handoff to Python. Owns the encoded `Vec<u8>` and
/// exposes it read-only via the buffer protocol, so `bytes(frame)` /
/// `memoryview(frame)` alias the Rust buffer instead of copying. Carries the
/// four stripe-metadata ints as Python attributes.
#[pyclass]
struct StripeFrame {
    data: Arc<Vec<u8>>,
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
    /// Hot-path constructor: shares the encoder's buffer by `Arc` (no copy) and carries stripe
    /// metadata as attributes, so the consumer can read it without parsing a header
    /// (required for omit_stripe_headers).
    fn new_owned_meta(data: Arc<Vec<u8>>, data_type: i32, stripe_y_start: i32, stripe_height: i32, frame_id: i32) -> Self {
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
        Self { data: Arc::new(data), data_type, stripe_y_start, stripe_height, frame_id }
    }

    fn __len__(&self) -> usize {
        self.data.len()
    }

    /// Expose the owned bytes read-only through the Python buffer protocol.
    ///
    /// `PyBuffer_FillInfo` INCREFs `slf` into `view->obj`, pinning the `Vec` until every view is
    /// released, so memoryviews can outlive the Python `frame` handle.
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        let r = unsafe {
            pyo3::ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                slf.data.as_ptr() as *mut std::os::raw::c_void,
                slf.data.len() as pyo3::ffi::Py_ssize_t,
                1,
                flags,
            )
        };
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
    /// Wakes the calloop after each command send: the command channel itself is drained in
    /// place by the compositor thread (render tick and wake handler), not registered as its
    /// own source, so input never waits behind an in-flight render tick's timer wakeup.
    wake_tx: smithay::reexports::calloop::channel::Sender<()>,
}

impl WaylandBackend {
    fn send(&self, cmd: ThreadCommand) -> Result<(), String> {
        self.tx.send(cmd).map_err(|e| e.to_string())?;
        let _ = self.wake_tx.send(());
        Ok(())
    }
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
        let (wake_tx, wake_rx) = smithay::reexports::calloop::channel::channel();
        let cu_tx = tx.clone();
        thread::spawn(move || {
            crate::boost_thread_priority(-15);
            run_wayland_thread(WaylandThreadConfig {
                command_rx: rx,
                wake_rx,
                command_tx: cu_tx,
                initial_width: width,
                initial_height: height,
                explicit_dri_node: dri_node,
                auto_gpu_selected,
                cursor_size,
            });
        });
        WaylandBackend { tx, wake_tx }
    }

    /// Begin a capture with the given frame callback and settings. The target display is
    /// the settings' `display_id` attribute (absent = 0, the primary); each display id runs
    /// at most one capture, independent of every other display's.
    ///
    /// Issuing the start also clears the interpreter-teardown gate: starting from Python proves the
    /// interpreter is live again after a manual atexit sweep.
    fn start_capture(&self, callback: Py<PyAny>, settings: &Bound<'_, PyAny>) -> PyResult<()> {
        let rust_settings = extract_settings(settings)?;
        let display_id = read_display_id(settings);

        PY_SHUTDOWN.store(false, Ordering::Relaxed);
        self.send(ThreadCommand::StartCapture { display_id, callback: Some(callback), settings: rust_settings })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to send start command: {}", e)))?;
        Ok(())
    }

    /// Stop the capture bound to `display_id` (default: the primary display).
    #[pyo3(signature = (display_id = 0))]
    fn stop_capture(&self, display_id: u32) -> PyResult<()> {
        self.send(ThreadCommand::StopCapture { display_id })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to send stop command: {}", e)))?;
        Ok(())
    }

    /// Create an additional output (`WxH` physical pixels at fractional `scale`) mapped into
    /// the layout at offset `(x, y)`; `id` is the display key used by every per-display API.
    /// False when the id is taken, the geometry/scale is invalid, the rectangle overlaps a
    /// live output, or the GPU render target cannot be allocated. Capture reconfigures
    /// (`start_capture` on an existing output) resize without this validation, so a
    /// multi-step relayout must stay overlap-free at every step by caller ordering.
    // The parameter list is the Python signature; grouping it would change the ABI.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (id, width, height, x = 0, y = 0, scale = 1.0))]
    fn create_output(
        &self,
        py: Python<'_>,
        id: u32,
        width: i32,
        height: i32,
        x: i32,
        y: i32,
        scale: f64,
    ) -> PyResult<bool> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<bool>();
        self.send(ThreadCommand::CreateOutput { id, width, height, x, y, scale, reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create output: {}", e)))?;
        Ok(py
            .detach(move || reply_rx.recv_timeout(Duration::from_secs(2)))
            .unwrap_or(false))
    }

    /// Destroy a secondary output: its capture ends cleanly and its windows relocate to the
    /// primary output. False for the primary (id 0) or an unknown id.
    fn destroy_output(&self, py: Python<'_>, id: u32) -> PyResult<bool> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<bool>();
        self.send(ThreadCommand::DestroyOutput { id, reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to destroy output: {}", e)))?;
        Ok(py
            .detach(move || reply_rx.recv_timeout(Duration::from_secs(2)))
            .unwrap_or(false))
    }

    /// Move an existing output (the primary, id 0, included) to layout offset `(x, y)`;
    /// its windows, absolute input injection, and cursor compositing follow. False for an
    /// unknown id or a destination overlapping a live output. Capture reconfigures
    /// (`start_capture` on an existing output) resize without this validation, so a
    /// multi-step relayout must stay overlap-free at every step by caller ordering.
    fn reposition_output(&self, py: Python<'_>, id: u32, x: i32, y: i32) -> PyResult<bool> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<bool>();
        self.send(ThreadCommand::RepositionOutput { id, x, y, reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to reposition output: {}", e)))?;
        Ok(py
            .detach(move || reply_rx.recv_timeout(Duration::from_secs(2)))
            .unwrap_or(false))
    }

    /// Every live output as `(id, x, y, width, height, scale, capturing)` — width/height in
    /// physical pixels, `(x, y)` the layout offset.
    fn list_outputs(&self, py: Python<'_>) -> PyResult<Vec<OutputDesc>> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.send(ThreadCommand::ListOutputs { reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to list outputs: {}", e)))?;
        Ok(py
            .detach(move || reply_rx.recv_timeout(Duration::from_secs(2)))
            .unwrap_or_default())
    }

    /// How many displays this backend can back with real content: -1 when
    /// self-compositing (outputs are created on demand, no fixed bound), otherwise the
    /// host compositor's output count. Host-capture mode reports -1 until the first
    /// capture start establishes the host session; 0 when the backend is unresponsive.
    fn output_capacity(&self, py: Python<'_>) -> PyResult<i64> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<i64>();
        self.send(ThreadCommand::OutputCapacity { reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to query output capacity: {}", e)))?;
        Ok(py
            .detach(move || reply_rx.recv_timeout(Duration::from_secs(2)))
            .unwrap_or(0))
    }

    /// Move the window with the given id onto output `output_id`, fullscreened at that
    /// output's logical size. False for an unknown window or output id.
    fn move_window_to_output(&self, py: Python<'_>, window_id: u32, output_id: u32) -> PyResult<bool> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<bool>();
        self.send(ThreadCommand::MoveWindowToOutput { window_id, output_id, reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to move window: {}", e)))?;
        Ok(py
            .detach(move || reply_rx.recv_timeout(Duration::from_secs(2)))
            .unwrap_or(false))
    }

    /// Every mapped window as `(window_id, title, app_id, output_id, waiting)`. A waiting
    /// window is tagged for that output but mapped clear of every one of them, holding its
    /// size until an output exists for it — a nested session's spare screens.
    fn list_windows(&self, py: Python<'_>) -> PyResult<Vec<WindowDesc>> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.send(ThreadCommand::ListWindows { reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to list windows: {}", e)))?;
        Ok(py
            .detach(move || reply_rx.recv_timeout(Duration::from_secs(2)))
            .unwrap_or_default())
    }

    fn set_cursor_callback(&self, callback: Py<PyAny>) -> PyResult<()> {
        self.send(ThreadCommand::SetCursorCallback(callback))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set cursor callback: {}", e)))?;
        Ok(())
    }

    /// Recreate the cursor theme at `size` pixels — no restart: subsequent named-cursor
    /// callbacks and the burned-in cursor overlay render at the new size. False for a
    /// non-positive size.
    fn set_cursor_size(&self, py: Python<'_>, size: i32) -> PyResult<bool> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<bool>();
        self.send(ThreadCommand::SetCursorSize { size, reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set cursor size: {}", e)))?;
        Ok(py
            .detach(move || reply_rx.recv_timeout(Duration::from_secs(2)))
            .unwrap_or(false))
    }

    /// cb(mime: str, data: bytes) fires when a client app copies to the clipboard.
    fn set_clipboard_callback(&self, callback: Py<PyAny>) -> PyResult<()> {
        self.send(ThreadCommand::SetClipboardCallback(callback))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set clipboard callback: {}", e)))?;
        Ok(())
    }

    /// Compositor-side clipboard offer: serve `data` as `mime` to pasting clients.
    fn set_clipboard(&self, mime: String, data: Vec<u8>) -> PyResult<()> {
        self.send(ThreadCommand::SetClipboard { mime, data })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set clipboard: {}", e)))?;
        Ok(())
    }

    fn inject_key(&self, scancode: u32, state: u32) -> PyResult<()> {
        self.send(ThreadCommand::KeyboardKey { scancode, state })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject key: {}", e)))?;
        Ok(())
    }

    /// Inject an ordered run of `(keycode, state)` events as one message, so a paste
    /// costs one channel send and one wake instead of one per event.
    fn inject_keys(&self, events: Vec<(u32, u32)>) -> PyResult<()> {
        if events.is_empty() {
            return Ok(());
        }
        self.send(ThreadCommand::KeyboardKeys { events })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject keys: {}", e)))?;
        Ok(())
    }

    /// Swap the seat keyboard's xkb keymap (XKB_KEYMAP_FORMAT_TEXT_V1 text). The caller
    /// owns keysym-to-keycode policy: define keycodes here, then press them via
    /// `inject_key`. Ordered with key events on the one compositor channel.
    /// Bind explicit `(keycode, keysym)` pairs onto the current base keymap in ONE swap.
    ///
    /// The caller owns the assignment — which keysym goes to which keycode, and when to
    /// recycle one — because that tracks layouts and user reports. This end only assembles
    /// the xkb text and delivers it, reusing the installed base rather than recompiling a
    /// re-supplied one. False when no base keymap is installed yet.
    fn set_keymap_overlay(&self, binds: Vec<(u32, u32)>) -> PyResult<()> {
        self.send(ThreadCommand::SetKeymapOverlay { binds })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set overlay: {}", e)))?;
        Ok(())
    }

    fn set_keymap_string(&self, text: String) -> PyResult<()> {
        self.send(ThreadCommand::SetKeymapString(text))
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
        self.send(ThreadCommand::GetXkbKeymap { reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to request keymap: {}", e)))?;
        let result = py.detach(move || reply_rx.recv_timeout(Duration::from_secs(2)));
        match result {
            Ok(s) => Ok(s),
            Err(_) => Ok(String::new()),
        }
    }

    fn inject_mouse_move(&self, x: f64, y: f64) -> PyResult<()> {
        self.send(ThreadCommand::PointerMotion { x, y })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject motion: {}", e)))?;
        Ok(())
    }

    fn inject_relative_mouse_move(&self, dx: f64, dy: f64) -> PyResult<()> {
        self.send(ThreadCommand::PointerRelativeMotion { dx, dy })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject relative motion: {}", e)))?;
        Ok(())
    }

    fn inject_mouse_button(&self, btn: u32, state: u32) -> PyResult<()> {
        self.send(ThreadCommand::PointerButton { btn, state })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject button: {}", e)))?;
        Ok(())
    }

    fn inject_mouse_scroll(&self, x: f64, y: f64) -> PyResult<()> {
        self.send(ThreadCommand::PointerAxis { x, y })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject axis: {}", e)))?;
        Ok(())
    }

    fn set_cursor_rendering(&self, enabled: bool) -> PyResult<()> {
        self.send(ThreadCommand::UpdateCursorConfig { render_on_framebuffer: enabled })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set cursor config: {}", e)))?;
        Ok(())
    }

    /// Forces an IDR/keyframe on the next captured frame so a (re)connecting client
    /// or a decoder reset can resume immediately. With the default infinite GOP this
    /// is the only recovery path, so every consumer that can lose decoder state must
    /// call it. No-op cost on the JPEG/software path (keyframes are N/A).
    #[pyo3(signature = (display_id = 0))]
    fn request_idr_frame(&self, display_id: u32) -> PyResult<()> {
        self.send(ThreadCommand::RequestIdr { display_id })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to request IDR: {}", e)))?;
        Ok(())
    }

    /// Apply a live bitrate (kbps) / VBV (kb) / framerate change to the given display's
    /// running capture.
    #[pyo3(signature = (bitrate_kbps = None, vbv_multiplier = None, fps = None, display_id = 0))]
    fn update_rate(&self, bitrate_kbps: Option<i32>, vbv_multiplier: Option<f64>, fps: Option<f64>, display_id: u32) -> PyResult<()> {
        self.send(ThreadCommand::UpdateRate { display_id, bitrate_kbps, vbv_multiplier, fps })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to update rate: {}", e)))?;
        Ok(())
    }

    /// Set the seat's BASE xkb layout from RMLVO names at runtime (empty strings select the
    /// xkbcommon defaults). Returns whether the layout compiled and was applied; overlay binds
    /// rebuild on top with their keycodes unchanged.
    #[pyo3(signature = (layout, variant = String::new(), options = String::new(), model = String::new(), rules = String::new()))]
    fn set_xkb_layout(
        &self,
        py: Python<'_>,
        layout: String,
        variant: String,
        options: String,
        model: String,
        rules: String,
    ) -> PyResult<bool> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<bool>();
        self.send(ThreadCommand::SetXkbLayout { rules, model, layout, variant, options, reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set layout: {}", e)))?;
        Ok(py
            .detach(move || reply_rx.recv_timeout(Duration::from_secs(2)))
            .unwrap_or(false))
    }

    /// Debug/verification readback of the seat keyboard: `(pressed_keycodes, modifier_mask)`
    /// with mask bits 1 ctrl, 2 shift, 4 alt, 8 logo, 16 caps, 32 num, 64 altgr, 128 level5.
    fn get_keyboard_state(&self, py: Python<'_>) -> PyResult<(Vec<u32>, u32)> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<(Vec<u32>, u32)>();
        self.send(ThreadCommand::GetKeyboardState { reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to read keyboard state: {}", e)))?;
        Ok(py
            .detach(move || reply_rx.recv_timeout(Duration::from_secs(2)))
            .unwrap_or_default())
    }

    /// The capture geometry actually live on the given display: `(width, height, scale)` in
    /// physical pixels, or `None` when the compositor did not answer in time. Reflects any
    /// degrade a `start_capture` performed (H.264 even-masking, GBM allocation failure keeping
    /// the previous mode, a host that kept its own mode), and the command channel is FIFO, so
    /// calling this after `start_capture` returns what that start realized. A host-capture
    /// start answers once the host has ruled on the requested mode (the compositor thread
    /// keeps serving input meanwhile), so a slow-but-live start reads its real size instead
    /// of being reported as a timeout; the `None` a genuine timeout returns lets the caller
    /// treat the geometry as unknown, not as "nothing to reconcile", which a `(0, 0, 0.0)`
    /// sentinel could not.
    #[pyo3(signature = (display_id = 0))]
    fn get_realized_geometry(
        &self,
        py: Python<'_>,
        display_id: u32,
    ) -> PyResult<Option<(i32, i32, f64)>> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<(i32, i32, f64)>();
        self.send(ThreadCommand::CuGetInfo { display_id, resp: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to read geometry: {}", e)))?;
        Ok(py
            .detach(move || reply_rx.recv_timeout(GEOMETRY_BARRIER_TIMEOUT))
            .ok())
    }

    /// Lifecycle of `display_id`'s capture as `(state, last_error)`: `state` is `"running"`
    /// (pipeline live), `"failed"` (start left no live pipeline) or `"idle"` (never started
    /// / stopped clean); `last_error` is the reason a start failed, or a caveat a live
    /// capture came up with (host connect refused -> local compositing, a hardware encoder
    /// that fell back to CPU, a refused resize). Read straight from the recorded outcome, so
    /// pair it with a prior `get_realized_geometry` when the ordering after a start matters.
    #[pyo3(signature = (display_id = 0))]
    fn capture_state(&self, display_id: u32) -> (String, Option<String>) {
        wayland_capture_state(display_id)
    }
}

/// Bound on the geometry read-back barrier. Behind a host-capture start the read is held
/// until the host has ruled on the requested mode, which the host's control thread bounds
/// by `LAYOUT_DEADLINE`, so this clears that with margin or a live-but-slow start would read
/// as a timeout.
const GEOMETRY_BARRIER_TIMEOUT: Duration = Duration::from_secs(6);

/// `(state, last_error)` for `display_id`'s Wayland capture, from the shared liveness and
/// outcome maps (no command round-trip).
fn wayland_capture_state(display_id: u32) -> (String, Option<String>) {
    let running = wayland_alive().lock().unwrap().contains(&display_id);
    let last_error = wayland_capture_err().lock().unwrap().get(&display_id).cloned();
    let state = if running {
        "running"
    } else if last_error.is_some() {
        "failed"
    } else {
        "idle"
    };
    (state.to_string(), last_error)
}

impl WaylandBackend {
    fn update_tunables(&self, display_id: u32, t: LiveTunables) -> PyResult<()> {
        self.send(ThreadCommand::UpdateTunables { display_id, tunables: t })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to update tunables: {}", e)))?;
        Ok(())
    }
}

/// The optional `display_id` attribute on a settings object (absent/invalid = 0, the
/// primary display).
fn read_display_id(settings: &Bound<'_, PyAny>) -> u32 {
    settings
        .getattr("display_id")
        .ok()
        .and_then(|v| v.extract::<u32>().ok())
        .unwrap_or(0)
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
    StripeFrame::new_owned_meta(Arc::new(data), data_type, stripe_y_start, stripe_height, frame_id)
}

/// Capture configuration read by `start_capture` (each field by attribute name via
/// `extract_settings`, so the field names must match exactly). Declared `dict` so callers
/// can stash extra attributes not listed here.
#[pyclass(dict)]
struct CaptureSettings {
    /// Wayland display key this capture binds to (0 = primary output); ignored on X11.
    #[pyo3(get, set)] display_id: u32,
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
    /// Wayland display of an EXTERNAL compositor to capture (host-capture mode).
    #[pyo3(get, set)] wayland_host_display: Py<PyAny>,
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
            display_id: 0,
            capture_width: 1920, capture_height: 1080, scale: 1.0, capture_x: 0, capture_y: 0,
            target_fps: 60.0, jpeg_quality: 85, paint_over_jpeg_quality: 95,
            use_paint_over_quality: false, paint_over_trigger_frames: 10,
            damage_block_threshold: 15, damage_block_duration: 30, output_mode: 0,
            video_crf: 25, video_paintover_crf: 18, video_paintover_burst_frames: 5,
            video_fullcolor: false, video_fullframe: false, video_streaming_mode: false,
            capture_cursor: false, watermark_path: py.None(), watermark_location_enum: 0,
            encode_node_index: -2, use_cpu: false, debug_logging: false,
            video_cbr_mode: false, video_bitrate_kbps: 4000, video_vbv_multiplier: 0.0,
            keyframe_interval_s: 0.0,
            video_min_qp: 0, video_max_qp: 0,
            auto_adjust_screen_capture_size: false, omit_stripe_headers: false,
            encode_node_path: py.None(),
            render_node_path: py.None(), auto_gpu: py.None(), use_wayland: py.None(),
            recording_socket: py.None(), wayland_host_display: py.None(),
            cursor_size: -1, cursor_size_cap: 32,
        }
    }
}

/// Process-wide Wayland backend: input and capture share ONE compositor (constructed lazily).
static WAYLAND_BACKEND: OnceLock<Mutex<Option<Py<WaylandBackend>>>> = OnceLock::new();
/// The compositor's auto-picked socket name (e.g. "wayland-1"), published by the compositor
/// thread once its listening socket exists. `ListeningSocketSource::new_auto` binds the first
/// FREE wayland-N, which need not match any configured index — consumers must read the real
/// name from here instead of assuming one.
static WAYLAND_SOCKET_NAME: Mutex<Option<String>> = Mutex::new(None);
static WAYLAND_SOCKET_CV: Condvar = Condvar::new();

fn publish_socket_name(name: &str) {
    *WAYLAND_SOCKET_NAME.lock().unwrap() = Some(name.to_string());
    WAYLAND_SOCKET_CV.notify_all();
}

/// Wait (bounded) for the compositor thread to publish its socket name.
fn wait_socket_name(timeout: Duration) -> Option<String> {
    let deadline = Instant::now() + timeout;
    let mut g = WAYLAND_SOCKET_NAME.lock().unwrap();
    loop {
        if let Some(name) = g.as_ref() {
            return Some(name.clone());
        }
        let now = Instant::now();
        if now >= deadline {
            return None;
        }
        let (gg, _) = WAYLAND_SOCKET_CV.wait_timeout(g, deadline - now).unwrap();
        g = gg;
    }
}
/// Cursor callback registered before the backend exists (selkies registers it pre-start);
/// applied when the backend is created, which is deferred to capture start so the real
/// render node (not a placeholder) reaches the compositor.
static PENDING_CURSOR_CALLBACK: Mutex<Option<Py<PyAny>>> = Mutex::new(None);
/// Interpreter-teardown gate, set by the atexit sweep: the detached compositor and delivery
/// threads must never attach to a finalizing interpreter (aborts the process pre-3.13).
/// Cleared by a fresh capture start (only a live interpreter can start one).
pub(crate) static PY_SHUTDOWN: AtomicBool = AtomicBool::new(false);
/// Per-display capture ownership: display id -> the ScreenCapture id that owns that
/// display's capture. Only the owner may stop it, so an input-only or stale instance can't
/// tear down a live capture.
static WAYLAND_OWNERS: OnceLock<Mutex<std::collections::HashMap<u32, u64>>> = OnceLock::new();

fn wayland_owners() -> &'static Mutex<std::collections::HashMap<u32, u64>> {
    WAYLAND_OWNERS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Display ids whose capture pipeline is actually running (StartCapture inserts,
/// StopCapture/DestroyOutput remove), so the Python-facing is_capturing() reports pipeline
/// liveness, not merely which ScreenCapture owns the backend.
static WAYLAND_ALIVE_DISPLAYS: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();

fn wayland_alive() -> &'static Mutex<std::collections::HashSet<u32>> {
    WAYLAND_ALIVE_DISPLAYS.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}
/// Per-display outcome of the most recent capture start, recorded by the calloop thread
/// so the Python-facing `capture_state` can report it. A start that came up clean removes
/// its entry; a hard failure (no output, ...) or a caveat that still capturing (host
/// connect failed -> local compositing, hardware encoder fell back to CPU) leaves the
/// reason here. Distinct from `wayland_alive`, which says only whether the pipeline runs.
static WAYLAND_CAPTURE_ERR: OnceLock<Mutex<std::collections::HashMap<u32, String>>> =
    OnceLock::new();

fn wayland_capture_err() -> &'static Mutex<std::collections::HashMap<u32, String>> {
    WAYLAND_CAPTURE_ERR.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Record (or clear, with `None`) the last-start outcome for `display_id`.
fn set_wayland_capture_err(display_id: u32, err: Option<String>) {
    let mut map = wayland_capture_err().lock().unwrap();
    match err {
        Some(e) => {
            map.insert(display_id, e);
        }
        None => {
            map.remove(&display_id);
        }
    }
}
/// Hands each `ScreenCapture` a unique, monotonic id — the token `WAYLAND_OWNERS` compares to
/// decide which instance is allowed to stop a display's capture. Starts at 1 so 0 can
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
    display_id: u32,
    bitrate_kbps: Option<i32>,
    vbv_multiplier: Option<f64>,
    fps: Option<f64>,
) {
    if let Some(slot) = WAYLAND_BACKEND.get()
        && let Some(be) = slot.lock().unwrap().as_ref() {
            let _ = be.bind(py).borrow().update_rate(bitrate_kbps, vbv_multiplier, fps, display_id);
        }
}

/// Forward live per-frame tunables to the shared Wayland backend (no-op if none is running).
fn wayland_update_tunables(py: Python<'_>, display_id: u32, t: LiveTunables) {
    if let Some(slot) = WAYLAND_BACKEND.get()
        && let Some(be) = slot.lock().unwrap().as_ref() {
            let _ = be.bind(py).borrow().update_tunables(display_id, t);
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
        if node.is_none()
            && let Some(request) = parse_auto_gpu(&auto_gpu) {
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
    /// This capture holds one reference on the shared X11 cursor monitor. Set in
    /// the same locked section as `backend = 1` and TAKEN in the same locked
    /// section a stop reads the backend, so acquire/release pair exactly per
    /// capture — inferring the reference from `backend` alone would let a stop
    /// that interleaves with a start release a reference not yet taken (leaking
    /// the monitor once the acquire lands). The GIL happens to serialize that
    /// window today; the pairing must not depend on it.
    cursor_ref: bool,
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
    /// The Wayland display id this instance's capture is bound to (backend == 2).
    wl_display: u32,
    /// The X11 capture thread's exit error (backend == 1): the thread records why it died
    /// mid-run here, so `capture_state` can report it instead of the error scrolling past in
    /// a single stderr line. `None` while healthy.
    err: Option<Arc<Mutex<Option<String>>>>,
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
        let (handle, deliver_handle, same_thread, backend, controls, wl_display, cursor_ref) = {
            let mut st = self.inner.lock().unwrap();
            if let Some(c) = &st.controls {
                c.stop.store(true, Ordering::Relaxed);
            }
            let cur = Some(thread::current().id());
            if st.encode_thread_id.is_none()
                && let Some(rx) = st.encode_tid_rx.as_ref()
                && let Ok(id) = rx.try_recv() {
                        st.encode_thread_id = Some(id);
                    }
            let same = st.cap_thread_id == cur
                || st.encode_thread_id == cur
                || st.deliver_thread_id == cur;
            let controls = st.controls.take();
            let handle = st.handle.take();
            let deliver_handle = st.deliver_handle.take();
            let backend = st.backend;
            let cursor_ref = std::mem::take(&mut st.cursor_ref);
            let wl_display = st.wl_display;
            st.backend = 0;
            st.cap_thread_id = None;
            st.encode_thread_id = None;
            st.encode_tid_rx = None;
            st.deliver_thread_id = None;
            st.wl_display = 0;
            st.err = None;
            (handle, deliver_handle, same, backend, controls, wl_display, cursor_ref)
        };
        if let Some(c) = &controls {
            live_x11().lock().unwrap().retain(|x| !Arc::ptr_eq(x, c));
        }
        if cursor_ref {
            crate::x11::cursor::release(py);
        }
        if backend == 2 {
            let did = wl_display;
            let owned = {
                let mut owners = wayland_owners().lock().unwrap();
                if owners.get(&did) == Some(&self.id) {
                    owners.remove(&did);
                    true
                } else {
                    false
                }
            };
            if owned
                && let Some(slot) = WAYLAND_BACKEND.get()
                && let Some(be) = slot.lock().unwrap().as_ref() {
                        let _ = be.bind(py).borrow().stop_capture(did);
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
                cursor_ref: false,
                controls: None,
                handle: None,
                cap_thread_id: None,
                encode_thread_id: None,
                encode_tid_rx: None,
                deliver_handle: None,
                deliver_thread_id: None,
                wl_display: 0,
                err: None,
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
        let display_id = read_display_id(settings);
        let live_wayland_restart = want_wayland(settings)
            && {
                let st = self.inner.lock().unwrap();
                st.backend == 2 && st.wl_display == display_id
            }
            && wayland_owners().lock().unwrap().get(&display_id) == Some(&self.id)
            && wayland_alive().lock().unwrap().contains(&display_id);
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
            wayland_owners().lock().unwrap().insert(display_id, self.id);
            {
                let mut st = self.inner.lock().unwrap();
                st.backend = 2;
                st.wl_display = display_id;
            }
            return Ok(());
        }

        // A fresh start proves the interpreter is alive: clear the teardown flag the atexit
        // sweep may have set (the Wayland start path already does the same), or the delivery
        // thread below would drop every frame.
        PY_SHUTDOWN.store(false, Ordering::Relaxed);
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
            if let Some(request) = parse_auto_gpu(&auto_gpu)
                && let Some(picked) = auto_select_render_node(request.as_deref())
                && let Some(idx) = picked
                        .strip_prefix("/dev/dri/renderD")
                        .and_then(|s| s.parse::<i32>().ok())
                    {
                        println!("[x11] AUTO_GPU enabled. Selected: {picked}");
                        rs.encode_node_index = idx - 128;
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
            // run_capture has joined its encode thread by now (NVENC/CUDA session dropped),
            // so the atexit sweep may finalize the interpreter without racing that drop.
            c3.finished.store(true, Ordering::Release);
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
        // Take the monitor reference BEFORE publishing the capture: a stop that
        // observes this capture must always find the reference it is to release.
        crate::x11::cursor::acquire(cursor_cap);
        let mut st = self.inner.lock().unwrap();
        st.backend = 1;
        st.cursor_ref = true;
        st.controls = Some(controls);
        st.handle = Some(handle);
        st.cap_thread_id = tid;
        st.encode_thread_id = etid;
        st.encode_tid_rx = late_etid_rx;
        st.deliver_handle = Some(deliver_handle);
        st.deliver_thread_id = Some(deliver_thread_id);
        st.err = Some(err_slot);
        drop(st);
        Ok(())
    }

    fn stop_capture(&self, py: Python<'_>) -> PyResult<()> {
        self.stop_internal(py)
    }

    fn request_idr_frame(&self, py: Python<'_>) -> PyResult<()> {
        let (backend, controls, did) = {
            let st = self.inner.lock().unwrap();
            (st.backend, st.controls.clone(), st.wl_display)
        };
        match backend {
            1 => {
                if let Some(c) = controls {
                    c.force_idr.store(true, Ordering::Relaxed);
                }
            }
            2 => {
                if let Some(slot) = WAYLAND_BACKEND.get()
                    && let Some(be) = slot.lock().unwrap().as_ref() {
                        let _ = be.bind(py).borrow().request_idr_frame(did);
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
        let (backend, controls, did) = {
            let st = self.inner.lock().unwrap();
            (st.backend, st.controls.clone(), st.wl_display)
        };
        match backend {
            1 => {
                if let Some(c) = &controls {
                    c.bitrate_kbps.store(kbps, Ordering::Relaxed);
                    c.rate_dirty.store(true, Ordering::Release);
                }
            }
            2 => wayland_update_rate(py, did, Some(kbps), None, None),
            _ => {}
        }
        Ok(())
    }

    fn update_framerate(&self, py: Python<'_>, fps: f64) -> PyResult<()> {
        let (backend, controls, did) = {
            let st = self.inner.lock().unwrap();
            (st.backend, st.controls.clone(), st.wl_display)
        };
        match backend {
            1 => {
                if let Some(c) = &controls {
                    c.fps_milli.store((fps.max(1.0) * 1000.0) as u64, Ordering::Relaxed);
                    c.rate_dirty.store(true, Ordering::Release);
                }
            }
            2 => wayland_update_rate(py, did, None, None, Some(fps)),
            _ => {}
        }
        Ok(())
    }

    /// Live CBR VBV change, as a multiple of one frame's bit budget (<= 0 = policy default).
    fn update_vbv_multiplier(&self, py: Python<'_>, multiplier: f64) -> PyResult<()> {
        let (backend, controls, did) = {
            let st = self.inner.lock().unwrap();
            (st.backend, st.controls.clone(), st.wl_display)
        };
        match backend {
            1 => {
                if let Some(c) = &controls {
                    c.vbv_mult_milli
                        .store((multiplier * 1000.0).round() as i32, Ordering::Relaxed);
                    c.rate_dirty.store(true, Ordering::Release);
                }
            }
            2 => wayland_update_rate(py, did, None, Some(multiplier), None),
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
        let (backend, controls, did) = {
            let st = self.inner.lock().unwrap();
            (st.backend, st.controls.clone(), st.wl_display)
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
            2 => wayland_update_tunables(py, did, t),
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
            2 => wayland_owners().lock().unwrap().get(&st.wl_display) == Some(&self.id)
                && wayland_alive().lock().unwrap().contains(&st.wl_display),
            _ => false,
        }
    }

    fn inject_key(&self, py: Python<'_>, scancode: u32, state: u32) -> PyResult<()> {
        wayland_backend_running(py).map_or(Ok(()), |be| be.bind(py).borrow().inject_key(scancode, state))
    }
    /// Inject an ordered run of `(keycode, state)` events in one message.
    fn inject_keys(&self, py: Python<'_>, events: Vec<(u32, u32)>) -> PyResult<()> {
        wayland_backend_running(py).map_or(Ok(()), |be| be.bind(py).borrow().inject_keys(events))
    }
    fn set_keymap_string(&self, py: Python<'_>, text: String) -> PyResult<()> {
        wayland_backend_running(py).map_or(Ok(()), |be| be.bind(py).borrow().set_keymap_string(text))
    }
    /// Bind explicit `(keycode, keysym)` pairs onto the current base keymap in one swap;
    /// false when no backend runs or no base keymap is installed. See the backend method
    /// for the split of responsibility.
    fn set_keymap_overlay(&self, py: Python<'_>, binds: Vec<(u32, u32)>) -> PyResult<()> {
        wayland_backend_running(py)
            .map_or(Ok(()), |be| be.bind(py).borrow().set_keymap_overlay(binds))
    }
    /// Compositor apps run under (a nested labwc/kwin session pixelflux captures),
    /// the target for Computer-Use text injection; selkies resolves it and hands it
    /// over. Empty clears it. Stored process-wide since the CU server is per-process.
    fn set_app_wayland_display(&self, display: String) {
        crate::computer_use::set_app_wayland_display(
            if display.is_empty() { None } else { Some(display) },
        );
    }
    /// Type `text` through `display`'s zwp_virtual_keyboard_manager_v1 as a one-shot
    /// client: selkies' text-injection path, targeting whichever compositor the apps
    /// live under (the nested session's, or pixelflux's own in a direct session).
    /// Blocking; releases the GIL for the duration. Raises
    /// [`VirtualKeyboardUnavailable`] when the compositor lacks the protocol.
    fn type_text_wayland(&self, py: Python<'_>, display: String, text: String) -> PyResult<()> {
        py.detach(move || {
            let path = crate::wayland::wlclient::socket_path(&display)
                .ok_or_else(|| "XDG_RUNTIME_DIR is unset".to_string())?;
            crate::wayland::vkclient::type_text_to(&path, &text)
        })
        .map_err(|e: String| {
            if e.contains("zwp_virtual_keyboard_manager_v1") {
                VirtualKeyboardUnavailable::new_err(e)
            } else {
                pyo3::exceptions::PyRuntimeError::new_err(e)
            }
        })
    }
    /// Tap `keysyms` in order through `display`'s virtual keyboard, verbatim: the
    /// caller owns which keysym spells which character (selkies' policy layer);
    /// this owns delivery. Same one-shot client and errors as `type_text_wayland`.
    fn type_keysyms_wayland(
        &self,
        py: Python<'_>,
        display: String,
        keysyms: Vec<u32>,
    ) -> PyResult<()> {
        py.detach(move || {
            let path = crate::wayland::wlclient::socket_path(&display)
                .ok_or_else(|| "XDG_RUNTIME_DIR is unset".to_string())?;
            crate::wayland::vkclient::type_keysyms_to(&path, &keysyms)
        })
        .map_err(|e: String| {
            if e.contains("zwp_virtual_keyboard_manager_v1") {
                VirtualKeyboardUnavailable::new_err(e)
            } else {
                pyo3::exceptions::PyRuntimeError::new_err(e)
            }
        })
    }
    /// Scale the app compositor's `index`-th screen, so its applications draw
    /// larger while the capture keeps its full resolution. False when that
    /// compositor manages no outputs for clients (KWin), whose scale follows
    /// the capture output's instead.
    fn set_app_output_scale(
        &self,
        py: Python<'_>,
        display: String,
        index: usize,
        scale: f64,
    ) -> PyResult<bool> {
        py.detach(move || {
            let path = crate::wayland::wlclient::socket_path(&display)
                .ok_or_else(|| "XDG_RUNTIME_DIR is unset".to_string())?;
            crate::wayland::outclient::set_output_scale(&path, index, scale)
        })
        .map(|outcome| matches!(outcome, crate::wayland::outclient::ScaleOutcome::Applied))
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    /// Give the app compositor's `index`-th screen this mode and scale in one
    /// configuration, so the session lays its desktop out once. Setting them
    /// separately exposes a geometry that never exists — the pre-connect mode at
    /// the new scale — and a client that does not lay out again keeps it.
    /// False = that compositor manages no outputs for clients.
    fn set_app_screen_geometry(
        &self,
        py: Python<'_>,
        display: String,
        index: usize,
        width: i32,
        height: i32,
        scale: f64,
    ) -> PyResult<bool> {
        py.detach(move || {
            let path = crate::wayland::wlclient::socket_path(&display)
                .ok_or_else(|| "XDG_RUNTIME_DIR is unset".to_string())?;
            crate::wayland::outclient::set_screen_geometry(&path, index, (width, height), scale)
        })
        .map(|outcome| matches!(outcome, crate::wayland::outclient::ScaleOutcome::Applied))
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    /// Lay the app compositor's screens out at `rects` — `(x, y, width, height)`
    /// per screen in screen order — so the session arranges its desktop the way
    /// the capture outputs were placed rather than by its own default rule.
    /// Returns how many were positioned; 0 = that compositor manages no outputs
    /// for clients (KWin, which offers `kde_output_management_v2` instead), and
    /// its arrangement is left as it chose.
    fn set_app_screen_layout(
        &self,
        py: Python<'_>,
        display: String,
        rects: Vec<(i32, i32, i32, i32)>,
    ) -> PyResult<usize> {
        py.detach(move || {
            let path = crate::wayland::wlclient::socket_path(&display)
                .ok_or_else(|| "XDG_RUNTIME_DIR is unset".to_string())?;
            crate::wayland::outclient::set_screen_layout(&path, rects)
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    /// The app compositor's enabled screens as `(name, x, y)` in screen order,
    /// which is what it did with a layout rather than what it was asked for.
    /// Empty = that compositor manages no outputs for clients.
    fn list_app_screens(&self, py: Python<'_>, display: String) -> PyResult<Vec<(String, i32, i32)>> {
        py.detach(move || {
            let path = crate::wayland::wlclient::socket_path(&display)
                .ok_or_else(|| "XDG_RUNTIME_DIR is unset".to_string())?;
            crate::wayland::outclient::list_screens(&path)
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    /// Hold the app compositor's screens past the first `keep` at a small size,
    /// so a session that opened more screens than there are capture outputs does
    /// not lay its desktop out across one nobody sees. Returns how many were
    /// resized (0 = that compositor manages no outputs for clients).
    fn hold_spare_app_screens(
        &self,
        py: Python<'_>,
        display: String,
        keep: usize,
        width: i32,
        height: i32,
    ) -> PyResult<usize> {
        py.detach(move || {
            let path = crate::wayland::wlclient::socket_path(&display)
                .ok_or_else(|| "XDG_RUNTIME_DIR is unset".to_string())?;
            crate::wayland::outclient::hold_spare_screens(&path, keep, (width, height))
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    /// Mimes the app compositor's current selection offers (empty = nothing copied).
    fn clipboard_types_app(&self, py: Python<'_>, display: String) -> PyResult<Vec<String>> {
        py.detach(move || {
            crate::wayland::dcclient::list_types(&app_socket_path(&display)?)
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    /// The app compositor selection's payload for `mime`, or None when nothing is
    /// copied or the selection does not offer that mime.
    fn clipboard_read_app(
        &self,
        py: Python<'_>,
        display: String,
        mime: String,
    ) -> PyResult<Option<Py<pyo3::types::PyBytes>>> {
        let data = py
            .detach(move || crate::wayland::dcclient::read(&app_socket_path(&display)?, &mime))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(data.map(|d| pyo3::types::PyBytes::new(py, &d).unbind()))
    }
    /// Take the app compositor's selection, serving `entries` (mime, bytes) to
    /// every paster from a background thread until another client copies.
    fn clipboard_write_app(
        &self,
        py: Python<'_>,
        display: String,
        entries: Vec<(String, Vec<u8>)>,
    ) -> PyResult<()> {
        py.detach(move || crate::wayland::dcclient::write(&app_socket_path(&display)?, entries))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    /// Drop the app compositor's selection.
    fn clipboard_clear_app(&self, py: Python<'_>, display: String) -> PyResult<()> {
        py.detach(move || crate::wayland::dcclient::clear(&app_socket_path(&display)?))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    /// Invoke `callback(mimes: list[str])` from a background thread on every
    /// selection change in the app compositor (including the one current at call
    /// time). A second watch for the same display replaces the first.
    fn clipboard_watch_app(
        &self,
        py: Python<'_>,
        display: String,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        py.detach(move || crate::wayland::dcclient::watch(&app_socket_path(&display)?, callback))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    /// Stop the selection watch for `display` (no-op without one).
    fn clipboard_unwatch_app(&self, py: Python<'_>, display: String) {
        let _ = py.detach(move || {
            crate::wayland::dcclient::unwatch(&app_socket_path(&display)?);
            Ok::<(), String>(())
        });
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
    /// Set the seat's BASE xkb layout from RMLVO names; false when no backend runs or the
    /// layout fails to compile.
    #[pyo3(signature = (layout, variant = String::new(), options = String::new(), model = String::new(), rules = String::new()))]
    fn set_xkb_layout(
        &self,
        py: Python<'_>,
        layout: String,
        variant: String,
        options: String,
        model: String,
        rules: String,
    ) -> PyResult<bool> {
        wayland_backend_running(py).map_or(Ok(false), |be| {
            be.bind(py).borrow().set_xkb_layout(py, layout, variant, options, model, rules)
        })
    }
    /// Seat keyboard readback: `(pressed_keycodes, modifier_mask)`; empty when no backend runs.
    fn get_keyboard_state(&self, py: Python<'_>) -> PyResult<(Vec<u32>, u32)> {
        wayland_backend_running(py)
            .map_or(Ok((Vec::new(), 0)), |be| be.bind(py).borrow().get_keyboard_state(py))
    }
    /// The capture geometry actually live on the given display `(width, height, scale)`;
    /// `None` when no Wayland backend runs or the compositor did not answer in time (a
    /// timeout is distinct from a real answer so the caller treats it as unknown rather than
    /// "nothing to reconcile"). X11 has no compositor to read back and returns `None`.
    #[pyo3(signature = (display_id = 0))]
    fn get_realized_geometry(
        &self,
        py: Python<'_>,
        display_id: u32,
    ) -> PyResult<Option<(i32, i32, f64)>> {
        wayland_backend_running(py)
            .map_or(Ok(None), |be| be.bind(py).borrow().get_realized_geometry(py, display_id))
    }
    /// Lifecycle of this capture as `(state, last_error)`: `state` is `"running"`, `"failed"`
    /// or `"idle"`; `last_error` gives the reason a start failed, or a caveat a live capture
    /// came up with (encoder fell back to CPU, host connect refused, a refused resize). The
    /// Wayland outcome is recorded by the compositor thread; the X11 outcome is the capture
    /// thread's own exit error. `display_id` selects a Wayland output; X11 ignores it.
    #[pyo3(signature = (display_id = 0))]
    fn capture_state(&self, display_id: u32) -> (String, Option<String>) {
        let (backend, running, err) = {
            let st = self.inner.lock().unwrap();
            let running = st
                .controls
                .as_ref()
                .map(|c| !c.stop.load(Ordering::Relaxed))
                .unwrap_or(false);
            let err = st.err.as_ref().and_then(|e| e.lock().ok().and_then(|g| g.clone()));
            (st.backend, running, err)
        };
        match backend {
            1 => {
                let state = if running {
                    "running"
                } else if err.is_some() {
                    "failed"
                } else {
                    "idle"
                };
                (state.to_string(), err)
            }
            2 => wayland_capture_state(display_id),
            _ => ("idle".to_string(), None),
        }
    }
    /// Create an additional Wayland output (see `WaylandBackend.create_output`); false when
    /// no backend runs.
    // The parameter list is the Python signature; grouping it would change the ABI.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (id, width, height, x = 0, y = 0, scale = 1.0))]
    fn create_output(
        &self,
        py: Python<'_>,
        id: u32,
        width: i32,
        height: i32,
        x: i32,
        y: i32,
        scale: f64,
    ) -> PyResult<bool> {
        wayland_backend_running(py).map_or(Ok(false), |be| {
            be.bind(py).borrow().create_output(py, id, width, height, x, y, scale)
        })
    }
    /// Destroy a secondary Wayland output; false when no backend runs.
    fn destroy_output(&self, py: Python<'_>, id: u32) -> PyResult<bool> {
        wayland_backend_running(py)
            .map_or(Ok(false), |be| be.bind(py).borrow().destroy_output(py, id))
    }
    /// Move a Wayland output (the primary included) to layout offset `(x, y)`; false when
    /// no backend runs or the id is unknown.
    fn reposition_output(&self, py: Python<'_>, id: u32, x: i32, y: i32) -> PyResult<bool> {
        wayland_backend_running(py)
            .map_or(Ok(false), |be| be.bind(py).borrow().reposition_output(py, id, x, y))
    }
    /// Recreate the Wayland cursor theme at `size` pixels (named-cursor callbacks and the
    /// burned-in overlay); false when no backend runs or the size is non-positive.
    fn set_cursor_size(&self, py: Python<'_>, size: i32) -> PyResult<bool> {
        wayland_backend_running(py)
            .map_or(Ok(false), |be| be.bind(py).borrow().set_cursor_size(py, size))
    }
    /// Every live Wayland output as `(id, x, y, width, height, scale, capturing)`; empty
    /// when no backend runs.
    fn list_outputs(&self, py: Python<'_>) -> PyResult<Vec<OutputDesc>> {
        wayland_backend_running(py)
            .map_or(Ok(Vec::new()), |be| be.bind(py).borrow().list_outputs(py))
    }
    /// Display capacity (see `WaylandBackend.output_capacity`); -1 when no backend runs.
    fn output_capacity(&self, py: Python<'_>) -> PyResult<i64> {
        wayland_backend_running(py)
            .map_or(Ok(-1), |be| be.bind(py).borrow().output_capacity(py))
    }
    /// Move a window onto an output (fullscreened there); false when no backend runs.
    fn move_window_to_output(&self, py: Python<'_>, window_id: u32, output_id: u32) -> PyResult<bool> {
        wayland_backend_running(py).map_or(Ok(false), |be| {
            be.bind(py).borrow().move_window_to_output(py, window_id, output_id)
        })
    }
    /// Every mapped window as `(window_id, title, app_id, output_id, waiting)`; empty when
    /// no backend runs.
    fn list_windows(&self, py: Python<'_>) -> PyResult<Vec<WindowDesc>> {
        wayland_backend_running(py)
            .map_or(Ok(Vec::new()), |be| be.bind(py).borrow().list_windows(py))
    }
}

/// Best-effort teardown: flag the capture thread to exit without joining.
///
/// Joining would need the GIL (the thread calls back into Python), which `Drop` cannot safely take,
/// so this only sets the stop flag; the actual join is left to an explicit `stop_capture` or the
/// atexit sweep.
impl Drop for ScreenCapture {
    fn drop(&mut self) {
        if let Ok(mut st) = self.inner.lock() {
            if let Some(c) = &st.controls {
                c.stop.store(true, Ordering::Relaxed);
            }
            // A Wayland capture is owned by this instance: release it, or a GC'd handle
            // would leave the compositor encoding and delivering forever.
            if st.backend == 2 && !crate::PY_SHUTDOWN.load(Ordering::Relaxed) {
                let did = st.wl_display;
                let owned = {
                    let mut owners = wayland_owners().lock().unwrap();
                    if owners.get(&did) == Some(&self.id) {
                        owners.remove(&did);
                        true
                    } else {
                        false
                    }
                };
                if owned
                    && let Some(slot) = WAYLAND_BACKEND.get()
                    && let Some(be) = slot.lock().unwrap().as_ref() {
                            Python::attach(|py| {
                                let _ = be.bind(py).borrow().stop_capture(did);
                            });
                        }
            }
            // Pair the cursor-monitor acquire from start_capture: a dropped
            // handle that never got stop_capture would pin the refcount and its
            // monitor thread forever.
            if std::mem::take(&mut st.cursor_ref) && !crate::PY_SHUTDOWN.load(Ordering::Relaxed) {
                Python::attach(crate::x11::cursor::release);
            }
        }
    }
}

/// Build a Python dict from a recorder status snapshot (one shape for status and stop).
fn recording_status_dict(py: Python<'_>, s: &crate::recorder::RecordingStatus) -> PyResult<Py<PyAny>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("active", s.active)?;
    d.set_item("path", &s.path)?;
    d.set_item("backend", s.backend)?;
    d.set_item("mode", s.mode)?;
    d.set_item("frames", s.frames)?;
    d.set_item("sync_frames", s.sync_frames)?;
    d.set_item("dropped", s.dropped)?;
    d.set_item("skipped_non_h264", s.skipped_non_h264)?;
    d.set_item("bytes", s.bytes)?;
    d.set_item("duration_s", s.duration_s)?;
    d.set_item("width", s.width)?;
    d.set_item("height", s.height)?;
    d.set_item("error", s.error.as_deref())?;
    Ok(d.into_any().unbind())
}

/// Start the built-in MP4 recorder. Works with no capture and no client running: the
/// recorder owns an independent capture (X11 root, or a Wayland output of the in-process
/// compositor) and taps a live streaming session instead of restarting it. `settings` is an
/// optional `CaptureSettings` for a recorder-owned capture (H.264 only; `display_id`
/// selects the Wayland output); when omitted, `PIXELFLUX_RECORD_*` environment variables
/// and full-screen defaults apply.
#[pyfunction]
#[pyo3(signature = (path, settings = None))]
fn start_recording(
    py: Python<'_>,
    path: String,
    settings: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let mut opts = crate::recorder::RecordOptions::from_env(path);
    if let Some(s) = settings {
        let rs = extract_settings(s)?;
        if rs.output_mode != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "recording requires H.264 capture settings (output_mode=1); JPEG cannot be recorded",
            ));
        }
        opts.display_id = read_display_id(s);
        if let Some(explicit) = s
            .getattr("use_wayland")
            .ok()
            .and_then(|v| v.extract::<bool>().ok())
        {
            opts.backend = Some(if explicit {
                crate::recorder::PreferredBackend::Wayland
            } else {
                crate::recorder::PreferredBackend::X11
            });
        }
        // Explicit settings are authoritative over the env knobs they subsume.
        opts.fps = 0.0;
        opts.bitrate_kbps = 0;
        opts.capture = Some(rs);
    }
    let status = py
        .detach(|| crate::recorder::start(opts))
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    recording_status_dict(py, &status)
}

/// Stop the active recording, finalize the MP4, and return the final status dict. Raises
/// when no recording is active or nothing recordable was captured.
#[pyfunction]
fn stop_recording(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let status = py
        .detach(crate::recorder::stop)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
    recording_status_dict(py, &status)
}

/// Status of the live recording (or the last finished one); `None` if this process has
/// never recorded.
#[pyfunction]
fn recording_status(py: Python<'_>) -> PyResult<Py<PyAny>> {
    match crate::recorder::status() {
        Some(s) => recording_status_dict(py, &s),
        None => Ok(py.None()),
    }
}

/// Bring the Wayland compositor socket up before any capture so apps launched early can
/// connect (sets WAYLAND_DISPLAY for children of this process). Idempotent; the running
/// backend keeps its node/dimensions on later calls. `render_node` is an explicit
/// /dev/dri/renderD* path; `auto_gpu` is a truthy string or vendor/driver token; empty
/// values mean software rendering. Returns the compositor's actual socket name (the
/// auto-picked `wayland-N`, which need not match any configured index); empty string only
/// if the socket did not come up in time.
#[pyfunction]
#[pyo3(signature = (width = 0, height = 0, render_node = String::new(), auto_gpu = String::new(), cursor_size = -1))]
fn ensure_wayland_display(
    py: Python<'_>,
    width: i32,
    height: i32,
    render_node: String,
    auto_gpu: String,
    cursor_size: i32,
) -> PyResult<String> {
    ensure_wayland_backend(py, width, height, render_node, auto_gpu, String::new(), cursor_size)?;
    Ok(py
        .detach(|| wait_socket_name(Duration::from_secs(5)))
        .unwrap_or_default())
}

/// Whether a Wayland session here would be hardware accelerated, and whether there is a GPU
/// to accelerate it.
///
/// Device paths do not answer this: a render node can exist with no working allocator or EGL
/// stack behind it, and a GPU can be present with no node at all (an NVIDIA container without
/// the graphics driver capability). So the compositor's own bring-up is run against the node
/// it would resolve, through allocating a render target and exporting it as a dmabuf. Nothing
/// is left running; a compositor started afterwards is unaffected.
///
/// A node that only reaches a software rasterizer is reported unaccelerated: Mesa answers
/// with llvmpipe rather than failing, so the bring-up succeeding is not on its own proof of
/// a GPU behind it.
///
/// `render_node` and `auto_gpu` take the same values as the capture settings of those names.
/// Returns `node` (the resolved render node, empty when none), `accelerated`, `gpu` (a GPU is
/// exposed here at all), `renderer` (the GL renderer reached, empty when none was), and
/// `error` (the step that failed; empty when accelerated).
#[pyfunction]
#[pyo3(signature = (render_node = String::new(), auto_gpu = String::new()))]
fn probe_wayland_gpu(
    py: Python<'_>,
    render_node: String,
    auto_gpu: String,
) -> PyResult<Py<PyAny>> {
    let (node, name, error) = py.detach(|| {
        let node = if render_node.is_empty() {
            parse_auto_gpu(&auto_gpu).and_then(|token| auto_select_render_node(token.as_deref()))
        } else {
            Some(render_node)
        };
        let Some(node) = node else {
            return (String::new(), String::new(), "No render node".to_string());
        };
        let mut name = String::new();
        let result = gpu_render_init(std::path::Path::new(&node)).and_then(|(gbm, mut renderer)| {
            let bo = gbm
                .create_buffer_object::<()>(64, 64, GbmFormat::Argb8888, BufferObjectFlags::RENDERING)
                .map_err(|_| "Failed to allocate GBM buffer")?;
            bo.fd().map_err(|e| format!("Failed to export dmabuf: {e:?}"))?;
            name = gl_renderer_name(&mut renderer);
            let lowered = name.to_lowercase();
            if ["llvmpipe", "softpipe", "swrast", "software rasterizer"]
                .iter()
                .any(|sw| lowered.contains(sw))
            {
                return Err(format!("Software rasterizer only ({name})"));
            }
            Ok(())
        });
        (node, name, result.err().unwrap_or_default())
    });
    let d = pyo3::types::PyDict::new(py);
    d.set_item("node", &node)?;
    d.set_item("accelerated", error.is_empty())?;
    d.set_item("gpu", gpu_exposed())?;
    d.set_item("renderer", &name)?;
    d.set_item("error", &error)?;
    Ok(d.into_any().unbind())
}

/// The running compositor's Wayland socket name (e.g. "wayland-1"), or None when no
/// compositor thread has been started (this never creates one).
#[pyfunction]
fn get_wayland_display_name(py: Python<'_>) -> Option<String> {
    wayland_backend_running(py)?;
    py.detach(|| wait_socket_name(Duration::from_secs(2)))
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
    // Finalize any active recording first so its last buffered MP4 sample is flushed and
    // its own capture (if any) is stopped through the normal path.
    py.detach(crate::recorder::finalize_on_exit);
    *PENDING_CURSOR_CALLBACK.lock().unwrap() = None;
    crate::x11::cursor::shutdown();
    crate::wayland::dcclient::unwatch_all();
    // Flag every live X11 capture to stop, then wait (bounded) for each thread to actually
    // return: it joins its own encode thread on the way out, dropping the NVENC/CUDA
    // session, and letting the interpreter finalize while that drop is in flight segfaults —
    // the same hazard the Wayland Barrier below fences. A snapshot is taken so the registry
    // mutex is not held across the wait.
    let x11: Vec<Arc<crate::x11::Controls>> = live_x11().lock().unwrap().iter().cloned().collect();
    for c in &x11 {
        c.stop.store(true, Ordering::Relaxed);
    }
    if !x11.is_empty() {
        let deadline = Instant::now() + Duration::from_secs(2);
        py.detach(|| {
            for c in &x11 {
                while !c.finished.load(Ordering::Acquire) && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        });
    }
    // Clone the backend handle out of the slot so the WAYLAND_BACKEND mutex is released
    // before the Barrier wait below: a GC-triggered ScreenCapture::Drop under the GIL takes
    // that same mutex, and holding it across the detached wait would block that Drop until
    // the Barrier times out into the unsafe exit.
    let be = WAYLAND_BACKEND
        .get()
        .and_then(|slot| slot.lock().unwrap().as_ref().map(|b| b.clone_ref(py)));
    if let Some(be) = be {
        let be = be.bind(py).borrow();
        let mut displays: Vec<u32> = wayland_alive().lock().unwrap().iter().copied().collect();
        if !displays.contains(&0) {
            displays.push(0);
        }
        for did in displays {
            let _ = be.stop_capture(did);
        }
        // Wait (bounded) until the calloop finished processing the stops: dropping a
        // hardware encoder session (NVENC/CUDA) mid-process-exit segfaults, so the
        // interpreter must not finalize while that teardown is still running.
        let (ack_tx, ack_rx) = std::sync::mpsc::channel::<()>();
        if be.send(ThreadCommand::Barrier { reply: ack_tx }).is_ok() {
            let _ = py.detach(move || ack_rx.recv_timeout(Duration::from_secs(2)));
        }
    }
    wayland_owners().lock().unwrap().clear();
    wayland_alive().lock().unwrap().clear();
    py.detach(|| std::thread::sleep(Duration::from_millis(50)));
}

/// The `pixelflux` Python module: registers the exported classes and functions, and hooks
/// `_stop_all_captures` into `atexit` so every live capture is stopped before interpreter shutdown.
/// Socket path for a Wayland display name, with the ABI methods' error shape.
fn app_socket_path(display: &str) -> Result<String, String> {
    crate::wayland::wlclient::socket_path(display)
        .ok_or_else(|| "XDG_RUNTIME_DIR is unset".to_string())
}

pyo3::create_exception!(
    pixelflux,
    VirtualKeyboardUnavailable,
    pyo3::exceptions::PyRuntimeError,
    "The target compositor does not advertise zwp_virtual_keyboard_manager_v1."
);

/// Start the Computer-Use HTTP server: a bare port listens on all interfaces,
/// `host:port` scopes it. Idempotent; the PIXELFLUX_CU env var remains the
/// standalone fallback.
#[pyfunction]
fn start_computer_use(bind: String) {
    crate::computer_use::start_cu_server(&bind);
}

/// `gil_used = true`: the module has not been audited for free-threaded Python. The
/// detached compositor, capture, encode and delivery threads attach to the interpreter
/// and several native encoder sessions (NVENC/CUDA, VA-API) assume the GIL serializes
/// their Python-facing access; until that is proven safe the interpreter re-enables the
/// GIL for this module on a free-threaded build rather than silently defaulting to the
/// thread-safe claim pyo3 0.28+ makes.
#[pymodule(gil_used = true)]
fn pixelflux(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WaylandBackend>()?;
    m.add_class::<StripeFrame>()?;
    m.add_class::<CaptureSettings>()?;
    m.add_class::<ScreenCapture>()?;
    m.add_class::<webcam::VirtualCamera>()?;
    m.add_class::<webcam::VirtualCameraSettings>()?;
    // The software H.264 encoder this build resolved to ("x264" | "openh264"): what a CPU
    // H.264 session encodes with, so a consumer can pick rate-control defaults and name it.
    m.add("SOFTWARE_H264_ENCODER", encoders::SOFTWARE_H264_ENCODER)?;
    m.add_function(wrap_pyfunction!(stripe_frame_from_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(ensure_wayland_display, m)?)?;
    m.add_function(wrap_pyfunction!(get_wayland_display_name, m)?)?;
    m.add_function(wrap_pyfunction!(probe_wayland_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(start_recording, m)?)?;
    m.add_function(wrap_pyfunction!(stop_recording, m)?)?;
    m.add_function(wrap_pyfunction!(recording_status, m)?)?;
    m.add_function(wrap_pyfunction!(start_computer_use, m)?)?;
    m.add(
        "VirtualKeyboardUnavailable",
        m.py().get_type::<VirtualKeyboardUnavailable>(),
    )?;
    m.add_function(wrap_pyfunction!(_stop_all_captures, m)?)?;
    if let Ok(atexit) = m.py().import("atexit") {
        let _ = atexit.call_method1("register", (m.getattr("_stop_all_captures")?,));
    }
    // Standalone CU entry point: with PIXELFLUX_CU set the server binds at import, serving
    // X11 (via DISPLAY) until a Wayland compositor registers itself as the backend.
    crate::computer_use::spawn_cu_from_env();
    // PIXELFLUX_RECORD=<path>: start recording from process start (X11 immediately, or as
    // soon as the in-process Wayland compositor comes up).
    crate::recorder::autostart_from_env();


    Ok(())
}

#[cfg(test)]
mod shm_usage_tests {
    //! What the session reports as its /dev/shm usage: allocated blocks, so a sparse
    //! file is not counted as memory anyone holds.
    use super::shm_usage_in;

    #[test]
    fn only_allocated_blocks_count() {
        let dir = std::env::temp_dir().join(format!("pixelflux-shm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.to_str().unwrap().to_string();

        std::fs::File::create(dir.join("sparse"))
            .unwrap()
            .set_len(8 * 1024 * 1024 * 1024)
            .unwrap();
        let sparse = shm_usage_in(&path);

        std::fs::write(dir.join("dense"), vec![0u8; 4 * 1024 * 1024]).unwrap();
        let dense = shm_usage_in(&path);

        std::fs::remove_dir_all(&dir).ok();
        assert!(sparse < 1024 * 1024, "an 8 GiB sparse file counted {sparse} bytes");
        assert!(
            dense - sparse >= 4 * 1024 * 1024,
            "a 4 MiB written file added only {} bytes",
            dense - sparse
        );
    }
}

#[cfg(test)]
mod capture_state_tests {
    //! The `(state, last_error)` a Wayland display's capture reports: a live pipeline is
    //! "running" (with a caveat surfaced alongside), a recorded failure with no live
    //! pipeline is "failed", and a clean stop is "idle".
    use super::{set_wayland_capture_err, wayland_alive, wayland_capture_state};

    #[test]
    fn state_tracks_liveness_and_recorded_error() {
        // A display id unlikely to collide with any concurrent test's use of the globals.
        let did = 987_654;
        set_wayland_capture_err(did, None);
        wayland_alive().lock().unwrap().remove(&did);
        assert_eq!(wayland_capture_state(did), ("idle".to_string(), None));

        // Recorded failure, no live pipeline -> failed with the reason.
        set_wayland_capture_err(did, Some("no output".to_string()));
        assert_eq!(
            wayland_capture_state(did),
            ("failed".to_string(), Some("no output".to_string()))
        );

        // A live pipeline reads running even with a caveat still recorded.
        wayland_alive().lock().unwrap().insert(did);
        set_wayland_capture_err(did, Some("using CPU encode".to_string()));
        assert_eq!(
            wayland_capture_state(did),
            ("running".to_string(), Some("using CPU encode".to_string()))
        );

        // Clean stop clears both.
        wayland_alive().lock().unwrap().remove(&did);
        set_wayland_capture_err(did, None);
        assert_eq!(wayland_capture_state(did), ("idle".to_string(), None));
    }
}

#[cfg(test)]
mod host_layout_tests {
    //! How a host's verdict on a layout request resolves for the capture configured at the
    //! requested size: applied or already current -> nothing to do; kept a different mode
    //! -> follow it; mode unknown -> keep gating on the request.
    use super::host_layout_resolution;

    #[test]
    fn verdict_resolves_to_a_follow_size_only_when_the_host_kept_another_mode() {
        let want = (1920, 1080);
        // Applied: nothing changes, whatever the (possibly not yet announced) mode reads.
        assert_eq!(host_layout_resolution(true, want, Some((1280, 720))), None);
        assert_eq!(host_layout_resolution(true, want, None), None);
        // Declined but already at the requested size (a re-assertion on a host without
        // layout management): converged.
        assert_eq!(host_layout_resolution(false, want, Some(want)), None);
        // Declined and running something else: the capture follows the host.
        assert_eq!(host_layout_resolution(false, want, Some((2560, 1440))), Some((2560, 1440)));
        // Declined with no mode known: nothing to follow, the request stands.
        assert_eq!(host_layout_resolution(false, want, None), None);
    }
}

#[cfg(test)]
mod output_overlap_tests {
    //! Invariants of the output-placement intersection predicate: strict interior
    //! intersection (touching edges never overlap), containment and identity overlap,
    //! empty/negative rectangles never overlap, and extreme coordinates do not wrap.
    use super::rects_overlap;

    #[test]
    fn disjoint_rects_do_not_overlap() {
        assert!(!rects_overlap((0, 0, 100, 100), (200, 0, 100, 100)));
        assert!(!rects_overlap((0, 0, 100, 100), (0, 200, 100, 100)));
    }

    #[test]
    fn touching_edges_do_not_overlap() {
        // Right edge of a meets left edge of b, and bottom meets top.
        assert!(!rects_overlap((0, 0, 100, 100), (100, 0, 100, 100)));
        assert!(!rects_overlap((0, 0, 100, 100), (0, 100, 100, 100)));
        // Corner touch only.
        assert!(!rects_overlap((0, 0, 100, 100), (100, 100, 50, 50)));
    }

    #[test]
    fn one_pixel_intrusion_overlaps() {
        assert!(rects_overlap((0, 0, 100, 100), (99, 0, 100, 100)));
        assert!(rects_overlap((0, 0, 100, 100), (0, 99, 100, 100)));
    }

    #[test]
    fn containment_and_identity_overlap() {
        assert!(rects_overlap((0, 0, 100, 100), (25, 25, 10, 10)));
        assert!(rects_overlap((25, 25, 10, 10), (0, 0, 100, 100)));
        assert!(rects_overlap((5, 5, 50, 50), (5, 5, 50, 50)));
    }

    #[test]
    fn empty_or_negative_rects_never_overlap() {
        assert!(!rects_overlap((10, 10, 0, 50), (0, 0, 100, 100)));
        assert!(!rects_overlap((10, 10, 50, 0), (0, 0, 100, 100)));
        assert!(!rects_overlap((10, 10, -5, 5), (0, 0, 100, 100)));
        assert!(!rects_overlap((0, 0, 100, 100), (10, 10, 0, 0)));
    }

    #[test]
    fn negative_origins_overlap_correctly() {
        assert!(rects_overlap((-50, -50, 100, 100), (0, 0, 100, 100)));
        assert!(!rects_overlap((-100, -100, 100, 100), (0, 0, 100, 100)));
    }

    #[test]
    fn extreme_coordinates_do_not_wrap() {
        assert!(!rects_overlap((i32::MAX - 10, 0, 10, 10), (i32::MIN, 0, 10, 10)));
        assert!(rects_overlap((i32::MAX - 10, 0, 10, 10), (i32::MAX - 5, 0, 10, 10)));
    }
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
