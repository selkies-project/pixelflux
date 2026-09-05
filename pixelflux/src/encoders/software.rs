/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! CPU-based striped encoder: H.264 through the build's software encoder — libx264 with the
//! `gpl` feature, Cisco OpenH264 without it (`SOFTWARE_H264_ENCODER`) — and turbojpeg for JPEG.
//!
//! Frames are split into horizontal stripes processed in parallel via rayon. Each stripe is
//! independently hashed against the previous frame for change detection, and only dirty stripes
//! are encoded. The H.264 path maintains per-stripe encoder state across frames for
//! inter-prediction, and both libraries emit the same per-stripe wire framing; the JPEG path is
//! stateless.

use super::codec::{push_jpeg_header, push_video_header, Codec, FRAME_DELTA, FRAME_INTRA, FRAME_KEY};
use crate::RustCaptureSettings;
use rayon::prelude::*;
use smithay::utils::{Physical, Rectangle};
#[cfg(feature = "gpl")]
use std::ffi::CString;
#[cfg(feature = "gpl")]
use std::ptr;
use std::sync::Arc;
use yuv::{BufferStoreMut, YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix};

/// Upper bound on the horizontal stripes the CPU encoder splits a frame into, so the
/// persistent per-stripe state vector can be reserved to a fixed capacity once, up front.
///
/// With the vector reserved to this size at startup, the per-frame resize to the actual stripe count
/// stays a cheap in-place adjustment that preserves each stripe's reused encoder and scratch buffers,
/// rather than a reallocation that would churn them whenever the count changes.
pub const MAX_STRIPE_CAPACITY: usize = 64;

/// Convert a packed BGRA/RGBA buffer to planar YUV (4:2:0 or 4:4:4) for the software H.264
/// encoders, spreading the conversion across up to `bands` threads so it never bottlenecks a frame.
///
/// **Why the band split exists.** Colour conversion is a non-trivial slice of per-frame CPU. The
/// striped path already parallelizes it for free — each stripe converts on its own rayon worker —
/// but a single full-frame consumer (the whole-frame x264 stripe, or a full-frame OpenH264
/// instance, which passes `bands = 4`) would otherwise convert its entire image on one thread and
/// stall the frame there. Splitting into horizontal bands hands that lone conversion the same
/// multi-threading the striped path enjoys. The cut is horizontal because YUV planes are
/// row-major, so a horizontal boundary yields contiguous, non-overlapping plane sub-slices with no
/// per-row seam bookkeeping.
///
/// 1. **Plane strides**: `strides` gives the Y and chroma row pitches of the output planes
///    (tightly packed for the stripe buffers, padded for an AVFrame); the chroma planes are
///    `width` wide for 4:4:4 (`i444 == true`) or `width / 2` for 4:2:0. `rgba_input` selects
///    the source byte order and `i444` the subsampling, together choosing one of four `yuv`
///    crate routines — 4:4:4 uses **Full** range, 4:2:0 uses **Limited** range, and both use
///    the **BT.709** matrix and the **Fast** conversion mode.
/// 2. **Band split**: `band_h` is `height / bands` floored to an even number and at least 2 rows
///    (a band under 2 rows is not worth a thread). Keeping band boundaries even ensures a 4:2:0
///    chroma pair never straddles a seam. When `bands <= 1` or the whole image fits one band, the
///    conversion runs single-threaded in place.
/// 3. **Parallel bands**: otherwise a `std::thread::scope` carves `src` and the three output planes
///    into contiguous per-band sub-slices (chroma rows scaled by `uv_rows` — full height for 4:4:4,
///    half for 4:2:0) and spawns one thread per band. The final band absorbs any leftover rows,
///    taking all remaining rows whenever fewer than `band_h + 2` are left. Each thread's result is
///    joined and collected; a panicked join degrades to a `PointerOverflow` error, and the first
///    error wins.
#[allow(clippy::too_many_arguments)]
pub(crate) fn convert_to_yuv_mt(
    src: &[u8],
    src_stride: u32,
    width: usize,
    height: usize,
    rgba_input: bool,
    i444: bool,
    y_buf: &mut [u8],
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    strides: (usize, usize),
    bands: usize,
) -> Result<(), yuv::YuvError> {
    let (y_stride, uv_stride) = strides;

    let convert_band = |src_band: &[u8], y: &mut [u8], u: &mut [u8], v: &mut [u8], h: usize| {
        let mut img = YuvPlanarImageMut {
            y_plane: BufferStoreMut::Borrowed(y),
            y_stride: y_stride as u32,
            u_plane: BufferStoreMut::Borrowed(u),
            u_stride: uv_stride as u32,
            v_plane: BufferStoreMut::Borrowed(v),
            v_stride: uv_stride as u32,
            width: width as u32,
            height: h as u32,
        };
        match (i444, rgba_input) {
            (true, true) => yuv::rgba_to_yuv444(
                &mut img, src_band, src_stride, YuvRange::Full,
                YuvStandardMatrix::Bt709, YuvConversionMode::Fast,
            ),
            (true, false) => yuv::bgra_to_yuv444(
                &mut img, src_band, src_stride, YuvRange::Full,
                YuvStandardMatrix::Bt709, YuvConversionMode::Fast,
            ),
            (false, true) => yuv::rgba_to_yuv420(
                &mut img, src_band, src_stride, YuvRange::Limited,
                YuvStandardMatrix::Bt709, YuvConversionMode::Fast,
            ),
            (false, false) => yuv::bgra_to_yuv420(
                &mut img, src_band, src_stride, YuvRange::Limited,
                YuvStandardMatrix::Bt709, YuvConversionMode::Fast,
            ),
        }
    };

    let band_h = ((height / bands.max(1)) & !1).max(2);
    if bands <= 1 || height <= band_h {
        return convert_band(src, y_buf, u_buf, v_buf, height);
    }

    let uv_rows = |rows: usize| if i444 { rows } else { rows / 2 };
    let mut results: Vec<Result<(), yuv::YuvError>> = Vec::new();
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        let (mut src_rest, mut y_rest, mut u_rest, mut v_rest) = (src, y_buf, u_buf, v_buf);
        let mut row = 0;
        while row < height {
            let h = if height - row < band_h + 2 { height - row } else { band_h };
            let (src_band, s_next) = src_rest.split_at(h * src_stride as usize);
            let (y_band, y_next) = y_rest.split_at_mut(h * y_stride);
            let (u_band, u_next) = u_rest.split_at_mut(uv_rows(h) * uv_stride);
            let (v_band, v_next) = v_rest.split_at_mut(uv_rows(h) * uv_stride);
            src_rest = s_next;
            y_rest = y_next;
            u_rest = u_next;
            v_rest = v_next;
            row += h;
            handles.push(s.spawn(move || convert_band(src_band, y_band, u_band, v_band, h)));
        }
        for hnd in handles {
            results.push(hnd.join().unwrap_or(Err(yuv::YuvError::PointerOverflow)));
        }
    });
    results.into_iter().collect()
}

thread_local! {
    /// Reused libjpeg-turbo compressor kept per worker thread to avoid paying a
    /// `tjInitCompress`/`tjDestroy` round trip for every stripe of every frame.
    ///
    /// The striped JPEG path compresses one stripe per rayon worker, so the compressor is
    /// thread-local rather than shared: each worker creates its own lazily on first use and then
    /// holds it for the process lifetime. Making it thread-local also sidesteps the locking a shared
    /// compressor would otherwise need across the parallel stripe encoders.
    static JPEG_COMPRESSOR: std::cell::RefCell<Option<turbojpeg::Compressor>> =
        const { std::cell::RefCell::new(None) };
}

/// Process-global lock that serializes libx264 encoder open/close, because those calls are
/// not thread-safe yet the striped path opens encoders concurrently from many stripe workers.
///
/// libx264 mutates process-global state inside `x264_encoder_open`/`x264_encoder_close`, so two
/// stripe encoders opening at once — or two capture instances sharing one process — can race that
/// state and corrupt the heap. The lock is deliberately held only around open and close, never
/// around `x264_encoder_encode`, so serializing setup costs nothing in the hot per-stripe encode
/// path where the real parallelism lives.
#[cfg(feature = "gpl")]
static X264_OPEN_CLOSE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// One long-lived libx264 session for a stripe, holding the raw `x264_t` handle alongside a
/// mirror of its live parameters so the encoder can be retuned per frame instead of rebuilt.
///
/// Rebuilding an x264 encoder is expensive and forces a fresh IDR, so a stripe keeps its instance
/// across frames and only nudges CRF, bitrate, VBV, and frame rate live; the tracked `current_*`
/// fields are that mirror, letting a reconfigure skip the FFI call whenever nothing actually changed.
/// `is_i444` (4:4:4 vs 4:2:0) is baked into the encoder's colour space at open, so a change to it is
/// one of the few things that forces a full rebuild; `is_cbr` records which rate-control mode was
/// chosen at open and gates which of the live reconfigures apply. The manual `Send` impl exists only
/// because a raw pointer is not `Send` by default and the handle must move onto the rayon stripe
/// workers; `Drop` closes it under the global open/close lock for the same reason that lock exists.
#[cfg(feature = "gpl")]
pub struct H264EncoderWrapper {
    encoder: *mut x264_sys::x264_t,
    pub width: i32,
    pub height: i32,
    current_crf: i32,
    pub is_i444: bool,
    is_cbr: bool,
    current_bitrate: i32,
    current_vbv: i32,
    current_fps: u32,
    #[allow(dead_code)]
    full_range: bool,
    /// Open-time parameters retained so a frame-rate change can reopen the session: x264's live
    /// reconfigure cannot alter the frame rate, and CBR/VBV budgets are derived from it.
    threads: i32,
    min_qp: i32,
    max_qp: i32,
}

#[cfg(feature = "gpl")]
unsafe impl Send for H264EncoderWrapper {}

#[cfg(feature = "gpl")]
impl Drop for H264EncoderWrapper {
    fn drop(&mut self) {
        if !self.encoder.is_null() {
            let _guard = X264_OPEN_CLOSE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            unsafe { x264_sys::x264_encoder_close(self.encoder) };
            self.encoder = ptr::null_mut();
        }
    }
}

#[cfg(feature = "gpl")]
impl H264EncoderWrapper {
    /// Open an x264 encoder tuned for real-time screen streaming, or `None` on failure.
    ///
    /// **Why this configuration.** These frames are captured live and must ship immediately, so the
    /// encoder is optimized for latency over compression ratio: the `ultrafast` preset keeps encode
    /// time under the frame budget, and `zerolatency` bars the frame reordering and lookahead
    /// buffering that would otherwise add pipeline delay. Everything below then bends x264 toward the
    /// pipeline's own keyframe and colour model instead of its broadcast-oriented defaults.
    ///
    /// 1. **Preset/tune**: starts from the `ultrafast` preset with the `zerolatency` tune, then
    ///    overrides resolution, frame rate (floored to 30 fps when under 1), and thread count.
    /// 2. **Infinite GOP**: `i_keyint_max` is set to x264's infinite sentinel and adaptive scene-cut
    ///    is disabled (`i_scenecut_threshold = 0`), so the encoder never injects an unrequested IDR
    ///    on a scene change — keyframes are purely on-demand via the forced-IDR path, matching the
    ///    strict infinite-GOP model.
    /// 3. **Rate control**:
    ///    - **CBR** (`cbr_mode`): ABR targeting `bitrate_kbps` with a VBV cap pinned to the same
    ///      value (buffer `vbv_kbit`, precomputed by the caller from the frame-time multiplier
    ///      policy) and filler disabled. Optional QP clamps apply only when non-zero — `max_qp` is
    ///      the legibility floor (caps how ugly a rate-starved frame gets) and `min_qp` the waste
    ///      ceiling (stops over-spending on easy content); both are clamped to 51.
    ///    - **CRF** (default): constant-quality with `f_rf_constant = crf`.
    /// 4. **Colour**: I444 (full range) or I420 (limited range) CSP, BT.709 VUI primaries/transfer/
    ///    matrix, and the matching `high444` / `baseline` profile.
    /// 5. **Coding tools**: CABAC and the 8x8 transform are disabled, matching the low-latency
    ///    baseline profile — CAVLC entropy coding with no 8x8 DCT — for minimal encode cost.
    /// 6. **Output**: repeated headers (SPS/PPS before each keyframe) and Annex-B framing, with
    ///    x264's own logging silenced.
    ///
    /// The `x264_encoder_open` call is serialized under `X264_OPEN_CLOSE_LOCK` because it mutates
    /// libx264 global state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(width: i32, height: i32, crf: i32, is_i444: bool, fps: f64, threads: i32,
               cbr_mode: bool, bitrate_kbps: i32, vbv_kbit: i32,
               min_qp: i32, max_qp: i32) -> Option<Self> {
        unsafe {
            let mut param: x264_sys::x264_param_t = std::mem::zeroed();
            let preset = CString::new("ultrafast").unwrap();
            let tune = CString::new("zerolatency").unwrap();

            if x264_sys::x264_param_default_preset(&mut param, preset.as_ptr(), tune.as_ptr()) < 0 {
                return None;
            }

            param.i_width = width;
            param.i_height = height;
            param.i_fps_num = if fps < 1.0 { 30 } else { fps as u32 };
            param.i_fps_den = 1;
            param.i_keyint_max = x264_sys::X264_KEYINT_MAX_INFINITE as i32;
            param.i_scenecut_threshold = 0;
            if cbr_mode {
                let bk = bitrate_kbps.saturating_abs();
                param.rc.i_rc_method = x264_sys::X264_RC_ABR as i32;
                param.rc.i_bitrate = bk;
                param.rc.i_vbv_max_bitrate = bk;
                param.rc.i_vbv_buffer_size = vbv_kbit.max(1);
                param.rc.b_filler = 0;
                if min_qp > 0 {
                    param.rc.i_qp_min = min_qp.min(51);
                }
                if max_qp > 0 {
                    param.rc.i_qp_max = max_qp.min(51);
                }
            } else {
                param.rc.i_rc_method = x264_sys::X264_RC_CRF as i32;
                param.rc.f_rf_constant = crf as f32;
            }
            param.i_csp = if is_i444 {
                x264_sys::X264_CSP_I444
            } else {
                x264_sys::X264_CSP_I420
            } as i32;
            param.vui.b_fullrange = if is_i444 { 1 } else { 0 };
            param.vui.i_colorprim = 1;
            param.vui.i_transfer = 1;
            param.vui.i_colmatrix = 1;

            let profile = CString::new(if is_i444 { "high444" } else { "baseline" }).unwrap();
            x264_sys::x264_param_apply_profile(&mut param, profile.as_ptr());

            param.i_threads = threads;
            param.b_repeat_headers = 1;
            param.b_annexb = 1;
            param.i_log_level = x264_sys::X264_LOG_NONE;

            let encoder = {
                let _guard = X264_OPEN_CLOSE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                x264_sys::x264_encoder_open(&mut param)
            };
            if encoder.is_null() {
                None
            } else {
                Some(Self {
                    encoder,
                    width,
                    height,
                    current_crf: crf,
                    is_i444,
                    is_cbr: cbr_mode,
                    current_bitrate: bitrate_kbps.saturating_abs(),
                    current_vbv: vbv_kbit,
                    current_fps: if fps < 1.0 { 30 } else { fps as u32 },
                    full_range: param.vui.b_fullrange == 1,
                    threads,
                    min_qp,
                    max_qp,
                })
            }
        }
    }

    /// Retune the constant-quality CRF on the running encoder, so a quality change costs a
    /// parameter push rather than tearing down and rebuilding the session (a rebuild would force an
    /// IDR and drop encoder state).
    ///
    /// It is a no-op in CBR mode, where rate is bitrate-controlled and CRF simply does not apply, and
    /// a no-op when the value is unchanged — the tracked `current_crf` is what makes that cheap
    /// early-out possible. Otherwise it reads the encoder's live parameters, overwrites
    /// `f_rf_constant`, and pushes the change via `x264_encoder_reconfig`, advancing the tracked CRF
    /// only once the reconfig has actually succeeded so the mirror never drifts from the encoder.
    pub fn reconfigure_crf(&mut self, new_crf: i32) {
        if self.is_cbr || self.current_crf == new_crf {
            return;
        }
        unsafe {
            let mut param: x264_sys::x264_param_t = std::mem::zeroed();
            x264_sys::x264_encoder_parameters(self.encoder, &mut param);
            param.rc.f_rf_constant = new_crf as f32;
            if x264_sys::x264_encoder_reconfig(self.encoder, &mut param) == 0 {
                self.current_crf = new_crf;
            }
        }
    }

    /// Retune bitrate/VBV (CBR only) and/or frame rate to match the live settings, structured to
    /// be called unconditionally every frame so the caller need not track what changed itself.
    ///
    /// Because `encode_cpu` fires it on every frame, it first computes the would-be values and bails
    /// before touching the encoder when neither the CBR bitrate/VBV nor the frame rate differs from
    /// what is live — that self-gating keeps a per-frame call nearly free.
    ///
    /// A frame-rate change reopens the encoder rather than reconfiguring it: `x264_encoder_reconfig`
    /// does not apply `i_fps_*`, and the CBR/VBV per-frame budget is `bitrate / fps`, so a session
    /// left at its old rate ships roughly half the configured bitrate once fps halves. The reopen
    /// carries the new bitrate/VBV too, and a fresh session emits an IDR on its first frame; a failed
    /// reopen keeps the working session instead of nulling the handle. A bitrate/VBV-only change
    /// (CBR) stays a live `x264_encoder_reconfig`, and the tracked mirror advances only on success so
    /// it cannot drift from the encoder's real state.
    pub fn reconfigure_rate(&mut self, bitrate_kbps: i32, vbv_kbit: i32, fps: f64) {
        let bk = bitrate_kbps.saturating_abs();
        let new_fps = if fps < 1.0 { 30 } else { fps as u32 };
        let rate_changed =
            self.is_cbr && (self.current_bitrate != bk || self.current_vbv != vbv_kbit);
        let fps_changed = self.current_fps != new_fps;
        if !rate_changed && !fps_changed {
            return;
        }
        if fps_changed {
            if let Some(fresh) = H264EncoderWrapper::new(
                self.width,
                self.height,
                self.current_crf,
                self.is_i444,
                new_fps as f64,
                self.threads,
                self.is_cbr,
                bk,
                vbv_kbit,
                self.min_qp,
                self.max_qp,
            ) {
                *self = fresh;
            }
            return;
        }
        unsafe {
            let mut param: x264_sys::x264_param_t = std::mem::zeroed();
            x264_sys::x264_encoder_parameters(self.encoder, &mut param);
            param.rc.i_bitrate = bk;
            param.rc.i_vbv_max_bitrate = bk;
            param.rc.i_vbv_buffer_size = vbv_kbit.max(1);
            if x264_sys::x264_encoder_reconfig(self.encoder, &mut param) == 0 {
                self.current_bitrate = bk;
                self.current_vbv = vbv_kbit;
            }
        }
    }

    /// Encode one YUV frame into H.264 and frame it for the wire, reporting whether the
    /// encoder actually emitted a bitstream this call.
    ///
    /// The boolean return is load-bearing: `x264_encoder_encode` can legitimately produce nothing on
    /// a given call, and the caller must forward a stripe only when real bytes exist — never an empty
    /// or header-only packet. Framing is conditional because the transport needs the pipeline's small
    /// wire header to route the stripe, while `omit_headers` consumers take the bare Annex-B
    /// elementary stream.
    ///
    /// 1. **Picture setup**: wraps the borrowed Y/U/V planes and their strides in an
    ///    `x264_picture_t` with the encoder's CSP, stamps the presentation timestamp with `frame_id`,
    ///    and requests an IDR when `force_idr` is set (otherwise `X264_TYPE_AUTO`).
    /// 2. **Encode**: calls `x264_encoder_encode`; a non-positive returned size means no frame was
    ///    emitted this call, so the function returns `false` without writing output.
    /// 3. **Framing**: `output_buf` is cleared and refilled. Unless `omit_headers` is set, the wire
    ///    header is prepended with a frame kind read from the *actual* output picture type rather
    ///    than from `force_idr`, because the encoder may not honor a keyframe request and the
    ///    client keys its decode-recovery on the kind it truly received. With `omit_headers` the
    ///    output is bare Annex-B.
    /// 4. **Payload**: every NAL payload is appended to `output_buf` after the optional header,
    ///    so the bytes past the wire header are always a contiguous Annex-B access unit.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_with_headers(
        &mut self,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        y_stride: i32,
        u_stride: i32,
        v_stride: i32,
        frame_id: u16,
        y_start: u16,
        force_idr: bool,
        omit_headers: bool,
        output_buf: &mut Vec<u8>,
    ) -> bool {
        unsafe {
            let mut pic_in: x264_sys::x264_picture_t = std::mem::zeroed();
            x264_sys::x264_picture_init(&mut pic_in);

            pic_in.img.i_csp = if self.is_i444 {
                x264_sys::X264_CSP_I444
            } else {
                x264_sys::X264_CSP_I420
            } as i32;
            pic_in.img.i_plane = 3;
            pic_in.img.plane[0] = y.as_ptr() as *mut u8;
            pic_in.img.plane[1] = u.as_ptr() as *mut u8;
            pic_in.img.plane[2] = v.as_ptr() as *mut u8;
            pic_in.img.i_stride[0] = y_stride;
            pic_in.img.i_stride[1] = u_stride;
            pic_in.img.i_stride[2] = v_stride;
            pic_in.i_pts = frame_id as i64;
            pic_in.i_type = if force_idr {
                x264_sys::X264_TYPE_IDR
            } else {
                x264_sys::X264_TYPE_AUTO
            } as i32;

            let mut pic_out: x264_sys::x264_picture_t = std::mem::zeroed();
            let mut nals: *mut x264_sys::x264_nal_t = ptr::null_mut();
            let mut i_nals: i32 = 0;

            let frame_size = x264_sys::x264_encoder_encode(
                self.encoder,
                &mut nals,
                &mut i_nals,
                &mut pic_in,
                &mut pic_out,
            );

            if frame_size > 0 {
                output_buf.clear();
                output_buf.reserve(super::codec::VIDEO_HEADER_LEN + frame_size as usize);
                if !omit_headers {
                    let frame_type = if pic_out.i_type == x264_sys::X264_TYPE_IDR as i32 {
                        FRAME_KEY
                    } else if pic_out.i_type == x264_sys::X264_TYPE_I as i32 {
                        FRAME_INTRA
                    } else {
                        FRAME_DELTA
                    };
                    push_video_header(
                        output_buf,
                        Codec::H264,
                        frame_type,
                        frame_id,
                        y_start,
                        self.width as u16,
                        self.height as u16,
                    );
                }

                let nal_slice = std::slice::from_raw_parts(nals, i_nals as usize);
                for nal in nal_slice {
                    let payload = std::slice::from_raw_parts(nal.p_payload, nal.i_payload as usize);
                    output_buf.extend_from_slice(payload);
                }
                return true;
            }
        }
        false
    }
}

/// Everything one horizontal stripe must remember between frames: its reused buffers, its own
/// live encoder, and the motion / paint-over / damage bookkeeping that drives its send decision.
///
/// The frame is striped so independent screen regions can encode in parallel and an unchanged region
/// can be skipped on its own, and that only works if each stripe carries its *own* cross-frame
/// history. So one instance lives per stripe for the whole session and nothing per-stripe is rebuilt
/// or recomputed from scratch each frame:
/// - **Reused buffers**: `y_buf` / `u_buf` / `v_buf` hold the stripe's YUV planes and `packet_buf`
///   the encoded output, grown in place rather than reallocated per frame.
/// - **Encoder**: `h264_encoder` is the stripe's software H.264 instance — libx264 in a `gpl`
///   build, OpenH264 otherwise — reused until its geometry (or, for x264, chroma format) changes.
/// - **Paint-over / recovery**: `no_motion_frame_count` counts consecutive static frames,
///   `paint_over_sent` guards against re-sending a high-quality repaint of a still region, and
///   `h264_burst_frames_remaining` tracks a post-repaint or recovery streaming burst.
/// - **Content-hash damage** (only for sources without external damage, i.e. X11): `last_hash` is
///   the previous frame's content hash, `consecutive_changes` counts changed frames toward the
///   damage-block threshold, and `in_damage_block` / `damage_block_frames_remaining` /
///   `hash_at_block_start` drive the sustained-motion damage block managed by `content_dirty`.
#[derive(Default)]
pub struct StripeState {
    pub no_motion_frame_count: u32,
    pub paint_over_sent: bool,
    #[cfg(feature = "gpl")]
    pub h264_encoder: Option<H264EncoderWrapper>,
    #[cfg(not(feature = "gpl"))]
    pub h264_encoder: Option<crate::encoders::oh264::Openh264Encoder>,
    pub h264_burst_frames_remaining: i32,
    #[cfg(feature = "gpl")]
    pub y_buf: Vec<u8>,
    #[cfg(feature = "gpl")]
    pub u_buf: Vec<u8>,
    #[cfg(feature = "gpl")]
    pub v_buf: Vec<u8>,
    pub packet_buf: Vec<u8>,
    pub last_hash: u64,
    pub consecutive_changes: u32,
    pub in_damage_block: bool,
    pub damage_block_frames_remaining: i32,
    pub hash_at_block_start: u64,
}

/// Fast, non-cryptographic 64-bit content hash used only for in-memory change detection.
///
/// Uses xxh3: a SIMD-friendly hash that processes 64-byte blocks with parallel lanes,
/// delivering near memory-bandwidth throughput. The value is never persisted or sent on the
/// wire, so only the property that identical bytes hash identically matters. A collision
/// between two distinct stripes is ~2^-64, and the next real content change or a requested
/// keyframe repaints any missed update anyway.
fn fast_hash(bytes: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64_with_seed(bytes, 0)
}

impl StripeState {
    /// Stand in for the compositor damage that X11 capture does not provide: hash this stripe
    /// to decide whether it changed since last frame, and once it is clearly in motion, stop
    /// re-hashing it every frame by committing to a sustained-motion "damage block".
    ///
    /// The hash is not free, and a region that changes every frame would otherwise be re-hashed
    /// forever while always reporting dirty anyway. So after `threshold` consecutive changes the
    /// stripe enters a damage block that just reports dirty for `duration` frames and re-hashes only
    /// once, at the end, to decide whether to extend the block or let it lapse — trading a little
    /// extra sending for far fewer hashes on exactly the regions that need them least:
    ///
    /// 1. **Inside a damage block**: the stripe is treated as dirty without re-hashing, and the
    ///    block's remaining-frame counter is decremented. Only when the counter reaches zero is the
    ///    stripe re-hashed — if it differs from the hash captured at block start the block is renewed
    ///    for another `duration` frames, otherwise the block exits and the change counter resets.
    ///    This keeps a continuously-moving region streaming for `duration` frames per re-check rather
    ///    than hashing every frame.
    /// 2. **Outside a block**: the stripe is hashed and compared to the previous frame. A change
    ///    increments `consecutive_changes`, and reaching `threshold` consecutive changes opens a new
    ///    damage block; an unchanged frame resets the counter to zero.
    ///
    /// Returns `true` whenever the stripe is considered dirty (always true while inside a block).
    pub fn content_dirty(&mut self, bytes: &[u8], threshold: u32, duration: i32) -> bool {
        if self.in_damage_block {
            self.damage_block_frames_remaining -= 1;
            if self.damage_block_frames_remaining <= 0 {
                let h = fast_hash(bytes);
                if h != self.hash_at_block_start {
                    self.damage_block_frames_remaining = duration;
                    self.hash_at_block_start = h;
                } else {
                    self.in_damage_block = false;
                    self.consecutive_changes = 0;
                }
                self.last_hash = h;
            }
            return true;
        }
        let h = fast_hash(bytes);
        let changed = h != self.last_hash;
        self.last_hash = h;
        if changed {
            self.consecutive_changes += 1;
            if self.consecutive_changes >= threshold {
                self.in_damage_block = true;
                self.damage_block_frames_remaining = duration;
                self.hash_at_block_start = h;
            }
        } else {
            self.consecutive_changes = 0;
        }
        changed
    }
}

/// One encoded stripe: the compressed bytes plus geometry and identity metadata.
///
/// The consumer can place and attribute the stripe even when the payload has no header. In
/// `omit_headers` mode the per-stripe wire header is stripped from the bytes; the struct fields
/// carry that information out-of-band.
///
/// # Fields
///
/// * `data` - Compressed payload (JPEG, or the video codec's bitstream). `Arc`-shared so every
///   delivery-layer consumer can retain the frame without copying the bytes.
/// * `codec` - The codec of the payload.
/// * `stripe_y_start` - Y pixel coordinate of the stripe's top edge within the frame.
/// * `stripe_height` - Height of the stripe in pixels.
/// * `frame_id` - Frame sequence number this stripe belongs to.
pub struct EncodedStripe {
    pub data: Arc<Vec<u8>>,
    pub codec: Codec,
    pub stripe_y_start: i32,
    pub stripe_height: i32,
    pub frame_id: i32,
}

/// The software encoder's per-frame entry point: split the frame into horizontal stripes,
/// decide per stripe whether it needs sending, and encode only those as JPEG or H.264 (libx264
/// or OpenH264, by build) across the rayon pool.
///
/// Two pressures drive the design: CPU H.264/JPEG is expensive, so the frame is cut into
/// parallel stripes; bandwidth is precious, so unchanged stripes are skipped. Each stripe is
/// independently hashed against the previous frame for change detection, and only dirty stripes
/// are encoded. The H.264 path maintains per-stripe encoder state across frames for
/// inter-prediction; the JPEG path is stateless.
///
/// # Arguments
///
/// * `stripes` - Persistent per-stripe state vector (resized as needed, encoder state preserved).
/// * `raw_pixels` - Packed BGRA/RGBA pixel buffer (`width * height * 4` bytes).
/// * `width` - Frame width in pixels.
/// * `height` - Frame height in pixels.
/// * `damage_rects` - Wayland damage rectangles (empty for X11 hash-based detection).
/// * `settings` - Capture settings (quality, mode, rate control, etc.).
/// * `frame_counter` - Current frame number (wrapping `u16`).
/// * `use_gpu` - `true` when the source is RGBA (GLES readback); `false` for BGRA (X11 host).
/// * `hash_damage` - `true` for X11 stripe-hash change detection; `false` when damage rects
///   are provided.
/// * `force_idr_all` - Force a keyframe on every stripe (client join / reset / periodic IDR).
///
/// # Returns
///
/// Vec of [`EncodedStripe`] — empty when nothing changed.
/// repainting a stalled region at full quality, and letting a freshly-joined or reset client recover
/// a clean picture. Persistent `StripeState` is what makes both affordable: encoders and buffers
/// survive across frames instead of being rebuilt, and the motion/paint-over history the decision
/// needs lives right beside them. The per-stripe decision mirrors `decide_hw_fullframe`'s policy for
/// the hardware full-frame encoders; it is kept as separate code here because the striped path also
/// chooses JPEG-vs-H.264 and derives its own damage.
///
/// 1. **Stripe count**: defaults to the core count so the fan-out matches the hardware, but
///    collapses to a single full-frame stripe when H.264 full-frame is requested or the frame is
///    shorter than the 64-row minimum, and is otherwise capped so no stripe is thinner than 64 rows —
///    below that the per-stripe encoder and thread overhead outweighs the parallelism and the tiny
///    H.264 slices compress poorly. The persistent `stripes` vector is resized to match, preserving
///    per-stripe state across frames.
/// 2. **Idle fast path**: a frame on which no stripe can emit anything (no damage / clean
///    hashes, no paint-over due, no burst, no recovery IDR, not streaming) only advances the
///    per-stripe no-motion bookkeeping inline and returns without dispatching the stripe
///    fan-out, so a static capture never wakes the rayon pool.
/// 3. **Dirty map**: with external compositor damage (`hash_damage == false`) each `damage_rects`
///    rectangle marks every stripe whose row range it overlaps. With `hash_damage == true` (X11,
///    which has no compositor damage) per-stripe content hashing drives dirtiness instead — except
///    in streaming H.264, where every stripe is sent unconditionally so the hash is skipped.
/// 4. **Per-stripe decision** (in `stripe_body`): a stripe is sent when it is dirty, when a
///    paint-over / recovery burst is in flight, when streaming mode is on, or when `force_idr_all`
///    is set. Quality is chosen per case — base JPEG quality / base CRF for live content, the
///    paint-over quality/CRF after `paint_over_trigger_frames` static frames (once per still region,
///    guarded by `paint_over_sent`), and `burst_crf` during a burst (the paint-over CRF when it is
///    enabled and actually lower, else the base CRF, since a recovery burst still needs to stream so
///    CBR can refine it). A newly dirty frame cancels any pending burst or paint-over and reverts to
///    base quality.
/// 5. **Recovery IDR** (`force_idr_all`): forces a send on every stripe even when static so a
///    reconnecting client can resume. For H.264 it forces an IDR and arms a short streaming burst
///    (unless one is already pending, so it cannot preempt an in-flight burst) because the keyframe
///    is base-quality — worsened further by CBR — and a damage-gated static stream would otherwise
///    never refine it; for JPEG, where every stripe is already intra, it resends a
///    previously-painted-over stripe at the paint-over quality already on screen so a joining viewer
///    does not see a downgrade.
/// 6. **Encoding**:
///    - **JPEG**: source byte order is RGBA on the GPU readback path and BGRA on
///      X11; each worker thread reuses its thread-local TurboJPEG compressor. Header-less output
///      hands the compressed buffer straight through; otherwise a 6-byte stripe header (`0x03` tag,
///      a reserved byte, frame number, y-start) is prepended to match the H.264 path's native
///      framing so the transport can forward the buffer without re-framing.
///    - **H.264**: the stripe's encoder is reused unless the width, height, or
///      (x264) chroma format changed, in which case it is rebuilt and an IDR forced; otherwise CRF
///      and rate are reconfigured live. With libx264, ARGB is converted to YUV here (a conversion
///      failure skips the stripe rather than encoding garbage) and an 8-byte fixed header (frame
///      number, y-start, width, height) is emitted; OpenH264 converts and frames inside
///      `encode_stripe_argb` with the same header layout, and encodes a 4:4:4 request 4:2:0 (said
///      once per process). The live CBR budget is recomputed here from the bitrate/fps so it
///      rescales with live changes.
/// 7. **Dispatch**: a single full-frame stripe runs inline (sequential — empirically faster than a
///    one-element rayon job) with one fewer encode thread than the available cores, clamped to
///    `[1, 4]` (x264 with a single-band colour conversion; OpenH264 adds four slices and a four-band
///    conversion of its own). The slice threads keep the in-frame encode latency inside the frame
///    budget at high resolutions; the cap is four because `zerolatency` makes x264 slice-threaded
///    and more than four slices trips decode glitches in some Chromium builds, and the minus-one
///    leaves headroom for the capture thread. Multiple stripes instead run across the rayon pool
///    with a single encode thread and one conversion band each, since the parallelism there
///    already comes from encoding the stripes concurrently.
#[allow(clippy::too_many_arguments)]
/// No stripe is shorter than a macroblock row.
const MIN_STRIPE_HEIGHT: i32 = 64;
/// How fast the smoothed count of budget-carrying stripes follows the frame's.
const CARRY_RISE: f32 = 0.3;
const CARRY_FALL: f32 = 0.05;

pub fn encode_cpu(
    stripes: &mut Vec<StripeState>,
    carrying: &mut f32,
    raw_pixels: &[u8],
    width: i32,
    height: i32,
    damage_rects: &[Rectangle<i32, Physical>],
    settings: &RustCaptureSettings,
    frame_counter: u16,
    use_gpu: bool,
    hash_damage: bool,
    force_idr_all: bool,
) -> Vec<EncodedStripe> {
    let codec = settings.codec;
    let n_processing_stripes = stripe_count(height, codec, settings.video_fullframe);

    if stripes.len() != n_processing_stripes {
        stripes.resize_with(n_processing_stripes, StripeState::default);
    }

    let stripe_geometries = compute_stripe_geometries(height as usize, n_processing_stripes, codec);

    // Idle fast path: a static frame must still advance every stripe's paint-over countdown,
    // but nothing else — so when no stripe can emit anything this frame, do that bookkeeping
    // inline and return before the rayon fan-out. Waking the whole worker pool 60x/s for
    // no-op stripes is the dominant idle cost (tens of percent of a core), dwarfing the real
    // per-frame work. "Static" is known up front for damage-authoritative sources (Wayland:
    // empty damage list); hash-damage sources (X11) instead take a sequential early-exit
    // hash scan, probing the most-recently-dirty stripe first so live content bails out
    // after a single stripe hash. A clean scan performs exactly the state transitions
    // `content_dirty` would (hash unchanged, change streak reset), so the damage-block
    // machinery observes no difference.
    let idle_candidate = damage_rects.is_empty()
        && !force_idr_all
        && !(codec.is_video() && settings.video_streaming_mode);
    if idle_candidate {
        let paint_over_armed = settings.use_paint_over_quality
            && if !codec.is_video() {
                settings.paint_over_jpeg_quality > settings.jpeg_quality
            } else {
                settings.video_paintover_crf < settings.video_crf
            };
        let no_pending_send = |st: &StripeState| {
            (!codec.is_video() || st.h264_burst_frames_remaining <= 0)
                && (!paint_over_armed
                    || st.paint_over_sent
                    || st.no_motion_frame_count.saturating_add(1)
                        < settings.paint_over_trigger_frames)
        };
        let quiescent = if !hash_damage {
            stripes.iter().all(no_pending_send)
        } else {
            let width_bytes = width as usize * 4;
            let hint = stripes
                .iter()
                .enumerate()
                .min_by_key(|(_, st)| st.no_motion_frame_count)
                .map(|(i, _)| i)
                .unwrap_or(0);
            let clean = |i: usize| {
                let st = &stripes[i];
                if !no_pending_send(st) || st.in_damage_block {
                    return false;
                }
                let (y, h) = stripe_geometries[i];
                let bytes = &raw_pixels[y * width_bytes..(y + h) * width_bytes];
                fast_hash(bytes) == st.last_hash
            };
            clean(hint) && (0..stripes.len()).filter(|&i| i != hint).all(clean)
        };
        if quiescent {
            for st in stripes.iter_mut() {
                st.no_motion_frame_count = st.no_motion_frame_count.saturating_add(1);
                st.consecutive_changes = 0;
            }
            return Vec::new();
        }
    }
    let mut stripe_is_dirty = vec![false; n_processing_stripes];
    if !damage_rects.is_empty() {
        for rect in damage_rects {
            let r_y_start = rect.loc.y.max(0) as usize;
            let r_y_end = (rect.loc.y + rect.size.h).min(height) as usize;
            if r_y_start < r_y_end {
                for (i, &(s_y, s_h)) in stripe_geometries.iter().enumerate() {
                    let s_end = s_y + s_h;
                    if r_y_start < s_end && r_y_end > s_y {
                        stripe_is_dirty[i] = true;
                    }
                }
            }
        }
    }

    let width_usize = width as usize;
    let video = codec.is_video();
    let video_crf = settings.video_crf;
    let video_po_crf = settings.video_paintover_crf;
    let video_burst = settings.video_paintover_burst_frames;
    let video_fullcolor = settings.video_fullcolor;
    let video_streaming = settings.video_streaming_mode;
    let jpeg_q = settings.jpeg_quality;
    let paint_q = settings.paint_over_jpeg_quality;
    let trigger_frames = settings.paint_over_trigger_frames;
    let use_paint_over = settings.use_paint_over_quality;
    let burst_crf = if use_paint_over && video_po_crf < video_crf { video_po_crf } else { video_crf };
    let target_fps = settings.target_fps;
    let omit_headers = settings.omit_stripe_headers;
    let damage_block_threshold = settings.damage_block_threshold;
    let damage_block_duration = settings.damage_block_duration as i32;
    #[cfg(feature = "gpl")]
    let video_cbr = settings.video_cbr_mode;
    // The requested rate is a whole-screen budget, and CRF needs no division
    // at all (a per-quality target). OpenH264 sizes its own buffer, so only
    // x264 reads the VBV share.
    #[cfg_attr(not(feature = "gpl"), allow(unused_variables))]
    let (video_bitrate, video_vbv) =
        stripe_rate_control(settings, *carrying, n_processing_stripes);
    // Full-frame x264 threads: one fewer than the cores (headroom for the
    // capture thread), clamped to [1, 4] to match the four-slice ceiling below.
    // A full-frame OpenH264 instance applies the same policy internally.
    #[cfg(feature = "gpl")]
    let h264_threads = if n_processing_stripes == 1 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .saturating_sub(1)
            .clamp(1, 4) as i32
    } else {
        1
    };
    #[cfg(feature = "gpl")]
    let csc_bands = 1;
    if video && video_fullcolor && !crate::encoders::SOFTWARE_H264_FULLCOLOR {
        static FULLCOLOR_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !FULLCOLOR_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("[software] 4:4:4 full-color requested; OpenH264 is 4:2:0-only, encoding 4:2:0.");
        }
    }

    let stripe_body = |(i, stripe_state): (usize, &mut StripeState)| -> Option<EncodedStripe> {
            if i >= stripe_geometries.len() {
                return None;
            }
            let (y_start, actual_height) = stripe_geometries[i];
            let start_idx = y_start * width_usize * 4;
            let end_idx = start_idx + (actual_height * width_usize * 4);
            let stripe_bytes = &raw_pixels[start_idx..end_idx];

            let mut send_this_stripe = false;
            let mut quality_or_crf = if !video { jpeg_q } else { video_crf };
            let mut force_idr = false;
            let is_dirty = if !hash_damage {
                stripe_is_dirty[i]
            } else if video && video_streaming {
                false
            } else {
                stripe_state.content_dirty(stripe_bytes, damage_block_threshold, damage_block_duration)
            };

            if video && stripe_state.h264_burst_frames_remaining > 0 {
                send_this_stripe = true;
                quality_or_crf = burst_crf;
                stripe_state.h264_burst_frames_remaining -= 1;

                if is_dirty {
                    stripe_state.h264_burst_frames_remaining = 0;
                    stripe_state.paint_over_sent = false;
                    quality_or_crf = video_crf;
                }
            }

            if !send_this_stripe && video && video_streaming {
                send_this_stripe = true;
            }

            if is_dirty {
                send_this_stripe = true;
                stripe_state.no_motion_frame_count = 0;
                stripe_state.paint_over_sent = false;
                stripe_state.h264_burst_frames_remaining = 0;
                quality_or_crf = if !video { jpeg_q } else { video_crf };
            } else if !send_this_stripe {
                stripe_state.no_motion_frame_count += 1;

                if use_paint_over
                    && stripe_state.no_motion_frame_count >= trigger_frames
                    && !stripe_state.paint_over_sent
                {
                    if !video && paint_q > jpeg_q {
                        send_this_stripe = true;
                        quality_or_crf = paint_q;
                        stripe_state.paint_over_sent = true;
                    } else if video && video_po_crf < video_crf {
                        send_this_stripe = true;
                        stripe_state.paint_over_sent = true;
                        quality_or_crf = video_po_crf;
                        force_idr = true;
                        stripe_state.h264_burst_frames_remaining = video_burst - 1;
                    }
                }
            }

            if force_idr_all {
                send_this_stripe = true;
                if video {
                    force_idr = true;
                    if stripe_state.h264_burst_frames_remaining <= 0 && video_burst > 0 {
                        stripe_state.paint_over_sent = true;
                        stripe_state.h264_burst_frames_remaining = video_burst;
                    }
                } else if stripe_state.paint_over_sent && use_paint_over && paint_q > jpeg_q {
                    quality_or_crf = paint_q;
                }
            }

            if send_this_stripe {
                if !video {
                    let pixel_format = if use_gpu {
                        turbojpeg::PixelFormat::RGBA
                    } else {
                        turbojpeg::PixelFormat::BGRA
                    };
                    let img = turbojpeg::Image {
                        pixels: stripe_bytes,
                        width: width_usize,
                        pitch: width_usize * 4,
                        height: actual_height,
                        format: pixel_format,
                    };
                    JPEG_COMPRESSOR.with(|cell| -> Option<EncodedStripe> {
                        let mut slot = cell.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(turbojpeg::Compressor::new().ok()?);
                        }
                        let compressor = slot.as_mut().unwrap();
                        compressor.set_quality(quality_or_crf).ok()?;
                        let jpeg = compressor.compress_to_vec(img).ok()?;
                        let data = if omit_headers {
                            jpeg
                        } else {
                            stripe_state.packet_buf.clear();
                            push_jpeg_header(&mut stripe_state.packet_buf, frame_counter, y_start as u16);
                            stripe_state.packet_buf.extend_from_slice(&jpeg);
                            std::mem::take(&mut stripe_state.packet_buf)
                        };
                        Some(EncodedStripe {
                            data: Arc::new(data),
                            codec: Codec::Jpeg,
                            stripe_y_start: y_start as i32,
                            stripe_height: actual_height as i32,
                            frame_id: frame_counter as i32,
                        })
                    })
                } else {
                    cfg_if::cfg_if! {
                        if #[cfg(feature = "gpl")] {
                    let needs_reinit = if let Some(ref enc) = stripe_state.h264_encoder {
                        enc.width != width_usize as i32
                            || enc.height != actual_height as i32
                            || enc.is_i444 != video_fullcolor
                    } else {
                        true
                    };

                    if needs_reinit {
                        stripe_state.h264_encoder = H264EncoderWrapper::new(
                            width_usize as i32,
                            actual_height as i32,
                            quality_or_crf,
                            video_fullcolor,
                            target_fps,
                            h264_threads,
                            video_cbr,
                            video_bitrate,
                            video_vbv,
                            settings.video_min_qp,
                            settings.video_max_qp,
                        );
                        force_idr = true;
                    } else if let Some(ref mut enc) = stripe_state.h264_encoder {
                        enc.reconfigure_crf(quality_or_crf);
                        enc.reconfigure_rate(video_bitrate, video_vbv, target_fps);
                    }

                    if let Some(ref mut enc) = stripe_state.h264_encoder {
                        let y_size = width_usize * actual_height;
                        let uv_size = if video_fullcolor { y_size } else { y_size / 4 };
                        if stripe_state.y_buf.len() != y_size {
                            stripe_state.y_buf.resize(y_size, 0);
                        }
                        if stripe_state.u_buf.len() != uv_size {
                            stripe_state.u_buf.resize(uv_size, 0);
                        }
                        if stripe_state.v_buf.len() != uv_size {
                            stripe_state.v_buf.resize(uv_size, 0);
                        }

                        let y_stride = width_usize as i32;
                        let uv_stride =
                            (if video_fullcolor { width_usize } else { width_usize / 2 }) as i32;
                        let conversion_result = convert_to_yuv_mt(
                            stripe_bytes,
                            (width_usize * 4) as u32,
                            width_usize,
                            actual_height,
                            use_gpu,
                            video_fullcolor,
                            &mut stripe_state.y_buf,
                            &mut stripe_state.u_buf,
                            &mut stripe_state.v_buf,
                            (y_stride as usize, uv_stride as usize),
                            csc_bands,
                        );

                        if let Err(e) = conversion_result {
                            eprintln!(
                                "[software] YUV conversion failed for {}x{} stripe: {:?}; skipping",
                                width_usize, actual_height, e
                            );
                            return None;
                        }

                        if enc.encode_with_headers(
                            &stripe_state.y_buf,
                            &stripe_state.u_buf,
                            &stripe_state.v_buf,
                            y_stride,
                            uv_stride,
                            uv_stride,
                            frame_counter,
                            y_start as u16,
                            force_idr,
                            omit_headers,
                            &mut stripe_state.packet_buf,
                        ) {
                            Some(EncodedStripe {
                                data: Arc::new(std::mem::take(&mut stripe_state.packet_buf)),
                                codec: Codec::H264,
                                stripe_y_start: y_start as i32,
                                stripe_height: actual_height as i32,
                                frame_id: frame_counter as i32,
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                        } else {
                    use crate::encoders::oh264::Openh264Encoder;
                    let needs_reinit = stripe_state.h264_encoder.as_ref().is_none_or(|enc| {
                        enc.width() != width_usize || enc.height() != actual_height
                    });
                    if needs_reinit {
                        stripe_state.h264_encoder = Openh264Encoder::new_stripe(
                            settings,
                            width_usize,
                            actual_height,
                            quality_or_crf,
                            video_bitrate,
                            n_processing_stripes == 1,
                        );
                        if stripe_state.h264_encoder.is_none() {
                            // Once per process: a geometry OpenH264 refuses (wider than
                            // 3840, say) would otherwise log on every stripe of every frame.
                            static INIT_FAILED_LOGGED: std::sync::atomic::AtomicBool =
                                std::sync::atomic::AtomicBool::new(false);
                            if !INIT_FAILED_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                eprintln!(
                                    "[software] OpenH264 init failed for a {}x{} stripe; no software H.264 for it",
                                    width_usize, actual_height
                                );
                            }
                        }
                        force_idr = true;
                    } else if let Some(ref mut enc) = stripe_state.h264_encoder {
                        enc.update_qp(quality_or_crf.max(0) as u32);
                        enc.reconfigure_rate(video_bitrate, target_fps);
                    }

                    let enc = stripe_state.h264_encoder.as_mut()?;
                    match enc.encode_stripe_argb(
                        stripe_bytes,
                        width_usize * 4,
                        frame_counter as u64,
                        y_start as u16,
                        force_idr,
                        use_gpu,
                    ) {
                        Ok(data) if !data.is_empty() => Some(EncodedStripe {
                            data: Arc::new(data),
                            codec: Codec::H264,
                            stripe_y_start: y_start as i32,
                            stripe_height: actual_height as i32,
                            frame_id: frame_counter as i32,
                        }),
                        Ok(_) => None,
                        Err(e) => {
                            eprintln!("[software] OpenH264 encode failed for stripe at y={y_start}: {e}");
                            None
                        }
                    }
                        }
                    }
                }
            } else {
                None
            }
    };
    let encoded: Vec<EncodedStripe> = if n_processing_stripes <= 1 {
        stripes.iter_mut().enumerate().filter_map(&stripe_body).collect()
    } else {
        stripes.par_iter_mut().enumerate().filter_map(&stripe_body).collect()
    };
    // Follow motion spreading out quickly and narrowing slowly: the budget is
    // better spent late than overshot the moment a screen goes still again.
    let sent = encoded.len() as f32;
    let alpha = if sent > *carrying { CARRY_RISE } else { CARRY_FALL };
    *carrying += (sent - *carrying) * alpha;
    encoded
}

/// How many horizontal stripes a frame of `height` is split into, which is the choice of how
/// much encode parallelism to spend on it.
///
/// A full-frame session is one contiguous stream and so a single stripe; otherwise the frame
/// fans out across cores, bounded so no stripe is shorter than a macroblock row. Both the
/// encoder and the settings line report from here, so what is logged is what is encoded.
pub fn stripe_count(height: i32, codec: Codec, fullframe: bool) -> usize {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    if !codec.stripes() || (codec.is_video() && fullframe) || height < MIN_STRIPE_HEIGHT {
        return 1;
    }
    cores.min((height / MIN_STRIPE_HEIGHT) as usize).max(1)
}

/// Split the configured CBR budget across the stripes carrying it, returning the
/// `(bitrate_kbps, vbv_kbit)` each stripe's encoder is programmed with.
///
/// Every stripe runs its own encoder and rate control is per instance, metered against the
/// declared frame rate rather than against the frames that stripe was actually sent. So the
/// screen's rate is one stripe's rate times the number of stripes that carry motion, and the
/// budget is divided by that number — not by the stripe count, which on a screen where one
/// corner moves would spend a fraction of what was configured. The divisor is the smoothed
/// count so it changes on the scale of a moving average and not every frame: a rate that
/// swings frame to frame leaves the encoder chasing it and delivers less than either rate would.
fn stripe_rate_control(
    settings: &RustCaptureSettings, carrying: f32, n_stripes: usize,
) -> (i32, i32) {
    let divisor = (carrying.round().max(1.0) as usize).min(n_stripes.max(1)) as i32;
    let bitrate = (settings.video_bitrate_kbps / divisor).max(1);
    let vbv = (crate::encoders::vbv_bits(
        (bitrate as u32).saturating_mul(1000),
        settings.target_fps,
        settings.keyframe_interval_s,
        settings.video_vbv_multiplier,
    ) / 1000)
        .max(1) as i32;
    (bitrate, vbv)
}

/// Divide `height` into `n` contiguous stripes as `(y_start, stripe_height)`, with the split
/// rule differing by codec because only H.264 constrains stripe height.
///
/// - **JPEG**: JPEG has no vertical subsampling, so stripes may be any height; the heights differ
///   by at most one row — the first `remainder` stripes take one extra each — and every row of
///   the frame is covered.
/// - **Video**: 4:2:0 pairs chroma rows vertically, so every stripe height is forced even and the
///   remainder is handed out two rows at a time. The deliberate cost is that a single trailing
///   odd row may be left uncovered — preferable to an odd-height stripe the encoder cannot
///   represent.
fn compute_stripe_geometries(height: usize, n: usize, codec: Codec) -> Vec<(usize, usize)> {
    let mut geoms = Vec::with_capacity(n);
    let mut current_y = 0;
    if !codec.is_video() {
        let base_h = height / n;
        let remainder = height - base_h * n;
        for i in 0..n {
            let s_h = base_h + if i < remainder { 1 } else { 0 };
            geoms.push((current_y, s_h));
            current_y += s_h;
        }
    } else {
        let base_h = (height / n) & !1;
        let remainder = height - base_h * n;
        let stripes_with_extra = remainder / 2;
        for i in 0..n {
            let s_h = base_h + if i < stripes_with_extra { 2 } else { 0 };
            geoms.push((current_y, s_h));
            current_y += s_h;
        }
    }
    geoms
}

#[cfg(test)]
mod tests {
    /// The configured bitrate is a budget for the screen, not for each stripe: every stripe
    /// runs its own rate control, so what reaches an encoder is the budget over the number of
    /// stripes carrying motion. Dividing by the stripe count instead spends a fraction of the
    /// configured rate whenever only part of the screen moves, and dividing by nothing at all
    /// spends a multiple of it whenever the whole screen does.
    #[test]
    fn cbr_budget_is_split_across_the_stripes_carrying_it() {
        use crate::RustCaptureSettings;
        for &kbps in &[500i32, 4000, 8000, 20000] {
            let settings = RustCaptureSettings {
                video_cbr_mode: true,
                video_bitrate_kbps: kbps,
                ..Default::default()
            };
            for &n in &[1usize, 2, 4, 12, 64] {
                let (all, _) = super::stripe_rate_control(&settings, n as f32, n);
                let total = all * n as i32;
                assert!(
                    total <= kbps && kbps - total < n as i32,
                    "{n} stripes at {all} kbps must sum to the configured {kbps}"
                );
                let (one, _) = super::stripe_rate_control(&settings, 1.0, n);
                assert_eq!(one, kbps, "a lone moving stripe carries the whole budget");
                let (over, _) = super::stripe_rate_control(&settings, n as f32 * 4.0, n);
                assert_eq!(over, all, "the divisor never exceeds the stripes that exist");
                let (under, _) = super::stripe_rate_control(&settings, 0.0, n);
                assert_eq!(under, kbps, "and never falls below one");
            }
            let (vbv_one, whole) = super::stripe_rate_control(&settings, 1.0, 8);
            let (_, share) = super::stripe_rate_control(&settings, 8.0, 8);
            assert_eq!(vbv_one, kbps);
            assert!(
                (share * 8 - whole).abs() <= 9,
                "each stripe's buffer is its share of the whole-screen one: {share}x8 vs {whole}"
            );
        }
    }

    /// The divisor follows the screen rather than the configuration: full-screen motion moves
    /// it to the stripe count within a few frames, and it comes back down when the motion
    /// stops. A divisor recomputed per frame would swing between those two ends every frame,
    /// which leaves the encoder chasing a square wave and delivering less than either rate.
    #[test]
    fn the_budget_divisor_follows_motion_and_is_smoothed() {
        use crate::RustCaptureSettings;
        let (w, h) = (64, 512);
        let settings = RustCaptureSettings {
            width: w,
            height: h,
            codec: Codec::Jpeg,
            jpeg_quality: 40,
            use_paint_over_quality: false,
            ..Default::default()
        };
        let full = [smithay::utils::Rectangle::new((0, 0).into(), (w, h).into())];
        let stripes_n = super::stripe_count(h, settings.codec, settings.video_fullframe);
        if stripes_n < 2 {
            return;
        }
        let mut stripes = Vec::new();
        let mut carrying = 1.0f32;
        for frame in 0..40u16 {
            let shade = 40u8.wrapping_add(frame.wrapping_mul(7) as u8);
            let px = vec![shade; (w * h * 4) as usize];
            super::encode_cpu(
                &mut stripes, &mut carrying, &px, w, h, &full, &settings, frame,
                false, false, false,
            );
        }
        assert!(
            carrying > stripes_n as f32 * 0.75,
            "full-screen motion must move the divisor toward the {stripes_n} stripes it uses, \
             not leave it at {carrying}"
        );
        let moved = carrying;
        // Motion that narrows to one corner narrows the divisor with it, so the budget
        // follows the stripes that are actually spending it. A frame with no motion at all
        // encodes nothing and carries nothing, so it leaves the divisor where it was.
        let band = [smithay::utils::Rectangle::new((0, 0).into(), (w, 64).into())];
        for frame in 40..120u16 {
            let shade = 40u8.wrapping_add(frame.wrapping_mul(11) as u8);
            let mut px = vec![200u8; (w * h * 4) as usize];
            for byte in px.iter_mut().take((w * 64 * 4) as usize) {
                *byte = shade;
            }
            super::encode_cpu(
                &mut stripes, &mut carrying, &px, w, h, &band, &settings, frame,
                false, false, false,
            );
        }
        assert!(
            carrying < moved * 0.5,
            "motion in one stripe must bring the divisor back down: {carrying} vs {moved}"
        );
    }
    use super::{compute_stripe_geometries, Codec, StripeState};

    /// Without `gpl` the striped H.264 path runs one OpenH264 instance per stripe and speaks
    /// the x264 stripes' protocol: the first frame emits every stripe as an IDR whose wire header
    /// carries that stripe's y-start and geometry, each stripe is an independently decodable
    /// stream (a decoder fed only that stripe's bytes yields a picture of the stripe's size), a
    /// static follow-up frame sends nothing, and motion confined to the top rows re-sends only
    /// the top stripe, as a delta frame.
    #[cfg(not(feature = "gpl"))]
    #[test]
    fn openh264_stripes_are_independent_streams() {
        use crate::RustCaptureSettings;
        use openh264::decoder::Decoder;
        use openh264::formats::YUVSource;
        let (w, h) = (128, 512);
        let settings = RustCaptureSettings {
            width: w,
            height: h,
            codec: Codec::H264,
            video_crf: 25,
            use_paint_over_quality: false,
            video_streaming_mode: false,
            ..Default::default()
        };
        let n = super::stripe_count(h, settings.codec, settings.video_fullframe);
        if n < 2 {
            return;
        }
        let mut stripes = Vec::new();
        let mut carrying = 1.0f32;
        let px: Vec<u8> = (0..(w * h * 4) as usize).map(|i| (i % 251) as u8).collect();
        let first = super::encode_cpu(
            &mut stripes, &mut carrying, &px, w, h, &[], &settings, 0, false, true, false,
        );
        assert_eq!(first.len(), n, "every stripe is sent on the first frame");
        for (stripe, (y, sh)) in first.iter().zip(compute_stripe_geometries(h as usize, n, Codec::H264)) {
            let d = &stripe.data;
            assert_eq!(d[0], 0x04, "H.264 stripe tag");
            assert_eq!(d[1], 0x11, "first frame of a stripe is an H.264 key frame");
            assert_eq!(u16::from_be_bytes([d[2], d[3]]), 0, "frame number");
            assert_eq!(u16::from_be_bytes([d[4], d[5]]) as usize, y, "y-start");
            assert_eq!(u16::from_be_bytes([d[6], d[7]]) as i32, w, "width");
            assert_eq!(u16::from_be_bytes([d[8], d[9]]) as usize, sh, "stripe height");
            assert_eq!((stripe.stripe_y_start as usize, stripe.stripe_height as usize), (y, sh));
            let mut dec = Decoder::new().expect("decoder");
            let img = dec.decode(&d[10..]).expect("decode").expect("an IDR decodes on its own");
            assert_eq!(img.dimensions(), (w as usize, sh), "each stripe is its own stream");
        }
        let quiet = super::encode_cpu(
            &mut stripes, &mut carrying, &px, w, h, &[], &settings, 1, false, true, false,
        );
        assert!(quiet.is_empty(), "a static frame sends nothing");
        let mut moved = px.clone();
        for b in moved.iter_mut().take((w * 8 * 4) as usize) {
            *b = b.wrapping_add(97);
        }
        let top = super::encode_cpu(
            &mut stripes, &mut carrying, &moved, w, h, &[], &settings, 2, false, true, false,
        );
        assert_eq!(top.len(), 1, "motion in the top rows re-sends the top stripe alone");
        assert_eq!(top[0].stripe_y_start, 0);
        assert_eq!(top[0].data[1], 0x10, "an unforced follow-up is an H.264 delta frame");
        assert_eq!(u16::from_be_bytes([top[0].data[2], top[0].data[3]]), 2, "frame number");
    }

    /// With `threshold = 2` and `duration = 3`, a first change reads dirty and two consecutive
    /// changes open a damage block that holds dirty for three frames without re-hashing; once content
    /// has gone static, the end-of-block re-hash exits the block and the stripe reads clean again.
    #[test]
    fn content_dirty_detects_change_and_damage_block() {
        let mut st = StripeState::default();
        let a = vec![1u8; 256];
        let b = vec![2u8; 256];
        assert!(st.content_dirty(&a, 2, 3));
        assert!(!st.content_dirty(&a, 2, 3));
        assert!(st.content_dirty(&b, 2, 3));
        assert!(st.content_dirty(&a, 2, 3));
        assert!(st.in_damage_block);
        assert!(st.content_dirty(&a, 2, 3));
        assert!(st.content_dirty(&a, 2, 3));
        assert!(st.content_dirty(&a, 2, 3));
        assert!(!st.in_damage_block);
        assert!(!st.content_dirty(&a, 2, 3));
    }

    /// With compositor damage as the authority (Wayland), a clean frame must still advance the
    /// paint-over countdown and fire the repaint at the trigger, and once every stripe has
    /// latched (`paint_over_sent`) further clean frames must produce nothing — that quiescent
    /// tail is the idle fast path, which skips the stripe fan-out entirely.
    #[test]
    fn clean_frames_countdown_fire_paintover_then_go_quiescent() {
        use crate::RustCaptureSettings;
        let (w, h) = (64, 128);
        let pixels = vec![128u8; (w * h * 4) as usize];
        let settings = RustCaptureSettings {
            width: w,
            height: h,
            codec: Codec::Jpeg,
            jpeg_quality: 60,
            paint_over_jpeg_quality: 90,
            use_paint_over_quality: true,
            paint_over_trigger_frames: 5,
            ..Default::default()
        };
        let mut stripes = Vec::new();
        let mut carrying = 1.0f32;
        let full = [smithay::utils::Rectangle::new(
            (0, 0).into(),
            (w, h).into(),
        )];
        let dirty = super::encode_cpu(
            &mut stripes, &mut carrying, &pixels, w, h, &full, &settings, 0, false, false, false,
        );
        assert!(!dirty.is_empty(), "damaged frame must encode");

        let mut fired_at = None;
        for frame in 1..=20u16 {
            let out = super::encode_cpu(
                &mut stripes, &mut carrying, &pixels, w, h, &[], &settings, frame, false, false, false,
            );
            if !out.is_empty() {
                assert!(fired_at.is_none(), "paint-over must fire exactly once");
                fired_at = Some(frame);
            }
        }
        assert_eq!(fired_at, Some(settings.paint_over_trigger_frames as u16));
        assert!(
            stripes.iter().all(|st| st.paint_over_sent),
            "all stripes latched after the repaint"
        );
    }

    /// Hash-damage sources (X11) take the sequential-scan fast path: static frames advance
    /// the countdown and fire the paint-over exactly once, the quiescent tail emits nothing,
    /// and a subsequent content change is still detected and encoded (streak state reset by
    /// the fast path must not swallow the wake-up).
    #[test]
    fn hash_scan_idles_after_paintover_and_wakes_on_change() {
        use crate::RustCaptureSettings;
        let (w, h) = (64, 128);
        let static_px = vec![128u8; (w * h * 4) as usize];
        let changed_px = vec![200u8; (w * h * 4) as usize];
        let settings = RustCaptureSettings {
            width: w,
            height: h,
            codec: Codec::Jpeg,
            jpeg_quality: 60,
            paint_over_jpeg_quality: 90,
            use_paint_over_quality: true,
            paint_over_trigger_frames: 5,
            damage_block_threshold: 10,
            damage_block_duration: 10,
            ..Default::default()
        };
        let mut stripes = Vec::new();
        let mut carrying = 1.0f32;
        let first = super::encode_cpu(
            &mut stripes, &mut carrying, &static_px, w, h, &[], &settings, 0, false, true, false,
        );
        assert!(!first.is_empty(), "first frame hashes as changed and encodes");

        let mut fired_at = None;
        for frame in 1..=20u16 {
            let out = super::encode_cpu(
                &mut stripes, &mut carrying, &static_px, w, h, &[], &settings, frame, false, true, false,
            );
            if !out.is_empty() {
                assert!(fired_at.is_none(), "paint-over must fire exactly once while static");
                fired_at = Some(frame);
            }
        }
        assert_eq!(fired_at, Some(settings.paint_over_trigger_frames as u16));

        let woke = super::encode_cpu(
            &mut stripes, &mut carrying, &changed_px, w, h, &[], &settings, 21, false, true, false,
        );
        assert!(!woke.is_empty(), "content change after idle must encode");
    }

    /// Total rows covered by a geometry — the sum of all stripe heights.
    fn covered(geoms: &[(usize, usize)]) -> usize {
        geoms.iter().map(|&(_, h)| h).sum()
    }

    /// Assert the stripes tile the frame with no gaps or overlap: each stripe's `y_start`
    /// equals the running sum of the preceding heights.
    fn assert_contiguous(geoms: &[(usize, usize)]) {
        let mut y = 0;
        for &(sy, sh) in geoms {
            assert_eq!(sy, y, "stripes must be contiguous");
            y += sh;
        }
    }

    /// JPEG geometry covers the full frame height with contiguous stripes, across a range of
    /// heights (odd ones included) and stripe counts.
    #[test]
    fn jpeg_covers_every_row_including_odd() {
        for &h in &[1usize, 63, 720, 721, 1079, 1080, 1081] {
            for &n in &[1usize, 2, 3, 8, 16] {
                let g = compute_stripe_geometries(h, n, Codec::Jpeg);
                assert_eq!(g.len(), n);
                assert_eq!(covered(&g), h, "JPEG must cover full height h={} n={}", h, n);
                assert_contiguous(&g);
            }
        }
    }

    /// H.264 geometry yields even, contiguous stripe heights that cover the whole frame
    /// except at most one trailing odd row, across a range of heights and stripe counts.
    #[test]
    fn h264_stripes_even_and_within_bounds() {
        for &h in &[64usize, 720, 721, 1080, 1081] {
            for &n in &[1usize, 2, 8] {
                let g = compute_stripe_geometries(h, n, Codec::H264);
                assert_eq!(g.len(), n);
                for &(_, sh) in &g {
                    assert_eq!(sh % 2, 0, "H.264 stripe heights must be even h={} n={}", h, n);
                }
                assert_contiguous(&g);
                assert!(covered(&g) <= h);
                assert!(h - covered(&g) <= 1, "at most one trailing odd row uncovered");
            }
        }
    }
}

#[cfg(test)]
mod qp_bound_sweep {
    //! Invariants under test: the CBR QP clamp reaches libx264/OpenH264 (a max clamp must
    //! raise worst-case fidelity on rate-starved text at the cost of bitrate overshoot;
    //! a min clamp must cut spend on over-budgeted content) and defaults (0) leave the
    //! encoders' own behavior untouched. Each encoder is swept separately: the OpenH264
    //! sweep runs in every build (the crate is a dev-dependency), the x264 one needs `gpl`.
    #[cfg(feature = "gpl")]
    use super::H264EncoderWrapper;
    use crate::encoders::oh264::Openh264Encoder;
    use crate::encoders::Codec;
    use crate::RustCaptureSettings;
    use openh264::decoder::Decoder;
    use openh264::formats::YUVSource;

    const W: usize = 1280;
    const H: usize = 720;
    const FRAMES: usize = 60;

    /// Build a scrolling terminal-like luma frame: an 8x12 glyph grid seeded by an LCG and
    /// scrolled 4 px per frame — the worst case for screen-share rate control, with dense
    /// high-contrast detail (~40% lit pixels per glyph row) under full-frame motion.
    fn text_luma(frame: usize) -> Vec<u8> {
        let mut y = vec![18u8; W * H];
        let scroll = frame * 4;
        for row in 0..H {
            let srow = row + scroll;
            let cell_y = srow / 12;
            let in_glyph_y = srow % 12;
            if in_glyph_y >= 10 {
                continue;
            }
            for col in 0..W {
                let cell_x = col / 8;
                let in_glyph_x = col % 8;
                if in_glyph_x >= 7 {
                    continue;
                }
                let mut s = (cell_x as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add((cell_y as u32).wrapping_mul(40503))
                    .wrapping_add(1);
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                if (s >> ((in_glyph_y * 3 + in_glyph_x) % 29)) & 1 == 1 {
                    y[row * W + col] = 224;
                }
            }
        }
        y
    }

    /// Encode `FRAMES` scrolling-text luma frames through the x264 stripe encoder at the
    /// given rate-control settings (constant grey chroma), returning each frame's raw bitstream.
    #[cfg(feature = "gpl")]
    fn encode_x264(cbr: bool, kbps: i32, crf: i32, min_qp: i32, max_qp: i32) -> Vec<Vec<u8>> {
        let mut enc = H264EncoderWrapper::new(
            W as i32, H as i32, crf, false, 60.0, 4, cbr, kbps, 50, min_qp, max_qp,
        )
        .expect("x264 init");
        let u = vec![128u8; (W / 2) * (H / 2)];
        let v = vec![128u8; (W / 2) * (H / 2)];
        (0..FRAMES)
            .map(|i| {
                let y = text_luma(i);
                let mut out = Vec::new();
                enc.encode_with_headers(
                    &y, &u, &v, W as i32, (W / 2) as i32, (W / 2) as i32,
                    i as u16, 0, i == 0, true, &mut out,
                );
                out
            })
            .collect()
    }

    /// Encode the same scrolling-text sequence through the OpenH264 full-frame encoder (luma
    /// broadcast to a grey BGRA frame), returning each frame's bitstream for comparison with the
    /// x264 run.
    fn encode_oh264(cbr: bool, kbps: i32, crf: i32, min_qp: i32, max_qp: i32) -> Vec<Vec<u8>> {
        let s = RustCaptureSettings {
            width: W as i32,
            height: H as i32,
            target_fps: 60.0,
            codec: Codec::H264,
            video_cbr_mode: cbr,
            video_bitrate_kbps: kbps,
            video_crf: crf,
            video_min_qp: min_qp,
            video_max_qp: max_qp,
            ..Default::default()
        };
        let mut enc = Openh264Encoder::new(&s).expect("oh264 init");
        (0..FRAMES)
            .map(|i| {
                let y = text_luma(i);
                let mut bgra = vec![255u8; W * H * 4];
                for (px, &l) in bgra.chunks_exact_mut(4).zip(y.iter()) {
                    px[0] = l;
                    px[1] = l;
                    px[2] = l;
                }
                enc.encode_host_argb(&bgra, W * 4, i as u64, i == 0, false)
                    .expect("oh264 encode")
            })
            .collect()
    }

    /// Decode a sequence of H.264 frames back to tightly-packed luma planes (dropping empty
    /// frames and the decoded chroma) for PSNR comparison.
    fn decode_luma(frames: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut dec = Decoder::new().expect("decoder");
        let mut out = Vec::new();
        for f in frames {
            if f.is_empty() {
                continue;
            }
            if let Ok(Some(img)) = dec.decode(f) {
                let (w, h) = img.dimensions();
                let stride = img.strides().0;
                let mut y = vec![0u8; w * h];
                for r in 0..h {
                    y[r * w..r * w + w].copy_from_slice(&img.y()[r * stride..r * stride + w]);
                }
                out.push(y);
            }
        }
        out
    }

    /// Mean per-frame luma PSNR (dB) between two decoded sequences, treating a zero-MSE frame
    /// as 99 dB.
    fn mean_psnr(a: &[Vec<u8>], b: &[Vec<u8>]) -> f64 {
        let n = a.len().min(b.len());
        let mut acc = 0.0;
        for i in 0..n {
            let mse: f64 = a[i]
                .iter()
                .zip(b[i].iter())
                .map(|(&x, &y)| {
                    let d = x as f64 - y as f64;
                    d * d
                })
                .sum::<f64>()
                / a[i].len() as f64;
            acc += if mse <= 0.0 { 99.0 } else { 10.0 * (255.0f64 * 255.0 / mse).log10() };
        }
        acc / n.max(1) as f64
    }

    /// Average encoded bitrate (kbps) of a frame sequence, assuming 60 fps playback.
    fn kbps(frames: &[Vec<u8>]) -> f64 {
        frames.iter().map(|f| f.len()).sum::<usize>() as f64 * 8.0 * 60.0
            / FRAMES as f64
            / 1000.0
    }

    /// Diagnostic that the CBR QP clamp is actually plumbed through to x264, printing a
    /// bitrate/PSNR table on scrolling text and asserting the effect.
    ///
    /// Encodes worst-case scrolling text at 2 Mbps CBR across a sweep of `max_qp` values (plus a
    /// separate `min_qp` sweep on an over-provisioned 12 Mbps budget), measuring luma PSNR against a
    /// near-lossless CRF-12 reference from the same encoder so colour-conversion differences cancel
    /// out. Capping `max_qp` at 30 on rate-starved content must lift fidelity by more than 0.5 dB
    /// over the unclamped run — proving the clamp reaches the encoder rather than being silently
    /// dropped (paid for in bitrate overshoot).
    #[cfg(feature = "gpl")]
    #[test]
    fn cbr_qp_bound_sweep_x264() {
        let reference = decode_luma(&encode_x264(false, 0, 12, 0, 0));

        println!("scrolling-text 720p60 @ 2 Mbps CBR, x264 (PSNR vs own CRF-12 decode):");
        let mut rows = Vec::new();
        for &max_qp in &[0i32, 45, 40, 35, 30] {
            let x = encode_x264(true, 2000, 25, 0, max_qp);
            let psnr = mean_psnr(&decode_luma(&x), &reference);
            println!("  max_qp {:>2}: {:>8.1} kbps / {:>5.2} dB", max_qp, kbps(&x), psnr);
            rows.push((max_qp, psnr));
        }
        println!("scrolling-text 720p60 @ 12 Mbps CBR, x264 min-QP sweep:");
        for &min_qp in &[0i32, 10, 15] {
            let x = encode_x264(true, 12000, 25, min_qp, 0);
            let psnr = mean_psnr(&decode_luma(&x), &reference);
            println!("  min_qp {:>2}: {:>8.1} kbps / {:>5.2} dB", min_qp, kbps(&x), psnr);
        }

        let base = rows[0].1;
        let capped = rows.last().unwrap().1;
        assert!(
            capped > base + 0.5,
            "x264 max-QP clamp had no effect: {capped:.2} vs {base:.2} dB"
        );
    }

    /// The OpenH264 counterpart of [`cbr_qp_bound_sweep_x264`]: the software H.264 sweep of a
    /// build without the `gpl` feature, where OpenH264 is the encoder behind every stripe.
    ///
    /// Same scrolling-text workload and 2 Mbps CBR `max_qp` sweep, with luma PSNR measured against
    /// this encoder's own near-lossless QP-12 reference. Capping `max_qp` at 30 must lift fidelity
    /// by more than 0.5 dB over the unclamped run.
    #[test]
    fn cbr_qp_bound_sweep_openh264() {
        let reference = decode_luma(&encode_oh264(false, 0, 12, 0, 0));

        println!("scrolling-text 720p60 @ 2 Mbps CBR, oh264 (PSNR vs own QP-12 decode):");
        let mut rows = Vec::new();
        for &max_qp in &[0i32, 45, 40, 35, 30] {
            let o = encode_oh264(true, 2000, 25, 0, max_qp);
            let psnr = mean_psnr(&decode_luma(&o), &reference);
            println!("  max_qp {:>2}: {:>8.1} kbps / {:>5.2} dB", max_qp, kbps(&o), psnr);
            rows.push((max_qp, psnr));
        }

        let base = rows[0].1;
        let capped = rows.last().unwrap().1;
        assert!(
            capped > base + 0.5,
            "oh264 max-QP clamp had no effect: {capped:.2} vs {base:.2} dB"
        );
    }

    /// After a live frame-rate change the CBR stream still tracks the configured bitrate.
    ///
    /// x264's per-frame CBR budget is `bitrate / fps`, so halving the frame rate at a fixed kbps
    /// budget must roughly double each encoded frame while the per-second bitrate holds. This encodes
    /// incompressible full-frame noise at 4 Mbps CBR (content the rate controller cannot undershoot,
    /// so per-frame size sits at the budget), measures the mean encoded frame size at 60 fps, drops
    /// to 30 fps through `reconfigure_rate`, and requires the per-frame size to roughly double — so
    /// the per-second bitrate is preserved rather than collapsing to half, which is what the pre-fix
    /// path did by leaving the session budgeting for 60 fps (`x264_encoder_reconfig` never applies a
    /// frame-rate change). Single-threaded so the rate-control measurement is deterministic; warmup
    /// frames are discarded so the ABR controller and the post-reopen IDR do not skew the mean.
    #[cfg(feature = "gpl")]
    #[test]
    fn cbr_bitrate_tracks_configured_rate_after_fps_change() {
        const TARGET_KBPS: i32 = 4000;
        const WARMUP: usize = 24;
        const MEASURED: usize = 96;
        let u = vec![128u8; (W / 2) * (H / 2)];
        let v = vec![128u8; (W / 2) * (H / 2)];
        let vbv = |fps: f64| {
            (crate::encoders::vbv_bits((TARGET_KBPS as u32) * 1000, fps, 0.0, 0.0) / 1000).max(1)
                as i32
        };
        // A fresh incompressible luma plane every frame, so inter-prediction cannot cheapen a
        // frame and CBR must spend its whole per-frame budget: the per-frame size then reads the
        // budget directly, which is exactly what a frame-rate change is supposed to move.
        let noise_luma = |frame: usize| -> Vec<u8> {
            let mut y = vec![0u8; W * H];
            let mut s = (frame as u32).wrapping_mul(2654435761).wrapping_add(1);
            for p in y.iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *p = (s >> 24) as u8;
            }
            y
        };

        let mut enc =
            H264EncoderWrapper::new(W as i32, H as i32, 25, false, 60.0, 1, true, TARGET_KBPS, vbv(60.0), 0, 0)
                .expect("x264 init");

        let measure = |enc: &mut H264EncoderWrapper, start: usize| -> f64 {
            let mut bytes = 0usize;
            let mut counted = 0usize;
            for i in start..start + WARMUP + MEASURED {
                let y = noise_luma(i);
                let mut out = Vec::new();
                enc.encode_with_headers(
                    &y, &u, &v, W as i32, (W / 2) as i32, (W / 2) as i32, i as u16, 0, i == start,
                    true, &mut out,
                );
                if i >= start + WARMUP && !out.is_empty() {
                    bytes += out.len();
                    counted += 1;
                }
            }
            bytes as f64 / counted.max(1) as f64
        };

        let per_frame_60 = measure(&mut enc, 0);
        enc.reconfigure_rate(TARGET_KBPS, vbv(30.0), 30.0);
        let per_frame_30 = measure(&mut enc, 1000);

        let eff_60 = per_frame_60 * 8.0 * 60.0 / 1000.0;
        let eff_30 = per_frame_30 * 8.0 * 30.0 / 1000.0;
        println!(
            "x264 CBR {TARGET_KBPS} kbps: 60fps {eff_60:.0} kbps ({per_frame_60:.0} B/frame), 30fps {eff_30:.0} kbps ({per_frame_30:.0} B/frame)"
        );

        let ratio = per_frame_30 / per_frame_60.max(1.0);
        assert!(
            (1.5..2.6).contains(&ratio),
            "30fps per-frame size {per_frame_30:.0} B vs 60fps {per_frame_60:.0} B (ratio {ratio:.2}); halving fps should roughly double each frame"
        );
        assert!(
            eff_30 > eff_60 * 0.75,
            "30fps effective bitrate {eff_30:.0} kbps collapsed from the 60fps {eff_60:.0} kbps; fps change did not re-budget the bitrate"
        );
    }
}
