/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Frame-processing policy shared by the two capture backends. It lives in its own module for one
//! reason: the Wayland path (dmabuf, compositor damage) and the X11 path (host-ARGB, stripe-hash
//! damage) capture pixels in completely different ways, but a viewer must never be able to tell
//! which one produced a frame — a paint-over refresh or a recovery keyframe has to behave
//! identically either way. Keeping the decision logic here, source-agnostic, is what guarantees it.

use crate::encoders::nvenc::NvencEncoder;
use crate::encoders::oh264::Openh264Encoder;
use crate::encoders::software::{encode_cpu, EncodedStripe, StripeState};
use crate::encoders::vaapi::VaapiEncoder;
use crate::RustCaptureSettings;
use std::sync::Arc;

/// Outcome of the full-frame H.264 send decision produced by `decide_hw_fullframe`.
pub struct HwFrameDecision {
    pub send: bool,
    pub force_idr: bool,
    pub target_qp: u32,
}

/// Whether a scheduled keyframe is due this tick.
///
/// The default (`keyframe_interval_s <= 0`) is an infinite GOP with no scheduled IDRs. A positive
/// interval buys a fixed ~N-second recovery cadence for consumers that cannot request one on demand.
///
/// # Arguments
///
/// * `settings` - Capture settings; reads `keyframe_interval_s` and `target_fps`.
/// * `frame_counter` - Current frame number (wrapping `u16`).
///
/// # Returns
///
/// `true` if a periodic IDR should be forced on this frame.
pub fn periodic_idr_due(settings: &RustCaptureSettings, frame_counter: u16) -> bool {
    let secs = settings.keyframe_interval_s;
    if secs <= 0.0 {
        return false;
    }
    let safe_fps = settings.target_fps.max(1.0);
    let interval = ((safe_fps * secs).round() as u64).max(1);
    (frame_counter as u64).is_multiple_of(interval)
}

/// The send / quality / keyframe policy every full-frame H.264 encoder obeys.
///
/// A static screen costs almost nothing to stream; a client that just joined or reset can always
/// recover a clean picture. The GOP is left infinite and an IDR is forced only when a consumer
/// genuinely needs a fresh decode entry point. Every forced IDR is followed by a short "recovery
/// burst" that keeps streaming until rate control converges. All three encoders (NVENC / VAAPI /
/// OpenH264) share this one function so they cannot drift apart.
///
/// Priority order: (1) recovery burst in progress, (2) always-on streaming/animated modes,
/// (3) motion detected, (4) recovery keyframe on static screen, (5) paint-over refresh.
///
/// # Arguments
///
/// * `st` - Per-stripe mutable state carrying paint-over and burst bookkeeping.
/// * `settings` - Capture settings; reads CRF, paint-over CRF, burst frames, trigger frames,
///   streaming mode, and keyframe interval.
/// * `frame_counter` - Current frame number (wrapping `u16`).
/// * `is_dirty` - Motion signal from the backend (compositor damage or stripe-hash change).
/// * `is_animated` - Forces a send for animated overlays.
/// * `requested_idr` - On-demand IDR request (client join / reset / recording cadence).
///
/// # Returns
///
/// [`HwFrameDecision`] with `send` (whether to encode this frame), `force_idr` (force a
/// keyframe), and `target_qp` (quality target for rate control).
pub fn decide_hw_fullframe(
    st: &mut StripeState,
    settings: &RustCaptureSettings,
    frame_counter: u16,
    is_dirty: bool,
    is_animated: bool,
    requested_idr: bool,
) -> HwFrameDecision {
    let normal_qp = settings.video_crf as u32;
    let paint_qp = settings.video_paintover_crf as u32;
    let trigger_frames = settings.paint_over_trigger_frames;
    let use_paint_over = settings.use_paint_over_quality;
    let burst = settings.video_paintover_burst_frames;
    let streaming = settings.video_streaming_mode;
    let burst_qp = if use_paint_over && paint_qp < normal_qp { paint_qp } else { normal_qp };

    let mut send_frame = false;
    let mut force_idr = false;
    let mut target_qp = normal_qp;

    if st.h264_burst_frames_remaining > 0 {
        send_frame = true;
        target_qp = burst_qp;
        st.h264_burst_frames_remaining -= 1;

        if is_dirty {
            st.h264_burst_frames_remaining = 0;
            st.paint_over_sent = false;
            target_qp = normal_qp;
        }
    }

    if !send_frame && (streaming || is_animated) {
        send_frame = true;
    }

    let recovery_idr = requested_idr || periodic_idr_due(settings, frame_counter);

    if is_dirty {
        send_frame = true;
        force_idr = recovery_idr;
        st.no_motion_frame_count = 0;
        st.paint_over_sent = false;
        st.h264_burst_frames_remaining = 0;
        target_qp = normal_qp;
    } else if recovery_idr {
        send_frame = true;
        force_idr = true;
        if st.h264_burst_frames_remaining <= 0 {
            target_qp = normal_qp;
            if burst > 0 {
                st.paint_over_sent = true;
                st.h264_burst_frames_remaining = burst;
            }
        }
    } else if !send_frame {
        st.no_motion_frame_count += 1;

        if use_paint_over
            && st.no_motion_frame_count >= trigger_frames
            && !st.paint_over_sent
            && paint_qp < normal_qp
        {
            send_frame = true;
            st.paint_over_sent = true;
            target_qp = paint_qp;
            st.h264_burst_frames_remaining = burst - 1;
        }
    }

    HwFrameDecision { send: send_frame, force_idr, target_qp }
}

/// Hardware encoder bound to the X11 (host-ARGB) pipeline. Software JPEG/x264 needs no
/// persistent encoder object (`encode_cpu` owns per-stripe x264 state), so it is `None`.
#[allow(clippy::large_enum_variant)]
enum X11Encoder {
    None,
    Nvenc(NvencEncoder),
    Vaapi(VaapiEncoder),
    Openh264(Openh264Encoder),
}

/// Software H.264 encoder used when hardware encoders are unavailable or the CPU is forced.
/// With GPL components this is the striped x264 path (its state lives per-stripe inside
/// `encode_cpu`, so no persistent object exists here); without them (the crate built with the
/// `gpl` feature disabled) the BSD-licensed OpenH264 full-frame encoder is substituted instead.
#[cfg(feature = "gpl")]
fn software_h264_encoder(settings: &RustCaptureSettings) -> X11Encoder {
    let _ = settings;
    X11Encoder::None
}

#[cfg(not(feature = "gpl"))]
fn software_h264_encoder(settings: &RustCaptureSettings) -> X11Encoder {
    if settings.output_mode != 1 {
        return X11Encoder::None;
    }
    println!("[x11] pixelflux was built without GPL components (libx264 disabled); substituting the BSD-licensed OpenH264 software encoder.");
    match Openh264Encoder::new(settings) {
        Some(e) => X11Encoder::Openh264(e),
        None => {
            eprintln!("[x11] OpenH264 init failed; no software H.264 encoder is available in this GPL-free build.");
            X11Encoder::None
        }
    }
}

/// Choose the full-frame encoder for the X11 host-ARGB path, following the settings and the
/// effective encode device. Used both at construction and when a live session has to be rebuilt
/// after a streak of encode failures.
fn select_encoder(settings: &RustCaptureSettings) -> X11Encoder {
    if settings.output_mode == 1 && settings.use_openh264 {
        println!("[x11] OpenH264 software encoder selected.");
        return match Openh264Encoder::new(settings) {
            Some(e) => X11Encoder::Openh264(e),
            None => {
                #[cfg(feature = "gpl")]
                eprintln!("[x11] OpenH264 init failed; falling back to software x264");
                #[cfg(not(feature = "gpl"))]
                eprintln!("[x11] OpenH264 init failed; no software H.264 encoder is available in this GPL-free build.");
                X11Encoder::None
            }
        };
    }
    if settings.output_mode == 1 && !settings.use_cpu && settings.encode_node_index != -1 {
        let encode_driver = crate::get_gpu_driver(settings.encode_node_index.max(0));
        println!(
            "[x11] Encode Node Index: {} | Driver: {}",
            settings.encode_node_index.max(0), encode_driver
        );
        if !encode_driver.is_empty() && !encode_driver.contains("nvidia") {
            println!("[x11] Initializing Unified VAAPI Encoder...");
            match VaapiEncoder::new_host(settings) {
                Ok(e) => {
                    println!(
                        "[x11] VAAPI Encoder initialized successfully ({}).",
                        if e.is_fullcolor() { "4:4:4" } else { "4:2:0" }
                    );
                    return X11Encoder::Vaapi(e);
                }
                Err(err) => {
                    eprintln!("[x11] Failed to init VAAPI: {err}. Falling back to CPU.");
                    return software_h264_encoder(settings);
                }
            }
        }
        println!("[x11] Nvidia Encoder detected. Initializing NVENC...");
        return match NvencEncoder::new(settings, std::ptr::null()) {
            Ok(e) => {
                println!("[x11] NVENC Encoder initialized successfully.");
                X11Encoder::Nvenc(e)
            }
            Err(err) => {
                eprintln!("[x11] Failed to init NVENC: {err}. Falling back to CPU.");
                software_h264_encoder(settings)
            }
        };
    }
    if settings.output_mode == 1 {
        println!("[x11] No GPU Encoder available -> Using CPU Software Encoding.");
    }
    software_h264_encoder(settings)
}

/// Everything the X11 host-ARGB path has to remember between frames.
///
/// Unlike the Wayland backend, X11 capture has no compositor to report what changed, so this
/// context exists to hold the state that stands in for that missing damage signal: the per-stripe
/// hashes and the persistent encoder session that let `process()` discover damage by comparing
/// content frame-to-frame. Full-frame H.264 runs through `decide_hw_fullframe`; striped JPEG/x264
/// runs through `encode_cpu` with `hash_damage=true`.
///
/// Recording fan-out (socket sink and MP4 recorder) is handled at the delivery layer; a
/// consumer needing a keyframe goes through [`X11Pipeline::request_idr`] like everyone else.
pub struct X11Pipeline {
    settings: RustCaptureSettings,
    stripes: Vec<StripeState>,
    /// Smoothed number of stripes carrying the encode budget (see `stripe_rate_control`).
    stripes_carrying: f32,
    hw: X11Encoder,
    hw_state: StripeState,
    frame_counter: u16,
    pending_force_idr: bool,
    /// Consecutive hardware encode failures, whether this pipeline already spent its one
    /// rebuild, and whether it has reached the bottom of the ladder; together they drive
    /// `recover_hw`.
    hw_error_streak: u32,
    hw_rebuilt: bool,
    hw_recovery_exhausted: bool,
}

impl X11Pipeline {
    /// Build the context, choosing the full-frame encoder for the X11 host-ARGB path.
    ///
    /// # Arguments
    ///
    /// * `settings` - Capture configuration. The `output_mode`, `use_openh264`, `use_cpu`, and
    ///   `encode_node_index` fields drive encoder selection.
    ///
    /// # Encoder selection
    ///
    /// 1. **OpenH264** — when `use_openh264` is set (explicit opt-in, full-frame software).
    /// 2. **NVENC** — on an NVIDIA driver (or no detectable GPU, since the attempt is cheap).
    /// 3. **VA-API** — on any other GPU driver. A 4:4:4 request is carried into the attempt, so a
    ///    device that cannot encode 4:4:4 fails here and falls through rather than being ruled out
    ///    in advance.
    /// 4. **Software** — `software_h264_encoder`: `X11Encoder::None`, meaning the striped x264
    ///    inside `encode_cpu`, in a GPL build; the full-frame OpenH264 encoder when the `gpl`
    ///    feature is disabled. This is also where a 4:4:4 request lands when no hardware encoder
    ///    can carry it, since x264 can.
    pub fn new(settings: RustCaptureSettings) -> Self {
        let hw = select_encoder(&settings);
        Self {
            settings,
            stripes: Vec::new(),
            stripes_carrying: 1.0,
            hw,
            hw_state: StripeState::default(),
            frame_counter: 0,
            pending_force_idr: false,
            hw_error_streak: 0,
            hw_rebuilt: false,
            hw_recovery_exhausted: false,
        }
    }

    /// Whether this build offers an encoder below the one currently running. Without GPL
    /// components the software rung *is* OpenH264, so an OpenH264 session has nothing under it
    /// and the ladder ends with it; with libx264 the rung below is always the striped software
    /// path, which `X11Encoder::None` selects.
    fn has_rung_below_hw(&self) -> bool {
        #[cfg(not(feature = "gpl"))]
        if self.settings.output_mode == 1 && matches!(self.hw, X11Encoder::Openh264(_)) {
            return false;
        }
        true
    }

    /// React to a streak of hardware encode failures: rebuild the session once with the startup
    /// selection, and demote to the software encoder when a fresh session fails the same way.
    /// A session whose encodes keep failing still constructs, so the rebuild only counts as
    /// recovery until an encode succeeds; otherwise the stream would rebuild in a loop and never
    /// demote. Streaming nothing forever is not an option.
    ///
    /// The demote is the last rung: once it lands on the encoder the ladder ends with, the
    /// caller stops re-entering, so a persistently failing software encoder is not rebuilt on
    /// every further streak.
    fn recover_hw(&mut self) {
        self.hw_error_streak = 0;
        if self.hw_rebuilt {
            self.hw_recovery_exhausted = !self.has_rung_below_hw();
            if self.hw_recovery_exhausted {
                eprintln!("[x11] software encoder still failing; nothing below it in this build.");
            } else {
                eprintln!("[x11] HW encoder unrecoverable; demoting to software encoding.");
            }
            // The broken session is released before its replacement is built: these failures
            // are usually device memory pressure, and holding both at once is what would make
            // the replacement fail too.
            self.hw = X11Encoder::None;
            self.hw = software_h264_encoder(&self.settings);
        } else {
            eprintln!("[x11] rebuilding HW encoder after repeated encode errors.");
            self.hw = X11Encoder::None;
            self.hw = select_encoder(&self.settings);
            self.hw_rebuilt = true;
        }
        self.hw_state = StripeState::default();
        self.stripes.clear();
        self.pending_force_idr = true;
    }

    /// Request an on-demand keyframe on the next processed frame.
    pub fn request_idr(&mut self) {
        self.pending_force_idr = true;
    }

    /// Return a human-readable encoder type string for logging.
    pub fn encoder_name(&self) -> &str {
        match &self.hw {
            X11Encoder::Nvenc(_) => "NVENC",
            X11Encoder::Vaapi(_) => "VAAPI",
            X11Encoder::Openh264(_) => "OpenH264",
            X11Encoder::None => "CPU",
        }
    }

    /// The `Colorspace:` field for this pipeline's stream log, describing what its encoder settled
    /// on: VA-API only reaches 4:4:4 when the driver and FFmpeg build carry it, and OpenH264 is
    /// 4:2:0-only whatever was asked for.
    pub fn colorspace_desc(&self) -> &'static str {
        let fullcolor = match &self.hw {
            X11Encoder::Vaapi(enc) => enc.is_fullcolor(),
            X11Encoder::Openh264(_) => false,
            _ => self.settings.video_fullcolor,
        };
        crate::encoders::colorspace_desc(fullcolor, matches!(self.hw, X11Encoder::None))
    }

    /// Adapt the live pipeline to recreated capture surfaces without rebuilding it.
    ///
    /// # Arguments
    ///
    /// * `settings` - New geometry plus current live rates.
    /// * `size_changed` - Whether the capture dimensions changed.
    ///
    /// # Returns
    ///
    /// `true` if the pipeline was successfully adapted in place; `false` when the active encoder
    /// cannot follow (VAAPI, OpenH264 on resize) and the caller must rebuild.
    pub fn reshape(&mut self, settings: &RustCaptureSettings, size_changed: bool) -> bool {
        if !size_changed {
            if let X11Encoder::Nvenc(enc) = &mut self.hw {
                enc.release_pinned_hosts();
            }
            self.settings = settings.clone();
            return true;
        }
        match &mut self.hw {
            X11Encoder::Nvenc(enc) => {
                if let Err(e) = enc.reconfigure_resolution(settings) {
                    eprintln!("[x11] NVENC in-place resize unavailable ({e}); rebuilding");
                    return false;
                }
            }
            X11Encoder::None => {}
            _ => return false,
        }
        self.settings = settings.clone();
        self.stripes.clear();
        self.hw_state = StripeState::default();
        true
    }

    /// Apply a runtime rate-control / framerate change: the CBR target bitrate + VBV (kbps /
    /// kb; ignored unless CBR is active) and the target fps. NVENC and OpenH264 reconfigure their
    /// live session immediately; VAAPI re-opens its codec context to apply the new rate; the x264
    /// software path picks the new values up on the next `process()` (encode_cpu reads the updated
    /// settings and reconfigures each stripe).
    pub fn update_rate(&mut self, bitrate_kbps: i32, vbv_multiplier: f64, fps: f64) {
        self.settings.video_bitrate_kbps = bitrate_kbps;
        self.settings.video_vbv_multiplier = vbv_multiplier;
        if fps > 0.0 {
            self.settings.target_fps = fps;
        }
        match &mut self.hw {
            X11Encoder::Nvenc(enc) => enc.reconfigure_rate(&self.settings),
            X11Encoder::Openh264(enc) => enc.reconfigure_rate(bitrate_kbps, fps),
            X11Encoder::Vaapi(enc) => enc.reconfigure_rate(&self.settings),
            _ => {}
        }
    }

    /// Apply live per-frame tunables (quality, paint-over, streaming mode, keyframe
    /// cadence); every encoder re-reads them from the settings on the next process().
    pub fn update_tunables(&mut self, t: &crate::LiveTunables) {
        t.apply_to(&mut self.settings);
    }

    /// Encode one host-ARGB frame and return the encoded stripes.
    ///
    /// # Arguments
    ///
    /// * `argb` - Packed BGRA pixel buffer (B,G,R,A byte order, `stride` bytes per row).
    /// * `stride` - Bytes per row (must equal `width * 4` for the software path).
    ///
    /// # Returns
    ///
    /// Vec of [`EncodedStripe`] — empty when nothing changed.
    pub fn process(&mut self, argb: &[u8], stride: usize) -> Vec<EncodedStripe> {
        let width = self.settings.width;
        let height = self.settings.height;
        let requested = self.pending_force_idr;
        let threshold = self.settings.damage_block_threshold;
        let duration = self.settings.damage_block_duration as i32;

        let out = if !matches!(self.hw, X11Encoder::None) {
            let is_dirty = if self.settings.video_streaming_mode {
                false
            } else {
                self.hw_state.content_dirty(argb, threshold, duration)
            };
            let d = decide_hw_fullframe(
                &mut self.hw_state,
                &self.settings,
                self.frame_counter,
                is_dirty,
                false,
                requested,
            );
            if d.send {
                let fc = self.frame_counter as u64;
                let force_idr = d.force_idr;
                let res = match &mut self.hw {
                    X11Encoder::Nvenc(enc) => {
                        enc.encode_cpu_argb(argb, stride, fc, d.target_qp, force_idr)
                    }
                    X11Encoder::Vaapi(enc) => {
                        enc.encode_host_argb(argb, stride, fc, d.target_qp, force_idr)
                    }
                    X11Encoder::Openh264(enc) => {
                        enc.update_qp(d.target_qp);
                        enc.encode_host_argb(argb, stride, fc, force_idr, false)
                    }
                    X11Encoder::None => unreachable!(),
                };
                match res {
                    Ok(data) if !data.is_empty() => {
                        self.hw_error_streak = 0;
                        self.hw_rebuilt = false;
                        self.hw_recovery_exhausted = false;
                        vec![EncodedStripe {
                            data: Arc::new(data),
                            data_type: 2,
                            stripe_y_start: 0,
                            stripe_height: height,
                            frame_id: self.frame_counter as i32,
                        }]
                    }
                    Ok(_) => {
                        self.hw_error_streak = 0;
                        self.hw_rebuilt = false;
                        self.hw_recovery_exhausted = false;
                        Vec::new()
                    }
                    Err(e) => {
                        // One line per recovery window: a session failing at frame rate would
                        // otherwise write a line per frame for the life of the capture.
                        if self.hw_error_streak % crate::HW_ERROR_RECOVERY_THRESHOLD == 0 {
                            eprintln!("[x11] HW encode error: {e}");
                        }
                        self.hw_error_streak = self.hw_error_streak.saturating_add(1);
                        if self.hw_error_streak >= crate::HW_ERROR_RECOVERY_THRESHOLD
                            && !self.hw_recovery_exhausted
                        {
                            self.recover_hw();
                        }
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            }
        } else {
            debug_assert_eq!(
                stride,
                width as usize * 4,
                "software encode path assumes tightly-packed rows (stride == width*4)"
            );
            let force_idr_all = requested
                || (self.settings.output_mode == 1
                    && periodic_idr_due(&self.settings, self.frame_counter));
            encode_cpu(
                &mut self.stripes,
                &mut self.stripes_carrying,
                argb,
                width,
                height,
                &[],
                &self.settings,
                self.frame_counter,
                false,
                true,
                force_idr_all,
            )
        };

        // An unserved request stays armed: on an infinite GOP an IDR lost to an encode
        // error or skip would never self-heal, leaving a joining consumer with an
        // undecodable stream. A rebuilt or demoted encoder arms one the same way, which
        // is what the second read of the flag picks up.
        self.pending_force_idr = (requested || self.pending_force_idr) && out.is_empty();
        self.frame_counter = self.frame_counter.wrapping_add(1);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Software JPEG path emits on change and stays silent while static: the first frame
    /// sends every stripe (all dirty vs init), an identical static frame sends nothing, and a
    /// frame with changed top rows re-sends the dirty stripes. `use_cpu` forces the software path
    /// and paint-over is disabled to keep the static-frame assertions clean.
    #[test]
    fn x11_software_emits_on_change_and_stays_quiet_when_static() {
        let s = RustCaptureSettings {
            width: 128,
            height: 128,
            output_mode: 0,
            use_cpu: true,
            jpeg_quality: 60,
            use_paint_over_quality: false,
            ..Default::default()
        };
        let mut p = X11Pipeline::new(s);
        let stride = 128 * 4;
        let frame_a = vec![10u8; stride * 128];
        let mut frame_b = frame_a.clone();
        for px in frame_b.iter_mut().take(stride * 40) {
            *px = 200;
        }
        let n1 = p.process(&frame_a, stride).len();
        let n2 = p.process(&frame_a, stride).len();
        let n3 = p.process(&frame_b, stride).len();
        assert!(n1 > 0, "first frame should emit (all stripes dirty vs init)");
        assert_eq!(n2, 0, "identical static frame should emit nothing");
        assert!(n3 > 0, "changed frame should emit dirty stripes");
    }

    /// Software x264, paint-over off, non-streaming: a requested IDR on a static screen is
    /// followed by a short recovery burst so rate control can refine the keyframe, instead of the
    /// stream going silent and stranding an unrefined keyframe. The stream goes quiet again once
    /// the burst ends.
    #[test]
    fn x11_software_h264_streams_recovery_burst_after_requested_idr() {
        let s = RustCaptureSettings {
            width: 128,
            height: 128,
            output_mode: 1,
            use_cpu: true,
            video_crf: 25,
            video_paintover_burst_frames: 5,
            use_paint_over_quality: false,
            video_streaming_mode: false,
            target_fps: 60.0,
            ..Default::default()
        };
        let mut p = X11Pipeline::new(s);
        let stride = 128 * 4;
        let frame = vec![10u8; stride * 128];
        assert!(!p.process(&frame, stride).is_empty(), "first frame emits");
        for _ in 0..4 {
            let _ = p.process(&frame, stride);
        }
        assert!(p.process(&frame, stride).is_empty(), "static screen is quiet before the request");
        p.request_idr();
        assert!(!p.process(&frame, stride).is_empty(), "requested IDR emits on a static screen");
        for i in 0..5 {
            assert!(!p.process(&frame, stride).is_empty(), "recovery burst frame {i} streams while static");
        }
        assert!(p.process(&frame, stride).is_empty(), "stream goes quiet again after the recovery burst");
    }

    /// The software path is the one encoder that always carries a 4:4:4 request, and carries it at
    /// full range, so its log line says so; without the request it reports 4:2:0. This is the same
    /// string the Wayland log builds from the same shared helper, so an identical session reads
    /// identically on both backends.
    #[test]
    fn x11_colorspace_desc_reports_what_the_software_encoder_carries() {
        for (fullcolor, expected) in
            [(true, "I444 (Full Range)"), (false, "I420 (Limited Range)")]
        {
            let p = X11Pipeline::new(RustCaptureSettings {
                width: 64,
                height: 64,
                output_mode: 1,
                use_cpu: true,
                video_fullcolor: fullcolor,
                ..Default::default()
            });
            assert_eq!(p.encoder_name(), "CPU");
            assert_eq!(p.colorspace_desc(), expected);
            assert_eq!(
                p.colorspace_desc(),
                crate::encoders::colorspace_desc(fullcolor, true),
                "X11 and Wayland must describe the same session identically"
            );
        }
    }

    fn settings() -> RustCaptureSettings {
        RustCaptureSettings {
            video_crf: 25,
            video_paintover_crf: 18,
            paint_over_trigger_frames: 3,
            use_paint_over_quality: true,
            video_paintover_burst_frames: 5,
            video_streaming_mode: false,
            target_fps: 60.0,
            ..Default::default()
        }
    }

    /// With no motion and no request, nothing is emitted at any frame-counter position —
    /// the GOP is infinite, so there is no scheduled IDR to break the silence.
    #[test]
    fn static_frames_stay_silent_without_request() {
        let mut s = settings();
        s.use_paint_over_quality = false;
        let mut st = StripeState::default();
        for fc in [0u16, 1, 120, 240] {
            let d = decide_hw_fullframe(&mut st, &s, fc, false, false, false);
            assert!(!d.send && !d.force_idr, "frame {fc} should stay idle");
        }
    }

    /// A requested IDR on a static screen (client resume / join) emits a base-QP keyframe
    /// and arms a recovery burst; subsequent static frames stream burst frames at the paint QP
    /// with no new keyframe, and real motion aborts the burst and reverts to the base QP.
    #[test]
    fn requested_idr_forces_send_even_on_static() {
        let s = settings();
        let mut st = StripeState::default();
        let d = decide_hw_fullframe(&mut st, &s, 5, false, false, true);
        assert!(d.send && d.force_idr);
        assert_eq!(d.target_qp, 25);
        assert_eq!(st.h264_burst_frames_remaining, 5);
        let d = decide_hw_fullframe(&mut st, &s, 6, false, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 18);
        assert_eq!(st.h264_burst_frames_remaining, 4);
        let d = decide_hw_fullframe(&mut st, &s, 7, true, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 25);
        assert_eq!(st.h264_burst_frames_remaining, 0);
    }

    /// With paint-over off, a forced keyframe still needs following frames so CBR rate
    /// control can refine the static image (streaming mode would mask this by always sending);
    /// the recovery burst supplies them at the base QP.
    #[test]
    fn requested_idr_recovers_even_without_paint_over() {
        let mut s = settings();
        s.use_paint_over_quality = false;
        let mut st = StripeState::default();
        let d = decide_hw_fullframe(&mut st, &s, 5, false, false, true);
        assert!(d.send && d.force_idr);
        assert_eq!(st.h264_burst_frames_remaining, 5, "recovery burst armed without paint-over");
        let d = decide_hw_fullframe(&mut st, &s, 6, false, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 25);
    }

    #[test]
    fn configured_interval_restores_scheduled_keyframes() {
        let mut s = settings();
        s.keyframe_interval_s = 2.0;
        assert!(periodic_idr_due(&s, 0));
        assert!(!periodic_idr_due(&s, 1));
        assert!(periodic_idr_due(&s, 120));
        let mut st = StripeState::default();
        let d = decide_hw_fullframe(&mut st, &s, 120, false, false, false);
        assert!(d.send && d.force_idr, "interval keyframe fires on a static screen");
        s.keyframe_interval_s = 0.0;
        assert!(!periodic_idr_due(&s, 0) && !periodic_idr_due(&s, 120));
    }

    /// After `paint_over_trigger_frames` idle frames, paint-over fires as a refining
    /// P-frame at the paint QP (no IDR spike) and arms a burst; the next frame continues the burst
    /// at the paint QP, still without a forced IDR.
    #[test]
    fn paint_over_fires_after_trigger_then_bursts() {
        let s = settings();
        let mut st = StripeState::default();
        for fc in 1..=2 {
            let d = decide_hw_fullframe(&mut st, &s, fc, false, false, false);
            assert!(!d.send, "frame {fc} should stay idle");
        }
        let d = decide_hw_fullframe(&mut st, &s, 3, false, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 18);
        assert_eq!(st.h264_burst_frames_remaining, 4);
        let d = decide_hw_fullframe(&mut st, &s, 4, false, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 18);
        assert_eq!(st.h264_burst_frames_remaining, 3);
    }

    #[test]
    fn motion_resets_paintover_and_uses_normal_qp() {
        let s = settings();
        let mut st = StripeState {
            paint_over_sent: true,
            h264_burst_frames_remaining: 2,
            no_motion_frame_count: 9,
            ..Default::default()
        };
        let d = decide_hw_fullframe(&mut st, &s, 7, true, false, false);
        assert!(d.send);
        assert_eq!(d.target_qp, 25);
        assert!(!st.paint_over_sent);
        assert_eq!(st.h264_burst_frames_remaining, 0);
        assert_eq!(st.no_motion_frame_count, 0);
    }
}

#[cfg(test)]
mod vbv_tests {
    /// VBV sizing policy: an infinite GOP uses 1.5 frames of headroom, scheduled keyframes
    /// relax to 3 frames, and an explicit multiplier overrides both and rescales with bitrate.
    #[test]
    fn vbv_policy() {
        use crate::encoders::vbv_bits;
        let frame = 4_000_000f64 / 60.0;
        assert_eq!(vbv_bits(4_000_000, 60.0, 0.0, 0.0), (frame * 1.5).round() as u32);
        assert_eq!(vbv_bits(4_000_000, 60.0, 2.0, 0.0), (frame * 3.0).round() as u32);
        assert_eq!(vbv_bits(4_000_000, 60.0, 2.0, 1.0), frame.round() as u32);
        assert_eq!(vbv_bits(8_000_000, 60.0, 0.0, 1.0), (2.0 * frame).round() as u32);
    }
}
