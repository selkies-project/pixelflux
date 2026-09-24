/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Software VP8 and VP9 through libvpx, with every frame's references under the session's
//! control.
//!
//! One session type serves both codecs. libvpx runs at the lowest latency it offers in real
//! time -- one frame in, one packet out, no lag, no alternate-reference lookahead -- with screen
//! tuning on VP9, and its rate control is either constant-rate at the session's bitrate and VBV
//! or a quantizer pinned between an equal minimum and maximum, which is how a quality change
//! reaches a running session without a key frame. A VP9 session runs in libvpx's flexible mode,
//! where the codec's eight buffers are addressed by slot, so a frame a client lost leaves the
//! predictions the way a decoded picture buffer allows; a VP8 session steers its three buffers
//! with the per-frame reference and refresh flags. What a lost frame would leave behind besides
//! its picture is kept out too: VP8's entropy probabilities persist across frames unless a
//! frame declines to update them, so no frame does, and a VP9 decoder takes the previous
//! frame's motion vectors and probability contexts whatever the references say, which only the
//! codec's error-resilient mode switches off.

use std::ffi::{c_int, c_long, c_void, CStr};
use std::ptr;

use codec_sys::vpx::*;

use super::codec::{frame_type_from_key, push_video_header, vp8_is_key, vp9_is_key, vpx_level, Codec, VIDEO_HEADER_LEN};
use super::reference::{Reference, ReferenceSlots, ReferenceWindow, SlotPlan, SlotRefresh, REFERENCE_FRAMES};
use super::session::{encode_threads, Pending, Planes, Quality, RateSettings};
use crate::RustCaptureSettings;

/// The rate target, in kbit/s, a pinned-quantizer session names and never reaches.
const RATE_CEILING_KBPS: u32 = 100_000;

/// The temporal layers a VP9 session declares. Every frame is coded in the base layer; the
/// second exists because libvpx sets up and re-targets its per-layer rate control, which the
/// flexible reference mode runs on, only for a session with more than one layer.
const VP9_LAYERS: u32 = 2;
/// The libvpx quantizer levels a session leaves to the library's own bounds.
const VP8_DEFAULT_MIN_LEVEL: u32 = 4;

/// The references a session keeps: VP9's eight slots as a sliding window, VP8's three buffers.
enum References {
    Window(ReferenceWindow),
    Slots(ReferenceSlots),
}

/// One libvpx session for one capture.
pub struct VpxEncoder {
    codec: Codec,
    ctx: vpx_codec_ctx_t,
    cfg: vpx_codec_enc_cfg_t,
    planes: Planes,
    threads: u32,
    quality: Quality,
    rate: RateSettings,
    omit_headers: bool,
    references: References,
    last_reference: Reference,
    pending: Pending,
    /// The slot plan of the frame being encoded, recorded once its packet returns (VP8).
    plan: SlotPlan,
}

unsafe impl Send for VpxEncoder {}

impl Drop for VpxEncoder {
    fn drop(&mut self) {
        unsafe { vpx_codec_destroy(&mut self.ctx) };
    }
}

fn error(ctx: &vpx_codec_ctx_t, what: &str) -> String {
    unsafe {
        let detail = vpx_codec_error_detail(ctx);
        let text = CStr::from_ptr(vpx_codec_err_to_string(ctx.err)).to_string_lossy();
        if detail.is_null() {
            format!("{what}: {text}")
        } else {
            format!("{what}: {text} ({})", CStr::from_ptr(detail).to_string_lossy())
        }
    }
}

impl VpxEncoder {
    /// Open a VP8 or VP9 session on host frames in the byte order `rgba` names.
    pub fn new(settings: &RustCaptureSettings, codec: Codec, rgba: bool) -> Result<Self, String> {
        if !matches!(codec, Codec::Vp8 | Codec::Vp9) {
            return Err(format!("libvpx encodes no {}", codec.display()));
        }
        let _ = rgba;
        let fullcolor = codec == Codec::Vp9 && settings.video_fullcolor;
        let threads = encode_threads() as u32;
        let rate = RateSettings::new(settings);
        let quality = Quality::new(codec.quantizer(settings.video_crf));
        let iface = unsafe { if codec == Codec::Vp8 { vpx_codec_vp8_cx() } else { vpx_codec_vp9_cx() } };
        let mut cfg: vpx_codec_enc_cfg_t = unsafe { std::mem::zeroed() };
        if unsafe { vpx_codec_enc_config_default(iface, &mut cfg, 0) } != VPX_CODEC_OK {
            return Err("libvpx refused its default configuration".into());
        }
        cfg.g_w = settings.width.max(1) as u32;
        cfg.g_h = settings.height.max(1) as u32;
        cfg.g_timebase = vpx_rational { num: 1, den: rate.fps };
        cfg.g_threads = threads.min(64);
        cfg.g_lag_in_frames = 0;
        cfg.g_pass = VPX_RC_ONE_PASS;
        cfg.g_error_resilient = if codec == Codec::Vp9 { VPX_ERROR_RESILIENT_DEFAULT } else { 0 };
        cfg.g_profile = if fullcolor { 1 } else { 0 };
        cfg.g_bit_depth = 8;
        cfg.g_input_bit_depth = 8;
        cfg.rc_dropframe_thresh = 0;
        cfg.rc_2pass_vbr_bias_pct = 50;
        cfg.rc_2pass_vbr_minsection_pct = 100;
        cfg.rc_2pass_vbr_maxsection_pct = 100;
        cfg.kf_max_dist = c_int::MAX as u32;
        if codec == Codec::Vp9 {
            cfg.ss_number_layers = 1;
            cfg.ts_number_layers = VP9_LAYERS;
            cfg.ts_rate_decimator[..VP9_LAYERS as usize].fill(1);
            cfg.ts_periodicity = 1;
            cfg.temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_BYPASS as c_int;
        }
        let mut me = Self {
            codec,
            ctx: unsafe { std::mem::zeroed() },
            cfg,
            planes: Planes::new(settings.width.max(1) as usize, settings.height.max(1) as usize, fullcolor),
            threads,
            quality,
            rate,
            omit_headers: settings.omit_stripe_headers,
            references: if codec == Codec::Vp9 {
                References::Window(ReferenceWindow::new(REFERENCE_FRAMES))
            } else {
                References::Slots(ReferenceSlots::new())
            },
            last_reference: Reference::Untracked,
            pending: Pending::default(),
            plan: SlotPlan::KEY,
        };
        let q = me.quality.current;
        me.program_rate(rate, q);
        let res = unsafe { vpx_codec_enc_init_ver(&mut me.ctx, iface, &me.cfg, 0, VPX_ENCODER_ABI_VERSION as c_int) };
        if res != VPX_CODEC_OK {
            return Err(error(&me.ctx, "libvpx refused the session"));
        }
        me.control(VP8E_SET_CPUUSED, if codec == Codec::Vp9 { 8 } else { 16 })?;
        me.control(VP8E_SET_ENABLEAUTOALTREF, 0)?;
        me.control(VP8E_SET_STATIC_THRESHOLD, 0)?;
        me.control(VP8E_SET_MAX_INTRA_BITRATE_PCT, 0)?;
        if codec == Codec::Vp8 {
            me.control(VP8E_SET_NOISE_SENSITIVITY, 0)?;
            me.control(VP8E_SET_TOKEN_PARTITIONS, 0)?;
        } else {
            // Column threading is per tile and VP9's narrowest tile is 256 pixels, so the
            // width sets how many columns the encode can spread across.
            let tile_columns = (settings.width / 256).max(1).ilog2().min(6);
            me.control(VP9E_SET_TILE_COLUMNS, tile_columns as c_int)?;
            me.control(VP9E_SET_FRAME_PARALLEL_DECODING, 0)?;
            me.control(VP9E_SET_COLOR_SPACE, VPX_CS_BT_709 as c_int)?;
            me.control(VP9E_SET_COLOR_RANGE, VPX_CR_STUDIO_RANGE as c_int)?;
            me.control(VP9E_SET_TARGET_LEVEL, 255)?;
            me.control(VP9E_SET_ROW_MT, 1)?;
            me.control(VP9E_SET_TUNE_CONTENT, VP9E_CONTENT_SCREEN as c_int)?;
            me.control(VP9E_SET_SVC, 1)?;
            me.program_layer()?;
        }
        Ok(me)
    }

    fn control(&mut self, id: vp8e_enc_control_id, value: c_int) -> Result<(), String> {
        if unsafe { vpx_codec_control_(&mut self.ctx, id as c_int, value) } != VPX_CODEC_OK {
            return Err(error(&self.ctx, &format!("libvpx refused control {id}")));
        }
        Ok(())
    }

    /// The layers of a VP9 session, whose quantizer bounds libvpx reads from here rather
    /// than from the configuration once the flexible mode is on.
    fn program_layer(&mut self) -> Result<(), String> {
        let mut layer: vpx_svc_parameters = unsafe { std::mem::zeroed() };
        for i in 0..VP9_LAYERS as usize {
            layer.max_quantizers[i] = self.cfg.rc_max_quantizer as c_int;
            layer.min_quantizers[i] = self.cfg.rc_min_quantizer as c_int;
            layer.speed_per_layer[i] = 8;
        }
        layer.scaling_factor_num[0] = 1;
        layer.scaling_factor_den[0] = 1;
        layer.temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_BYPASS as c_int;
        let res = unsafe {
            vpx_codec_control_(&mut self.ctx, VP9E_SET_SVC_PARAMETERS as c_int, &mut layer as *mut vpx_svc_parameters as *mut c_void)
        };
        if res != VPX_CODEC_OK {
            return Err(error(&self.ctx, "libvpx refused the layer parameters"));
        }
        Ok(())
    }

    /// Write `rate` and the quantizer `q` (the codec's own domain) into the configuration:
    /// a constant rate holds the target and VBV with the quantizer bounds the settings name,
    /// a pinned quantizer names a rate it never reaches with both bounds at the quantizer.
    fn program_rate(&mut self, rate: RateSettings, q: u32) {
        let cfg = &mut self.cfg;
        cfg.g_timebase = vpx_rational { num: 1, den: rate.fps };
        cfg.rc_end_usage = VPX_CBR;
        if rate.cbr {
            let kbps = (rate.bps() / 1000).clamp(1, u32::MAX as u64) as u32;
            cfg.rc_target_bitrate = kbps;
            let (lo, hi) = (self.codec.quantizer_bound(rate.min_qp), self.codec.quantizer_bound(rate.max_qp));
            cfg.rc_min_quantizer = if lo > 0 { vpx_level(self.codec, lo) } else if self.codec == Codec::Vp8 { VP8_DEFAULT_MIN_LEVEL } else { 0 };
            cfg.rc_max_quantizer = if hi > 0 { vpx_level(self.codec, hi) } else { 63 };
            let ms = (rate.vbv() as u64 * 1000 / rate.bps().max(1)) as u32;
            cfg.rc_buf_sz = ms;
            cfg.rc_buf_initial_sz = ms;
            cfg.rc_buf_optimal_sz = ms * 5 / 6;
        } else {
            cfg.rc_target_bitrate = RATE_CEILING_KBPS;
            cfg.rc_min_quantizer = vpx_level(self.codec, q);
            cfg.rc_max_quantizer = vpx_level(self.codec, q);
        }
        cfg.ss_target_bitrate[0] = cfg.rc_target_bitrate;
        cfg.ts_target_bitrate[..VP9_LAYERS as usize].fill(cfg.rc_target_bitrate);
        cfg.layer_target_bitrate[..VP9_LAYERS as usize].fill(cfg.rc_target_bitrate);
    }

    /// Apply the configuration to the running session.
    fn reconfigure(&mut self) -> Result<(), String> {
        if unsafe { vpx_codec_enc_config_set(&mut self.ctx, &self.cfg) } != VPX_CODEC_OK {
            return Err(error(&self.ctx, "libvpx refused the new configuration"));
        }
        if self.codec == Codec::Vp9 {
            self.program_layer()?;
        }
        Ok(())
    }

    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// The library behind the session, as the logs name it.
    pub fn library(&self) -> &'static str {
        "libvpx"
    }

    /// Whether the session carries 4:4:4 (VP9 profile 1).
    pub fn is_fullcolor(&self) -> bool {
        self.planes.i444
    }

    /// VP9 keeps the limited range of its 4:2:0 in profile 1, so the decoder hint a client
    /// sends for it stays true; VP8 has no other.
    pub fn is_full_range(&self) -> bool {
        false
    }

    /// The frame the last encoded frame predicted from.
    pub fn last_reference(&self) -> Reference {
        self.last_reference
    }

    /// Leave frame `frame_id` and every frame after it out of the predictions; the next frame
    /// predicts from an earlier buffer, or is a key frame when none is left.
    pub fn invalidate_reference(&mut self, frame_id: u16) -> bool {
        match &mut self.references {
            References::Window(w) => w.invalidate(frame_id),
            References::Slots(s) => s.invalidate(frame_id),
        };
        true
    }

    /// Re-program the rate control when a rate or frame-rate setting changed, live.
    pub fn reconfigure_rate(&mut self, settings: &RustCaptureSettings) -> Result<(), String> {
        let Some(rate) = self.rate.changed(settings) else { return Ok(()) };
        self.rate = rate;
        self.program_rate(rate, self.quality.current);
        self.reconfigure()
    }

    /// Encode one packed host frame at the quality index `crf`, as a key frame when
    /// `force_idr` or when no reference is left to predict from.
    pub fn encode_host(
        &mut self,
        pixels: &[u8],
        stride: usize,
        rgba: bool,
        frame_number: u64,
        crf: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        if !self.rate.cbr && let Some(q) = self.quality.update(self.codec.quantizer(crf as i32)) {
            self.program_rate(self.rate, q);
            self.reconfigure()?;
        }
        let bt601 = self.codec == Codec::Vp8;
        self.planes.convert(pixels, stride, rgba, false, bt601, self.threads as usize)?;

        let key = force_idr
            || match &self.references {
                References::Window(w) => !w.has_reference(),
                References::Slots(s) => !s.has_reference(),
            };
        let pts = match &self.references {
            References::Window(w) => w.next_pts(),
            References::Slots(s) => s.next_pts(),
        };
        let mut flags: c_long = if key { VPX_EFLAG_FORCE_KF as c_long } else { 0 };
        match &self.references {
            References::Window(w) => {
                let mut refs: vpx_svc_ref_frame_config = unsafe { std::mem::zeroed() };
                if key {
                    refs.update_buffer_slot[0] = 0xff;
                } else {
                    let (_, ref_pts) = w.newest_valid().expect("a reference or a key frame");
                    let slot = (ref_pts % REFERENCE_FRAMES as u64) as c_int;
                    refs.lst_fb_idx[0] = slot;
                    refs.gld_fb_idx[0] = slot;
                    refs.alt_fb_idx[0] = slot;
                    refs.reference_last[0] = 1;
                    refs.update_buffer_slot[0] = 1 << (pts % REFERENCE_FRAMES as u64);
                }
                let res = unsafe {
                    vpx_codec_control_(
                        &mut self.ctx,
                        VP9E_SET_SVC_REF_FRAME_CONFIG as c_int,
                        &mut refs as *mut vpx_svc_ref_frame_config as *mut c_void,
                    )
                };
                if res != VPX_CODEC_OK {
                    return Err(error(&self.ctx, "libvpx refused the reference configuration"));
                }
            }
            References::Slots(s) => {
                self.plan = s.plan(key);
                flags |= VP8_EFLAG_NO_UPD_ENTROPY as c_long;
                if !key {
                    for (buffer, no_ref, no_upd, force) in [
                        (SlotRefresh::LAST, VP8_EFLAG_NO_REF_LAST, VP8_EFLAG_NO_UPD_LAST, 0),
                        (SlotRefresh::GOLDEN, VP8_EFLAG_NO_REF_GF, VP8_EFLAG_NO_UPD_GF, VP8_EFLAG_FORCE_GF),
                        (SlotRefresh::ALTREF, VP8_EFLAG_NO_REF_ARF, VP8_EFLAG_NO_UPD_ARF, VP8_EFLAG_FORCE_ARF),
                    ] {
                        if self.plan.predict_from != buffer {
                            flags |= no_ref as c_long;
                        }
                        if self.plan.refresh.refreshes(buffer) {
                            flags |= force as c_long;
                        } else {
                            flags |= no_upd as c_long;
                        }
                    }
                }
            }
        }

        let mut img: vpx_image = unsafe { std::mem::zeroed() };
        let fmt = if self.planes.i444 { VPX_IMG_FMT_I444 } else { VPX_IMG_FMT_I420 };
        unsafe {
            vpx_img_wrap(&mut img, fmt, self.planes.width as u32, self.planes.height as u32, 1, self.planes.y.as_mut_ptr());
        }
        img.planes[0] = self.planes.y.as_mut_ptr();
        img.planes[1] = self.planes.u.as_mut_ptr();
        img.planes[2] = self.planes.v.as_mut_ptr();
        img.stride[0] = self.planes.width as c_int;
        img.stride[1] = self.planes.chroma_width() as c_int;
        img.stride[2] = self.planes.chroma_width() as c_int;
        img.range = VPX_CR_STUDIO_RANGE;
        self.pending.push(pts, frame_number as u16);
        let res = unsafe { vpx_codec_encode(&mut self.ctx, &img, pts as i64, 1, flags, VPX_DL_REALTIME as u64) };
        if res != VPX_CODEC_OK {
            self.pending.undo();
            return Err(error(&self.ctx, "libvpx refused the frame"));
        }

        let mut output = Vec::new();
        let mut iter: vpx_codec_iter_t = ptr::null_mut();
        loop {
            let pkt = unsafe { vpx_codec_get_cx_data(&mut self.ctx, &mut iter) };
            if pkt.is_null() {
                break;
            }
            let pkt = unsafe { &*pkt };
            if pkt.kind != VPX_CODEC_CX_FRAME_PKT {
                continue;
            }
            let frame = unsafe { pkt.data.frame };
            let bytes = unsafe { std::slice::from_raw_parts(frame.buf as *const u8, frame.sz) };
            let is_key = if self.codec == Codec::Vp8 { vp8_is_key(bytes) } else { vp9_is_key(bytes) };
            let id = self.pending.take(frame.pts as u64).unwrap_or(frame_number as u16);
            self.last_reference = match &mut self.references {
                References::Window(w) => w.record(id, is_key),
                References::Slots(s) => {
                    let plan = if is_key { s.plan(true) } else { self.plan };
                    s.record(id, plan)
                }
            };
            if !self.omit_headers {
                output.reserve(VIDEO_HEADER_LEN + bytes.len());
                push_video_header(
                    &mut output,
                    self.codec,
                    frame_type_from_key(is_key),
                    id,
                    0,
                    self.planes.width as u16,
                    self.planes.height as u16,
                    self.last_reference,
                );
            }
            output.extend_from_slice(bytes);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::codec::{parse_video_type, FRAME_DELTA, FRAME_KEY};
    use crate::webcam::decode::{Decoder as _, VideoDecoder};

    const W: usize = 320;
    const H: usize = 240;

    fn settings(codec: Codec) -> RustCaptureSettings {
        RustCaptureSettings { width: W as i32, height: H as i32, target_fps: 30.0, codec, video_crf: 25, use_cpu: true, ..Default::default() }
    }

    /// A frame whose luma spells its index in blocks, so a decoded picture names the frame it is.
    fn frame(t: usize) -> Vec<u8> {
        let mut f = vec![0u8; W * H * 4];
        let (bx, by) = ((t * 9) % (W - 40), (t * 5) % (H - 30));
        for y in 0..H {
            for x in 0..W {
                let i = (y * W + x) * 4;
                let g = ((x * 255) / W) as u8;
                let cell = (x / 8 + y / 12) % 3 == 0 && x % 8 < 6 && y % 12 < 9;
                let (b, gr, r) = if x >= bx && x < bx + 40 && y >= by && y < by + 30 {
                    (40, 220, 250)
                } else if cell {
                    (30, 30, 30)
                } else {
                    (g, 200 - g / 2, 120)
                };
                f[i..i + 4].copy_from_slice(&[b, gr, r, 255]);
            }
        }
        f
    }

    fn luma_distance(a: &crate::webcam::convert::I420View<'_>, b: &crate::webcam::convert::I420View<'_>) -> f64 {
        a.y.chunks(a.y_stride)
            .zip(b.y.chunks(b.y_stride))
            .take(a.height)
            .flat_map(|(ra, rb)| ra[..a.width].iter().zip(&rb[..a.width]).map(|(&x, &y)| (x as f64 - y as f64).abs()))
            .sum::<f64>()
            / (a.width * a.height) as f64
    }

    /// A frame a client lost is left out of the predictions: the next frame names the newest
    /// frame before it, and a decoder that never saw the lost frames decodes it exactly as one
    /// that saw everything. VP9 reaches back through its eight slots, VP8 through its anchors.
    #[test]
    fn a_lost_frame_is_predicted_past() {
        for codec in [Codec::Vp8, Codec::Vp9] {
            let s = settings(codec);
            let mut enc = VpxEncoder::new(&s, codec, false).expect("session");
            let mut frames = Vec::new();
            for t in 0..8u64 {
                let out = enc.encode_host(&frame(t as usize), W * 4, false, t, 25, t == 0).expect("encode");
                assert_eq!(parse_video_type(out[1]).map(|(_, k)| k), Some(if t == 0 { FRAME_KEY } else { FRAME_DELTA }), "{codec:?} {t}");
                assert_eq!(enc.last_reference(), if t == 0 { Reference::None } else { Reference::Frame(t as u16 - 1) }, "{codec:?} {t}");
                frames.push(out);
            }
            // Frame 5 is reported lost once 6 and 7 have gone out: VP9 predicts from 4 through
            // its slots, VP8 from the altref anchor, the key frame, since its golden holds 6.
            assert!(enc.invalidate_reference(5));
            let out = enc.encode_host(&frame(8), W * 4, false, 8, 25, false).expect("encode");
            let anchor = if codec == Codec::Vp9 { 4 } else { 0 };
            assert_eq!(enc.last_reference(), Reference::Frame(anchor), "{codec:?}");
            assert_eq!(parse_video_type(out[1]).map(|(_, k)| k), Some(FRAME_DELTA), "{codec:?}");
            frames.push(out);
            frames.push(enc.encode_host(&frame(9), W * 4, false, 9, 25, false).expect("encode"));
            assert_eq!(enc.last_reference(), Reference::Frame(8), "{codec:?}");
            let (mut whole, mut lossy) = (VideoDecoder::new(codec).unwrap(), VideoDecoder::new(codec).unwrap());
            for (i, f) in frames.iter().enumerate() {
                assert!(whole.decode(&f[VIDEO_HEADER_LEN..]).expect("decode"), "{codec:?} frame {i}");
                if !(5..8).contains(&i) {
                    assert!(lossy.decode(&f[VIDEO_HEADER_LEN..]).expect("decode without 5-7"), "{codec:?} frame {i}");
                }
            }
            let apart = luma_distance(&whole.frame().unwrap(), &lossy.frame().unwrap());
            assert!(apart == 0.0, "{codec:?}: the decoder that lost frames 5-7 shows frame 9 {apart:.2} off the one that saw them");
        }
    }

    /// A loss no buffer reaches past costs a key frame, and the count restarts there.
    #[test]
    fn a_loss_past_the_window_costs_a_key_frame() {
        for codec in [Codec::Vp8, Codec::Vp9] {
            let s = settings(codec);
            let mut enc = VpxEncoder::new(&s, codec, false).expect("session");
            for t in 0..20u64 {
                enc.encode_host(&frame(t as usize), W * 4, false, t, 25, t == 0).expect("encode");
            }
            assert!(enc.invalidate_reference(1));
            let out = enc.encode_host(&frame(20), W * 4, false, 20, 25, false).expect("encode");
            assert_eq!(parse_video_type(out[1]).map(|(_, k)| k), Some(FRAME_KEY), "{codec:?}");
            assert_eq!(enc.last_reference(), Reference::None);
            enc.encode_host(&frame(21), W * 4, false, 21, 25, false).expect("encode");
            assert_eq!(enc.last_reference(), Reference::Frame(20), "{codec:?}");
        }
    }

    /// A quality change reaches a running session without a key frame, and a rate change too.
    #[test]
    fn quality_and_rate_move_without_a_key_frame() {
        for codec in [Codec::Vp8, Codec::Vp9] {
            let mut s = settings(codec);
            s.video_crf = 40;
            let mut enc = VpxEncoder::new(&s, codec, false).expect("session");
            let coarse = enc.encode_host(&frame(0), W * 4, false, 0, 40, true).unwrap();
            let fine = enc.encode_host(&frame(1), W * 4, false, 1, 15, false).unwrap();
            assert_eq!(enc.quality.current, codec.quantizer(15));
            assert_eq!(parse_video_type(fine[1]).map(|(_, k)| k), Some(FRAME_DELTA), "{codec:?}");
            let again = enc.encode_host(&frame(2), W * 4, false, 2, 15, false).unwrap();
            assert!(again.len() + fine.len() > coarse.len() / 2, "{codec:?}: the finer quantizer spends more");
            s.target_fps = 15.0;
            enc.reconfigure_rate(&s).expect("rate");
            let after = enc.encode_host(&frame(3), W * 4, false, 3, 15, false).unwrap();
            assert_eq!(parse_video_type(after[1]).map(|(_, k)| k), Some(FRAME_DELTA), "{codec:?}");
            let mut dec = VideoDecoder::new(codec).unwrap();
            for f in [&coarse, &fine, &again, &after] {
                assert!(dec.decode(&f[VIDEO_HEADER_LEN..]).unwrap(), "{codec:?}");
            }
        }
    }
}
