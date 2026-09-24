/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Software AV1 through SVT-AV1 at preset 11 in its real-time mode, one frame in and one
//! packet out (two frames deep before 2.3), with rate control either constant-rate at the
//! session's bitrate and VBV or a constant quantizer. The library takes no live change of
//! either and exposes no reference control, so a quality, rate, or frame-rate change re-opens
//! the encoder, and a frame a client lost costs the key frame the caller codes on the refusal,
//! which SVT-AV1 before 2.0 codes only from a re-open as well.

use std::ffi::CString;
use std::ptr;

use codec_sys::svtav1::*;

use super::codec::{av1_is_key, frame_type_from_key, push_video_header, vpx_level, Codec, VIDEO_HEADER_LEN};
use super::reference::Reference;
use super::session::{encode_threads, Pending, Planes, Quality, RateSettings};
use crate::RustCaptureSettings;

/// The lowest quantizer level the real-time mode runs at; below it the library faults.
const RTC_MIN_LEVEL: u32 = 3;

/// SVT-AV1 before 2.0 codes a key frame asked for mid-stream only in its random-access quality
/// mode; a session on one re-opens the encoder for it, and the first frame answers.
const KEYFRAME_BY_REOPEN: bool = SVT_AV1_VERSION_MAJOR < 2;

/// One SVT-AV1 session for one capture.
pub struct SvtAv1Encoder {
    handle: *mut EbComponentType,
    config: Box<EbSvtAv1EncConfiguration>,
    /// The input picture the library reads: the header and the plane pointers it carries.
    input: Box<(EbBufferHeaderType, EbSvtIOFormat)>,
    planes: Planes,
    threads: i32,
    quality: Quality,
    rate: RateSettings,
    omit_headers: bool,
    fresh: bool,
    next_pts: u64,
    pending: Pending,
}

unsafe impl Send for SvtAv1Encoder {}

fn error(what: &str, code: EbErrorType) -> String {
    format!("{what} (SVT-AV1 error {code:#x})")
}

impl Drop for SvtAv1Encoder {
    fn drop(&mut self) {
        self.close();
    }
}

impl SvtAv1Encoder {
    /// Open a session on host frames; `rgba` names the byte order they arrive in.
    pub fn new(settings: &RustCaptureSettings, rgba: bool) -> Result<Self, String> {
        let _ = rgba;
        let width = settings.width.max(1) as usize;
        let height = settings.height.max(1) as usize;
        let mut me = Self {
            handle: ptr::null_mut(),
            config: Box::new(unsafe { std::mem::zeroed() }),
            input: Box::new(unsafe { std::mem::zeroed() }),
            planes: Planes::new(width, height, false),
            threads: encode_threads(),
            quality: Quality::new(Codec::Av1.quantizer(settings.video_crf)),
            rate: RateSettings::new(settings),
            omit_headers: settings.omit_stripe_headers,
            fresh: true,
            next_pts: 0,
            pending: Pending::default(),
        };
        me.open()?;
        Ok(me)
    }

    /// The quantizer level the library takes for a quantizer index of AV1's domain.
    fn level(q: u32) -> u32 {
        vpx_level(Codec::Av1, q).max(RTC_MIN_LEVEL)
    }

    fn set(&mut self, name: &str, value: &str) -> Result<(), String> {
        let (n, v) = (CString::new(name).unwrap(), CString::new(value).unwrap());
        let code = unsafe { svt_av1_enc_parse_parameter(&mut *self.config, n.as_ptr(), v.as_ptr()) };
        if code != EB_ErrorNone {
            return Err(error(&format!("SVT-AV1 refused {name}={value}"), code));
        }
        Ok(())
    }

    /// Stand the encoder up with the live settings. The first frame it codes is a key frame.
    fn open(&mut self) -> Result<(), String> {
        self.close();
        let code = unsafe { init_handle(&mut self.handle, &mut *self.config) };
        if code != EB_ErrorNone || self.handle.is_null() {
            return Err(error("SVT-AV1 handed out no encoder handle", code));
        }
        let rate = self.rate;
        let bps = rate.bps();
        {
            let cfg = &mut *self.config;
            cfg.enc_mode = 11;
            cfg.source_width = self.planes.width as u32;
            cfg.source_height = self.planes.height as u32;
            cfg.frame_rate_numerator = rate.fps as u32;
            cfg.frame_rate_denominator = 1;
            cfg.encoder_bit_depth = 8;
            cfg.encoder_color_format = EB_YUV420;
            cfg.color_primaries = EB_CICP_CP_BT_709;
            cfg.transfer_characteristics = 1;
            cfg.matrix_coefficients = 1;
            cfg.color_range = 0;
            cfg.intra_refresh_type = 2;
            if rate.cbr {
                let (lo, hi) = (Codec::Av1.quantizer_bound(rate.min_qp), Codec::Av1.quantizer_bound(rate.max_qp));
                cfg.target_bit_rate = bps as u32;
                cfg.rate_control_mode = 2;
                cfg.max_qp_allowed = if hi > 0 { Self::level(hi) } else { 63 };
                cfg.min_qp_allowed = if lo > 0 { Self::level(lo) } else { 1 };
                // The library refuses a rate-control buffer shorter than 20 ms, which the
                // 1.5-frame VBV falls under above 75 fps.
                let vbv = (rate.vbv() as u64).max(bps / 50);
                cfg.maximum_buffer_size_ms = (vbv * 1000 / bps.max(1)) as i64;
            }
        }
        // Preset 11 in the real-time mode: a quarter less encode time than preset 10 for more
        // bytes at a fixed quantizer, which the quantizer table absorbs; the presets above it
        // are no faster. `lp` is a level of parallelism, 0..=6, not a thread count.
        if HAS_RTC {
            self.set("rtc", "1")?;
        }
        self.set("pred-struct", "1")?;
        self.set("lookahead", "0")?;
        self.set("keyint", "-1")?;
        self.set("tile-columns", "0")?;
        self.set("tile-rows", "0")?;
        self.set("lp", &self.threads.min(6).to_string())?;
        if rate.cbr {
            self.set("rc", "2")?;
        } else {
            self.set("rc", "0")?;
            self.set("qp", &Self::level(self.quality.current).to_string())?;
        }
        let code = unsafe { svt_av1_enc_set_parameter(self.handle, &mut *self.config) };
        if code != EB_ErrorNone {
            return Err(error("SVT-AV1 refused the parameters", code));
        }
        let code = unsafe { svt_av1_enc_init(self.handle) };
        if code != EB_ErrorNone {
            return Err(error("SVT-AV1 refused the session", code));
        }
        let (header, io) = &mut *self.input;
        *header = unsafe { std::mem::zeroed() };
        header.size = std::mem::size_of::<EbBufferHeaderType>() as u32;
        header.p_buffer = io as *mut EbSvtIOFormat as *mut u8;
        header.n_alloc_len = (self.planes.y.len() + self.planes.u.len() + self.planes.v.len()) as u32;
        self.fresh = true;
        Ok(())
    }

    /// Tell the library the stream ended, drain what it still holds, and free the encoder.
    fn close(&mut self) {
        if self.handle.is_null() {
            return;
        }
        unsafe {
            if !self.fresh {
                let mut last: EbBufferHeaderType = std::mem::zeroed();
                last.pic_type = EB_AV1_INVALID_PICTURE;
                last.flags = EB_BUFFERFLAG_EOS;
                svt_av1_enc_send_picture(self.handle, &mut last);
                loop {
                    let mut packet: *mut EbBufferHeaderType = ptr::null_mut();
                    if svt_av1_enc_get_packet(self.handle, &mut packet, 1) != EB_ErrorNone {
                        break;
                    }
                    let Some(header) = packet.as_ref() else { break };
                    let done = header.flags & EB_BUFFERFLAG_EOS != 0;
                    svt_av1_enc_release_out_buffer(&mut packet);
                    if done {
                        break;
                    }
                }
            }
            svt_av1_enc_deinit(self.handle);
            svt_av1_enc_deinit_handle(self.handle);
        }
        self.handle = ptr::null_mut();
    }

    pub fn codec(&self) -> Codec {
        Codec::Av1
    }

    /// The library behind the session, as the logs name it.
    pub fn library(&self) -> &'static str {
        "svt-av1"
    }

    pub fn is_fullcolor(&self) -> bool {
        false
    }

    pub fn is_full_range(&self) -> bool {
        false
    }

    pub fn last_reference(&self) -> Reference {
        Reference::Untracked
    }

    /// The library keeps its reference structure to itself: refused, so the caller codes a
    /// key frame.
    pub fn invalidate_reference(&mut self, _frame_id: u16) -> bool {
        false
    }

    /// Re-open the encoder when a rate or frame-rate setting changed.
    pub fn reconfigure_rate(&mut self, settings: &RustCaptureSettings) -> Result<(), String> {
        let Some(rate) = self.rate.changed(settings) else { return Ok(()) };
        self.rate = rate;
        self.open()
    }

    /// Encode one packed host frame at the quality index `crf`, as a key frame when `force_idr`.
    pub fn encode_host(
        &mut self,
        pixels: &[u8],
        stride: usize,
        rgba: bool,
        frame_number: u64,
        crf: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        let quality_changed = !self.rate.cbr && self.quality.update(Codec::Av1.quantizer(crf as i32)).is_some();
        if quality_changed || (force_idr && !self.fresh && KEYFRAME_BY_REOPEN) {
            self.open()?;
        }
        self.planes.convert(pixels, stride, rgba, false, false, self.threads as usize)?;
        let pts = self.next_pts;
        {
            let (header, io) = &mut *self.input;
            io.luma = self.planes.y.as_mut_ptr();
            io.cb = self.planes.u.as_mut_ptr();
            io.cr = self.planes.v.as_mut_ptr();
            io.y_stride = self.planes.width as u32;
            io.cb_stride = self.planes.chroma_width() as u32;
            io.cr_stride = self.planes.chroma_width() as u32;
            header.n_filled_len = header.n_alloc_len;
            header.pts = pts as i64;
            header.flags = 0;
            header.pic_type = if force_idr || self.fresh { EB_AV1_KEY_PICTURE } else { EB_AV1_INVALID_PICTURE };
        }
        let code = unsafe { svt_av1_enc_send_picture(self.handle, &mut self.input.0) };
        if code != EB_ErrorNone {
            return Err(error("SVT-AV1 refused the frame", code));
        }
        self.next_pts += 1;
        self.fresh = false;
        self.pending.push(pts, frame_number as u16);

        let mut output = Vec::new();
        loop {
            let mut packet: *mut EbBufferHeaderType = ptr::null_mut();
            let code = unsafe { svt_av1_enc_get_packet(self.handle, &mut packet, 0) };
            if code == EB_NoErrorEmptyQueue {
                break;
            }
            if code != EB_ErrorNone {
                return Err(error("SVT-AV1 handed out no packet", code));
            }
            let Some(p) = (unsafe { packet.as_ref() }) else { break };
            let bytes = unsafe { std::slice::from_raw_parts(p.p_buffer, p.n_filled_len as usize) }.to_vec();
            let packet_pts = p.pts.max(0) as u64;
            unsafe { svt_av1_enc_release_out_buffer(&mut packet) };
            let id = self.pending.take(packet_pts).unwrap_or(frame_number as u16);
            if !self.omit_headers {
                output.reserve(VIDEO_HEADER_LEN + bytes.len());
                push_video_header(
                    &mut output,
                    Codec::Av1,
                    frame_type_from_key(av1_is_key(&bytes)),
                    id,
                    0,
                    self.planes.width as u16,
                    self.planes.height as u16,
                    Reference::Untracked,
                );
            }
            output.extend_from_slice(&bytes);
            if self.pending.is_empty() {
                break;
            }
        }
        Ok(output)
    }
}
