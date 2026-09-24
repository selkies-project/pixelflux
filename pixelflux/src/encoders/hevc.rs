/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Software HEVC: x265 with the `gpl` feature, kvazaar without it, each at the fastest setting
//! its library offers in real time (ultrafast, wavefront threads, no lookahead, no reordering)
//! with the parameter sets repeated on every key frame. x265 carries 4:4:4 at full range like
//! x264; kvazaar encodes 4:2:0. Neither library lets a session steer what a frame predicts
//! from, so the session names no reference and refuses an invalidation, and the caller codes
//! a key frame instead.

use std::ffi::CString;

use super::codec::{h265_frame_type, push_video_header, Codec, VIDEO_HEADER_LEN};
use super::reference::Reference;
use super::session::{encode_threads, Pending, Planes, Quality, RateSettings};
use crate::RustCaptureSettings;

/// One software HEVC session for one capture.
pub struct HevcEncoder {
    backend: Backend,
    planes: Planes,
    threads: i32,
    quality: Quality,
    rate: RateSettings,
    omit_headers: bool,
    /// Whether the next frame is the first after an open, which the encoder codes as a key
    /// frame on its own.
    fresh: bool,
    next_pts: u64,
    pending: Pending,
}

unsafe impl Send for HevcEncoder {}

/// What a backend answered for one submitted picture.
struct Coded {
    bytes: Vec<u8>,
    pts: u64,
}

impl HevcEncoder {
    /// Open a session on host frames; `rgba` names the byte order they arrive in.
    pub fn new(settings: &RustCaptureSettings, rgba: bool) -> Result<Self, String> {
        let _ = rgba;
        let fullcolor = settings.video_fullcolor && Backend::FULLCOLOR;
        let threads = encode_threads();
        let rate = RateSettings::new(settings);
        let quality = Quality::new(Codec::H265.quantizer(settings.video_crf));
        let width = settings.width.max(1) as usize;
        let height = settings.height.max(1) as usize;
        let backend = Backend::open(width, height, threads, fullcolor, rate, quality.current)?;
        Ok(Self {
            backend,
            planes: Planes::new(width, height, fullcolor),
            threads,
            quality,
            rate,
            omit_headers: settings.omit_stripe_headers,
            fresh: true,
            next_pts: 0,
            pending: Pending::default(),
        })
    }

    pub fn codec(&self) -> Codec {
        Codec::H265
    }

    /// The library behind the session, as the logs name it.
    pub fn library(&self) -> &'static str {
        Backend::LIBRARY
    }

    pub fn is_fullcolor(&self) -> bool {
        self.planes.i444
    }

    /// The software 4:4:4 of x264's kind signals full range.
    pub fn is_full_range(&self) -> bool {
        self.planes.i444
    }

    pub fn last_reference(&self) -> Reference {
        Reference::Untracked
    }

    /// The library keeps its reference lists to itself: refused, so the caller codes a key frame.
    pub fn invalidate_reference(&mut self, _frame_id: u16) -> bool {
        false
    }

    /// Open a fresh encoder with the live settings, whose first frame is a key frame.
    fn reopen(&mut self) -> Result<(), String> {
        self.backend = Backend::open(self.planes.width, self.planes.height, self.threads, self.planes.i444, self.rate, self.quality.current)?;
        self.fresh = true;
        Ok(())
    }

    /// Apply a rate or frame-rate change: live where the library takes one, else by re-opening.
    pub fn reconfigure_rate(&mut self, settings: &RustCaptureSettings) -> Result<(), String> {
        let Some(rate) = self.rate.changed(settings) else { return Ok(()) };
        let fps_changed = rate.fps != self.rate.fps;
        self.rate = rate;
        if fps_changed || !self.backend.reconfigure_rate(rate) {
            self.reopen()?;
        }
        Ok(())
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
        if !self.rate.cbr
            && let Some(q) = self.quality.update(Codec::H265.quantizer(crf as i32))
            && !self.backend.reconfigure_quality(q)
        {
            self.reopen()?;
        }
        if force_idr && !self.fresh && !Backend::KEY_ON_REQUEST {
            self.reopen()?;
        }
        let key = force_idr || self.fresh;
        let full_range = self.is_full_range();
        let pts = self.next_pts;
        let coded = self.backend.encode(&mut self.planes, pixels, stride, rgba, full_range, self.threads as usize, pts, key)?;
        self.next_pts += 1;
        self.fresh = false;
        self.pending.push(pts, frame_number as u16);
        let Some(coded) = coded else { return Ok(Vec::new()) };
        let id = self.pending.take(coded.pts).unwrap_or(frame_number as u16);
        let mut output = Vec::with_capacity(VIDEO_HEADER_LEN + coded.bytes.len());
        if !self.omit_headers {
            push_video_header(
                &mut output,
                Codec::H265,
                h265_frame_type(&coded.bytes),
                id,
                0,
                self.planes.width as u16,
                self.planes.height as u16,
                Reference::Untracked,
            );
        }
        output.extend_from_slice(&coded.bytes);
        Ok(output)
    }
}

/// The CBR quantizer bounds in HEVC's domain, zero where the settings name none.
fn qp_bounds(rate: RateSettings) -> (u32, u32) {
    (Codec::H265.quantizer_bound(rate.min_qp), Codec::H265.quantizer_bound(rate.max_qp))
}

#[cfg(feature = "gpl")]
use x265::Backend;
#[cfg(not(feature = "gpl"))]
use kvazaar::Backend;

#[cfg(feature = "gpl")]
mod x265 {
    //! x265 through its versioned API table, configured by name so the session reads no field
    //! of a parameter structure that moves between releases.
    use std::ffi::c_int;
    use std::ptr;

    use codec_sys::x265::*;

    use super::*;

    pub struct Backend {
        api: *const x265_api,
        params: *mut x265_param,
        encoder: *mut x265_encoder,
    }

    impl Drop for Backend {
        fn drop(&mut self) {
            unsafe {
                let api = &*self.api;
                if !self.encoder.is_null() {
                    (api.encoder_close.unwrap())(self.encoder);
                }
                if !self.params.is_null() {
                    (api.param_free.unwrap())(self.params);
                }
            }
        }
    }

    impl Backend {
        pub const LIBRARY: &'static str = "x265";
        pub const FULLCOLOR: bool = true;
        pub const KEY_ON_REQUEST: bool = true;

        fn set(&self, name: &str, value: &str) -> Result<(), String> {
            let (n, v) = (CString::new(name).unwrap(), CString::new(value).unwrap());
            match unsafe { ((*self.api).param_parse.unwrap())(self.params, n.as_ptr(), v.as_ptr()) } {
                0 => Ok(()),
                X265_PARAM_BAD_NAME => Err(format!("x265 knows no option {name}")),
                _ => Err(format!("x265 refused {name}={value}")),
            }
        }

        /// The rate-control options of `rate` at quantizer `q`: a constant rate held to its
        /// VBV with the quantizer capped at 51, since the library's default ceiling admits
        /// out-of-spec quantizers that only force skips on a VBV underflow; else a constant
        /// rate factor.
        fn rate_options(rate: RateSettings, q: u32) -> Vec<(&'static str, String)> {
            let mut options = Vec::new();
            if rate.cbr {
                let kbps = (rate.bps() / 1000).max(1);
                let vbv_kbit = (rate.vbv() / 1000).max(1);
                options.push(("bitrate", kbps.to_string()));
                options.push(("vbv-maxrate", kbps.to_string()));
                options.push(("vbv-bufsize", vbv_kbit.to_string()));
                if rate.vbv() > 1000 {
                    options.push(("vbv-init", vbv_kbit.to_string()));
                }
                options.push(("strict-cbr", "1".into()));
                let (lo, hi) = qp_bounds(rate);
                if lo > 0 {
                    options.push(("qpmin", lo.to_string()));
                }
                options.push(("qpmax", if hi > 0 { hi } else { 51 }.to_string()));
            } else {
                options.push(("crf", q.to_string()));
            }
            options
        }

        pub fn open(width: usize, height: usize, threads: i32, i444: bool, rate: RateSettings, q: u32) -> Result<Self, String> {
            let api = unsafe { x265_api_get(8) };
            if api.is_null() {
                return Err("x265 carries no 8-bit encoder".into());
            }
            let params = unsafe { ((*api).param_alloc.unwrap())() };
            if params.is_null() {
                return Err("x265 allocated no parameters".into());
            }
            let mut me = Self { api, params, encoder: ptr::null_mut() };
            let (preset, tune) = (c"ultrafast", c"zerolatency");
            if unsafe { ((*api).param_default_preset.unwrap())(params, preset.as_ptr(), tune.as_ptr()) } < 0 {
                return Err("x265 refused the ultrafast zerolatency preset".into());
            }
            if width < 16 || height < 16 {
                return Err(format!("x265 encodes no picture as small as {width}x{height}"));
            }
            let mut options: Vec<(&str, String)> = vec![
                ("log-level", "none".into()),
                ("input-res", format!("{width}x{height}")),
                ("input-csp", if i444 { "i444" } else { "i420" }.into()),
                ("fps", format!("{}/1", rate.fps)),
                ("range", if i444 { "full" } else { "limited" }.into()),
                ("colorprim", "bt709".into()),
                ("transfer", "bt709".into()),
                ("colormatrix", "bt709".into()),
                ("keyint", "-1".into()),
                ("scenecut", "0".into()),
                ("repeat-headers", "1".into()),
                ("annexb", "1".into()),
                ("aud", "0".into()),
                ("rc-lookahead", "0".into()),
                ("bframes", "0".into()),
                ("frame-threads", "1".into()),
                ("pools", threads.to_string()),
                ("wpp", "1".into()),
            ];
            if width < 32 || height < 32 {
                options.push(("ctu", "16".into()));
            } else if width < 64 || height < 64 {
                options.push(("ctu", "32".into()));
            }
            options.extend(Self::rate_options(rate, q));
            for (name, value) in &options {
                me.set(name, value)?;
            }
            if i444 {
                let profile = c"main444-8";
                if unsafe { ((*api).param_apply_profile.unwrap())(params, profile.as_ptr()) } < 0 {
                    return Err("x265 refused the main444-8 profile".into());
                }
            }
            me.encoder = unsafe { ((*api).encoder_open.unwrap())(params) };
            if me.encoder.is_null() {
                return Err("x265 refused the session".into());
            }
            Ok(me)
        }

        /// Push the live parameters back into the running encoder; false where it refuses,
        /// and the session re-opens.
        fn reconfigure(&mut self, options: &[(&str, String)]) -> bool {
            unsafe {
                let api = &*self.api;
                (api.encoder_parameters.unwrap())(self.encoder, self.params);
                if options.iter().any(|(name, value)| self.set(name, value).is_err()) {
                    return false;
                }
                (api.encoder_reconfig.unwrap())(self.encoder, self.params) == 0
            }
        }

        pub fn reconfigure_quality(&mut self, q: u32) -> bool {
            self.reconfigure(&[("crf", q.to_string())])
        }

        pub fn reconfigure_rate(&mut self, rate: RateSettings) -> bool {
            let options = Self::rate_options(rate, 0);
            self.reconfigure(&options[..3])
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode(
            &mut self,
            planes: &mut Planes,
            pixels: &[u8],
            stride: usize,
            rgba: bool,
            full_range: bool,
            threads: usize,
            pts: u64,
            key: bool,
        ) -> Result<Option<Coded>, String> {
            planes.convert(pixels, stride, rgba, full_range, false, threads)?;
            let api = unsafe { &*self.api };
            let mut pic: x265_picture = unsafe { std::mem::zeroed() };
            let mut out: [x265_picture; 2] = unsafe { std::mem::zeroed() };
            unsafe {
                (api.picture_init.unwrap())(self.params, &mut pic);
                for o in &mut out {
                    (api.picture_init.unwrap())(self.params, o);
                }
            }
            // Written per index: x265 4 declares a fourth plane for alpha, x265 3 does not.
            pic.planes[0] = planes.y.as_mut_ptr().cast();
            pic.planes[1] = planes.u.as_mut_ptr().cast();
            pic.planes[2] = planes.v.as_mut_ptr().cast();
            pic.stride[0] = planes.width as c_int;
            pic.stride[1] = planes.chroma_width() as c_int;
            pic.stride[2] = planes.chroma_width() as c_int;
            pic.pts = pts as i64;
            pic.bitDepth = 8;
            pic.sliceType = if key { X265_TYPE_IDR as c_int } else { X265_TYPE_AUTO as c_int };
            let mut nals: *mut x265_nal = ptr::null_mut();
            let mut count: u32 = 0;
            let ret = unsafe { encoder_encode(api, self.encoder, &mut nals, &mut count, &mut pic, out.as_mut_ptr()) };
            if ret < 0 {
                return Err("x265 refused the frame".into());
            }
            if count == 0 {
                return Ok(None);
            }
            let mut bytes = Vec::new();
            for nal in unsafe { std::slice::from_raw_parts(nals, count as usize) } {
                bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(nal.payload, nal.sizeBytes as usize) });
            }
            Ok(Some(Coded { bytes, pts: out[0].pts.max(0) as u64 }))
        }
    }
}

#[cfg(not(feature = "gpl"))]
mod kvazaar {
    //! kvazaar through its API table. The library takes no per-frame key-frame request and no
    //! live rate change, so both re-open the encoder.
    use std::ptr;

    use codec_sys::kvazaar::*;

    use super::*;

    pub struct Backend {
        api: *const kvz_api,
        config: *mut kvz_config,
        encoder: *mut kvz_encoder,
    }

    impl Drop for Backend {
        fn drop(&mut self) {
            unsafe {
                let api = &*self.api;
                if !self.encoder.is_null() {
                    (api.encoder_close.unwrap())(self.encoder);
                }
                if !self.config.is_null() {
                    (api.config_destroy.unwrap())(self.config);
                }
            }
        }
    }

    impl Backend {
        pub const LIBRARY: &'static str = "kvazaar";
        pub const FULLCOLOR: bool = false;
        pub const KEY_ON_REQUEST: bool = false;

        fn set(&self, name: &str, value: &str) -> Result<(), String> {
            let (n, v) = (CString::new(name).unwrap(), CString::new(value).unwrap());
            if unsafe { ((*self.api).config_parse.unwrap())(self.config, n.as_ptr(), v.as_ptr()) } == 0 {
                return Err(format!("kvazaar refused {name}={value}"));
            }
            Ok(())
        }

        pub fn open(width: usize, height: usize, threads: i32, i444: bool, rate: RateSettings, q: u32) -> Result<Self, String> {
            let _ = i444;
            if !width.is_multiple_of(8) || !height.is_multiple_of(8) {
                return Err(format!("kvazaar takes only pictures a multiple of eight, not {width}x{height}"));
            }
            let api = unsafe { kvz_api_get(8) };
            if api.is_null() {
                return Err("kvazaar carries no 8-bit encoder".into());
            }
            let config = unsafe { ((*api).config_alloc.unwrap())() };
            if config.is_null() {
                return Err("kvazaar allocated no configuration".into());
            }
            let me = Self { api, config, encoder: ptr::null_mut() };
            unsafe {
                if ((*api).config_init.unwrap())(config) == 0 {
                    return Err("kvazaar refused to initialize its configuration".into());
                }
                let cfg = &mut *config;
                cfg.width = width as i32;
                cfg.height = height as i32;
                cfg.framerate_num = rate.fps;
                cfg.framerate_denom = 1;
                cfg.vui.fullrange = 0;
                cfg.vui.colorprim = 1;
                cfg.vui.transfer = 1;
                cfg.vui.colormatrix = 1;
            }
            let mut options: Vec<(&str, String)> = vec![
                ("preset", "ultrafast".into()),
                ("gop", "0".into()),
                ("period", "0".into()),
                ("vps-period", "0".into()),
                ("threads", threads.to_string()),
                ("owf", "0".into()),
                ("wpp", "1".into()),
            ];
            if rate.cbr {
                options.push(("bitrate", rate.bps().to_string()));
                options.push(("rc-algorithm", "oba".into()));
            } else {
                options.push(("qp", q.to_string()));
            }
            for (name, value) in &options {
                me.set(name, value)?;
            }
            let mut me = me;
            me.encoder = unsafe { ((*api).encoder_open.unwrap())(config) };
            if me.encoder.is_null() {
                return Err("kvazaar refused the session".into());
            }
            Ok(me)
        }

        pub fn reconfigure_quality(&mut self, _q: u32) -> bool {
            false
        }

        pub fn reconfigure_rate(&mut self, _rate: RateSettings) -> bool {
            false
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode(
            &mut self,
            planes: &mut Planes,
            pixels: &[u8],
            stride: usize,
            rgba: bool,
            full_range: bool,
            threads: usize,
            pts: u64,
            _key: bool,
        ) -> Result<Option<Coded>, String> {
            let api = unsafe { &*self.api };
            let pic = unsafe { (api.picture_alloc.unwrap())(planes.width as i32, planes.height as i32) };
            if pic.is_null() {
                return Err("kvazaar allocated no picture".into());
            }
            let result = unsafe {
                let p = &mut *pic;
                let luma_stride = p.stride as usize;
                let chroma_stride = luma_stride / 2;
                let (w, h) = (planes.width, planes.height);
                let y = std::slice::from_raw_parts_mut(p.y, luma_stride * h);
                let u = std::slice::from_raw_parts_mut(p.u, chroma_stride * h.div_ceil(2));
                let v = std::slice::from_raw_parts_mut(p.v, chroma_stride * h.div_ceil(2));
                p.pts = pts as i64;
                super::super::session::convert_into(pixels, stride, w, h, rgba, false, full_range, false, threads, y, u, v, (luma_stride, chroma_stride))
            };
            if let Err(e) = result {
                unsafe { (api.picture_free.unwrap())(pic) };
                return Err(e);
            }
            let mut data: *mut kvz_data_chunk = ptr::null_mut();
            let mut len: u32 = 0;
            let mut recon: *mut kvz_picture = ptr::null_mut();
            let mut info: kvz_frame_info = unsafe { std::mem::zeroed() };
            let ok = unsafe {
                (api.encoder_encode.unwrap())(self.encoder, pic, &mut data, &mut len, &mut recon, ptr::null_mut(), &mut info)
            };
            let mut coded = None;
            if ok != 0 && !data.is_null() {
                let mut bytes = Vec::with_capacity(len as usize);
                let mut chunk = data;
                while !chunk.is_null() {
                    let c = unsafe { &*chunk };
                    bytes.extend_from_slice(&c.data[..c.len as usize]);
                    chunk = c.next;
                }
                let out_pts = if recon.is_null() { pts } else { unsafe { (*recon).pts.max(0) as u64 } };
                coded = Some(Coded { bytes, pts: out_pts });
            }
            unsafe {
                (api.picture_free.unwrap())(pic);
                (api.picture_free.unwrap())(recon);
                (api.chunk_free.unwrap())(data);
            }
            if ok == 0 {
                return Err("kvazaar refused the frame".into());
            }
            Ok(coded)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A session's first frame is a key frame, a forced one mid-stream is too, and the ones
    /// between are deltas that name no reference.
    #[test]
    fn key_frames_come_on_request_and_deltas_name_no_reference() {
        let settings = RustCaptureSettings { width: 64, height: 48, target_fps: 30.0, codec: Codec::H265, video_crf: 25, ..Default::default() };
        let mut enc = HevcEncoder::new(&settings, false).expect("session");
        assert_eq!(enc.library(), Backend::LIBRARY);
        let bgra = vec![128u8; 64 * 48 * 4];
        let mut kinds = Vec::new();
        for t in 0..4u64 {
            let out = enc.encode_host(&bgra, 64 * 4, false, t, 25, t == 2).expect("encode");
            kinds.push(super::super::codec::parse_video_type(out[1]).map(|(_, k)| k));
            assert_eq!(enc.last_reference(), Reference::Untracked);
        }
        assert_eq!(kinds, [Some(super::super::codec::FRAME_KEY), Some(super::super::codec::FRAME_DELTA), Some(super::super::codec::FRAME_KEY), Some(super::super::codec::FRAME_DELTA)]);
        assert!(!enc.invalidate_reference(1));
    }
}
