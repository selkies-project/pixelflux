//! Decoders for the client camera uplink.
//!
//! Clients send whatever their browser can produce: H.264 (WebCodecs or the WebRTC media track),
//! VP8/VP9/AV1/HEVC (same sources), or MJPEG from the canvas fallback. Every codec lands in the same
//! I420 view so the rest of the pipeline is codec-agnostic. Inter-coded codecs go through FFmpeg's
//! avcodec (already linked for VA-API); MJPEG goes through TurboJPEG, which pixelflux already uses for
//! its own JPEG stripes. Decoding is software here; a hardware decoder only changes which `Decoder`
//! is constructed.

use std::ffi::{c_int, CStr};
use std::ptr;

use ffmpeg_sys_next as ff;
use turbojpeg::{Decompressor, Image, PixelFormat, Subsamp, YuvImage};

use super::convert::{I420Buffer, I420View};

/// Input codecs, by wire id. The ids are part of the Selkies WebSocket framing and are exported to
/// Python as `VirtualCamera.CODEC_*`.
pub use crate::encoders::codec::Codec;
use crate::encoders::codec::{av1_is_key, h264_frame_type, h265_frame_type, vp8_is_key, vp9_is_key, FRAME_KEY};

impl Codec {
    fn av_codec_id(self) -> ff::AVCodecID {
        match self {
            Codec::Jpeg => ff::AVCodecID::AV_CODEC_ID_MJPEG,
            Codec::H264 => ff::AVCodecID::AV_CODEC_ID_H264,
            Codec::Vp8 => ff::AVCodecID::AV_CODEC_ID_VP8,
            Codec::Vp9 => ff::AVCodecID::AV_CODEC_ID_VP9,
            Codec::Av1 => ff::AVCodecID::AV_CODEC_ID_AV1,
            Codec::H265 => ff::AVCodecID::AV_CODEC_ID_HEVC,
        }
    }
}

/// Whether an encoded frame is a key frame, read from the bitstream itself: every codec here
/// declares it in its first bytes or NAL types.
pub fn sniff_keyframe(codec: Codec, data: &[u8]) -> Option<bool> {
    Some(match codec {
        Codec::Jpeg => true,
        Codec::H264 => h264_frame_type(data) == FRAME_KEY,
        Codec::H265 => h265_frame_type(data) == FRAME_KEY,
        Codec::Vp8 => vp8_is_key(data),
        Codec::Vp9 => vp9_is_key(data),
        Codec::Av1 => av1_is_key(data),
    })
}

/// Padding avcodec requires past the end of any packet it parses.
const AV_INPUT_BUFFER_PADDING: usize = 64;

#[derive(Debug)]
pub enum DecodeError {
    /// The packet could not be decoded; the stream needs a keyframe to resynchronize.
    Corrupt(String),
    /// The decoder itself failed and must be recreated.
    Fatal(String),
}

pub trait Decoder {
    fn codec(&self) -> Codec;
    /// Decode one encoded frame. `Ok(true)` when `frame()` now holds a new picture.
    fn decode(&mut self, data: &[u8]) -> Result<bool, DecodeError>;
    fn frame(&self) -> Option<I420View<'_>>;
}

pub fn new_decoder(codec: Codec) -> Result<Box<dyn Decoder>, String> {
    match codec {
        Codec::Jpeg => Ok(Box::new(JpegDecoder::new()?)),
        _ => Ok(Box::new(AvDecoder::new(codec)?)),
    }
}

fn ff_err(code: c_int) -> String {
    let mut buf = [0 as libc::c_char; 128];
    unsafe {
        ff::av_strerror(code, buf.as_mut_ptr(), buf.len());
        CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
    }
}

/// avcodec-backed decoder for the inter-coded codecs.
pub struct AvDecoder {
    codec: Codec,
    ctx: *mut ff::AVCodecContext,
    pkt: *mut ff::AVPacket,
    frame: *mut ff::AVFrame,
    /// Receive target; each decoded picture is moved from here into `frame`.
    scratch: *mut ff::AVFrame,
    /// Encoded input with avcodec's trailing padding, reused across frames.
    input: Vec<u8>,
    /// Planar copy of frames whose avcodec pixel format is not I420.
    converted: I420Buffer,
    have_frame: bool,
    from_converted: bool,
}

unsafe impl Send for AvDecoder {}

impl AvDecoder {
    pub fn new(codec: Codec) -> Result<Self, String> {
        unsafe {
            static QUIET: std::sync::Once = std::sync::Once::new();
            QUIET.call_once(|| ff::av_log_set_level(ff::AV_LOG_ERROR));
            let c = ff::avcodec_find_decoder(codec.av_codec_id());
            if c.is_null() {
                return Err(format!("no avcodec decoder for {}", codec.name()));
            }
            let ctx = ff::avcodec_alloc_context3(c);
            if ctx.is_null() {
                return Err("avcodec_alloc_context3 failed".into());
            }
            (*ctx).thread_count = 2;
            (*ctx).thread_type = ff::FF_THREAD_SLICE;
            (*ctx).flags |= ff::AV_CODEC_FLAG_LOW_DELAY as c_int;
            let rc = ff::avcodec_open2(ctx, c, ptr::null_mut());
            if rc < 0 {
                let mut p = ctx;
                ff::avcodec_free_context(&mut p);
                return Err(format!("avcodec_open2({}) failed: {}", codec.name(), ff_err(rc)));
            }
            let pkt = ff::av_packet_alloc();
            let frame = ff::av_frame_alloc();
            let scratch = ff::av_frame_alloc();
            if pkt.is_null() || frame.is_null() || scratch.is_null() {
                let mut p = ctx;
                ff::avcodec_free_context(&mut p);
                return Err("avcodec packet/frame allocation failed".into());
            }
            Ok(AvDecoder {
                codec,
                ctx,
                pkt,
                frame,
                scratch,
                input: Vec::new(),
                converted: I420Buffer::new(2, 2),
                have_frame: false,
                from_converted: false,
            })
        }
    }

    /// Copy a non-I420 planar/semi-planar frame into the I420 scratch. Returns false for formats
    /// browsers never produce (they would need a full colorspace conversion).
    fn convert_frame(&mut self) -> bool {
        unsafe {
            let f = &*self.frame;
            let w = f.width as usize;
            let h = f.height as usize;
            let cw = w.div_ceil(2);
            let ch = h.div_ceil(2);
            self.converted.resize(w, h);
            let y_len = self.converted.y_len();
            let uv_len = self.converted.uv_len();
            let fmt: ff::AVPixelFormat = std::mem::transmute(f.format);
            let (yp, rest) = self.converted.data.split_at_mut(y_len);
            let (up, vp) = rest.split_at_mut(uv_len);
            for row in 0..h {
                let s = f.data[0].add(row * f.linesize[0] as usize);
                ptr::copy_nonoverlapping(s, yp.as_mut_ptr().add(row * w), w);
            }
            match fmt {
                ff::AVPixelFormat::AV_PIX_FMT_NV12 | ff::AVPixelFormat::AV_PIX_FMT_NV21 => {
                    let swap = fmt == ff::AVPixelFormat::AV_PIX_FMT_NV21;
                    for row in 0..ch {
                        let s = f.data[1].add(row * f.linesize[1] as usize);
                        for x in 0..cw {
                            let a = *s.add(2 * x);
                            let b = *s.add(2 * x + 1);
                            up[row * cw + x] = if swap { b } else { a };
                            vp[row * cw + x] = if swap { a } else { b };
                        }
                    }
                    true
                }
                ff::AVPixelFormat::AV_PIX_FMT_YUV422P | ff::AVPixelFormat::AV_PIX_FMT_YUVJ422P => {
                    for row in 0..ch {
                        let r0 = (2 * row).min(h - 1);
                        let r1 = (2 * row + 1).min(h - 1);
                        for (plane, dst) in [(1usize, &mut *up), (2usize, &mut *vp)] {
                            let s0 = f.data[plane].add(r0 * f.linesize[plane] as usize);
                            let s1 = f.data[plane].add(r1 * f.linesize[plane] as usize);
                            for x in 0..cw {
                                dst[row * cw + x] = ((*s0.add(x) as u32 + *s1.add(x) as u32 + 1) / 2) as u8;
                            }
                        }
                    }
                    true
                }
                ff::AVPixelFormat::AV_PIX_FMT_YUV444P | ff::AVPixelFormat::AV_PIX_FMT_YUVJ444P => {
                    for row in 0..ch {
                        let r0 = (2 * row).min(h - 1);
                        let r1 = (2 * row + 1).min(h - 1);
                        for (plane, dst) in [(1usize, &mut *up), (2usize, &mut *vp)] {
                            let s0 = f.data[plane].add(r0 * f.linesize[plane] as usize);
                            let s1 = f.data[plane].add(r1 * f.linesize[plane] as usize);
                            for x in 0..cw {
                                let x0 = 2 * x;
                                let x1 = (2 * x + 1).min(w - 1);
                                let sum = *s0.add(x0) as u32 + *s0.add(x1) as u32 + *s1.add(x0) as u32 + *s1.add(x1) as u32;
                                dst[row * cw + x] = ((sum + 2) / 4) as u8;
                            }
                        }
                    }
                    true
                }
                _ => false,
            }
        }
    }
}

impl Drop for AvDecoder {
    fn drop(&mut self) {
        unsafe {
            ff::av_frame_free(&mut self.frame);
            ff::av_frame_free(&mut self.scratch);
            ff::av_packet_free(&mut self.pkt);
            ff::avcodec_free_context(&mut self.ctx);
        }
    }
}

impl Decoder for AvDecoder {
    fn codec(&self) -> Codec {
        self.codec
    }

    fn decode(&mut self, data: &[u8]) -> Result<bool, DecodeError> {
        self.have_frame = false;
        self.input.clear();
        self.input.extend_from_slice(data);
        self.input.resize(data.len() + AV_INPUT_BUFFER_PADDING, 0);
        unsafe {
            ff::av_packet_unref(self.pkt);
            (*self.pkt).data = self.input.as_mut_ptr();
            (*self.pkt).size = data.len() as c_int;
            let rc = ff::avcodec_send_packet(self.ctx, self.pkt);
            if rc < 0 && rc != ff::AVERROR(libc::EAGAIN) {
                return Err(DecodeError::Corrupt(ff_err(rc)));
            }
            // avcodec_receive_frame unreferences its destination first, so drain
            // into the scratch frame and keep the newest picture in `frame`.
            let mut got = false;
            loop {
                let rc = ff::avcodec_receive_frame(self.ctx, self.scratch);
                if rc == ff::AVERROR(libc::EAGAIN) || rc == ff::AVERROR_EOF {
                    break;
                }
                if rc < 0 {
                    return Err(DecodeError::Corrupt(ff_err(rc)));
                }
                ff::av_frame_unref(self.frame);
                ff::av_frame_move_ref(self.frame, self.scratch);
                got = true;
            }
            if !got {
                return Ok(false);
            }
            let fmt: ff::AVPixelFormat = std::mem::transmute((*self.frame).format);
            self.from_converted = !matches!(fmt, ff::AVPixelFormat::AV_PIX_FMT_YUV420P | ff::AVPixelFormat::AV_PIX_FMT_YUVJ420P);
            if self.from_converted && !self.convert_frame() {
                return Err(DecodeError::Fatal(format!("unsupported decoded pixel format {}", (*self.frame).format)));
            }
            self.have_frame = true;
            Ok(true)
        }
    }

    fn frame(&self) -> Option<I420View<'_>> {
        if !self.have_frame {
            return None;
        }
        unsafe {
            let f = &*self.frame;
            let fmt: ff::AVPixelFormat = std::mem::transmute(f.format);
            let full_range = f.color_range == ff::AVColorRange::AVCOL_RANGE_JPEG
                || matches!(fmt, ff::AVPixelFormat::AV_PIX_FMT_YUVJ420P | ff::AVPixelFormat::AV_PIX_FMT_YUVJ422P | ff::AVPixelFormat::AV_PIX_FMT_YUVJ444P);
            if self.from_converted {
                return Some(self.converted.view(full_range));
            }
            let w = f.width as usize;
            let h = f.height as usize;
            let cw = w.div_ceil(2);
            let ch = h.div_ceil(2);
            let ys = f.linesize[0] as usize;
            let us = f.linesize[1] as usize;
            let vs = f.linesize[2] as usize;
            Some(I420View {
                width: w,
                height: h,
                y: std::slice::from_raw_parts(f.data[0], ys * (h - 1) + w),
                u: std::slice::from_raw_parts(f.data[1], us * (ch - 1) + cw),
                v: std::slice::from_raw_parts(f.data[2], vs * (ch - 1) + cw),
                y_stride: ys,
                uv_stride: us.max(vs).min(us),
                full_range,
            })
        }
    }
}

/// TurboJPEG decoder: 4:2:0 JPEGs decode straight into I420 planes; other subsamplings take the
/// RGB route and are converted.
pub struct JpegDecoder {
    dec: Decompressor,
    out: I420Buffer,
    rgb: Vec<u8>,
    have_frame: bool,
}

impl JpegDecoder {
    pub fn new() -> Result<Self, String> {
        Ok(JpegDecoder {
            dec: Decompressor::new().map_err(|e| format!("turbojpeg: {}", e))?,
            out: I420Buffer::new(2, 2),
            rgb: Vec::new(),
            have_frame: false,
        })
    }
}

impl Decoder for JpegDecoder {
    fn codec(&self) -> Codec {
        Codec::Jpeg
    }

    fn decode(&mut self, data: &[u8]) -> Result<bool, DecodeError> {
        self.have_frame = false;
        let hdr = self.dec.read_header(data).map_err(|e| DecodeError::Corrupt(format!("jpeg header: {}", e)))?;
        if hdr.width == 0 || hdr.height == 0 {
            return Err(DecodeError::Corrupt("empty jpeg".into()));
        }
        self.out.resize(hdr.width, hdr.height);
        if hdr.subsamp == Subsamp::Sub2x2 {
            let img = YuvImage { pixels: &mut self.out.data[..], width: hdr.width, align: 1, height: hdr.height, subsamp: Subsamp::Sub2x2 };
            self.dec.decompress_to_yuv(data, img).map_err(|e| DecodeError::Corrupt(format!("jpeg: {}", e)))?;
        } else {
            self.rgb.resize(hdr.width * hdr.height * 4, 0);
            let img = Image { pixels: &mut self.rgb[..], width: hdr.width, pitch: hdr.width * 4, height: hdr.height, format: PixelFormat::RGBA };
            self.dec.decompress(data, img).map_err(|e| DecodeError::Corrupt(format!("jpeg: {}", e)))?;
            let y_len = self.out.y_len();
            let uv_len = self.out.uv_len();
            let cw = hdr.width.div_ceil(2);
            let (yp, rest) = self.out.data.split_at_mut(y_len);
            let (up, vp) = rest.split_at_mut(uv_len);
            let mut planar = yuv::YuvPlanarImageMut {
                y_plane: yuv::BufferStoreMut::Borrowed(yp),
                y_stride: hdr.width as u32,
                u_plane: yuv::BufferStoreMut::Borrowed(up),
                u_stride: cw as u32,
                v_plane: yuv::BufferStoreMut::Borrowed(vp),
                v_stride: cw as u32,
                width: hdr.width as u32,
                height: hdr.height as u32,
            };
            yuv::rgba_to_yuv420(&mut planar, &self.rgb, (hdr.width * 4) as u32, yuv::YuvRange::Full, yuv::YuvStandardMatrix::Bt601, yuv::YuvConversionMode::Fast)
                .map_err(|e| DecodeError::Corrupt(format!("rgb->i420: {:?}", e)))?;
        }
        self.have_frame = true;
        Ok(true)
    }

    fn frame(&self) -> Option<I420View<'_>> {
        if self.have_frame {
            Some(self.out.view(true))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_ids_round_trip() {
        for id in 0..6 {
            let c = Codec::from_id(id).unwrap();
            assert_eq!(c as u32, id);
        }
        assert!(Codec::from_id(6).is_none());
    }

    #[test]
    fn h264_keyframe_sniff() {
        let idr = [0, 0, 0, 1, 0x67, 0x42, 0, 0, 0, 1, 0x68, 0xCE, 0, 0, 1, 0x65, 0x88];
        assert_eq!(sniff_keyframe(Codec::H264, &idr), Some(true));
        let p = [0, 0, 0, 1, 0x41, 0x9A];
        assert_eq!(sniff_keyframe(Codec::H264, &p), Some(false));
        assert_eq!(sniff_keyframe(Codec::Vp8, &[0x10, 0, 0]), Some(true));
        assert_eq!(sniff_keyframe(Codec::Vp8, &[0x11, 0, 0]), Some(false));
        assert_eq!(sniff_keyframe(Codec::Vp9, &[0x82, 0x49, 0x83]), Some(true));
        assert_eq!(sniff_keyframe(Codec::Vp9, &[0x86, 0]), Some(false));
        assert_eq!(sniff_keyframe(Codec::H265, &[0, 0, 1, 0x26, 0x01]), Some(true));
        assert_eq!(sniff_keyframe(Codec::H265, &[0, 0, 1, 0x02, 0x01]), Some(false));
        assert_eq!(sniff_keyframe(Codec::Av1, &[0x12, 0, 0x32, 0x01, 0x10]), Some(true));
        assert_eq!(sniff_keyframe(Codec::Av1, &[0x32, 0x01, 0x30]), Some(false));
    }

    #[test]
    fn jpeg_round_trip() {
        let w = 64;
        let h = 48;
        let mut img = I420Buffer::new(w, h);
        let yl = img.y_len();
        let ul = img.uv_len();
        img.data[..yl].fill(200);
        img.data[yl..yl + ul].fill(100);
        img.data[yl + ul..].fill(150);
        let src = YuvImage { pixels: &img.data[..], width: w, align: 1, height: h, subsamp: Subsamp::Sub2x2 };
        let jpeg = turbojpeg::compress_yuv(src, 90).unwrap();
        let mut dec = JpegDecoder::new().unwrap();
        assert!(dec.decode(&jpeg).unwrap());
        let v = dec.frame().unwrap();
        assert_eq!((v.width, v.height), (w, h));
        assert!(v.full_range);
        let mid = v.y[(h / 2) * v.y_stride + w / 2];
        assert!((mid as i32 - 200).abs() <= 3, "luma {}", mid);
        let cu = v.u[(h / 4) * v.uv_stride + w / 4];
        assert!((cu as i32 - 100).abs() <= 3, "cb {}", cu);
    }

    #[test]
    fn av_decoder_constructs_for_browser_codecs() {
        for c in [Codec::H264, Codec::Vp8, Codec::Vp9] {
            let d = AvDecoder::new(c).unwrap();
            assert_eq!(d.codec(), c);
            assert!(d.frame().is_none());
        }
    }

    #[test]
    fn h264_round_trip_through_avcodec() {
        use openh264::encoder::Encoder;
        use openh264::formats::YUVBuffer;
        let (w, h) = (64usize, 48usize);
        let mut yuv = vec![0u8; w * h * 3 / 2];
        yuv[..w * h].fill(145);
        yuv[w * h..w * h + w * h / 4].fill(54);
        yuv[w * h + w * h / 4..].fill(34);
        let source = YUVBuffer::from_vec(yuv, w, h);
        let mut enc = Encoder::new().unwrap();
        let mut dec = AvDecoder::new(Codec::H264).unwrap();
        let mut decoded = 0;
        for _ in 0..6 {
            let bitstream = enc.encode(&source).unwrap().to_vec();
            assert!(!bitstream.is_empty());
            if dec.decode(&bitstream).unwrap() {
                decoded += 1;
                let v = dec.frame().unwrap();
                assert_eq!((v.width, v.height), (w, h));
                assert!(!v.full_range);
                let y = v.y[(h / 2) * v.y_stride + w / 2];
                let u = v.u[(h / 4) * v.uv_stride + w / 4];
                let vv = v.v[(h / 4) * v.uv_stride + w / 4];
                assert!((y as i32 - 145).abs() <= 6 && (u as i32 - 54).abs() <= 6 && (vv as i32 - 34).abs() <= 6, "yuv {} {} {}", y, u, vv);
            }
        }
        assert!(decoded >= 5, "decoded {} of 6 frames", decoded);
    }

    #[test]
    fn av_decoder_rejects_garbage_without_fatal() {
        let mut d = AvDecoder::new(Codec::H264).unwrap();
        match d.decode(&[1, 2, 3, 4, 5, 6, 7, 8]) {
            Ok(false) | Err(DecodeError::Corrupt(_)) => {}
            Ok(true) => panic!("garbage decoded to a frame"),
            Err(DecodeError::Fatal(e)) => panic!("fatal: {}", e),
        }
    }
}
