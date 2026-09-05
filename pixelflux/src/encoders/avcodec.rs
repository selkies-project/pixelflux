/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! libavcodec-backed encoders: VA-API hardware sessions on a DRM render node for every
//! video codec, and the software HEVC / VP8 / VP9 / AV1 encoders the linked FFmpeg carries.
//!
//! One session type serves both. The codec context, packet drain, wire framing, the
//! rate-control re-open with its QP hysteresis and the keyframe policy are shared; only how
//! pixels reach the codec differs. A hardware session lands them on a VA surface — a Wayland
//! dmabuf mapped in place, or a packed host frame uploaded — and runs VA-VPP (`scale_vaapi`)
//! to convert to the surface format on the GPU, so no colorspace conversion happens on the
//! CPU. A software session converts a packed host frame into its planar input frame on the
//! encode thread and hands the planes to the codec without a further copy.
//!
//! Chroma follows `video_fullcolor` where the codec carries 4:4:4 (H.264 and H.265): a
//! hardware session negotiates the 4:4:4 surface format the driver reports, a software one
//! takes planar 4:4:4 at full range like x264. Anything else encodes 4:2:0.

// Every operation in these functions is an FFmpeg or VA-API call, or a dereference of a
// pointer one handed back; the safety contract is carried by the function signatures.
#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::mem;
use std::os::fd::AsRawFd;
use std::ptr;

use ffmpeg_sys_next as ff;
use libc::{close, dup, lseek, SEEK_END};

use super::codec::{
    av1_is_key, av1_level, frame_type_from_key, h264_frame_type, h264_level, h265_frame_type,
    h265_level, push_video_header, vp8_is_key, vp9_is_key, vpx_level, Codec, VIDEO_HEADER_LEN,
};
use super::software::convert_to_yuv_mt;
use super::QP_HYSTERESIS_LIMIT;
use crate::RustCaptureSettings;
use smithay::backend::allocator::{dmabuf::Dmabuf, Buffer};

/// Plane/object fan-out of the `AVDRM*` descriptors, matching FFmpeg's `AV_DRM_MAX_PLANES`.
const AV_DRM_MAX_PLANES: usize = 4;
/// The 8-bit 4:4:4 surface formats FFmpeg's VA-API hardware context can carry, in the order a
/// hardware session wants them: planar first, since it is what the host paths hold.
const FULLCOLOR_SW_FORMATS: [ff::AVPixelFormat; 2] = [
    ff::AVPixelFormat::AV_PIX_FMT_YUV444P,
    ff::AVPixelFormat::AV_PIX_FMT_VUYX,
];
/// A bitrate ceiling (100 Mbps) programmed wherever a constant-quantizer session needs a rate
/// target the encoder API demands but must never bind.
const BITRATE_CEILING_BPS: i64 = 100_000_000;

/// Mirrors FFmpeg's `libavutil/hwcontext_drm.h` ABI so a Wayland dmabuf can be handed to the
/// `hwmap` filter without a copy. FFmpeg reinterprets these bytes directly, so every field,
/// order and `#[repr(C)]` layout must stay bit-identical to the C definitions.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMObjectDescriptor {
    pub fd: c_int,
    pub size: usize,
    pub format_modifier: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMPlaneDescriptor {
    pub object_index: c_int,
    pub offset: isize,
    pub pitch: isize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMLayerDescriptor {
    pub format: u32,
    pub nb_planes: c_int,
    pub planes: [AVDRMPlaneDescriptor; AV_DRM_MAX_PLANES],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMFrameDescriptor {
    pub nb_objects: c_int,
    pub objects: [AVDRMObjectDescriptor; AV_DRM_MAX_PLANES],
    pub nb_layers: c_int,
    pub layers: [AVDRMLayerDescriptor; AV_DRM_MAX_PLANES],
}

/// The dup'd dmabuf fds FFmpeg is handed for one in-flight frame; `release_drm_frame` closes
/// them when FFmpeg tears the wrapping buffer down, so the compositor's own fds stay its own.
struct DmabufResources {
    fds: Vec<c_int>,
}

/// FFmpeg buffer-free callback for the custom DRM-PRIME frames: closes the dmabuf fds and
/// frees the descriptor. Runs inside `catch_unwind` because a panic must not cross the
/// `extern "C"` boundary. `data` is null only on the construction error path, where the
/// caller frees the descriptor itself.
unsafe extern "C" fn release_drm_frame(opaque: *mut c_void, data: *mut u8) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let resources = Box::from_raw(opaque as *mut DmabufResources);
        for &fd in &resources.fds {
            close(fd);
        }
        if !data.is_null() {
            ff::av_free(data as *mut c_void);
        }
    }));
}

/// An FFmpeg pixel format's canonical name, as a filter argument and in failure messages.
fn pix_fmt_name(fmt: ff::AVPixelFormat) -> String {
    unsafe {
        let name = ff::av_get_pix_fmt_name(fmt);
        if name.is_null() {
            format!("{fmt:?}")
        } else {
            CStr::from_ptr(name).to_string_lossy().into_owned()
        }
    }
}

/// The 4:4:4 surface format to encode into on this VA device, or `None` when the driver
/// carries none. This answers only the driver half: `av_hwframe_ctx_init` and
/// `avcodec_open2` still have to accept the format, and each reports its own refusal.
unsafe fn fullcolor_sw_format(device: *mut ff::AVBufferRef) -> Option<ff::AVPixelFormat> {
    let constraints = ff::av_hwdevice_get_hwframe_constraints(device, ptr::null());
    if constraints.is_null() {
        return None;
    }
    let mut carried = Vec::new();
    let mut fmt = (*constraints).valid_sw_formats;
    if !fmt.is_null() {
        while *fmt != ff::AVPixelFormat::AV_PIX_FMT_NONE {
            carried.push(*fmt);
            fmt = fmt.add(1);
        }
    }
    let mut owned = constraints;
    ff::av_hwframe_constraints_free(&mut owned);
    preferred_fullcolor_format(&carried)
}

/// The pick out of the formats a device reports, in `FULLCOLOR_SW_FORMATS` order.
fn preferred_fullcolor_format(carried: &[ff::AVPixelFormat]) -> Option<ff::AVPixelFormat> {
    FULLCOLOR_SW_FORMATS.into_iter().find(|wanted| carried.contains(wanted))
}

/// Format an FFmpeg error code through `av_strerror`.
fn ff_err_str(err: i32) -> String {
    unsafe {
        let mut errbuf = [0 as c_char; 128];
        ff::av_strerror(err, errbuf.as_mut_ptr(), 128);
        CStr::from_ptr(errbuf.as_ptr()).to_string_lossy().into_owned()
    }
}

/// Set one string option on a dictionary.
unsafe fn dict_set(d: &mut *mut ff::AVDictionary, key: &str, value: &str) {
    let ck = CString::new(key).unwrap();
    let cv = CString::new(value).unwrap();
    ff::av_dict_set(d, ck.as_ptr(), cv.as_ptr(), 0);
}

/// Where a session runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// A VA-API session on the render node the settings select.
    Vaapi,
    /// The software encoder the build resolved for the codec.
    Software,
}

/// How frames reach a session: a Wayland DRM-PRIME dmabuf (hardware only), or packed host
/// pixels in B,G,R,A (`rgba: false`) or R,G,B,A (`rgba: true`) byte order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Dmabuf,
    Host { rgba: bool },
}

/// The VA-API half of a session: device and frame contexts, the upload/convert filter graph,
/// and the reusable frames, all freed in dependency order by `Drop`.
struct VaapiSession {
    hw_device_ctx: *mut ff::AVBufferRef,
    drm_device_ctx: *mut ff::AVBufferRef,
    drm_frames_ctx: *mut ff::AVBufferRef,
    enc_frames_ctx: *mut ff::AVBufferRef,
    filter_graph: *mut ff::AVFilterGraph,
    buffersrc_ctx: *mut ff::AVFilterContext,
    buffersink_ctx: *mut ff::AVFilterContext,
    filtered_frame: *mut ff::AVFrame,
    /// Whether the codec was opened on the low-power (VDENC) entry point, which some Intel
    /// generations expose as the only one for HEVC, VP9 and AV1.
    low_power: bool,
}

impl Drop for VaapiSession {
    fn drop(&mut self) {
        unsafe {
            if !self.filtered_frame.is_null() {
                ff::av_frame_free(&mut self.filtered_frame);
            }
            if !self.filter_graph.is_null() {
                ff::avfilter_graph_free(&mut self.filter_graph);
            }
            for r in [
                &mut self.enc_frames_ctx,
                &mut self.drm_frames_ctx,
                &mut self.hw_device_ctx,
                &mut self.drm_device_ctx,
            ] {
                if !r.is_null() {
                    ff::av_buffer_unref(r);
                }
            }
        }
    }
}

/// One libavcodec encoder session, hardware or software, for one capture.
///
/// `current_qp` / `qp_hysteresis_counter` drive the constant-quantizer hysteresis of
/// `update_qp`; `cbr_mode`, `current_bitrate_kbps`, `current_vbv_mult` and `current_kf_s`
/// cache the live rate-control state so `reconfigure_rate` re-opens the codec only when a
/// value actually changes. `sw_format` is the format frames reach the codec in: the surface
/// format a hardware session negotiated, or the planar format a software one converts into.
pub struct AvcodecEncoder {
    codec: Codec,
    backend: Backend,
    input: Input,
    library: &'static str,
    avcodec: *const ff::AVCodec,
    encoder_ctx: *mut ff::AVCodecContext,
    hw: Option<VaapiSession>,
    frame: *mut ff::AVFrame,
    packet: *mut ff::AVPacket,
    width: i32,
    height: i32,
    fps: i32,
    threads: i32,
    sw_format: ff::AVPixelFormat,
    /// Whether a forced key frame has to re-open the codec because the encoder takes no
    /// per-frame request (kvazaar).
    keyframe_by_reopen: bool,
    /// Whether the next frame is the first after an open, which every encoder here emits as
    /// a key frame on its own.
    fresh: bool,
    current_qp: u32,
    qp_hysteresis_counter: u32,
    cbr_mode: bool,
    current_bitrate_kbps: i32,
    current_vbv_mult: f64,
    current_kf_s: f64,
    min_qp: i32,
    max_qp: i32,
    omit_stripe_headers: bool,
}

/// The raw FFmpeg pointers are owned exclusively and the session is driven from one capture
/// thread, so moving it across threads adds no aliasing.
unsafe impl Send for AvcodecEncoder {}

impl Drop for AvcodecEncoder {
    fn drop(&mut self) {
        unsafe {
            self.close_codec();
            if !self.packet.is_null() {
                ff::av_packet_free(&mut self.packet);
            }
            if !self.frame.is_null() {
                ff::av_frame_free(&mut self.frame);
            }
            self.hw.take();
        }
    }
}

impl AvcodecEncoder {
    /// Stand up a session for `codec` on `backend`, fed through `input`.
    ///
    /// A hardware session opens the DRM render node the settings select (`renderD128` when
    /// none is), derives the VA-API device, negotiates the surface format, opens the codec
    /// (retrying on the low-power entry point when the default one refuses) and builds the
    /// upload/convert filter graph. A software session resolves the build's encoder for the
    /// codec, opens it on a planar frame and allocates that frame. Every failure unwinds
    /// what was built so far and names the layer that refused, so the caller can fall back.
    pub fn new(
        settings: &RustCaptureSettings,
        codec: Codec,
        backend: Backend,
        input: Input,
    ) -> Result<Self, String> {
        if !codec.is_video() {
            return Err("JPEG has no libavcodec encoder".into());
        }
        if backend == Backend::Software && input == Input::Dmabuf {
            return Err("a software session takes host frames, not dmabufs".into());
        }
        let width = settings.width;
        let height = settings.height;
        let fps = (settings.target_fps as i32).max(1);
        let (library, name) = match backend {
            Backend::Vaapi => ("vaapi", format!("{}_vaapi", vaapi_codec_name(codec))),
            Backend::Software => {
                let enc = super::software_encoder(codec)
                    .filter(|e| !e.avcodec.is_empty())
                    .ok_or_else(|| format!("this FFmpeg carries no software {} encoder", codec.display()))?;
                (enc.library, enc.avcodec.to_string())
            }
        };
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .saturating_sub(1)
            .clamp(1, 8) as i32;

        unsafe {
            let cname = CString::new(name.clone()).unwrap();
            let avcodec = ff::avcodec_find_encoder_by_name(cname.as_ptr());
            if avcodec.is_null() {
                return Err(format!("{name} encoder not found in this FFmpeg"));
            }
            let mut me = Self {
                codec,
                backend,
                input,
                library,
                avcodec,
                encoder_ctx: ptr::null_mut(),
                hw: None,
                frame: ptr::null_mut(),
                packet: ptr::null_mut(),
                width,
                height,
                fps,
                threads,
                sw_format: ff::AVPixelFormat::AV_PIX_FMT_YUV420P,
                keyframe_by_reopen: library == "kvazaar",
                fresh: true,
                current_qp: codec.quantizer(settings.video_crf),
                qp_hysteresis_counter: 0,
                cbr_mode: settings.video_cbr_mode,
                current_bitrate_kbps: settings.video_bitrate_kbps,
                current_vbv_mult: settings.video_vbv_multiplier,
                current_kf_s: settings.keyframe_interval_s,
                min_qp: settings.video_min_qp,
                max_qp: settings.video_max_qp,
                omit_stripe_headers: settings.omit_stripe_headers,
            };
            me.packet = ff::av_packet_alloc();
            let fullcolor = settings.video_fullcolor && codec.fullcolor();
            match backend {
                Backend::Vaapi => me.open_vaapi(settings, fullcolor)?,
                Backend::Software => {
                    me.sw_format = if fullcolor && super::software_fullcolor(codec) {
                        ff::AVPixelFormat::AV_PIX_FMT_YUV444P
                    } else {
                        ff::AVPixelFormat::AV_PIX_FMT_YUV420P
                    };
                    me.open_codec(me.current_qp)?;
                    me.frame = ff::av_frame_alloc();
                    (*me.frame).format = me.sw_format as i32;
                    (*me.frame).width = width;
                    (*me.frame).height = height;
                    let ret = ff::av_frame_get_buffer(me.frame, 0);
                    if ret < 0 {
                        return Err(format!("Failed to allocate the input frame: {}", ff_err_str(ret)));
                    }
                }
            }
            Ok(me)
        }
    }

    /// The codec of this session.
    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// Where the session runs.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// The library or driver behind the session, as the logs name it.
    pub fn library(&self) -> &'static str {
        self.library
    }

    /// Whether this session negotiated 4:4:4 chroma. The request alone does not settle it —
    /// the driver, the FFmpeg build and the codec all have to carry it — so callers describing
    /// the active colorspace ask the encoder rather than the settings.
    pub fn is_fullcolor(&self) -> bool {
        !matches!(
            self.sw_format,
            ff::AVPixelFormat::AV_PIX_FMT_NV12 | ff::AVPixelFormat::AV_PIX_FMT_YUV420P
        )
    }

    /// Open the VA-API device, surface pool and filter graph, then the codec.
    unsafe fn open_vaapi(&mut self, settings: &RustCaptureSettings, fullcolor: bool) -> Result<(), String> {
        let render_node = if settings.encode_node_index >= 0 {
            format!("/dev/dri/renderD{}", 128 + settings.encode_node_index)
        } else {
            "/dev/dri/renderD128".to_string()
        };
        let device_url = CString::new(render_node).unwrap();
        let mut drm_device_ctx: *mut ff::AVBufferRef = ptr::null_mut();
        let ret = ff::av_hwdevice_ctx_create(
            &mut drm_device_ctx,
            ff::AVHWDeviceType::AV_HWDEVICE_TYPE_DRM,
            device_url.as_ptr(),
            ptr::null_mut(),
            0,
        );
        if ret < 0 {
            return Err(format!("Failed to create DRM device: {}", ff_err_str(ret)));
        }
        let mut session = VaapiSession {
            hw_device_ctx: ptr::null_mut(),
            drm_device_ctx,
            drm_frames_ctx: ptr::null_mut(),
            enc_frames_ctx: ptr::null_mut(),
            filter_graph: ptr::null_mut(),
            buffersrc_ctx: ptr::null_mut(),
            buffersink_ctx: ptr::null_mut(),
            filtered_frame: ptr::null_mut(),
            low_power: false,
        };
        let ret = ff::av_hwdevice_ctx_create_derived(
            &mut session.hw_device_ctx,
            ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
            drm_device_ctx,
            0,
        );
        if ret < 0 {
            return Err(format!("Failed to derive VAAPI device: {}", ff_err_str(ret)));
        }

        self.sw_format = if fullcolor {
            fullcolor_sw_format(session.hw_device_ctx)
                .ok_or("4:4:4 requested but this VA-API driver carries no 4:4:4 surface format")?
        } else {
            ff::AVPixelFormat::AV_PIX_FMT_NV12
        };

        let host_format = match self.input {
            Input::Dmabuf => ff::AVPixelFormat::AV_PIX_FMT_BGRA,
            Input::Host { rgba: false } => ff::AVPixelFormat::AV_PIX_FMT_BGRA,
            Input::Host { rgba: true } => ff::AVPixelFormat::AV_PIX_FMT_RGBA,
        };
        if self.input == Input::Dmabuf {
            session.drm_frames_ctx = ff::av_hwframe_ctx_alloc(drm_device_ctx);
            if session.drm_frames_ctx.is_null() {
                return Err("Failed to alloc DRM frames ctx".into());
            }
            let drm_frames = (*session.drm_frames_ctx).data as *mut ff::AVHWFramesContext;
            (*drm_frames).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME;
            (*drm_frames).sw_format = host_format;
            (*drm_frames).width = self.width;
            (*drm_frames).height = self.height;
            (*drm_frames).initial_pool_size = 0;
            if ff::av_hwframe_ctx_init(session.drm_frames_ctx) < 0 {
                return Err("Failed to init DRM frames ctx".into());
            }
        }

        session.enc_frames_ctx = ff::av_hwframe_ctx_alloc(session.hw_device_ctx);
        if session.enc_frames_ctx.is_null() {
            return Err("Failed to allocate encoder frames ctx".into());
        }
        let enc_frames = (*session.enc_frames_ctx).data as *mut ff::AVHWFramesContext;
        (*enc_frames).format = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
        (*enc_frames).sw_format = self.sw_format;
        (*enc_frames).width = (self.width + 31) & !31;
        (*enc_frames).height = (self.height + 31) & !31;
        (*enc_frames).initial_pool_size = 20;
        if ff::av_hwframe_ctx_init(session.enc_frames_ctx) < 0 {
            return Err(format!(
                "Failed to init encoder frames ctx: this VA-API driver allocates no {} surfaces",
                pix_fmt_name(self.sw_format)
            ));
        }

        // The codec context must reference the frames pool before the graph, and the
        // low-power retry needs the pool too, so the session is parked on self for the
        // shared open to reach.
        self.hw = Some(session);
        let qp = self.current_qp;
        if let Err(e) = self.open_codec(qp) {
            self.hw.as_mut().unwrap().low_power = true;
            if let Err(lp) = self.open_codec(qp) {
                return Err(if self.is_fullcolor() {
                    format!(
                        "Failed to open {} for 4:4:4 ({}): {e}; low-power entry point: {lp}",
                        self.library,
                        pix_fmt_name(self.sw_format)
                    )
                } else {
                    format!("Failed to open encoder: {e}; low-power entry point: {lp}")
                });
            }
        }
        self.build_graph(host_format)
    }

    /// Build the `buffersrc` → `hwmap`/`hwupload` + `scale_vaapi` → `buffersink` chain that
    /// lands every input on a GPU surface in `sw_format`, BT.709 limited range.
    ///
    /// The chain is staged with the segment API (parse, create filters, attach the VA device
    /// to every filter, apply) rather than the one-shot parser, because `hwupload`
    /// initializes during the parse and fails without a device, and a host buffersrc carries
    /// no frames context to derive one from.
    unsafe fn build_graph(&mut self, host_format: ff::AVPixelFormat) -> Result<(), String> {
        let session = self.hw.as_mut().unwrap();
        session.filter_graph = ff::avfilter_graph_alloc();
        let graph = session.filter_graph;
        let buffersrc = ff::avfilter_get_by_name(c"buffer".as_ptr());
        let buffersink = ff::avfilter_get_by_name(c"buffersink".as_ptr());
        session.buffersrc_ctx = ff::avfilter_graph_alloc_filter(graph, buffersrc, c"in".as_ptr());

        let par = ff::av_buffersrc_parameters_alloc();
        if par.is_null() {
            return Err("Failed to alloc buffersrc parameters".into());
        }
        if self.input == Input::Dmabuf {
            (*par).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32;
            (*par).hw_frames_ctx = ff::av_buffer_ref(session.drm_frames_ctx);
        } else {
            (*par).format = host_format as i32;
        }
        (*par).width = self.width;
        (*par).height = self.height;
        (*par).time_base = ff::AVRational { num: 1, den: self.fps };
        let ret = ff::av_buffersrc_parameters_set(session.buffersrc_ctx, par);
        if !(*par).hw_frames_ctx.is_null() {
            ff::av_buffer_unref(&mut (*par).hw_frames_ctx);
        }
        ff::av_free(par as *mut c_void);
        if ret < 0 {
            return Err(format!("Failed to set buffersrc parameters: {}", ff_err_str(ret)));
        }
        let args = CString::new(format!(
            "video_size={}x{}:time_base=1/{}:pixel_aspect=1/1",
            self.width, self.height, self.fps
        ))
        .unwrap();
        if ff::avfilter_init_str(session.buffersrc_ctx, args.as_ptr()) < 0 {
            return Err("Failed to init buffersrc".into());
        }
        if ff::avfilter_graph_create_filter(
            &mut session.buffersink_ctx,
            buffersink,
            c"out".as_ptr(),
            ptr::null(),
            ptr::null_mut(),
            graph,
        ) < 0
        {
            return Err("Failed to create buffersink".into());
        }

        let stage = if self.input == Input::Dmabuf { "hwmap" } else { "hwupload" };
        let filters_desc = CString::new(format!(
            "{},scale_vaapi=w={}:h={}:format={}:out_color_matrix=bt709:out_range=tv",
            stage,
            self.width,
            self.height,
            pix_fmt_name(self.sw_format)
        ))
        .unwrap();
        let mut seg: *mut ff::AVFilterGraphSegment = ptr::null_mut();
        let mut seg_inputs: *mut ff::AVFilterInOut = ptr::null_mut();
        let mut seg_outputs: *mut ff::AVFilterInOut = ptr::null_mut();
        let seg_ok = ff::avfilter_graph_segment_parse(graph, filters_desc.as_ptr(), 0, &mut seg) >= 0
            && ff::avfilter_graph_segment_create_filters(seg, 0) >= 0
            && {
                for i in 0..(*graph).nb_filters {
                    let f = *(*graph).filters.add(i as usize);
                    if (*f).hw_device_ctx.is_null() {
                        (*f).hw_device_ctx = ff::av_buffer_ref(session.hw_device_ctx);
                    }
                }
                ff::avfilter_graph_segment_apply(seg, 0, &mut seg_inputs, &mut seg_outputs) >= 0
            }
            && !seg_inputs.is_null()
            && !seg_outputs.is_null()
            && ff::avfilter_link(
                session.buffersrc_ctx,
                0,
                (*seg_inputs).filter_ctx,
                (*seg_inputs).pad_idx as u32,
            ) >= 0
            && ff::avfilter_link(
                (*seg_outputs).filter_ctx,
                (*seg_outputs).pad_idx as u32,
                session.buffersink_ctx,
                0,
            ) >= 0;
        ff::avfilter_inout_free(&mut seg_inputs);
        ff::avfilter_inout_free(&mut seg_outputs);
        ff::avfilter_graph_segment_free(&mut seg);
        if !seg_ok {
            return Err("Failed to build filter graph".into());
        }
        if ff::avfilter_graph_config(graph, ptr::null_mut()) < 0 {
            return Err("Failed to config filter graph".into());
        }
        session.filtered_frame = ff::av_frame_alloc();
        self.frame = ff::av_frame_alloc();
        Ok(())
    }

    /// Drain and free the open codec context, if any. A software encoder is told the stream
    /// ended first, since some (SVT-AV1) flush their threads only on that and complain
    /// otherwise; the packets that come out are discarded, as no frame is pending.
    unsafe fn close_codec(&mut self) {
        if self.encoder_ctx.is_null() {
            return;
        }
        if self.backend == Backend::Software && !self.fresh {
            if ff::avcodec_send_frame(self.encoder_ctx, ptr::null()) >= 0 {
                while ff::avcodec_receive_packet(self.encoder_ctx, self.packet) >= 0 {
                    ff::av_packet_unref(self.packet);
                }
            }
        }
        ff::avcodec_free_context(&mut self.encoder_ctx);
    }

    /// Open a fresh codec context at quantizer `qp` with the session's live rate-control
    /// state, replacing any open one. The first frame of a fresh context is a key frame.
    unsafe fn open_codec(&mut self, qp: u32) -> Result<(), String> {
        self.close_codec();
        let ctx = ff::avcodec_alloc_context3(self.avcodec);
        if ctx.is_null() {
            return Err("Failed to allocate encoder context".into());
        }
        self.encoder_ctx = ctx;
        (*ctx).width = self.width;
        (*ctx).height = self.height;
        (*ctx).time_base = ff::AVRational { num: 1, den: self.fps };
        (*ctx).framerate = ff::AVRational { num: self.fps, den: 1 };
        (*ctx).max_b_frames = 0;
        (*ctx).gop_size = c_int::MAX;
        (*ctx).thread_count = self.threads;
        let fullcolor = self.is_fullcolor();
        let software = self.backend == Backend::Software;
        (*ctx).color_range = if fullcolor && software {
            ff::AVColorRange::AVCOL_RANGE_JPEG
        } else {
            ff::AVColorRange::AVCOL_RANGE_MPEG
        };
        (*ctx).colorspace = ff::AVColorSpace::AVCOL_SPC_BT709;
        (*ctx).color_primaries = ff::AVColorPrimaries::AVCOL_PRI_BT709;
        (*ctx).color_trc = ff::AVColorTransferCharacteristic::AVCOL_TRC_BT709;
        if let Some(session) = self.hw.as_ref() {
            (*ctx).pix_fmt = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
            (*ctx).hw_device_ctx = ff::av_buffer_ref(session.hw_device_ctx);
            (*ctx).hw_frames_ctx = ff::av_buffer_ref(session.enc_frames_ctx);
            (*ctx).compression_level = 6;
            if matches!(self.codec, Codec::H264 | Codec::H265) {
                (*ctx).slices = 4;
            }
        } else {
            (*ctx).pix_fmt = self.sw_format;
        }

        let mut opts: *mut ff::AVDictionary = ptr::null_mut();
        let bps = (self.current_bitrate_kbps.max(0) as i64).saturating_mul(1000);
        let vbv = super::vbv_bits(
            bps.min(u32::MAX as i64) as u32,
            self.fps as f64,
            self.current_kf_s,
            self.current_vbv_mult,
        ) as i64;
        if self.cbr_mode {
            (*ctx).bit_rate = bps;
            (*ctx).rc_max_rate = bps;
            (*ctx).rc_min_rate = bps;
            (*ctx).rc_buffer_size = vbv.min(i32::MAX as i64) as i32;
            (*ctx).rc_initial_buffer_occupancy = (*ctx).rc_buffer_size;
            let (lo, hi) = (
                self.codec.quantizer_bound(self.min_qp),
                self.codec.quantizer_bound(self.max_qp),
            );
            if lo > 0 {
                (*ctx).qmin = self.encoder_quantizer(lo) as i32;
            }
            if hi > 0 {
                (*ctx).qmax = self.encoder_quantizer(hi) as i32;
            }
        }
        match self.backend {
            Backend::Vaapi => self.vaapi_options(&mut opts, qp),
            Backend::Software => self.software_options(ctx, &mut opts, qp),
        }
        let ret = ff::avcodec_open2(ctx, self.avcodec, &mut opts);
        ff::av_dict_free(&mut opts);
        if ret < 0 {
            // A failed open leaves the context unopened; it is freed outright so the encode
            // entry points refuse it and the caller rebuilds the session.
            ff::avcodec_free_context(&mut self.encoder_ctx);
            return Err(format!("Failed to open {}: {}", self.library, ff_err_str(ret)));
        }
        self.current_qp = qp;
        self.fresh = true;
        Ok(())
    }

    /// The value a quantizer of the codec's domain is programmed as on this session's
    /// encoder: the libvpx and SVT-AV1 encoders take the 0..=63 level, everything else the
    /// domain value itself.
    fn encoder_quantizer(&self, q: u32) -> u32 {
        if self.backend == Backend::Software && matches!(self.codec, Codec::Vp8 | Codec::Vp9 | Codec::Av1) {
            vpx_level(self.codec, q)
        } else {
            q
        }
    }

    /// The private options of a VA-API session: rate-control mode and quantizer, a single
    /// frame in flight, the profile the surface format implies, the lowest fitting level,
    /// and the low-power entry point when the default one refused.
    unsafe fn vaapi_options(&self, opts: &mut *mut ff::AVDictionary, qp: u32) {
        let ctx = self.encoder_ctx;
        let (w, h, fps) = (self.width as u32, self.height as u32, self.fps as u32);
        if self.cbr_mode {
            dict_set(opts, "rc_mode", "CBR");
        } else {
            dict_set(opts, "rc_mode", "CQP");
            match self.codec {
                Codec::H264 | Codec::H265 => dict_set(opts, "qp", &qp.to_string()),
                _ => (*ctx).global_quality = qp.max(1) as i32,
            }
        }
        dict_set(opts, "async_depth", "1");
        dict_set(opts, "idr_interval", "0");
        if self.hw.as_ref().is_some_and(|s| s.low_power) {
            dict_set(opts, "low_power", "1");
        }
        match self.codec {
            Codec::H264 => {
                // Naming a profile pins 4:2:0 to High; a 4:4:4 session leaves it for FFmpeg
                // to match against the surface format.
                if !self.is_fullcolor() {
                    dict_set(opts, "profile", "high");
                }
                dict_set(opts, "level", &h264_level(w, h, fps).to_string());
            }
            Codec::H265 => {
                dict_set(opts, "profile", if self.is_fullcolor() { "rext" } else { "main" });
                dict_set(opts, "level", &h265_level(w, h, fps).to_string());
            }
            Codec::Av1 => {
                dict_set(opts, "profile", "main");
                dict_set(opts, "level", &av1_level(w, h, fps).to_string());
            }
            Codec::Vp8 | Codec::Vp9 | Codec::Jpeg => {}
        }
    }

    /// The options of a software session: every encoder is tuned for the lowest latency it
    /// offers (no frame threading or lookahead, no reordering, an unbounded GOP with key
    /// frames only on request, parameter sets repeated on every key frame) and for screen
    /// content where it has such a mode, with rate control either CBR at the session's
    /// bitrate and VBV or a constant quantizer.
    unsafe fn software_options(&self, ctx: *mut ff::AVCodecContext, opts: &mut *mut ff::AVDictionary, qp: u32) {
        let q = self.encoder_quantizer(qp);
        match self.library {
            "x265" => {
                dict_set(opts, "preset", "ultrafast");
                dict_set(opts, "tune", "zerolatency");
                dict_set(opts, "forced-idr", "1");
                let mut params = format!(
                    "keyint=-1:scenecut=0:repeat-headers=1:annexb=1:aud=0:rc-lookahead=0:bframes=0:frame-threads=1:pools={}:wpp=1:log-level=none",
                    self.threads
                );
                if self.cbr_mode {
                    params.push_str(":strict-cbr=1");
                } else {
                    params.push_str(&format!(":crf={q}"));
                }
                dict_set(opts, "x265-params", &params);
                if self.is_fullcolor() {
                    dict_set(opts, "profile", "main444-8");
                }
            }
            "kvazaar" => {
                let mut params = format!(
                    "preset=ultrafast,gop=0,intra-period=0,vps-period=0,threads={},owf=0,wpp=1",
                    self.threads
                );
                if self.cbr_mode {
                    params.push_str(&format!(",bitrate={},rc-algorithm=oba", (*ctx).bit_rate));
                } else {
                    params.push_str(&format!(",qp={q}"));
                }
                dict_set(opts, "kvazaar-params", &params);
            }
            "libvpx" => {
                dict_set(opts, "deadline", "realtime");
                dict_set(opts, "cpu-used", if self.codec == Codec::Vp9 { "7" } else { "8" });
                dict_set(opts, "lag-in-frames", "0");
                dict_set(opts, "auto-alt-ref", "0");
                dict_set(opts, "error-resilient", "0");
                dict_set(opts, "static-thresh", "0");
                dict_set(opts, "max-intra-rate", "0");
                if self.codec == Codec::Vp9 {
                    dict_set(opts, "row-mt", "1");
                    dict_set(opts, "tune-content", "screen");
                    dict_set(opts, "frame-parallel", "0");
                    let tile_columns = (self.width / 512).max(1).ilog2().min(6);
                    dict_set(opts, "tile-columns", &tile_columns.to_string());
                }
                if !self.cbr_mode {
                    // A pinned quantizer: libvpx holds a level fixed only between qmin and
                    // qmax, and the rate target that mode still wants must never bind.
                    (*ctx).qmin = q as i32;
                    (*ctx).qmax = q as i32;
                    (*ctx).bit_rate = BITRATE_CEILING_BPS;
                    (*ctx).rc_max_rate = BITRATE_CEILING_BPS;
                    (*ctx).rc_min_rate = BITRATE_CEILING_BPS;
                }
            }
            "svt-av1" => {
                dict_set(opts, "preset", "10");
                let mut params = format!(
                    "pred-struct=1:lookahead=0:force-key-frames=1:keyint=-2:scm=1:fast-decode=1:tile-columns=0:tile-rows=0:lp={}",
                    self.threads
                );
                if self.cbr_mode {
                    params.push_str(":rc=2");
                } else {
                    params.push_str(&format!(":rc=0:qp={q}"));
                }
                dict_set(opts, "svtav1-params", &params);
            }
            _ => {}
        }
    }

    /// The encode entry points run behind this: a rate or QP re-open that failed leaves no
    /// codec context, and the session has to be rebuilt rather than encoded into.
    fn require_open_codec(&self) -> Result<(), String> {
        if self.encoder_ctx.is_null() {
            return Err("no open codec context after a failed re-open; the session needs a rebuild".into());
        }
        Ok(())
    }

    /// Move the constant quantizer toward the one the session quality index `crf` selects,
    /// weighing each change against the re-open (and key frame) it costs: a decrease sharpens
    /// the picture and applies at once, an increase waits out `QP_HYSTERESIS_LIMIT`
    /// consecutive requests so transient motion does not make quality blink. A no-op in CBR
    /// mode.
    unsafe fn update_qp(&mut self, crf: u32) -> Result<(), String> {
        if self.cbr_mode {
            return Ok(());
        }
        let target_qp = self.codec.quantizer(crf as i32);
        if target_qp == self.current_qp {
            self.qp_hysteresis_counter = 0;
            return Ok(());
        }
        if target_qp < self.current_qp {
            self.qp_hysteresis_counter = 0;
            self.open_codec(target_qp)?;
        } else {
            self.qp_hysteresis_counter += 1;
            if self.qp_hysteresis_counter > QP_HYSTERESIS_LIMIT {
                self.qp_hysteresis_counter = 0;
                self.open_codec(target_qp)?;
            }
        }
        Ok(())
    }

    /// Re-open the codec only when a rate-control or frame-rate setting actually changed:
    /// in CBR a different bitrate or VBV multiplier, in any mode a different fps. `Err`
    /// means the re-open failed and the session has no codec context: the caller rebuilds.
    pub fn reconfigure_rate(&mut self, settings: &RustCaptureSettings) -> Result<(), String> {
        let mut changed = false;
        if self.cbr_mode
            && (settings.video_bitrate_kbps != self.current_bitrate_kbps
                || settings.video_vbv_multiplier != self.current_vbv_mult)
        {
            changed = true;
        }
        let new_fps = settings.target_fps.max(1.0) as i32;
        if new_fps != self.fps {
            changed = true;
        }
        if !changed {
            return Ok(());
        }
        self.fps = new_fps;
        self.current_bitrate_kbps = settings.video_bitrate_kbps;
        self.current_vbv_mult = settings.video_vbv_multiplier;
        self.current_kf_s = settings.keyframe_interval_s;
        unsafe { self.open_codec(self.current_qp) }
    }

    /// Send one frame, marked as a key frame when `force_idr`, then drain every packet the
    /// encoder has into `output` behind the wire header (unless headers are omitted). An
    /// encoder that takes no per-frame key request gets a fresh context instead.
    unsafe fn encode_frame(
        &mut self,
        frame: *mut ff::AVFrame,
        frame_number: u64,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        if force_idr && !self.fresh && self.keyframe_by_reopen {
            self.open_codec(self.current_qp)?;
        }
        (*frame).pts = frame_number as i64;
        (*frame).pict_type = if force_idr {
            ff::AVPictureType::AV_PICTURE_TYPE_I
        } else {
            ff::AVPictureType::AV_PICTURE_TYPE_NONE
        };
        let ret = ff::avcodec_send_frame(self.encoder_ctx, frame);
        if ret < 0 {
            return Err(format!("Failed to send frame to encoder: {}", ff_err_str(ret)));
        }
        self.fresh = false;
        let mut output = Vec::new();
        while ff::avcodec_receive_packet(self.encoder_ctx, self.packet) == 0 {
            let size = (*self.packet).size as usize;
            let bytes = std::slice::from_raw_parts((*self.packet).data, size);
            if !self.omit_stripe_headers {
                let frame_type = match self.codec {
                    Codec::H264 => h264_frame_type(bytes),
                    Codec::H265 => h265_frame_type(bytes),
                    Codec::Vp8 => frame_type_from_key(vp8_is_key(bytes)),
                    Codec::Vp9 => frame_type_from_key(vp9_is_key(bytes)),
                    _ => frame_type_from_key(av1_is_key(bytes)),
                };
                output.reserve(VIDEO_HEADER_LEN + size);
                push_video_header(
                    &mut output,
                    self.codec,
                    frame_type,
                    frame_number as u16,
                    0,
                    self.width as u16,
                    self.height as u16,
                );
            }
            output.extend_from_slice(bytes);
            ff::av_packet_unref(self.packet);
        }
        Ok(output)
    }

    /// Push a frame through the filter graph and encode whatever comes out of the sink.
    unsafe fn encode_through_graph(&mut self, frame_number: u64, force_idr: bool) -> Result<Vec<u8>, String> {
        let session = self.hw.as_ref().unwrap();
        let (src, sink, filtered) = (session.buffersrc_ctx, session.buffersink_ctx, session.filtered_frame);
        if ff::av_buffersrc_add_frame(src, self.frame) < 0 {
            ff::av_frame_unref(self.frame);
            return Err("Failed to feed filter graph".into());
        }
        let mut output = Vec::new();
        while ff::av_buffersink_get_frame(sink, filtered) >= 0 {
            let result = self.encode_frame(filtered, frame_number, force_idr);
            ff::av_frame_unref(filtered);
            output.extend(result?);
        }
        Ok(output)
    }

    /// Encode one Wayland DRM-PRIME dmabuf: the buffer is described to FFmpeg by a DRM frame
    /// descriptor over dup'd fds (FFmpeg closes what it is handed), `hwmap`ped onto a VA
    /// surface and converted there before encode.
    pub fn encode_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        frame_number: u64,
        crf: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        if self.input != Input::Dmabuf {
            return Err("this session takes host frames".into());
        }
        unsafe {
            self.update_qp(crf)?;
            self.require_open_codec()?;

            let desc_size = mem::size_of::<AVDRMFrameDescriptor>();
            let desc_ptr = ff::av_mallocz(desc_size) as *mut AVDRMFrameDescriptor;
            if desc_ptr.is_null() {
                return Err("OOM".into());
            }
            let mut resources = DmabufResources { fds: Vec::new() };
            let fail = move |resources: &DmabufResources, msg: &str| -> String {
                for &fd in &resources.fds {
                    close(fd);
                }
                ff::av_free(desc_ptr as *mut c_void);
                msg.to_string()
            };

            (*desc_ptr).nb_objects = dmabuf.handles().count() as i32;
            (*desc_ptr).nb_layers = 1;
            for (i, handle) in dmabuf.handles().enumerate() {
                let fd = dup(handle.as_raw_fd());
                if fd < 0 {
                    return Err(fail(&resources, "Failed to dup fd"));
                }
                resources.fds.push(fd);
                (*desc_ptr).objects[i].fd = fd;
                // The fd reports the object's real size; stride times height is wrong for
                // tiled or compressed layouts and for a BO padded beyond the image rows.
                let object_size = lseek(fd, 0, SEEK_END);
                if object_size <= 0 {
                    return Err(fail(&resources, "Failed to query the dmabuf object size"));
                }
                (*desc_ptr).objects[i].size = object_size as usize;
                (*desc_ptr).objects[i].format_modifier = u64::from(dmabuf.format().modifier);
            }
            (*desc_ptr).layers[0].format = dmabuf.format().code as u32;
            (*desc_ptr).layers[0].nb_planes = dmabuf.num_planes() as i32;
            let single_object = dmabuf.handles().count() == 1;
            for (i, (stride, offset)) in dmabuf.strides().zip(dmabuf.offsets()).enumerate() {
                (*desc_ptr).layers[0].planes[i].object_index = if single_object { 0 } else { i as i32 };
                (*desc_ptr).layers[0].planes[i].offset = offset as isize;
                (*desc_ptr).layers[0].planes[i].pitch = stride as isize;
            }

            ff::av_frame_unref(self.frame);
            (*self.frame).width = self.width;
            (*self.frame).height = self.height;
            (*self.frame).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32;
            (*self.frame).data[0] = desc_ptr as *mut u8;
            let opaque = Box::into_raw(Box::new(resources));
            let buf_ref = ff::av_buffer_create(
                desc_ptr as *mut u8,
                desc_size,
                Some(release_drm_frame),
                opaque as *mut c_void,
                0,
            );
            if buf_ref.is_null() {
                release_drm_frame(opaque as *mut c_void, ptr::null_mut());
                ff::av_free(desc_ptr as *mut c_void);
                return Err("Failed to create buffer ref".into());
            }
            (*self.frame).buf[0] = buf_ref;
            (*self.frame).pts = frame_number as i64;
            (*self.frame).hw_frames_ctx = ff::av_buffer_ref(self.hw.as_ref().unwrap().drm_frames_ctx);
            self.encode_through_graph(frame_number, force_idr)
        }
    }

    /// Encode one packed host frame (`stride` bytes per row, in the byte order the session
    /// was built for) at the quality index `crf`. A hardware session uploads it and converts
    /// on the GPU; a software session converts it into its planar input frame — 4:2:0 limited
    /// range, or 4:4:4 full range — across the encode threads and hands the planes to the
    /// codec.
    pub fn encode_host(
        &mut self,
        pixels: &[u8],
        stride: usize,
        frame_number: u64,
        crf: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        let Input::Host { rgba } = self.input else {
            return Err("this session takes dmabufs".into());
        };
        let h = self.height as usize;
        let row_bytes = (self.width as usize) * 4;
        let needed = if h == 0 { 0 } else { stride.checked_mul(h - 1).ok_or("stride overflow")? + row_bytes };
        if stride < row_bytes || pixels.len() < needed {
            return Err("Input buffer too small".into());
        }
        unsafe {
            self.update_qp(crf)?;
            self.require_open_codec()?;
            if self.hw.is_some() {
                ff::av_frame_unref(self.frame);
                (*self.frame).width = self.width;
                (*self.frame).height = self.height;
                (*self.frame).format = if rgba {
                    ff::AVPixelFormat::AV_PIX_FMT_RGBA as i32
                } else {
                    ff::AVPixelFormat::AV_PIX_FMT_BGRA as i32
                };
                if ff::av_frame_get_buffer(self.frame, 0) < 0 {
                    return Err("Failed to allocate the host frame".into());
                }
                let dst = (*self.frame).data[0];
                let dst_stride = (*self.frame).linesize[0] as usize;
                for row in 0..h {
                    ptr::copy_nonoverlapping(
                        pixels.as_ptr().add(row * stride),
                        dst.add(row * dst_stride),
                        row_bytes,
                    );
                }
                (*self.frame).pts = frame_number as i64;
                return self.encode_through_graph(frame_number, force_idr);
            }
            let ret = ff::av_frame_make_writable(self.frame);
            if ret < 0 {
                return Err(format!("Failed to make the input frame writable: {}", ff_err_str(ret)));
            }
            let i444 = self.is_fullcolor();
            let w = self.width as usize;
            let uv_rows = if i444 { h } else { h.div_ceil(2) };
            let (ys, us, vs) = (
                (*self.frame).linesize[0] as usize,
                (*self.frame).linesize[1] as usize,
                (*self.frame).linesize[2] as usize,
            );
            let y = std::slice::from_raw_parts_mut((*self.frame).data[0], ys * h);
            let u = std::slice::from_raw_parts_mut((*self.frame).data[1], us * uv_rows);
            let v = std::slice::from_raw_parts_mut((*self.frame).data[2], vs * uv_rows);
            convert_to_yuv_mt(pixels, stride as u32, w, h, rgba, i444, y, u, v, (ys, us), self.threads as usize)
                .map_err(|e| format!("rgb-to-yuv conversion failed: {e:?}"))?;
            self.encode_frame(self.frame, frame_number, force_idr)
        }
    }
}

/// The codec half of a VA-API encoder's libavcodec name.
fn vaapi_codec_name(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "h264",
        Codec::H265 => "hevc",
        Codec::Vp8 => "vp8",
        Codec::Vp9 => "vp9",
        Codec::Av1 => "av1",
        Codec::Jpeg => "mjpeg",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The planar surface wins whenever a device carries it, because the host paths hold
    /// planar 4:4:4; the packed one is taken only when it is the sole 4:4:4 format on offer,
    /// and a device carrying neither yields nothing. The names handed to `scale_vaapi` have
    /// to be the ones FFmpeg parses.
    #[test]
    fn fullcolor_surface_preference_and_names() {
        use ff::AVPixelFormat::*;
        assert_eq!(preferred_fullcolor_format(&[AV_PIX_FMT_NV12]), None);
        assert_eq!(preferred_fullcolor_format(&[AV_PIX_FMT_NV12, AV_PIX_FMT_VUYX]), Some(AV_PIX_FMT_VUYX));
        assert_eq!(preferred_fullcolor_format(&[AV_PIX_FMT_VUYX, AV_PIX_FMT_YUV444P]), Some(AV_PIX_FMT_YUV444P));
        for fmt in FULLCOLOR_SW_FORMATS {
            let name = pix_fmt_name(fmt);
            let round_trip = unsafe { ff::av_get_pix_fmt(CString::new(name.clone()).unwrap().as_ptr()) };
            assert_eq!(round_trip, fmt, "{name} did not parse back");
        }
    }

    /// Every VA-API encoder name this module can ask for is one FFmpeg registers, whether or
    /// not a device exists to run it.
    #[test]
    fn vaapi_encoder_names_are_registered() {
        for codec in Codec::VIDEO {
            let name = CString::new(format!("{}_vaapi", vaapi_codec_name(codec))).unwrap();
            assert!(
                !unsafe { ff::avcodec_find_encoder_by_name(name.as_ptr()) }.is_null(),
                "{codec:?}"
            );
        }
    }

    /// Construction either stands a session up or says why it could not; a half-built
    /// encoder must never reach a caller, and a session never quietly changes chroma. Runs
    /// everywhere: a host without a VA-API device exercises the error path.
    #[test]
    fn vaapi_construction_answers_or_refuses() {
        let mut settings = RustCaptureSettings {
            width: 128,
            height: 128,
            codec: Codec::H264,
            video_fullcolor: true,
            ..Default::default()
        };
        for codec in Codec::VIDEO {
            settings.codec = codec;
            match AvcodecEncoder::new(&settings, codec, Backend::Vaapi, Input::Host { rgba: false }) {
                Ok(enc) => assert_eq!(enc.is_fullcolor(), codec.fullcolor(), "{codec:?}"),
                Err(e) => assert!(!e.is_empty(), "refusal must carry a reason"),
            }
        }
        settings.video_fullcolor = false;
        match AvcodecEncoder::new(&settings, Codec::H264, Backend::Vaapi, Input::Host { rgba: false }) {
            Ok(enc) => assert!(!enc.is_fullcolor(), "4:2:0 session reports 4:4:4"),
            Err(e) => assert!(!e.is_empty(), "refusal must carry a reason"),
        }
        assert!(AvcodecEncoder::new(&settings, Codec::Jpeg, Backend::Software, Input::Host { rgba: false }).is_err());
        assert!(AvcodecEncoder::new(&settings, Codec::Vp8, Backend::Software, Input::Dmabuf).is_err());
    }
}

#[cfg(test)]
mod software_tests {
    //! The software sessions the linked FFmpeg carries, driven end to end: every frame they
    //! emit is wire-framed for its codec and decodes back to the picture that went in, key
    //! frames come on request and are self-contained, the quantizer and the CBR target reach
    //! the encoder, and a live rate or quality change keeps the stream decodable.
    use super::*;
    use crate::encoders::codec::{parse_video_type, FRAME_DELTA, FRAME_KEY, WIRE_VIDEO};
    use crate::webcam::convert::I420View;
    use crate::webcam::decode::{AvDecoder, Decoder};

    const W: usize = 320;
    const H: usize = 240;

    fn settings(codec: Codec) -> RustCaptureSettings {
        RustCaptureSettings {
            width: W as i32,
            height: H as i32,
            target_fps: 30.0,
            codec,
            video_crf: 25,
            use_cpu: true,
            ..Default::default()
        }
    }

    /// The video codecs this build has a software encoder for, other than H.264.
    fn software_codecs() -> Vec<Codec> {
        [Codec::H265, Codec::Vp8, Codec::Vp9, Codec::Av1]
            .into_iter()
            .filter(|&c| super::super::software_encoder(c).is_some())
            .collect()
    }

    fn session(codec: Codec, s: &RustCaptureSettings, rgba: bool) -> AvcodecEncoder {
        AvcodecEncoder::new(s, codec, Backend::Software, Input::Host { rgba })
            .unwrap_or_else(|e| panic!("{codec:?} software session: {e}"))
    }

    /// A desktop-like BGRA frame: a diagonal gradient with a grid of dark glyph cells and a
    /// bright block that moves with `t`, so inter frames carry real motion.
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
                f[i] = b;
                f[i + 1] = gr;
                f[i + 2] = r;
                f[i + 3] = 255;
            }
        }
        f
    }

    /// Incompressible content, so rate control has to spend its whole budget.
    fn noise(t: usize) -> Vec<u8> {
        let mut f = vec![255u8; W * H * 4];
        let mut s = (t as u32).wrapping_mul(2654435761).wrapping_add(7);
        for px in f.chunks_exact_mut(4) {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            px[0] = (s >> 24) as u8;
            px[1] = (s >> 16) as u8;
            px[2] = (s >> 8) as u8;
        }
        f
    }

    /// Luma PSNR of a decoded frame against the BGRA source it came from, with the source's
    /// luma derived by the BT.709 limited-range formula the encoder's conversion uses.
    fn luma_psnr(decoded: &I420View<'_>, bgra: &[u8]) -> f64 {
        assert_eq!((decoded.width, decoded.height), (W, H));
        let mut mse = 0f64;
        for y in 0..H {
            for x in 0..W {
                let i = (y * W + x) * 4;
                let (b, g, r) = (bgra[i] as f64, bgra[i + 1] as f64, bgra[i + 2] as f64);
                let luma = 16.0 + (0.2126 * r + 0.7152 * g + 0.0722 * b) * 219.0 / 255.0;
                let d = decoded.y[y * decoded.y_stride + x] as f64 - luma;
                mse += d * d;
            }
        }
        mse /= (W * H) as f64;
        if mse <= 0.0 { 99.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
    }

    fn decode_one(dec: &mut AvDecoder, packet: &[u8]) -> bool {
        dec.decode(&packet[VIDEO_HEADER_LEN..]).unwrap_or_else(|e| panic!("decode: {e:?}"))
    }

    /// Every codec's frames carry its own wire id and kind, decode back to the source picture,
    /// and a key frame forced mid-stream starts a fresh decoder on its own.
    #[test]
    fn software_frames_decode_back_to_the_source() {
        for codec in software_codecs() {
            let s = settings(codec);
            let mut enc = session(codec, &s, false);
            let mut dec = AvDecoder::new(codec).expect("decoder");
            for t in 0..6usize {
                let src = frame(t);
                let out = enc
                    .encode_host(&src, W * 4, t as u64, 25, t == 0)
                    .unwrap_or_else(|e| panic!("{codec:?} encode {t}: {e}"));
                assert!(out.len() > VIDEO_HEADER_LEN, "{codec:?} frame {t} is empty");
                assert_eq!(out[0], WIRE_VIDEO);
                let kind = if t == 0 { FRAME_KEY } else { FRAME_DELTA };
                assert_eq!(parse_video_type(out[1]), Some((codec, kind)), "{codec:?} frame {t}");
                assert_eq!(u16::from_be_bytes([out[2], out[3]]) as usize, t);
                assert_eq!(&out[4..10], &[0, 0, (W >> 8) as u8, W as u8, (H >> 8) as u8, H as u8]);
                assert!(decode_one(&mut dec, &out), "{codec:?} frame {t} decoded nothing");
                let psnr = luma_psnr(&dec.frame().unwrap(), &src);
                assert!(psnr > 28.0, "{codec:?} frame {t}: luma PSNR {psnr:.1} dB");
            }
            let src = frame(6);
            let key = enc.encode_host(&src, W * 4, 6, 25, true).expect("forced key");
            assert_eq!(parse_video_type(key[1]), Some((codec, FRAME_KEY)), "{codec:?} forced key");
            let mut fresh = AvDecoder::new(codec).expect("decoder");
            assert!(decode_one(&mut fresh, &key), "{codec:?}: a forced key frame must decode alone");
            assert!(luma_psnr(&fresh.frame().unwrap(), &src) > 28.0);
            let next = enc.encode_host(&frame(7), W * 4, 7, 25, false).expect("delta after key");
            assert_eq!(parse_video_type(next[1]), Some((codec, FRAME_DELTA)));
            assert!(decode_one(&mut fresh, &next));
        }
    }

    /// The byte order a session is built for reaches the conversion: a red picture handed as
    /// B,G,R,A and as R,G,B,A decodes to the same red on both.
    #[test]
    fn host_byte_order_is_honoured() {
        for codec in software_codecs() {
            let s = settings(codec);
            let mut means = Vec::new();
            for rgba in [false, true] {
                let mut px = [0u8; 4];
                if rgba { px[0] = 220 } else { px[2] = 220 }
                px[3] = 255;
                let src: Vec<u8> = px.repeat(W * H);
                let mut enc = session(codec, &s, rgba);
                let out = enc.encode_host(&src, W * 4, 0, 20, true).expect("encode");
                let mut dec = AvDecoder::new(codec).expect("decoder");
                assert!(decode_one(&mut dec, &out));
                let f = dec.frame().unwrap();
                let cw = f.chroma_width();
                let ch = f.chroma_height();
                let v: f64 = (0..ch).flat_map(|y| (0..cw).map(move |x| (x, y))).map(|(x, y)| f.v[y * f.uv_stride + x] as f64).sum::<f64>() / (cw * ch) as f64;
                means.push(v);
            }
            assert!(means[0] > 180.0, "{codec:?}: red must land high in V, got {:.0}", means[0]);
            assert!((means[0] - means[1]).abs() < 6.0, "{codec:?}: BGRA {:.0} vs RGBA {:.0}", means[0], means[1]);
        }
    }

    /// A higher session quality index (a coarser quantizer) shrinks the stream, and CBR holds a
    /// noise stream near its bitrate target.
    #[test]
    fn quantizer_and_bitrate_reach_the_encoder() {
        for codec in software_codecs() {
            let run = |crf: i32| -> usize {
                let mut s = settings(codec);
                s.video_crf = crf;
                let mut enc = session(codec, &s, false);
                (0..12usize)
                    .map(|t| enc.encode_host(&frame(t), W * 4, t as u64, crf as u32, t == 0).unwrap().len())
                    .sum()
            };
            let (fine, coarse) = (run(15), run(45));
            assert!(coarse * 2 < fine, "{codec:?}: crf 45 = {coarse} bytes vs crf 15 = {fine}");

            const KBPS: i32 = 800;
            let mut s = settings(codec);
            s.video_cbr_mode = true;
            s.video_bitrate_kbps = KBPS;
            let mut enc = session(codec, &s, false);
            let mut bytes = 0usize;
            for t in 0..90usize {
                let out = enc.encode_host(&noise(t), W * 4, t as u64, 25, t == 0).unwrap();
                if t >= 30 {
                    bytes += out.len() - VIDEO_HEADER_LEN;
                }
            }
            let kbps = bytes as f64 * 8.0 * 30.0 / 60.0 / 1000.0;
            println!("{codec:?} CBR {KBPS} kbps on noise: {kbps:.0} kbps");
            assert!(kbps > KBPS as f64 * 0.5 && kbps < KBPS as f64 * 1.6, "{codec:?}: {kbps:.0} kbps");
        }
    }

    /// A quality increase applies at once and keeps the stream decodable through the re-open,
    /// an increase waits out the hysteresis, and a frame-rate change re-opens the codec with
    /// the stream still decodable.
    #[test]
    fn live_quality_and_rate_changes_keep_the_stream_decodable() {
        for codec in software_codecs() {
            let mut s = settings(codec);
            s.video_crf = 40;
            let mut enc = session(codec, &s, false);
            let mut dec = AvDecoder::new(codec).expect("decoder");
            let coarse = enc.encode_host(&frame(0), W * 4, 0, 40, true).unwrap();
            assert!(decode_one(&mut dec, &coarse));
            let fine = enc.encode_host(&frame(1), W * 4, 1, 15, false).unwrap();
            assert_eq!(enc.current_qp, codec.quantizer(15), "{codec:?}: a quality increase applies at once");
            assert_eq!(parse_video_type(fine[1]), Some((codec, FRAME_KEY)), "{codec:?}: a re-open starts with a key frame");
            assert!(decode_one(&mut dec, &fine));
            let held = enc.encode_host(&frame(2), W * 4, 2, 40, false).unwrap();
            assert_eq!(enc.current_qp, codec.quantizer(15), "{codec:?}: a single decrease waits out the hysteresis");
            assert_eq!(parse_video_type(held[1]), Some((codec, FRAME_DELTA)));
            assert!(decode_one(&mut dec, &held));
            s.target_fps = 15.0;
            enc.reconfigure_rate(&s).expect("rate reconfigure");
            let after = enc.encode_host(&frame(3), W * 4, 3, 15, false).unwrap();
            assert_eq!(parse_video_type(after[1]), Some((codec, FRAME_KEY)));
            assert!(decode_one(&mut dec, &after));
            assert!(luma_psnr(&dec.frame().unwrap(), &frame(3)) > 28.0);
        }
    }

    /// 4:4:4 is carried only where the software encoder does (x265), never quietly elsewhere.
    #[test]
    fn fullcolor_follows_the_software_encoder() {
        for codec in software_codecs() {
            let mut s = settings(codec);
            s.video_fullcolor = true;
            let mut enc = session(codec, &s, false);
            assert_eq!(enc.is_fullcolor(), super::super::software_fullcolor(codec), "{codec:?}");
            let out = enc.encode_host(&frame(0), W * 4, 0, 25, true).unwrap();
            let mut dec = AvDecoder::new(codec).expect("decoder");
            assert!(decode_one(&mut dec, &out));
        }
    }
}
