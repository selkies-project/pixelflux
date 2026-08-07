/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Hardware-accelerated H.264 encoding on VA-API through FFmpeg's `h264_vaapi`
//! encoder. Frames reach the GPU by one of three entry points — a Wayland
//! DRM-PRIME dmabuf (`encode_dmabuf`), a host BGRA frame from the X11 capture
//! path (`encode_host_argb`), or already-planar NV12 pixels (`encode_raw`) — and
//! an FFmpeg filter graph runs VA-VPP (`scale_vaapi`) to land the pixels as NV12
//! in BT.709 limited range on the GPU before encode, so no colorspace conversion
//! happens on the CPU.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::mem;
use std::os::fd::AsRawFd;
use std::ptr;
use std::sync::Once;

use ffmpeg_sys_next as ff;
use libc::{close, dup};

use crate::RustCaptureSettings;
use smithay::backend::allocator::{dmabuf::Dmabuf, Buffer};

/// One-time FFmpeg global-init barrier; the closure is empty because modern FFmpeg needs
/// no explicit codec/filter registration, leaving only the run-once guarantee worth keeping.
static FF_INIT: Once = Once::new();
/// Plane/object fan-out of the `AVDRM*` descriptors, matching FFmpeg's `AV_DRM_MAX_PLANES`.
const AV_DRM_MAX_PLANES: usize = 4;
/// Damps visible quality "blinking": the number of consecutive frames a QP *increase* (a
/// quality drop under sustained motion) must be requested before `update_qp` commits it. A CQP
/// re-open is the only way to move the quantizer, so acting on every transient increase — and then
/// reversing it as motion settles — would make the picture pulse; requiring the drop to persist
/// this long absorbs those transients. Quality *decreases*, which only ever sharpen the picture,
/// apply at once and never wait.
const QP_HYSTERESIS_LIMIT: u32 = 60;

/// Exists only to mirror FFmpeg's `libavutil/hwcontext_drm.h` ABI so a Wayland dmabuf can be
/// handed to the `hwmap` filter without a copy.
///
/// FFmpeg (C) reinterprets these bytes directly, so this and its sibling `AVDRM*` descriptors carry
/// no abstraction of their own: every field, order, and `#[repr(C)]` layout must stay
/// **bit-identical** to the C definitions or the driver reads garbage. This particular struct
/// describes one backing DRM object (a dmabuf) — its fd, byte size, and DRM format modifier.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMObjectDescriptor {
    pub fd: c_int,
    pub size: usize,
    pub format_modifier: u64,
}

/// One plane within a layer: which object holds it, and the byte offset + pitch of the
/// plane inside that object.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMPlaneDescriptor {
    pub object_index: c_int,
    pub offset: isize,
    pub pitch: isize,
}

/// One layer (a single image format) built from up to `AV_DRM_MAX_PLANES` planes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMLayerDescriptor {
    pub format: u32,
    pub nb_planes: c_int,
    pub planes: [AVDRMPlaneDescriptor; AV_DRM_MAX_PLANES],
}

/// Top-level DRM frame descriptor: the set of backing objects plus the layers that index
/// into them, as consumed by FFmpeg's DRM-PRIME `hwmap`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMFrameDescriptor {
    pub nb_objects: c_int,
    pub objects: [AVDRMObjectDescriptor; AV_DRM_MAX_PLANES],
    pub nb_layers: c_int,
    pub layers: [AVDRMLayerDescriptor; AV_DRM_MAX_PLANES],
}

/// Decouples fd ownership between FFmpeg and the compositor for one in-flight frame: FFmpeg
/// closes the fds it is handed, so it must be handed `dup`'d copies rather than the originals the
/// smithay `Dmabuf` still owns. This box holds those dup'd fds and gives them a defined lifetime —
/// `release_drm_frame` closes them when FFmpeg tears the wrapping buffer down.
struct DmabufResources {
    fds: Vec<c_int>,
}

/// FFmpeg buffer-free callback for the custom DRM-PRIME frames — closes the dmabuf fds and
/// frees the descriptor.
///
/// FFmpeg (C) invokes this when it tears an `av_buffer_create` buffer down, so the whole body runs
/// inside `catch_unwind`: a panic must not unwind across the `extern "C"` boundary (that would
/// abort the process). The `opaque` pointer is reclaimed as the boxed `DmabufResources` and each
/// dup'd fd is closed. `data` is the descriptor FFmpeg now owns, freed here when non-null; it is
/// null only on the construction error path, where the caller frees the descriptor itself.
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

/// Format an FFmpeg error code as its human-readable string via `av_strerror`.
fn ff_err_str(err: i32) -> String {
    unsafe {
        let mut errbuf = [0 as c_char; 128];
        ff::av_strerror(err, errbuf.as_mut_ptr(), 128);
        CStr::from_ptr(errbuf.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

/// Hardware-accelerated H.264 encoder built on FFmpeg's `h264_vaapi`, owning the whole
/// VA-API pipeline for one capture: device contexts, the NV12 surface pool, the color-convert
/// filter graph, the reusable frames/packet, and live rate-control state.
///
/// The pointer members are raw FFmpeg objects freed in `Drop`. Three groups matter:
///
/// 1. **Device / frames contexts**: `drm_device_ctx` → derived `hw_device_ctx`; `drm_frames_ctx`
///    describes the incoming DMA-BUF, `enc_frames_ctx` the NV12 VA-surface pool the encoder draws
///    from. `enc_frames_ctx` is kept referenced so `reopen_codec` can rebuild the codec against the
///    same pool.
/// 2. **Filter graph**: `buffersrc_ctx` → hwmap/hwupload + `scale_vaapi` → `buffersink_ctx`, which
///    lands every input as GPU NV12.
/// 3. **Reusable frames**: `video_frame` feeds the graph on the dmabuf/host paths, `sw_frame` +
///    `hw_frame` stage the direct NV12 upload in `encode_raw`, and `packet` is the shared output.
///
/// `current_qp` / `qp_hysteresis_counter` drive the CQP hysteresis in `update_qp`. `cbr_mode`,
/// `current_bitrate_kbps`, `current_vbv_mult`, and `current_kf_s` cache the live rate-control state
/// so `reconfigure_rate` re-opens the codec only when a value actually changes.
/// `omit_stripe_headers` drops the 10-byte framing when the consumer wants a bare Annex-B stream.
pub struct VaapiEncoder {
    encoder_ctx: *mut ff::AVCodecContext,
    codec: *const ff::AVCodec,

    #[allow(dead_code)]
    hw_device_ctx: *mut ff::AVBufferRef,
    #[allow(dead_code)]
    drm_device_ctx: *mut ff::AVBufferRef,
    #[allow(dead_code)]
    drm_frames_ctx: *mut ff::AVBufferRef,
    
    enc_frames_ctx: *mut ff::AVBufferRef,

    filter_graph: *mut ff::AVFilterGraph,
    buffersrc_ctx: *mut ff::AVFilterContext,
    buffersink_ctx: *mut ff::AVFilterContext,

    video_frame: *mut ff::AVFrame,
    sw_frame: *mut ff::AVFrame,
    hw_frame: *mut ff::AVFrame,

    packet: *mut ff::AVPacket,

    width: i32,
    height: i32,
    fps: i32,

    current_qp: u32,
    qp_hysteresis_counter: u32,

    cbr_mode: bool,
    current_bitrate_kbps: i32,
    current_vbv_mult: f64,
    current_kf_s: f64,

    omit_stripe_headers: bool,
}

/// Assert `VaapiEncoder` is `Send`: its raw FFmpeg pointers are owned exclusively and the
/// encoder is driven from a single capture thread, so moving the whole object across threads adds
/// no aliasing.
unsafe impl Send for VaapiEncoder {}

/// Tear down every FFmpeg object in dependency order so nothing is freed while still
/// referenced: first the reusable packet and frames, then the filter graph and codec context, and
/// last the frames/device contexts they pointed at (encoder pool, DRM frames, VA-API device, DRM
/// device). Each pointer is null-checked so a partially-built encoder unwinds cleanly.
impl Drop for VaapiEncoder {
    fn drop(&mut self) {
        unsafe {
            if !self.packet.is_null() {
                ff::av_packet_free(&mut self.packet);
            }
            if !self.video_frame.is_null() {
                ff::av_frame_free(&mut self.video_frame);
            }
            if !self.sw_frame.is_null() {
                ff::av_frame_free(&mut self.sw_frame);
            }
            if !self.hw_frame.is_null() {
                ff::av_frame_free(&mut self.hw_frame);
            }

            if !self.filter_graph.is_null() {
                ff::avfilter_graph_free(&mut self.filter_graph);
            }
            if !self.encoder_ctx.is_null() {
                ff::avcodec_free_context(&mut self.encoder_ctx);
            }

            if !self.enc_frames_ctx.is_null() {
                ff::av_buffer_unref(&mut self.enc_frames_ctx);
            }
            if !self.drm_frames_ctx.is_null() {
                ff::av_buffer_unref(&mut self.drm_frames_ctx);
            }
            if !self.hw_device_ctx.is_null() {
                ff::av_buffer_unref(&mut self.hw_device_ctx);
            }
            if !self.drm_device_ctx.is_null() {
                ff::av_buffer_unref(&mut self.drm_device_ctx);
            }
        }
    }
}

impl VaapiEncoder {
    /// Build a VA-API encoder for the Wayland **dmabuf** path — the source is a DRM-PRIME
    /// dmabuf that the filter graph `hwmap`s onto a VA surface. Thin wrapper over `new_impl` with
    /// `host_input = false`.
    pub fn new(
        settings: &RustCaptureSettings,
    ) -> Result<Self, String> {
        Self::new_impl(settings, false)
    }

    /// Build a VA-API encoder for the X11 **host-ARGB** path — the source is a CPU BGRA frame
    /// that the filter graph `hwupload`s onto a VA surface. Thin wrapper over `new_impl` with
    /// `host_input = true`; the GPU still does the ARGB→NV12 convert, so there is no CPU colorspace
    /// conversion.
    pub fn new_host(
        settings: &RustCaptureSettings,
    ) -> Result<Self, String> {
        Self::new_impl(settings, true)
    }

    /// Stand up the whole VA-API pipeline for one capture, shared by the dmabuf and
    /// host-ARGB entry points and selected by `host_input`.
    ///
    /// **Why it is built this way.** Every piece here exists to push all pixel-format work onto the
    /// GPU: the capture thread should hand over a dmabuf or a raw host frame and get H.264 back
    /// without the CPU ever performing a colorspace conversion. One function serves both capture
    /// backends so their device, pool, codec, and filter-graph setup cannot drift apart; `host_input`
    /// selects only the two points where the paths genuinely differ — the buffersrc pixel format and
    /// the `hwupload` vs `hwmap` staging filter.
    ///
    /// 1. **Devices**: open a DRM device on the chosen render node (`/dev/dri/renderD{128+index}`,
    ///    or `renderD128` when no index is set) and derive a VA-API device from it.
    /// 2. **Frames contexts**: a DRM-PRIME frames context (sw-format BGRA) describes the incoming
    ///    Wayland dmabufs; the encoder frames context is a pool of `initial_pool_size = 20` VA-API
    ///    surfaces in NV12 at macroblock-aligned dimensions (width→16, height→32). A second ref to
    ///    the encoder pool is saved so `reopen_codec` can rebuild the codec against it.
    /// 3. **Codec context**: `h264_vaapi` with an effectively **infinite GOP** (`gop_size = INT_MAX`,
    ///    IDRs only on demand), **no B-frames** (low latency), **4 slices** (lets client decoders
    ///    parallelize; more than 4 upsets Chromium), and `compression_level = 6` (the VA quality
    ///    knob, biased toward speed — higher is faster). Rate control is either **CBR** (program
    ///    `bit_rate`/`rc_max_rate` and a `vbv_bits`-derived `rc_buffer_size`, `rc_mode=CBR`) or
    ///    **CQP** (`rc_mode=CQP` plus a fixed `qp`); both then pin `profile=high`, `level=4.1`,
    ///    `async_depth=1`.
    /// 4. **Filter graph**: an explicit `buffersrc` → `hwmap`/`hwupload` + `scale_vaapi` →
    ///    `buffersink` chain. The buffersrc format is DRM-PRIME (carrying the DRM frames context) for
    ///    dmabuf input, or plain BGRA for host input. `scale_vaapi` does the ARGB→NV12 convert on the
    ///    GPU in **BT.709 limited range** (`out_range=tv`) so VA-API output matches the NVENC/x264
    ///    4:2:0 color — an explicit convert is used rather than trusting encoder-side RGB CSC, which
    ///    varies across VA drivers.
    /// 5. **Graph staging**: the chain is built with the segment API (parse → create filters →
    ///    attach the VA device to every filter → apply → link our endpoints to the dangling pads)
    ///    rather than the one-shot parser. `hwupload` initializes *during* the parse and fails
    ///    without a device, and on the host path the BGRA buffersrc carries no frames context to
    ///    derive one from; staging mirrors what the ffmpeg CLI does so the device is attached before
    ///    those filters init.
    /// 6. **Preflight**: allocate the reusable `video_frame`/`sw_frame`/`hw_frame` and pull one
    ///    buffer from the encoder pool to prove the NV12 surface path is live before returning.
    ///
    /// Every failure step unwinds by unref-ing exactly the contexts allocated so far, in reverse
    /// order, so no FFmpeg object leaks on the error path.
    fn new_impl(
        settings: &RustCaptureSettings,
        host_input: bool,
    ) -> Result<Self, String> {
        FF_INIT.call_once(|| {});

        let width = settings.width;
        let height = settings.height;
        // AVRational.den must stay nonzero: a sub-1 fps setting would truncate to 0.
        let fps = (settings.target_fps as i32).max(1);

        unsafe {
            let mut drm_device_ctx: *mut ff::AVBufferRef = ptr::null_mut();
            let render_node = if settings.encode_node_index >= 0 {
                format!("/dev/dri/renderD{}", 128 + settings.encode_node_index)
            } else {
                "/dev/dri/renderD128".to_string()
            };
            let device_url = CString::new(render_node).unwrap();

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

            let mut hw_device_ctx: *mut ff::AVBufferRef = ptr::null_mut();
            let ret = ff::av_hwdevice_ctx_create_derived(
                &mut hw_device_ctx,
                ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                drm_device_ctx,
                0,
            );
            if ret < 0 {
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err(format!(
                    "Failed to derive VAAPI device: {}",
                    ff_err_str(ret)
                ));
            }

            let mut drm_frames_ref = ff::av_hwframe_ctx_alloc(drm_device_ctx);
            if drm_frames_ref.is_null() {
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to alloc DRM frames ctx".into());
            }

            let drm_frames = (*drm_frames_ref).data as *mut ff::AVHWFramesContext;
            (*drm_frames).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME;
            (*drm_frames).sw_format = ff::AVPixelFormat::AV_PIX_FMT_BGRA;
            (*drm_frames).width = width;
            (*drm_frames).height = height;
            (*drm_frames).initial_pool_size = 0;

            if ff::av_hwframe_ctx_init(drm_frames_ref) < 0 {
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to init DRM frames ctx".into());
            }

            let codec_name = CString::new("h264_vaapi").unwrap();
            let codec = ff::avcodec_find_encoder_by_name(codec_name.as_ptr());
            if codec.is_null() {
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("h264_vaapi encoder not found".into());
            }

            let aligned_width = (width + 15) & !15;
            let aligned_height = (height + 31) & !31;

            let mut enc_frames_ref = ff::av_hwframe_ctx_alloc(hw_device_ctx);
            if enc_frames_ref.is_null() {
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to allocate encoder frames ctx".into());
            }
            let enc_frames = (*enc_frames_ref).data as *mut ff::AVHWFramesContext;
            (*enc_frames).format = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
            (*enc_frames).sw_format = ff::AVPixelFormat::AV_PIX_FMT_NV12;
            (*enc_frames).width = aligned_width;
            (*enc_frames).height = aligned_height;
            (*enc_frames).initial_pool_size = 20;

            if ff::av_hwframe_ctx_init(enc_frames_ref) < 0 {
                ff::av_buffer_unref(&mut enc_frames_ref);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to init encoder frames ctx".into());
            }

            let mut saved_enc_frames_ctx = ff::av_buffer_ref(enc_frames_ref);

            let mut encoder_ctx = ff::avcodec_alloc_context3(codec);
            if encoder_ctx.is_null() {
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut enc_frames_ref);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to allocate encoder context".into());
            }
            (*encoder_ctx).width = width;
            (*encoder_ctx).height = height;
            (*encoder_ctx).time_base = ff::AVRational { num: 1, den: fps };
            (*encoder_ctx).framerate = ff::AVRational { num: fps, den: 1 };
            (*encoder_ctx).pix_fmt = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
            (*encoder_ctx).hw_device_ctx = ff::av_buffer_ref(hw_device_ctx);
            (*encoder_ctx).hw_frames_ctx = ff::av_buffer_ref(enc_frames_ref);
            (*encoder_ctx).max_b_frames = 0;
            (*encoder_ctx).gop_size = std::ffi::c_int::MAX;
            (*encoder_ctx).slices = 4;
            (*encoder_ctx).compression_level = 6;

            ff::av_buffer_unref(&mut enc_frames_ref);

            let mut opts: *mut ff::AVDictionary = ptr::null_mut();
            let set_opt = |d: &mut *mut ff::AVDictionary, k: &str, v: &str| {
                let ck = CString::new(k).unwrap();
                let cv = CString::new(v).unwrap();
                ff::av_dict_set(d, ck.as_ptr(), cv.as_ptr(), 0);
            };

            if settings.video_cbr_mode {
                let bps = (settings.video_bitrate_kbps.max(0) as i64).saturating_mul(1000);
                let vbv = crate::encoders::vbv_bits(
                    bps.min(u32::MAX as i64) as u32,
                    settings.target_fps,
                    settings.keyframe_interval_s,
                    settings.video_vbv_multiplier,
                );
                (*encoder_ctx).bit_rate = bps;
                (*encoder_ctx).rc_max_rate = bps;
                (*encoder_ctx).rc_buffer_size = vbv.min(i32::MAX as u32) as i32;
                set_opt(&mut opts, "rc_mode", "CBR");
            } else {
                set_opt(&mut opts, "rc_mode", "CQP");
                set_opt(&mut opts, "qp", &settings.video_crf.to_string());
            }
            set_opt(&mut opts, "async_depth", "1");
            set_opt(&mut opts, "profile", "high");
            set_opt(
                &mut opts,
                "level",
                &super::min_h264_level(width as u32, height as u32, fps as u32).to_string(),
            );

            let ret = ff::avcodec_open2(encoder_ctx, codec, &mut opts);
            ff::av_dict_free(&mut opts);
            if ret < 0 {
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err(format!("Failed to open encoder: {}", ff_err_str(ret)));
            }

            let mut filter_graph = ff::avfilter_graph_alloc();
            let buffersrc = ff::avfilter_get_by_name(CString::new("buffer").unwrap().as_ptr());
            let buffersink =
                ff::avfilter_get_by_name(CString::new("buffersink").unwrap().as_ptr());
            let name_in = CString::new("in").unwrap();
            let name_out = CString::new("out").unwrap();

            let buffersrc_ctx =
                ff::avfilter_graph_alloc_filter(filter_graph, buffersrc, name_in.as_ptr());

            let par = ff::av_buffersrc_parameters_alloc();
            if par.is_null() {
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to alloc buffersrc parameters".into());
            }
            if host_input {
                (*par).format = ff::AVPixelFormat::AV_PIX_FMT_BGRA as i32;
            } else {
                (*par).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32;
                (*par).hw_frames_ctx = ff::av_buffer_ref(drm_frames_ref);
            }
            (*par).width = width;
            (*par).height = height;
            (*par).time_base = ff::AVRational { num: 1, den: fps };

            let ret = ff::av_buffersrc_parameters_set(buffersrc_ctx, par);
            if !(*par).hw_frames_ctx.is_null() {
                ff::av_buffer_unref(&mut (*par).hw_frames_ctx);
            }
            ff::av_free(par as *mut c_void);
            if ret < 0 {
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err(format!(
                    "Failed to set buffersrc parameters: {}",
                    ff_err_str(ret)
                ));
            }

            let args_str = format!(
                "video_size={}x{}:time_base=1/{}:pixel_aspect=1/1",
                width, height, fps
            );
            let args = CString::new(args_str).unwrap();
            if ff::avfilter_init_str(buffersrc_ctx, args.as_ptr()) < 0 {
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to init buffersrc".into());
            }

            let mut buffersink_ctx: *mut ff::AVFilterContext = ptr::null_mut();
            if ff::avfilter_graph_create_filter(
                &mut buffersink_ctx,
                buffersink,
                name_out.as_ptr(),
                ptr::null(),
                ptr::null_mut(),
                filter_graph,
            ) < 0
            {
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to create buffersink".into());
            }

            let stage = if host_input { "hwupload" } else { "hwmap" };
            let filters_desc = CString::new(format!(
                "{},scale_vaapi=w={}:h={}:format=nv12:out_color_matrix=bt709:out_range=tv",
                stage, width, height
            ))
            .unwrap();
            let mut seg: *mut ff::AVFilterGraphSegment = ptr::null_mut();
            let mut seg_inputs: *mut ff::AVFilterInOut = ptr::null_mut();
            let mut seg_outputs: *mut ff::AVFilterInOut = ptr::null_mut();
            let seg_ok = ff::avfilter_graph_segment_parse(
                filter_graph,
                filters_desc.as_ptr(),
                0,
                &mut seg,
            ) >= 0
                && ff::avfilter_graph_segment_create_filters(seg, 0) >= 0
                && {
                    for i in 0..(*filter_graph).nb_filters {
                        let f = *(*filter_graph).filters.add(i as usize);
                        if (*f).hw_device_ctx.is_null() {
                            (*f).hw_device_ctx = ff::av_buffer_ref(hw_device_ctx);
                        }
                    }
                    ff::avfilter_graph_segment_apply(seg, 0, &mut seg_inputs, &mut seg_outputs)
                        >= 0
                }
                && !seg_inputs.is_null()
                && !seg_outputs.is_null()
                && ff::avfilter_link(
                    buffersrc_ctx,
                    0,
                    (*seg_inputs).filter_ctx,
                    (*seg_inputs).pad_idx as u32,
                ) >= 0
                && ff::avfilter_link(
                    (*seg_outputs).filter_ctx,
                    (*seg_outputs).pad_idx as u32,
                    buffersink_ctx,
                    0,
                ) >= 0;
            ff::avfilter_inout_free(&mut seg_inputs);
            ff::avfilter_inout_free(&mut seg_outputs);
            ff::avfilter_graph_segment_free(&mut seg);
            if !seg_ok {
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to build filter graph".into());
            }

            if ff::avfilter_graph_config(filter_graph, ptr::null_mut()) < 0 {
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to config filter graph".into());
            }

            let mut video_frame = ff::av_frame_alloc();
            let mut sw_frame = ff::av_frame_alloc();
            let mut hw_frame = ff::av_frame_alloc();

            if ff::av_hwframe_get_buffer((*encoder_ctx).hw_frames_ctx, hw_frame, 0) < 0 {
                ff::av_frame_free(&mut hw_frame);
                ff::av_frame_free(&mut sw_frame);
                ff::av_frame_free(&mut video_frame);
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to allocate HW frame for NV12 path".into());
            }

            Ok(Self {
                encoder_ctx,
                codec,
                hw_device_ctx,
                drm_device_ctx,
                drm_frames_ctx: drm_frames_ref,
                enc_frames_ctx: saved_enc_frames_ctx,
                filter_graph,
                buffersrc_ctx,
                buffersink_ctx,
                video_frame,
                sw_frame,
                hw_frame,
                packet: ff::av_packet_alloc(),
                width,
                height,
                fps,
                current_qp: settings.video_crf as u32,
                qp_hysteresis_counter: 0,
                cbr_mode: settings.video_cbr_mode,
                current_bitrate_kbps: settings.video_bitrate_kbps,
                current_vbv_mult: settings.video_vbv_multiplier,
                current_kf_s: settings.keyframe_interval_s,
                omit_stripe_headers: settings.omit_stripe_headers,
            })
        }
    }

    /// Applies every live QP / bitrate / fps change by re-opening the whole codec context,
    /// because VA-API drivers do not reliably honor an in-place reconfigure — a fresh
    /// `AVCodecContext` is the one portable way to make a new rate-control setting actually take
    /// effect.
    ///
    /// The price of that reliability is that a re-opened context always emits an IDR as its first
    /// frame, which is affordable here precisely because that IDR simply re-anchors the reference
    /// chain, so the stream self-heals across the swap rather than breaking. Only the codec context
    /// is torn down and rebuilt: the VA device, the encoder frames pool, and the filter graph all
    /// persist, and the new context is re-allocated against the same `codec` and re-bound to the
    /// saved `enc_frames_ctx` pool and `hw_device_ctx`, with the same GOP / slice / compression
    /// settings as the initial open. **CBR** reprograms `bit_rate` / `rc_max_rate` / `rc_buffer_size`
    /// (VBV from `vbv_bits`); **CQP** reprograms the quantizer to `qp` and records it in `current_qp`.
    unsafe fn reopen_codec(&mut self, qp: u32) -> Result<(), String> {
        if !self.encoder_ctx.is_null() {
            ff::avcodec_free_context(&mut self.encoder_ctx);
        }

        self.encoder_ctx = ff::avcodec_alloc_context3(self.codec);
        if self.encoder_ctx.is_null() {
            return Err("Failed to re-alloc encoder context".into());
        }

        (*self.encoder_ctx).width = self.width;
        (*self.encoder_ctx).height = self.height;
        (*self.encoder_ctx).time_base = ff::AVRational { num: 1, den: self.fps };
        (*self.encoder_ctx).framerate = ff::AVRational { num: self.fps, den: 1 };
        (*self.encoder_ctx).pix_fmt = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
        (*self.encoder_ctx).hw_device_ctx = ff::av_buffer_ref(self.hw_device_ctx);
        (*self.encoder_ctx).hw_frames_ctx = ff::av_buffer_ref(self.enc_frames_ctx);
        (*self.encoder_ctx).max_b_frames = 0;
        (*self.encoder_ctx).gop_size = std::ffi::c_int::MAX;
        (*self.encoder_ctx).slices = 4;
        (*self.encoder_ctx).compression_level = 6;

        let mut opts: *mut ff::AVDictionary = ptr::null_mut();
        let set_opt = |d: &mut *mut ff::AVDictionary, k: &str, v: &str| {
            let ck = CString::new(k).unwrap();
            let cv = CString::new(v).unwrap();
            ff::av_dict_set(d, ck.as_ptr(), cv.as_ptr(), 0);
        };

        if self.cbr_mode {
            let bps = (self.current_bitrate_kbps.max(0) as i64).saturating_mul(1000);
            let vbv = crate::encoders::vbv_bits(
                bps.min(u32::MAX as i64) as u32,
                self.fps.max(1) as f64,
                self.current_kf_s,
                self.current_vbv_mult,
            );
            (*self.encoder_ctx).bit_rate = bps;
            (*self.encoder_ctx).rc_max_rate = bps;
            (*self.encoder_ctx).rc_buffer_size = vbv.min(i32::MAX as u32) as i32;
            set_opt(&mut opts, "rc_mode", "CBR");
        } else {
            set_opt(&mut opts, "rc_mode", "CQP");
            set_opt(&mut opts, "qp", &qp.to_string());
        }
        set_opt(&mut opts, "async_depth", "1");
        set_opt(&mut opts, "profile", "high");
        set_opt(
            &mut opts,
            "level",
            &super::min_h264_level(self.width as u32, self.height as u32, self.fps.max(1) as u32)
                .to_string(),
        );

        let ret = ff::avcodec_open2(self.encoder_ctx, self.codec, &mut opts);
        ff::av_dict_free(&mut opts);

        if ret < 0 {
            return Err(format!("Failed to re-open encoder: {}", ff_err_str(ret)));
        }

        self.current_qp = qp;
        Ok(())
    }

    /// Moves the CQP quantizer toward `target_qp`, but weighs each change against the cost of
    /// acting on it, because applying a quantizer is not free — it forces a codec re-open (and thus
    /// an IDR) — and the two directions are not equally worth that cost.
    ///
    /// A *decrease* sharpens the picture and is always worth an immediate switch; an *increase* dulls
    /// it and is worth committing to only once the drop has clearly persisted, so transient motion
    /// does not make quality blink. That asymmetry is the whole point of the hysteresis. Resolved in
    /// priority order:
    ///
    /// 1. **CBR**: no-op — the bitrate target, not the quantizer, governs quality, so re-opening per
    ///    frame would only flip the context into CQP and abandon the configured bitrate.
    /// 2. **Unchanged QP**: reset the hysteresis counter and return.
    /// 3. **QP decrease** (higher quality, e.g. a paint-over refresh): apply immediately via
    ///    `reopen_codec` — a sharper static image is always worth the switch.
    /// 4. **QP increase** (lower quality, e.g. sustained motion): count consecutive requests and only
    ///    re-open once the counter exceeds `QP_HYSTERESIS_LIMIT`, so brief motion does not make
    ///    quality visibly blink between re-opens.
    unsafe fn update_qp(&mut self, target_qp: u32) -> Result<(), String> {
        if self.cbr_mode {
            return Ok(());
        }

        if target_qp == self.current_qp {
            self.qp_hysteresis_counter = 0;
            return Ok(());
        }

        if target_qp < self.current_qp {
            self.qp_hysteresis_counter = 0;
            self.reopen_codec(target_qp)?;
        } else {
            self.qp_hysteresis_counter += 1;
            if self.qp_hysteresis_counter > QP_HYSTERESIS_LIMIT {
                self.qp_hysteresis_counter = 0;
                self.reopen_codec(target_qp)?;
            }
        }

        Ok(())
    }

    /// Stays cheap enough for the pipeline to call on every single frame by re-opening the
    /// codec only when a rate-control or framerate setting has actually changed — an unconditional
    /// re-open here would force a needless IDR every frame and cripple the stream.
    ///
    /// The change test is deliberately narrow, to keep that guard tight: in CBR, a different target
    /// bitrate or VBV multiplier; in any mode, a different target fps. When nothing changed it returns
    /// without touching the codec. On a real change it caches the new fps / bitrate / VBV / keyframe
    /// interval and calls `reopen_codec` carrying the **current** QP, so a CQP stream keeps its
    /// quantizer across the change.
    pub fn reconfigure_rate(&mut self, settings: &RustCaptureSettings) {
        unsafe {
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
                return;
            }
            self.fps = new_fps;
            self.current_bitrate_kbps = settings.video_bitrate_kbps;
            self.current_vbv_mult = settings.video_vbv_multiplier;
            self.current_kf_s = settings.keyframe_interval_s;
            if let Err(e) = self.reopen_codec(self.current_qp) {
                eprintln!("[vaapi] rate reconfigure failed: {e}");
            }
        }
    }

    /// Feeds each finished packet to two consumers that need different framing, which is why
    /// this is more than a plain receive loop.
    ///
    /// The network `output` gets each packet wrapped in the 10-byte stripe header the client demuxer
    /// expects, so a full frame can ride the exact same path as the striped modes — it is simply
    /// described to the client as one full-height stripe: tag `0x04`, a keyframe flag (`0x01`/`0x00`
    /// from `AV_PKT_FLAG_KEY`), the frame number as a big-endian `u16`, a `0` y-start (a whole frame
    /// starts at the top), then width and height as big-endian `u16`s, followed by the raw Annex-B
    /// payload.
    /// `omit_stripe_headers` drops the header on the network path too, for a consumer that already
    /// wants raw Annex-B. The loop drains `avcodec_receive_packet` until the encoder is empty,
    /// unref'ing each packet before the next iteration.
    unsafe fn collect_packet(&mut self, frame_number: u64, output: &mut Vec<u8>) {
        while ff::avcodec_receive_packet(self.encoder_ctx, self.packet) == 0 {
            let size = (*self.packet).size as usize;
            let data = (*self.packet).data;
            let is_key = ((*self.packet).flags & ff::AV_PKT_FLAG_KEY) != 0;

            let header_sz = if self.omit_stripe_headers { 0 } else { 10 };
            output.reserve(header_sz + size);
            if !self.omit_stripe_headers {
                output.push(0x04);
                output.push(if is_key { 0x01 } else { 0x00 });
                output.extend_from_slice(&(frame_number as u16).to_be_bytes());
                output.extend_from_slice(&0u16.to_be_bytes());
                output.extend_from_slice(&(self.width as u16).to_be_bytes());
                output.extend_from_slice(&(self.height as u16).to_be_bytes());
            }

            let slice = std::slice::from_raw_parts(data, size);
            output.extend_from_slice(slice);

            ff::av_packet_unref(self.packet);
        }
    }

    /// Encode one Wayland DRM-PRIME dmabuf by wrapping it in an FFmpeg DRM frame descriptor
    /// and pushing it through the filter graph, which `hwmap`s it to a VA surface and converts to
    /// NV12 before encode.
    ///
    /// 1. **Quantizer**: apply the requested `qp` through `update_qp` (hysteresis / CBR-aware).
    /// 2. **Descriptor**: allocate a zeroed `AVDRMFrameDescriptor` and populate it from the dmabuf —
    ///    one object per handle with a freshly `dup`'d fd (so FFmpeg owns independent fds), the object
    ///    size computed from the plane stride and the height rounded up to 32, the format modifier,
    ///    and one layer whose planes carry each plane's offset and pitch. A single-handle multi-plane
    ///    buffer points all planes at object 0.
    /// 3. **Ownership**: the dup'd fds live in a boxed `DmabufResources` handed to `av_buffer_create`
    ///    as `release_drm_frame`'s opaque, so FFmpeg closes them on teardown. If building that buffer
    ///    fails, `release_drm_frame` is called directly to clean up.
    /// 4. **Submit**: point `video_frame` at the descriptor, tag it DRM-PRIME with the DRM frames
    ///    context, and feed the graph. `av_buffersrc_add_frame` consumes the frame's refs only on
    ///    success; on error the frame is untouched, so `video_frame` is unref'd to release `buf[0]`
    ///    (which runs `release_drm_frame` and closes the fds) — no manual fd close, which would
    ///    double-close.
    /// 5. **Collect**: pull each converted frame from the sink, stamp `pict_type = I` when
    ///    `force_idr`, send it to the encoder, and drain packets via `collect_packet`.
    pub fn encode_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        frame_number: u64,
        qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.update_qp(qp)?;

            let desc_size = mem::size_of::<AVDRMFrameDescriptor>();
            let desc_ptr = ff::av_mallocz(desc_size) as *mut AVDRMFrameDescriptor;
            if desc_ptr.is_null() {
                return Err("OOM".into());
            }

            let mut resources = DmabufResources { fds: Vec::new() };
            let strides: Vec<u32> = dmabuf.strides().collect();

            (*desc_ptr).nb_objects = dmabuf.handles().count() as i32;
            (*desc_ptr).nb_layers = 1;

            for (i, (handle, _)) in dmabuf.handles().zip(dmabuf.offsets()).enumerate() {
                let fd = dup(handle.as_raw_fd());
                if fd < 0 {
                    for &dup_fd in &resources.fds {
                        close(dup_fd);
                    }
                    ff::av_free(desc_ptr as *mut c_void);
                    return Err("Failed to dup fd".into());
                }
                resources.fds.push(fd);
                (*desc_ptr).objects[i].fd = fd;
                
                let stride = strides.get(i).copied().unwrap_or(strides[0]);
                let aligned_height = (self.height + 31) & !31;
                (*desc_ptr).objects[i].size = (stride as usize) * (aligned_height as usize);
                
                (*desc_ptr).objects[i].format_modifier = u64::from(dmabuf.format().modifier);
            }

            (*desc_ptr).layers[0].format = dmabuf.format().code as u32;
            (*desc_ptr).layers[0].nb_planes = dmabuf.num_planes() as i32;

            for (i, (stride, offset)) in dmabuf.strides().zip(dmabuf.offsets()).enumerate() {
                (*desc_ptr).layers[0].planes[i].object_index = i as i32;
                (*desc_ptr).layers[0].planes[i].offset = offset as isize;
                (*desc_ptr).layers[0].planes[i].pitch = stride as isize;
            }

            if dmabuf.handles().count() == 1 && dmabuf.num_planes() > 1 {
                for i in 0..dmabuf.num_planes() {
                    (*desc_ptr).layers[0].planes[i].object_index = 0;
                }
            }

            ff::av_frame_unref(self.video_frame);
            (*self.video_frame).width = self.width;
            (*self.video_frame).height = self.height;
            (*self.video_frame).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32;
            (*self.video_frame).data[0] = desc_ptr as *mut u8;

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
            (*self.video_frame).buf[0] = buf_ref;
            (*self.video_frame).pts = frame_number as i64;
            (*self.video_frame).hw_frames_ctx = ff::av_buffer_ref(self.drm_frames_ctx);

            if ff::av_buffersrc_add_frame(self.buffersrc_ctx, self.video_frame) < 0 {
                ff::av_frame_unref(self.video_frame);
                return Err("Failed to feed filter graph".into());
            }

            let mut output = Vec::new();
            let mut filtered_frame = ff::av_frame_alloc();

            while ff::av_buffersink_get_frame(self.buffersink_ctx, filtered_frame) >= 0 {
                if force_idr {
                    (*filtered_frame).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_I;
                }

                if ff::avcodec_send_frame(self.encoder_ctx, filtered_frame) < 0 {
                    ff::av_frame_free(&mut filtered_frame);
                    return Err("Failed to send frame to encoder".into());
                }
                ff::av_frame_unref(filtered_frame);

                self.collect_packet(frame_number, &mut output);
            }
            ff::av_frame_free(&mut filtered_frame);

            Ok(output)
        }
    }

    /// Encode one host BGRA frame (X11 path) by staging it onto a VA surface and letting the
    /// GPU do the color convert — valid only on an encoder built with `new_host`.
    ///
    /// `bgra` is B,G,R,A in memory at `stride` bytes per row (padding allowed) and must hold at least
    /// `stride * height` bytes. The flow: apply `qp` via `update_qp`; allocate a fresh refcounted
    /// BGRA `video_frame` and copy the host rows in (the sole copy — plain data movement to a
    /// GPU-uploadable frame, clamped to `min(width*4, stride, dst_stride)` per row, **not** a
    /// colorspace conversion); feed the graph, where `hwupload` stages it onto a VA surface and
    /// `scale_vaapi` converts ARGB→NV12; then pull converted frames, stamp `pict_type = I` when
    /// `force_idr`, encode, and drain packets via `collect_packet`.
    pub fn encode_host_argb(
        &mut self,
        bgra: &[u8],
        stride: usize,
        frame_number: u64,
        qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.update_qp(qp)?;

            let h = self.height as usize;
            let needed = stride.checked_mul(h).ok_or("stride overflow")?;
            if bgra.len() < needed {
                return Err("Input buffer too small".into());
            }

            ff::av_frame_unref(self.video_frame);
            (*self.video_frame).width = self.width;
            (*self.video_frame).height = self.height;
            (*self.video_frame).format = ff::AVPixelFormat::AV_PIX_FMT_BGRA as i32;
            if ff::av_frame_get_buffer(self.video_frame, 0) < 0 {
                return Err("Failed to allocate host BGRA frame".into());
            }
            let dst = (*self.video_frame).data[0];
            let dst_stride = (*self.video_frame).linesize[0] as usize;
            let row_bytes = (self.width as usize) * 4;
            let copy_bytes = row_bytes.min(stride).min(dst_stride);
            for row in 0..h {
                ptr::copy_nonoverlapping(
                    bgra.as_ptr().add(row * stride),
                    dst.add(row * dst_stride),
                    copy_bytes,
                );
            }
            (*self.video_frame).pts = frame_number as i64;

            if ff::av_buffersrc_add_frame(self.buffersrc_ctx, self.video_frame) < 0 {
                ff::av_frame_unref(self.video_frame);
                return Err("Failed to feed filter graph".into());
            }

            let mut output = Vec::new();
            let mut filtered_frame = ff::av_frame_alloc();
            while ff::av_buffersink_get_frame(self.buffersink_ctx, filtered_frame) >= 0 {
                if force_idr {
                    (*filtered_frame).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_I;
                }
                if ff::avcodec_send_frame(self.encoder_ctx, filtered_frame) < 0 {
                    ff::av_frame_free(&mut filtered_frame);
                    return Err("Failed to send frame to encoder".into());
                }
                ff::av_frame_unref(filtered_frame);
                self.collect_packet(frame_number, &mut output);
            }
            ff::av_frame_free(&mut filtered_frame);

            Ok(output)
        }
    }

    /// Encode already-planar NV12 pixels by uploading them straight to a VA surface,
    /// bypassing the filter graph (there is no color convert to do).
    ///
    /// Wraps the input as an NV12 `sw_frame` — Y plane at offset 0, interleaved UV at `width*height`,
    /// both with `linesize = width` — after checking the buffer holds the full `w*h + w*h/2` bytes.
    /// A fresh `hw_frame` is pulled from the encoder pool (the frame is unref'd first, or the prior
    /// surface would leak), `av_hwframe_transfer_data` copies CPU→GPU, and the `sw_frame` is released.
    /// Keyframes are forced through `pict_type = I` (`AV_PKT_FLAG_KEY` is a *packet* flag, unusable on
    /// a frame); otherwise `pict_type = NONE`. The GPU frame is sent to the encoder and packets are
    /// drained via `collect_packet`.
    pub fn encode_raw(
        &mut self,
        nv12_pixels: &[u8],
        frame_number: u64,
        qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.update_qp(qp)?;

            let width = self.width as usize;
            let height = self.height as usize;
            let required_size = width * height + (width * height / 2);

            if nv12_pixels.len() < required_size {
                return Err("Input buffer too small".into());
            }

            ff::av_frame_unref(self.sw_frame);
            (*self.sw_frame).format = ff::AVPixelFormat::AV_PIX_FMT_NV12 as i32;
            (*self.sw_frame).width = self.width;
            (*self.sw_frame).height = self.height;

            (*self.sw_frame).data[0] = nv12_pixels.as_ptr() as *mut u8;
            (*self.sw_frame).linesize[0] = self.width;

            (*self.sw_frame).data[1] = nv12_pixels.as_ptr().add(width * height) as *mut u8;
            (*self.sw_frame).linesize[1] = self.width;

            ff::av_frame_unref(self.hw_frame);
            if ff::av_hwframe_get_buffer((*self.encoder_ctx).hw_frames_ctx, self.hw_frame, 0) < 0 {
                return Err("Failed to allocate HW frame for NV12 path".into());
            }
            (*self.hw_frame).width = self.width;
            (*self.hw_frame).height = self.height;

            if ff::av_hwframe_transfer_data(self.hw_frame, self.sw_frame, 0) < 0 {
                return Err("Failed to upload frame to GPU".into());
            }

            ff::av_frame_unref(self.sw_frame);

            (*self.hw_frame).pts = frame_number as i64;
            if force_idr {
                (*self.hw_frame).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_I;
            } else {
                (*self.hw_frame).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_NONE;
            }

            if ff::avcodec_send_frame(self.encoder_ctx, self.hw_frame) < 0 {
                return Err("Error sending frame to encoder".into());
            }

            let mut output = Vec::new();
            self.collect_packet(frame_number, &mut output);

            Ok(output)
        }
    }
}

#[cfg(test)]
mod graph_ordering_tests {
    use super::*;

    /// Build the same shape of graph as `new_impl`'s host path — explicit buffersrc/sink
    /// endpoints around an `hwupload,…` chain — against a CUDA device, to exercise the device-attach
    /// ordering without needing a VA-API device.
    ///
    /// `hwupload` is the generic filter VA-API also uses, so CUDA stands in on a box that has no VA
    /// device. `staged = true` builds via the segment API (parse → create filters → attach device →
    /// init/link); `staged = false` uses the one-shot `avfilter_graph_parse_ptr` with the device
    /// attached only afterwards. The helper then configures the graph and pushes one 64×64 BGRA frame
    /// through it, returning `Err` unless a frame actually flows out — proving the graph passes
    /// pixels, not merely that it configured.
    unsafe fn build_hwupload_graph(staged: bool) -> Result<(), String> {
        let mut dev: *mut ff::AVBufferRef = ptr::null_mut();
        if ff::av_hwdevice_ctx_create(
            &mut dev,
            ff::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
            ptr::null(),
            ptr::null_mut(),
            0,
        ) < 0
        {
            return Err("no CUDA device".into());
        }

        let graph = ff::avfilter_graph_alloc();
        let mut src: *mut ff::AVFilterContext = ptr::null_mut();
        let args = CString::new("video_size=64x64:pix_fmt=bgra:time_base=1/30").unwrap();
        let r = ff::avfilter_graph_create_filter(
            &mut src,
            ff::avfilter_get_by_name(CString::new("buffer").unwrap().as_ptr()),
            CString::new("in").unwrap().as_ptr(),
            args.as_ptr(),
            ptr::null_mut(),
            graph,
        );
        assert!(r >= 0, "buffersrc create");
        let mut sink: *mut ff::AVFilterContext = ptr::null_mut();
        let r = ff::avfilter_graph_create_filter(
            &mut sink,
            ff::avfilter_get_by_name(CString::new("buffersink").unwrap().as_ptr()),
            CString::new("out").unwrap().as_ptr(),
            ptr::null(),
            ptr::null_mut(),
            graph,
        );
        assert!(r >= 0, "buffersink create");

        let desc = CString::new("hwupload,hwdownload,format=bgra").unwrap();
        let result: Result<(), String> = if staged {
            let mut seg: *mut ff::AVFilterGraphSegment = ptr::null_mut();
            let mut ins: *mut ff::AVFilterInOut = ptr::null_mut();
            let mut outs: *mut ff::AVFilterInOut = ptr::null_mut();
            let ok = ff::avfilter_graph_segment_parse(graph, desc.as_ptr(), 0, &mut seg) >= 0
                && ff::avfilter_graph_segment_create_filters(seg, 0) >= 0
                && {
                    for i in 0..(*graph).nb_filters {
                        let f = *(*graph).filters.add(i as usize);
                        if (*f).hw_device_ctx.is_null() {
                            (*f).hw_device_ctx = ff::av_buffer_ref(dev);
                        }
                    }
                    ff::avfilter_graph_segment_apply(seg, 0, &mut ins, &mut outs) >= 0
                }
                && !ins.is_null()
                && !outs.is_null()
                && ff::avfilter_link(src, 0, (*ins).filter_ctx, (*ins).pad_idx as u32) >= 0
                && ff::avfilter_link((*outs).filter_ctx, (*outs).pad_idx as u32, sink, 0) >= 0;
            ff::avfilter_inout_free(&mut ins);
            ff::avfilter_inout_free(&mut outs);
            ff::avfilter_graph_segment_free(&mut seg);
            if ok { Ok(()) } else { Err("segment build failed".into()) }
        } else {
            let mut inputs = ff::avfilter_inout_alloc();
            let mut outputs = ff::avfilter_inout_alloc();
            (*inputs).name = ff::av_strdup(CString::new("in").unwrap().as_ptr());
            (*inputs).filter_ctx = src;
            (*inputs).pad_idx = 0;
            (*inputs).next = ptr::null_mut();
            (*outputs).name = ff::av_strdup(CString::new("out").unwrap().as_ptr());
            (*outputs).filter_ctx = sink;
            (*outputs).pad_idx = 0;
            (*outputs).next = ptr::null_mut();
            let r = ff::avfilter_graph_parse_ptr(
                graph,
                desc.as_ptr(),
                &mut outputs,
                &mut inputs,
                ptr::null_mut(),
            );
            ff::avfilter_inout_free(&mut inputs);
            ff::avfilter_inout_free(&mut outputs);
            if r < 0 {
                Err(format!("parse failed before the attach loop could run: {}", ff_err_str(r)))
            } else {
                for i in 0..(*graph).nb_filters {
                    let f = *(*graph).filters.add(i as usize);
                    if (*f).hw_device_ctx.is_null() {
                        (*f).hw_device_ctx = ff::av_buffer_ref(dev);
                    }
                }
                Ok(())
            }
        };

        let result = result.and_then(|()| {
            let r = ff::avfilter_graph_config(graph, ptr::null_mut());
            if r < 0 { Err(format!("config failed: {}", ff_err_str(r))) } else { Ok(()) }
        });

        let result = result.and_then(|()| {
            let frame = ff::av_frame_alloc();
            (*frame).format = ff::AVPixelFormat::AV_PIX_FMT_BGRA as i32;
            (*frame).width = 64;
            (*frame).height = 64;
            if ff::av_frame_get_buffer(frame, 0) < 0 {
                return Err("frame alloc".into());
            }
            for y in 0..64 {
                let row = (*frame).data[0].add(y * (*frame).linesize[0] as usize);
                std::ptr::write_bytes(row, 0x80, 64 * 4);
            }
            let mut fr = frame;
            let ok = ff::av_buffersrc_add_frame(src, fr) >= 0 && {
                let out = ff::av_frame_alloc();
                let got = ff::av_buffersink_get_frame(sink, out) >= 0
                    && (*out).width == 64
                    && !(*out).data[0].is_null();
                let mut o = out;
                ff::av_frame_free(&mut o);
                got
            };
            ff::av_frame_free(&mut fr);
            if ok { Ok(()) } else { Err("frame did not flow through the graph".into()) }
        });

        let mut g = graph;
        ff::avfilter_graph_free(&mut g);
        ff::av_buffer_unref(&mut dev);
        result
    }

    /// Verify the device-attach ordering the host filter graph depends on: on the same
    /// machine, the staged segment build must pass pixels end-to-end while the one-shot parser must
    /// fail. The one-shot parser initializes `hwupload` during the parse — before any device-attach
    /// loop can run — so it never gets a device and would force a software fallback, which is exactly
    /// why `new_impl` stages the graph. Ignored by default (needs a working hardware device).
    #[test]
    #[ignore]
    fn gpu_hwupload_device_attach_ordering() {
        unsafe {
            let old = build_hwupload_graph(false);
            let new = build_hwupload_graph(true);
            println!("one-shot parse+attach-after: {old:?}");
            println!("staged segment build:        {new:?}");
            assert!(new.is_ok(), "staged build must work: {new:?}");
            assert!(old.is_err(), "expected the one-shot ordering to fail on this FFmpeg");
        }
    }
}
