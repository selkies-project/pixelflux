/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! NVENC hardware H.264 encoder: CUDA-bound sessions that encode ARGB — or raw NV12 / YUV444 —
//! frames from the X11 host-ARGB and Wayland dmabuf capture paths.
//!
//! The module dynamically loads `libcuda`, `libnvidia-encode` and `libEGL` at runtime, negotiates
//! the NVENC API version against the installed driver (set-once per process), and stamps every
//! NVENCAPI struct with the exact `NV_ENC_*_VER` word the negotiated SDK defines, so one binary
//! drives drivers from NVENC 10.0 (~R445) through 13.0. Frames reach the GPU three ways: a
//! zero-copy dmabuf import (EGLImage → CUDA), a pinned host→device copy of packed ARGB, and a raw
//! planar upload. Sessions reconfigure resolution and rate control in place, so a resize or
//! bitrate change costs a few milliseconds instead of a full rebuild.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::os::unix::io::AsRawFd;
use std::ptr;
use std::sync::Arc;

use libloading::{Library, Symbol};
use smithay::backend::allocator::{dmabuf::Dmabuf, Buffer};

use crate::RustCaptureSettings;
use nvcodec_sys::cuda::*;
use nvcodec_sys::*;

/// EGL C-interop type aliases and the `EGL_*` attribute constants used to wrap a dmabuf as
/// an `EGLImageKHR` for CUDA import.
type EGLDisplay = *const c_void;
type EGLImageKHR = *mut c_void;
type EGLint = i32;
type EGLenum = u32;
type EGLBoolean = u32;

const EGL_NO_IMAGE_KHR: EGLImageKHR = ptr::null_mut();
const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
const EGL_DMA_BUF_PLANE0_FD_EXT: EGLint = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EGLint = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: EGLint = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EGLint = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EGLint = 0x3444;
const EGL_WIDTH: EGLint = 0x3057;
const EGL_HEIGHT: EGLint = 0x3056;
const EGL_LINUX_DRM_FOURCC_EXT: EGLint = 0x3271;
const EGL_NONE: EGLint = 0x3038;

/// Opaque CUDA graphics-resource handle for the EGL interop path — an `EGLImageKHR`
/// registered with CUDA maps to one of these.
type CUgraphicsResource = *mut c_void;

/// A CUDA frame mapped from an EGLImage: the `cuGraphicsResourceGetMappedEglFrame` result
/// describing the imported dmabuf's plane pointers, geometry, pitch and pixel format.
#[repr(C)]
#[derive(Clone, Copy)]
struct CUeglFrame {
    frame: CUeglFrameUnion,
    width: u32,
    height: u32,
    depth: u32,
    pitch: u32,
    plane_count: u32,
    num_channels: u32,
    frame_type: u32,
    egl_color_format: u32,
    cu_format: u32,
}

/// The mapped frame's plane pointers, as either CUDA arrays or pitch-linear device
/// pointers — `CUeglFrame::frame_type` selects which arm of the union is valid.
#[repr(C)]
#[derive(Clone, Copy)]
union CUeglFrameUnion {
    p_array: [CUarray; 3],
    p_pitch: [*mut c_void; 3],
}

type EglCreateImageKhrFn = unsafe extern "C" fn(
    dpy: EGLDisplay,
    ctx: *mut c_void,
    target: EGLenum,
    buffer: *mut c_void,
    attrib_list: *const EGLint,
) -> EGLImageKHR;
type EglDestroyImageKhrFn = unsafe extern "C" fn(dpy: EGLDisplay, image: EGLImageKHR) -> EGLBoolean;

/// Dynamically loaded EGL entry points (from `libEGL`) for creating and destroying the
/// `EGLImageKHR` that wraps a dmabuf. `_lib` keeps the library resident for the pointers' life.
struct EglFunctions {
    _lib: Library,
    eglGetProcAddress: unsafe extern "C" fn(procname: *const c_char) -> *mut c_void,
    eglCreateImageKHR: EglCreateImageKhrFn,
    eglDestroyImageKHR: EglDestroyImageKhrFn,
}

/// Dynamically loaded CUDA driver-API entry points (from `libcuda`) for context, device,
/// memory, host-pin and EGL-interop calls. `_lib` keeps the library resident for the pointers' life.
struct CudaFunctions {
    _lib: Library,
    cuInit: unsafe extern "C" fn(flags: u32) -> CUresult,
    cuDeviceGet: unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult,
    cuDeviceGetByPCIBusId: unsafe extern "C" fn(dev: *mut CUdevice, pciBusId: *const c_char) -> CUresult,
    cuCtxCreate_v2: unsafe extern "C" fn(
        pctx: *mut CUcontext,
        flags: u32,
        dev: CUdevice,
    ) -> CUresult,
    cuCtxPushCurrent_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    cuCtxPopCurrent_v2: unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult,
    cuCtxDestroy_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    cuMemAlloc_v2: unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    cuMemAllocPitch_v2: unsafe extern "C" fn(
        dptr: *mut CUdeviceptr,
        pPitch: *mut usize,
        WidthInBytes: usize,
        Height: usize,
        ElementSizeBytes: u32,
    ) -> CUresult,
    cuMemFree_v2: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
    cuMemcpyHtoD_v2: unsafe extern "C" fn(
        dstDevice: CUdeviceptr,
        srcHost: *const c_void,
        ByteCount: usize,
    ) -> CUresult,
    cuMemcpyDtoH_v2: unsafe extern "C" fn(
        dstHost: *mut c_void,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult,
    cuMemcpy2D_v2: unsafe extern "C" fn(pCopy: *const CUDA_MEMCPY2D) -> CUresult,
    cuMemHostRegister_v2: unsafe extern "C" fn(p: *mut c_void, bytesize: usize, flags: u32) -> CUresult,
    cuMemHostUnregister: unsafe extern "C" fn(p: *mut c_void) -> CUresult,
    cuGraphicsEGLRegisterImage: unsafe extern "C" fn(
        pCudaResource: *mut CUgraphicsResource,
        image: EGLImageKHR,
        flags: u32,
    ) -> CUresult,
    cuGraphicsUnregisterResource: unsafe extern "C" fn(resource: CUgraphicsResource) -> CUresult,
    cuGraphicsResourceGetMappedEglFrame: unsafe extern "C" fn(
        pEglFrame: *mut CUeglFrame,
        resource: CUgraphicsResource,
        index: u32,
        mipLevel: u32,
    ) -> CUresult,
    cuDeviceGetCount: unsafe extern "C" fn(count: *mut i32) -> CUresult,
    cuDeviceGetName: unsafe extern "C" fn(name: *mut c_char, len: i32, dev: CUdevice) -> CUresult,
    cuDeviceGetUuid: unsafe extern "C" fn(uuid: *mut CUuuid, dev: CUdevice) -> CUresult,
    cuGetErrorName: unsafe extern "C" fn(error: CUresult, pStr: *mut *const c_char) -> CUresult,
}

/// Dynamically loaded NVENC entry points (from `libnvidia-encode`).
///
/// - **`create_instance`** (`NvEncodeAPICreateInstance`): fills an `NV_ENCODE_API_FUNCTION_LIST`
///   with the driver's encode entry points for a requested API-version word.
/// - **`get_max_version`** (`NvEncodeAPIGetMaxSupportedVersion`): the highest API version the
///   driver supports, used to cap version probing. `Option` because very old drivers lack it, in
///   which case probing relies on `create_instance` acceptance alone.
///
/// `_lib` keeps the library resident for the function pointers' life.
struct NvencLibrary {
    _lib: Library,
    create_instance: unsafe extern "C" fn(
        functionList: *mut NV_ENCODE_API_FUNCTION_LIST,
    ) -> NVENCSTATUS,
    get_max_version: Option<unsafe extern "C" fn(*mut u32) -> NVENCSTATUS>,
}

/// Negotiated NVENC API version `(major, minor)`, resolved once per process. `None` until
/// `nvenc_negotiate` runs; every struct-version word and the session `apiVersion` derive from it.
static NVENC_NEG_VER: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();

/// The NVENCAPI structs this encoder must stamp with a per-SDK version word — enumerated
/// here precisely because getting that word exactly right is the whole mechanism that lets one
/// compiled binary satisfy every driver's version check.
///
/// Each NVENCAPI struct carries a `version` field that the driver validates against the exact word
/// its own SDK defined for that struct, rejecting anything else outright with
/// `NV_ENC_ERR_INVALID_VERSION`. Only two parts of that packed word move between SDKs — the struct
/// **revision** (bits 16-23) and the **`1<<31` flag** — so a session that has down-negotiated to an
/// older API cannot send the compiled 13.0 words; it must stamp each struct with precisely the word
/// that older SDK defined, while a current driver still receives its own native word. Naming the
/// structs here is what lets `NvStruct::rev` supply the per-version `(revision, flag)` and
/// `nvenc_struct_ver` assemble the word.
#[derive(Clone, Copy, Debug)]
enum NvStruct {
    FunctionList,
    OpenSessionExParams,
    Config,
    RcParams,
    PresetConfig,
    InitializeParams,
    ReconfigureParams,
    RegisterResource,
    MapInputResource,
    CreateBitstreamBuffer,
    PicParams,
    LockBitstream,
}

impl NvStruct {
    /// The `(struct revision, 1<<31 flag)` this struct uses under the SDK identified by the
    /// packed API version `api` (`(major<<4)|minor`) — the only two sub-fields that move between
    /// SDKs, and thus the entire per-version knowledge stamping a struct actually needs.
    ///
    /// The revision lands in bits 16-23 of the version word and the flag in bit 31; every other bit
    /// is fixed, which is exactly why matching just these two on `api` reproduces each SDK's word.
    /// The values are transcribed verbatim from `nvEncodeAPI.h` at the FFmpeg nv-codec-headers tags
    /// n10.0.26.2, n11.0.10.3, n11.1.5.3, n12.0.16.1, n12.1.14.0, n12.2.72.0 and n13.0.19.0, so they
    /// are ground truth rather than anything derived that could drift. Structs whose layout is stable
    /// across those SDKs return a constant pair; the rest match on `api`. 10.0 is the negotiation
    /// floor, so the oldest match arm also covers anything below it.
    fn rev(self, api: u32) -> (u32, bool) {
        match self {
            NvStruct::FunctionList => (2, false),
            NvStruct::OpenSessionExParams => (1, false),
            NvStruct::Config => match api {
                0xC2.. => (9, true),
                0xC0..=0xC1 => (8, true),
                _ => (7, true),
            },
            NvStruct::RcParams => (1, false),
            NvStruct::PresetConfig => (if api >= 0xC2 { 5 } else { 4 }, true),
            NvStruct::InitializeParams => match api {
                0xC2.. => (7, true),
                0xC1 => (6, true),
                _ => (5, true),
            },
            NvStruct::ReconfigureParams => (if api >= 0xC2 { 2 } else { 1 }, true),
            NvStruct::RegisterResource => match api {
                0xC2.. => (5, false),
                0xC0..=0xC1 => (4, false),
                _ => (3, false),
            },
            NvStruct::MapInputResource => (4, false),
            NvStruct::CreateBitstreamBuffer => (1, false),
            NvStruct::PicParams => match api {
                0xC2.. => (7, true),
                0xC0..=0xC1 => (6, true),
                _ => (4, true),
            },
            NvStruct::LockBitstream => match api {
                0xC2.. => (2, true),
                0xC1 => (1, true),
                0xC0 => (2, false),
                _ => (1, false),
            },
        }
    }
}

/// Assemble the `NVENCAPI_STRUCT_VERSION` word for struct `s` at API version `(maj, min)`.
///
/// The 32-bit word packs, from `NvStruct::rev` and the API version:
///
/// 1. **API major** in bits 0-7, **API minor** in bits 24-27.
/// 2. **Struct revision** (`rev`) in bits 16-23.
/// 3. **Magic `0x7`** in bits 28-30.
/// 4. **The `1<<31` flag** in bit 31, when this struct sets it at this version.
///
/// For the pinned nvcodec-sys headers this reproduces the compile-time `NV_ENC_*_VER` constants
/// exactly, so a current driver is stamped byte-for-byte identically to its own SDK's constant; the
/// `version_tests` module asserts that identity.
fn nvenc_struct_ver(s: NvStruct, maj: u32, min: u32) -> u32 {
    let (rev, high_bit) = s.rev((maj << 4) | (min & 0xF));
    (maj & 0xFF) | ((min & 0xF) << 24) | (rev << 16) | (0x7 << 28) | ((high_bit as u32) << 31)
}

/// The process's effective NVENC API version `(major, minor)`: the negotiated value once
/// `nvenc_negotiate` has run, otherwise the pinned `NVENCAPI_VERSION` decomposed (major in the low
/// byte, minor at bit 24) as the pre-negotiation fallback.
#[inline]
fn nvenc_cur_ver() -> (u32, u32) {
    NVENC_NEG_VER
        .get()
        .copied()
        .unwrap_or((NVENCAPI_VERSION & 0xFF, (NVENCAPI_VERSION >> 24) & 0xFF))
}

/// The struct-version word for `s` tagged with the process's negotiated API version — the
/// value every NVENCAPI struct literal assigns to its `version` field.
#[inline]
fn sv(s: NvStruct) -> u32 {
    let (m, n) = nvenc_cur_ver();
    nvenc_struct_ver(s, m, n)
}

/// The negotiated session `apiVersion` word (`major | minor<<24`) passed to
/// `NvEncOpenEncodeSessionEx` — note the minor sits at bit 24 here, unlike the `(major<<4)|minor`
/// packing that `NvStruct::rev` matches on.
#[inline]
fn neg_api() -> u32 {
    let (m, n) = nvenc_cur_ver();
    m | (n << 24)
}

/// Resolve the process-wide NVENC API version once, by probing the driver newest-first and
/// remembering the highest version it accepts.
///
/// The bundled nv-codec-headers are NVENC 13.0 (`pinned`), so a current driver negotiates 13.0
/// natively while older drivers down-negotiate through 12.x / 11.x to the 10.0 floor (~R445). The
/// compiled struct *layouts* are always the 13.0 ones; only the version *words* change per
/// negotiated version (via the `NvStruct::rev` table), so each struct is stamped with the exact word
/// the negotiated SDK defined. Steps:
///
/// 1. **Cap the search** by the driver's max: query `get_max_version` when present, then optionally
///    lower it further from `PIXELFLUX_NVENC_MAX_API` (e.g. `"11.0"`) for testing / pinning. A cap
///    of 0 means unknown, and probing then relies on `create_instance` acceptance alone.
/// 2. **Probe candidates** newest-first (`pinned`, 12.1, 12.0, 11.1, 11.0, 10.0), skipping any
///    above the cap. Each probe stamps an `NV_ENCODE_API_FUNCTION_LIST` with that version's word and
///    calls `create_instance`.
/// 3. **Require the whole encode path**, not just a success code: the session opener plus
///    `nvEncInitializeEncoder`, `nvEncGetEncodePresetConfigEx`, `nvEncEncodePicture` and
///    `nvEncLockBitstream` must all be non-null, because a driver can accept the function-list word
///    yet leave newer entry points null. The first fully-populated version wins.
/// 4. **Fall back** to `pinned` if nothing qualifies. Stored in `NVENC_NEG_VER`, set-once.
fn nvenc_negotiate(lib: &NvencLibrary) {
    NVENC_NEG_VER.get_or_init(|| {
        let pinned = (NVENCAPI_VERSION & 0xFF, (NVENCAPI_VERSION >> 24) & 0xFF);
        let mut drv_max: u32 = 0;
        if let Some(get_max) = lib.get_max_version {
            let mut m: u32 = 0;
            if unsafe { get_max(&mut m) } == NVENCSTATUS::NV_ENC_SUCCESS {
                drv_max = m;
            }
        }
        if let Ok(cap) = std::env::var("PIXELFLUX_NVENC_MAX_API") {
            let mut it = cap.split('.');
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                if let (Ok(cm), Ok(cn)) = (a.parse::<u32>(), b.parse::<u32>()) {
                    let capv = (cm << 4) | (cn & 0xF);
                    if capv != 0 && (drv_max == 0 || capv < drv_max) {
                        drv_max = capv;
                    }
                }
            }
        }
        let candidates = [pinned, (12, 1), (12, 0), (11, 1), (11, 0), (10, 0)];
        for (maj, min) in candidates {
            let vv = (maj << 4) | min;
            if drv_max != 0 && vv > drv_max {
                continue;
            }
            let mut probe = NV_ENCODE_API_FUNCTION_LIST {
                version: nvenc_struct_ver(NvStruct::FunctionList, maj, min),
                ..Default::default()
            };
            let st = unsafe { (lib.create_instance)(&mut probe) };
            if st == NVENCSTATUS::NV_ENC_SUCCESS
                && probe.nvEncOpenEncodeSessionEx.is_some()
                && probe.nvEncInitializeEncoder.is_some()
                && probe.nvEncGetEncodePresetConfigEx.is_some()
                && probe.nvEncEncodePicture.is_some()
                && probe.nvEncLockBitstream.is_some()
            {
                eprintln!("[pixelflux] NVENC API version negotiated: {}.{}", maj, min);
                return (maj, min);
            }
        }
        pinned
    });
}

/// Cached CUDA import of a dmabuf, keyed by fd so a recurring capture buffer is imported
/// once: the `EGLImageKHR`, the CUDA graphics resource it registers as, and the mapped `CUeglFrame`.
/// Torn down on drop / reconfigure.
struct CachedDmaBuf {
    egl_image: EGLImageKHR,
    cuda_resource: CUgraphicsResource,
    egl_frame: CUeglFrame,
}

/// GUID selecting the H.264 **High** profile (4:2:0) for `NV_ENC_CONFIG::profileGUID`.
const NV_ENC_H264_PROFILE_HIGH_GUID: GUID = GUID {
    Data1: 0x205b553d,
    Data2: 0x5f01,
    Data3: 0x4d9e,
    Data4: [0x91, 0x84, 0xda, 0x32, 0x77, 0x5b, 0x55, 0x9b],
};

/// GUID selecting the H.264 **High 4:4:4 Predictive** profile for full-color encoding.
const NV_ENC_H264_PROFILE_HIGH_444_GUID: GUID = GUID {
    Data1: 0x7ac663cb,
    Data2: 0xa598,
    Data3: 0x4960,
    Data4: [0xb8, 0x44, 0x33, 0x9b, 0x26, 0x1a, 0x7d, 0x5c],
};

/// A live NVENC H.264 encoder session with its CUDA context and interop resources.
///
/// One instance owns a CUDA context bound to a specific GPU plus an NVENC session and everything
/// the three input paths need:
///
/// - **ARGB path**: a pitched device buffer (`input_device_ptr` / `input_pitch`) registered and
///   mapped as the NVENC input (`registered_input_resource` / `mapped_input_buffer`), fed either by
///   a dmabuf copy or a host→device ARGB upload.
/// - **Raw planar path**: a lazily-allocated NV12 / YUV444 device buffer (the `nv12_*` fields),
///   created on first `encode_raw`.
/// - **Zero-copy dmabuf path**: `dmabuf_cache` memoizes each fd's EGLImage → CUDA import.
///
/// `bitstream_buffers` is a small ring (`current_buffer_idx` cycles it) of output buffers.
/// `pinned_hosts` maps each page-locked host upload source's base pointer to its registered length,
/// with a `0` length recording a failed registration so that address is never re-pinned.
/// `current_qp` tracks the live ConstQP so a paint-over reconfigure is skipped when unchanged.
/// `encode_config` and `init_params` are retained so in-place reconfigure can resubmit them.
/// `omit_stripe_headers` drops the 10-byte wire
/// header, and `node_index` is the effective CUDA device this session is bound to — a reuse across
/// captures that now targets a different device must rebuild rather than reconfigure.
pub struct NvencEncoder {
    encoder_session: *mut c_void,
    cuda_context: CUcontext,
    egl_display: EGLDisplay,
    width: u32,
    height: u32,
    current_qp: u32,
    encode_config: NV_ENC_CONFIG,
    init_params: NV_ENC_INITIALIZE_PARAMS,
    input_device_ptr: CUdeviceptr,
    input_pitch: usize,
    registered_input_resource: NV_ENC_REGISTERED_PTR,
    mapped_input_buffer: NV_ENC_INPUT_PTR,
    nv12_device_ptr: Option<CUdeviceptr>,
    nv12_pitch: usize,
    nv12_registered_resource: Option<NV_ENC_REGISTERED_PTR>,
    nv12_mapped_buffer: Option<NV_ENC_INPUT_PTR>,
    bitstream_buffers: Vec<NV_ENC_OUTPUT_PTR>,
    current_buffer_idx: usize,
    dmabuf_cache: HashMap<i32, CachedDmaBuf>,
    pinned_hosts: HashMap<usize, usize>,
    cuda: Arc<CudaFunctions>,
    egl: Arc<EglFunctions>,
    _nvenc_lib: Arc<NvencLibrary>,
    nvenc_funcs: NV_ENCODE_API_FUNCTION_LIST,
    omit_stripe_headers: bool,
    node_index: i32,
}

unsafe impl Send for NvencEncoder {}

/// Release every GPU resource the session holds, in the one teardown order the drivers
/// tolerate, so nothing leaks and no still-referenced handle is ever freed out from under the
/// driver.
///
/// The whole sequence runs with the owning CUDA context pushed current, because the `cuMemFree` /
/// `cuGraphicsUnregisterResource` / `cuMemHostUnregister` calls each act on the *current* context —
/// pop it first and the frees silently do nothing, leaking device memory. Within that, resources
/// go inner-handle before the outer handle that owns it, since freeing an owner first orphans or
/// faults on what still points into it: unmap the ARGB and raw-plane inputs before unregistering
/// them, free their device buffers, destroy the bitstream buffers, and unregister-and-destroy every
/// cached dmabuf import — all session-owned — before the encoder session itself, and destroy that
/// session before the CUDA context it was opened against. The page-locked host sources are unpinned
/// in the same pass, each only when its recorded length is non-zero (a `0` marks a registration that
/// failed and so was never pinned).
impl Drop for NvencEncoder {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);

            if !self.mapped_input_buffer.is_null() {
                (self.nvenc_funcs.nvEncUnmapInputResource.unwrap())(
                    self.encoder_session,
                    self.mapped_input_buffer,
                );
            }
            if !self.registered_input_resource.is_null() {
                (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                    self.encoder_session,
                    self.registered_input_resource,
                );
            }
            if self.input_device_ptr != 0 {
                (self.cuda.cuMemFree_v2)(self.input_device_ptr);
            }

            if let Some(mapped) = self.nv12_mapped_buffer {
                (self.nvenc_funcs.nvEncUnmapInputResource.unwrap())(
                    self.encoder_session,
                    mapped,
                );
            }
            if let Some(registered) = self.nv12_registered_resource {
                (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                    self.encoder_session,
                    registered,
                );
            }
            if let Some(ptr) = self.nv12_device_ptr {
                (self.cuda.cuMemFree_v2)(ptr);
            }

            for &bs in &self.bitstream_buffers {
                (self.nvenc_funcs.nvEncDestroyBitstreamBuffer.unwrap())(
                    self.encoder_session,
                    bs,
                );
            }

            for (_, cache) in self.dmabuf_cache.drain() {
                (self.cuda.cuGraphicsUnregisterResource)(cache.cuda_resource);
                (self.egl.eglDestroyImageKHR)(self.egl_display, cache.egl_image);
            }

            for (base, len) in &self.pinned_hosts {
                if *len > 0 {
                    (self.cuda.cuMemHostUnregister)(*base as *mut c_void);
                }
            }

            if !self.encoder_session.is_null() {
                (self.nvenc_funcs.nvEncDestroyEncoder.unwrap())(self.encoder_session);
            }

            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
            (self.cuda.cuCtxDestroy_v2)(self.cuda_context);
        }
    }
}

use super::min_h264_level;

/// The level an NVENC session advertises for this geometry, floored at 5.2 (`NV_ENC_LEVEL`
/// shares level_idc's numbering: 52/60/61/62 for 5.2/6.0/6.1/6.2).
///
/// `reconfigure_resolution` resizes a live session in place, so the level has to cover every
/// geometry that session can still reach — a level bump mid-GOP forces some hardware decoders to
/// re-initialize. 5.2's MaxFS of 36864 macroblocks (≈ 4096×2304) spans everything up to 4K, so
/// pinning that floor makes the whole range resolve to High@5.2; only beyond 4K does the shared
/// ladder step up to 6.0 / 6.1 / 6.2.
fn nvenc_h264_level(width: u32, height: u32, fps: u32) -> u32 {
    min_h264_level(width, height, fps).max(52)
}

impl NvencEncoder {
    /// Resolve EGL at runtime rather than link against it, so one binary boots even on hosts
    /// without EGL — it is needed only by the zero-copy dmabuf path — and reach the
    /// `eglCreateImageKHR` / `eglDestroyImageKHR` entry points through `eglGetProcAddress` because
    /// they are KHR *extensions* the base `libEGL` is not obliged to export as plain symbols.
    /// Erroring when the library or either extension is missing lets the caller fall back to another
    /// encoder instead of crashing at the first dmabuf import.
    fn load_egl() -> Result<EglFunctions, String> {
        unsafe {
            let lib_name = "libEGL.so.1";
            let lib = Library::new(lib_name)
                .or_else(|_| Library::new("libEGL.so"))
                .map_err(|e| format!("Could not load EGL library: {}", e))?;

            let get_proc_addr_sym: Symbol<unsafe extern "C" fn(*const c_char) -> *mut c_void> = lib
                .get(b"eglGetProcAddress\0")
                .map_err(|e| format!("Missing symbol eglGetProcAddress: {}", e))?;

            let eglGetProcAddress = *get_proc_addr_sym;

            let load_extension = |name: &str| -> Result<*mut c_void, String> {
                let c_name = CString::new(name).unwrap();
                let addr = eglGetProcAddress(c_name.as_ptr());
                if addr.is_null() {
                    Err(format!("EGL Extension not found: {}", name))
                } else {
                    Ok(addr)
                }
            };

            let create_addr = load_extension("eglCreateImageKHR")?;
            let destroy_addr = load_extension("eglDestroyImageKHR")?;

            Ok(EglFunctions {
                _lib: lib,
                eglGetProcAddress,
                eglCreateImageKHR: std::mem::transmute::<*mut c_void, EglCreateImageKhrFn>(create_addr),
                eglDestroyImageKHR: std::mem::transmute::<*mut c_void, EglDestroyImageKhrFn>(destroy_addr),
            })
        }
    }

    /// Resolve the CUDA driver library (`libcuda.so.1`, or `nvcuda.dll` on Windows) at
    /// runtime so the crate links against no CUDA SDK and still runs wherever a driver is installed,
    /// binding every `cu*` entry point up front so the per-frame hot path is plain indirect calls
    /// with no repeated symbol lookups. A missing symbol errors with its name, turning an ABI
    /// mismatch into a legible message instead of a later null-pointer call.
    fn load_cuda() -> Result<CudaFunctions, String> {
        unsafe {
            let lib_name = if cfg!(windows) {
                "nvcuda.dll"
            } else {
                "libcuda.so.1"
            };
            let lib = Library::new(lib_name)
                .map_err(|e| format!("Could not load CUDA library ({}): {}", lib_name, e))?;

            macro_rules! load {
                ($lib:expr, $name:expr) => {
                    *$lib.get($name).map_err(|e| {
                        format!(
                            "Missing symbol {}: {}",
                            std::str::from_utf8($name).unwrap(),
                            e
                        )
                    })?
                };
            }

            Ok(CudaFunctions {
                cuInit: load!(lib, b"cuInit\0"),
                cuDeviceGet: load!(lib, b"cuDeviceGet\0"),
                cuDeviceGetByPCIBusId: load!(lib, b"cuDeviceGetByPCIBusId\0"),
                cuCtxCreate_v2: load!(lib, b"cuCtxCreate_v2\0"),
                cuCtxPushCurrent_v2: load!(lib, b"cuCtxPushCurrent_v2\0"),
                cuCtxPopCurrent_v2: load!(lib, b"cuCtxPopCurrent_v2\0"),
                cuCtxDestroy_v2: load!(lib, b"cuCtxDestroy_v2\0"),
                cuMemAlloc_v2: load!(lib, b"cuMemAlloc_v2\0"),
                cuMemAllocPitch_v2: load!(lib, b"cuMemAllocPitch_v2\0"),
                cuMemFree_v2: load!(lib, b"cuMemFree_v2\0"),
                cuMemcpyHtoD_v2: load!(lib, b"cuMemcpyHtoD_v2\0"),
                cuMemcpyDtoH_v2: load!(lib, b"cuMemcpyDtoH_v2\0"),
                cuMemcpy2D_v2: load!(lib, b"cuMemcpy2D_v2\0"),
                cuMemHostRegister_v2: load!(lib, b"cuMemHostRegister_v2\0"),
                cuMemHostUnregister: load!(lib, b"cuMemHostUnregister\0"),
                cuGraphicsEGLRegisterImage: load!(lib, b"cuGraphicsEGLRegisterImage\0"),
                cuGraphicsUnregisterResource: load!(lib, b"cuGraphicsUnregisterResource\0"),
                cuGraphicsResourceGetMappedEglFrame: load!(
                    lib,
                    b"cuGraphicsResourceGetMappedEglFrame\0"
                ),
                cuDeviceGetCount: load!(lib, b"cuDeviceGetCount\0"),
                cuDeviceGetName: load!(lib, b"cuDeviceGetName\0"),
                cuDeviceGetUuid: load!(lib, b"cuDeviceGetUuid\0"),
                cuGetErrorName: load!(lib, b"cuGetErrorName\0"),
                _lib: lib,
            })
        }
    }

    /// Resolve `libnvidia-encode` at runtime for the same reason as CUDA — no SDK to link,
    /// runs against whatever driver ships — binding `NvEncodeAPICreateInstance` as the sole entry
    /// point every later encode call is reached through. `NvEncodeAPIGetMaxSupportedVersion` is kept
    /// optional and bound only when present, because very old drivers lack it; negotiation then falls
    /// back to probing `create_instance` acceptance directly rather than failing to load.
    fn load_nvenc() -> Result<NvencLibrary, String> {
        unsafe {
            let lib_name = NVENC_DLL_NAME;
            let lib = Library::new(lib_name)
                .map_err(|e| format!("Could not load NVENC library ({}): {}", lib_name, e))?;

            let create_instance = *lib
                .get(NV_ENCODE_API_CREATE_INSTANCE_FN_NAME)
                .map_err(|e| e.to_string())?;
            let get_max_version = lib
                .get::<NvEncodeApiGetMaxSupportedVersionFn>(
                    NV_ENCODE_API_GET_MAX_SUPPORTED_VERSION_FN_NAME,
                )
                .map(|s| *s)
                .ok();
            Ok(NvencLibrary {
                create_instance,
                get_max_version,
                _lib: lib,
            })
        }
    }

    /// Turn a `CUresult` into the driver's own error name via `cuGetErrorName` so a failure
    /// logs something diagnosable (e.g. `CUDA_ERROR_OUT_OF_MEMORY`) instead of a bare integer,
    /// falling back to the numeric code only when the name is unavailable.
    unsafe fn get_error_string(cuda: &CudaFunctions, err: CUresult) -> String {
        let mut p_str: *const c_char = ptr::null();
        if (cuda.cuGetErrorName)(err, &mut p_str) == CUresult::CUDA_SUCCESS && !p_str.is_null() {
            CStr::from_ptr(p_str).to_string_lossy().into_owned()
        } else {
            format!("Unknown CUDA Error ({})", err.0)
        }
    }

    /// Log the CUDA devices CUDA can enumerate — a debug aid when session init fails to find
    /// or bind the expected GPU.
    unsafe fn probe_devices(cuda: &CudaFunctions) {
        let mut count = 0;
        if (cuda.cuDeviceGetCount)(&mut count) != CUresult::CUDA_SUCCESS {
            return;
        }
        println!("[NVENC] Found {} CUDA devices:", count);
        for i in 0..count {
            let mut dev = 0;
            (cuda.cuDeviceGet)(&mut dev, i);
            let mut name_buf = [0 as c_char; 256];
            (cuda.cuDeviceGetName)(name_buf.as_mut_ptr(), 256, dev);
            let name = CStr::from_ptr(name_buf.as_ptr()).to_string_lossy();
            println!("[NVENC]   Device {}: {}", i, name);
        }
    }

    /// The PCI bus ID of the GPU behind `/dev/dri/renderD<128+index>`, read from the sysfs
    /// device symlink, so CUDA can bind to the same physical GPU the capture render node lives on.
    fn get_pci_bus_id(render_index: i32) -> Option<String> {
        let path = format!("/sys/class/drm/renderD{}/device", 128 + render_index);
        if let Ok(target) = std::fs::read_link(&path) {
            if let Some(name) = target.file_name() {
                if let Some(name_str) = name.to_str() {
                    return Some(name_str.to_string());
                }
            }
        }
        None
    }

    /// Build a live NVENC session: bind CUDA to the target GPU, open and configure the H.264
    /// encoder, and allocate its input and output buffers.
    ///
    /// The sequence:
    ///
    /// 1. **Load and negotiate**: dlopen EGL / CUDA / NVENC, then `nvenc_negotiate` resolves the API
    ///    version against the driver (set-once) before any struct is version-tagged. The multi-GPU
    ///    `GET_ATTACHED_IDS` ioctl filter is installed after the NVIDIA libraries are loaded — so
    ///    their GOTs can be patched — and before `cuInit` enumerates devices (a no-op unless a host
    ///    GPU is hidden from this container). The three library `Arc`s are leaked once per process so
    ///    the resolved function pointers stay valid for the program's life.
    /// 2. **Bind the device**: `cuInit`, then bind by the render node's PCI bus ID
    ///    (`encode_node_index`, with auto `<0` meaning device 0), falling back to CUDA device 0, and
    ///    create the CUDA context.
    /// 3. **Allocate input**: a pitched ARGB device buffer (`cuMemAllocPitch`, 16-byte element
    ///    alignment) that hardware CSC turns into YUV.
    /// 4. **Open the session**: create the function-list instance, open the session with the
    ///    negotiated `apiVersion`, and pull a preset config (P4, ultra-low-latency). A failed preset
    ///    lookup logs the driver's error string and proceeds with the zeroed default rather than
    ///    aborting.
    /// 5. **Configure the stream** (mutating the returned preset config, whose `version` word is
    ///    re-stamped while its embedded `rcParams` keeps the version the preset fill set): High or
    ///    High-4:4:4 profile; CBR (two-pass quarter-resolution for tighter per-frame rate adherence,
    ///    VBV sizing, optional min/max QP clamps) or ConstQP; infinite GOP (`gopLength` / `idrPeriod`
    ///    = `0xFFFFFFFF`); `zeroReorderDelay` plus a bitstream-restriction VUI (`max_num_reorder_frames=0`)
    ///    so no-reorder decoders don't buffer; an explicit Annex-A level from `nvenc_h264_level`
    ///    pinned from frame 1 so the level never bumps mid-stream; BT.709 VUI primaries and
    ///    transfer for the sRGB source, with an SMPTE170M matrix and limited range to match the
    ///    hardware ARGB CSC; chroma 4:2:0 or 4:4:4; repeated SPS/PPS; CABAC; no AUD; strict GOP
    ///    target; and lookahead disabled for real-time latency.
    /// 6. **Initialize with resize headroom**: `maxEncodeWidth` / `maxEncodeHeight` are raised to at
    ///    least 4096×2304 (the 5.2 ceiling) so `reconfigure_resolution` can grow in place; this costs
    ///    ~290 MiB of device memory, so a failed init retries at the exact size (in-place resize then
    ///    falls back to a rebuild).
    /// 7. **Register, map, and buffer**: register and map the ARGB input surface, and create a
    ///    4-deep ring of bitstream output buffers.
    ///
    /// Every failure after the CUDA allocation unwinds the resources created so far — buffers,
    /// session, context — before returning `Err`. EGL is only needed by the zero-copy dmabuf path,
    /// so callers on the host-ARGB path pass a null `egl_display`. The retained
    /// `init_params.encodeConfig` raw pointer is nulled before the struct is returned (it points at
    /// a local `config` about to move); the reconfigure paths repoint it at `self.encode_config`
    /// when they resubmit.
    pub fn new(
        settings: &RustCaptureSettings,
        egl_display: *const c_void,
    ) -> Result<Self, String> {
        println!("[NVENC] Initializing...");

        let egl = Arc::new(Self::load_egl()?);
        let cuda = Arc::new(Self::load_cuda()?);
        let nvenc_lib = Arc::new(Self::load_nvenc()?);
        nvenc_negotiate(&nvenc_lib);

        crate::nvgpufilter::install();

        static LEAK_ONCE: std::sync::Once = std::sync::Once::new();
        LEAK_ONCE.call_once(|| {
            std::mem::forget(egl.clone());
            std::mem::forget(cuda.clone());
            std::mem::forget(nvenc_lib.clone());
        });

        unsafe {
            let res = (cuda.cuInit)(0);
            if res != CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "Init CUDA failed: {}",
                    Self::get_error_string(&cuda, res)
                ));
            }

            Self::probe_devices(&cuda);

            let mut cu_device: CUdevice = 0;
            let mut device_found = false;

            if let Some(pci_bus_id) = Self::get_pci_bus_id(settings.encode_node_index.max(0)) {
                let c_pci_bus_id = CString::new(pci_bus_id.clone()).unwrap();
                if (cuda.cuDeviceGetByPCIBusId)(&mut cu_device, c_pci_bus_id.as_ptr()) == CUresult::CUDA_SUCCESS {
                    println!("[NVENC] Bound to CUDA device via PCI Bus ID: {}", pci_bus_id);
                    device_found = true;
                }
            }

            if !device_found {
                let res = (cuda.cuDeviceGet)(&mut cu_device, 0);
                if res != CUresult::CUDA_SUCCESS {
                    return Err("Failed to get default CUDA device".into());
                }
            }

            let mut cu_context: CUcontext = ptr::null_mut();
            let res = (cuda.cuCtxCreate_v2)(&mut cu_context, 0, cu_device);
            if res != CUresult::CUDA_SUCCESS {
                return Err("Failed to create CUDA Context".into());
            }

            let width = settings.width as u32;
            let height = settings.height as u32;
            let mut input_device_ptr: CUdeviceptr = 0;
            let mut input_pitch: usize = 0;

            let res = (cuda.cuMemAllocPitch_v2)(
                &mut input_device_ptr,
                &mut input_pitch,
                (width * 4) as usize,
                height as usize,
                16,
            );
            if res != CUresult::CUDA_SUCCESS {
                (cuda.cuCtxDestroy_v2)(cu_context);
                return Err("Failed to allocate ARGB input buffer on GPU".into());
            }

            let mut function_list = NV_ENCODE_API_FUNCTION_LIST {
                version: sv(NvStruct::FunctionList),
                ..Default::default()
            };
            if (nvenc_lib.create_instance)(&mut function_list) != NVENCSTATUS::NV_ENC_SUCCESS {
                (cuda.cuMemFree_v2)(input_device_ptr);
                (cuda.cuCtxDestroy_v2)(cu_context);
                return Err("NvEncodeAPICreateInstance failed".into());
            }

            let mut session_params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
                version: sv(NvStruct::OpenSessionExParams),
                deviceType: NV_ENC_DEVICE_TYPE::NV_ENC_DEVICE_TYPE_CUDA,
                device: cu_context as *mut c_void,
                apiVersion: neg_api(),
                ..Default::default()
            };

            let mut encoder_session: *mut c_void = ptr::null_mut();
            let open_fn = function_list.nvEncOpenEncodeSessionEx.unwrap();
            if open_fn(&mut session_params, &mut encoder_session) != NVENCSTATUS::NV_ENC_SUCCESS {
                (cuda.cuMemFree_v2)(input_device_ptr);
                (cuda.cuCtxDestroy_v2)(cu_context);
                return Err("Failed to open NVENC session".into());
            }

            let is_444 = settings.video_fullcolor;
            let profile_guid = if is_444 {
                NV_ENC_H264_PROFILE_HIGH_444_GUID
            } else {
                NV_ENC_H264_PROFILE_HIGH_GUID
            };

            let mut config = NV_ENC_CONFIG {
                version: sv(NvStruct::Config),
                ..Default::default()
            };
            let mut preset_config = NV_ENC_PRESET_CONFIG {
                version: sv(NvStruct::PresetConfig),
                presetCfg: config,
                ..Default::default()
            };

            let get_preset_ex = function_list.nvEncGetEncodePresetConfigEx.unwrap();
            let preset_status = get_preset_ex(
                encoder_session,
                NV_ENC_CODEC_H264_GUID,
                NV_ENC_PRESET_P4_GUID,
                NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
                &mut preset_config,
            );
            if preset_status != NVENCSTATUS::NV_ENC_SUCCESS {
                let detail = function_list.nvEncGetLastErrorString.and_then(|f| {
                    let p = f(encoder_session);
                    if p.is_null() {
                        None
                    } else {
                        Some(CStr::from_ptr(p).to_string_lossy().into_owned())
                    }
                });
                eprintln!(
                    "[NVENC] nvEncGetEncodePresetConfigEx failed ({preset_status:?}): {}",
                    detail.as_deref().unwrap_or("no error string")
                );
            }

            config = preset_config.presetCfg;
            config.version = sv(NvStruct::Config);
            config.profileGUID = profile_guid;
            if settings.video_cbr_mode {
                let bps = (settings.video_bitrate_kbps.max(0) as u32).saturating_mul(1000);
                config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR;
                config.rcParams.multiPass = NV_ENC_MULTI_PASS::NV_ENC_TWO_PASS_QUARTER_RESOLUTION;
                config.rcParams.averageBitRate = bps;
                config.rcParams.maxBitRate = bps;
                config.rcParams.vbvBufferSize = crate::encoders::vbv_bits(
                    bps,
                    settings.target_fps,
                    settings.keyframe_interval_s,
                    settings.video_vbv_multiplier,
                );
                if settings.video_min_qp > 0 {
                    let q = settings.video_min_qp.min(51) as u32;
                    config.rcParams.set_enableMinQP(1);
                    config.rcParams.minQP.qpInterP = q;
                    config.rcParams.minQP.qpInterB = q;
                    config.rcParams.minQP.qpIntra = q;
                }
                if settings.video_max_qp > 0 {
                    let q = settings.video_max_qp.min(51) as u32;
                    config.rcParams.set_enableMaxQP(1);
                    config.rcParams.maxQP.qpInterP = q;
                    config.rcParams.maxQP.qpInterB = q;
                    config.rcParams.maxQP.qpIntra = q;
                }
            } else {
                config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CONSTQP;
                config.rcParams.constQP.qpInterP = settings.video_crf as u32;
                config.rcParams.constQP.qpInterB = settings.video_crf as u32;
                config.rcParams.constQP.qpIntra = settings.video_crf as u32;
            }
            config.frameIntervalP = 1;
            config.gopLength = 0xFFFFFFFF;
            config.rcParams.set_zeroReorderDelay(1);
            config.encodeCodecConfig.h264Config.h264VUIParameters.bitstreamRestrictionFlag = 1;
            config.encodeCodecConfig.h264Config.level =
                nvenc_h264_level(width, height, settings.target_fps as u32);
            config.encodeCodecConfig.h264Config.idrPeriod = 0xFFFFFFFF;
            config.encodeCodecConfig.h264Config.h264VUIParameters.videoSignalTypePresentFlag = 1;
            config.encodeCodecConfig.h264Config.h264VUIParameters.videoFormat =
                NV_ENC_VUI_VIDEO_FORMAT::NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
            config.encodeCodecConfig.h264Config.h264VUIParameters.colourDescriptionPresentFlag = 1;
            // Primaries and transfer describe the source, which is sRGB desktop pixels —
            // sRGB shares BT.709's primaries and transfer function. Only the matrix follows
            // the encoder: NVENC's ARGB hardware CSC is fixed at BT.601, so a client that
            // inverts BT.709 shifts saturated colour badly.
            config.encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries =
                NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709;
            config.encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics =
                NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
            config.encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix =
                NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_SMPTE170M;
            config.encodeCodecConfig.h264Config.chromaFormatIDC = if is_444 { 3 } else { 1 };
            // That same hardware CSC emits limited range in every chroma format. The
            // host-planar path could encode full range, but a capture moves between the
            // two at runtime when zero-copy demotes to readback, and the declared range
            // has to hold across that, so every NVENC session declares limited.
            config.encodeCodecConfig.h264Config.h264VUIParameters.videoFullRangeFlag = 0;
            config.encodeCodecConfig.h264Config.set_repeatSPSPPS(1);
            config.encodeCodecConfig.h264Config.entropyCodingMode =
                NV_ENC_H264_ENTROPY_CODING_MODE::NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;
            config.encodeCodecConfig.h264Config.set_outputAUD(0);
            config.rcParams.set_strictGOPTarget(1);
            config.rcParams.set_enableLookahead(0);
            config.rcParams.lookaheadDepth = 0;

            let mut init_params = NV_ENC_INITIALIZE_PARAMS {
                version: sv(NvStruct::InitializeParams),
                encodeGUID: NV_ENC_CODEC_H264_GUID,
                presetGUID: NV_ENC_PRESET_P4_GUID,
                tuningInfo: NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
                encodeWidth: width,
                encodeHeight: height,
                darWidth: width,
                darHeight: height,
                frameRateNum: settings.target_fps.max(1.0) as u32,
                frameRateDen: 1,
                enablePTD: 1,
                encodeConfig: &mut config,
                maxEncodeWidth: width.max(4096),
                maxEncodeHeight: height.max(2304),
                ..Default::default()
            };

            let init_fn = function_list.nvEncInitializeEncoder.unwrap();
            if init_fn(encoder_session, &mut init_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                init_params.maxEncodeWidth = width;
                init_params.maxEncodeHeight = height;
                if init_fn(encoder_session, &mut init_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                    (function_list.nvEncDestroyEncoder.unwrap())(encoder_session);
                    (cuda.cuMemFree_v2)(input_device_ptr);
                    (cuda.cuCtxDestroy_v2)(cu_context);
                    return Err("Failed to initialize encoder".into());
                }
                eprintln!("[NVENC] Init with resize headroom failed; running without it.");
            }

            init_params.encodeConfig = ptr::null_mut();

            let mut reg_res = NV_ENC_REGISTER_RESOURCE {
                version: sv(NvStruct::RegisterResource),
                resourceType: NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                width,
                height,
                resourceToRegister: input_device_ptr as *mut c_void,
                pitch: input_pitch as u32,
                bufferFormat: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
                bufferUsage: NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE,
                ..Default::default()
            };

            let register_fn = function_list.nvEncRegisterResource.unwrap();
            if register_fn(encoder_session, &mut reg_res) != NVENCSTATUS::NV_ENC_SUCCESS {
                (function_list.nvEncDestroyEncoder.unwrap())(encoder_session);
                (cuda.cuMemFree_v2)(input_device_ptr);
                (cuda.cuCtxDestroy_v2)(cu_context);
                return Err("Failed to register input buffer".into());
            }

            let mut map_params = NV_ENC_MAP_INPUT_RESOURCE {
                version: sv(NvStruct::MapInputResource),
                registeredResource: reg_res.registeredResource,
                ..Default::default()
            };
            let map_fn = function_list.nvEncMapInputResource.unwrap();
            if map_fn(encoder_session, &mut map_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                (function_list.nvEncUnregisterResource.unwrap())(
                    encoder_session,
                    reg_res.registeredResource,
                );
                (function_list.nvEncDestroyEncoder.unwrap())(encoder_session);
                (cuda.cuMemFree_v2)(input_device_ptr);
                (cuda.cuCtxDestroy_v2)(cu_context);
                return Err("Failed to map input buffer".into());
            }

            let mut bitstream_buffers = Vec::new();
            let create_bs_fn = function_list.nvEncCreateBitstreamBuffer.unwrap();
            for _ in 0..4 {
                let mut bitstream_params = NV_ENC_CREATE_BITSTREAM_BUFFER {
                    version: sv(NvStruct::CreateBitstreamBuffer),
                    ..Default::default()
                };
                if create_bs_fn(encoder_session, &mut bitstream_params)
                    != NVENCSTATUS::NV_ENC_SUCCESS
                {
                    for &bs in &bitstream_buffers {
                        (function_list.nvEncDestroyBitstreamBuffer.unwrap())(encoder_session, bs);
                    }
                    (function_list.nvEncUnmapInputResource.unwrap())(
                        encoder_session,
                        map_params.mappedResource,
                    );
                    (function_list.nvEncUnregisterResource.unwrap())(
                        encoder_session,
                        reg_res.registeredResource,
                    );
                    (function_list.nvEncDestroyEncoder.unwrap())(encoder_session);
                    (cuda.cuMemFree_v2)(input_device_ptr);
                    (cuda.cuCtxDestroy_v2)(cu_context);
                    return Err("Failed to create bitstream buffer".into());
                }
                bitstream_buffers.push(bitstream_params.bitstreamBuffer);
            }

            println!("[NVENC] Initialized successfully (4:4:4 mode: {}).", is_444);

            Ok(Self {
                encoder_session,
                cuda_context: cu_context,
                egl_display: egl_display as EGLDisplay,
                width,
                height,
                current_qp: settings.video_crf as u32,
                encode_config: config,
                init_params,
                input_device_ptr,
                input_pitch,
                registered_input_resource: reg_res.registeredResource,
                mapped_input_buffer: map_params.mappedResource,
                nv12_device_ptr: None,
                nv12_pitch: 0,
                nv12_registered_resource: None,
                nv12_mapped_buffer: None,
                bitstream_buffers,
                current_buffer_idx: 0,
                dmabuf_cache: HashMap::new(),
                pinned_hosts: HashMap::new(),
                cuda,
                egl,
                _nvenc_lib: nvenc_lib,
                nvenc_funcs: function_list,
                omit_stripe_headers: settings.omit_stripe_headers,
                node_index: settings.encode_node_index.max(0),
            })
        }
    }

    /// Resize the live session to `settings` in place, folding in the current rate / QP /
    /// fps, without tearing it down.
    ///
    /// The NVENC session, CUDA context and bitstream buffers survive, so a resize costs a few
    /// milliseconds instead of a full rebuild. Flow:
    ///
    /// 1. **Reject the unchangeable**: a different encode device, a chroma-format flip (4:4:4), an
    ///    RC-mode flip, or dimensions of zero or beyond the init-time `maxEncode` headroom all return
    ///    `Err` so the caller rebuilds. Chroma and RC mode are read back from the live
    ///    `encode_config` (the H.264 arm of the codec-config union is the one this encoder fills).
    /// 2. **Release geometry-dependent state** under the pushed CUDA context: unmap / unregister /
    ///    free the ARGB surface, the raw-plane buffer, every cached dmabuf import, and every pinned
    ///    host. The raw-plane buffer and dmabuf imports are re-created lazily by their encode paths;
    ///    pinned hosts are dropped because the source shm segments are recreated on resize and may
    ///    reuse the same base addresses.
    /// 3. **Reconfigure the session**: update the level for the new size, the CBR bitrate + VBV or
    ///    the ConstQP, and the new dimensions / DAR / frame rate, then `NvEncReconfigureEncoder` with
    ///    `resetEncoder` and `forceIDR` so the stream restarts cleanly at the new size. Driver
    ///    rejection returns `Err`.
    /// 4. **Reallocate the ARGB input** at the new size and register + map it exactly as init does.
    ///
    /// On success the next encoded frame is a reset-RC IDR.
    pub fn reconfigure_resolution(&mut self, settings: &RustCaptureSettings) -> Result<(), String> {
        let new_w = settings.width as u32;
        let new_h = settings.height as u32;
        let is_444 =
            unsafe { self.encode_config.encodeCodecConfig.h264Config.chromaFormatIDC == 3 };
        let is_cbr = self.encode_config.rcParams.rateControlMode
            == NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR;
        if settings.encode_node_index.max(0) != self.node_index {
            return Err("encode device changed".into());
        }
        if settings.video_fullcolor != is_444 {
            return Err("chroma format changed".into());
        }
        if settings.video_cbr_mode != is_cbr {
            return Err("rate-control mode changed".into());
        }
        if new_w == 0
            || new_h == 0
            || new_w > self.init_params.maxEncodeWidth
            || new_h > self.init_params.maxEncodeHeight
        {
            return Err(format!(
                "{}x{} outside reconfigure headroom {}x{}",
                new_w, new_h, self.init_params.maxEncodeWidth, self.init_params.maxEncodeHeight
            ));
        }

        unsafe {
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);
            if !self.mapped_input_buffer.is_null() {
                (self.nvenc_funcs.nvEncUnmapInputResource.unwrap())(
                    self.encoder_session,
                    self.mapped_input_buffer,
                );
                self.mapped_input_buffer = ptr::null_mut();
            }
            if !self.registered_input_resource.is_null() {
                (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                    self.encoder_session,
                    self.registered_input_resource,
                );
                self.registered_input_resource = ptr::null_mut();
            }
            if self.input_device_ptr != 0 {
                (self.cuda.cuMemFree_v2)(self.input_device_ptr);
                self.input_device_ptr = 0;
            }
            if let Some(mapped) = self.nv12_mapped_buffer.take() {
                (self.nvenc_funcs.nvEncUnmapInputResource.unwrap())(self.encoder_session, mapped);
            }
            if let Some(registered) = self.nv12_registered_resource.take() {
                (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                    self.encoder_session,
                    registered,
                );
            }
            if let Some(ptr) = self.nv12_device_ptr.take() {
                (self.cuda.cuMemFree_v2)(ptr);
            }
            self.nv12_pitch = 0;
            for (_, cache) in self.dmabuf_cache.drain() {
                (self.cuda.cuGraphicsUnregisterResource)(cache.cuda_resource);
                (self.egl.eglDestroyImageKHR)(self.egl_display, cache.egl_image);
            }
            for (base, len) in self.pinned_hosts.drain() {
                if len > 0 {
                    (self.cuda.cuMemHostUnregister)(base as *mut c_void);
                }
            }

            self.encode_config.encodeCodecConfig.h264Config.level =
                nvenc_h264_level(new_w, new_h, settings.target_fps as u32);
            if is_cbr {
                let bps = (settings.video_bitrate_kbps.max(0) as u32).saturating_mul(1000);
                self.encode_config.rcParams.averageBitRate = bps;
                self.encode_config.rcParams.maxBitRate = bps;
                self.encode_config.rcParams.vbvBufferSize = crate::encoders::vbv_bits(
                    bps,
                    settings.target_fps,
                    settings.keyframe_interval_s,
                    settings.video_vbv_multiplier,
                );
            } else {
                let qp = settings.video_crf as u32;
                self.encode_config.rcParams.constQP.qpInterP = qp;
                self.encode_config.rcParams.constQP.qpInterB = qp;
                self.encode_config.rcParams.constQP.qpIntra = qp;
                self.current_qp = qp;
            }
            self.init_params.encodeWidth = new_w;
            self.init_params.encodeHeight = new_h;
            self.init_params.darWidth = new_w;
            self.init_params.darHeight = new_h;
            self.init_params.frameRateNum = (settings.target_fps.max(1.0)) as u32;
            self.init_params.frameRateDen = 1;
            self.init_params.encodeConfig = &mut self.encode_config;
            let mut reconfig_params = NV_ENC_RECONFIGURE_PARAMS {
                version: sv(NvStruct::ReconfigureParams),
                reInitEncodeParams: self.init_params,
                ..Default::default()
            };
            reconfig_params.set_resetEncoder(1);
            reconfig_params.set_forceIDR(1);
            let reconfig_fn = self.nvenc_funcs.nvEncReconfigureEncoder.unwrap();
            if reconfig_fn(self.encoder_session, &mut reconfig_params)
                != NVENCSTATUS::NV_ENC_SUCCESS
            {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("NvEncReconfigureEncoder rejected the resolution change".into());
            }
            self.width = new_w;
            self.height = new_h;

            let mut input_device_ptr: CUdeviceptr = 0;
            let mut input_pitch: usize = 0;
            let res = (self.cuda.cuMemAllocPitch_v2)(
                &mut input_device_ptr,
                &mut input_pitch,
                (new_w * 4) as usize,
                new_h as usize,
                16,
            );
            if res != CUresult::CUDA_SUCCESS {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("Failed to allocate ARGB input buffer on GPU".into());
            }
            let mut reg_res = NV_ENC_REGISTER_RESOURCE {
                version: sv(NvStruct::RegisterResource),
                resourceType: NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                width: new_w,
                height: new_h,
                resourceToRegister: input_device_ptr as *mut c_void,
                pitch: input_pitch as u32,
                bufferFormat: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
                bufferUsage: NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE,
                ..Default::default()
            };
            let register_fn = self.nvenc_funcs.nvEncRegisterResource.unwrap();
            if register_fn(self.encoder_session, &mut reg_res) != NVENCSTATUS::NV_ENC_SUCCESS {
                (self.cuda.cuMemFree_v2)(input_device_ptr);
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("Failed to register input buffer".into());
            }
            let mut map_params = NV_ENC_MAP_INPUT_RESOURCE {
                version: sv(NvStruct::MapInputResource),
                registeredResource: reg_res.registeredResource,
                ..Default::default()
            };
            let map_fn = self.nvenc_funcs.nvEncMapInputResource.unwrap();
            if map_fn(self.encoder_session, &mut map_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                    self.encoder_session,
                    reg_res.registeredResource,
                );
                (self.cuda.cuMemFree_v2)(input_device_ptr);
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("Failed to map input buffer".into());
            }
            self.input_device_ptr = input_device_ptr;
            self.input_pitch = input_pitch;
            self.registered_input_resource = reg_res.registeredResource;
            self.mapped_input_buffer = map_params.mappedResource;
            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
        }
        self.omit_stripe_headers = settings.omit_stripe_headers;
        Ok(())
    }

    /// Drop every page-locked host registration, under the pushed CUDA context.
    ///
    /// Called when the capture's shm segments are recreated at unchanged dimensions: the new
    /// segments often reuse the old base addresses, so a stale registration would alias fresh memory.
    /// Subsequent uploads re-pin lazily. A `0`-length entry marks a registration that failed and so
    /// is not unregistered.
    pub fn release_pinned_hosts(&mut self) {
        if self.pinned_hosts.is_empty() {
            return;
        }
        unsafe {
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);
            for (base, len) in self.pinned_hosts.drain() {
                if len > 0 {
                    (self.cuda.cuMemHostUnregister)(base as *mut c_void);
                }
            }
            let _ = (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
        }
    }

    /// Reconfigure the live session's ConstQP when `target_qp` differs from the current QP,
    /// returning whether a reconfigure actually happened.
    ///
    /// A no-op in CBR mode (bitrate-controlled, so QP-based paint-over does not apply) and when the
    /// QP is unchanged. When it does apply, the three `constQP` fields are updated and the session is
    /// reconfigured **without** a forced IDR: a lower-QP P-frame refines the static image against the
    /// existing reference chain (paint-over) with no intra-frame bitrate spike, so the GOP continues
    /// seamlessly across the reconfigure.
    unsafe fn reconfigure_if_needed(&mut self, target_qp: u32) -> bool {
        if self.encode_config.rcParams.rateControlMode
            == NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR
        {
            return false;
        }
        if self.current_qp != target_qp {
            self.encode_config.rcParams.constQP.qpInterP = target_qp;
            self.encode_config.rcParams.constQP.qpInterB = target_qp;
            self.encode_config.rcParams.constQP.qpIntra = target_qp;
            self.init_params.encodeConfig = &mut self.encode_config;

            let mut reconfig_params = NV_ENC_RECONFIGURE_PARAMS {
                version: sv(NvStruct::ReconfigureParams),
                reInitEncodeParams: self.init_params,
                ..Default::default()
            };

            let reconfig_fn = self.nvenc_funcs.nvEncReconfigureEncoder.unwrap();
            if reconfig_fn(self.encoder_session, &mut reconfig_params)
                == NVENCSTATUS::NV_ENC_SUCCESS
            {
                self.current_qp = target_qp;
                return true;
            } else {
                eprintln!("[NVENC] Reconfigure failed.");
            }
        }
        false
    }

    /// Apply a runtime rate-control / frame-rate change to the live session.
    ///
    /// In CBR mode the target bitrate, max bitrate and VBV buffer size are updated (the VBV is
    /// ignored outside CBR); the target fps is updated in either mode. The session is reconfigured
    /// only when one of these actually changed — no forced IDR, no RC reset — so calling it every
    /// frame is cheap.
    pub fn reconfigure_rate(&mut self, settings: &RustCaptureSettings) {
        unsafe {
            let mut changed = false;
            if self.encode_config.rcParams.rateControlMode
                == NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR
            {
                let bps = (settings.video_bitrate_kbps.max(0) as u32).saturating_mul(1000);
                let vbv = crate::encoders::vbv_bits(
                    bps,
                    settings.target_fps,
                    settings.keyframe_interval_s,
                    settings.video_vbv_multiplier,
                );
                if self.encode_config.rcParams.averageBitRate != bps
                    || self.encode_config.rcParams.maxBitRate != bps
                    || self.encode_config.rcParams.vbvBufferSize != vbv
                {
                    self.encode_config.rcParams.averageBitRate = bps;
                    self.encode_config.rcParams.maxBitRate = bps;
                    self.encode_config.rcParams.vbvBufferSize = vbv;
                    changed = true;
                }
            }
            let fps = (settings.target_fps.max(1.0)) as u32;
            if self.init_params.frameRateNum != fps {
                self.init_params.frameRateNum = fps;
                self.init_params.frameRateDen = 1;
                self.encode_config.encodeCodecConfig.h264Config.level = nvenc_h264_level(
                    self.init_params.encodeWidth,
                    self.init_params.encodeHeight,
                    fps,
                );
                changed = true;
            }
            if !changed {
                return;
            }
            self.init_params.encodeConfig = &mut self.encode_config;
            let mut reconfig_params = NV_ENC_RECONFIGURE_PARAMS {
                version: sv(NvStruct::ReconfigureParams),
                reInitEncodeParams: self.init_params,
                ..Default::default()
            };
            let reconfig_fn = self.nvenc_funcs.nvEncReconfigureEncoder.unwrap();
            if reconfig_fn(self.encoder_session, &mut reconfig_params)
                != NVENCSTATUS::NV_ENC_SUCCESS
            {
                eprintln!("[NVENC] Rate reconfigure failed.");
            }
        }
    }

    /// Encode one mapped input picture and return its bitstream bytes behind the wire header.
    ///
    /// The shared tail of all three encode paths:
    ///
    /// 1. **Pick an output buffer** from the ring (`current_buffer_idx` advances modulo the ring
    ///    length) and submit the picture with `nvEncEncodePicture`; `force_idr` sets the force-IDR
    ///    pic flag.
    /// 2. **Lock the bitstream** (`nvEncLockBitstream`, blocking) to read the encoded bytes.
    /// 3. **Frame the output**: unless `omit_stripe_headers` is set, prepend the 10-byte wire header
    ///    — a `0x04` tag, a picture-type byte derived from the *actual* encoded `pictureType`
    ///    (IDR = `0x01`, I = `0x02`, P = `0x00`) rather than the `force_idr` request, the low 16 bits
    ///    of the frame number, a zero field, and the width and height (all big-endian).
    /// 4. **Emit**: append the encoded bytes, unlock the bitstream, and return the framed buffer.
    unsafe fn submit_frame(
        &mut self,
        mapped_buffer: NV_ENC_INPUT_PTR,
        buffer_format: NV_ENC_BUFFER_FORMAT,
        frame_number: u64,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        let output_bitstream = self.bitstream_buffers[self.current_buffer_idx];
        self.current_buffer_idx = (self.current_buffer_idx + 1) % self.bitstream_buffers.len();

        let mut pic_params = NV_ENC_PIC_PARAMS {
            version: sv(NvStruct::PicParams),
            inputWidth: self.width,
            inputHeight: self.height,
            inputBuffer: mapped_buffer,
            outputBitstream: output_bitstream,
            bufferFmt: buffer_format,
            pictureStruct: NV_ENC_PIC_STRUCT::NV_ENC_PIC_STRUCT_FRAME,
            encodePicFlags: if force_idr {
                NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_FORCEIDR as u32
            } else {
                0
            },
            ..Default::default()
        };

        let encode_fn = self.nvenc_funcs.nvEncEncodePicture.unwrap();
        let res = encode_fn(self.encoder_session, &mut pic_params);
        if res != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("Encode Picture failed: {:?}", res));
        }

        let mut lock_params = NV_ENC_LOCK_BITSTREAM {
            version: sv(NvStruct::LockBitstream),
            outputBitstream: output_bitstream,
            ..Default::default()
        };
        lock_params.set_doNotWait(0);

        let lock_fn = self.nvenc_funcs.nvEncLockBitstream.unwrap();
        if lock_fn(self.encoder_session, &mut lock_params) != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err("Lock Bitstream failed".into());
        }

        let data_ptr = lock_params.bitstreamBufferPtr as *const u8;
        let data_size = lock_params.bitstreamSizeInBytes as usize;
        let header_sz = if self.omit_stripe_headers { 0 } else { 10 };
        let mut output = Vec::with_capacity(header_sz + data_size);

        if !self.omit_stripe_headers {
            let type_hdr = match lock_params.pictureType {
                NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR => 0x01u8,
                NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_I => 0x02u8,
                _ => 0x00u8,
            };
            output.push(0x04);
            output.push(type_hdr);
            output.extend_from_slice(&(frame_number as u16).to_be_bytes());
            output.extend_from_slice(&0u16.to_be_bytes());
            output.extend_from_slice(&(self.width as u16).to_be_bytes());
            output.extend_from_slice(&(self.height as u16).to_be_bytes());
        }

        if data_size > 0 && !data_ptr.is_null() {
            let slice = std::slice::from_raw_parts(data_ptr, data_size);
            output.extend_from_slice(slice);
        }

        (self.nvenc_funcs.nvEncUnlockBitstream.unwrap())(self.encoder_session, output_bitstream);
        Ok(output)
    }

    /// Encode a dmabuf frame zero-copy, by importing it through EGL into CUDA.
    ///
    /// Applies any pending ConstQP change, then works under the pushed CUDA context:
    ///
    /// 1. **Import once, cache by fd**: on a cache miss, build an `EGLImageKHR` from the dmabuf's
    ///    fd / offset / pitch / modifier, register it as a CUDA graphics resource, and map it to a
    ///    `CUeglFrame`; the result is memoized in `dmabuf_cache` so a recurring capture buffer pays
    ///    the import cost only once. Each failure destroys what it created and pops the context.
    /// 2. **Copy into the ARGB input**: a `cuMemcpy2D` from the mapped frame — the array plane or the
    ///    pitch-linear plane, per `frame_type` — into the registered ARGB surface, sanitizing pitch
    ///    and format for NVENC.
    /// 3. **Submit** the ARGB input via `submit_frame`, then pop the context.
    ///
    /// The dmabuf fd is read out before the context is pushed so an early `?` return cannot leave the
    /// CUDA context stack imbalanced.
    pub fn encode(
        &mut self,
        dmabuf: &Dmabuf,
        frame_number: u64,
        target_qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.reconfigure_if_needed(target_qp);
            let fd = dmabuf.handles().next().ok_or("No handles")?.as_raw_fd();
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);

            if !self.dmabuf_cache.contains_key(&fd) {
                let stride = dmabuf.strides().next().unwrap_or(0) as i32;
                let offset = dmabuf.offsets().next().unwrap_or(0) as i32;
                let fmt = dmabuf.format();
                let modifier: u64 = fmt.modifier.into();

                let attribs = [
                    EGL_WIDTH,
                    self.width as i32,
                    EGL_HEIGHT,
                    self.height as i32,
                    EGL_LINUX_DRM_FOURCC_EXT,
                    fmt.code as i32,
                    EGL_DMA_BUF_PLANE0_FD_EXT,
                    fd,
                    EGL_DMA_BUF_PLANE0_OFFSET_EXT,
                    offset,
                    EGL_DMA_BUF_PLANE0_PITCH_EXT,
                    stride,
                    EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                    (modifier & 0xFFFFFFFF) as i32,
                    EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                    (modifier >> 32) as i32,
                    EGL_NONE,
                ];

                let egl_image = (self.egl.eglCreateImageKHR)(
                    self.egl_display,
                    ptr::null_mut(),
                    EGL_LINUX_DMA_BUF_EXT,
                    ptr::null_mut(),
                    attribs.as_ptr(),
                );
                if egl_image == EGL_NO_IMAGE_KHR {
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to create EGLImage".into());
                }

                let mut cuda_resource: CUgraphicsResource = ptr::null_mut();
                if (self.cuda.cuGraphicsEGLRegisterImage)(&mut cuda_resource, egl_image, 1)
                    != CUresult::CUDA_SUCCESS
                {
                    (self.egl.eglDestroyImageKHR)(self.egl_display, egl_image);
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to register EGLImage".into());
                }

                let mut egl_frame: CUeglFrame = std::mem::zeroed();
                if (self.cuda.cuGraphicsResourceGetMappedEglFrame)(
                    &mut egl_frame,
                    cuda_resource,
                    0,
                    0,
                ) != CUresult::CUDA_SUCCESS
                {
                    (self.cuda.cuGraphicsUnregisterResource)(cuda_resource);
                    (self.egl.eglDestroyImageKHR)(self.egl_display, egl_image);
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to map EGL frame".into());
                }

                self.dmabuf_cache.insert(
                    fd,
                    CachedDmaBuf {
                        egl_image,
                        cuda_resource,
                        egl_frame,
                    },
                );
            }

            let cached = self.dmabuf_cache.get(&fd).unwrap();
            let mut copy_params = CUDA_MEMCPY2D {
                srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcHost: ptr::null(),
                srcDevice: 0,
                srcArray: ptr::null_mut(),
                srcPitch: 0,
                dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstHost: ptr::null_mut(),
                dstDevice: self.input_device_ptr,
                dstArray: ptr::null_mut(),
                dstPitch: self.input_pitch,
                WidthInBytes: (self.width * 4) as usize,
                Height: self.height as usize,
                ..Default::default()
            };

            if cached.egl_frame.frame_type == 0 {
                copy_params.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
                copy_params.srcArray = cached.egl_frame.frame.p_array[0];
            } else {
                copy_params.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
                copy_params.srcDevice = cached.egl_frame.frame.p_pitch[0] as CUdeviceptr;
                copy_params.srcPitch = cached.egl_frame.pitch as usize;
            }

            if (self.cuda.cuMemcpy2D_v2)(&copy_params) != CUresult::CUDA_SUCCESS {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("Sanitization copy failed".into());
            }

            let result = self.submit_frame(
                self.mapped_input_buffer,
                NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
                frame_number,
                force_idr,
            );
            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
            result
        }
    }

    /// Encode a host ARGB frame by uploading it straight into the ARGB input surface, with no
    /// CPU-side colour conversion. NVENC's hardware ARGB→YUV conversion is fixed at BT.601
    /// limited range, which is what the session VUI declares; a CPU prepass to BT.709 would cost
    /// this path its zero-copy property.
    ///
    /// The packed rows are copied host→device into the registered ARGB surface and NVENC's hardware
    /// CSC produces the YUV. Bytes must be in NVENC ARGB order (`B,G,R,A` in memory) — the host BGRA
    /// layout an XShm grab yields — and `src_stride` is the source row stride in bytes (`>= width*4`).
    /// Steps, under the pushed CUDA context after any pending QP change:
    ///
    /// 1. **Bounds-check** the source against `stride × (rows-1) + width*4`, erroring rather than
    ///    reading past a short buffer.
    /// 2. **Pin the source once**: unless `PIXELFLUX_NVENC_PIN=0`, page-lock each distinct source
    ///    base address (`cuMemHostRegister`) so the copy is a direct pinned DMA instead of a pageable
    ///    copy staged through a driver bounce buffer. The persistent, bounded shm sources make this a
    ///    one-time bounded cost; a failed registration is recorded as length `0` and never re-probed.
    /// 3. **Copy and submit**: `cuMemcpy2D` the rows into the ARGB surface honoring `src_stride`,
    ///    then `submit_frame`.
    pub fn encode_cpu_argb(
        &mut self,
        argb: &[u8],
        src_stride: usize,
        frame_number: u64,
        target_qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.reconfigure_if_needed(target_qp);
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);

            let width_bytes = (self.width * 4) as usize;
            let rows = self.height as usize;
            let needed = if rows == 0 { 0 } else { src_stride * (rows - 1) + width_bytes };
            if src_stride < width_bytes || argb.len() < needed {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err(format!(
                    "ARGB buffer too small: len={} need>={} (stride={}, {}x{})",
                    argb.len(), needed, src_stride, self.width, self.height
                ));
            }

            if std::env::var("PIXELFLUX_NVENC_PIN").as_deref() != Ok("0") {
                let base = argb.as_ptr() as usize;
                if let std::collections::hash_map::Entry::Vacant(e) = self.pinned_hosts.entry(base) {
                    let st = (self.cuda.cuMemHostRegister_v2)(argb.as_ptr() as *mut c_void, argb.len(), 0);
                    e.insert(if st == CUresult::CUDA_SUCCESS { argb.len() } else { 0 });
                }
            }

            let copy = CUDA_MEMCPY2D {
                srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                srcHost: argb.as_ptr() as *const c_void,
                srcPitch: src_stride,
                dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstDevice: self.input_device_ptr,
                dstPitch: self.input_pitch,
                WidthInBytes: width_bytes,
                Height: rows,
                ..Default::default()
            };
            if (self.cuda.cuMemcpy2D_v2)(&copy) != CUresult::CUDA_SUCCESS {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("ARGB host->device copy failed".into());
            }

            let result = self.submit_frame(
                self.mapped_input_buffer,
                NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
                frame_number,
                force_idr,
            );
            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
            result
        }
    }

    /// Encode a raw planar frame — NV12 (4:2:0) or YUV444 — uploaded host→device.
    ///
    /// The planar-input counterpart to `encode_cpu_argb`, used when the caller has already produced
    /// YUV. The chroma format follows the session's `chromaFormatIDC` (3 ⇒ YUV444, else NV12). Flow
    /// under the pushed CUDA context after any pending QP change:
    ///
    /// 1. **Lazily allocate** the planar device buffer on first use: a pitched allocation tall enough
    ///    for three full planes (YUV444) or Y plus half-height interleaved UV (NV12), registered and
    ///    mapped with the matching buffer format.
    /// 2. **Upload each plane** with its own `cuMemcpy2D`. Every copy is bounds-checked against the
    ///    host slice: the Y plane is required in full (a short buffer errors), and each chroma plane
    ///    is copied only if its **entire** span — not merely its start offset — is present, so a
    ///    truncated buffer never reads past its end.
    /// 3. **Submit** the mapped planar input via `submit_frame`.
    pub fn encode_raw(
        &mut self,
        raw_data: &[u8],
        frame_number: u64,
        target_qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.reconfigure_if_needed(target_qp);
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);

            let is_444 = self.encode_config.encodeCodecConfig.h264Config.chromaFormatIDC == 3;

            if self.nv12_device_ptr.is_none() {
                let mut d_ptr: CUdeviceptr = 0;
                let mut pitch: usize = 0;

                let alloc_height = if is_444 {
                    self.height * 3
                } else {
                    self.height + (self.height / 2)
                };

                let res = (self.cuda.cuMemAllocPitch_v2)(
                    &mut d_ptr,
                    &mut pitch,
                    self.width as usize,
                    alloc_height as usize,
                    16,
                );
                if res != CUresult::CUDA_SUCCESS {
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to allocate GPU buffer for raw input".into());
                }

                let buffer_fmt = if is_444 {
                    NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_YUV444
                } else {
                    NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12
                };

                let mut reg_res = NV_ENC_REGISTER_RESOURCE {
                    version: sv(NvStruct::RegisterResource),
                    resourceType:
                        NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                    width: self.width,
                    height: self.height,
                    resourceToRegister: d_ptr as *mut c_void,
                    pitch: pitch as u32,
                    bufferFormat: buffer_fmt,
                    bufferUsage: NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE,
                    ..Default::default()
                };

                let register_fn = self.nvenc_funcs.nvEncRegisterResource.unwrap();
                if register_fn(self.encoder_session, &mut reg_res) != NVENCSTATUS::NV_ENC_SUCCESS {
                    (self.cuda.cuMemFree_v2)(d_ptr);
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to register raw input buffer".into());
                }

                let mut map_params = NV_ENC_MAP_INPUT_RESOURCE {
                    version: sv(NvStruct::MapInputResource),
                    registeredResource: reg_res.registeredResource,
                    ..Default::default()
                };
                let map_fn = self.nvenc_funcs.nvEncMapInputResource.unwrap();
                if map_fn(self.encoder_session, &mut map_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                    (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                        self.encoder_session,
                        reg_res.registeredResource,
                    );
                    (self.cuda.cuMemFree_v2)(d_ptr);
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to map raw input buffer".into());
                }

                self.nv12_device_ptr = Some(d_ptr);
                self.nv12_pitch = pitch;
                self.nv12_registered_resource = Some(reg_res.registeredResource);
                self.nv12_mapped_buffer = Some(map_params.mappedResource);
            }

            let dev_ptr = self.nv12_device_ptr.unwrap();
            let dev_pitch = self.nv12_pitch;
            let width_bytes = self.width as usize;
            let height = self.height as usize;

            if is_444 {
                let plane_size = width_bytes * height;
                if raw_data.len() < plane_size {
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("raw frame smaller than the Y plane (444)".into());
                }

                let copy_y = CUDA_MEMCPY2D {
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: raw_data.as_ptr() as *const c_void,
                    srcPitch: width_bytes,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstDevice: dev_ptr,
                    dstPitch: dev_pitch,
                    WidthInBytes: width_bytes,
                    Height: height,
                    ..Default::default()
                };
                if (self.cuda.cuMemcpy2D_v2)(&copy_y) != CUresult::CUDA_SUCCESS {
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to copy Y plane (444)".into());
                }

                if raw_data.len() >= 2 * plane_size {
                    let copy_u = CUDA_MEMCPY2D {
                        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                        srcHost: raw_data[plane_size..].as_ptr() as *const c_void,
                        srcPitch: width_bytes,
                        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                        dstDevice: dev_ptr + (dev_pitch * height) as u64,
                        dstPitch: dev_pitch,
                        WidthInBytes: width_bytes,
                        Height: height,
                        ..Default::default()
                    };
                    if (self.cuda.cuMemcpy2D_v2)(&copy_u) != CUresult::CUDA_SUCCESS {
                        (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                        return Err("Failed to copy U plane (444)".into());
                    }
                }

                if raw_data.len() >= 3 * plane_size {
                    let copy_v = CUDA_MEMCPY2D {
                        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                        srcHost: raw_data[2 * plane_size..].as_ptr() as *const c_void,
                        srcPitch: width_bytes,
                        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                        dstDevice: dev_ptr + (dev_pitch * height * 2) as u64,
                        dstPitch: dev_pitch,
                        WidthInBytes: width_bytes,
                        Height: height,
                        ..Default::default()
                    };
                    if (self.cuda.cuMemcpy2D_v2)(&copy_v) != CUresult::CUDA_SUCCESS {
                        (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                        return Err("Failed to copy V plane (444)".into());
                    }
                }
            } else {
                let y_size = width_bytes * height;
                if raw_data.len() < y_size {
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("raw frame smaller than the Y plane".into());
                }
                let copy_y = CUDA_MEMCPY2D {
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: raw_data.as_ptr() as *const c_void,
                    srcPitch: width_bytes,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstDevice: dev_ptr,
                    dstPitch: dev_pitch,
                    WidthInBytes: width_bytes,
                    Height: height,
                    ..Default::default()
                };
                if (self.cuda.cuMemcpy2D_v2)(&copy_y) != CUresult::CUDA_SUCCESS {
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to copy Y plane".into());
                }

                let uv_offset = y_size;
                if raw_data.len() >= uv_offset + width_bytes * (height / 2) {
                    let copy_uv = CUDA_MEMCPY2D {
                        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                        srcHost: raw_data[uv_offset..].as_ptr() as *const c_void,
                        srcPitch: width_bytes,
                        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                        dstDevice: dev_ptr + (dev_pitch * height) as u64,
                        dstPitch: dev_pitch,
                        WidthInBytes: width_bytes,
                        Height: height / 2,
                        ..Default::default()
                    };
                    if (self.cuda.cuMemcpy2D_v2)(&copy_uv) != CUresult::CUDA_SUCCESS {
                        (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                        return Err("Failed to copy UV plane".into());
                    }
                }
            }

            let raw_format = if self.encode_config.encodeCodecConfig.h264Config.chromaFormatIDC == 3 {
                NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_YUV444
            } else {
                NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12
            };
            let result =
                self.submit_frame(self.nv12_mapped_buffer.unwrap(), raw_format, frame_number, force_idr);
            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
            result
        }
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    /// Test helper: H.264 full-frame capture settings at `w×h`, `fps`, CRF 25.
    fn settings(w: i32, h: i32, fps: f64) -> RustCaptureSettings {
        RustCaptureSettings {
            width: w,
            height: h,
            output_mode: 1,
            target_fps: fps,
            video_crf: 25,
            ..Default::default()
        }
    }

    /// Test helper: a `w×h` BGRA frame filled with a hashed gradient (offset by `seed`) so
    /// the content has structure and encodes are non-trivial.
    fn frame(w: usize, h: usize, seed: u8) -> Vec<u8> {
        let mut f = vec![0u8; w * h * 4];
        for (i, px) in f.chunks_exact_mut(4).enumerate() {
            let v = ((i as u32).wrapping_mul(2654435761) >> 24) as u8;
            px[0] = v.wrapping_add(seed);
            px[1] = v ^ seed;
            px[2] = seed;
            px[3] = 255;
        }
        f
    }

    /// Test helper: read the big-endian width/height (bytes 6-9) from a 10-byte wire header.
    fn wire_dims(pkt: &[u8]) -> (u16, u16) {
        (
            u16::from_be_bytes([pkt[6], pkt[7]]),
            u16::from_be_bytes([pkt[8], pkt[9]]),
        )
    }

    /// End-to-end in-place resize on a real GPU: encode 720p, grow to 1080p and verify the
    /// first post-resize frame is an IDR (steady frames are P), shrink to 480p, exercise the
    /// rejection cases (beyond headroom, chroma flip, RC-mode flip) and confirm the session still
    /// encodes afterward, and optionally dump a decodable stream. Ignored by default; run with
    /// `cargo test gpu_ -- --ignored --nocapture --test-threads=1` — building NVENC sessions
    /// concurrently races in the driver and intermittently faults, so the GPU set runs serially.
    #[test]
    #[ignore]
    fn gpu_resolution_reconfigure_roundtrip() {
        let mut s = settings(1280, 720, 60.0);
        let t0 = std::time::Instant::now();
        let mut enc = NvencEncoder::new(&s, ptr::null()).expect("NVENC init");
        let init_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut stream: Vec<u8> = Vec::new();
        let f720 = frame(1280, 720, 10);
        for i in 0..5u64 {
            let pkt = enc
                .encode_cpu_argb(&f720, 1280 * 4, i, 25, i == 0)
                .expect("encode 720p");
            assert_eq!(wire_dims(&pkt), (1280, 720));
            stream.extend_from_slice(&pkt[10..]);
        }

        s.width = 1920;
        s.height = 1080;
        let t1 = std::time::Instant::now();
        enc.reconfigure_resolution(&s).expect("grow reconfigure");
        let grow_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let f1080 = frame(1920, 1080, 40);
        let pkt = enc
            .encode_cpu_argb(&f1080, 1920 * 4, 5, 25, false)
            .expect("encode 1080p");
        assert_eq!(pkt[0], 0x04);
        assert_eq!(pkt[1], 0x01, "first frame after a resize must be an IDR");
        assert_eq!(wire_dims(&pkt), (1920, 1080));
        stream.extend_from_slice(&pkt[10..]);
        for i in 6..10u64 {
            let pkt = enc
                .encode_cpu_argb(&f1080, 1920 * 4, i, 25, false)
                .expect("encode 1080p");
            assert_eq!(pkt[1], 0x00, "steady frames after the IDR are P frames");
            stream.extend_from_slice(&pkt[10..]);
        }

        s.width = 640;
        s.height = 480;
        let t2 = std::time::Instant::now();
        enc.reconfigure_resolution(&s).expect("shrink reconfigure");
        let shrink_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let f480 = frame(640, 480, 70);
        let pkt = enc
            .encode_cpu_argb(&f480, 640 * 4, 10, 25, false)
            .expect("encode 480p");
        assert_eq!(pkt[1], 0x01);
        assert_eq!(wire_dims(&pkt), (640, 480));
        stream.extend_from_slice(&pkt[10..]);

        s.width = 4100;
        s.height = 2400;
        assert!(enc.reconfigure_resolution(&s).is_err(), "beyond headroom");
        s.width = 640;
        s.height = 480;
        s.video_fullcolor = true;
        assert!(enc.reconfigure_resolution(&s).is_err(), "chroma flip");
        s.video_fullcolor = false;
        s.video_cbr_mode = true;
        assert!(enc.reconfigure_resolution(&s).is_err(), "RC mode flip");
        s.video_cbr_mode = false;
        let pkt = enc
            .encode_cpu_argb(&f480, 640 * 4, 11, 25, false)
            .expect("session survives rejected reconfigures");
        stream.extend_from_slice(&pkt[10..]);

        println!(
            "init={init_ms:.1}ms grow(720p->1080p)={grow_ms:.1}ms shrink(1080p->480p)={shrink_ms:.1}ms"
        );
        if let Ok(path) = std::env::var("NVENC_TEST_DUMP") {
            std::fs::write(&path, &stream).unwrap();
            println!("wrote {} bytes to {path}", stream.len());
        }
    }

    /// On a real GPU, a CBR session resized 720p→1080p folds the new bitrate into the resize
    /// reconfigure (asserts `averageBitRate` updated to 8 Mbit/s) and the first post-resize frame is
    /// an IDR at the new dimensions. Ignored by default.
    #[test]
    #[ignore]
    fn gpu_resolution_reconfigure_cbr() {
        let mut s = settings(1280, 720, 60.0);
        s.video_cbr_mode = true;
        s.video_bitrate_kbps = 4000;
        let mut enc = NvencEncoder::new(&s, ptr::null()).expect("NVENC init");
        let f720 = frame(1280, 720, 10);
        for i in 0..3u64 {
            enc.encode_cpu_argb(&f720, 1280 * 4, i, 25, i == 0)
                .expect("encode 720p");
        }
        s.width = 1920;
        s.height = 1080;
        s.video_bitrate_kbps = 8000;
        enc.reconfigure_resolution(&s).expect("cbr resize+rate");
        assert_eq!(enc.encode_config.rcParams.averageBitRate, 8_000_000);
        let f1080 = frame(1920, 1080, 40);
        let pkt = enc
            .encode_cpu_argb(&f1080, 1920 * 4, 3, 25, false)
            .expect("encode 1080p");
        assert_eq!(pkt[1], 0x01);
        assert_eq!(wire_dims(&pkt), (1920, 1080));
    }

    /// On a real GPU, print the device-memory cost of one 1080p session (via `nvidia-smi`),
    /// for measuring the reconfigure-headroom overhead. Ignored by default.
    #[test]
    #[ignore]
    fn gpu_vram_probe() {
        fn used_mb() -> i64 {
            let out = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
                .output()
                .expect("nvidia-smi");
            String::from_utf8_lossy(&out.stdout).trim().parse().expect("parse MiB")
        }
        let s = settings(1920, 1080, 60.0);
        let before = used_mb();
        let mut enc = NvencEncoder::new(&s, ptr::null()).expect("init");
        let f = frame(1920, 1080, 5);
        for i in 0..3u64 {
            enc.encode_cpu_argb(&f, 1920 * 4, i, 25, i == 0).expect("encode");
        }
        println!("VRAM delta for one 1080p session: {} MiB", used_mb() - before);
    }

    /// On a real GPU, a session that starts taller than the default 2304 headroom (portrait
    /// 4K: 2160×4096, within NVENC's 4096 H.264 cap) takes its own size as the `maxEncode` ceiling
    /// and encodes at that resolution. Ignored by default.
    /// The VUI has to describe what the session actually emits. NVENC converts its ARGB
    /// input with a fixed BT.601 matrix at limited range in both chroma formats, so the
    /// matrix and range follow the hardware; the primaries and transfer follow the source,
    /// which is sRGB desktop pixels and therefore BT.709. A client that inverts the wrong
    /// matrix, or expands a limited-range frame as full-range, shifts colour visibly.
    #[test]
    #[ignore]
    fn gpu_vui_describes_the_hardware_csc() {
        for fullcolor in [false, true] {
            let mut s = settings(1280, 720, 60.0);
            s.video_fullcolor = fullcolor;
            let enc = NvencEncoder::new(&s, ptr::null()).expect("NVENC init");
            // encodeCodecConfig is a union; a successful init leaves the H.264 arm live.
            let h264 = unsafe { &enc.encode_config.encodeCodecConfig.h264Config };
            let vui = &h264.h264VUIParameters;
            assert_eq!(vui.colourMatrix as u32,
                NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_SMPTE170M as u32);
            assert_eq!(vui.colourPrimaries as u32,
                NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709 as u32);
            assert_eq!(vui.transferCharacteristics as u32,
                NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709 as u32);
            assert_eq!(vui.videoFullRangeFlag, 0, "fullcolor={fullcolor}");
            assert_eq!(h264.chromaFormatIDC, if fullcolor { 3 } else { 1 });
        }
    }

    #[test]
    #[ignore]
    fn gpu_init_above_default_headroom() {
        let s = settings(2160, 4096, 30.0);
        let mut enc = NvencEncoder::new(&s, ptr::null()).expect("NVENC init portrait 4K");
        assert_eq!(enc.init_params.maxEncodeWidth, 4096);
        assert_eq!(enc.init_params.maxEncodeHeight, 4096);
        let f = frame(2160, 4096, 20);
        let pkt = enc
            .encode_cpu_argb(&f, 2160 * 4, 0, 25, true)
            .expect("encode portrait 4K");
        assert_eq!(wire_dims(&pkt), (2160, 4096));
    }
}

#[cfg(test)]
mod version_tests {
    use super::*;

    /// Every version-tagged struct, in one fixed order, paired with its pinned compile-time
    /// `NV_ENC_*_VER` constant — the reference both version tests iterate.
    const ALL: [(NvStruct, u32); 12] = [
        (NvStruct::FunctionList, NV_ENCODE_API_FUNCTION_LIST_VER),
        (NvStruct::OpenSessionExParams, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER),
        (NvStruct::Config, NV_ENC_CONFIG_VER),
        (NvStruct::RcParams, NV_ENC_RC_PARAMS_VER),
        (NvStruct::PresetConfig, NV_ENC_PRESET_CONFIG_VER),
        (NvStruct::InitializeParams, NV_ENC_INITIALIZE_PARAMS_VER),
        (NvStruct::ReconfigureParams, NV_ENC_RECONFIGURE_PARAMS_VER),
        (NvStruct::RegisterResource, NV_ENC_REGISTER_RESOURCE_VER),
        (NvStruct::MapInputResource, NV_ENC_MAP_INPUT_RESOURCE_VER),
        (NvStruct::CreateBitstreamBuffer, NV_ENC_CREATE_BITSTREAM_BUFFER_VER),
        (NvStruct::PicParams, NV_ENC_PIC_PARAMS_VER),
        (NvStruct::LockBitstream, NV_ENC_LOCK_BITSTREAM_VER),
    ];

    /// For the pinned nvcodec-sys version, `nvenc_struct_ver` must reproduce every
    /// compile-time `NV_ENC_*_VER` constant exactly — guaranteeing a current driver is stamped
    /// byte-for-byte identically — and the packed major/minor must round-trip `NVENCAPI_VERSION`.
    /// Fails loudly if the bundled header is bumped without extending the revision table.
    #[test]
    fn table_is_identity_for_pinned_version() {
        let maj = NVENCAPI_VERSION & 0xFF;
        let min = (NVENCAPI_VERSION >> 24) & 0xFF;
        for (s, base) in ALL {
            assert_eq!(nvenc_struct_ver(s, maj, min), base, "{:?}", s);
        }
        assert_eq!(maj | (min << 24), NVENCAPI_VERSION);
    }

    /// `nvenc_struct_ver` must reproduce the exact `NV_ENC_*_VER` words each SDK defined, for
    /// every negotiable version 10.0 through 13.0.
    ///
    /// The expected words are hardcoded from `nvEncodeAPI.h` at the FFmpeg nv-codec-headers tags
    /// listed in `NvStruct::rev`, one row per SDK version in `ALL` order. (The n10.0.26.2 header
    /// spells the flag `1<<31` rather than `1u<<31`; the bit is the same.) This is what lets the
    /// 13.0-layout structs be stamped with an older SDK's word when the session down-negotiates.
    #[test]
    fn table_matches_historical_headers() {
        #[rustfmt::skip]
        let expected: [(u32, u32, [u32; 12]); 7] = [
            (10, 0, [0x7002000A, 0x7001000A, 0xF007000A, 0x7001000A, 0xF004000A, 0xF005000A, 0xF001000A,
                     0x7003000A, 0x7004000A, 0x7001000A, 0xF004000A, 0x7001000A]),
            (11, 0, [0x7002000B, 0x7001000B, 0xF007000B, 0x7001000B, 0xF004000B, 0xF005000B, 0xF001000B,
                     0x7003000B, 0x7004000B, 0x7001000B, 0xF004000B, 0x7001000B]),
            (11, 1, [0x7102000B, 0x7101000B, 0xF107000B, 0x7101000B, 0xF104000B, 0xF105000B, 0xF101000B,
                     0x7103000B, 0x7104000B, 0x7101000B, 0xF104000B, 0x7101000B]),
            (12, 0, [0x7002000C, 0x7001000C, 0xF008000C, 0x7001000C, 0xF004000C, 0xF005000C, 0xF001000C,
                     0x7004000C, 0x7004000C, 0x7001000C, 0xF006000C, 0x7002000C]),
            (12, 1, [0x7102000C, 0x7101000C, 0xF108000C, 0x7101000C, 0xF104000C, 0xF106000C, 0xF101000C,
                     0x7104000C, 0x7104000C, 0x7101000C, 0xF106000C, 0xF101000C]),
            (12, 2, [0x7202000C, 0x7201000C, 0xF209000C, 0x7201000C, 0xF205000C, 0xF207000C, 0xF202000C,
                     0x7205000C, 0x7204000C, 0x7201000C, 0xF207000C, 0xF202000C]),
            (13, 0, [0x7002000D, 0x7001000D, 0xF009000D, 0x7001000D, 0xF005000D, 0xF007000D, 0xF002000D,
                     0x7005000D, 0x7004000D, 0x7001000D, 0xF007000D, 0xF002000D]),
        ];
        for (maj, min, words) in expected {
            for ((s, _), want) in ALL.iter().zip(words) {
                assert_eq!(
                    nvenc_struct_ver(*s, maj, min),
                    want,
                    "{:?} at {}.{}",
                    s,
                    maj,
                    min
                );
            }
        }
    }
}
