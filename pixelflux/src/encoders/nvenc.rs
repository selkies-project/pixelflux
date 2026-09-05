/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! NVENC hardware H.264 / HEVC / AV1 encoder: CUDA-bound sessions that encode packed BGRA or
//! RGBA frames from the X11 host and Wayland readback paths, or Wayland dmabufs in place.
//!
//! The module dynamically loads `libcuda`, `libnvidia-encode` and `libEGL` at runtime, negotiates
//! the NVENC API version against the installed driver (set-once per process), and stamps every
//! NVENCAPI struct with the exact `NV_ENC_*_VER` word the negotiated SDK defines, so one binary
//! drives drivers from NVENC 10.0 (~R445) through 13.0. Frames reach the GPU two ways: a
//! zero-copy dmabuf import (EGLImage → CUDA, the mapped plane registered with NVENC in place as
//! pitch-linear memory or as a CUDA array), and a pinned host→device upload of packed BGRA / RGBA
//! that the hardware CSC converts. The codec is a session parameter: the same rate control,
//! GOP, VUI and latency posture is programmed into whichever of the three codec configurations
//! the device offers, and a codec the device lacks (AV1 before Ada) is refused at open so the
//! caller falls back. Sessions reconfigure resolution and rate control in place, so a resize or
//! bitrate change costs a few milliseconds instead of a full rebuild.

// The NVENC and CUDA entry points are called through function pointers
// resolved at runtime, so the safety contract is carried by the function
// signatures rather than by a block around each call.
#![allow(unsafe_op_in_unsafe_fn)]

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::os::unix::io::AsRawFd;
use std::ptr;
use std::sync::Arc;

use libloading::{Library, Symbol};
use smithay::backend::allocator::{dmabuf::Dmabuf, Buffer, Fourcc};

use super::codec::{
    av1_level, h264_level, h265_level, push_video_header, Codec, FRAME_DELTA, FRAME_INTRA, FRAME_KEY,
    VIDEO_HEADER_LEN,
};
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

/// `CUeglFrame::frame_type` values: the mapped planes are CUDA arrays, or pitch-linear device
/// memory.
const CU_EGL_FRAME_TYPE_ARRAY: u32 = 0;
const CU_EGL_FRAME_TYPE_PITCH: u32 = 1;
/// `CUeglFrame::cu_format` of an 8-bit-per-channel plane (`CU_AD_FORMAT_UNSIGNED_INT8`).
const CU_AD_FORMAT_U8: u32 = 1;

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
    cuDevicePrimaryCtxRetain: unsafe extern "C" fn(
        pctx: *mut CUcontext,
        dev: CUdevice,
    ) -> CUresult,
    cuCtxPushCurrent_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    cuCtxPopCurrent_v2: unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult,
    cuDevicePrimaryCtxRelease_v2: unsafe extern "C" fn(dev: CUdevice) -> CUresult,
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
    cuMemcpy2DAsync_v2: unsafe extern "C" fn(pCopy: *const CUDA_MEMCPY2D, hStream: CUstream) -> CUresult,
    cuStreamSynchronize: unsafe extern "C" fn(hStream: CUstream) -> CUresult,
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
    CapsParam,
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
            NvStruct::CapsParam => (1, false),
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
            if let (Some(a), Some(b)) = (it.next(), it.next())
                && let (Ok(cm), Ok(cn)) = (a.parse::<u32>(), b.parse::<u32>()) {
                    let capv = (cm << 4) | (cn & 0xF);
                    if capv != 0 && (drv_max == 0 || capv < drv_max) {
                        drv_max = capv;
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
/// once: the `EGLImageKHR`, the CUDA graphics resource it registers as, the mapped `CUeglFrame`,
/// and how that frame reaches NVENC. Torn down on drop / reconfigure, or evicted when the fd it is
/// keyed by no longer names the same buffer (see `DmaBufIdentity`).
struct CachedDmaBuf {
    identity: DmaBufIdentity,
    egl_image: EGLImageKHR,
    cuda_resource: CUgraphicsResource,
    egl_frame: CUeglFrame,
    input: DmaBufInput,
}

/// How a cached dmabuf import feeds the encoder.
///
/// `Direct` is the zero-copy case: the mapped frame's first plane is itself registered and mapped
/// as an NVENC input — as a pitch-linear device pointer or as a CUDA array, per `DirectPlane` — so
/// encoding reads the capture buffer in place. `Copy` covers the rest: a plane `direct_plane`
/// rules out, one the driver declined to register, or the direct path switched off; the plane is
/// then copied into the session's packed input surface each frame.
#[derive(Clone, Copy)]
enum DmaBufInput {
    Direct {
        registered: NV_ENC_REGISTERED_PTR,
        mapped: NV_ENC_INPUT_PTR,
        format: NV_ENC_BUFFER_FORMAT,
    },
    Copy,
}

/// The NVENC packed input format whose byte order is a dmabuf's: the XR24 / AR24 family is
/// B,G,R,A in memory (NVENC's word-ordered `ARGB`), the XB24 / AB24 family R,G,B,A (`ABGR`).
/// `None` for any other fourcc — nothing NVENC reads as packed 8-bit RGB.
fn fourcc_nvenc_format(code: Fourcc) -> Option<NV_ENC_BUFFER_FORMAT> {
    match code {
        Fourcc::Argb8888 | Fourcc::Xrgb8888 => Some(NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB),
        Fourcc::Abgr8888 | Fourcc::Xbgr8888 => Some(NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ABGR),
        _ => None,
    }
}

/// How the first plane of a mapped `CUeglFrame` registers with NVENC in place: the
/// `NV_ENC_REGISTER_RESOURCE` resource type and the `pitch` word that type expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectPlane {
    /// Pitch-linear device memory, registered as a CUDA device pointer at this row pitch.
    Pitch(u32),
    /// A two-dimensional CUDA array of four 8-bit channels, registered as a CUDA array; the
    /// value is the array's row width in bytes (`Width × NumChannels`), which is what NVENC
    /// takes as the pitch of an array resource.
    Array(u32),
}

/// Whether NVENC can read a mapped `CUeglFrame` in place, and how.
///
/// Either frame kind has to be usable as the session's `width × height` input: a first plane
/// present and non-null, and a geometry of at least the session's. A pitch-linear plane also needs
/// a row pitch covering `width * 4` bytes at the 4-byte alignment `NV_ENC_REGISTER_RESOURCE`
/// requires; a CUDA-array plane has to be four 8-bit channels, the layout NVENC's packed formats
/// describe. `None` sends the frame down the per-frame copy into the session's own input surface.
fn direct_plane(frame: &CUeglFrame, width: u32, height: u32) -> Option<DirectPlane> {
    if frame.plane_count < 1 || width == 0 || height == 0 {
        return None;
    }
    if frame.width < width || frame.height < height {
        return None;
    }
    match frame.frame_type {
        CU_EGL_FRAME_TYPE_PITCH => {
            let plane = unsafe { frame.frame.p_pitch[0] };
            let pitch_ok = frame.pitch >= width.saturating_mul(4) && frame.pitch % 4 == 0;
            (!plane.is_null() && pitch_ok).then_some(DirectPlane::Pitch(frame.pitch))
        }
        CU_EGL_FRAME_TYPE_ARRAY => {
            let array = unsafe { frame.frame.p_array[0] };
            let packed_8bit = frame.cu_format == CU_AD_FORMAT_U8 && frame.num_channels == 4;
            (!array.is_null() && packed_8bit).then_some(DirectPlane::Array(frame.width * 4))
        }
        _ => None,
    }
}

/// The stable identity of a dmabuf, so a cache keyed by the raw fd number cannot hand back a
/// stale EGLImage after that fd was closed and recycled onto a different buffer.
///
/// The fd integer alone is not an identity: the host's slot renegotiation frees the backing buffer
/// objects and the kernel reissues the same small fd numbers for the new ones. Since Linux 5.3 each
/// dma-buf carries its own inode, so `(st_dev, st_ino)` distinguishes two buffers that reuse one fd
/// number, and `size` reports the true allocation; the DRM format modifier and the geometry round
/// out the identity for older kernels where the inode is shared. A cache hit requires every field to
/// match, so a recycled fd whose buffer differs in any of them is re-imported instead of reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DmaBufIdentity {
    dev: u64,
    ino: u64,
    size: i64,
    modifier: u64,
    width: u32,
    height: u32,
}

impl DmaBufIdentity {
    /// Read the identity of the buffer behind `fd`: `fstat` supplies the inode and allocation
    /// size, and the caller supplies the modifier and geometry from the dmabuf descriptor. A failed
    /// `fstat` leaves the inode/size zero, which still combines with the modifier and geometry.
    fn probe(fd: i32, modifier: u64, width: u32, height: u32) -> Self {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        let (dev, ino, size) = if unsafe { libc::fstat(fd, &mut st) } == 0 {
            (st.st_dev as u64, st.st_ino as u64, st.st_size as i64)
        } else {
            (0, 0, 0)
        };
        Self { dev, ino, size, modifier, width, height }
    }
}

/// The chroma format and dimensional feasibility a session settles on given the driver's
/// reported capabilities, so init degrades cleanly instead of failing opaquely.
///
/// - `fullcolor` is the chroma actually used: 4:4:4 only when it was requested and the GPU carries
///   it, otherwise 4:2:0.
/// - `downgraded_color` records that a 4:4:4 request was met with 4:2:0, so the caller says so once.
/// - `too_large` carries the driver's `(max_w, max_h)` when the requested geometry exceeds it; the
///   caller then declines NVENC and falls back to software rather than failing to initialize.
///
/// A capability that could not be queried is `None` and does not gate: 4:4:4 stays as requested and
/// the dimension test is skipped, so an unavailable answer never forces a false downgrade or refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CapsDecision {
    fullcolor: bool,
    downgraded_color: bool,
    too_large: Option<(i32, i32)>,
}

/// Resolve the requested chroma and geometry against the driver caps (`None` = unknown, ungated).
fn decide_caps(
    req_fullcolor: bool,
    req_w: i32,
    req_h: i32,
    cap_yuv444: Option<i32>,
    cap_width_max: Option<i32>,
    cap_height_max: Option<i32>,
) -> CapsDecision {
    let downgraded_color = req_fullcolor && cap_yuv444 == Some(0);
    let fullcolor = req_fullcolor && !downgraded_color;
    let exceeds = |req: i32, cap: Option<i32>| cap.is_some_and(|m| m > 0 && req > m);
    let too_large = if exceeds(req_w, cap_width_max) || exceeds(req_h, cap_height_max) {
        Some((cap_width_max.unwrap_or(0), cap_height_max.unwrap_or(0)))
    } else {
        None
    };
    CapsDecision { fullcolor, downgraded_color, too_large }
}

/// The in-place resize headroom for one axis: the requested size lifted to `floor` (the 5.2
/// ceiling the level is pinned at) but never past the driver's reported maximum, so initializing
/// with headroom cannot itself exceed what the GPU supports.
fn nvenc_headroom(size: u32, floor: u32, cap: Option<i32>) -> u32 {
    let want = size.max(floor);
    match cap {
        Some(m) if m > 0 => want.min(m as u32),
        _ => want,
    }
}

/// Query one NVENC capability of `codec` on an open session, returning the driver's integer
/// answer or `None` when the entry point is absent or the query fails — `decide_caps` reads
/// `None` as "do not gate", so a query failure never becomes a false refusal.
unsafe fn query_cap(
    funcs: &NV_ENCODE_API_FUNCTION_LIST,
    session: *mut c_void,
    codec: GUID,
    cap: NV_ENC_CAPS,
) -> Option<i32> {
    let get = funcs.nvEncGetEncodeCaps?;
    let mut param = NV_ENC_CAPS_PARAM {
        version: sv(NvStruct::CapsParam),
        capsToQuery: cap,
        reserved: [0u32; 62],
    };
    let mut val: i32 = 0;
    if get(session, codec, &mut param, &mut val) == NVENCSTATUS::NV_ENC_SUCCESS {
        Some(val)
    } else {
        None
    }
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

/// Whether two GUIDs are the same.
fn guid_eq(a: &GUID, b: &GUID) -> bool {
    a.Data1 == b.Data1 && a.Data2 == b.Data2 && a.Data3 == b.Data3 && a.Data4 == b.Data4
}

/// The NVENC codec GUID of a codec the encoder serves, or `None` for one NVENC has no engine
/// for.
fn codec_guid(codec: Codec) -> Option<GUID> {
    match codec {
        Codec::H264 => Some(NV_ENC_CODEC_H264_GUID),
        Codec::H265 => Some(NV_ENC_CODEC_HEVC_GUID),
        Codec::Av1 => Some(NV_ENC_CODEC_AV1_GUID),
        Codec::Jpeg | Codec::Vp8 | Codec::Vp9 => None,
    }
}

/// The profile of a session: the 4:2:0 profile of the codec, or its 4:4:4 one where the codec
/// has it.
fn profile_guid(codec: Codec, fullcolor: bool) -> GUID {
    match (codec, fullcolor) {
        (Codec::H264, true) => NV_ENC_H264_PROFILE_HIGH_444_GUID,
        (Codec::H264, false) => NV_ENC_H264_PROFILE_HIGH_GUID,
        (Codec::H265, true) => NV_ENC_HEVC_PROFILE_FREXT_GUID,
        (Codec::H265, false) => NV_ENC_HEVC_PROFILE_MAIN_GUID,
        _ => NV_ENC_AV1_PROFILE_MAIN_GUID,
    }
}

/// A live NVENC encoder session with its CUDA context and interop resources.
///
/// One instance owns a CUDA context bound to a specific GPU plus an NVENC session and everything
/// the two input paths need:
///
/// - **Packed path**: a pitched device buffer (`input_device_ptr` / `input_pitch`) registered and
///   mapped as the NVENC input (`registered_input_resource` / `mapped_input_buffer`) in the byte
///   order `input_format` names (re-registered in place when a source of the other order arrives),
///   fed either by a host→device upload or by the copy arm of the dmabuf path.
/// - **Zero-copy dmabuf path**: `dmabuf_cache` memoizes each fd's EGLImage → CUDA import, keyed by
///   fd but validated against the buffer's `DmaBufIdentity` so a recycled fd re-imports; an import
///   whose plane NVENC can take as it is — pitch-linear memory or a packed 8-bit CUDA array — is
///   registered in place (`DmaBufInput::Direct`) unless `direct_dmabuf` was switched off, anything
///   else is copied into the packed input each frame.
///
/// `bitstream_buffers` is a small ring (`current_buffer_idx` cycles it) of output buffers.
/// `pinned_hosts` maps each page-locked host upload source's base pointer to its registered length,
/// with a `0` length recording a failed registration so that address is never re-pinned.
/// `codec` and `fullcolor` name the session's codec and negotiated chroma; `current_qp` tracks the
/// live ConstQP so a paint-over reconfigure is skipped when unchanged. `encode_config` and
/// `init_params` are retained so in-place reconfigure can resubmit them.
/// `omit_stripe_headers` drops the 10-byte wire
/// header, and `node_index` is the effective CUDA device this session is bound to — a reuse across
/// captures that now targets a different device must rebuild rather than reconfigure.
pub struct NvencEncoder {
    encoder_session: *mut c_void,
    cuda_context: CUcontext,
    cuda_device: CUdevice,
    egl_display: EGLDisplay,
    codec: Codec,
    fullcolor: bool,
    width: u32,
    height: u32,
    current_qp: u32,
    encode_config: NV_ENC_CONFIG,
    init_params: NV_ENC_INITIALIZE_PARAMS,
    input_device_ptr: CUdeviceptr,
    input_pitch: usize,
    input_format: NV_ENC_BUFFER_FORMAT,
    registered_input_resource: NV_ENC_REGISTERED_PTR,
    mapped_input_buffer: NV_ENC_INPUT_PTR,
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
    /// Resolved once at init from `PIXELFLUX_NVENC_PIN`: page-lock the host upload sources so the
    /// copy is a direct pinned DMA rather than a pageable copy staged through a bounce buffer.
    pin_uploads: bool,
    /// Resolved once at init from `PIXELFLUX_NVENC_DIRECT`: register pitch-linear dmabuf imports
    /// with NVENC in place instead of copying them into the packed input each frame.
    direct_dmabuf: bool,
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
/// faults on what still points into it: unmap the packed input before unregistering it, free its
/// device buffer, destroy the bitstream buffers, and release every cached dmabuf
/// import (`release_dmabuf_import`: its NVENC mapping and registration, then the CUDA resource and
/// the EGLImage) — all session-owned — before the encoder session itself, and destroy that
/// session before releasing the device's primary CUDA context it was opened against (the retain is
/// refcounted, so the context lives until the last session on that device releases it). The
/// page-locked host sources are unpinned in the same pass, each only when its recorded length is
/// non-zero (a `0` marks a registration that failed and so was never pinned).
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

            for &bs in &self.bitstream_buffers {
                (self.nvenc_funcs.nvEncDestroyBitstreamBuffer.unwrap())(
                    self.encoder_session,
                    bs,
                );
            }

            let imports: Vec<CachedDmaBuf> = self.dmabuf_cache.drain().map(|(_, c)| c).collect();
            for cache in imports {
                self.release_dmabuf_import(cache);
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
            (self.cuda.cuDevicePrimaryCtxRelease_v2)(self.cuda_device);
        }
    }
}

/// The level an NVENC session advertises for this geometry, floored at the level that spans
/// the in-place resize headroom (`NV_ENC_LEVEL` shares each codec's own numbering: level_idc
/// for H.264, general_level_idc for HEVC, seq_level_idx for AV1).
///
/// `reconfigure_resolution` resizes a live session in place, so the level has to cover every
/// geometry that session can still reach — a level bump mid-GOP forces some hardware decoders to
/// re-initialize. The floors (H.264 5.2, HEVC 5.2, AV1 5.0) span everything up to 4096×2304 at
/// 60 fps, so the whole range resolves to one level; only beyond it does the shared ladder step
/// up.
fn nvenc_level(codec: Codec, width: u32, height: u32, fps: u32) -> u32 {
    match codec {
        Codec::H265 => h265_level(width, height, fps).max(156),
        Codec::Av1 => av1_level(width, height, fps).max(12),
        _ => h264_level(width, height, fps).max(52),
    }
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
                cuDevicePrimaryCtxRetain: load!(lib, b"cuDevicePrimaryCtxRetain\0"),
                cuCtxPushCurrent_v2: load!(lib, b"cuCtxPushCurrent_v2\0"),
                cuCtxPopCurrent_v2: load!(lib, b"cuCtxPopCurrent_v2\0"),
                cuDevicePrimaryCtxRelease_v2: load!(lib, b"cuDevicePrimaryCtxRelease_v2\0"),
                cuMemAlloc_v2: load!(lib, b"cuMemAlloc_v2\0"),
                cuMemAllocPitch_v2: load!(lib, b"cuMemAllocPitch_v2\0"),
                cuMemFree_v2: load!(lib, b"cuMemFree_v2\0"),
                cuMemcpyHtoD_v2: load!(lib, b"cuMemcpyHtoD_v2\0"),
                cuMemcpyDtoH_v2: load!(lib, b"cuMemcpyDtoH_v2\0"),
                cuMemcpy2D_v2: load!(lib, b"cuMemcpy2D_v2\0"),
                cuMemcpy2DAsync_v2: load!(lib, b"cuMemcpy2DAsync_v2\0"),
                cuStreamSynchronize: load!(lib, b"cuStreamSynchronize\0"),
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
        if let Ok(target) = std::fs::read_link(&path)
            && let Some(name) = target.file_name()
            && let Some(name_str) = name.to_str() {
                    return Some(name_str.to_string());
                }
        None
    }

    /// Build a live NVENC session for the settings' codec: bind CUDA to the target GPU, open and
    /// configure the encoder, and allocate its input and output buffers.
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
    ///    retain the device's primary CUDA context — shared and refcounted across every session on
    ///    that device rather than a fresh 100-300 MiB context each — pushing it current.
    /// 3. **Allocate input**: a pitched ARGB device buffer (`cuMemAllocPitch`, 16-byte element
    ///    alignment) that hardware CSC turns into YUV.
    /// 4. **Open the session and query caps**: create the function-list instance, open the session
    ///    with the negotiated `apiVersion`, refuse a codec the device lists no engine for, and
    ///    query `nvEncGetEncodeCaps` so init degrades rather than fails — a 4:4:4 request on a GPU
    ///    or codec without it drops to 4:2:0, and a capture beyond the encoder's max dimensions
    ///    returns `Err` so the caller falls back to software. Then pull a preset config (P4,
    ///    ultra-low-latency); a failed preset lookup logs the driver's error string and proceeds
    ///    with the zeroed default rather than aborting.
    /// 5. **Configure the stream** (mutating the returned preset config, whose `version` word is
    ///    re-stamped while its embedded `rcParams` keeps the version the preset fill set): the
    ///    codec's 4:2:0 or 4:4:4 profile; CBR (two-pass quarter-resolution for tighter per-frame
    ///    rate adherence, VBV sizing, optional min/max QP clamps in the codec's quantizer domain)
    ///    or ConstQP; infinite GOP (`gopLength` / `idrPeriod` = `0xFFFFFFFF`); `zeroReorderDelay`
    ///    and, for H.264, a bitstream-restriction VUI (`max_num_reorder_frames=0`) so no-reorder
    ///    decoders don't buffer; an explicit level from `nvenc_level` pinned from frame 1 so the
    ///    level never bumps mid-stream; BT.709 primaries and transfer for the sRGB source, with an
    ///    SMPTE170M matrix and limited range to match the hardware ARGB CSC; repeated parameter
    ///    sets on every key frame; H.264 CABAC; no AUD; one AV1 tile; strict GOP target; and
    ///    lookahead disabled for real-time latency.
    /// 6. **Initialize with resize headroom**: `maxEncodeWidth` / `maxEncodeHeight` are raised to at
    ///    least 4096×2304 (the 5.2 ceiling) so `reconfigure_resolution` can grow in place, but never
    ///    past the driver's reported maximum; this costs ~290 MiB of device memory, so a failed init
    ///    retries at the exact size (in-place resize then falls back to a rebuild).
    /// 7. **Register, map, and buffer**: register and map the packed input surface (as `ARGB`;
    ///    `set_input_format` re-registers it for an RGBA source), and create a
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
        let codec = settings.codec;
        let codec_guid = codec_guid(codec)
            .ok_or_else(|| format!("NVENC has no {} encoder", codec.display()))?;
        println!("[NVENC] Initializing {}...", codec.display());

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

            // One primary context per device, shared and refcounted across every session on that
            // device, rather than a fresh 100-300 MiB context each: a second display or a rebuild
            // retains the same context instead of allocating another. Retain does not make it
            // current, so it is pushed here to run the allocations below and left current for the
            // encode paths — matching the current-context state the removed cuCtxCreate produced.
            let mut cu_context: CUcontext = ptr::null_mut();
            let res = (cuda.cuDevicePrimaryCtxRetain)(&mut cu_context, cu_device);
            if res != CUresult::CUDA_SUCCESS {
                return Err("Failed to retain the device's primary CUDA context".into());
            }
            if (cuda.cuCtxPushCurrent_v2)(cu_context) != CUresult::CUDA_SUCCESS {
                (cuda.cuDevicePrimaryCtxRelease_v2)(cu_device);
                return Err("Failed to make the primary CUDA context current".into());
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
                (cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                (cuda.cuDevicePrimaryCtxRelease_v2)(cu_device);
                return Err("Failed to allocate ARGB input buffer on GPU".into());
            }

            let mut function_list = NV_ENCODE_API_FUNCTION_LIST {
                version: sv(NvStruct::FunctionList),
                ..Default::default()
            };
            if (nvenc_lib.create_instance)(&mut function_list) != NVENCSTATUS::NV_ENC_SUCCESS {
                (cuda.cuMemFree_v2)(input_device_ptr);
                (cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                (cuda.cuDevicePrimaryCtxRelease_v2)(cu_device);
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
                (cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                (cuda.cuDevicePrimaryCtxRelease_v2)(cu_device);
                return Err("Failed to open NVENC session".into());
            }

            // The device's engine list decides whether the codec exists here at all (AV1
            // arrived with Ada), and a refusal has to name that rather than fail an init.
            if !Self::device_encodes(&function_list, encoder_session, &codec_guid) {
                (function_list.nvEncDestroyEncoder.unwrap())(encoder_session);
                (cuda.cuMemFree_v2)(input_device_ptr);
                (cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                (cuda.cuDevicePrimaryCtxRelease_v2)(cu_device);
                return Err(format!("this GPU's NVENC has no {} engine", codec.display()));
            }

            // Query caps so init degrades instead of failing opaquely: a 4:4:4 request on a GPU
            // or codec without it drops to 4:2:0, and a capture beyond the encoder's max
            // dimensions declines NVENC so the caller falls back to software.
            let caps_444 = if codec.fullcolor() {
                query_cap(
                    &function_list,
                    encoder_session,
                    codec_guid,
                    NV_ENC_CAPS::NV_ENC_CAPS_SUPPORT_YUV444_ENCODE,
                )
            } else {
                Some(0)
            };
            let caps_wmax =
                query_cap(&function_list, encoder_session, codec_guid, NV_ENC_CAPS::NV_ENC_CAPS_WIDTH_MAX);
            let caps_hmax =
                query_cap(&function_list, encoder_session, codec_guid, NV_ENC_CAPS::NV_ENC_CAPS_HEIGHT_MAX);
            let caps = decide_caps(
                settings.video_fullcolor,
                width as i32,
                height as i32,
                caps_444,
                caps_wmax,
                caps_hmax,
            );
            if let Some((mw, mh)) = caps.too_large {
                (function_list.nvEncDestroyEncoder.unwrap())(encoder_session);
                (cuda.cuMemFree_v2)(input_device_ptr);
                (cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                (cuda.cuDevicePrimaryCtxRelease_v2)(cu_device);
                return Err(format!(
                    "NVENC maximum encode size {mw}x{mh} exceeded by {width}x{height}; using software"
                ));
            }
            if caps.downgraded_color {
                eprintln!(
                    "[NVENC] {} 4:4:4 (YUV444) encoding unsupported on this GPU; encoding 4:2:0.",
                    codec.display()
                );
            }

            let is_444 = caps.fullcolor;

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
                codec_guid,
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
            config.profileGUID = profile_guid(codec, is_444);
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
                let lo = codec.quantizer_bound(settings.video_min_qp);
                if lo > 0 {
                    config.rcParams.set_enableMinQP(1);
                    config.rcParams.minQP.qpInterP = lo;
                    config.rcParams.minQP.qpInterB = lo;
                    config.rcParams.minQP.qpIntra = lo;
                }
                let hi = codec.quantizer_bound(settings.video_max_qp);
                if hi > 0 {
                    config.rcParams.set_enableMaxQP(1);
                    config.rcParams.maxQP.qpInterP = hi;
                    config.rcParams.maxQP.qpInterB = hi;
                    config.rcParams.maxQP.qpIntra = hi;
                }
            } else {
                let q = codec.quantizer(settings.video_crf);
                config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CONSTQP;
                config.rcParams.constQP.qpInterP = q;
                config.rcParams.constQP.qpInterB = q;
                config.rcParams.constQP.qpIntra = q;
            }
            config.frameIntervalP = 1;
            config.gopLength = 0xFFFFFFFF;
            config.rcParams.set_zeroReorderDelay(1);
            config.rcParams.set_strictGOPTarget(1);
            config.rcParams.set_enableLookahead(0);
            config.rcParams.lookaheadDepth = 0;
            Self::configure_codec(&mut config, codec, is_444, width, height, settings.target_fps as u32);

            let mut init_params = NV_ENC_INITIALIZE_PARAMS {
                version: sv(NvStruct::InitializeParams),
                encodeGUID: codec_guid,
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
                maxEncodeWidth: nvenc_headroom(width, 4096, caps_wmax),
                maxEncodeHeight: nvenc_headroom(height, 2304, caps_hmax),
                ..Default::default()
            };

            let init_fn = function_list.nvEncInitializeEncoder.unwrap();
            if init_fn(encoder_session, &mut init_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                init_params.maxEncodeWidth = width;
                init_params.maxEncodeHeight = height;
                if init_fn(encoder_session, &mut init_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                    (function_list.nvEncDestroyEncoder.unwrap())(encoder_session);
                    (cuda.cuMemFree_v2)(input_device_ptr);
                    (cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    (cuda.cuDevicePrimaryCtxRelease_v2)(cu_device);
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
                (cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                (cuda.cuDevicePrimaryCtxRelease_v2)(cu_device);
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
                (cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                (cuda.cuDevicePrimaryCtxRelease_v2)(cu_device);
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
                    (cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    (cuda.cuDevicePrimaryCtxRelease_v2)(cu_device);
                    return Err("Failed to create bitstream buffer".into());
                }
                bitstream_buffers.push(bitstream_params.bitstreamBuffer);
            }

            println!("[NVENC] {} initialized (4:4:4 mode: {}).", codec.display(), is_444);

            Ok(Self {
                encoder_session,
                cuda_context: cu_context,
                cuda_device: cu_device,
                egl_display: egl_display as EGLDisplay,
                codec,
                fullcolor: is_444,
                width,
                height,
                current_qp: codec.quantizer(settings.video_crf),
                encode_config: config,
                init_params,
                input_device_ptr,
                input_pitch,
                input_format: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
                registered_input_resource: reg_res.registeredResource,
                mapped_input_buffer: map_params.mappedResource,
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
                pin_uploads: std::env::var("PIXELFLUX_NVENC_PIN").as_deref() != Ok("0"),
                direct_dmabuf: std::env::var("PIXELFLUX_NVENC_DIRECT").as_deref() != Ok("0"),
            })
        }
    }

    /// Whether the open session's device lists an encode engine for `codec`.
    unsafe fn device_encodes(funcs: &NV_ENCODE_API_FUNCTION_LIST, session: *mut c_void, codec: &GUID) -> bool {
        let (Some(count_fn), Some(list_fn)) = (funcs.nvEncGetEncodeGUIDCount, funcs.nvEncGetEncodeGUIDs) else {
            return true;
        };
        let mut count = 0u32;
        if count_fn(session, &mut count) != NVENCSTATUS::NV_ENC_SUCCESS || count == 0 {
            return false;
        }
        let mut guids = vec![GUID::default(); count as usize];
        let mut listed = 0u32;
        if list_fn(session, guids.as_mut_ptr(), count, &mut listed) != NVENCSTATUS::NV_ENC_SUCCESS {
            return false;
        }
        guids.iter().take(listed as usize).any(|g| guid_eq(g, codec))
    }

    /// Program the codec-specific arm of `config`: the level for the geometry, an infinite IDR
    /// period, the chroma format, 8-bit input and output, parameter sets repeated on every key
    /// frame, and the colour description of the hardware CSC.
    ///
    /// Primaries and transfer describe the source, which is sRGB desktop pixels — sRGB shares
    /// BT.709's primaries and transfer function. Only the matrix follows the encoder: NVENC's
    /// ARGB hardware CSC is fixed at BT.601, so a client that inverts BT.709 shifts saturated
    /// colour badly; that same CSC emits limited range in every chroma format, and both capture
    /// paths go through it, so every session declares limited. H.264 additionally restricts
    /// reordering in its VUI so no-reorder decoders don't buffer, and codes CABAC; AV1 keeps
    /// one tile, since tiles cost bitrate and buy no quality.
    fn configure_codec(config: &mut NV_ENC_CONFIG, codec: Codec, fullcolor: bool, width: u32, height: u32, fps: u32) {
        let level = nvenc_level(codec, width, height, fps);
        let primaries = NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709;
        let transfer = NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
        let matrix = NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_SMPTE170M;
        let vui = |vui: &mut NV_ENC_CONFIG_H264_VUI_PARAMETERS| {
            vui.videoSignalTypePresentFlag = 1;
            vui.videoFormat = NV_ENC_VUI_VIDEO_FORMAT::NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
            vui.colourDescriptionPresentFlag = 1;
            vui.colourPrimaries = primaries;
            vui.transferCharacteristics = transfer;
            vui.colourMatrix = matrix;
            vui.videoFullRangeFlag = 0;
        };
        unsafe {
            match codec {
                Codec::H265 => {
                    let c = &mut config.encodeCodecConfig.hevcConfig;
                    c.level = level;
                    c.tier = 0;
                    c.idrPeriod = 0xFFFFFFFF;
                    c.set_chromaFormatIDC(if fullcolor { 3 } else { 1 });
                    c.inputBitDepth = NV_ENC_BIT_DEPTH::NV_ENC_BIT_DEPTH_8;
                    c.outputBitDepth = NV_ENC_BIT_DEPTH::NV_ENC_BIT_DEPTH_8;
                    c.set_repeatSPSPPS(1);
                    c.set_outputAUD(0);
                    vui(&mut c.hevcVUIParameters);
                }
                Codec::Av1 => {
                    let c = &mut config.encodeCodecConfig.av1Config;
                    c.level = level;
                    c.tier = 0;
                    c.idrPeriod = 0xFFFFFFFF;
                    c.set_chromaFormatIDC(1);
                    c.inputBitDepth = NV_ENC_BIT_DEPTH::NV_ENC_BIT_DEPTH_8;
                    c.outputBitDepth = NV_ENC_BIT_DEPTH::NV_ENC_BIT_DEPTH_8;
                    c.set_repeatSeqHdr(1);
                    c.set_outputAnnexBFormat(0);
                    c.set_enableBitstreamPadding(0);
                    c.numTileColumns = 1;
                    c.numTileRows = 1;
                    c.colorPrimaries = primaries;
                    c.transferCharacteristics = transfer;
                    c.matrixCoefficients = matrix;
                    c.colorRange = 0;
                }
                _ => {
                    let c = &mut config.encodeCodecConfig.h264Config;
                    c.level = level;
                    c.idrPeriod = 0xFFFFFFFF;
                    c.chromaFormatIDC = if fullcolor { 3 } else { 1 };
                    c.set_repeatSPSPPS(1);
                    c.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE::NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;
                    c.set_outputAUD(0);
                    c.h264VUIParameters.bitstreamRestrictionFlag = 1;
                    vui(&mut c.h264VUIParameters);
                }
            }
        }
    }

    /// Write the level for a new geometry or frame rate into the live config's codec arm.
    fn set_level(&mut self, width: u32, height: u32, fps: u32) {
        let level = nvenc_level(self.codec, width, height, fps);
        match self.codec {
            Codec::H265 => self.encode_config.encodeCodecConfig.hevcConfig.level = level,
            Codec::Av1 => self.encode_config.encodeCodecConfig.av1Config.level = level,
            _ => self.encode_config.encodeCodecConfig.h264Config.level = level,
        }
    }

    /// The codec the session emits.
    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// Whether the session negotiated 4:4:4 chroma.
    pub fn is_fullcolor(&self) -> bool {
        self.fullcolor
    }

    /// Resize the live session to `settings` in place, folding in the current rate / QP /
    /// fps, without tearing it down.
    ///
    /// The NVENC session, CUDA context and bitstream buffers survive, so a resize costs a few
    /// milliseconds instead of a full rebuild. Flow:
    ///
    /// 1. **Reject the unchangeable**: a different encode device or codec, a chroma-format flip
    ///    (4:4:4), an RC-mode flip, or dimensions of zero or beyond the init-time `maxEncode`
    ///    headroom all return `Err` so the caller rebuilds.
    /// 2. **Release geometry-dependent state** under the pushed CUDA context: unmap / unregister /
    ///    free the packed input surface, every cached dmabuf import (with the NVENC registration a
    ///    direct import holds), and every pinned host. The dmabuf imports are re-created lazily by
    ///    the encode path; pinned hosts are dropped because the source shm segments are recreated
    ///    on resize and may reuse the same base addresses.
    /// 3. **Reconfigure the session**: update the level for the new size, the CBR bitrate + VBV or
    ///    the ConstQP, and the new dimensions / DAR / frame rate, then `NvEncReconfigureEncoder` with
    ///    `resetEncoder` and `forceIDR` so the stream restarts cleanly at the new size. Driver
    ///    rejection returns `Err`.
    /// 4. **Reallocate the packed input** at the new size and register + map it as init does, in
    ///    the byte order the session was last fed.
    ///
    /// On success the next encoded frame is a reset-RC IDR.
    pub fn reconfigure_resolution(&mut self, settings: &RustCaptureSettings) -> Result<(), String> {
        let new_w = settings.width as u32;
        let new_h = settings.height as u32;
        let is_cbr = self.encode_config.rcParams.rateControlMode
            == NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR;
        if settings.encode_node_index.max(0) != self.node_index {
            return Err("encode device changed".into());
        }
        if settings.codec != self.codec {
            return Err("codec changed".into());
        }
        if (settings.video_fullcolor && self.codec.fullcolor()) != self.fullcolor {
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
            let imports: Vec<CachedDmaBuf> = self.dmabuf_cache.drain().map(|(_, c)| c).collect();
            for cache in imports {
                self.release_dmabuf_import(cache);
            }
            for (base, len) in self.pinned_hosts.drain() {
                if len > 0 {
                    (self.cuda.cuMemHostUnregister)(base as *mut c_void);
                }
            }

            self.set_level(new_w, new_h, settings.target_fps as u32);
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
                let qp = self.codec.quantizer(settings.video_crf);
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
                bufferFormat: self.input_format,
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

    /// Page-lock one host upload source's base address once, under the already-current CUDA
    /// context, so the copy is a direct pinned DMA instead of a pageable copy staged through a driver
    /// bounce buffer. A `0`-length entry records a failed registration so the address is never
    /// re-probed; the persistent, bounded shm / reused planar sources make this a one-time cost.
    unsafe fn pin_host_source(&mut self, base: usize, len: usize) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.pinned_hosts.entry(base) {
            let st = (self.cuda.cuMemHostRegister_v2)(base as *mut c_void, len, 0);
            e.insert(if st == CUresult::CUDA_SUCCESS { len } else { 0 });
        }
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

    /// Tear down one cached dmabuf import under the already-current CUDA context, inner handle
    /// first: the NVENC mapping and registration a direct import holds, then the CUDA graphics
    /// resource, then the EGLImage it was built from. The encode that last read the import has
    /// completed (`submit_frame` waits for the bitstream), so nothing is still in flight on it.
    unsafe fn release_dmabuf_import(&self, cache: CachedDmaBuf) {
        if let DmaBufInput::Direct { registered, mapped, .. } = cache.input {
            (self.nvenc_funcs.nvEncUnmapInputResource.unwrap())(self.encoder_session, mapped);
            (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(self.encoder_session, registered);
        }
        (self.cuda.cuGraphicsUnregisterResource)(cache.cuda_resource);
        (self.egl.eglDestroyImageKHR)(self.egl_display, cache.egl_image);
    }

    /// Register the packed input surface with NVENC in the byte order `format` names, when it is
    /// not already: the surface memory is unchanged, only its registration (and mapping) is
    /// replaced, so a session fed first from one source order and then the other keeps one surface.
    /// Runs under the already-current CUDA context; a failed re-registration leaves the surface
    /// unregistered and returns `Err`, so the caller's encode fails visibly instead of encoding
    /// swapped channels.
    unsafe fn set_input_format(&mut self, format: NV_ENC_BUFFER_FORMAT) -> Result<(), String> {
        if self.input_format == format && !self.registered_input_resource.is_null() {
            return Ok(());
        }
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
        let mut reg_res = NV_ENC_REGISTER_RESOURCE {
            version: sv(NvStruct::RegisterResource),
            resourceType: NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
            width: self.width,
            height: self.height,
            resourceToRegister: self.input_device_ptr as *mut c_void,
            pitch: self.input_pitch as u32,
            bufferFormat: format,
            bufferUsage: NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE,
            ..Default::default()
        };
        if (self.nvenc_funcs.nvEncRegisterResource.unwrap())(self.encoder_session, &mut reg_res)
            != NVENCSTATUS::NV_ENC_SUCCESS
        {
            return Err(format!("Failed to register input buffer as {format:?}"));
        }
        let mut map_params = NV_ENC_MAP_INPUT_RESOURCE {
            version: sv(NvStruct::MapInputResource),
            registeredResource: reg_res.registeredResource,
            ..Default::default()
        };
        if (self.nvenc_funcs.nvEncMapInputResource.unwrap())(self.encoder_session, &mut map_params)
            != NVENCSTATUS::NV_ENC_SUCCESS
        {
            (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                self.encoder_session,
                reg_res.registeredResource,
            );
            return Err(format!("Failed to map input buffer as {format:?}"));
        }
        self.registered_input_resource = reg_res.registeredResource;
        self.mapped_input_buffer = map_params.mappedResource;
        self.input_format = format;
        Ok(())
    }

    /// Reconfigure the live session's ConstQP when the quantizer the session quality index
    /// `crf` selects differs from the current one, returning whether a reconfigure actually
    /// happened.
    ///
    /// A no-op in CBR mode (bitrate-controlled, so QP-based paint-over does not apply) and when the
    /// QP is unchanged. When it does apply, the three `constQP` fields are updated and the session is
    /// reconfigured **without** a forced IDR: a lower-QP P-frame refines the static image against the
    /// existing reference chain (paint-over) with no intra-frame bitrate spike, so the GOP continues
    /// seamlessly across the reconfigure.
    unsafe fn reconfigure_if_needed(&mut self, crf: u32) -> bool {
        if self.encode_config.rcParams.rateControlMode
            == NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR
        {
            return false;
        }
        let target_qp = self.codec.quantizer(crf as i32);
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
                self.set_level(self.init_params.encodeWidth, self.init_params.encodeHeight, fps);
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
    /// 3. **Frame the output**: unless `omit_stripe_headers` is set, prepend the wire header with
    ///    the frame kind derived from the *actual* encoded `pictureType` (IDR = key, I = intra,
    ///    else delta) rather than the `force_idr` request.
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
        let header_sz = if self.omit_stripe_headers { 0 } else { VIDEO_HEADER_LEN };
        let mut output = Vec::with_capacity(header_sz + data_size);

        if !self.omit_stripe_headers {
            let frame_type = match lock_params.pictureType {
                NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR => FRAME_KEY,
                NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_I => FRAME_INTRA,
                _ => FRAME_DELTA,
            };
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

        if data_size > 0 && !data_ptr.is_null() {
            let slice = std::slice::from_raw_parts(data_ptr, data_size);
            output.extend_from_slice(slice);
        }

        (self.nvenc_funcs.nvEncUnlockBitstream.unwrap())(self.encoder_session, output_bitstream);
        Ok(output)
    }

    /// Encode a dmabuf frame zero-copy, by importing it through EGL into CUDA and, where the
    /// driver allows, handing the mapped plane to NVENC as its input.
    ///
    /// Applies any pending ConstQP change, then works under the pushed CUDA context:
    ///
    /// 1. **Import once, cache by fd with an identity check**: the cache is keyed by the dmabuf fd
    ///    but each entry stores the buffer's `DmaBufIdentity`; an entry whose identity no longer
    ///    matches (a recycled fd) is released first. On a miss, build an `EGLImageKHR` from the
    ///    dmabuf's fd / offset / pitch / modifier, register it as a CUDA graphics resource, map it to
    ///    a `CUeglFrame`, and settle how it feeds the encoder: a first plane that `direct_plane`
    ///    accepts (and `direct_dmabuf` on) is registered with NVENC in place — a pitch-linear plane
    ///    as a CUDA device pointer at its own pitch, a four-channel 8-bit CUDA array as a CUDA
    ///    array — in the byte order the dmabuf fourcc names, and mapped once (`DmaBufInput::Direct`);
    ///    any other plane, or a registration the driver refuses, takes `DmaBufInput::Copy`. The
    ///    result is memoized so a recurring capture buffer pays the import cost only once. Each
    ///    failure destroys what it created and pops the context.
    /// 2. **Feed the encoder**: a direct import is submitted as it is — no copy at all. A copy
    ///    import is copied with `cuMemcpy2DAsync` on the default stream — the array plane or the
    ///    pitch-linear plane, per `frame_type` — into the packed input surface, re-registered in the
    ///    dmabuf's byte order when it differs; NVENC processes its input on that same stream, so the
    ///    copy is ordered before the encode without a host wait.
    /// 3. **Submit** via `submit_frame`, then pop the context.
    ///
    /// The dmabuf fd is read out before the context is pushed so an early `?` return cannot leave the
    /// CUDA context stack imbalanced.
    pub fn encode(
        &mut self,
        dmabuf: &Dmabuf,
        frame_number: u64,
        crf: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.reconfigure_if_needed(crf);
            let fd = dmabuf.handles().next().ok_or("No handles")?.as_raw_fd();
            let fmt = dmabuf.format();
            let modifier: u64 = fmt.modifier.into();
            let identity = DmaBufIdentity::probe(fd, modifier, self.width, self.height);
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);

            // A raw fd number is not an identity: the host recycles fd numbers across slot
            // renegotiations, so an entry whose stored identity no longer matches is torn down and
            // re-imported rather than returning a stale EGLImage for a buffer the fd no longer names.
            if self.dmabuf_cache.get(&fd).is_some_and(|c| c.identity != identity)
                && let Some(stale) = self.dmabuf_cache.remove(&fd)
            {
                self.release_dmabuf_import(stale);
            }

            if !self.dmabuf_cache.contains_key(&fd) {
                let stride = dmabuf.strides().next().unwrap_or(0) as i32;
                let offset = dmabuf.offsets().next().unwrap_or(0) as i32;

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

                let input = match (
                    self.direct_dmabuf,
                    direct_plane(&egl_frame, self.width, self.height),
                    fourcc_nvenc_format(fmt.code),
                ) {
                    (true, Some(plane), Some(format)) => {
                        self.register_direct_input(&egl_frame, plane, format)
                    }
                    _ => DmaBufInput::Copy,
                };
                println!(
                    "[NVENC] dmabuf imported as a {} frame ({} planes, {}x{}, pitch {}, {} channels of element format {}): {}.",
                    match egl_frame.frame_type {
                        CU_EGL_FRAME_TYPE_PITCH => "pitch-linear",
                        CU_EGL_FRAME_TYPE_ARRAY => "CUDA-array",
                        _ => "unknown-kind",
                    },
                    egl_frame.plane_count,
                    egl_frame.width,
                    egl_frame.height,
                    egl_frame.pitch,
                    egl_frame.num_channels,
                    egl_frame.cu_format,
                    match input {
                        DmaBufInput::Direct { .. } => "encoding in place",
                        DmaBufInput::Copy => "copying per frame",
                    }
                );

                self.dmabuf_cache.insert(
                    fd,
                    CachedDmaBuf {
                        identity,
                        egl_image,
                        cuda_resource,
                        egl_frame,
                        input,
                    },
                );
            }

            let (egl_frame, input) = {
                let cached = self.dmabuf_cache.get(&fd).unwrap();
                (cached.egl_frame, cached.input)
            };
            let (mapped, format) = match input {
                DmaBufInput::Direct { mapped, format, .. } => (mapped, format),
                DmaBufInput::Copy => {
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
                    if egl_frame.frame_type == CU_EGL_FRAME_TYPE_ARRAY {
                        copy_params.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
                        copy_params.srcArray = egl_frame.frame.p_array[0];
                    } else {
                        copy_params.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
                        copy_params.srcDevice = egl_frame.frame.p_pitch[0] as CUdeviceptr;
                        copy_params.srcPitch = egl_frame.pitch as usize;
                    }
                    // A fourcc without a packed NVENC equivalent keeps the surface's current
                    // registration; the copy still lands the bytes, as it always has.
                    if let Some(format) = fourcc_nvenc_format(fmt.code)
                        && let Err(e) = self.set_input_format(format)
                    {
                        (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                        return Err(e);
                    }
                    if (self.cuda.cuMemcpy2DAsync_v2)(&copy_params, ptr::null_mut())
                        != CUresult::CUDA_SUCCESS
                    {
                        (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                        return Err("Sanitization copy failed".into());
                    }
                    (self.mapped_input_buffer, self.input_format)
                }
            };

            let result = self.submit_frame(mapped, format, frame_number, force_idr);
            if result.is_err() {
                (self.cuda.cuStreamSynchronize)(ptr::null_mut());
            }
            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
            result
        }
    }

    /// Register the first plane of a mapped dmabuf frame with NVENC in place — as a pitch-linear
    /// CUDA device pointer or as a CUDA array, per `plane` — and map it as an input, under the
    /// already-current CUDA context. Either step failing falls back to `DmaBufInput::Copy` — the
    /// per-frame copy then serves that import for as long as it is cached, so a driver that declines
    /// the direct path costs one failed registration, not a failed frame.
    unsafe fn register_direct_input(
        &mut self,
        frame: &CUeglFrame,
        plane: DirectPlane,
        format: NV_ENC_BUFFER_FORMAT,
    ) -> DmaBufInput {
        let (resource_type, resource, pitch) = match plane {
            DirectPlane::Pitch(pitch) => (
                NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                frame.frame.p_pitch[0],
                pitch,
            ),
            DirectPlane::Array(row_bytes) => (
                NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY,
                frame.frame.p_array[0] as *mut c_void,
                row_bytes,
            ),
        };
        let mut reg_res = NV_ENC_REGISTER_RESOURCE {
            version: sv(NvStruct::RegisterResource),
            resourceType: resource_type,
            width: self.width,
            height: self.height,
            resourceToRegister: resource,
            pitch,
            bufferFormat: format,
            bufferUsage: NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE,
            ..Default::default()
        };
        let st = (self.nvenc_funcs.nvEncRegisterResource.unwrap())(self.encoder_session, &mut reg_res);
        if st != NVENCSTATUS::NV_ENC_SUCCESS {
            eprintln!("[NVENC] dmabuf plane registration ({plane:?}) refused ({st:?}); copying per frame.");
            return DmaBufInput::Copy;
        }
        let mut map_params = NV_ENC_MAP_INPUT_RESOURCE {
            version: sv(NvStruct::MapInputResource),
            registeredResource: reg_res.registeredResource,
            ..Default::default()
        };
        let st = (self.nvenc_funcs.nvEncMapInputResource.unwrap())(self.encoder_session, &mut map_params);
        if st != NVENCSTATUS::NV_ENC_SUCCESS {
            (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                self.encoder_session,
                reg_res.registeredResource,
            );
            eprintln!("[NVENC] dmabuf plane mapping refused ({st:?}); copying per frame.");
            return DmaBufInput::Copy;
        }
        DmaBufInput::Direct {
            registered: reg_res.registeredResource,
            mapped: map_params.mappedResource,
            format,
        }
    }

    /// Encode a host BGRA frame (B,G,R,A in memory, NVENC's word-ordered `ARGB` — the layout an
    /// XShm grab or the pixman framebuffer yields) through `encode_cpu_packed`.
    pub fn encode_cpu_argb(
        &mut self,
        argb: &[u8],
        src_stride: usize,
        frame_number: u64,
        crf: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        self.encode_cpu_packed(argb, src_stride, false, frame_number, crf, force_idr)
    }

    /// Encode a host packed-pixel frame by uploading it straight into the packed input surface,
    /// with no CPU-side colour conversion: NVENC's hardware RGB→YUV conversion is fixed at BT.601
    /// limited range, which is what the session VUI declares, and a CPU prepass to BT.709 would
    /// cost this path its copy-free property.
    ///
    /// `rgba_input` names the byte order — `false` for B,G,R,A (X11 XShm, the pixman framebuffer),
    /// `true` for R,G,B,A (a GLES readback) — and the input surface is registered with NVENC in
    /// that order (`ARGB` / `ABGR`, re-registered in place when it changes), so both arrive at the
    /// hardware CSC untouched. `src_stride` is the source row stride in bytes (`>= width*4`).
    /// Steps, under the pushed CUDA context after any pending QP change:
    ///
    /// 1. **Bounds-check** the source against `stride × (rows-1) + width*4`, erroring rather than
    ///    reading past a short buffer.
    /// 2. **Pin the source once**: unless pinning was disabled at init (`PIXELFLUX_NVENC_PIN=0`, read
    ///    once into `pin_uploads`), page-lock each distinct source base address via `pin_host_source`
    ///    so the upload is a direct DMA from the caller's buffer instead of a pageable copy staged
    ///    through a driver bounce buffer. The persistent, bounded shm / pool sources make this a
    ///    one-time bounded cost.
    /// 3. **Upload and submit**: `cuMemcpy2DAsync` the rows into the input surface on the default
    ///    stream honoring `src_stride`, then `submit_frame`. NVENC processes its input on that same
    ///    stream, so the upload is ordered before the encode without a host wait, and the blocking
    ///    bitstream lock inside `submit_frame` (or the stream sync on its error path) guarantees the
    ///    upload has finished reading `pixels` by the time this returns — the caller may reuse the
    ///    buffer immediately.
    pub fn encode_cpu_packed(
        &mut self,
        pixels: &[u8],
        src_stride: usize,
        rgba_input: bool,
        frame_number: u64,
        crf: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.reconfigure_if_needed(crf);
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);

            let width_bytes = (self.width * 4) as usize;
            let rows = self.height as usize;
            let needed = if rows == 0 { 0 } else { src_stride * (rows - 1) + width_bytes };
            if src_stride < width_bytes || pixels.len() < needed {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err(format!(
                    "packed buffer too small: len={} need>={} (stride={}, {}x{})",
                    pixels.len(), needed, src_stride, self.width, self.height
                ));
            }

            let format = if rgba_input {
                NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ABGR
            } else {
                NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB
            };
            if let Err(e) = self.set_input_format(format) {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err(e);
            }

            if self.pin_uploads {
                self.pin_host_source(pixels.as_ptr() as usize, pixels.len());
            }

            let copy = CUDA_MEMCPY2D {
                srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                srcHost: pixels.as_ptr() as *const c_void,
                srcPitch: src_stride,
                dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstDevice: self.input_device_ptr,
                dstPitch: self.input_pitch,
                WidthInBytes: width_bytes,
                Height: rows,
                ..Default::default()
            };
            if (self.cuda.cuMemcpy2DAsync_v2)(&copy, ptr::null_mut()) != CUresult::CUDA_SUCCESS {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("packed host->device upload failed".into());
            }

            let result = self.submit_frame(
                self.mapped_input_buffer,
                self.input_format,
                frame_number,
                force_idr,
            );
            if result.is_err() {
                (self.cuda.cuStreamSynchronize)(ptr::null_mut());
            }
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
            codec: Codec::H264,
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
        assert_eq!(pkt[1] & 0x0f, FRAME_KEY, "first frame after a resize must be an IDR");
        assert_eq!(wire_dims(&pkt), (1920, 1080));
        stream.extend_from_slice(&pkt[10..]);
        for i in 6..10u64 {
            let pkt = enc
                .encode_cpu_argb(&f1080, 1920 * 4, i, 25, false)
                .expect("encode 1080p");
            assert_eq!(pkt[1] & 0x0f, FRAME_DELTA, "steady frames after the IDR are P frames");
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
        assert_eq!(pkt[1] & 0x0f, FRAME_KEY);
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

    /// On a real GPU: every codec the device lists an engine for comes up, frames it
    /// emits carry the codec's wire id and a kind the bitstream agrees with, decode back
    /// to the painted picture, and a key frame forced mid-stream starts a fresh decoder on
    /// its own; a codec the device lacks (AV1 before Ada) is refused with a message naming
    /// it. HEVC 4:4:4 is taken where the device carries it, AV1 never quietly. Ignored by
    /// default.
    #[test]
    #[ignore]
    fn gpu_codec_sessions_encode_and_decode() {
        use crate::encoders::codec::{av1_is_key, h264_frame_type, h265_frame_type, parse_video_type, FRAME_DELTA, FRAME_KEY};
        use crate::webcam::decode::{AvDecoder, Decoder};
        let (w, h) = (1280usize, 720usize);
        for codec in [Codec::H264, Codec::H265, Codec::Av1] {
            let mut s = settings(w as i32, h as i32, 60.0);
            s.codec = codec;
            let mut enc = match NvencEncoder::new(&s, ptr::null()) {
                Ok(enc) => enc,
                Err(e) => {
                    println!("{codec:?}: {e}");
                    assert!(e.contains("engine"), "a missing codec must be refused as such: {e}");
                    continue;
                }
            };
            assert_eq!(enc.codec(), codec);
            let mut dec = AvDecoder::new(codec).expect("decoder");
            for i in 0..6u64 {
                let src = frame(w, h, 10 + i as u8);
                let pkt = enc.encode_cpu_argb(&src, w * 4, i, 25, i == 0).expect("encode");
                let (wire_codec, kind) = parse_video_type(pkt[1]).expect("video type byte");
                assert_eq!(wire_codec, codec);
                assert_eq!(kind, if i == 0 { FRAME_KEY } else { FRAME_DELTA }, "{codec:?} frame {i}");
                let payload = &pkt[10..];
                let read = match codec {
                    Codec::H264 => h264_frame_type(payload),
                    Codec::H265 => h265_frame_type(payload),
                    _ => if av1_is_key(payload) { FRAME_KEY } else { FRAME_DELTA },
                };
                assert_eq!(read, kind, "{codec:?} frame {i}: the bitstream disagrees with pictureType");
                assert!(dec.decode(payload).expect("decode"), "{codec:?} frame {i} decoded nothing");
                let f = dec.frame().unwrap();
                assert_eq!((f.width, f.height), (w, h));
            }
            let key = enc.encode_cpu_argb(&frame(w, h, 99), w * 4, 6, 25, true).expect("forced key");
            assert_eq!(parse_video_type(key[1]), Some((codec, FRAME_KEY)));
            let mut fresh = AvDecoder::new(codec).expect("decoder");
            assert!(fresh.decode(&key[10..]).expect("decode"), "{codec:?}: a forced key frame must decode alone");

            let mut full = s.clone();
            full.video_fullcolor = true;
            let enc = NvencEncoder::new(&full, ptr::null()).expect("4:4:4 request");
            if codec.fullcolor() {
                println!("{codec:?} 4:4:4: {}", enc.is_fullcolor());
            } else {
                assert!(!enc.is_fullcolor(), "{codec:?} never carries 4:4:4");
            }
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
        assert_eq!(pkt[1] & 0x0f, FRAME_KEY);
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

    /// Thread CPU time, for the per-frame CPU cost of an encode path independent of how long
    /// the thread waited on the GPU.
    fn thread_cpu() -> std::time::Duration {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        std::time::Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
    }

    /// Test helper: run `f` `n` times and report wall and thread-CPU microseconds per call.
    fn per_frame(label: &str, n: usize, mut f: impl FnMut(usize)) -> (f64, f64) {
        let t0 = std::time::Instant::now();
        let c0 = thread_cpu();
        for i in 0..n {
            f(i);
        }
        let wall = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        let cpu = (thread_cpu() - c0).as_secs_f64() * 1e6 / n as f64;
        println!("{label}: {wall:.0} us wall/frame, {cpu:.0} us cpu/frame ({n} frames)");
        (wall, cpu)
    }

    /// Test helper: the raw GBM device and GLES renderer of the render node named by
    /// `PIXELFLUX_TEST_RENDER_NODE` (default `/dev/dri/renderD128`), brought up exactly as the
    /// compositor brings them up.
    fn gpu_render() -> (gbm::Device<std::fs::File>, smithay::backend::renderer::gles::GlesRenderer) {
        let node = std::env::var("PIXELFLUX_TEST_RENDER_NODE")
            .unwrap_or_else(|_| "/dev/dri/renderD128".to_string());
        crate::gpu_render_init(std::path::Path::new(&node)).expect("GPU render init")
    }

    /// Background and block colours painted into test dmabufs, as `Color32F` components.
    const BG: [f32; 3] = [0.1, 0.2, 0.8];
    const FG: [f32; 3] = [0.9, 0.3, 0.1];

    /// BT.601 limited-range Y/Cb/Cr of an RGB triple in 0..1 — what NVENC's hardware CSC emits.
    fn ycbcr_601(rgb: [f32; 3]) -> [f64; 3] {
        let [r, g, b] = rgb.map(|c| c as f64);
        [
            16.0 + 219.0 * (0.299 * r + 0.587 * g + 0.114 * b),
            128.0 + 224.0 * (-0.168736 * r - 0.331264 * g + 0.5 * b),
            128.0 + 224.0 * (0.5 * r - 0.418688 * g - 0.081312 * b),
        ]
    }

    /// Where the foreground block of a `seed`-painted `w×h` frame sits: a quarter-size block
    /// whose origin moves with the seed.
    fn block_rect(w: u32, h: u32, seed: u32) -> (i32, i32, i32, i32) {
        let x = ((seed * 37) % (w / 2)) as i32 & !1;
        let y = ((seed * 53) % (h / 2)) as i32 & !1;
        (x, y, (w / 4) as i32 & !1, (h / 4) as i32 & !1)
    }

    /// Test helper: allocate a `w×h` ARGB8888 render-target dmabuf on `gbm` and paint it with
    /// the GLES renderer — `BG` everywhere and an `FG` block at `block_rect(seed)` — waiting for
    /// the render to land before returning, as the compositor does before encoding.
    fn painted_dmabuf(
        gbm: &gbm::Device<std::fs::File>,
        renderer: &mut smithay::backend::renderer::gles::GlesRenderer,
        w: u32,
        h: u32,
        seed: u32,
    ) -> (gbm::BufferObject<()>, Dmabuf) {
        use gbm::{BufferObjectFlags, Format as GbmFormat};
        use smithay::backend::renderer::{Bind, Color32F, Frame, Renderer};
        use smithay::utils::{Physical, Rectangle, Size, Transform};
        let bo = gbm
            .create_buffer_object::<()>(w, h, GbmFormat::Argb8888, BufferObjectFlags::RENDERING)
            .expect("GBM buffer");
        let mut dmabuf = crate::create_dmabuf_from_bo(&bo);
        {
            let mut fb = renderer.bind(&mut dmabuf).expect("bind dmabuf");
            let size: Size<i32, Physical> = (w as i32, h as i32).into();
            let mut frame = renderer.render(&mut fb, size, Transform::Normal).expect("render");
            let full: Rectangle<i32, Physical> = Rectangle::from_size(size);
            frame.clear(Color32F::new(BG[0], BG[1], BG[2], 1.0), &[full]).expect("clear");
            let (x, y, bw, bh) = block_rect(w, h, seed);
            let block: Rectangle<i32, Physical> = Rectangle::new((x, y).into(), (bw, bh).into());
            frame
                .draw_solid(
                    block,
                    &[Rectangle::from_size(block.size)],
                    Color32F::new(FG[0], FG[1], FG[2], 1.0),
                )
                .expect("draw block");
            let sync = frame.finish().expect("finish");
            let _ = sync.wait();
        }
        (bo, dmabuf)
    }

    /// Test helper: decode one H.264 access unit (the bytes behind the 10-byte wire header)
    /// with the crate's avcodec decoder and return the mean Y/Cb/Cr inside `rect` and outside it.
    fn decoded_means(
        dec: &mut crate::webcam::decode::AvDecoder,
        pkt: &[u8],
        rect: (i32, i32, i32, i32),
    ) -> ([f64; 3], [f64; 3]) {
        use crate::webcam::decode::Decoder;
        assert!(dec.decode(&pkt[10..]).expect("decode"), "no picture from this access unit");
        let v = dec.frame().expect("decoded frame");
        let (rx, ry, rw, rh) = rect;
        let inside = |x: usize, y: usize| {
            x as i32 >= rx && (x as i32) < rx + rw && y as i32 >= ry && (y as i32) < ry + rh
        };
        let mut acc = [[0f64; 3]; 2];
        let mut cnt = [0f64; 2];
        for y in 0..v.height {
            for x in 0..v.width {
                let k = if inside(x, y) { 0 } else { 1 };
                acc[k][0] += v.y[y * v.y_stride + x] as f64;
                acc[k][1] += v.u[(y / 2) * v.uv_stride + x / 2] as f64;
                acc[k][2] += v.v[(y / 2) * v.uv_stride + x / 2] as f64;
                cnt[k] += 1.0;
            }
        }
        let mean = |k: usize| [acc[k][0] / cnt[k], acc[k][1] / cnt[k], acc[k][2] / cnt[k]];
        (mean(0), mean(1))
    }

    /// Assert decoded region means sit within `tol` of the BT.601 limited-range values of the
    /// painted colours — a wrong pitch, byte order or stale buffer lands far outside this.
    fn assert_painted(label: &str, block: [f64; 3], bg: [f64; 3], tol: f64) {
        let (eb, eg) = (ycbcr_601(FG), ycbcr_601(BG));
        for i in 0..3 {
            assert!(
                (block[i] - eb[i]).abs() <= tol,
                "{label}: block plane {i} = {:.1}, expected {:.1}",
                block[i],
                eb[i]
            );
            assert!(
                (bg[i] - eg[i]).abs() <= tol,
                "{label}: background plane {i} = {:.1}, expected {:.1}",
                bg[i],
                eg[i]
            );
        }
    }

    /// Whether every cached dmabuf import of `enc` is registered with NVENC in place.
    fn all_direct(enc: &NvencEncoder) -> bool {
        !enc.dmabuf_cache.is_empty()
            && enc.dmabuf_cache.values().all(|c| matches!(c.input, DmaBufInput::Direct { .. }))
    }

    /// How the driver mapped the cached dmabuf imports of `enc`, for the test output.
    fn mapped_kind(enc: &NvencEncoder) -> &'static str {
        match enc.dmabuf_cache.values().next().map(|c| c.egl_frame.frame_type) {
            Some(CU_EGL_FRAME_TYPE_PITCH) => "pitch-linear",
            Some(CU_EGL_FRAME_TYPE_ARRAY) => "as a CUDA array",
            _ => "in an unknown frame kind",
        }
    }

    /// On a real GPU with a render node: two GLES-painted dmabufs encode through the dmabuf path
    /// and decode to the painted colours at the painted positions, first with the direct
    /// registration enabled (in place when the driver maps the import pitch-linear, otherwise the
    /// copy arm) and then with it disabled — the two streams must agree, and the decoded content
    /// of both must match the paint. Prints which path the driver gave. Ignored by default; needs
    /// `PIXELFLUX_TEST_RENDER_NODE` or `/dev/dri/renderD128` backed by the NVIDIA GPU.
    #[test]
    #[ignore]
    fn gpu_dmabuf_direct_and_copy_paths_decode_to_the_paint() {
        use crate::webcam::decode::{AvDecoder, Codec};
        let (w, h) = (1920u32, 1080u32);
        let s = settings(w as i32, h as i32, 60.0);
        let (gbm, mut renderer) = gpu_render();
        let egl_display = renderer.egl_context().display().get_display_handle().handle;
        let bufs: Vec<_> = (1..=2u32).map(|seed| (seed, painted_dmabuf(&gbm, &mut renderer, w, h, seed))).collect();
        let mut enc = NvencEncoder::new(&s, egl_display).expect("NVENC init");

        let run = |enc: &mut NvencEncoder, label: &str| -> Vec<Vec<u8>> {
            let mut dec = AvDecoder::new(Codec::H264).expect("avcodec h264");
            let mut out = Vec::new();
            for i in 0..6u64 {
                let (seed, (_, dmabuf)) = &bufs[(i % 2) as usize];
                let pkt = enc.encode(dmabuf, i, 25, i == 0).expect("dmabuf encode");
                assert_eq!(wire_dims(&pkt), (w as u16, h as u16));
                let (block, bg) = decoded_means(&mut dec, &pkt, block_rect(w, h, *seed));
                assert_painted(&format!("{label} frame {i}"), block, bg, 6.0);
                out.push(pkt[10..].to_vec());
            }
            out
        };

        let direct = run(&mut enc, "direct");
        println!(
            "driver mapped the dmabuf {}: {}",
            mapped_kind(&enc),
            if all_direct(&enc) { "registered in place" } else { "direct registration unavailable, copy arm used" }
        );

        enc.direct_dmabuf = false;
        enc.reconfigure_resolution(&s).expect("same-size reconfigure drains the import cache");
        let copied = run(&mut enc, "copy");
        assert!(!all_direct(&enc));
        let identical = direct.iter().zip(&copied).all(|(a, b)| a == b);
        println!(
            "direct vs copy streams: {} ({} vs {} bytes)",
            if identical { "byte-identical" } else { "differ" },
            direct.iter().map(Vec::len).sum::<usize>(),
            copied.iter().map(Vec::len).sum::<usize>()
        );
        if let Ok(dir) = std::env::var("NVENC_TEST_DUMP_DIR") {
            std::fs::write(format!("{dir}/dmabuf-direct.h264"), direct.concat()).unwrap();
            std::fs::write(format!("{dir}/dmabuf-copy.h264"), copied.concat()).unwrap();
        }
    }

    /// On a real GPU: a host frame handed over as BGRA (`rgba_input = false`) and the same image
    /// handed over as RGBA bytes (`rgba_input = true`) both decode to the painted colours — the
    /// input surface is re-registered in the other byte order in place — and the session keeps
    /// encoding across the switch. Ignored by default.
    #[test]
    #[ignore]
    fn gpu_packed_bgra_and_rgba_inputs_agree() {
        use crate::webcam::decode::{AvDecoder, Codec};
        let (w, h) = (1280u32, 720u32);
        let s = settings(w as i32, h as i32, 60.0);
        let rect = block_rect(w, h, 3);
        let paint = |rgba: bool| -> Vec<u8> {
            let to_u8 = |c: f32| (c * 255.0).round() as u8;
            let mut f = vec![0u8; (w * h * 4) as usize];
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    let inside = x >= rect.0 && x < rect.0 + rect.2 && y >= rect.1 && y < rect.1 + rect.3;
                    let c = if inside { FG } else { BG };
                    let px = &mut f[((y as u32 * w + x as u32) * 4) as usize..][..4];
                    let (r, g, b) = (to_u8(c[0]), to_u8(c[1]), to_u8(c[2]));
                    if rgba {
                        px.copy_from_slice(&[r, g, b, 255]);
                    } else {
                        px.copy_from_slice(&[b, g, r, 255]);
                    }
                }
            }
            f
        };
        let bgra = paint(false);
        let rgba = paint(true);
        let mut enc = NvencEncoder::new(&s, ptr::null()).expect("NVENC init");
        let mut dec = AvDecoder::new(Codec::H264).expect("avcodec h264");
        let stride = (w * 4) as usize;
        for (i, (buf, is_rgba)) in [(&bgra, false), (&rgba, true), (&bgra, false), (&rgba, true)]
            .into_iter()
            .enumerate()
        {
            let pkt = enc
                .encode_cpu_packed(buf, stride, is_rgba, i as u64, 25, i == 0)
                .expect("packed encode");
            assert_eq!(
                enc.input_format,
                if is_rgba {
                    NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ABGR
                } else {
                    NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB
                }
            );
            let (block, bg) = decoded_means(&mut dec, &pkt, rect);
            assert_painted(&format!("frame {i} rgba={is_rgba}"), block, bg, 6.0);
        }
    }

    /// On a real GPU with a render node: per-frame wall and CPU cost of the dmabuf path with the
    /// in-place registration (when the driver maps the import pitch-linear) against the per-frame
    /// copy, 1080p, two painted buffers alternating. Prints both; ignored by default.
    #[test]
    #[ignore]
    fn gpu_bench_dmabuf_paths() {
        let (w, h) = (1920u32, 1080u32);
        let n: usize = std::env::var("NVENC_BENCH_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
        let s = settings(w as i32, h as i32, 60.0);
        let (gbm, mut renderer) = gpu_render();
        let egl_display = renderer.egl_context().display().get_display_handle().handle;
        let bufs: Vec<_> = (1..=2u32).map(|seed| painted_dmabuf(&gbm, &mut renderer, w, h, seed).1).collect();
        let mut enc = NvencEncoder::new(&s, egl_display).expect("NVENC init");
        for pass in 0..2 {
            enc.direct_dmabuf = pass == 0;
            enc.reconfigure_resolution(&s).expect("reconfigure drains the import cache");
            enc.encode(&bufs[0], 0, 25, true).expect("warm-up");
            enc.encode(&bufs[1], 1, 25, false).expect("warm-up");
            let label = if all_direct(&enc) {
                format!("dmabuf registered in place ({})", mapped_kind(&enc))
            } else if pass == 0 {
                format!("dmabuf copy arm (direct registration unavailable, mapped {})", mapped_kind(&enc))
            } else {
                format!("dmabuf per-frame copy (mapped {})", mapped_kind(&enc))
            };
            per_frame(&label, n, |i| {
                enc.encode(&bufs[i % 2], 2 + i as u64, 25, false).expect("encode");
            });
        }
    }

    /// On a real GPU: per-frame wall and CPU cost of the readback→NVENC hand-over, 1080p host
    /// frames: the packed upload with the hardware CSC (BGRA and RGBA), and that same packed
    /// upload as a synchronous copy. Prints all; ignored by default.
    #[test]
    #[ignore]
    fn gpu_bench_readback_upload() {
        let (w, h) = (1920u32, 1080u32);
        let n: usize = std::env::var("NVENC_BENCH_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
        let s = settings(w as i32, h as i32, 60.0);
        let frames: Vec<Vec<u8>> = (0..4u8).map(|k| frame(w as usize, h as usize, 10 + 40 * k)).collect();
        let stride = (w * 4) as usize;
        let mut enc = NvencEncoder::new(&s, ptr::null()).expect("NVENC init");

        enc.reconfigure_resolution(&s).expect("reconfigure");
        enc.encode_cpu_packed(&frames[0], stride, false, 0, 25, true).expect("warm-up");
        per_frame("encode_cpu_packed BGRA (pinned, async upload, hardware CSC)", n, |i| {
            enc.encode_cpu_packed(&frames[i % 4], stride, false, 1 + i as u64, 25, false).expect("packed");
        });

        enc.reconfigure_resolution(&s).expect("reconfigure");
        enc.encode_cpu_packed(&frames[0], stride, true, 0, 25, true).expect("warm-up");
        per_frame("encode_cpu_packed RGBA (pinned, async upload, hardware CSC)", n, |i| {
            enc.encode_cpu_packed(&frames[i % 4], stride, true, 1 + i as u64, 25, false).expect("packed");
        });

        enc.reconfigure_resolution(&s).expect("reconfigure");
        enc.encode_cpu_packed(&frames[0], stride, false, 0, 25, true).expect("warm-up");
        per_frame("packed BGRA with a synchronous cuMemcpy2D upload", n, |i| unsafe {
            let _ = (enc.cuda.cuCtxPushCurrent_v2)(enc.cuda_context);
            let src = &frames[i % 4];
            enc.pin_host_source(src.as_ptr() as usize, src.len());
            let copy = CUDA_MEMCPY2D {
                srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                srcHost: src.as_ptr() as *const c_void,
                srcPitch: stride,
                dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstDevice: enc.input_device_ptr,
                dstPitch: enc.input_pitch,
                WidthInBytes: stride,
                Height: h as usize,
                ..Default::default()
            };
            assert_eq!((enc.cuda.cuMemcpy2D_v2)(&copy), CUresult::CUDA_SUCCESS);
            enc.submit_frame(enc.mapped_input_buffer, enc.input_format, 1 + i as u64, false)
                .expect("submit");
            (enc.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
        });
    }
}

#[cfg(test)]
mod version_tests {
    use super::*;

    /// Every version-tagged struct, in one fixed order, paired with its pinned compile-time
    /// `NV_ENC_*_VER` constant — the reference both version tests iterate.
    const ALL: [(NvStruct, u32); 13] = [
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
        (NvStruct::CapsParam, NV_ENC_CAPS_PARAM_VER),
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

#[cfg(test)]
mod decision_tests {
    use super::*;

    /// A caps query that returns `Some(0)` means the GPU lacks 4:4:4, so a 4:4:4 request is met
    /// with 4:2:0 and flagged as a downgrade; `Some(1)` keeps 4:4:4; an unqueryable cap (`None`)
    /// leaves the request untouched rather than downgrading on missing information.
    #[test]
    fn caps_chroma_downgrade() {
        let d = decide_caps(true, 1920, 1080, Some(0), None, None);
        assert!(!d.fullcolor && d.downgraded_color && d.too_large.is_none());

        let d = decide_caps(true, 1920, 1080, Some(1), None, None);
        assert!(d.fullcolor && !d.downgraded_color);

        let d = decide_caps(true, 1920, 1080, None, None, None);
        assert!(d.fullcolor && !d.downgraded_color);

        // A 4:2:0 request is never a downgrade whatever the cap says.
        let d = decide_caps(false, 1920, 1080, Some(0), None, None);
        assert!(!d.fullcolor && !d.downgraded_color);
    }

    /// A capture beyond the driver's reported maximum dimensions is flagged `too_large` (the
    /// caller then declines NVENC and uses software); a capture within them, or one whose caps are
    /// unknown, is not.
    #[test]
    fn caps_dimension_gate() {
        assert_eq!(
            decide_caps(false, 5120, 2160, None, Some(4096), Some(4096)).too_large,
            Some((4096, 4096))
        );
        assert_eq!(
            decide_caps(false, 3840, 4320, None, Some(4096), Some(4096)).too_large,
            Some((4096, 4096))
        );
        assert!(decide_caps(false, 3840, 2160, None, Some(4096), Some(4096)).too_large.is_none());
        assert!(decide_caps(false, 7680, 4320, None, None, None).too_large.is_none());
        // A zero cap is treated as unknown, not as "everything is too large".
        assert!(decide_caps(false, 3840, 2160, None, Some(0), Some(0)).too_large.is_none());
    }

    /// The resize headroom lifts the request to the 5.2-ceiling floor but never past the driver
    /// maximum, so initializing with headroom cannot itself exceed what the GPU supports.
    #[test]
    fn headroom_is_floored_and_capped() {
        assert_eq!(nvenc_headroom(1920, 4096, Some(8192)), 4096);
        assert_eq!(nvenc_headroom(3840, 4096, Some(4096)), 4096);
        assert_eq!(nvenc_headroom(6000, 4096, Some(8192)), 6000);
        assert_eq!(nvenc_headroom(1920, 4096, None), 4096);
        assert_eq!(nvenc_headroom(1920, 4096, Some(2048)), 2048);
    }

    /// Two buffers that reuse one fd number but differ in any identity field are distinct, so a
    /// cache keyed by fd cannot return a stale import: a new inode (the kernel reissues one per
    /// dma-buf since 5.3), a new size, or new geometry each breaks the match.
    #[test]
    fn dmabuf_identity_distinguishes_recycled_fd() {
        let base = DmaBufIdentity { dev: 1, ino: 10, size: 100, modifier: 0, width: 1920, height: 1080 };
        assert_eq!(base, base);
        let mut new_ino = base;
        new_ino.ino = 11;
        assert_ne!(base, new_ino);
        let mut new_size = base;
        new_size.size = 200;
        assert_ne!(base, new_size);
        let mut new_mod = base;
        new_mod.modifier = 1;
        assert_ne!(base, new_mod);
        let mut new_geom = base;
        new_geom.width = 1280;
        assert_ne!(base, new_geom);
    }

    /// `probe` reads a real fd deterministically and folds the modifier and geometry into the
    /// identity: the same fd and parameters yield equal identities, and changing the modifier or the
    /// geometry changes the identity even on the same fd.
    #[test]
    fn dmabuf_identity_probe_is_stable_and_parameterized() {
        use std::os::fd::AsRawFd;
        let f = std::fs::File::open("/dev/null").expect("open /dev/null");
        let fd = f.as_raw_fd();
        let a = DmaBufIdentity::probe(fd, 0x1234, 1920, 1080);
        assert_eq!(a, DmaBufIdentity::probe(fd, 0x1234, 1920, 1080));
        assert_ne!(a, DmaBufIdentity::probe(fd, 0x9999, 1920, 1080));
        assert_ne!(a, DmaBufIdentity::probe(fd, 0x1234, 1280, 720));
    }

    /// Test helper: a mapped `CUeglFrame` of one plane with the given kind, geometry and pitch,
    /// its first plane at `plane` (a device pointer for the pitch kind, an array handle for the
    /// array kind) in four 8-bit channels.
    fn egl_frame(frame_type: u32, w: u32, h: u32, pitch: u32, plane: usize) -> CUeglFrame {
        let mut f: CUeglFrame = unsafe { std::mem::zeroed() };
        f.frame_type = frame_type;
        f.plane_count = 1;
        f.width = w;
        f.height = h;
        f.pitch = pitch;
        f.num_channels = 4;
        f.cu_format = CU_AD_FORMAT_U8;
        f.frame.p_pitch = [plane as *mut c_void, ptr::null_mut(), ptr::null_mut()];
        f
    }

    /// A pitch-linear mapping whose first plane covers the session geometry at a 4-byte-aligned
    /// pitch of at least `width * 4` is registered with NVENC in place as a device pointer at its
    /// own pitch; a null plane, a short or unaligned pitch, a frame smaller than the session, or a
    /// frame without planes each take the per-frame copy.
    #[test]
    fn pitch_linear_frame_direct_registration_rules() {
        let pitch_ok = egl_frame(CU_EGL_FRAME_TYPE_PITCH, 1920, 1080, 7680, 0x1000);
        assert_eq!(direct_plane(&pitch_ok, 1920, 1080), Some(DirectPlane::Pitch(7680)));
        let padded = egl_frame(CU_EGL_FRAME_TYPE_PITCH, 1920, 1080, 8192, 0x1000);
        assert_eq!(direct_plane(&padded, 1920, 1080), Some(DirectPlane::Pitch(8192)));
        let larger = egl_frame(CU_EGL_FRAME_TYPE_PITCH, 2048, 1200, 8192, 0x1000);
        assert_eq!(direct_plane(&larger, 1920, 1080), Some(DirectPlane::Pitch(8192)));

        let null_plane = egl_frame(CU_EGL_FRAME_TYPE_PITCH, 1920, 1080, 7680, 0);
        assert_eq!(direct_plane(&null_plane, 1920, 1080), None);
        let short_pitch = egl_frame(CU_EGL_FRAME_TYPE_PITCH, 1920, 1080, 7676, 0x1000);
        assert_eq!(direct_plane(&short_pitch, 1920, 1080), None);
        let unaligned = egl_frame(CU_EGL_FRAME_TYPE_PITCH, 1920, 1080, 7682, 0x1000);
        assert_eq!(direct_plane(&unaligned, 1920, 1080), None);
        let smaller = egl_frame(CU_EGL_FRAME_TYPE_PITCH, 1280, 720, 7680, 0x1000);
        assert_eq!(direct_plane(&smaller, 1920, 1080), None);
        let mut no_planes = pitch_ok;
        no_planes.plane_count = 0;
        assert_eq!(direct_plane(&no_planes, 1920, 1080), None);
        assert_eq!(direct_plane(&pitch_ok, 0, 1080), None);
        let mut unknown_kind = pitch_ok;
        unknown_kind.frame_type = 7;
        assert_eq!(direct_plane(&unknown_kind, 1920, 1080), None);
    }

    /// A CUDA-array mapping of four 8-bit channels covering the session geometry is registered in
    /// place as a CUDA array whose pitch word is the array's row width in bytes; an array of any
    /// other element layout, a null array, or one smaller than the session takes the copy.
    #[test]
    fn cuda_array_frame_direct_registration_rules() {
        let array = egl_frame(CU_EGL_FRAME_TYPE_ARRAY, 1920, 1080, 0, 0x2000);
        assert_eq!(direct_plane(&array, 1920, 1080), Some(DirectPlane::Array(7680)));
        let wider = egl_frame(CU_EGL_FRAME_TYPE_ARRAY, 2048, 1080, 0, 0x2000);
        assert_eq!(direct_plane(&wider, 1920, 1080), Some(DirectPlane::Array(8192)));

        let null_array = egl_frame(CU_EGL_FRAME_TYPE_ARRAY, 1920, 1080, 0, 0);
        assert_eq!(direct_plane(&null_array, 1920, 1080), None);
        let mut one_channel = array;
        one_channel.num_channels = 1;
        assert_eq!(direct_plane(&one_channel, 1920, 1080), None);
        let mut wide_elements = array;
        wide_elements.cu_format = 3;
        assert_eq!(direct_plane(&wide_elements, 1920, 1080), None);
        let smaller = egl_frame(CU_EGL_FRAME_TYPE_ARRAY, 1280, 720, 0, 0x2000);
        assert_eq!(direct_plane(&smaller, 1920, 1080), None);
    }

    /// The dmabuf fourcc picks the NVENC packed format with the same byte order: XR24 / AR24 are
    /// B,G,R,A in memory and so NVENC `ARGB`; XB24 / AB24 are R,G,B,A and so `ABGR`; anything else
    /// has no packed 8-bit NVENC equivalent.
    #[test]
    fn fourcc_selects_nvenc_byte_order() {
        for code in [Fourcc::Argb8888, Fourcc::Xrgb8888] {
            assert_eq!(fourcc_nvenc_format(code), Some(NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB));
        }
        for code in [Fourcc::Abgr8888, Fourcc::Xbgr8888] {
            assert_eq!(fourcc_nvenc_format(code), Some(NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ABGR));
        }
        for code in [Fourcc::Rgb565, Fourcc::Nv12, Fourcc::Argb2101010, Fourcc::Bgra8888, Fourcc::Rgba8888] {
            assert_eq!(fourcc_nvenc_format(code), None, "{code:?}");
        }
    }
}
