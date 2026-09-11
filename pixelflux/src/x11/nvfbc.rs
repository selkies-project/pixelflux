/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! NVIDIA Framebuffer Capture (NvFBC): the zero-copy X11 capture path, where the NVIDIA driver
//! composites the X screen straight into video memory and NVENC encodes it there.
//!
//! The XShm path in [`super`] is the general one, and it copies once: the X server blits the
//! screen into a shared-memory surface, and a hardware session then uploads that surface across
//! PCIe. NvFBC removes both. The capture subsystem lives in the NVIDIA X driver, composites each
//! frame into a buffer it owns in video memory, and hands back a `CUdeviceptr` — which is
//! registered with NVENC **in place**, exactly as a Wayland dmabuf is, so a frame is never read,
//! written or copied by the CPU between the screen and the bitstream.
//!
//! Three properties follow from that, and the module is built around keeping all three:
//!
//! 1. **Zero copy.** The device pointer is stable for as long as the geometry is, so it is
//!    registered and mapped once (`NvencEncoder::encode_cuda_pitch`) and every later frame reuses
//!    that registration.
//! 2. **Lowest latency.** The session runs in push model — the driver generates a frame when an
//!    application damages the screen rather than on a sampling timer — and each grab is issued
//!    with `NOWAIT_IF_NEW_FRAME_READY`, so a frame that already exists is taken immediately and
//!    one that does not is waited for only until the frame deadline. Direct capture is allowed,
//!    which lets the driver attach to a fullscreen unoccluded application and have it present
//!    into the capture buffer, bypassing the X server entirely.
//! 3. **Unrestricted frame rate.** Capture and encode share one thread with no hand-off, and
//!    neither does any per-pixel CPU work: no shm round-trip, no content hashing, no cursor
//!    blend. What the loop can sustain is what NVENC can sustain.
//!
//! Requirements are checked rather than assumed, because none of them holds everywhere: the
//! session must encode on NVENC (the device pointer is meaningless to any other encoder), the
//! driver must offer NvFBC on this GPU and X server, and a watermark keeps the XShm path since
//! blending it would mean reading the frame back. `open` reports what it found and returns
//! `None`, and the caller streams through XShm instead.
//!
//! The API is reached the way NVENC's is: `libnvidia-fbc.so.1` is loaded at run time and every
//! entry point comes from the function table `NvFBCCreateInstance` fills, so the build links no
//! NVIDIA library and one binary runs wherever a driver is installed. The structures below are
//! the ABI of NvFBC 1.8 (`nvcodec-sys/headers/NvFBC.h`); each carries a version word packing its
//! own `size_of`, which is what makes the layout assertions in `abi_tests` a real check.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::upper_case_acronyms)]

use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use libloading::Library;

use super::Controls;
use crate::encoders::nvenc::NvencEncoder;
use crate::encoders::software::{EncodedStripe, StripeState};
use crate::pipeline::decide_hw_fullframe;
use crate::recording_sink::RecordingSink;
use crate::RustCaptureSettings;
use nvcodec_sys::cuda::CUdeviceptr;

/// NvFBC API version this module speaks: minor in the low byte, major above it.
const NVFBC_VERSION: u32 = 8 | (1 << 8);

/// The version word an NvFBC parameter struct carries: its own size, its revision, and the
/// API version, packed as the SDK's `NVFBC_STRUCT_VERSION` macro packs them (the API version
/// lands at bit 24, where only its low byte survives the 32-bit result).
const fn struct_ver(size: usize, rev: u32) -> u32 {
    (size as u32) | (rev << 16) | (NVFBC_VERSION << 24)
}

type NVFBCSTATUS = u32;
const NVFBC_SUCCESS: NVFBCSTATUS = 0;
/// The X server is gone; the handle cannot be reused, only destroyed and recreated.
const NVFBC_ERR_X: NVFBCSTATUS = 10;
/// A modeset invalidated the session; recreating it resumes capture.
const NVFBC_ERR_MUST_RECREATE: NVFBCSTATUS = 16;

type NVFBC_BOOL = u32;
const NVFBC_FALSE: NVFBC_BOOL = 0;
const NVFBC_TRUE: NVFBC_BOOL = 1;

type NVFBC_SESSION_HANDLE = u64;

/// Capture into a CUDA buffer in video memory, the only capture type this module uses.
const NVFBC_CAPTURE_SHARED_CUDA: u32 = 1;
/// Track the entire X screen (the framebuffer), rather than one RandR output.
const NVFBC_TRACKING_SCREEN: u32 = 2;
/// The driver's native buffer format: B,G,R,A in memory, 32 bpp, needing no conversion pass.
const NVFBC_BUFFER_FORMAT_BGRA: u32 = 5;
/// Return whatever is in the capture buffer without waiting for the driver to render.
const NVFBC_TOCUDA_GRAB_FLAGS_NOWAIT: u32 = 1 << 0;
/// Take a frame that is already waiting without blocking, otherwise wait for one.
const NVFBC_TOCUDA_GRAB_FLAGS_NOWAIT_IF_NEW_FRAME_READY: u32 = 1 << 2;

const NVFBC_OUTPUT_MAX: usize = 5;
const NVFBC_OUTPUT_NAME_LEN: usize = 128;

/// A region of the tracked framebuffer, in its own coordinates.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct NVFBC_BOX {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// A frame's pixel dimensions.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct NVFBC_SIZE {
    w: u32,
    h: u32,
}

/// What the driver reports about a captured frame: its geometry and byte size, whether it is
/// one the client has not seen, how many the driver generated meanwhile, and whether it needed
/// a conversion pass or came straight from a fullscreen application.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NVFBC_FRAME_GRAB_INFO {
    dwWidth: u32,
    dwHeight: u32,
    dwByteSize: u32,
    dwCurrentFrame: u32,
    bIsNewFrame: NVFBC_BOOL,
    ulTimestampUs: u64,
    dwMissedFrames: u32,
    bRequiredPostProcessing: NVFBC_BOOL,
    bDirectCapture: NVFBC_BOOL,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NVFBC_CREATE_HANDLE_PARAMS {
    dwVersion: u32,
    privateData: *const c_void,
    privateDataSize: u32,
    bExternallyManagedContext: NVFBC_BOOL,
    glxCtx: *mut c_void,
    glxFBConfig: *mut c_void,
}

/// The only parameter block of the destroy / context calls: a version word.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NVFBC_VERSION_ONLY_PARAMS {
    dwVersion: u32,
}

/// One RandR output of the tracked X screen, as `NvFBCGetStatus` reports it.
#[repr(C)]
#[derive(Clone, Copy)]
struct NVFBC_RANDR_OUTPUT_INFO {
    dwId: u32,
    name: [c_char; NVFBC_OUTPUT_NAME_LEN],
    trackedBox: NVFBC_BOX,
}

impl Default for NVFBC_RANDR_OUTPUT_INFO {
    fn default() -> Self {
        Self { dwId: 0, name: [0; NVFBC_OUTPUT_NAME_LEN], trackedBox: NVFBC_BOX::default() }
    }
}

/// What the driver says about capture on this X server: whether it is possible at all, whether
/// a session can be created right now, the framebuffer size, and the connected outputs.
#[repr(C)]
#[derive(Clone, Copy)]
struct NVFBC_GET_STATUS_PARAMS {
    dwVersion: u32,
    bIsCapturePossible: NVFBC_BOOL,
    bCurrentlyCapturing: NVFBC_BOOL,
    bCanCreateNow: NVFBC_BOOL,
    screenSize: NVFBC_SIZE,
    bXRandRAvailable: NVFBC_BOOL,
    outputs: [NVFBC_RANDR_OUTPUT_INFO; NVFBC_OUTPUT_MAX],
    dwOutputNum: u32,
    dwNvFBCVersion: u32,
    bInModeset: NVFBC_BOOL,
}

impl Default for NVFBC_GET_STATUS_PARAMS {
    fn default() -> Self {
        Self {
            dwVersion: 0,
            bIsCapturePossible: 0,
            bCurrentlyCapturing: 0,
            bCanCreateNow: 0,
            screenSize: NVFBC_SIZE::default(),
            bXRandRAvailable: 0,
            outputs: [NVFBC_RANDR_OUTPUT_INFO::default(); NVFBC_OUTPUT_MAX],
            dwOutputNum: 0,
            dwNvFBCVersion: 0,
            bInModeset: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NVFBC_CREATE_CAPTURE_SESSION_PARAMS {
    dwVersion: u32,
    eCaptureType: u32,
    eTrackingType: u32,
    dwOutputId: u32,
    captureBox: NVFBC_BOX,
    frameSize: NVFBC_SIZE,
    bWithCursor: NVFBC_BOOL,
    bDisableAutoModesetRecovery: NVFBC_BOOL,
    bRoundFrameSize: NVFBC_BOOL,
    dwSamplingRateMs: u32,
    bPushModel: NVFBC_BOOL,
    bAllowDirectCapture: NVFBC_BOOL,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NVFBC_TOCUDA_SETUP_PARAMS {
    dwVersion: u32,
    eBufferFormat: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NVFBC_TOCUDA_GRAB_FRAME_PARAMS {
    dwVersion: u32,
    dwFlags: u32,
    pCUDADeviceBuffer: *mut c_void,
    pFrameGrabInfo: *mut NVFBC_FRAME_GRAB_INFO,
    dwTimeoutMs: u32,
}

/// The driver's entry points, as `NvFBCCreateInstance` fills them. The retired slots are the
/// deprecated hardware-encoder interface and are never called.
#[repr(C)]
#[derive(Clone, Copy)]
struct NVFBC_API_FUNCTION_LIST {
    dwVersion: u32,
    nvFBCGetLastErrorStr: Option<unsafe extern "C" fn(NVFBC_SESSION_HANDLE) -> *const c_char>,
    nvFBCCreateHandle:
        Option<unsafe extern "C" fn(*mut NVFBC_SESSION_HANDLE, *mut NVFBC_CREATE_HANDLE_PARAMS) -> NVFBCSTATUS>,
    nvFBCDestroyHandle:
        Option<unsafe extern "C" fn(NVFBC_SESSION_HANDLE, *mut NVFBC_VERSION_ONLY_PARAMS) -> NVFBCSTATUS>,
    nvFBCGetStatus:
        Option<unsafe extern "C" fn(NVFBC_SESSION_HANDLE, *mut NVFBC_GET_STATUS_PARAMS) -> NVFBCSTATUS>,
    nvFBCCreateCaptureSession: Option<
        unsafe extern "C" fn(NVFBC_SESSION_HANDLE, *mut NVFBC_CREATE_CAPTURE_SESSION_PARAMS) -> NVFBCSTATUS,
    >,
    nvFBCDestroyCaptureSession:
        Option<unsafe extern "C" fn(NVFBC_SESSION_HANDLE, *mut NVFBC_VERSION_ONLY_PARAMS) -> NVFBCSTATUS>,
    nvFBCToSysSetUp: *mut c_void,
    nvFBCToSysGrabFrame: *mut c_void,
    nvFBCToCudaSetUp:
        Option<unsafe extern "C" fn(NVFBC_SESSION_HANDLE, *mut NVFBC_TOCUDA_SETUP_PARAMS) -> NVFBCSTATUS>,
    nvFBCToCudaGrabFrame:
        Option<unsafe extern "C" fn(NVFBC_SESSION_HANDLE, *mut NVFBC_TOCUDA_GRAB_FRAME_PARAMS) -> NVFBCSTATUS>,
    pad1: *mut c_void,
    pad2: *mut c_void,
    pad3: *mut c_void,
    nvFBCBindContext:
        Option<unsafe extern "C" fn(NVFBC_SESSION_HANDLE, *mut NVFBC_VERSION_ONLY_PARAMS) -> NVFBCSTATUS>,
    nvFBCReleaseContext:
        Option<unsafe extern "C" fn(NVFBC_SESSION_HANDLE, *mut NVFBC_VERSION_ONLY_PARAMS) -> NVFBCSTATUS>,
    pad4: *mut c_void,
    pad5: *mut c_void,
    pad6: *mut c_void,
    pad7: *mut c_void,
    nvFBCToGLSetUp: *mut c_void,
    nvFBCToGLGrabFrame: *mut c_void,
}

impl Default for NVFBC_API_FUNCTION_LIST {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

type NvFBCCreateInstanceFn = unsafe extern "C" fn(*mut NVFBC_API_FUNCTION_LIST) -> NVFBCSTATUS;

/// Runtime name of the NvFBC library. The NVIDIA container toolkit injects it under the `video`
/// driver capability, alongside `libnvidia-encode.so.1`.
const NVFBC_LIBRARY: &str = "libnvidia-fbc.so.1";

/// A driver call that failed: the status the driver returned, and its own account of it.
///
/// The status is kept because what to do next follows from it and nothing else: a session
/// invalidated by a modeset is rebuilt, a lost X server needs a whole new handle, and anything
/// else ends the capture.
#[derive(Debug)]
struct NvfbcError {
    status: NVFBCSTATUS,
    message: String,
}

impl std::fmt::Display for NvfbcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.message.is_empty() {
            write!(f, "({})", self.status)
        } else {
            write!(f, "{} ({})", self.message, self.status)
        }
    }
}

/// One frame the driver composited into video memory.
struct GrabbedFrame {
    device_ptr: CUdeviceptr,
    width: u32,
    height: u32,
    byte_size: u32,
    /// Whether the driver rendered this frame since the previous grab. A static screen reports
    /// `false` and costs nothing to skip.
    is_new: bool,
    /// Whether a fullscreen application presented straight into the capture buffer, bypassing
    /// the X server.
    direct: bool,
    /// Whether the driver had to run a conversion pass, which would mean an extra frame copy;
    /// the native BGRA request is chosen so that it does not.
    post_processed: bool,
}

/// A live NvFBC capture: the loaded library, the driver's function table, the client handle, and
/// the capture session bound to it.
///
/// The handle owns an OpenGL and a CUDA context that the driver binds to the thread that created
/// it, and the captured device pointer belongs to whichever CUDA context was current at setup —
/// which is why the whole object is built, used and dropped on one thread, with NVENC's own
/// primary context pushed current. Sharing that context is what makes the capture zero-copy: the
/// pointer the driver hands back is one NVENC can register directly.
struct NvfbcSession {
    _lib: Library,
    funcs: NVFBC_API_FUNCTION_LIST,
    handle: NVFBC_SESSION_HANDLE,
    capture_open: bool,
    /// The region of the framebuffer being captured, and the frame size it is delivered at.
    region: NVFBC_BOX,
    size: NVFBC_SIZE,
    with_cursor: bool,
}

impl NvfbcSession {
    /// Load `libnvidia-fbc.so.1` and fill the driver's function table, erroring when the library
    /// is absent (no NVIDIA driver, or a container without the `video` capability) or too old to
    /// answer for this API version.
    fn load() -> Result<(Library, NVFBC_API_FUNCTION_LIST), String> {
        unsafe {
            let lib = Library::new(NVFBC_LIBRARY)
                .map_err(|e| format!("could not load {NVFBC_LIBRARY}: {e}"))?;
            let create: libloading::Symbol<NvFBCCreateInstanceFn> = lib
                .get(b"NvFBCCreateInstance\0")
                .map_err(|e| format!("missing symbol NvFBCCreateInstance: {e}"))?;
            let mut funcs = NVFBC_API_FUNCTION_LIST { dwVersion: NVFBC_VERSION, ..Default::default() };
            let st = create(&mut funcs);
            if st != NVFBC_SUCCESS {
                return Err(format!("NvFBCCreateInstance failed ({st})"));
            }
            if funcs.nvFBCCreateHandle.is_none()
                || funcs.nvFBCGetStatus.is_none()
                || funcs.nvFBCCreateCaptureSession.is_none()
                || funcs.nvFBCToCudaSetUp.is_none()
                || funcs.nvFBCToCudaGrabFrame.is_none()
            {
                return Err("the driver's NvFBC table has no CUDA capture entry points".into());
            }
            Ok((lib, funcs))
        }
    }

    /// The driver's message for the last failure on this handle, for a log line that says what
    /// went wrong rather than only that something did.
    fn last_error(&self) -> String {
        match self.funcs.nvFBCGetLastErrorStr {
            Some(f) => unsafe {
                let p = f(self.handle);
                if p.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            },
            None => String::new(),
        }
    }

    /// Turn a driver status into an error carrying both the code and the driver's own message.
    fn fail(&self, what: &str, status: NVFBCSTATUS) -> NvfbcError {
        let detail = self.last_error();
        let message = if detail.is_empty() {
            format!("{what} failed")
        } else {
            format!("{what} failed: {detail}")
        };
        NvfbcError { status, message }
    }

    /// Open a client handle on the X server named by the environment, letting NvFBC create and
    /// manage the OpenGL context it captures through.
    ///
    /// The handle is bound to the calling thread, and every later call has to come from it.
    fn open() -> Result<Self, String> {
        let (lib, funcs) = Self::load()?;
        let mut handle: NVFBC_SESSION_HANDLE = 0;
        let mut params = NVFBC_CREATE_HANDLE_PARAMS {
            dwVersion: struct_ver(std::mem::size_of::<NVFBC_CREATE_HANDLE_PARAMS>(), 2),
            bExternallyManagedContext: NVFBC_FALSE,
            ..Default::default()
        };
        let st = unsafe { (funcs.nvFBCCreateHandle.unwrap())(&mut handle, &mut params) };
        if st != NVFBC_SUCCESS {
            return Err(format!("NvFBCCreateHandle failed ({st})"));
        }
        Ok(Self {
            _lib: lib,
            funcs,
            handle,
            capture_open: false,
            region: NVFBC_BOX::default(),
            size: NVFBC_SIZE::default(),
            with_cursor: false,
        })
    }

    /// Ask the driver whether capture is possible here and how large the framebuffer is.
    fn status(&self) -> Result<NVFBC_GET_STATUS_PARAMS, NvfbcError> {
        let mut params = NVFBC_GET_STATUS_PARAMS {
            dwVersion: struct_ver(std::mem::size_of::<NVFBC_GET_STATUS_PARAMS>(), 2),
            ..Default::default()
        };
        let st = unsafe { (self.funcs.nvFBCGetStatus.unwrap())(self.handle, &mut params) };
        if st != NVFBC_SUCCESS {
            return Err(self.fail("NvFBCGetStatus", st));
        }
        Ok(params)
    }

    /// Start capturing `region` of the framebuffer into video memory, delivered at `size`.
    ///
    /// The session is programmed for latency rather than for throughput smoothing: push model, so
    /// the driver generates a frame when an application damages the screen instead of polling on
    /// a timer; direct capture allowed, so a fullscreen unoccluded application presents into the
    /// capture buffer without going through the X server (which the driver takes up only when the
    /// cursor is not composited); and the native BGRA format, which needs no conversion pass and
    /// so no extra copy. Modeset recovery is left to the driver, which rebuilds the session and
    /// resumes on its own.
    fn start(
        &mut self,
        region: NVFBC_BOX,
        size: NVFBC_SIZE,
        with_cursor: bool,
    ) -> Result<(), NvfbcError> {
        self.stop();
        let mut params = NVFBC_CREATE_CAPTURE_SESSION_PARAMS {
            dwVersion: struct_ver(std::mem::size_of::<NVFBC_CREATE_CAPTURE_SESSION_PARAMS>(), 6),
            eCaptureType: NVFBC_CAPTURE_SHARED_CUDA,
            eTrackingType: NVFBC_TRACKING_SCREEN,
            dwOutputId: 0,
            captureBox: region,
            frameSize: size,
            bWithCursor: if with_cursor { NVFBC_TRUE } else { NVFBC_FALSE },
            bDisableAutoModesetRecovery: NVFBC_FALSE,
            bRoundFrameSize: NVFBC_FALSE,
            dwSamplingRateMs: 0,
            bPushModel: NVFBC_TRUE,
            bAllowDirectCapture: NVFBC_TRUE,
        };
        let st = unsafe { (self.funcs.nvFBCCreateCaptureSession.unwrap())(self.handle, &mut params) };
        if st != NVFBC_SUCCESS {
            return Err(self.fail("NvFBCCreateCaptureSession", st));
        }
        self.capture_open = true;
        let mut setup = NVFBC_TOCUDA_SETUP_PARAMS {
            dwVersion: struct_ver(std::mem::size_of::<NVFBC_TOCUDA_SETUP_PARAMS>(), 1),
            eBufferFormat: NVFBC_BUFFER_FORMAT_BGRA,
        };
        let st = unsafe { (self.funcs.nvFBCToCudaSetUp.unwrap())(self.handle, &mut setup) };
        if st != NVFBC_SUCCESS {
            let msg = self.fail("NvFBCToCudaSetUp", st);
            self.stop();
            return Err(msg);
        }
        self.region = region;
        self.size = size;
        self.with_cursor = with_cursor;
        Ok(())
    }

    /// End the capture session, leaving the handle open for another `start`.
    fn stop(&mut self) {
        if !self.capture_open {
            return;
        }
        let mut params = NVFBC_VERSION_ONLY_PARAMS {
            dwVersion: struct_ver(std::mem::size_of::<NVFBC_VERSION_ONLY_PARAMS>(), 1),
        };
        if let Some(f) = self.funcs.nvFBCDestroyCaptureSession {
            unsafe { f(self.handle, &mut params) };
        }
        self.capture_open = false;
    }

    /// Take the next frame, waiting at most `timeout` for the driver to produce one.
    ///
    /// A frame the client has not seen yet is returned without waiting at all; otherwise the call
    /// blocks until one is rendered or the timeout expires, and then returns the frame already in
    /// the buffer with `is_new` false. Nothing is copied either way: the device pointer addresses
    /// the driver's own capture buffer, which stays valid until the next grab.
    fn grab(&mut self, timeout: Duration) -> Result<GrabbedFrame, NvfbcError> {
        self.grab_with(NVFBC_TOCUDA_GRAB_FLAGS_NOWAIT_IF_NEW_FRAME_READY, timeout)
    }

    /// The grab above, with the driver's wait behaviour named explicitly.
    fn grab_with(&mut self, flags: u32, timeout: Duration) -> Result<GrabbedFrame, NvfbcError> {
        let mut device_ptr: CUdeviceptr = 0;
        let mut info = NVFBC_FRAME_GRAB_INFO::default();
        let mut params = NVFBC_TOCUDA_GRAB_FRAME_PARAMS {
            dwVersion: struct_ver(std::mem::size_of::<NVFBC_TOCUDA_GRAB_FRAME_PARAMS>(), 2),
            dwFlags: flags,
            pCUDADeviceBuffer: &mut device_ptr as *mut CUdeviceptr as *mut c_void,
            pFrameGrabInfo: &mut info,
            dwTimeoutMs: timeout.as_millis().clamp(1, u32::MAX as u128) as u32,
        };
        let st = unsafe { (self.funcs.nvFBCToCudaGrabFrame.unwrap())(self.handle, &mut params) };
        if st != NVFBC_SUCCESS {
            return Err(self.fail("NvFBCToCudaGrabFrame", st));
        }
        Ok(GrabbedFrame {
            device_ptr,
            width: info.dwWidth,
            height: info.dwHeight,
            byte_size: info.dwByteSize,
            is_new: info.bIsNewFrame == NVFBC_TRUE,
            direct: info.bDirectCapture == NVFBC_TRUE,
            post_processed: info.bRequiredPostProcessing == NVFBC_TRUE,
        })
    }
}

impl Drop for NvfbcSession {
    fn drop(&mut self) {
        self.stop();
        let mut params = NVFBC_VERSION_ONLY_PARAMS {
            dwVersion: struct_ver(std::mem::size_of::<NVFBC_VERSION_ONLY_PARAMS>(), 1),
        };
        if let Some(f) = self.funcs.nvFBCDestroyHandle {
            unsafe { f(self.handle, &mut params) };
        }
    }
}

/// What a failed driver call leaves the capture able to do.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Recovery {
    /// The capture session is gone but the client handle is live: create the session again.
    Session,
    /// The X server the handle talks to is gone. The API has no way back from this other than a
    /// new handle, so the old one is destroyed (which may leak the X, GLX and GL resources it can
    /// no longer reach) and the capture rebuilt around a fresh one.
    Handle,
    /// Nothing this path can do; the caller stops.
    None,
}

fn recovery_for(err: &NvfbcError) -> Recovery {
    match err.status {
        NVFBC_ERR_MUST_RECREATE => Recovery::Session,
        NVFBC_ERR_X => Recovery::Handle,
        _ => Recovery::None,
    }
}

/// The capture region and delivered frame size for a set of settings against a framebuffer of
/// `screen`, in the terms the XShm path resolves them: the origin clamped into the screen, the
/// size following the root when auto-adjust is on and clamped to what is left of it otherwise,
/// and both dimensions even because every video codec needs them to be.
///
/// A zero-sized box means the whole tracked region, so a capture that covers the screen asks for
/// it that way and keeps following a resize.
fn resolve_region(screen: NVFBC_SIZE, s: &RustCaptureSettings) -> (NVFBC_BOX, NVFBC_SIZE) {
    let (w, h) = super::resolve_dims(screen.w.min(u16::MAX as u32) as u16, screen.h.min(u16::MAX as u32) as u16, s);
    let x = super::clamp_offset(s.capture_x, screen.w.min(u16::MAX as u32) as u16).max(0) as u32;
    let y = super::clamp_offset(s.capture_y, screen.h.min(u16::MAX as u32) as u16).max(0) as u32;
    let size = NVFBC_SIZE { w: w as u32, h: h as u32 };
    // An empty box is the whole tracked region, which is what keeps a whole-screen capture
    // following a resize. It may only stand in for a frame that *is* the screen: a frame size
    // differing from the captured region makes the driver scale into it, which is both a
    // conversion pass and the end of direct capture.
    let full = x == 0 && y == 0 && size.w == screen.w && size.h == screen.h;
    let region = if full {
        NVFBC_BOX::default()
    } else {
        NVFBC_BOX { x, y, w: size.w, h: size.h }
    };
    (region, size)
}

/// A running zero-copy capture: the NvFBC session and the NVENC session it feeds, on the thread
/// that owns both contexts.
struct GpuCapture {
    nvfbc: NvfbcSession,
    encoder: NvencEncoder,
    /// The live session: its geometry is what is actually being captured and encoded, and its
    /// rate and quality fields follow the cross-thread controls.
    settings: RustCaptureSettings,
    /// The geometry that was asked for, which a region change updates and every region is
    /// resolved from. Kept apart from `settings` because a capture that follows the framebuffer
    /// would otherwise take its own current size for the request and stop following.
    request: RustCaptureSettings,
    screen: NVFBC_SIZE,
}

/// Why the NvFBC path was not taken, for the one line that says so.
fn declined(reason: &str) -> Option<GpuCapture> {
    println!("[x11] GPU capture (NvFBC) unavailable: {reason}. Capturing through XShm.");
    None
}

/// Build the zero-copy capture for these settings, or report why it cannot serve them.
///
/// Everything fallible happens here, before a single frame is delivered, so the caller can fall
/// back to the XShm path with nothing half-built. The order matters: NVENC is opened first
/// because it owns the CUDA context, and NvFBC is then set up with that context current, which is
/// what makes the device pointer it hands back one NVENC can register in place.
///
/// The path is declined, rather than failed, when it cannot be zero-copy: a codec no NVENC engine
/// serves, software encoding requested, a device that is not NVIDIA, a watermark (which is
/// composited into host pixels and would mean reading the frame back), or a driver that offers no
/// NvFBC on this X server.
fn open(settings: &RustCaptureSettings) -> Option<GpuCapture> {
    if !settings.codec.is_video() {
        return declined("the codec is JPEG");
    }
    if settings.use_cpu || settings.encode_node_index == -1 {
        return declined("software encoding was requested");
    }
    if !settings.watermark_path.is_empty() {
        return declined("a watermark is composited into the frame");
    }
    let node = settings.encode_node_index.max(0);
    let driver = crate::get_gpu_driver(node);
    if !crate::driver_selects_nvenc(&driver) {
        return declined(&format!("the encode node's driver is {driver}"));
    }
    // The driver is asked before an encoder is built, so a host without NvFBC never pays for an
    // NVENC session the XShm path would immediately build again.
    let probed = (|| -> Result<(NvfbcSession, NVFBC_SIZE), String> {
        let nvfbc = NvfbcSession::open()?;
        let status = nvfbc.status().map_err(|e| e.to_string())?;
        if status.bIsCapturePossible != NVFBC_TRUE {
            return Err("the driver reports capture is not possible on this X server".into());
        }
        if status.bInModeset == NVFBC_TRUE {
            return Err("the X server is in a modeset".into());
        }
        Ok((nvfbc, status.screenSize))
    })();
    let (mut nvfbc, screen) = match probed {
        Ok(v) => v,
        Err(e) => return declined(&e),
    };

    let mut encoder = match NvencEncoder::new(settings, std::ptr::null()) {
        Ok(enc) => enc,
        Err(e) => return declined(&format!("NVENC {} did not open: {e}", settings.codec.display())),
    };
    // The capture session is created with the encoder's context current, so the device pointer it
    // hands back is one the encoder can register.
    if !encoder.push_context() {
        return declined("the encoder's CUDA context could not be made current");
    }
    let (region, size) = resolve_region(screen, settings);
    if let Err(e) = nvfbc.start(region, size, settings.capture_cursor) {
        encoder.pop_context();
        return declined(&e.to_string());
    }

    let request = settings.clone();
    let mut settings = settings.clone();
    settings.width = nvfbc.size.w as i32;
    settings.height = nvfbc.size.h as i32;
    if (settings.width != encoder.width() as i32 || settings.height != encoder.height() as i32)
        && let Err(e) = encoder.reconfigure_resolution(&settings)
    {
        nvfbc.stop();
        encoder.pop_context();
        return declined(&format!("NVENC could not follow the captured size: {e}"));
    }
    println!(
        "[x11] GPU capture (NvFBC) active: {}x{} composited into video memory and encoded in place.",
        settings.width, settings.height
    );
    Some(GpuCapture { nvfbc, encoder, settings, request, screen })
}

impl Drop for GpuCapture {
    /// Tear down in the one order that keeps the driver's bookkeeping honest: the encoder's
    /// registration of the capture buffer goes first, because the memory behind it belongs to the
    /// NvFBC session; then the capture session; then the CUDA context both were sharing.
    fn drop(&mut self) {
        self.encoder.release_external_input();
        self.nvfbc.stop();
        self.encoder.pop_context();
    }
}

/// Frames between framebuffer-size polls, matching the XShm path's geometry cadence.
const GEOMETRY_POLL_FRAMES: i32 = 30;

/// Run the zero-copy X11 capture until `controls.stop` is set, or report that it cannot start.
///
/// `None` means NvFBC cannot serve this session and nothing was delivered, which is the caller's
/// signal to run the XShm path instead. Anything else means the capture ran — every frame going
/// from the driver's video-memory buffer into NVENC without a copy — and carries how it ended.
///
/// Each iteration:
///
/// 1. **Paces** to the live target frame rate, which is re-read every tick, so a rate change
///    applies immediately and the loop never runs ahead of the session's contract.
/// 2. **Applies the cross-thread controls** on the thread that owns the sessions: a requested key
///    frame, a rate change, live tunables, a cursor toggle or a region change (the last two are
///    session parameters, so they restart the capture session in place).
/// 3. **Grabs** the next frame, which returns immediately when one is already waiting and
///    otherwise waits for one until the frame deadline.
/// 4. **Follows the geometry**: a framebuffer resize restarts the capture session at the new size
///    and reconfigures NVENC in place.
/// 5. **Encodes** through the same send / quality / key-frame policy every full-frame encoder
///    obeys, with the driver's own "this frame is new" report standing in for the damage signal,
///    and hands the bitstream to the recording sink and the delivery callback.
pub fn run_capture<F>(
    settings: RustCaptureSettings,
    controls: Arc<Controls>,
    encode_tid_tx: std::sync::mpsc::Sender<std::thread::ThreadId>,
    on_frame: &mut F,
) -> Option<Result<(), String>>
where
    F: FnMut(Vec<EncodedStripe>),
{
    let mut gpu = open(&settings)?;
    // Capture and encode are the same thread here, so the re-entrant-stop guard watches it as
    // both.
    let _ = encode_tid_tx.send(std::thread::current().id());
    controls.codec.store(gpu.encoder.codec().id(), Ordering::Relaxed);

    let recording_sink = RecordingSink::try_bind(&settings.recording_socket, settings.target_fps);
    let mut state = StripeState::default();
    let mut frame_counter: u16 = 0;
    let mut pending_force_idr = false;
    let mut next_frame = Instant::now();
    let mut geometry_check = GEOMETRY_POLL_FRAMES;
    let mut error_streak = 0u32;
    let mut last_log = Instant::now();
    let mut sent_frames: u64 = 0;
    let mut new_frames: u64 = 0;
    let mut direct_frames: u64 = 0;

    while !controls.stop.load(Ordering::Relaxed) {
        let fps = (controls.fps_milli.load(Ordering::Relaxed).max(1) as f64) / 1000.0;
        let frame_dur = Duration::from_secs_f64(1.0 / fps.max(1.0));
        let now = Instant::now();
        if now < next_frame {
            std::thread::sleep(next_frame - now);
        }
        next_frame += frame_dur;
        let now = Instant::now();
        if next_frame < now {
            next_frame = now;
        }
        if controls.stop.load(Ordering::Relaxed) {
            break;
        }

        if controls.force_idr.swap(false, Ordering::Relaxed)
            || recording_sink.as_ref().is_some_and(|s| s.should_force_idr())
        {
            pending_force_idr = true;
        }
        if controls.rate_dirty.swap(false, Ordering::Acquire) {
            gpu.settings.video_bitrate_kbps = controls.bitrate_kbps.load(Ordering::Relaxed);
            gpu.settings.video_vbv_multiplier =
                controls.vbv_mult_milli.load(Ordering::Relaxed) as f64 / 1000.0;
            gpu.settings.target_fps = fps;
            gpu.encoder.reconfigure_rate(&gpu.settings);
        }
        if controls.tunables_dirty.swap(false, Ordering::Acquire)
            && let Some(t) = controls.tunables.lock().unwrap().take()
        {
            t.apply_to(&mut gpu.settings);
        }

        let want_cursor = controls.capture_cursor.load(Ordering::Relaxed);
        let mut recheck_geometry = controls.region_dirty.swap(false, Ordering::Acquire);
        if recheck_geometry {
            let (nx, ny, nw, nh) = *controls.region.lock().unwrap();
            gpu.request.capture_x = nx;
            gpu.request.capture_y = ny;
            gpu.request.width = nw;
            gpu.request.height = nh;
            // An explicit live size pins the region; `<= 0` keeps following the framebuffer.
            gpu.request.auto_adjust_screen_capture_size = nw <= 0 || nh <= 0;
        }
        geometry_check -= 1;
        if geometry_check <= 0 {
            geometry_check = GEOMETRY_POLL_FRAMES;
            if let Ok(status) = gpu.nvfbc.status() {
                recheck_geometry |= status.screenSize != gpu.screen;
                gpu.screen = status.screenSize;
            }
        }
        if recheck_geometry || want_cursor != gpu.nvfbc.with_cursor {
            let (region, size) = resolve_region(gpu.screen, &gpu.request);
            if region != gpu.nvfbc.region || size != gpu.nvfbc.size || want_cursor != gpu.nvfbc.with_cursor
            {
                match restart_session(&mut gpu, region, size, want_cursor) {
                    Ok(()) => {
                        state = StripeState::default();
                        pending_force_idr = true;
                    }
                    Err(e) => {
                        return Some(Err(format!("NvFBC could not follow the new geometry: {e}")));
                    }
                }
            }
        }

        let frame = match gpu.nvfbc.grab(frame_dur) {
            Ok(f) => {
                error_streak = 0;
                f
            }
            Err(e) => {
                error_streak += 1;
                let recovery = recovery_for(&e);
                if recovery == Recovery::None || error_streak > 5 {
                    return Some(Err(format!("NvFBC capture ended: {e}")));
                }
                let (region, size) = resolve_region(gpu.screen, &gpu.request);
                let rebuilt = if recovery == Recovery::Handle {
                    eprintln!("[x11] NvFBC lost the X server ({e}); rebuilding the client handle.");
                    rebuild_handle(&mut gpu, region, size, want_cursor)
                } else {
                    eprintln!("[x11] NvFBC grab failed ({e}); rebuilding the capture session.");
                    restart_session(&mut gpu, region, size, want_cursor).map_err(|e| e.to_string())
                };
                if let Err(e) = rebuilt {
                    return Some(Err(format!("NvFBC capture could not be rebuilt: {e}")));
                }
                state = StripeState::default();
                pending_force_idr = true;
                continue;
            }
        };
        if frame.device_ptr == 0 || frame.width == 0 || frame.height == 0 {
            continue;
        }
        new_frames += frame.is_new as u64;
        direct_frames += frame.direct as u64;

        if frame.width != gpu.encoder.width() || frame.height != gpu.encoder.height() {
            gpu.settings.width = frame.width as i32;
            gpu.settings.height = frame.height as i32;
            if let Err(e) = gpu.encoder.reconfigure_resolution(&gpu.settings) {
                return Some(Err(format!("NVENC could not follow the captured size: {e}")));
            }
            state = StripeState::default();
            pending_force_idr = true;
        }

        let decision = decide_hw_fullframe(
            &mut state,
            &gpu.settings,
            frame_counter,
            !gpu.settings.video_streaming_mode && frame.is_new,
            false,
            pending_force_idr,
        );
        let mut delivered = false;
        if decision.send {
            let pitch = frame_pitch(frame.byte_size, frame.width, frame.height);
            match gpu.encoder.encode_cuda_pitch(
                frame.device_ptr,
                pitch,
                false,
                frame_counter as u64,
                decision.target_qp,
                decision.force_idr,
            ) {
                Ok(data) if !data.is_empty() => {
                    error_streak = 0;
                    let stripes = vec![EncodedStripe {
                        data: Arc::new(data),
                        codec: gpu.settings.codec,
                        stripe_y_start: 0,
                        stripe_height: gpu.settings.height,
                        frame_id: frame_counter as i32,
                    }];
                    if let Some(sink) = &recording_sink {
                        sink.write_frame(&stripes, gpu.settings.width, gpu.settings.height);
                    }
                    sent_frames += 1;
                    delivered = true;
                    on_frame(stripes);
                }
                Ok(_) => {}
                Err(e) => {
                    error_streak += 1;
                    if error_streak % crate::HW_ERROR_RECOVERY_THRESHOLD == 1 {
                        eprintln!("[x11] NVENC encode error on the zero-copy path: {e}");
                    }
                    if error_streak >= crate::HW_ERROR_RECOVERY_THRESHOLD {
                        return Some(Err("NVENC failed repeatedly on the zero-copy path".to_string()));
                    }
                }
            }
        }
        // An unserved request stays armed: on an infinite GOP a key frame lost to an encode error
        // or a skipped tick would never come back on its own.
        pending_force_idr = (pending_force_idr || decision.force_idr) && !delivered;
        frame_counter = frame_counter.wrapping_add(1);

        let elapsed = last_log.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            if gpu.settings.debug_logging {
                println!(
                    "[x11] NvFBC {}x{} Encoder: NVENC EncFPS: {:.2} NewFrames/s: {:.2} Direct: {}",
                    gpu.settings.width,
                    gpu.settings.height,
                    sent_frames as f64 / elapsed,
                    new_frames as f64 / elapsed,
                    direct_frames > 0
                );
            }
            sent_frames = 0;
            new_frames = 0;
            direct_frames = 0;
            last_log = Instant::now();
        }
    }
    Some(Ok(()))
}

/// Restart the capture session at a new region, size or cursor mode, and bring NVENC to the same
/// geometry. The device pointer the driver returns afterwards is a new one, which the encoder's
/// registration follows on the next frame.
fn restart_session(
    gpu: &mut GpuCapture,
    region: NVFBC_BOX,
    size: NVFBC_SIZE,
    with_cursor: bool,
) -> Result<(), NvfbcError> {
    // The registration addresses the buffer the outgoing session owns, so it goes before the
    // session does.
    gpu.encoder.release_external_input();
    gpu.nvfbc.start(region, size, with_cursor)?;
    gpu.settings.width = size.w as i32;
    gpu.settings.height = size.h as i32;
    if gpu.settings.width != gpu.encoder.width() as i32 || gpu.settings.height != gpu.encoder.height() as i32
    {
        gpu.encoder
            .reconfigure_resolution(&gpu.settings)
            .map_err(|e| NvfbcError { status: NVFBC_SUCCESS, message: e })?;
    }
    Ok(())
}

/// Replace the client handle and the capture session on it, for the one failure the API offers no
/// other way back from: the X server the handle was opened on is gone.
///
/// The encoder, its CUDA context and the frames already delivered all survive; only the driver
/// objects are rebuilt, in the order that keeps the encoder from holding a registration of memory
/// the outgoing session owns.
fn rebuild_handle(
    gpu: &mut GpuCapture,
    region: NVFBC_BOX,
    size: NVFBC_SIZE,
    with_cursor: bool,
) -> Result<(), String> {
    gpu.encoder.release_external_input();
    gpu.nvfbc.stop();
    gpu.nvfbc = NvfbcSession::open()?;
    gpu.screen = gpu.nvfbc.status().map_err(|e| e.to_string())?.screenSize;
    let (region, size) = if region == NVFBC_BOX::default() {
        resolve_region(gpu.screen, &gpu.request)
    } else {
        (region, size)
    };
    restart_session(gpu, region, size, with_cursor).map_err(|e| e.to_string())
}

/// The row stride of a captured frame: the driver's own, derived from the buffer size it
/// reports, or the packed width when that size is not a whole number of rows.
fn frame_pitch(byte_size: u32, width: u32, height: u32) -> usize {
    let packed = width as usize * 4;
    if height == 0 {
        return packed;
    }
    let rows = byte_size as usize / height as usize;
    if rows >= packed && rows.is_multiple_of(4) && rows * height as usize == byte_size as usize {
        rows
    } else {
        packed
    }
}

#[cfg(test)]
mod abi_tests {
    use super::*;

    /// Every parameter struct carries its own `size_of` inside the version word the driver
    /// validates, so a layout that drifts from the SDK's is rejected at the first call with no
    /// other symptom. These are the sizes `NvFBC.h` (NvFBC 1.8) defines.
    #[test]
    fn struct_layouts_match_the_sdk() {
        assert_eq!(std::mem::size_of::<NVFBC_BOX>(), 16);
        assert_eq!(std::mem::size_of::<NVFBC_SIZE>(), 8);
        assert_eq!(std::mem::size_of::<NVFBC_FRAME_GRAB_INFO>(), 48);
        assert_eq!(std::mem::size_of::<NVFBC_CREATE_HANDLE_PARAMS>(), 40);
        assert_eq!(std::mem::size_of::<NVFBC_VERSION_ONLY_PARAMS>(), 4);
        assert_eq!(std::mem::size_of::<NVFBC_RANDR_OUTPUT_INFO>(), 148);
        assert_eq!(std::mem::size_of::<NVFBC_GET_STATUS_PARAMS>(), 780);
        assert_eq!(std::mem::size_of::<NVFBC_CREATE_CAPTURE_SESSION_PARAMS>(), 64);
        assert_eq!(std::mem::size_of::<NVFBC_TOCUDA_SETUP_PARAMS>(), 8);
        assert_eq!(std::mem::size_of::<NVFBC_TOCUDA_GRAB_FRAME_PARAMS>(), 32);
        assert_eq!(std::mem::size_of::<NVFBC_API_FUNCTION_LIST>(), 176);
    }

    /// The version words the driver compares against are the SDK's `NVFBC_*_VER` constants: the
    /// struct's size, its revision at bit 16, and the API version's low byte at bit 24.
    #[test]
    fn version_words_match_the_sdk() {
        assert_eq!(NVFBC_VERSION, 0x108);
        assert_eq!(struct_ver(40, 2), 0x0802_0028);
        assert_eq!(struct_ver(4, 1), 0x0801_0004);
        assert_eq!(struct_ver(780, 2), 0x0802_030C);
        assert_eq!(struct_ver(64, 6), 0x0806_0040);
        assert_eq!(struct_ver(8, 1), 0x0801_0008);
        assert_eq!(struct_ver(32, 2), 0x0802_0020);
    }
}

#[cfg(test)]
mod region_tests {
    use super::*;
    use crate::encoders::codec::Codec;

    fn settings(w: i32, h: i32) -> RustCaptureSettings {
        RustCaptureSettings { width: w, height: h, codec: Codec::H264, ..Default::default() }
    }

    /// A capture covering the whole framebuffer asks for the whole tracked region, so the session
    /// keeps following it; anything smaller or offset is a box.
    #[test]
    fn whole_screen_is_an_empty_box() {
        let screen = NVFBC_SIZE { w: 1920, h: 1080 };
        let mut s = settings(1920, 1080);
        let (region, size) = resolve_region(screen, &s);
        assert_eq!(region, NVFBC_BOX::default());
        assert_eq!(size, NVFBC_SIZE { w: 1920, h: 1080 });

        s.auto_adjust_screen_capture_size = true;
        let (region, size) = resolve_region(screen, &s);
        assert_eq!(region, NVFBC_BOX::default());
        assert_eq!(size, NVFBC_SIZE { w: 1920, h: 1080 });

        let mut region_only = settings(1280, 720);
        region_only.capture_x = 100;
        region_only.capture_y = 50;
        let (region, size) = resolve_region(screen, &region_only);
        assert_eq!(region, NVFBC_BOX { x: 100, y: 50, w: 1280, h: 720 });
        assert_eq!(size, NVFBC_SIZE { w: 1280, h: 720 });
    }

    /// Both dimensions are even, because every video codec requires it, and a region is clamped
    /// into the framebuffer rather than running past its edge.
    ///
    /// An odd screen therefore captures an even box of it, never the whole screen delivered at
    /// the even size: the driver would scale into the difference, which is a conversion pass and
    /// the end of direct capture.
    #[test]
    fn region_is_even_and_inside_the_screen() {
        let screen = NVFBC_SIZE { w: 1919, h: 1081 };
        let (region, size) = resolve_region(screen, &settings(0, 0));
        assert_eq!(size, NVFBC_SIZE { w: 1918, h: 1080 });
        assert_eq!(region, NVFBC_BOX { x: 0, y: 0, w: 1918, h: 1080 });

        let mut s = settings(1920, 1080);
        s.capture_x = 1000;
        s.capture_y = 800;
        let (region, size) = resolve_region(NVFBC_SIZE { w: 1920, h: 1080 }, &s);
        assert_eq!(region.x, 1000);
        assert_eq!(region.y, 800);
        assert_eq!(size, NVFBC_SIZE { w: 920, h: 280 });
        assert_eq!(region.w, size.w);
        assert_eq!(region.h, size.h);
    }

    /// What the capture does after a failed driver call follows from the status alone: a session
    /// the driver invalidated is created again, a lost X server needs a whole new handle, and
    /// nothing else is recoverable. The status is read from the error rather than matched in its
    /// text, where the driver's own message could carry the same digits.
    #[test]
    fn recovery_follows_the_status() {
        let session = |status| NvfbcError { status, message: String::new() };
        assert!(matches!(recovery_for(&session(NVFBC_ERR_MUST_RECREATE)), Recovery::Session));
        assert!(matches!(recovery_for(&session(NVFBC_ERR_X)), Recovery::Handle));
        assert!(matches!(recovery_for(&session(1)), Recovery::None));
        let misleading = NvfbcError { status: 1, message: "the display (16) went away (10)".into() };
        assert!(matches!(recovery_for(&misleading), Recovery::None));
    }

    /// The stride comes from the driver's own byte count when that is a whole number of rows,
    /// and falls back to the packed width when it is not.
    #[test]
    fn pitch_follows_the_reported_byte_size() {
        assert_eq!(frame_pitch(1920 * 1080 * 4, 1920, 1080), 1920 * 4);
        assert_eq!(frame_pitch(2048 * 4 * 1080, 1920, 1080), 2048 * 4);
        assert_eq!(frame_pitch(12345, 1920, 1080), 1920 * 4);
        assert_eq!(frame_pitch(0, 1920, 0), 1920 * 4);
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::encoders::codec::{parse_video_type, FRAME_DELTA, FRAME_KEY, VIDEO_HEADER_LEN};
    use crate::webcam::decode::{AvDecoder, Codec as DecCodec, Decoder};

    /// Test helper: full-frame capture settings for `codec` at CRF 25.
    fn settings(codec: crate::encoders::codec::Codec) -> RustCaptureSettings {
        RustCaptureSettings {
            codec,
            target_fps: 60.0,
            video_crf: 25,
            video_streaming_mode: true,
            ..Default::default()
        }
    }

    /// Test helper: paint the whole root of `$DISPLAY` one solid colour and let the server
    /// finish, so the next capture has a known picture in it.
    ///
    /// The screen saver is turned off first: a test display sees no input, so a server left with
    /// the default ten-minute blanking timeout hands the capture a black screen and every colour
    /// comparison below fails for a reason that has nothing to do with the capture.
    fn paint_root(rgb: (u8, u8, u8)) -> bool {
        let _ = std::process::Command::new("xset").args(["s", "off", "s", "noblank"]).output();
        let _ = std::process::Command::new("xset").arg("s").arg("reset").output();
        let spec = format!("#{:02x}{:02x}{:02x}", rgb.0, rgb.1, rgb.2);
        let out = std::process::Command::new("xsetroot").args(["-solid", &spec]).output();
        if !out.map(|o| o.status.success()).unwrap_or(false) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(120));
        true
    }

    /// BT.601 limited-range Y/Cb/Cr of an 8-bit RGB triple — what NVENC's hardware CSC emits for
    /// the packed input this path hands it.
    fn ycbcr_601(rgb: (u8, u8, u8)) -> [f64; 3] {
        let (r, g, b) = (rgb.0 as f64 / 255.0, rgb.1 as f64 / 255.0, rgb.2 as f64 / 255.0);
        [
            16.0 + 219.0 * (0.299 * r + 0.587 * g + 0.114 * b),
            128.0 + 224.0 * (-0.168736 * r - 0.331264 * g + 0.5 * b),
            128.0 + 224.0 * (0.5 * r - 0.418688 * g - 0.081312 * b),
        ]
    }

    /// Test helper: mean Y/Cb/Cr of a decoded picture.
    fn decoded_mean(dec: &mut AvDecoder, payload: &[u8]) -> [f64; 3] {
        assert!(dec.decode(payload).expect("decode"), "no picture from this access unit");
        let v = dec.frame().expect("decoded frame");
        let mut acc = [0f64; 3];
        let mut n = 0f64;
        for y in 0..v.height {
            for x in 0..v.width {
                acc[0] += v.y[y * v.y_stride + x] as f64;
                acc[1] += v.u[(y / 2) * v.uv_stride + x / 2] as f64;
                acc[2] += v.v[(y / 2) * v.uv_stride + x / 2] as f64;
                n += 1.0;
            }
        }
        [acc[0] / n, acc[1] / n, acc[2] / n]
    }

    /// Thread CPU time, for the per-frame CPU cost of the path independent of GPU waiting.
    fn thread_cpu() -> Duration {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
    }

    /// On a real NVIDIA X server: the driver reports whether it can capture this screen, and at
    /// what size. Ignored by default; run with `DISPLAY` pointing at an NVIDIA X server:
    /// `cargo test gpu_nvfbc -- --ignored --nocapture --test-threads=1`.
    #[test]
    #[ignore]
    fn gpu_nvfbc_status_reports_capture() {
        let Ok(nvfbc) = NvfbcSession::open() else {
            println!("NvFBC handed out no capture handle on this host; nothing to report");
            return;
        };
        let status = nvfbc.status().expect("NvFBC status");
        println!(
            "capture possible={} can create now={} screen={}x{} outputs={} nvfbc version={} in modeset={}",
            status.bIsCapturePossible,
            status.bCanCreateNow,
            status.screenSize.w,
            status.screenSize.h,
            status.dwOutputNum,
            status.dwNvFBCVersion,
            status.bInModeset
        );
        assert_eq!(status.bIsCapturePossible, NVFBC_TRUE, "this GPU offers no NvFBC");
        assert!(status.screenSize.w > 0 && status.screenSize.h > 0);
    }

    /// Whether two captures of the same X screen can hold NvFBC sessions at once, which is what
    /// a second Selkies display on X11 needs to be zero-copy as well. A driver that allows only
    /// one leaves the second capture on the XShm path, which is a correct outcome but a slower
    /// one, so the answer is worth knowing rather than assuming. Ignored by default.
    #[test]
    #[ignore]
    fn gpu_nvfbc_two_sessions_on_one_screen() {
        let Some(mut first) = open(&settings(crate::encoders::codec::Codec::H264)) else {
            println!("the NvFBC path declined the first capture on this host; nothing to compare");
            return;
        };
        let second = open(&settings(crate::encoders::codec::Codec::H264));
        println!(
            "a second concurrent NvFBC capture of the same screen: {}",
            if second.is_some() { "opens" } else { "is declined, and falls back to XShm" }
        );
        // The first must keep working whatever the second did.
        let f = first.nvfbc.grab(Duration::from_millis(200)).expect("first still captures");
        let pitch = frame_pitch(f.byte_size, f.width, f.height);
        first
            .encoder
            .encode_cuda_pitch(f.device_ptr, pitch, false, 0, 25, true)
            .expect("first still encodes");
        drop(second);
        let f = first.nvfbc.grab(Duration::from_millis(200)).expect("first survives the second");
        assert!(f.width > 0);
    }

    /// The whole zero-copy path against a real X server and GPU: the driver composites the root
    /// into video memory, NVENC encodes that buffer in place, and the decoded picture is the
    /// colour that was painted.
    ///
    /// The zero-copy property is asserted rather than assumed, three ways: the driver reports it
    /// needed no post-processing pass (the native BGRA request means no conversion copy), every
    /// grab hands back the *same* device pointer (so the encoder's registration is made once and
    /// reused instead of per frame), and repainting the root changes the decoded colour, which it
    /// could not if the encoder were reading a stale copy. Ignored by default.
    #[test]
    #[ignore]
    fn gpu_nvfbc_zero_copy_encodes_the_painted_root() {
        const FIRST: (u8, u8, u8) = (0x20, 0x40, 0xc0);
        const SECOND: (u8, u8, u8) = (0xd0, 0x50, 0x18);
        if !paint_root(FIRST) {
            println!("no X root this host can paint (xsetroot, $DISPLAY); nothing to capture");
            return;
        }
        let Some(mut gpu) = open(&settings(crate::encoders::codec::Codec::H264)) else {
            println!("the NvFBC path declined this session on this host; nothing to capture");
            return;
        };
        let mut dec = AvDecoder::new(DecCodec::H264).expect("avcodec h264");
        let mut pointers = Vec::new();

        let encode = |gpu: &mut GpuCapture, i: u64, key: bool, pointers: &mut Vec<CUdeviceptr>| -> Vec<u8> {
            // A repaint is one frame; ask until the driver has produced it.
            let mut frame = gpu.nvfbc.grab(Duration::from_millis(200)).expect("grab");
            for _ in 0..30 {
                if frame.is_new {
                    break;
                }
                frame = gpu.nvfbc.grab(Duration::from_millis(200)).expect("grab");
            }
            assert_eq!(frame.width, gpu.encoder.width());
            assert_eq!(frame.height, gpu.encoder.height());
            assert!(
                !frame.post_processed,
                "the native format needed a conversion pass, which is an extra frame copy"
            );
            pointers.push(frame.device_ptr);
            let pitch = frame_pitch(frame.byte_size, frame.width, frame.height);
            gpu.encoder
                .encode_cuda_pitch(frame.device_ptr, pitch, false, i, 25, key)
                .expect("encode in place")
        };

        let pkt = encode(&mut gpu, 0, true, &mut pointers);
        assert_eq!(
            parse_video_type(pkt[1]),
            Some((crate::encoders::codec::Codec::H264, FRAME_KEY))
        );
        let mean = decoded_mean(&mut dec, &pkt[VIDEO_HEADER_LEN..]);
        let want = ycbcr_601(FIRST);
        for i in 0..3 {
            assert!(
                (mean[i] - want[i]).abs() <= 8.0,
                "plane {i}: captured {:.1}, painted {:.1}",
                mean[i],
                want[i]
            );
        }

        for i in 1..4u64 {
            let pkt = encode(&mut gpu, i, false, &mut pointers);
            assert_eq!(
                parse_video_type(pkt[1]),
                Some((crate::encoders::codec::Codec::H264, FRAME_DELTA))
            );
            let _ = decoded_mean(&mut dec, &pkt[VIDEO_HEADER_LEN..]);
        }

        assert!(paint_root(SECOND), "the root was painted once, so a repaint must work");
        let pkt = encode(&mut gpu, 4, false, &mut pointers);
        let mean = decoded_mean(&mut dec, &pkt[VIDEO_HEADER_LEN..]);
        let want = ycbcr_601(SECOND);
        for i in 0..3 {
            assert!(
                (mean[i] - want[i]).abs() <= 8.0,
                "after repaint, plane {i}: captured {:.1}, painted {:.1}",
                mean[i],
                want[i]
            );
        }

        assert!(
            pointers.windows(2).all(|w| w[0] == w[1]),
            "the capture buffer moved between frames, so the registration cannot be reused: {pointers:?}"
        );
        println!(
            "captured {}x{} into one device pointer {:#x} across {} frames, no post-processing pass",
            gpu.encoder.width(),
            gpu.encoder.height(),
            pointers[0],
            pointers.len()
        );
    }

    /// What the zero-copy path costs on a real GPU, per codec the device serves.
    ///
    /// Two numbers, because they answer different questions. The steady-state pass grabs without
    /// waiting, so it measures what capture and encode themselves cost with nothing else in the
    /// way — the ceiling the path imposes on frame rate. The moving-screen pass repaints the root
    /// between frames and times only the grab and encode, so the encoder is doing real work on
    /// changed content. In both, the host-CPU figure is the one that matters: nothing on the host
    /// touches a pixel, so it stays a small fraction of the wall time. Ignored by default.
    #[test]
    #[ignore]
    fn gpu_nvfbc_throughput() {
        let n: usize = std::env::var("NVFBC_BENCH_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(240);
        for codec in [
            crate::encoders::codec::Codec::H264,
            crate::encoders::codec::Codec::H265,
            crate::encoders::codec::Codec::Av1,
        ] {
            let mut s = settings(codec);
            s.auto_adjust_screen_capture_size = true;
            let Some(mut gpu) = open(&s) else {
                println!("{}: declined", codec.display());
                continue;
            };
            let (w, h) = (gpu.encoder.width(), gpu.encoder.height());
            let run = |label: &str, repaint: bool, gpu: &mut GpuCapture| {
                let f = gpu.nvfbc.grab(Duration::from_millis(200)).expect("grab");
                let pitch = frame_pitch(f.byte_size, f.width, f.height);
                gpu.encoder
                    .encode_cuda_pitch(f.device_ptr, pitch, false, 0, 25, true)
                    .expect("warm-up");
                let (mut wall, mut cpu, mut bytes, mut news) = (Duration::ZERO, Duration::ZERO, 0usize, 0usize);
                for i in 0..n {
                    if repaint {
                        let v = (i * 9) as u8;
                        let _ = std::process::Command::new("xsetroot")
                            .args(["-solid", &format!("#{:02x}{:02x}{:02x}", v, 0x40, 0x80)])
                            .output();
                    }
                    let t = Instant::now();
                    let c = thread_cpu();
                    let f = gpu
                        .nvfbc
                        .grab_with(NVFBC_TOCUDA_GRAB_FLAGS_NOWAIT, Duration::from_millis(50))
                        .expect("grab");
                    let pitch = frame_pitch(f.byte_size, f.width, f.height);
                    let pkt = gpu
                        .encoder
                        .encode_cuda_pitch(f.device_ptr, pitch, false, 1 + i as u64, 25, false)
                        .expect("encode");
                    wall += t.elapsed();
                    cpu += thread_cpu() - c;
                    bytes += pkt.len();
                    news += f.is_new as usize;
                }
                println!(
                    "{} {w}x{h} {label}: {:.0} fps ({:.2} ms/frame wall, {:.2} ms/frame host CPU), {news}/{n} new, {:.1} KiB/frame",
                    codec.display(),
                    n as f64 / wall.as_secs_f64(),
                    wall.as_secs_f64() * 1000.0 / n as f64,
                    cpu.as_secs_f64() * 1000.0 / n as f64,
                    bytes as f64 / n as f64 / 1024.0
                );
            };
            run("steady state", false, &mut gpu);
            run("moving screen", true, &mut gpu);
            shm_upload_baseline(codec, w, h, n);
        }
    }

    /// The same screen, same codec and same GPU through the general XShm path, for the comparison
    /// the zero-copy path exists to win: the X server blits the root into a shared-memory surface
    /// and the encoder uploads that surface across PCIe, where NvFBC hands the encoder a buffer
    /// the driver already composited in video memory.
    fn shm_upload_baseline(codec: crate::encoders::codec::Codec, w: u32, h: u32, n: usize) {
        use x11rb::connection::Connection;
        use x11rb::protocol::shm::ConnectionExt as ShmExt;
        let Ok((conn, screen)) = x11rb::connect(None) else { return };
        if conn.shm_query_version().ok().and_then(|c| c.reply().ok()).is_none() {
            return;
        }
        let root = conn.setup().roots[screen].root;
        let Ok(mut surface) = crate::x11::ShmSurface::create(&conn, w as u16, h as u16) else { return };
        let mut s = settings(codec);
        s.width = w as i32;
        s.height = h as i32;
        let mut enc = match NvencEncoder::new(&s, std::ptr::null()) {
            Ok(e) => e,
            Err(_) => {
                surface.destroy(&conn);
                return;
            }
        };
        let stride = w as usize * 4;
        let (mut wall, mut cpu) = (Duration::ZERO, Duration::ZERO);
        for i in 0..n {
            let t = Instant::now();
            let c = thread_cpu();
            if crate::x11::grab_frame(&conn, root, &surface, 0, 0, w as u16, h as u16).is_err() {
                break;
            }
            let _ = enc
                .encode_cpu_packed(surface.as_slice(), stride, false, i as u64, 25, i == 0)
                .expect("encode uploaded frame");
            wall += t.elapsed();
            cpu += thread_cpu() - c;
        }
        surface.destroy(&conn);
        println!(
            "{} {w}x{h} XShm grab + upload: {:.0} fps ({:.2} ms/frame wall, {:.2} ms/frame host CPU)",
            codec.display(),
            n as f64 / wall.as_secs_f64(),
            wall.as_secs_f64() * 1000.0 / n as f64,
            cpu.as_secs_f64() * 1000.0 / n as f64
        );
    }
}
