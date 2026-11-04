/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Windows desktop capture: Windows.Graphics.Capture hands each frame of the primary monitor
//! to the encoder as a D3D11 texture, and NVENC encodes that texture in place on the device
//! the capture runs on, so the encoded bitstream is the only thing that reaches the CPU.
//!
//! The frames come from a free-threaded capture frame pool, which needs no dispatcher and
//! hands out one [`ID3D11Texture2D`] per frame it has buffered. The pool cycles a few textures,
//! so the encoder registers each one with NVENC the first time it sees it and keeps the
//! registration for the session, cached by the texture's interface pointer; the frame is held
//! for the encode and released after, never copied. Windows.Graphics.Capture delivers a frame
//! only when the screen changed, so a still screen delivers nothing at all — the loop polls
//! briefly and waits — and it delivers the first frame the moment `StartCapture` runs, so a
//! fresh capture always opens with a frame to encode as an IDR. A pool that has no frame
//! reports that through a null result that reaches the projection as `HRESULT(0)`, which is
//! how an empty pool is told apart from a failed one.
//!
//! The encoder is an H.264 NVENC session bound to the capture's D3D11 device
//! (`NV_ENC_DEVICE_TYPE_DIRECTX`), tuned for end-to-end latency: preset P4 under the
//! ultra-low-latency tuning, no B-frames (`frameIntervalP` 1), no lookahead, no reorder
//! delay, and `repeatSPSPPS` so every IDR carries its parameter sets and a decoder that joins
//! the stream at any frame decodes the first keyframe it receives without waiting for one
//! that carries them. The rate is CBR at the settings' bitrate; the GOP is the settings'
//! keyframe interval, or infinite (IDRs only on demand, through [`Controls::force_idr`])
//! where the settings name none.
//!
//! Everything COM and NVENC this module touches is created, used, and released on the one
//! capture thread [`run_capture`] runs on; the handles it returns to the rest of the crate
//! are plain atomics, so nothing unsafe crosses a thread boundary.

// The NVENC entry points are called through function pointers resolved at runtime, so the
// safety contract is carried by the function signatures rather than by a block around each
// call.
#![allow(unsafe_op_in_unsafe_fn)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use libloading::Library;
use nvcodec_sys::*;

use windows::Graphics::Capture::{
    Direct3D11CaptureFrame, Direct3D11CaptureFramePool, GraphicsCaptureItem,
};
use windows::Graphics::DirectX::Direct3D11::{IDirect3DDevice, IDirect3DSurface};
use windows::Graphics::DirectX::DirectXPixelFormat;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE, D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_UNKNOWN,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_FLAG, D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
    D3D11_SDK_VERSION, D3D11CreateDevice, ID3D11Device, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_DESC1, IDXGIAdapter, IDXGIAdapter1, IDXGIFactory1,
};
use windows::Win32::Graphics::Gdi::{MONITOR_DEFAULTTOPRIMARY, MonitorFromWindow};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize};
use windows::Win32::System::WinRT::Direct3D11::{
    CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess,
};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
use windows::core::{IInspectable, Interface, factory};

use crate::CaptureSettings;
use crate::encoders::codec::{Codec, FRAME_DELTA, FRAME_INTRA, FRAME_KEY, push_video_header};
use crate::encoders::frame_rate::FrameRate;
use crate::encoders::reference::Reference;
use crate::encoders::software::{EncodedStripe, FrameTiming};

/// How long the capture loop sleeps when the frame pool holds no frame: the screen has not
/// changed, so nothing is due but the next poll.
const IDLE_POLL: Duration = Duration::from_millis(1);

/// The bitrate a capture serves (kbps) where the settings name none.
const DEFAULT_BITRATE_KBPS: i32 = 8000;

/// The CBR target the settings name, as bits per second.
fn bitrate_bps(settings: &CaptureSettings) -> u32 {
    (if settings.video_bitrate_kbps > 0 {
        settings.video_bitrate_kbps
    } else {
        DEFAULT_BITRATE_KBPS
    }) as u32
        * 1000
}

/// The GOP length in frames the settings name, over the rate the encoder runs at: the
/// keyframe interval in seconds, or infinite where the settings keep the GOP open.
fn gop_frames(settings: &CaptureSettings) -> u32 {
    if settings.keyframe_interval_s > 0.0 {
        (settings.keyframe_interval_s * FrameRate::of(settings.target_fps).fps())
            .round()
            .max(1.0) as u32
    } else {
        NVENC_INFINITE_GOPLENGTH
    }
}

/// Cross-thread controls for a running capture: the atomics the owning `ScreenCapture` sets
/// from its own thread and the capture loop reads at the top of each iteration. It exists so
/// nothing unsafe crosses the thread boundary; the COM and NVENC state the loop owns stays
/// on the capture thread.
///
/// 1. **Lifecycle**: `stop` ends the capture loop, which tears the NVENC session down on its
///    way out; `force_idr` requests an on-demand keyframe on the next encoded frame.
pub struct Controls {
    pub stop: AtomicBool,
    pub force_idr: AtomicBool,
}

impl Controls {
    pub fn new() -> Self {
        Self {
            stop: AtomicBool::new(false),
            force_idr: AtomicBool::new(false),
        }
    }
}

/// COM as multithreaded for the thread, released when the guard drops. `S_FALSE` (the thread
/// already had an apartment) balances its own `CoUninitialize`; a mode conflict leaves the
/// caller's apartment alone. Every activation and interface call the capture makes — the
/// capture item, the frame pool, the session — is a COM call on this thread.
struct ComApartment {
    uninit: bool,
}

impl ComApartment {
    /// Bring the apartment up for this thread, or note that it could not.
    fn init() -> Self {
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        Self { uninit: hr.is_ok() }
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        if self.uninit {
            unsafe { CoUninitialize() };
        }
    }
}

/// One NVENC H.264 session on the capture's D3D11 device.
///
/// The session registers each frame-pool texture the first time it carries a frame, keyed by
/// its interface pointer, and keeps the registration (and a reference on the texture) for
/// its life: the pool cycles a few textures, so the map stays small and no texture is
/// registered twice. One bitstream buffer serves the strictly serial encode of this thread.
/// `_lib` keeps the driver library resident for the function pointers' life. Everything the
/// struct holds is created and released on the capture thread.
struct NvencDirectX {
    _lib: Library,
    funcs: NV_ENCODE_API_FUNCTION_LIST,
    session: *mut c_void,
    bitstream: NV_ENC_OUTPUT_PTR,
    width: u32,
    height: u32,
    /// Each pool texture NVENC has registered: its interface pointer, the texture (held so
    /// the registration cannot outlive it), and the registered handle.
    registered: HashMap<*mut c_void, (ID3D11Texture2D, NV_ENC_REGISTERED_PTR)>,
}

impl NvencDirectX {
    /// Open the session on `device` for `width` x `height` H.264, at the settings' rate and
    /// GOP, and create its single bitstream buffer.
    ///
    /// The latency posture comes from the preset and tuning (P4, ultra-low-latency) plus the
    /// explicit no-B-frames, no-lookahead, no-reorder settings; `repeatSPSPPS` makes every
    /// IDR self-contained so a decoder joining at any frame decodes the first keyframe it
    /// receives; CBR runs at the settings' bitrate with the shared VBV policy
    /// ([`crate::encoders::vbv_bits`]).
    unsafe fn open(
        lib: Library,
        funcs: NV_ENCODE_API_FUNCTION_LIST,
        device: &ID3D11Device,
        width: u32,
        height: u32,
        settings: &CaptureSettings,
    ) -> Result<Self, String> {
        let mut session_params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
            version: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
            deviceType: NV_ENC_DEVICE_TYPE::NV_ENC_DEVICE_TYPE_DIRECTX,
            device: device.as_raw(),
            apiVersion: NVENCAPI_VERSION,
            ..Default::default()
        };
        let mut session: *mut c_void = ptr::null_mut();
        let status = (funcs.nvEncOpenEncodeSessionEx.unwrap())(&mut session_params, &mut session);
        if status != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("nvEncOpenEncodeSessionEx: {status:?}"));
        }

        let kbps = bitrate_bps(settings) / 1000;
        let vbv = crate::encoders::vbv_bits(
            kbps * 1000,
            settings.target_fps,
            settings.keyframe_interval_s,
            settings.video_vbv_multiplier,
        );
        let rate = FrameRate::of(settings.target_fps);
        let gop = gop_frames(settings);

        let mut preset = NV_ENC_PRESET_CONFIG {
            version: NV_ENC_PRESET_CONFIG_VER,
            presetCfg: NV_ENC_CONFIG {
                version: NV_ENC_CONFIG_VER,
                ..Default::default()
            },
            ..Default::default()
        };
        let status = (funcs.nvEncGetEncodePresetConfigEx.unwrap())(
            session,
            NV_ENC_CODEC_H264_GUID,
            NV_ENC_PRESET_P4_GUID,
            NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
            &mut preset,
        );
        if status != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("nvEncGetEncodePresetConfigEx: {status:?}"));
        }
        let mut config = preset.presetCfg;
        config.frameIntervalP = 1;
        config.gopLength = gop;
        config.encodeCodecConfig.h264Config.set_repeatSPSPPS(1);
        config.rcParams.version = NV_ENC_RC_PARAMS_VER;
        config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR;
        config.rcParams.averageBitRate = kbps * 1000;
        config.rcParams.maxBitRate = kbps * 1000;
        config.rcParams.vbvBufferSize = vbv;
        config.rcParams.vbvInitialDelay = vbv;
        config.rcParams.set_zeroReorderDelay(1);
        config.rcParams.set_strictGOPTarget(1);
        config.rcParams.set_enableLookahead(0);
        config.rcParams.lookaheadDepth = 0;

        let mut init_params = NV_ENC_INITIALIZE_PARAMS {
            version: NV_ENC_INITIALIZE_PARAMS_VER,
            encodeGUID: NV_ENC_CODEC_H264_GUID,
            presetGUID: NV_ENC_PRESET_P4_GUID,
            tuningInfo: NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
            encodeWidth: width,
            encodeHeight: height,
            darWidth: width,
            darHeight: height,
            frameRateNum: rate.num,
            frameRateDen: rate.den,
            enablePTD: 1,
            encodeConfig: &mut config,
            maxEncodeWidth: width,
            maxEncodeHeight: height,
            ..Default::default()
        };
        let status = (funcs.nvEncInitializeEncoder.unwrap())(session, &mut init_params);
        if status != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("nvEncInitializeEncoder: {status:?}"));
        }

        let mut bitstream = NV_ENC_CREATE_BITSTREAM_BUFFER {
            version: NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
            ..Default::default()
        };
        let status = (funcs.nvEncCreateBitstreamBuffer.unwrap())(session, &mut bitstream);
        if status != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("nvEncCreateBitstreamBuffer: {status:?}"));
        }
        Ok(Self {
            _lib: lib,
            funcs,
            session,
            bitstream: bitstream.bitstreamBuffer,
            width,
            height,
            registered: HashMap::new(),
        })
    }

    /// Encode one frame out of `texture`, registering it with NVENC the first time it
    /// appears, and return the bitstream bytes with the picture type the encoder says it
    /// produced. The texture is mapped for the encode and unmapped after, on every path.
    unsafe fn encode(
        &mut self,
        texture: &ID3D11Texture2D,
        force_idr: bool,
    ) -> Result<(Vec<u8>, NV_ENC_PIC_TYPE), String> {
        let key = texture.as_raw();
        let registered = match self.registered.get(&key) {
            Some((_, registered)) => *registered,
            None => {
                let mut reg = NV_ENC_REGISTER_RESOURCE {
                    version: NV_ENC_REGISTER_RESOURCE_VER,
                    resourceType: NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX,
                    width: self.width,
                    height: self.height,
                    resourceToRegister: key,
                    bufferFormat: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
                    bufferUsage: NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE,
                    ..Default::default()
                };
                let status = (self.funcs.nvEncRegisterResource.unwrap())(self.session, &mut reg);
                if status != NVENCSTATUS::NV_ENC_SUCCESS {
                    return Err(format!("nvEncRegisterResource: {status:?}"));
                }
                let registered = reg.registeredResource;
                self.registered.insert(key, (texture.clone(), registered));
                crate::log::debug!(
                    "[Windows] registered a capture texture with NVENC ({} seen)",
                    self.registered.len()
                );
                registered
            }
        };
        let mut map = NV_ENC_MAP_INPUT_RESOURCE {
            version: NV_ENC_MAP_INPUT_RESOURCE_VER,
            registeredResource: registered,
            ..Default::default()
        };
        let status = (self.funcs.nvEncMapInputResource.unwrap())(self.session, &mut map);
        if status != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("nvEncMapInputResource: {status:?}"));
        }
        let input = map.mappedResource;
        let encoded = self.encode_mapped(input, force_idr);
        // The mapping is released whether the encode worked or not, so no registration is
        // left mapped over the session's teardown.
        let _ = (self.funcs.nvEncUnmapInputResource.unwrap())(self.session, input);
        encoded
    }

    /// Encode from an already-mapped input, locking the bitstream and copying it out.
    unsafe fn encode_mapped(
        &mut self,
        input: NV_ENC_INPUT_PTR,
        force_idr: bool,
    ) -> Result<(Vec<u8>, NV_ENC_PIC_TYPE), String> {
        let mut pic = NV_ENC_PIC_PARAMS {
            version: NV_ENC_PIC_PARAMS_VER,
            inputBuffer: input,
            bufferFmt: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
            inputWidth: self.width,
            inputHeight: self.height,
            outputBitstream: self.bitstream,
            pictureStruct: NV_ENC_PIC_STRUCT::NV_ENC_PIC_STRUCT_FRAME,
            ..Default::default()
        };
        if force_idr {
            pic.encodePicFlags = NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_FORCEIDR as u32;
        }
        let status = (self.funcs.nvEncEncodePicture.unwrap())(self.session, &mut pic);
        if status != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("nvEncEncodePicture: {status:?}"));
        }
        let mut lock = NV_ENC_LOCK_BITSTREAM {
            version: NV_ENC_LOCK_BITSTREAM_VER,
            outputBitstream: self.bitstream,
            ..Default::default()
        };
        let status = (self.funcs.nvEncLockBitstream.unwrap())(self.session, &mut lock);
        if status != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("nvEncLockBitstream: {status:?}"));
        }
        let bytes = std::slice::from_raw_parts(
            lock.bitstreamBufferPtr as *const u8,
            lock.bitstreamSizeInBytes as usize,
        )
        .to_vec();
        let picture_type = lock.pictureType;
        let status = (self.funcs.nvEncUnlockBitstream.unwrap())(self.session, self.bitstream);
        if status != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("nvEncUnlockBitstream: {status:?}"));
        }
        Ok((bytes, picture_type))
    }
}

impl Drop for NvencDirectX {
    /// Tear the session down in the reverse order it came up: every registered texture is
    /// unregistered (their textures and the D3D11 device are still alive here, dropped after
    /// this struct on the same thread), the bitstream buffer destroyed, and the encoder
    /// session closed.
    fn drop(&mut self) {
        unsafe {
            for (_, (_, registered)) in self.registered.iter() {
                let _ = (self.funcs.nvEncUnregisterResource.unwrap())(self.session, *registered);
            }
            let _ = (self.funcs.nvEncDestroyBitstreamBuffer.unwrap())(self.session, self.bitstream);
            let _ = (self.funcs.nvEncDestroyEncoder.unwrap())(self.session);
        }
    }
}

/// Resolve the NVENC driver library and its function list, as the Linux backend does: the
/// driver is loaded at run time by the name the platform names it, and
/// `NvEncodeAPICreateInstance` fills the list for this API version.
fn load_nvenc_functions() -> Result<(Library, NV_ENCODE_API_FUNCTION_LIST), String> {
    unsafe {
        let lib = Library::new(nvcodec_sys::NVENC_DLL_NAME)
            .map_err(|e| format!("Could not load the NVENC library: {e}"))?;
        let create: libloading::Symbol<NvEncodeApiCreateInstanceFn> = lib
            .get(NV_ENCODE_API_CREATE_INSTANCE_FN_NAME)
            .map_err(|e| format!("Missing symbol NvEncodeAPICreateInstance: {e}"))?;
        let mut funcs = NV_ENCODE_API_FUNCTION_LIST {
            version: NV_ENCODE_API_FUNCTION_LIST_VER,
            ..Default::default()
        };
        create(&mut funcs);
        if funcs.nvEncOpenEncodeSessionEx.is_none() {
            return Err("NvEncodeAPICreateInstance filled no entry points".into());
        }
        Ok((lib, funcs))
    }
}

/// The adapter the capture encodes on: the first NVIDIA one (vendor `0x10DE`, whose driver
/// serves NVENC), the default adapter where none is present, or none where the system lists
/// no adapter at all (the D3D11 default hardware path picks one). The adapter's description
/// is returned with it for the log.
fn pick_adapter(factory: &IDXGIFactory1) -> Result<(Option<IDXGIAdapter1>, String), String> {
    let mut adapters: Vec<(IDXGIAdapter1, DXGI_ADAPTER_DESC1)> = Vec::new();
    let mut i = 0u32;
    loop {
        let adapter = unsafe { factory.EnumAdapters1(i) };
        let Ok(adapter) = adapter else {
            break;
        };
        let desc = unsafe { adapter.GetDesc1() }.map_err(|e| format!("GetDesc1: {e}"))?;
        adapters.push((adapter, desc));
        i += 1;
    }
    if let Some((adapter, desc)) = adapters.iter().find(|(_, d)| d.VendorId == 0x10DE) {
        return Ok((Some(adapter.clone()), adapter_name(desc)));
    }
    if let Some((adapter, desc)) = adapters.first() {
        eprintln!(
            "[Windows] no NVIDIA adapter (vendor 0x10DE) found; encoding on the default adapter"
        );
        return Ok((Some(adapter.clone()), adapter_name(desc)));
    }
    Ok((None, "the default adapter".to_string()))
}

/// An adapter's description, as the text it names itself with.
fn adapter_name(desc: &DXGI_ADAPTER_DESC1) -> String {
    desc.Description
        .iter()
        .take_while(|c| **c != 0)
        .map(|c| char::from_u32(*c as u32).unwrap_or('?'))
        .collect()
}

/// Create the D3D11 device the capture and the encoder share, on `adapter` where one was
/// picked, with the BGRA support Windows.Graphics.Capture writes and the video support NVENC
/// reads.
fn create_device(adapter: Option<&IDXGIAdapter1>) -> Result<ID3D11Device, String> {
    let flags = D3D11_CREATE_DEVICE_FLAG(
        D3D11_CREATE_DEVICE_VIDEO_SUPPORT.0 | D3D11_CREATE_DEVICE_BGRA_SUPPORT.0,
    );
    let driver: D3D_DRIVER_TYPE = match adapter {
        Some(_) => D3D_DRIVER_TYPE_UNKNOWN,
        None => D3D_DRIVER_TYPE_HARDWARE,
    };
    let target: Option<IDXGIAdapter> = adapter
        .map(|a| a.cast::<IDXGIAdapter>())
        .transpose()
        .map_err(|e| format!("IDXGIAdapter cast: {e}"))?;
    let mut device: Option<ID3D11Device> = None;
    unsafe {
        D3D11CreateDevice(
            target.as_ref(),
            driver,
            HMODULE::default(),
            flags,
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            None,
        )
    }
    .map_err(|e| format!("D3D11CreateDevice: {e}"))?;
    device.ok_or_else(|| "D3D11CreateDevice returned no device".to_string())
}

/// Wrap the D3D11 device as the `IDirect3DDevice` Windows.Graphics.Capture takes.
fn winrt_device(device: &ID3D11Device) -> Result<IDirect3DDevice, String> {
    let dxgi: windows::Win32::Graphics::Dxgi::IDXGIDevice = device
        .cast()
        .map_err(|e| format!("device as IDXGIDevice: {e}"))?;
    let inspectable: IInspectable = unsafe { CreateDirect3D11DeviceFromDXGIDevice(&dxgi) }
        .map_err(|e| format!("CreateDirect3D11DeviceFromDXGIDevice: {e}"))?;
    inspectable
        .cast()
        .map_err(|e| format!("as IDirect3DDevice: {e}"))
}

/// The primary monitor's capture item, over the interop interface that creates one from an
/// `HMONITOR` without a WinRT activation factory in a windowed context.
fn primary_item() -> Result<GraphicsCaptureItem, String> {
    let monitor = unsafe { MonitorFromWindow(Default::default(), MONITOR_DEFAULTTOPRIMARY) };
    let interop: IGraphicsCaptureItemInterop =
        factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()
            .map_err(|e| format!("IGraphicsCaptureItemInterop factory: {e}"))?;
    unsafe { interop.CreateForMonitor(monitor) }.map_err(|e| format!("CreateForMonitor: {e}"))
}

/// The D3D11 texture a capture frame carries, behind the surface it is handed over as.
fn frame_texture(frame: &Direct3D11CaptureFrame) -> Result<ID3D11Texture2D, String> {
    let surface: IDirect3DSurface = frame.Surface().map_err(|e| format!("frame Surface: {e}"))?;
    let access: IDirect3DDxgiInterfaceAccess = surface
        .cast()
        .map_err(|e| format!("surface as IDirect3DDxgiInterfaceAccess: {e}"))?;
    unsafe { access.GetInterface() }.map_err(|e| format!("frame as ID3D11Texture2D: {e}"))
}

/// Run the capture: bring the device, the capture item, the frame pool, and the NVENC
/// session up, signal `started` once the pipeline is live (`Err` where it came up short, so
/// the caller can fail the start), then encode every frame the pool delivers into one
/// [`EncodedStripe`] and hand it to `on_frame`, until `controls.stop`.
///
/// This is the body of the capture thread `ScreenCapture::start_capture` spawns: everything
/// it touches — the D3D11 device, the frame pool, the NVENC session — is created and
/// released on that thread, and the only things that cross a thread boundary are the
/// atomics in `controls` and the frames handed to `on_frame`. The frame is encoded in place
/// as a D3D11 texture registered with NVENC; the bitstream is copied out of the locked
/// output buffer, and that copy is the CPU's only share of the frame.
pub fn run_capture<F>(
    settings: CaptureSettings,
    controls: Arc<Controls>,
    started: std::sync::mpsc::Sender<Result<(), String>>,
    mut on_frame: F,
) -> Result<(), String>
where
    F: FnMut(Vec<EncodedStripe>),
{
    let _com = ComApartment::init();
    let (lib, funcs) = load_nvenc_functions()?;
    let factory: IDXGIFactory1 =
        unsafe { CreateDXGIFactory1() }.map_err(|e| format!("CreateDXGIFactory1: {e}"))?;
    let (adapter, adapter_desc) = pick_adapter(&factory)?;
    let device = create_device(adapter.as_ref())?;
    let winrt = winrt_device(&device)?;
    let item = primary_item()?;
    let size = item.Size().map_err(|e| format!("capture item size: {e}"))?;
    let (width, height) = (size.Width as u32, size.Height as u32);
    if (settings.width > 0 && settings.width != width as i32)
        || (settings.height > 0 && settings.height != height as i32)
    {
        eprintln!(
            "[Windows] asked for {}x{}, but this milestone captures the monitor's native {}x{}; scaling is not wired yet",
            settings.width, settings.height, width, height
        );
    }
    let pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        &winrt,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        3,
        size,
    )
    .map_err(|e| format!("CreateFreeThreaded: {e}"))?;
    let session = pool
        .CreateCaptureSession(&item)
        .map_err(|e| format!("CreateCaptureSession: {e}"))?;
    // The pointer is composited into the frames where the settings ask it drawn, exactly as
    // the Linux backends' in-frame cursor is; the out-of-band cursor callback is not wired
    // to Windows yet.
    let _ = session.SetIsCursorCaptureEnabled(settings.capture_cursor);
    session
        .StartCapture()
        .map_err(|e| format!("StartCapture: {e}"))?;

    let mut nvenc = unsafe { NvencDirectX::open(lib, funcs, &device, width, height, &settings)? };

    let kbps = bitrate_bps(&settings) / 1000;
    let gop = gop_frames(&settings);
    println!(
        "[Windows] Capture started for the primary monitor: {width}x{height} on {adapter_desc}, zero-copy through Windows.Graphics.Capture"
    );
    println!(
        "[Windows] Stream settings active -> Res: {width}x{height} | FPS: {:.1} | Encoder: NVENC | Mode: H264 | CBR {kbps} | GOP: {}",
        settings.target_fps,
        if gop == NVENC_INFINITE_GOPLENGTH {
            "infinite".to_string()
        } else {
            format!("{gop} frames")
        }
    );
    let _ = started.send(Ok(()));

    // The first frame is forced to an IDR: a stream that starts on a delta frame has
    // nothing a decoder can start from, and the first frame the pool delivers (the one
    // StartCapture produced) is the natural keyframe.
    let mut frame_id: i32 = 0;
    let mut first = true;
    let mut logged_pool_error = false;
    let mut frames_coded: u64 = 0;
    let mut report_window = std::time::Instant::now();
    loop {
        if controls.stop.load(Ordering::Relaxed) {
            break;
        }
        let frame = match pool.TryGetNextFrame() {
            Ok(frame) => {
                logged_pool_error = false;
                frame
            }
            // A null frame is how the projection reports the empty pool: nothing changed
            // on screen.
            Err(e) if e.code() == windows::core::HRESULT(0) => {
                thread::sleep(IDLE_POLL);
                continue;
            }
            Err(e) => {
                if !logged_pool_error {
                    eprintln!("[Windows] TryGetNextFrame: {e}; retrying");
                    logged_pool_error = true;
                }
                thread::sleep(IDLE_POLL);
                continue;
            }
        };
        let capture_ns = crate::clock::now_ns();
        let content = frame
            .ContentSize()
            .map_err(|e| format!("ContentSize: {e}"))?;
        if content.Width != width as i32 || content.Height != height as i32 {
            // The pool keeps its size when the monitor's mode changes; the encoder is bound
            // to that size, so the capture ends here rather than encoding letterboxed
            // frames. Live resize is parity work.
            return Err(format!(
                "the monitor changed to {}x{}; restart the capture at that size",
                content.Width, content.Height
            ));
        }
        let texture = frame_texture(&frame)?;
        let force_idr = first || controls.force_idr.swap(false, Ordering::AcqRel);
        let encode_start_ns = crate::clock::now_ns();
        let (bytes, picture_type) = unsafe { nvenc.encode(&texture, force_idr) }?;
        drop(frame);
        frame_id += 1;
        // What the bitstream carries decides the wire's frame kind and the reference it
        // names: the encoder inserts IDRs at the GOP boundary on its own, so the frame is
        // read back from the lock rather than assumed from the request.
        let wire_type = match picture_type {
            NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR => FRAME_KEY,
            NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_I => FRAME_INTRA,
            _ => FRAME_DELTA,
        };
        // With no B-frames, a predicted picture codes from the picture before it, which is
        // the frame this one follows in the stream.
        let reference = if wire_type == FRAME_DELTA {
            Reference::Frame((frame_id - 1) as u16)
        } else {
            Reference::None
        };
        let mut data = Vec::with_capacity(bytes.len() + 12);
        if !settings.omit_stripe_headers {
            push_video_header(
                &mut data,
                Codec::H264,
                wire_type,
                frame_id as u16,
                0,
                width as u16,
                height as u16,
                reference,
            );
        }
        data.extend_from_slice(&bytes);
        let mut stripes = vec![EncodedStripe {
            data: Arc::new(data),
            codec: Codec::H264,
            stripe_y_start: 0,
            stripe_height: height as i32,
            frame_id,
            timing: FrameTiming::default(),
            reference,
        }];
        FrameTiming::stamp(&mut stripes, capture_ns, encode_start_ns);
        first = false;
        frames_coded += 1;
        on_frame(stripes);
        if report_window.elapsed().as_secs_f64() >= 1.0 {
            crate::log::debug!(
                "[Windows] Res: {width}x{height} Encoder: NVENC EncFPS: {:.2}",
                frames_coded as f64 / report_window.elapsed().as_secs_f64()
            );
            frames_coded = 0;
            report_window = std::time::Instant::now();
        }
    }
    // The capture ends as the session and pool drop at the end of this scope, before the
    // D3D11 device does; `nvenc` drops ahead of them (it was declared after them), so the
    // NVENC teardown runs while the device and every registered texture are still alive.
    Ok(())
}
