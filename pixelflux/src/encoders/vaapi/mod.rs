/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! VA-API hardware sessions on a DRM render node for every video codec, driven through libva
//! directly.
//!
//! The session owns what a hardware encoder needs owned: the surfaces, the parameter buffers,
//! and, for the codecs that take them, the packed headers, so it also owns each picture's
//! reference lists. A frame a client lost is left out of the predictions the way NVENC and
//! libx264 allow (`encoders/reference.rs`): H.264 and HEVC keep the decoded picture buffer
//! the level admits and name the newest surviving frame in the slice header, VP9 and AV1
//! address their eight buffer slots, and VP8 steers its three buffers.
//!
//! Pixels reach the codec on a VA surface -- a Wayland dmabuf imported in place, or a packed
//! host frame uploaded -- and the video processor converts to the surface format on the GPU,
//! so no colorspace conversion happens on the CPU. Chroma follows `video_fullcolor` where the
//! codec carries 4:4:4 (HEVC, and VP9 as profile 1): a 4:4:4 session tries the surface formats
//! the driver allocates and its video processor renders, read through a `VAProfileNone`
//! configuration, until one survives the surface pool, the convert, and the codec open. Every
//! session converts with the BT.709 matrix the sRGB source's own primaries and transfer belong
//! to, at limited range, and declares it; VP8 is held to BT.601, the only matrix its bitstream
//! can name. libva itself is loaded at run time, so the host's copy serves the driver it was
//! built for, and a test can put its own functions behind the same table.

mod av1;
mod h264;
mod h265;
#[cfg(test)]
pub(crate) mod mock;
#[cfg(test)]
mod session_tests;
mod vp8;
mod vp9;

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr;
use std::sync::{Arc, OnceLock};

use smithay::backend::allocator::{dmabuf::Dmabuf, Buffer};
use va_sys::*;

use super::codec::{
    av1_is_key, frame_type_from_key, h264_frame_type, h265_frame_type, push_video_header, vp8_is_key, vp9_is_key,
    Codec, VIDEO_HEADER_LEN,
};
use super::reference::{Reference, ReferenceSlots, ReferenceWindow, SlotPlan, REFERENCE_FRAMES};
use super::session::{check_host_frame, RateSettings};
use crate::RustCaptureSettings;

/// The compression level asked of every session, the middle of the range a driver offers.
const QUALITY_LEVEL: u32 = 6;
/// The slices an H.264 or HEVC picture is cut into.
const SLICES: u32 = 4;
/// The bytes a coded buffer holds: the uncompressed picture and some, an upper bound on any
/// frame.
fn coded_buffer_size(width: u32, height: u32) -> u32 {
    3 * width * height + (1 << 16)
}

/// The libva of this process, loaded once, or why it could not be.
fn libva() -> Result<VaApi, String> {
    static LIBVA: OnceLock<Result<Libva, String>> = OnceLock::new();
    LIBVA.get_or_init(|| unsafe { Libva::load() }).as_ref().map(|lib| lib.api).map_err(Clone::clone)
}

/// How frames reach a session: a Wayland DRM-PRIME dmabuf, or packed host pixels in B,G,R,A
/// (`rgba: false`) or R,G,B,A (`rgba: true`) byte order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Dmabuf,
    Host { rgba: bool },
}

/// The 8-bit 4:4:4 surface formats a session tries, in order. The order only breaks a tie
/// on a driver whose video processor renders both: every frame reaches either through the
/// convert, so neither is nearer the host.
const FULLCOLOR_FOURCCS: [u32; 2] = [VA_FOURCC_444P, VA_FOURCC_XYUV];

/// The name of a surface format, for the session log.
pub(crate) fn fourcc_name(fourcc: u32) -> String {
    match fourcc {
        VA_FOURCC_NV12 => "nv12".into(),
        VA_FOURCC_444P => "yuv444p".into(),
        VA_FOURCC_XYUV => "vuyx".into(),
        VA_FOURCC_BGRA => "bgra".into(),
        VA_FOURCC_RGBA => "rgba".into(),
        other => String::from_utf8_lossy(&other.to_le_bytes()).trim().to_string(),
    }
}

unsafe extern "C" fn log_error(_user: *mut c_void, message: *const c_char) {
    if !message.is_null() {
        eprintln!("[vaapi] {}", unsafe { CStr::from_ptr(message) }.to_string_lossy().trim_end());
    }
}

unsafe extern "C" fn log_info(_user: *mut c_void, message: *const c_char) {
    if !message.is_null() {
        crate::log::debug!("[vaapi] {}", unsafe { CStr::from_ptr(message) }.to_string_lossy().trim_end());
    }
}

/// A VA display opened on a render node, terminated with the device.
pub(crate) struct Device {
    api: VaApi,
    display: VADisplay,
    _fd: OwnedFd,
    vendor: String,
}

/// The display is used from the one thread that holds the session.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { (self.api.vaTerminate)(self.display) };
    }
}

impl Device {
    /// Open the render node behind `encode_node_index` (`renderD128` when none is named) and
    /// initialize the driver on it.
    pub(crate) fn open(api: VaApi, encode_node_index: i32) -> Result<Self, String> {
        let render_node = format!("/dev/dri/renderD{}", 128 + encode_node_index.max(0));
        let path = CString::new(render_node.clone()).unwrap();
        let raw = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if raw < 0 {
            return Err(format!("{render_node}: {}", std::io::Error::last_os_error()));
        }
        Self::on(api, unsafe { OwnedFd::from_raw_fd(raw) }, &render_node)
    }

    /// Initialize the driver on the open render node `fd`, with its messages routed to the
    /// log; `render_node` names it in errors.
    pub(crate) fn on(api: VaApi, fd: OwnedFd, render_node: &str) -> Result<Self, String> {
        let display = unsafe { (api.vaGetDisplayDRM)(fd.as_raw_fd()) };
        if display.is_null() {
            return Err(format!("{render_node} carries no VA display"));
        }
        unsafe {
            (api.vaSetErrorCallback)(display, Some(log_error), ptr::null_mut());
            (api.vaSetInfoCallback)(display, Some(log_info), ptr::null_mut());
        }
        let (mut major, mut minor) = (0 as c_int, 0 as c_int);
        let status = unsafe { (api.vaInitialize)(display, &mut major, &mut minor) };
        if status != VA_STATUS_SUCCESS as VAStatus {
            return Err(format!("{render_node}: no VA driver initialized ({})", error_text(&api, status)));
        }
        let vendor = unsafe {
            let text = (api.vaQueryVendorString)(display);
            if text.is_null() { String::new() } else { CStr::from_ptr(text).to_string_lossy().into_owned() }
        };
        Ok(Self { api, display, _fd: fd, vendor })
    }

    fn check(&self, status: VAStatus, what: &str) -> Result<(), String> {
        if status == VA_STATUS_SUCCESS as VAStatus {
            Ok(())
        } else {
            Err(format!("{what}: {}", error_text(&self.api, status)))
        }
    }

    /// The profiles the driver lists.
    fn profiles(&self) -> Result<Vec<VAProfile>, String> {
        let mut profiles = vec![0 as VAProfile; unsafe { (self.api.vaMaxNumProfiles)(self.display) }.max(0) as usize];
        let mut listed: c_int = 0;
        self.check(unsafe { (self.api.vaQueryConfigProfiles)(self.display, profiles.as_mut_ptr(), &mut listed) }, "vaQueryConfigProfiles")?;
        profiles.truncate(listed.max(0) as usize);
        Ok(profiles)
    }

    /// The encode entry point of `profile`: the low-power one first, since recent Intel
    /// generations expose it as the only one for HEVC, VP9, and AV1 and it is the shorter path
    /// where both exist; None where the profile encodes on neither.
    fn encode_entrypoint(&self, profile: VAProfile) -> Option<VAEntrypoint> {
        let mut entrypoints = vec![0 as VAEntrypoint; unsafe { (self.api.vaMaxNumEntrypoints)(self.display) }.max(0) as usize];
        let mut listed: c_int = 0;
        let status = unsafe { (self.api.vaQueryConfigEntrypoints)(self.display, profile, entrypoints.as_mut_ptr(), &mut listed) };
        if status != VA_STATUS_SUCCESS as VAStatus {
            return None;
        }
        let listed = &entrypoints[..listed.max(0) as usize];
        [VAEntrypointEncSliceLP, VAEntrypointEncSlice].into_iter().find(|e| listed.contains(e))
    }

    /// One configuration attribute of a profile and entry point, None where the driver does
    /// not report it.
    fn attribute(&self, profile: VAProfile, entrypoint: VAEntrypoint, kind: VAConfigAttribType) -> Option<u32> {
        let mut attrib = VAConfigAttrib { type_: kind, value: 0 };
        let status = unsafe { (self.api.vaGetConfigAttributes)(self.display, profile, entrypoint, &mut attrib, 1) };
        (status == VA_STATUS_SUCCESS as VAStatus && attrib.value != VA_ATTRIB_NOT_SUPPORTED).then_some(attrib.value)
    }

    /// The surface attributes a configuration reports, asked for the way libva wants: once
    /// for the count, once for the values. Empty where the driver reports none.
    fn surface_attributes(&self, config: VAConfigID) -> Vec<VASurfaceAttrib> {
        let mut count: c_uint = 0;
        let mut attribs: Vec<VASurfaceAttrib> = Vec::new();
        for pass in 0..2 {
            let list = if pass == 0 { ptr::null_mut() } else { attribs.as_mut_ptr() };
            if unsafe { (self.api.vaQuerySurfaceAttributes)(self.display, config, list, &mut count) } != VA_STATUS_SUCCESS as VAStatus {
                return Vec::new();
            }
            attribs.resize(count as usize, unsafe { std::mem::zeroed() });
        }
        attribs
    }

    /// The surface formats a configuration renders into, empty where it constrains none.
    fn surface_fourccs(&self, config: VAConfigID) -> Vec<u32> {
        self.surface_attributes(config)
            .iter()
            .filter(|a| a.type_ == VASurfaceAttribPixelFormat)
            .map(|a| unsafe { a.value.value.i } as u32)
            .collect()
    }

    /// The surface alignment a configuration asks for, as `(width, height)`, one where it
    /// asks none.
    fn surface_alignment(&self, config: VAConfigID) -> (u32, u32) {
        self.surface_attributes(config)
            .iter()
            .find(|a| a.type_ == VASurfaceAttribAlignmentSize)
            .map(|a| {
                let v = unsafe { a.value.value.i } as u32;
                (1 << (v & 0xf), 1 << ((v & 0xf0) >> 4))
            })
            .unwrap_or((1, 1))
    }

    /// Surfaces of `fourcc` at `width` x `height`, `count` of them, on the driver's own memory.
    fn create_surfaces(&self, rt_format: u32, fourcc: u32, width: u32, height: u32, count: usize) -> Result<Vec<VASurfaceID>, String> {
        let mut attrib: VASurfaceAttrib = unsafe { std::mem::zeroed() };
        attrib.type_ = VASurfaceAttribPixelFormat;
        attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
        attrib.value.type_ = VAGenericValueTypeInteger;
        attrib.value.value.i = fourcc as i32;
        let mut surfaces = vec![VA_INVALID_SURFACE; count];
        self.check(
            unsafe {
                (self.api.vaCreateSurfaces)(self.display, rt_format, width, height, surfaces.as_mut_ptr(), count as c_uint, &mut attrib, 1)
            },
            &format!("this VA-API driver allocates no {} surfaces", fourcc_name(fourcc)),
        )?;
        Ok(surfaces)
    }

    fn destroy_surfaces(&self, surfaces: &mut [VASurfaceID]) {
        if !surfaces.is_empty() {
            unsafe { (self.api.vaDestroySurfaces)(self.display, surfaces.as_mut_ptr(), surfaces.len() as c_int) };
        }
    }

    fn create_buffer(&self, context: VAContextID, kind: VABufferType, data: &[u8]) -> Result<VABufferID, String> {
        let mut id = VA_INVALID_ID;
        self.check(
            unsafe {
                (self.api.vaCreateBuffer)(self.display, context, kind, data.len() as c_uint, 1, data.as_ptr() as *mut c_void, &mut id)
            },
            &format!("vaCreateBuffer(type {kind})"),
        )?;
        Ok(id)
    }
}

fn error_text(api: &VaApi, status: VAStatus) -> String {
    let text = unsafe { (api.vaErrorStr)(status) };
    if text.is_null() { format!("VA status {status}") } else { unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned() }
}

/// The video codecs the VA-API driver of the render node behind `encode_node_index` encodes:
/// those one of whose profiles the driver lists with an encode entry point, which is what
/// `vainfo` reports and what a session checks first. The device is opened the way a session
/// opens it and released. An error names the step that failed: no such node, no VA driver on
/// it, or a libva the probe cannot reach.
pub(crate) fn probe_codecs(encode_node_index: i32) -> Result<Vec<(Codec, bool)>, String> {
    probe_codecs_on(&Device::open(libva()?, encode_node_index)?)
}

/// `probe_codecs` on an open device: each codec a profile of its 4:2:0 ladder encodes, and
/// whether the profile its 4:4:4 session opens under encodes and renders 4:4:4 surfaces too.
pub(crate) fn probe_codecs_on(device: &Device) -> Result<Vec<(Codec, bool)>, String> {
    let profiles = device.profiles()?;
    let entrypoint = |profile: VAProfile| profiles.contains(&profile).then(|| device.encode_entrypoint(profile)).flatten();
    Ok(Codec::VIDEO
        .into_iter()
        .filter(|&codec| profile_ladder(codec, false).into_iter().any(|p| entrypoint(p).is_some()))
        .map(|codec| {
            let fullcolor = profile_ladder(codec, true).into_iter().any(|p| {
                entrypoint(p).is_some_and(|e| {
                    device.attribute(p, e, VAConfigAttribRTFormat).is_none_or(|f| f & VA_RT_FORMAT_YUV444 != 0)
                })
            });
            (codec, fullcolor)
        })
        .collect())
}

/// The VA profiles a session opens under, in order of preference: for 4:2:0, the ones an
/// 8-bit session comes up as; for a 4:4:4 request, the profile that carries it, or nothing
/// where the codec has none the session serves.
fn profile_ladder(codec: Codec, fullcolor: bool) -> Vec<VAProfile> {
    match (codec, fullcolor) {
        (Codec::H264, false) => vec![VAProfileH264High, VAProfileH264Main, VAProfileH264ConstrainedBaseline],
        (Codec::H265, false) => vec![VAProfileHEVCMain],
        (Codec::H265, true) => vec![VAProfileHEVCMain444],
        (Codec::Vp8, false) => vec![VAProfileVP8Version0_3],
        (Codec::Vp9, false) => vec![VAProfileVP9Profile0],
        (Codec::Vp9, true) => vec![VAProfileVP9Profile1],
        (Codec::Av1, false) => vec![VAProfileAV1Profile0],
        _ => Vec::new(),
    }
}

/// The parameter buffers of one picture, in the order they are rendered.
pub(super) struct Buffers {
    entries: Vec<(VABufferType, Vec<u8>)>,
}

impl Buffers {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// One parameter structure as a buffer of its type.
    pub(super) fn push<T: Copy>(&mut self, kind: VABufferType, value: &T) {
        let bytes = unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) };
        self.entries.push((kind, bytes.to_vec()));
    }

    /// One miscellaneous parameter, behind the header naming its type.
    pub(super) fn push_misc<T: Copy>(&mut self, kind: VAEncMiscParameterType, value: &T) {
        let mut bytes = kind.to_ne_bytes().to_vec();
        bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) });
        self.entries.push((VAEncMiscParameterBufferType, bytes));
    }

    /// One packed header: its parameter buffer, then its bytes.
    pub(super) fn push_packed(&mut self, kind: u32, bytes: &[u8], bit_length: u32) {
        let param = VAEncPackedHeaderParameterBuffer { type_: kind, bit_length, has_emulation_bytes: 1, va_reserved: [0; 4] };
        self.push(VAEncPackedHeaderParameterBufferType, &param);
        self.entries.push((VAEncPackedHeaderDataBufferType, bytes.to_vec()));
    }

    #[cfg(test)]
    pub(crate) fn entries(&self) -> &[(VABufferType, Vec<u8>)] {
        &self.entries
    }
}

/// What the session negotiated at open, which the codec arms read.
#[derive(Clone, Copy)]
pub(super) struct Negotiated {
    pub profile: VAProfile,
    pub entrypoint: VAEntrypoint,
    /// The packed headers the driver takes, `VA_ENC_PACKED_HEADER_*` bits.
    pub packed: u32,
    /// The rate-control mode, `VA_RC_CQP` or `VA_RC_CBR`.
    pub rc_mode: u32,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bits_per_second: u32,
    /// The reference frames the decoded picture buffer holds.
    pub dpb: u32,
    pub fullcolor: bool,
    /// The quantizer bounds of a constant-rate session in the codec's own domain, 0 for none.
    pub min_qp: u32,
    pub max_qp: u32,
}

/// One frame as the codec arms see it: what it is, where its reconstruction and coded output
/// go, and what it may predict from.
pub(super) struct Frame<'a> {
    pub key: bool,
    /// The session's count of encoded frames.
    pub pts: u64,
    /// The count at the last key frame, where a codec's own counters restart.
    pub key_pts: u64,
    pub recon: VASurfaceID,
    pub coded: VABufferID,
    /// The frame predicted from, by timestamp and reconstruction surface.
    pub reference: Option<(u64, VASurfaceID)>,
    /// Every frame the decoder holds, oldest first, with the surface each was reconstructed
    /// into and whether a client lost it.
    pub held: &'a [(u64, VASurfaceID, bool)],
    /// The quantizer, in the codec's domain, of a constant-quantizer session.
    pub qp: u32,
    /// VP8's buffer plan.
    pub slots: SlotPlan,
    /// The surface each VP8 buffer holds, LAST, GOLDEN, ALTREF.
    pub slot_surfaces: [VASurfaceID; 3],
}

/// The codec arm of a session.
enum Arm {
    H264(Box<h264::Arm>),
    H265(h265::Arm),
    Vp8(vp8::Arm),
    Vp9(vp9::Arm),
    Av1(av1::Arm),
}

impl Arm {
    /// The surface alignment the codec's blocks want.
    fn alignment(&self) -> (u32, u32) {
        match self {
            Arm::H264(_) | Arm::Vp8(_) => (16, 16),
            Arm::H265(a) => a.alignment(),
            Arm::Vp9(_) => (64, 64),
            Arm::Av1(a) => a.alignment(),
        }
    }

    /// The packed headers the codec writes itself.
    fn wanted_packed(&self) -> u32 {
        match self {
            Arm::H264(_) | Arm::H265(_) => VA_ENC_PACKED_HEADER_SEQUENCE | VA_ENC_PACKED_HEADER_SLICE,
            Arm::Av1(_) => VA_ENC_PACKED_HEADER_SEQUENCE | VA_ENC_PACKED_HEADER_PICTURE,
            Arm::Vp8(_) | Arm::Vp9(_) => 0,
        }
    }

    /// Whether the session can name what a frame predicts from and leave a lost one out: the
    /// codecs whose headers the session writes need the driver to take them, since a driver
    /// writing its own slice headers names the references it chooses.
    fn tracks_references(&self, packed: u32) -> bool {
        match self {
            Arm::H264(_) | Arm::H265(_) => packed & VA_ENC_PACKED_HEADER_SLICE != 0,
            Arm::Av1(_) => packed & VA_ENC_PACKED_HEADER_PICTURE != 0,
            Arm::Vp8(_) | Arm::Vp9(_) => true,
        }
    }

    fn sequence(&mut self, negotiated: &Negotiated, out: &mut Buffers) {
        match self {
            Arm::H264(a) => a.sequence(negotiated, out),
            Arm::H265(a) => a.sequence(negotiated, out),
            Arm::Vp8(a) => a.sequence(negotiated, out),
            Arm::Vp9(a) => a.sequence(negotiated, out),
            Arm::Av1(a) => a.sequence(negotiated, out),
        }
    }

    fn picture(&mut self, negotiated: &Negotiated, frame: &Frame, out: &mut Buffers) -> Result<(), String> {
        match self {
            Arm::H264(a) => a.picture(negotiated, frame, out),
            Arm::H265(a) => a.picture(negotiated, frame, out),
            Arm::Vp8(a) => a.picture(negotiated, frame, out),
            Arm::Vp9(a) => a.picture(negotiated, frame, out),
            Arm::Av1(a) => a.picture(negotiated, frame, out),
        }
    }
}

/// The references a session keeps: a decoded picture buffer, or VP8's three buffers.
enum References {
    Window(ReferenceWindow),
    Slots(ReferenceSlots),
}

/// The host frame's way onto a surface: through the surface's own image where the driver
/// derives one, else through an image the driver copies in.
struct HostUpload {
    surface: VASurfaceID,
    fourcc: u32,
    derive: bool,
    image: Option<VAImage>,
}

/// One VA-API encoder session, hardware, for one capture.
pub struct VaapiEncoder {
    codec: Codec,
    input: Input,
    device: Arc<Device>,
    config: VAConfigID,
    context: VAContextID,
    vpp_config: VAConfigID,
    vpp_context: VAContextID,
    /// The color standard the video processor is told to convert from and to.
    vpp_standards: (VAProcColorStandardType, VAProcColorStandardType),
    negotiated: Negotiated,
    fourcc: u32,
    rt_format: u32,
    surface_width: u32,
    surface_height: u32,
    low_power: bool,
    /// The reconstruction surfaces, one more than the frames the decoder holds.
    recon: Vec<VASurfaceID>,
    /// The converted picture the encoder reads.
    converted: Vec<VASurfaceID>,
    host: Option<HostUpload>,
    coded: VABufferID,
    arm: Arm,
    references: Option<References>,
    /// The surface each held frame was reconstructed into, by timestamp.
    surfaces_of: HashMap<u64, VASurfaceID>,
    last_reference: Reference,
    rate: RateSettings,
    qp: u32,
    /// The quality level the driver takes, clamped to its range; None where it reports none.
    quality_level: Option<u32>,
    /// Whether the next frame opens a sequence: a key frame carrying the sequence parameters
    /// and the rate control.
    sequence_start: bool,
    omit_headers: bool,
    fresh: bool,
}

unsafe impl Send for VaapiEncoder {}

impl Drop for VaapiEncoder {
    fn drop(&mut self) {
        let api = self.device.api;
        let display = self.device.display;
        unsafe {
            if self.coded != VA_INVALID_ID {
                (api.vaDestroyBuffer)(display, self.coded);
            }
            if let Some(host) = &self.host
                && let Some(image) = host.image
            {
                (api.vaDestroyImage)(display, image.image_id);
            }
            if self.vpp_context != VA_INVALID_ID {
                (api.vaDestroyContext)(display, self.vpp_context);
            }
            if self.context != VA_INVALID_ID {
                (api.vaDestroyContext)(display, self.context);
            }
            let mut surfaces = self.recon.clone();
            surfaces.extend(&self.converted);
            surfaces.extend(self.host.iter().map(|h| h.surface));
            self.device.destroy_surfaces(&mut surfaces);
            if self.vpp_config != VA_INVALID_ID {
                (api.vaDestroyConfig)(display, self.vpp_config);
            }
            if self.config != VA_INVALID_ID {
                (api.vaDestroyConfig)(display, self.config);
            }
        }
    }
}

impl VaapiEncoder {
    /// Stand up a session for `codec` on the render node the settings select, fed through
    /// `input`, with the libva of this process.
    pub fn new(settings: &RustCaptureSettings, codec: Codec, input: Input) -> Result<Self, String> {
        if !codec.is_video() {
            return Err("JPEG has no VA-API encoder".into());
        }
        Self::on_device(Arc::new(Device::open(libva()?, settings.encode_node_index)?), settings, codec, input)
    }

    /// `new` on an open device.
    pub(crate) fn on_device(device: Arc<Device>, settings: &RustCaptureSettings, codec: Codec, input: Input) -> Result<Self, String> {
        let fullcolor = settings.video_fullcolor && codec.fullcolor();
        let ladder = profile_ladder(codec, fullcolor);
        if ladder.is_empty() {
            return Err(format!("no VA-API profile carries {} 4:4:4", codec.display()));
        }
        let listed = device.profiles()?;
        let (profile, entrypoint) = ladder
            .iter()
            .filter(|p| listed.contains(p))
            .find_map(|&p| device.encode_entrypoint(p).map(|e| (p, e)))
            .ok_or_else(|| format!("this VA-API driver encodes no {} {}", codec.display(), super::chroma_name(fullcolor)))?;
        let mut last = None;
        let formats: Vec<u32> = if fullcolor { FULLCOLOR_FOURCCS.to_vec() } else { vec![VA_FOURCC_NV12] };
        for fourcc in formats {
            match Self::open(&device, settings, codec, input, profile, entrypoint, fourcc, fullcolor) {
                Ok(session) => return Ok(session),
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap_or_else(|| "no VA-API surface format to try".into()))
    }

    /// One bring-up on `fourcc` surfaces: the encode configuration and its attributes, the
    /// surfaces, the encode and processing contexts, and the codec arm.
    #[allow(clippy::too_many_arguments)]
    fn open(
        device: &Arc<Device>,
        settings: &RustCaptureSettings,
        codec: Codec,
        input: Input,
        profile: VAProfile,
        entrypoint: VAEntrypoint,
        fourcc: u32,
        fullcolor: bool,
    ) -> Result<Self, String> {
        let api = device.api;
        let rate = RateSettings::new(settings);
        let rc_mode = if rate.cbr { VA_RC_CBR } else { VA_RC_CQP };
        let rt_format = if fullcolor { VA_RT_FORMAT_YUV444 } else { VA_RT_FORMAT_YUV420 };
        let width = settings.width.max(1) as u32;
        let height = settings.height.max(1) as u32;
        let fps = rate.fps.max(1) as u32;
        let bits_per_second = if rate.cbr { rate.bps().min(u32::MAX as u64) as u32 } else { 0 };

        let mut attribs = Vec::new();
        if let Some(formats) = device.attribute(profile, entrypoint, VAConfigAttribRTFormat) {
            if formats & rt_format == 0 {
                return Err(format!("the {} entry point renders no {} surfaces", fourcc_name(fourcc), super::chroma_name(fullcolor)));
            }
            attribs.push(VAConfigAttrib { type_: VAConfigAttribRTFormat, value: rt_format });
        }
        match device.attribute(profile, entrypoint, VAConfigAttribRateControl) {
            Some(modes) if modes & rc_mode != 0 => attribs.push(VAConfigAttrib { type_: VAConfigAttribRateControl, value: rc_mode }),
            Some(_) => return Err(format!("this VA-API driver has no {} rate control for {}", if rate.cbr { "constant-rate" } else { "constant-quantizer" }, codec.display())),
            None if rate.cbr => return Err("this VA-API driver reports no rate control modes".into()),
            None => {}
        }
        let ref_l0 = device.attribute(profile, entrypoint, VAConfigAttribEncMaxRefFrames).map_or(0, |v| v & 0xffff);
        if ref_l0 < 1 {
            return Err("this VA-API driver takes no reference frames".into());
        }
        let level_bitrate = bits_per_second as u64;
        let dpb = match codec {
            Codec::H264 => super::codec::h264_dpb_frames(super::codec::h264_level(width, height, fps, level_bitrate), width, height),
            Codec::H265 => super::codec::h265_dpb_frames(super::codec::h265_level(width, height, fps, level_bitrate, true), width, height),
            _ => REFERENCE_FRAMES,
        };
        let mut arm = match codec {
            Codec::H264 => Arm::H264(Box::new(h264::Arm::new(profile))),
            Codec::H265 => Arm::H265(h265::Arm::new(
                profile,
                device.attribute(profile, entrypoint, VAConfigAttribEncHEVCFeatures),
                device.attribute(profile, entrypoint, VAConfigAttribEncHEVCBlockSizes),
            )),
            Codec::Vp8 => Arm::Vp8(vp8::Arm::new()),
            Codec::Vp9 => Arm::Vp9(vp9::Arm::new()),
            Codec::Av1 => Arm::Av1(av1::Arm::new(
                device.attribute(profile, entrypoint, VAConfigAttribEncAV1),
                device.attribute(profile, entrypoint, VAConfigAttribEncAV1Ext1),
                device.attribute(profile, entrypoint, VAConfigAttribEncAV1Ext2).ok_or("this VA-API driver reports no AV1 encoder attributes")?,
            )),
            Codec::Jpeg => unreachable!(),
        };
        let packed = device.attribute(profile, entrypoint, VAConfigAttribEncPackedHeaders).map_or(0, |v| v & arm.wanted_packed());
        if packed != 0 {
            attribs.push(VAConfigAttrib { type_: VAConfigAttribEncPackedHeaders, value: packed });
        }
        let quality_range = device.attribute(profile, entrypoint, VAConfigAttribEncQualityRange);
        let slice_caps = (
            device.attribute(profile, entrypoint, VAConfigAttribEncMaxSlices),
            device.attribute(profile, entrypoint, VAConfigAttribEncSliceStructure),
        );

        let mut config = VA_INVALID_ID;
        device.check(
            unsafe { (api.vaCreateConfig)(device.display, profile, entrypoint, attribs.as_mut_ptr(), attribs.len() as c_int, &mut config) },
            &format!("no VA-API encode configuration for {} on {} surfaces", codec.display(), fourcc_name(fourcc)),
        )?;
        let mut me = Self {
            codec,
            input,
            device: Arc::clone(device),
            config,
            context: VA_INVALID_ID,
            vpp_config: VA_INVALID_ID,
            vpp_context: VA_INVALID_ID,
            vpp_standards: (VAProcColorStandardNone, VAProcColorStandardNone),
            negotiated: Negotiated {
                profile,
                entrypoint,
                packed,
                rc_mode,
                width,
                height,
                fps,
                bits_per_second,
                dpb,
                fullcolor,
                min_qp: codec.quantizer_bound(rate.min_qp),
                max_qp: codec.quantizer_bound(rate.max_qp),
            },
            fourcc,
            rt_format,
            surface_width: 0,
            surface_height: 0,
            low_power: entrypoint == VAEntrypointEncSliceLP,
            recon: Vec::new(),
            converted: Vec::new(),
            host: None,
            coded: VA_INVALID_ID,
            arm: Arm::Vp8(vp8::Arm::new()),
            references: None,
            surfaces_of: HashMap::new(),
            last_reference: Reference::Untracked,
            rate,
            qp: codec.quantizer(settings.video_crf),
            quality_level: None,
            sequence_start: true,
            omit_headers: settings.omit_stripe_headers,
            fresh: true,
        };
        let (align_w, align_h) = {
            let (aw, ah) = arm.alignment();
            let (dw, dh) = device.surface_alignment(config);
            (aw.max(dw), ah.max(dh))
        };
        me.surface_width = width.div_ceil(align_w) * align_w;
        me.surface_height = height.div_ceil(align_h) * align_h;
        let rendered = device.surface_fourccs(config);
        if !rendered.is_empty() && !rendered.contains(&fourcc) {
            return Err(format!("the {} encoder takes no {} surfaces", codec.display(), fourcc_name(fourcc)));
        }
        let slices = match (&arm, slice_caps) {
            (Arm::H264(_) | Arm::H265(_), (Some(max), Some(structure))) => Some(slice_layout(structure, max, me.surface_height.div_ceil(arm_block(&arm)), SLICES)?),
            (Arm::H264(_) | Arm::H265(_), _) => Some((1, me.surface_height.div_ceil(arm_block(&arm)))),
            _ => None,
        };
        match &mut arm {
            Arm::H264(a) => a.configure(&me.negotiated, me.surface_width, me.surface_height, slices.unwrap()),
            Arm::H265(a) => a.configure(&me.negotiated, me.surface_width, me.surface_height, slices.unwrap()),
            Arm::Vp8(a) => a.configure(&me.negotiated),
            Arm::Vp9(a) => a.configure(&me.negotiated),
            Arm::Av1(a) => a.configure(&me.negotiated, me.surface_width, me.surface_height)?,
        }
        me.arm = arm;
        me.references = me.arm.tracks_references(packed).then(|| match me.arm {
            Arm::Vp8(_) => References::Slots(ReferenceSlots::new()),
            _ => References::Window(ReferenceWindow::new(dpb)),
        });
        if let Some(References::Window(w)) = &mut me.references
            && let Arm::H264(a) = &me.arm
        {
            w.set_frame_num_range(a.frame_num_range());
        }
        me.quality_level = quality_range.map(|max| QUALITY_LEVEL.min(max));

        let recon_count = match me.arm {
            Arm::Vp8(_) => 4,
            _ => dpb as usize + 1,
        };
        me.recon = device.create_surfaces(rt_format, fourcc, me.surface_width, me.surface_height, recon_count)?;
        me.converted = device.create_surfaces(rt_format, fourcc, me.surface_width, me.surface_height, 1)?;
        let mut render_targets = me.recon.clone();
        render_targets.extend(&me.converted);
        device.check(
            unsafe {
                (api.vaCreateContext)(
                    device.display,
                    config,
                    me.surface_width as c_int,
                    me.surface_height as c_int,
                    VA_PROGRESSIVE as c_int,
                    render_targets.as_mut_ptr(),
                    render_targets.len() as c_int,
                    &mut me.context,
                )
            },
            "no VA-API encode context",
        )?;
        me.coded = device.create_buffer(me.context, VAEncCodedBufferType, &vec![0u8; coded_buffer_size(me.surface_width, me.surface_height) as usize])?;
        me.open_vpp(device, input)?;
        Ok(me)
    }
}

/// The block a codec's slices are counted in: a macroblock, or the coding tree unit.
fn arm_block(arm: &Arm) -> u32 {
    match arm {
        Arm::H265(a) => a.ctu_size(),
        _ => 16,
    }
}

/// How many slices a picture of `rows` block rows is cut into and how many rows each
/// takes, for `wanted` slices under the driver's slice structure: arbitrary rows as asked,
/// a power of two of rows where that is all the driver takes, one row each where it takes
/// only equal rows.
fn slice_layout(structure: u32, max_slices: u32, rows: u32, wanted: u32) -> Result<(u32, u32), String> {
    let wanted = wanted.min(rows).max(1);
    let (count, size) = if structure & (VA_ENC_SLICE_STRUCTURE_ARBITRARY_ROWS | VA_ENC_SLICE_STRUCTURE_ARBITRARY_MACROBLOCKS) != 0 {
        (wanted, rows / wanted)
    } else if structure & VA_ENC_SLICE_STRUCTURE_POWER_OF_TWO_ROWS != 0 {
        let mut k = 1;
        while 2 * k * (wanted - 1) + 1 < rows {
            k *= 2;
        }
        (rows.div_ceil(k), k)
    } else if structure & VA_ENC_SLICE_STRUCTURE_EQUAL_ROWS != 0 {
        (rows, 1)
    } else {
        return Err(format!("this VA-API driver supports no usable slice structure ({structure:#x})"));
    };
    if count > max_slices {
        return Err(format!("this VA-API driver encodes at most {max_slices} slices, not {count}"));
    }
    Ok((count, size))
}

impl VaapiEncoder {
    /// The video processor: a configuration and a context on the converted surface, and the
    /// color standards it is told to convert between, chosen the way libavfilter's `scale_vaapi`
    /// chose them: explicit color properties where the driver takes them, else the standard
    /// nearest to the matrix and range asked for.
    fn open_vpp(&mut self, device: &Device, input: Input) -> Result<(), String> {
        let api = device.api;
        device.check(
            unsafe { (api.vaCreateConfig)(device.display, VAProfileNone, VAEntrypointVideoProc, ptr::null_mut(), 0, &mut self.vpp_config) },
            "no VA-API video processing configuration",
        )?;
        let rendered = device.surface_fourccs(self.vpp_config);
        if !rendered.is_empty() && !rendered.contains(&self.fourcc) {
            return Err(format!("this VA-API driver's video processor renders no {} surfaces", fourcc_name(self.fourcc)));
        }
        device.check(
            unsafe {
                (api.vaCreateContext)(
                    device.display,
                    self.vpp_config,
                    self.surface_width as c_int,
                    self.surface_height as c_int,
                    VA_PROGRESSIVE as c_int,
                    self.converted.as_mut_ptr(),
                    self.converted.len() as c_int,
                    &mut self.vpp_context,
                )
            },
            "no VA-API video processing context",
        )?;
        let mut caps: VAProcPipelineCaps = unsafe { std::mem::zeroed() };
        device.check(
            unsafe { (api.vaQueryVideoProcPipelineCaps)(device.display, self.vpp_context, ptr::null_mut(), 0, &mut caps) },
            "vaQueryVideoProcPipelineCaps",
        )?;
        let standards = |list: *mut VAProcColorStandardType, count: u32| -> Vec<VAProcColorStandardType> {
            if list.is_null() { Vec::new() } else { unsafe { std::slice::from_raw_parts(list, count as usize) }.to_vec() }
        };
        let input_standards = standards(caps.input_color_standards, caps.num_input_color_standards);
        let output_standards = standards(caps.output_color_standards, caps.num_output_color_standards);
        self.vpp_standards = (
            color_standard(&input_standards, ColorDescription::SRGB_SOURCE),
            color_standard(&output_standards, self.declared_color()),
        );
        if let Input::Host { rgba } = input {
            let fourcc = if rgba { VA_FOURCC_RGBA } else { VA_FOURCC_BGRA };
            let surface = device.create_surfaces(VA_RT_FORMAT_RGB32, fourcc, self.negotiated.width, self.negotiated.height, 1)?[0];
            let mut derived: VAImage = unsafe { std::mem::zeroed() };
            let derive = unsafe { (api.vaDeriveImage)(device.display, surface, &mut derived) } == VA_STATUS_SUCCESS as VAStatus && {
                let same = derived.format.fourcc == fourcc;
                unsafe { (api.vaDestroyImage)(device.display, derived.image_id) };
                same
            };
            let image = if derive {
                None
            } else {
                let mut format = image_format(device, fourcc).ok_or_else(|| format!("this VA-API driver has no {} image format", fourcc_name(fourcc)))?;
                let mut image: VAImage = unsafe { std::mem::zeroed() };
                device.check(
                    unsafe { (api.vaCreateImage)(device.display, &mut format, self.negotiated.width as c_int, self.negotiated.height as c_int, &mut image) },
                    "vaCreateImage",
                )?;
                Some(image)
            };
            self.host = Some(HostUpload { surface, fourcc, derive, image });
        }
        Ok(())
    }

    /// The color the session declares: BT.709, whose primaries and transfer the sRGB desktop
    /// source already carries, at limited range; VP8 is held to BT.601, the only matrix its
    /// keyframe header's one color-space bit can name.
    fn declared_color(&self) -> ColorDescription {
        ColorDescription { primaries: 1, transfer: 1, matrix: if self.codec == Codec::Vp8 { 6 } else { 1 }, full_range: false, rgb: false }
    }

    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// Whether this session negotiated 4:4:4 chroma.
    pub fn is_fullcolor(&self) -> bool {
        self.negotiated.fullcolor
    }

    /// The name of the surface format frames reach the codec in, for the session log.
    pub fn surface_format_name(&self) -> String {
        fourcc_name(self.fourcc)
    }

    /// A hardware session converts at limited range.
    pub fn is_full_range(&self) -> bool {
        false
    }

    /// The driver behind the session, as libva names it.
    pub fn vendor(&self) -> &str {
        &self.device.vendor
    }

    /// Whether the codec runs on the low-power entry point.
    pub fn low_power(&self) -> bool {
        self.low_power
    }

    /// The frame the last encoded frame predicted from.
    pub fn last_reference(&self) -> Reference {
        self.last_reference
    }

    /// Leave frame `frame_id` and every frame after it out of the predictions. False where the
    /// session tracks none, and the caller codes a key frame instead.
    pub fn invalidate_reference(&mut self, frame_id: u16) -> bool {
        match &mut self.references {
            Some(References::Window(w)) => {
                w.invalidate(frame_id);
                true
            }
            Some(References::Slots(s)) => {
                s.invalidate(frame_id);
                true
            }
            None => false,
        }
    }

    /// Take a rate or frame-rate change: the next frame opens a sequence with the new rate
    /// control, as a key frame.
    pub fn reconfigure_rate(&mut self, settings: &RustCaptureSettings) -> Result<(), String> {
        let Some(rate) = self.rate.changed(settings) else { return Ok(()) };
        self.rate = rate;
        self.negotiated.fps = rate.fps.max(1) as u32;
        self.negotiated.bits_per_second = if rate.cbr { rate.bps().min(u32::MAX as u64) as u32 } else { 0 };
        self.negotiated.min_qp = self.codec.quantizer_bound(rate.min_qp);
        self.negotiated.max_qp = self.codec.quantizer_bound(rate.max_qp);
        match &mut self.arm {
            Arm::H264(a) => a.configure(&self.negotiated, self.surface_width, self.surface_height, a.slices()),
            Arm::H265(a) => a.configure(&self.negotiated, self.surface_width, self.surface_height, a.slices()),
            Arm::Vp8(a) => a.configure(&self.negotiated),
            Arm::Vp9(a) => a.configure(&self.negotiated),
            Arm::Av1(a) => a.configure(&self.negotiated, self.surface_width, self.surface_height)?,
        }
        self.sequence_start = true;
        Ok(())
    }

    /// The rate control of a sequence: the target, buffer, and frame rate a constant-rate
    /// session holds, the frame rate and quality level of any.
    fn rate_control(&self, out: &mut Buffers) {
        if self.rate.cbr {
            let bps = self.negotiated.bits_per_second;
            let vbv = self.rate.vbv();
            let mut rc: VAEncMiscParameterRateControl = unsafe { std::mem::zeroed() };
            rc.bits_per_second = bps;
            rc.target_percentage = 100;
            rc.window_size = (vbv as u64 * 1000 / bps.max(1) as u64) as u32;
            rc.initial_qp = 0;
            rc.min_qp = self.negotiated.min_qp;
            rc.max_qp = self.negotiated.max_qp;
            rc.basic_unit_size = 0;
            rc.ICQ_quality_factor = 1;
            rc.quality_factor = 0;
            unsafe { rc.rc_flags.bits.set_mb_rate_control(2) };
            out.push_misc(VAEncMiscParameterTypeRateControl, &rc);
            let hrd = VAEncMiscParameterHRD { initial_buffer_fullness: vbv, buffer_size: vbv, va_reserved: [0; 4] };
            out.push_misc(VAEncMiscParameterTypeHRD, &hrd);
        }
        let mut frame_rate: VAEncMiscParameterFrameRate = unsafe { std::mem::zeroed() };
        frame_rate.framerate = (1 << 16) | self.negotiated.fps;
        out.push_misc(VAEncMiscParameterTypeFrameRate, &frame_rate);
        if let Some(level) = self.quality_level {
            let quality = VAEncMiscParameterBufferQualityLevel { quality_level: level, va_reserved: [0; 4] };
            out.push_misc(VAEncMiscParameterTypeQualityLevel, &quality);
        }
    }

    /// Encode one packed host frame (`stride` bytes per row, in the byte order the session
    /// was built for) at the quality index `crf`: uploaded straight from the caller's rows
    /// onto the host surface and converted on the GPU.
    pub fn encode_host(&mut self, pixels: &[u8], stride: usize, rgba: bool, frame_number: u64, crf: u32, force_idr: bool) -> Result<Vec<u8>, String> {
        let Input::Host { rgba: built } = self.input else { return Err("this session takes dmabufs".into()) };
        if built != rgba {
            return Err("this session was built for the other byte order".into());
        }
        let (width, height) = (self.negotiated.width as usize, self.negotiated.height as usize);
        check_host_frame(pixels, stride, width, height)?;
        let host = self.host.as_ref().ok_or("no host upload surface")?;
        let api = self.device.api;
        let display = self.device.display;
        unsafe {
            let mut image: VAImage = std::mem::zeroed();
            if host.derive {
                self.device.check((api.vaDeriveImage)(display, host.surface, &mut image), "vaDeriveImage")?;
            } else {
                image = host.image.unwrap();
            }
            let mut address: *mut c_void = ptr::null_mut();
            let mapped = self.device.check((api.vaMapBuffer)(display, image.buf, &mut address), "vaMapBuffer");
            if let Err(e) = mapped {
                if host.derive {
                    (api.vaDestroyImage)(display, image.image_id);
                }
                return Err(e);
            }
            let pitch = image.pitches[0] as usize;
            let dst = (address as *mut u8).add(image.offsets[0] as usize);
            for row in 0..height {
                ptr::copy_nonoverlapping(pixels.as_ptr().add(row * stride), dst.add(row * pitch), width * 4);
            }
            (api.vaUnmapBuffer)(display, image.buf);
            if host.derive {
                (api.vaDestroyImage)(display, image.image_id);
            } else {
                self.device.check(
                    (api.vaPutImage)(display, host.surface, image.image_id, 0, 0, width as c_uint, height as c_uint, 0, 0, width as c_uint, height as c_uint),
                    "vaPutImage",
                )?;
            }
        }
        let surface = host.surface;
        self.encode_surface(surface, frame_number, crf, force_idr)
    }

    /// Encode one Wayland DRM-PRIME dmabuf: imported as a VA surface in place for the frame and
    /// converted on the GPU.
    pub fn encode_dmabuf(&mut self, dmabuf: &Dmabuf, frame_number: u64, crf: u32, force_idr: bool) -> Result<Vec<u8>, String> {
        if self.input != Input::Dmabuf {
            return Err("this session takes host frames".into());
        }
        let surface = self.import_dmabuf(dmabuf)?;
        let result = self.encode_surface(surface, frame_number, crf, force_idr);
        self.device.destroy_surfaces(&mut [surface]);
        result
    }

    /// A VA surface over a dmabuf: through the PRIME 2 descriptor, which carries the format
    /// modifier, else the older external-buffer one.
    fn import_dmabuf(&mut self, dmabuf: &Dmabuf) -> Result<VASurfaceID, String> {
        let api = self.device.api;
        let display = self.device.display;
        let (width, height) = (self.negotiated.width, self.negotiated.height);
        let handles: Vec<i32> = dmabuf.handles().map(|h| h.as_raw_fd()).collect();
        if handles.len() != 1 {
            return Err("VA-API maps only a dmabuf made of a single object".into());
        }
        let size = unsafe { libc::lseek(handles[0], 0, libc::SEEK_END) };
        if size <= 0 {
            return Err("the dmabuf object reports no size".into());
        }
        let drm_format = dmabuf.format().code as u32;
        let fourcc = match drm_format {
            0x34325241 => VA_FOURCC_BGRA,
            0x34325258 => VA_FOURCC_BGRX,
            0x34324241 => VA_FOURCC_RGBA,
            0x34324258 => VA_FOURCC_RGBX,
            other => return Err(format!("DRM format {other:#x} is not one VA-API maps")),
        };
        let planes: Vec<(u32, u32)> = dmabuf.strides().zip(dmabuf.offsets()).collect();
        let mut surface = VA_INVALID_SURFACE;
        let mut attribs: [VASurfaceAttrib; 2] = unsafe { std::mem::zeroed() };
        attribs[0].type_ = VASurfaceAttribMemoryType;
        attribs[0].flags = VA_SURFACE_ATTRIB_SETTABLE;
        attribs[0].value.type_ = VAGenericValueTypeInteger;
        attribs[1].type_ = VASurfaceAttribExternalBufferDescriptor;
        attribs[1].flags = VA_SURFACE_ATTRIB_SETTABLE;
        attribs[1].value.type_ = VAGenericValueTypePointer;
        let modifier = u64::from(dmabuf.format().modifier);
        if modifier != 0x00ff_ffff_ffff_ffff {
            let mut desc: VADRMPRIMESurfaceDescriptor = unsafe { std::mem::zeroed() };
            desc.fourcc = fourcc;
            desc.width = width;
            desc.height = height;
            desc.num_objects = 1;
            desc.objects[0].fd = handles[0];
            desc.objects[0].size = size as u32;
            desc.objects[0].drm_format_modifier = modifier;
            desc.num_layers = 1;
            desc.layers[0].drm_format = drm_format;
            desc.layers[0].num_planes = planes.len() as u32;
            for (i, &(pitch, offset)) in planes.iter().enumerate().take(4) {
                desc.layers[0].object_index[i] = 0;
                desc.layers[0].offset[i] = offset;
                desc.layers[0].pitch[i] = pitch;
            }
            attribs[0].value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2 as i32;
            attribs[1].value.value.p = &mut desc as *mut VADRMPRIMESurfaceDescriptor as *mut c_void;
            let status = unsafe { (api.vaCreateSurfaces)(display, VA_RT_FORMAT_RGB32, width, height, &mut surface, 1, attribs.as_mut_ptr(), 2) };
            if status == VA_STATUS_SUCCESS as VAStatus {
                return Ok(surface);
            }
        }
        let mut handle = handles[0] as usize;
        let mut desc: VASurfaceAttribExternalBuffers = unsafe { std::mem::zeroed() };
        desc.pixel_format = fourcc;
        desc.width = width;
        desc.height = height;
        desc.data_size = size as u32;
        desc.buffers = &mut handle;
        desc.num_buffers = 1;
        desc.num_planes = planes.len() as u32;
        for (i, &(pitch, offset)) in planes.iter().enumerate().take(4) {
            desc.pitches[i] = pitch;
            desc.offsets[i] = offset;
        }
        attribs[0].value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME as i32;
        attribs[1].value.value.p = &mut desc as *mut VASurfaceAttribExternalBuffers as *mut c_void;
        self.device.check(
            unsafe { (api.vaCreateSurfaces)(display, VA_RT_FORMAT_RGB32, width, height, &mut surface, 1, attribs.as_mut_ptr(), 2) },
            "no VA surface over the dmabuf",
        )?;
        Ok(surface)
    }

    /// Convert `source` onto the encoder's surface: BT.709 primaries and transfer in, full
    /// range, the declared matrix out at limited range, chroma sited at the center of each
    /// block, which is where the software convert puts it.
    fn convert(&mut self, source: VASurfaceID) -> Result<(), String> {
        let api = self.device.api;
        let display = self.device.display;
        let region = VARectangle { x: 0, y: 0, width: self.negotiated.width as u16, height: self.negotiated.height as u16 };
        let declared = self.declared_color();
        let mut params: VAProcPipelineParameterBuffer = unsafe { std::mem::zeroed() };
        params.surface = source;
        params.surface_region = &region;
        params.output_region = ptr::null();
        params.output_background_color = 0xff00_0000;
        params.pipeline_flags = 0;
        params.filter_flags = VA_FRAME_PICTURE;
        params.rotation_state = VA_ROTATION_NONE;
        params.mirror_state = VA_MIRROR_NONE;
        params.surface_color_standard = self.vpp_standards.0;
        params.output_color_standard = self.vpp_standards.1;
        params.input_color_properties = ColorDescription::SRGB_SOURCE.properties();
        params.output_color_properties = declared.properties();
        let buffer = self.device.create_buffer(self.vpp_context, VAProcPipelineParameterBufferType, unsafe {
            std::slice::from_raw_parts(&params as *const VAProcPipelineParameterBuffer as *const u8, std::mem::size_of::<VAProcPipelineParameterBuffer>())
        })?;
        let mut buffers = [buffer];
        let result = (|| {
            self.device.check(unsafe { (api.vaBeginPicture)(display, self.vpp_context, self.converted[0]) }, "vaBeginPicture (video processing)")?;
            let rendered = self.device.check(unsafe { (api.vaRenderPicture)(display, self.vpp_context, buffers.as_mut_ptr(), 1) }, "vaRenderPicture (video processing)");
            let ended = self.device.check(unsafe { (api.vaEndPicture)(display, self.vpp_context) }, "vaEndPicture (video processing)");
            rendered.and(ended)
        })();
        unsafe { (api.vaDestroyBuffer)(display, buffer) };
        result
    }

    /// Encode the picture on `source`: convert it, then issue the frame with its parameter
    /// buffers and packed headers, wait for it, and frame the coded bytes for the wire.
    fn encode_surface(&mut self, source: VASurfaceID, frame_number: u64, crf: u32, force_idr: bool) -> Result<Vec<u8>, String> {
        if !self.rate.cbr {
            self.qp = self.codec.quantizer(crf as i32);
        }
        self.convert(source)?;
        let frame_id = frame_number as u16;
        let has_reference = match &self.references {
            Some(References::Window(w)) => w.has_reference(),
            Some(References::Slots(s)) => s.has_reference(),
            None => !self.fresh,
        };
        let key = force_idr || self.sequence_start || !has_reference;
        let pts = match &self.references {
            Some(References::Window(w)) => w.next_pts(),
            Some(References::Slots(s)) => s.next_pts(),
            None => self.surfaces_of.len() as u64,
        };
        let recon = self.recon[(pts % self.recon.len() as u64) as usize];
        let held: Vec<(u64, VASurfaceID, bool)> = match &self.references {
            Some(References::Window(w)) => w.held().filter_map(|(_, p, lost)| self.surfaces_of.get(&p).map(|&s| (p, s, lost))).collect(),
            _ => Vec::new(),
        };
        let (reference, key_pts, slots, slot_surfaces) = match &self.references {
            Some(References::Window(w)) => (
                if key { None } else { w.newest_valid().and_then(|(_, p)| self.surfaces_of.get(&p).map(|&s| (p, s))) },
                w.key_pts(),
                SlotPlan::KEY,
                [VA_INVALID_SURFACE; 3],
            ),
            Some(References::Slots(s)) => {
                let plan = s.plan(key);
                let surfaces = [1u8, 2, 4].map(|slot| s.slot(slot).and_then(|(_, p, _)| self.surfaces_of.get(&p).copied()).unwrap_or(VA_INVALID_SURFACE));
                let reference = (plan.predict_from != 0).then(|| s.slot(plan.predict_from)).flatten().and_then(|(_, p, _)| self.surfaces_of.get(&p).map(|&surf| (p, surf)));
                (reference, 0, plan, surfaces)
            }
            None => (if key { None } else { self.surfaces_of.get(&(pts - 1)).map(|&s| (pts - 1, s)) }, 0, SlotPlan::KEY, [VA_INVALID_SURFACE; 3]),
        };
        let frame = Frame { key, pts, key_pts, recon, coded: self.coded, reference, held: &held, qp: self.qp, slots, slot_surfaces };
        let mut out = Buffers::new();
        if key {
            self.arm.sequence(&self.negotiated, &mut out);
            self.rate_control(&mut out);
        }
        self.arm.picture(&self.negotiated, &frame, &mut out)?;
        let coded = self.issue(&out)?;
        self.fresh = false;
        self.sequence_start = false;
        self.surfaces_of.insert(pts, recon);
        self.surfaces_of.retain(|&p, _| p + self.recon.len() as u64 > pts);
        let frame_type = match self.codec {
            Codec::H264 => h264_frame_type(&coded),
            Codec::H265 => h265_frame_type(&coded),
            Codec::Vp8 => frame_type_from_key(vp8_is_key(&coded)),
            Codec::Vp9 => frame_type_from_key(vp9_is_key(&coded)),
            _ => frame_type_from_key(av1_is_key(&coded)),
        };
        let is_key = frame_type != super::codec::FRAME_DELTA;
        self.last_reference = match &mut self.references {
            Some(References::Window(w)) => w.record(frame_id, is_key),
            Some(References::Slots(s)) => {
                let plan = if is_key { s.plan(true) } else { slots };
                s.record(frame_id, plan)
            }
            None => Reference::Untracked,
        };
        let mut output = Vec::with_capacity(VIDEO_HEADER_LEN + coded.len());
        if !self.omit_headers {
            push_video_header(&mut output, self.codec, frame_type, frame_id, 0, self.negotiated.width as u16, self.negotiated.height as u16, self.last_reference);
        }
        output.extend_from_slice(&coded);
        Ok(output)
    }

    /// Render `buffers` as one picture on the encode context, wait for it, and read the coded
    /// bytes back.
    fn issue(&mut self, buffers: &Buffers) -> Result<Vec<u8>, String> {
        let api = self.device.api;
        let display = self.device.display;
        let mut ids = Vec::with_capacity(buffers.entries.len());
        let mut created = Ok(());
        for (kind, bytes) in &buffers.entries {
            match self.device.create_buffer(self.context, *kind, bytes) {
                Ok(id) => ids.push(id),
                Err(e) => {
                    created = Err(e);
                    break;
                }
            }
        }
        let result = created.and_then(|()| {
            self.device.check(unsafe { (api.vaBeginPicture)(display, self.context, self.converted[0]) }, "vaBeginPicture")?;
            let rendered = self.device.check(unsafe { (api.vaRenderPicture)(display, self.context, ids.as_mut_ptr(), ids.len() as c_int) }, "vaRenderPicture");
            let ended = self.device.check(unsafe { (api.vaEndPicture)(display, self.context) }, "vaEndPicture");
            rendered.and(ended)
        });
        for id in ids {
            unsafe { (api.vaDestroyBuffer)(display, id) };
        }
        result?;
        let synced = match api.vaSyncBuffer {
            Some(sync) => {
                let status = unsafe { sync(display, self.coded, u64::MAX) };
                if status == VA_STATUS_ERROR_UNIMPLEMENTED as VAStatus { None } else { Some(self.device.check(status, "vaSyncBuffer")) }
            }
            None => None,
        };
        match synced {
            Some(result) => result?,
            None => self.device.check(unsafe { (api.vaSyncSurface)(display, self.converted[0]) }, "vaSyncSurface")?,
        }
        let mut list: *mut c_void = ptr::null_mut();
        self.device.check(unsafe { (api.vaMapBuffer)(display, self.coded, &mut list) }, "vaMapBuffer (coded)")?;
        let mut out = Vec::new();
        let mut segment = unsafe { (list as *const VACodedBufferSegment).as_ref() };
        while let Some(s) = segment {
            if !s.buf.is_null() {
                out.extend_from_slice(unsafe { std::slice::from_raw_parts(s.buf as *const u8, s.size as usize) });
            }
            segment = unsafe { (s.next as *const VACodedBufferSegment).as_ref() };
        }
        unsafe { (api.vaUnmapBuffer)(display, self.coded) };
        Ok(out)
    }
}

/// A color description as the video processor is told it: the ISO/IEC 23091-2 code points
/// and the range, and whether the picture is RGB, whose matrix is fixed.
#[derive(Clone, Copy)]
struct ColorDescription {
    primaries: u8,
    transfer: u8,
    matrix: u8,
    full_range: bool,
    rgb: bool,
}

impl ColorDescription {
    /// The sRGB desktop picture every session starts from: BT.709 primaries and transfer,
    /// full range.
    const SRGB_SOURCE: Self = Self { primaries: 1, transfer: 1, matrix: 0, full_range: true, rgb: true };

    fn properties(self) -> VAProcColorProperties {
        VAProcColorProperties {
            chroma_sample_location: (VA_CHROMA_SITING_VERTICAL_CENTER | VA_CHROMA_SITING_HORIZONTAL_CENTER) as u8,
            color_range: if self.full_range { VA_SOURCE_RANGE_FULL } else { VA_SOURCE_RANGE_REDUCED } as u8,
            colour_primaries: self.primaries,
            transfer_characteristics: self.transfer,
            matrix_coefficients: self.matrix,
            reserved: [0; 3],
        }
    }
}

/// The color standard a video processor is told for `wanted`: the explicit one where the
/// driver takes it, since the properties then say everything; else the standard whose
/// matrix, transfer, and primaries come closest, weighted four, two, and one, or none when
/// nothing matches at all.
fn color_standard(offered: &[VAProcColorStandardType], wanted: ColorDescription) -> VAProcColorStandardType {
    if offered.contains(&VAProcColorStandardExplicit) {
        return VAProcColorStandardExplicit;
    }
    const TABLE: [(VAProcColorStandardType, u8, u8, u8); 12] = [
        (VAProcColorStandardBT601, 5, 6, 5),
        (VAProcColorStandardBT601, 6, 6, 6),
        (VAProcColorStandardBT709, 1, 1, 1),
        (VAProcColorStandardBT470M, 4, 4, 4),
        (VAProcColorStandardBT470BG, 5, 5, 5),
        (VAProcColorStandardSMPTE170M, 6, 6, 6),
        (VAProcColorStandardSMPTE240M, 7, 7, 7),
        (VAProcColorStandardGenericFilm, 8, 1, 1),
        (VAProcColorStandardSRGB, 1, 13, 0),
        (VAProcColorStandardXVYCC601, 1, 11, 5),
        (VAProcColorStandardXVYCC709, 1, 11, 1),
        (VAProcColorStandardBT2020, 9, 14, 9),
    ];
    let matrix_counts = !wanted.rgb;
    let worst = 4 * matrix_counts as u32 + 2 + 1;
    let mut best = (worst, VAProcColorStandardNone);
    for &standard in offered {
        for &(_, primaries, transfer, matrix) in TABLE.iter().filter(|t| t.0 == standard) {
            let score = 4 * (matrix_counts && wanted.matrix != matrix) as u32 + 2 * (wanted.transfer != transfer) as u32 + (wanted.primaries != primaries) as u32;
            if score < best.0 {
                best = (score, standard);
            }
        }
    }
    best.1
}

/// The driver's image format of `fourcc`, with the masks it wants an image created with.
fn image_format(device: &Device, fourcc: u32) -> Option<VAImageFormat> {
    let api = device.api;
    let mut formats: Vec<VAImageFormat> = vec![unsafe { std::mem::zeroed() }; unsafe { (api.vaMaxNumImageFormats)(device.display) }.max(0) as usize];
    let mut count: c_int = 0;
    if unsafe { (api.vaQueryImageFormats)(device.display, formats.as_mut_ptr(), &mut count) } != VA_STATUS_SUCCESS as VAStatus {
        return None;
    }
    formats[..count.max(0) as usize].iter().find(|f| f.fourcc == fourcc).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The planar surface is tried first wherever a device both allocates and renders it, the
    /// packed one after it, and the profile ladder names the 4:4:4 profiles only where a
    /// codec carries them.
    #[test]
    fn surface_and_profile_ladders() {
        assert_eq!(FULLCOLOR_FOURCCS, [VA_FOURCC_444P, VA_FOURCC_XYUV]);
        assert_eq!(fourcc_name(VA_FOURCC_XYUV), "vuyx");
        assert_eq!(fourcc_name(VA_FOURCC_444P), "yuv444p");
        assert!(profile_ladder(Codec::H264, true).is_empty(), "no 4:4:4 H.264 profile is served");
        assert_eq!(profile_ladder(Codec::H265, true), [VAProfileHEVCMain444]);
        assert_eq!(profile_ladder(Codec::Vp9, true), [VAProfileVP9Profile1]);
        assert!(profile_ladder(Codec::Av1, true).is_empty());
        for codec in Codec::VIDEO {
            assert!(!profile_ladder(codec, false).is_empty(), "{codec:?}");
        }
    }

    /// The color standard follows libavfilter's choice: explicit wherever the driver takes
    /// it, else the closest standard by matrix, transfer, and primaries, with an RGB source
    /// scored on transfer and primaries alone and a tie going to the one the driver lists
    /// first.
    #[test]
    fn the_color_standard_is_the_nearest_the_driver_offers() {
        let explicit = [VAProcColorStandardBT601, VAProcColorStandardExplicit];
        assert_eq!(color_standard(&explicit, ColorDescription::SRGB_SOURCE), VAProcColorStandardExplicit);
        let classic = [VAProcColorStandardBT601, VAProcColorStandardBT709, VAProcColorStandardSMPTE170M, VAProcColorStandardSRGB];
        assert_eq!(color_standard(&classic, ColorDescription::SRGB_SOURCE), VAProcColorStandardBT709);
        let bt709 = ColorDescription { primaries: 1, transfer: 1, matrix: 1, full_range: false, rgb: false };
        assert_eq!(color_standard(&classic, bt709), VAProcColorStandardBT709);
        let bt601 = ColorDescription { matrix: 6, ..bt709 };
        assert_eq!(color_standard(&classic, bt601), VAProcColorStandardBT601, "the driver's first of two equal matches");
        assert_eq!(color_standard(&classic[2..], bt601), VAProcColorStandardSMPTE170M);
        assert_eq!(color_standard(&[VAProcColorStandardBT2020], bt709), VAProcColorStandardNone, "a total mismatch names no standard");
    }

    /// Slices follow the driver's structure: as many rows as asked where rows are free, a
    /// power of two of rows where that is all the driver takes, and one row each where only
    /// equal rows are.
    #[test]
    fn slice_layout_follows_the_driver() {
        assert_eq!(slice_layout(VA_ENC_SLICE_STRUCTURE_ARBITRARY_ROWS, 32, 68, 4), Ok((4, 17)));
        assert_eq!(slice_layout(VA_ENC_SLICE_STRUCTURE_POWER_OF_TWO_ROWS, 32, 68, 4), Ok((5, 16)));
        assert_eq!(slice_layout(VA_ENC_SLICE_STRUCTURE_EQUAL_ROWS, 128, 68, 4), Ok((68, 1)));
        assert!(slice_layout(VA_ENC_SLICE_STRUCTURE_EQUAL_ROWS, 32, 68, 4).is_err(), "more slices than the driver takes");
        assert!(slice_layout(0, 32, 68, 4).is_err());
        assert_eq!(slice_layout(VA_ENC_SLICE_STRUCTURE_ARBITRARY_ROWS, 32, 2, 4), Ok((2, 1)), "no more slices than rows");
    }

    /// Construction either stands a session up or says why it could not; a half-built
    /// encoder must never reach a caller, and a session never quietly changes chroma. Runs
    /// everywhere: a host without a VA-API device exercises the error path.
    #[test]
    fn construction_answers_or_refuses() {
        let mut settings = RustCaptureSettings { width: 128, height: 128, codec: Codec::H264, video_fullcolor: true, ..Default::default() };
        for codec in Codec::VIDEO {
            settings.codec = codec;
            match VaapiEncoder::new(&settings, codec, Input::Host { rgba: false }) {
                Ok(enc) => assert_eq!(enc.is_fullcolor(), matches!(codec, Codec::H265 | Codec::Vp9), "{codec:?}"),
                Err(e) => assert!(!e.is_empty(), "refusal must carry a reason"),
            }
        }
        settings.video_fullcolor = false;
        match VaapiEncoder::new(&settings, Codec::H264, Input::Host { rgba: false }) {
            Ok(enc) => assert!(!enc.is_fullcolor(), "4:2:0 session reports 4:4:4"),
            Err(e) => assert!(!e.is_empty(), "refusal must carry a reason"),
        }
        assert!(VaapiEncoder::new(&settings, Codec::Jpeg, Input::Host { rgba: false }).is_err());
    }
}
