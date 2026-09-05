/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Encoder backends and what they share: the codec identities and wire framing, the
//! rate-control policy, and which software encoder a build resolves each codec to.

/// libavcodec-backed encoders: VA-API hardware sessions on a DRM render node, and the
/// software HEVC / VP8 / VP9 / AV1 encoders the linked FFmpeg carries.
pub mod avcodec;
/// Codec identities, wire framing, quantizer domains, level ladders, bitstream reads.
pub mod codec;
/// NVIDIA NVENC hardware H.264 / HEVC / AV1 encoder loaded via runtime `libcuda` /
/// `libnvidia-encode`.
pub mod nvenc;
/// Cisco OpenH264 software H.264 encoder (BSD-licensed): the software H.264 encoder of a
/// build without `gpl`, and always built for the test suite.
#[cfg(any(feature = "openh264", test))]
pub mod oh264;
/// PNG watermark overlay composited onto frames before encoding.
pub mod overlay;
/// CPU-based striped H.264 (libx264 or OpenH264, by build) / JPEG encoder with per-stripe
/// change detection.
pub mod software;

pub use codec::*;

use std::ffi::{c_void, CString};
use std::sync::OnceLock;

use avcodec::{AvcodecEncoder, Backend, Input};
use nvenc::NvencEncoder;
use smithay::backend::allocator::dmabuf::Dmabuf;

use crate::RustCaptureSettings;

#[cfg(not(any(feature = "gpl", feature = "openh264")))]
compile_error!(
    "pixelflux needs a software H.264 encoder: enable the `gpl` feature (libx264, the default) or `openh264`."
);

/// The software H.264 encoder this build resolved to, fixed by the crate features: `"x264"`
/// whenever `gpl` is on (libx264 wins even if `openh264` is also enabled), `"openh264"` for
/// a GPL-free build. It is what the striped software path and the full-frame software fallback
/// under NVENC/VA-API both encode with.
#[cfg(feature = "gpl")]
pub const SOFTWARE_H264_ENCODER: &str = "x264";
#[cfg(not(feature = "gpl"))]
pub const SOFTWARE_H264_ENCODER: &str = "openh264";

/// Whether the build's software H.264 encoder carries a 4:4:4 (`video_fullcolor`) request:
/// libx264 does (High 4:4:4, full range); OpenH264 is 4:2:0-only and encodes such a request
/// 4:2:0.
pub const SOFTWARE_H264_FULLCOLOR: bool = cfg!(feature = "gpl");

/// A software encoder the build can run for one codec: the library's name as reported to
/// Python and the logs, and the libavcodec encoder that reaches it (empty for H.264, whose
/// software encoders are linked directly).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftwareEncoder {
    pub library: &'static str,
    pub avcodec: &'static str,
}

/// Whether the software encoder of a codec carries 4:4:4: x264 and x265 do, the rest are
/// 4:2:0 encoders.
pub fn software_fullcolor(codec: Codec) -> bool {
    match software_encoder(codec) {
        Some(enc) => matches!(enc.library, "x264" | "x265"),
        None => false,
    }
}

/// The software encoder this build runs for `codec`, or `None` when the linked FFmpeg
/// carries none of the encoders the codec is served by.
///
/// H.264 is resolved at build time (`SOFTWARE_H264_ENCODER`). The other codecs are probed
/// once against the linked libavcodec: HEVC through x265 (GPL, so only with the `gpl`
/// feature) or kvazaar, VP8 and VP9 through libvpx, AV1 through SVT-AV1. A build without
/// `gpl` never picks x265 even from a system FFmpeg that has it, keeping the GPL-free
/// posture the feature promises.
pub fn software_encoder(codec: Codec) -> Option<SoftwareEncoder> {
    static PROBED: OnceLock<[Option<SoftwareEncoder>; 5]> = OnceLock::new();
    if codec == Codec::Jpeg {
        return None;
    }
    let table = PROBED.get_or_init(|| {
        let h265: &[SoftwareEncoder] = if cfg!(feature = "gpl") {
            &[
                SoftwareEncoder { library: "x265", avcodec: "libx265" },
                SoftwareEncoder { library: "kvazaar", avcodec: "libkvazaar" },
            ]
        } else {
            &[SoftwareEncoder { library: "kvazaar", avcodec: "libkvazaar" }]
        };
        let first_linked = |candidates: &[SoftwareEncoder]| {
            candidates.iter().copied().find(|c| avcodec_has_encoder(c.avcodec))
        };
        [
            Some(SoftwareEncoder { library: SOFTWARE_H264_ENCODER, avcodec: "" }),
            first_linked(&[SoftwareEncoder { library: "libvpx", avcodec: "libvpx" }]),
            first_linked(&[SoftwareEncoder { library: "libvpx", avcodec: "libvpx-vp9" }]),
            first_linked(&[SoftwareEncoder { library: "svt-av1", avcodec: "libsvtav1" }]),
            first_linked(h265),
        ]
    });
    table[Codec::VIDEO.iter().position(|&c| c == codec).unwrap()]
}

/// Whether the linked libavcodec registers an encoder of this name.
fn avcodec_has_encoder(name: &str) -> bool {
    let Ok(name) = CString::new(name) else { return false };
    unsafe { !ffmpeg_sys_next::avcodec_find_encoder_by_name(name.as_ptr()).is_null() }
}

/// Damps visible quality "blinking": the number of consecutive frames a QP *increase* (a
/// quality drop under sustained motion) must be requested before a fixed-QP encoder commits
/// it. Moving the quantizer costs a codec re-open (the libavcodec encoders) or a full encoder
/// rebuild (OpenH264), and either forces a key frame, so acting on every transient increase —
/// and then reversing it as motion settles — would make the picture pulse. Quality
/// *increases* (a lower QP, e.g. a paint-over refresh) apply at once and never wait. Shared so
/// the fixed-QP encoders cannot disagree about how long a drop must persist.
pub(crate) const QP_HYSTERESIS_LIMIT: u32 = 60;

/// Size the CBR VBV/HRD buffer so rate control has enough slack to hold quality steady
/// without letting end-to-end latency drift upward.
///
/// The size is expressed as a multiple of one frame's bit budget (`bitrate_bps / fps`) rather
/// than a fixed byte count so a live bitrate or framerate change rescales the buffer with it,
/// preserving the same latency behavior at every operating point.
///
/// # Arguments
///
/// * `bitrate_bps` - Target bitrate in bits per second.
/// * `fps` - Target frames per second.
/// * `keyframe_interval_s` - Seconds between scheduled keyframes; `<= 0` for infinite GOP.
/// * `multiplier` - Explicit buffer multiplier; `<= 0` selects the policy default (1.5 on
///   infinite GOP, 3 when keyframe interval is active).
///
/// # Returns
///
/// VBV buffer size in bits, clamped to `[1, u32::MAX]`.
pub fn vbv_bits(bitrate_bps: u32, fps: f64, keyframe_interval_s: f64, multiplier: f64) -> u32 {
    let frame_bits = bitrate_bps as f64 / fps.max(1.0);
    let mult = if multiplier > 0.0 {
        multiplier
    } else if keyframe_interval_s > 0.0 {
        3.0
    } else {
        1.5
    };
    (frame_bits * mult).round().max(1.0).min(u32::MAX as f64) as u32
}

/// The `Colorspace:` field of a stream log line, from what the session negotiated rather than
/// what was asked for: a hardware encoder can refuse 4:4:4, and only the software encoder
/// carries it at full range. Shared so the X11 and Wayland logs describe an identical session
/// identically.
pub fn colorspace_desc(fullcolor: bool, software: bool) -> &'static str {
    match (fullcolor, software) {
        (true, true) => "I444 (Full Range)",
        (true, false) => "I444 (Limited Range)",
        _ => "I420 (Limited Range)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The build always serves H.264 in software, JPEG never, and whatever the linked FFmpeg
    /// carries for the rest is reported by a library name with a libavcodec encoder behind it;
    /// a GPL-free build never names x265.
    #[test]
    fn software_encoders_follow_the_build() {
        let h264 = software_encoder(Codec::H264).expect("H.264 is always served");
        assert_eq!(h264.library, SOFTWARE_H264_ENCODER);
        assert!(h264.avcodec.is_empty());
        assert_eq!(software_encoder(Codec::Jpeg), None);
        for codec in [Codec::H265, Codec::Vp8, Codec::Vp9, Codec::Av1] {
            if let Some(enc) = software_encoder(codec) {
                assert!(avcodec_has_encoder(enc.avcodec), "{codec:?}: {}", enc.avcodec);
                assert!(!enc.library.is_empty());
                if !cfg!(feature = "gpl") {
                    assert_ne!(enc.library, "x265");
                }
            }
        }
        assert_eq!(software_fullcolor(Codec::H264), SOFTWARE_H264_FULLCOLOR);
        assert!(!software_fullcolor(Codec::Vp8));
    }
}

/// One full-frame encoder session, whichever backend produced it, so the render and delivery
/// code passes "the frame encoder" around without caring which vendor path or library
/// produced the frames.
#[allow(clippy::large_enum_variant)]
pub enum FrameEncoder {
    Nvenc(NvencEncoder),
    Avcodec(AvcodecEncoder),
}

impl FrameEncoder {
    /// The codec the session emits.
    pub fn codec(&self) -> Codec {
        match self {
            FrameEncoder::Nvenc(enc) => enc.codec(),
            FrameEncoder::Avcodec(enc) => enc.codec(),
        }
    }

    /// Whether the session encodes on a GPU.
    pub fn is_hardware(&self) -> bool {
        match self {
            FrameEncoder::Nvenc(_) => true,
            FrameEncoder::Avcodec(enc) => enc.backend() == Backend::Vaapi,
        }
    }

    /// The backend as the logs name it: `NVENC`, `VAAPI`, or the software library.
    pub fn backend_name(&self) -> &'static str {
        match self {
            FrameEncoder::Nvenc(_) => "NVENC",
            FrameEncoder::Avcodec(enc) if enc.backend() == Backend::Vaapi => "VAAPI",
            FrameEncoder::Avcodec(enc) => enc.library(),
        }
    }

    /// Whether the session negotiated 4:4:4 chroma.
    pub fn is_fullcolor(&self) -> bool {
        match self {
            FrameEncoder::Nvenc(enc) => enc.is_fullcolor(),
            FrameEncoder::Avcodec(enc) => enc.is_fullcolor(),
        }
    }

    /// Apply a live bitrate / VBV / frame-rate change. `Err` means the session lost its codec
    /// context and has to be rebuilt.
    pub fn reconfigure_rate(&mut self, settings: &RustCaptureSettings) -> Result<(), String> {
        match self {
            FrameEncoder::Nvenc(enc) => {
                enc.reconfigure_rate(settings);
                Ok(())
            }
            FrameEncoder::Avcodec(enc) => enc.reconfigure_rate(settings),
        }
    }

    /// Encode one packed host frame (`stride` bytes per row; `rgba` names R,G,B,A byte order,
    /// otherwise B,G,R,A) at quantizer `qp`, as a key frame when `force_idr`.
    pub fn encode_host(
        &mut self,
        pixels: &[u8],
        stride: usize,
        rgba: bool,
        frame_number: u64,
        qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        match self {
            FrameEncoder::Nvenc(enc) => enc.encode_cpu_packed(pixels, stride, rgba, frame_number, qp, force_idr),
            FrameEncoder::Avcodec(enc) => enc.encode_host(pixels, stride, frame_number, qp, force_idr),
        }
    }

    /// Encode one Wayland dmabuf in place (a zero-copy session).
    pub fn encode_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        frame_number: u64,
        qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        match self {
            FrameEncoder::Nvenc(enc) => enc.encode(dmabuf, frame_number, qp, force_idr),
            FrameEncoder::Avcodec(enc) => enc.encode_dmabuf(dmabuf, frame_number, qp, force_idr),
        }
    }
}

/// How a session's frames arrive: Wayland dmabufs (with the EGL display NVENC imports them
/// through), or packed host pixels in R,G,B,A (`rgba`) or B,G,R,A byte order.
#[derive(Clone, Copy)]
pub enum FrameSource {
    Dmabuf { egl_display: *const c_void },
    Host { rgba: bool },
}

/// Choose and build the full-frame encoder of a session, or `None` when the striped software
/// path serves it. One ladder for X11, Wayland zero-copy and Wayland readback, so the three
/// cannot pick differently for the same settings:
///
/// 1. Unless software encoding is forced (`use_cpu`, or encode node `-1`), the hardware
///    backend the encode node's driver selects — NVENC on the NVIDIA driver, VA-API otherwise.
///    A compatible NVENC session handed over in `prior` is reconfigured in place instead of
///    rebuilt. A hardware refusal is logged and falls through.
/// 2. A dmabuf source stops here: software cannot read dmabufs, and the caller's readback path
///    then runs this ladder again with host frames.
/// 3. The software encoder of the codec, except JPEG and H.264, whose software path is the
///    striped one. A codec this build has no software encoder for demotes the session to
///    H.264, rewriting `settings.codec`, so the stream comes up rather than staying black.
pub fn select_frame_encoder(
    settings: &mut RustCaptureSettings,
    source: FrameSource,
    prior: Option<FrameEncoder>,
    tag: &str,
) -> Option<FrameEncoder> {
    let codec = settings.codec;
    if !codec.is_video() {
        return None;
    }
    let software_forced = settings.use_cpu || settings.encode_node_index == -1;
    if !software_forced {
        let node = settings.encode_node_index.max(0);
        let driver = crate::get_gpu_driver(node);
        println!("[{tag}] Encode Node Index: {node} | Driver: {driver}");
        if crate::driver_selects_nvenc(&driver) {
            if let Some(FrameEncoder::Nvenc(mut enc)) = prior {
                match enc.reconfigure_resolution(settings) {
                    Ok(()) => {
                        println!("[{tag}] NVENC session reconfigured in place.");
                        return Some(FrameEncoder::Nvenc(enc));
                    }
                    Err(e) => eprintln!("[{tag}] NVENC in-place reconfigure unavailable ({e}); rebuilding."),
                }
            }
            let egl_display = match source {
                FrameSource::Dmabuf { egl_display } => egl_display,
                FrameSource::Host { .. } => std::ptr::null(),
            };
            match NvencEncoder::new(settings, egl_display) {
                Ok(enc) => {
                    println!("[{tag}] NVENC {} encoder initialized.", codec.display());
                    return Some(FrameEncoder::Nvenc(enc));
                }
                Err(e) => eprintln!("[{tag}] Failed to init NVENC {}: {e}", codec.display()),
            }
        } else {
            let input = match source {
                FrameSource::Dmabuf { .. } => Input::Dmabuf,
                FrameSource::Host { rgba } => Input::Host { rgba },
            };
            match AvcodecEncoder::new(settings, codec, Backend::Vaapi, input) {
                Ok(enc) => {
                    println!(
                        "[{tag}] VAAPI {} encoder initialized ({}).",
                        codec.display(),
                        if enc.is_fullcolor() { "4:4:4" } else { "4:2:0" }
                    );
                    return Some(FrameEncoder::Avcodec(enc));
                }
                Err(e) => eprintln!("[{tag}] Failed to init VAAPI {}: {e}", codec.display()),
            }
        }
    } else {
        println!("[{tag}] Software encoding selected (use_cpu=true or encode_node_index=-1).");
    }
    let FrameSource::Host { rgba } = source else {
        return None;
    };
    if codec == Codec::H264 {
        println!("[{tag}] Software H.264 ({}).", SOFTWARE_H264_ENCODER);
        return None;
    }
    match AvcodecEncoder::new(settings, codec, Backend::Software, Input::Host { rgba }) {
        Ok(enc) => {
            println!("[{tag}] Software {} ({}).", codec.display(), enc.library());
            Some(FrameEncoder::Avcodec(enc))
        }
        Err(e) => {
            eprintln!(
                "[{tag}] No {} encoder available: {e}. Encoding H.264 ({}) instead.",
                codec.display(),
                SOFTWARE_H264_ENCODER
            );
            settings.codec = Codec::H264;
            None
        }
    }
}

/// The chroma format a session actually carries, which is not always the one requested: a
/// hardware session only when the device carries it, the software path only when the build's
/// encoder for the codec does. Every consumer of "is this stream 4:4:4" reads it from here.
pub fn session_fullcolor(encoder: Option<&FrameEncoder>, settings: &RustCaptureSettings) -> bool {
    match encoder {
        Some(enc) => enc.is_fullcolor(),
        None => settings.video_fullcolor && software_fullcolor(settings.codec),
    }
}
