/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Encoder backends and what they share: the codec identities and wire framing, the
//! rate-control policy, which software encoder a build resolves each codec to, and which
//! hardware encoder a render node serves each with.

/// libavcodec-backed encoders: VA-API hardware sessions on a DRM render node, and the
/// software HEVC / VP8 / VP9 / AV1 encoders the linked FFmpeg carries.
pub mod avcodec;
/// Codec identities, wire framing, quantizer domains, level ladders, bitstream reads.
pub mod codec;
/// Tegra hardware video encoding through the vendor V4L2 encoder, loaded at runtime: the only path to a
/// Jetson's encoder, which carries no `libnvidia-encode` and no render node driver. Built for
/// `aarch64` alone — the vendor libraries and the encoder behind them exist on no other
/// architecture, so an x86_64 build carries none of this.
#[cfg(target_arch = "aarch64")]
pub mod tegra;
/// NVIDIA NVENC hardware H.264 / HEVC / AV1 encoder loaded via runtime `libcuda` /
/// `libnvidia-encode`.
pub mod nvenc;
/// Cisco OpenH264 software H.264 encoder (BSD-licensed): the software H.264 encoder of a
/// build without `gpl`, and always built for the test suite.
#[cfg(any(feature = "openh264", test))]
pub mod oh264;
/// PNG watermark overlay composited onto frames before encoding.
pub mod overlay;
pub mod reference;
/// CPU-based striped H.264 (libx264 or OpenH264, by build) / JPEG encoder with per-stripe
/// change detection.
pub mod software;
/// The colour an H.264 stream declares: read from a sequence parameter set, and written into
/// one for a device that converts without saying what it converted with.
pub mod sps;
/// Hardware H.264 through a generic stateful V4L2 M2M encoder: boards whose encoder sits
/// behind the kernel's own interface rather than a vendor library or a render node, such as
/// a Raspberry Pi 4, RK356x or i.MX8M. Built everywhere, since the interface is the kernel's.
pub mod v4l2m2m;

pub use codec::*;

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::{Mutex, OnceLock};

use avcodec::{AvcodecEncoder, Backend, Input};
use nvenc::NvencEncoder;
use smithay::backend::allocator::dmabuf::Dmabuf;

use crate::RustCaptureSettings;

#[cfg(not(any(feature = "gpl", feature = "openh264")))]
compile_error!(
    "pixelflux needs a software H.264 encoder: enable the `gpl` feature (libx264, the default) or `openh264`."
);

/// A software encoder the build can run for one codec: the library's name as reported to
/// Python and the logs, and the libavcodec encoder that reaches it (empty for H.264, whose
/// software encoders are linked directly).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftwareEncoder {
    pub library: &'static str,
    pub avcodec: &'static str,
}

/// The codecs a render node encodes in hardware, each with the backend's name.
/// Each video codec an engine on a node encodes, the backend's name, and whether the engine
/// takes a `video_fullcolor` session as 4:4:4 rather than 4:2:0.
pub type HardwareEncoders = Vec<(Codec, &'static str, bool)>;

/// The hardware backend that serves each video codec on an encode node, as the name a
/// session logs it in lower case (`"nvenc"`, `"vaapi"` or `"tegra"`), probed once per node and
/// remembered for the life of the process: the ladder picks the backend by the node's
/// driver exactly as `select_frame_encoder` does, and that backend lists the codecs its
/// device has an engine for (`nvenc::probe_codecs`, `avcodec::probe_codecs`). A node whose
/// backend cannot be brought up serves nothing, said once in the log, so a caller offers
/// the codec only where a session would come up on hardware rather than demote. What a
/// session is then refused for (a size past the engine's maximum, a 4:4:4 the engine lacks)
/// is still the ladder's to fall through on.
pub fn hardware_encoders(encode_node_index: i32) -> HardwareEncoders {
    static PROBED: OnceLock<Mutex<HashMap<i32, HardwareEncoders>>> = OnceLock::new();
    let node = encode_node_index.max(0);
    let mut probed = PROBED.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(served) = probed.get(&node) {
        return served.clone();
    }
    #[cfg(target_arch = "aarch64")]
    if tegra::available() {
        let served: HardwareEncoders = tegra::served().into_iter().map(|c| (c, "tegra", false)).collect();
        let names: Vec<&str> = served.iter().map(|(c, ..)| c.display()).collect();
        println!("[pixelflux] Render node {node} encodes {} on tegra.", names.join(", "));
        probed.insert(node, served.clone());
        return served;
    }
    let driver = crate::get_gpu_driver(node);
    let (backend, codecs) = if crate::driver_selects_nvenc(&driver) {
        ("nvenc", nvenc::probe_codecs(node))
    } else {
        ("vaapi", avcodec::probe_codecs(node))
    };
    let served: HardwareEncoders = match codecs {
        Ok(codecs) => codecs.into_iter().map(|(codec, fullcolor)| (codec, backend, fullcolor)).collect(),
        Err(e) => {
            eprintln!("[pixelflux] No hardware encoder on render node {node} ({backend}): {e}");
            Vec::new()
        }
    };
    if served.is_empty() && v4l2m2m::available() {
        // The node index names nothing here: an M2M encoder is not a render node, and the
        // answer is the same whichever index was asked about. It is cached under the key all
        // the same, so a caller asking twice is answered from the same probe.
        let codecs = v4l2m2m::served();
        let served: HardwareEncoders = codecs.iter().map(|&c| (c, "v4l2m2m", false)).collect();
        let names: Vec<&str> = codecs.iter().map(|c| c.display()).collect();
        println!("[pixelflux] A stateful V4L2 M2M encoder serves {}.", names.join(", "));
        probed.insert(node, served.clone());
        return served;
    }
    if !served.is_empty() {
        let names: Vec<&str> = served.iter().map(|(codec, ..)| codec.display()).collect();
        println!("[pixelflux] Render node {node} encodes {} on {backend}.", names.join(", "));
    }
    probed.insert(node, served.clone());
    served
}

/// Whether the software encoder of a codec carries a 4:4:4 (`video_fullcolor`) request: x264
/// (High 4:4:4, full range), x265 and libvpx's VP9 (profile 1) do; OpenH264, kvazaar, VP8 and
/// SVT-AV1 encode such a request 4:2:0.
pub fn software_fullcolor(codec: Codec) -> bool {
    match software_encoder(codec) {
        Some(enc) => matches!(enc.library, "x264" | "x265") || (codec == Codec::Vp9 && enc.library == "libvpx"),
        None => false,
    }
}

/// The name of the software encoder this build runs for `codec`, for logs; `"none"` where the
/// linked FFmpeg carries none.
pub fn software_library(codec: Codec) -> &'static str {
    software_encoder(codec).map_or("none", |enc| enc.library)
}

/// The software encoder this build runs for `codec` on this machine, or `None` when the linked
/// FFmpeg carries none of the encoders the codec is served by that run here.
///
/// H.264 is fixed by the crate features: libx264 whenever `gpl` is on (it wins even if
/// `openh264` is also enabled), Cisco OpenH264 for a GPL-free build; it is what the striped
/// software path and the full-frame software fallback under NVENC/VA-API both encode with.
/// The other codecs are probed once against the linked libavcodec: HEVC through x265 (GPL,
/// so only with the `gpl` feature) or kvazaar, VP8 and VP9 through libvpx, AV1 through
/// SVT-AV1, each opened on a frame in a forked child first, so an encoder that takes its
/// process down on this machine is one the build does not carry here and the next candidate
/// is tried. A build without `gpl` never picks x265 even from a system FFmpeg that has it,
/// keeping the GPL-free posture the feature promises.
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
        let first_linked = |codec: Codec, candidates: &[SoftwareEncoder]| {
            candidates.iter().copied().find(|c| avcodec_has_encoder(c.avcodec) && encodes_in_child(codec, *c))
        };
        [
            Some(SoftwareEncoder {
                library: if cfg!(feature = "gpl") { "x264" } else { "openh264" },
                avcodec: "",
            }),
            first_linked(Codec::Vp8, &[SoftwareEncoder { library: "libvpx", avcodec: "libvpx" }]),
            first_linked(Codec::Vp9, &[SoftwareEncoder { library: "libvpx", avcodec: "libvpx-vp9" }]),
            first_linked(Codec::Av1, &[SoftwareEncoder { library: "svt-av1", avcodec: "libsvtav1" }]),
            first_linked(Codec::H265, h265),
        ]
    });
    table[Codec::VIDEO.iter().position(|&c| c == codec).unwrap()]
}

/// Whether the linked libavcodec registers an encoder of this name.
fn avcodec_has_encoder(name: &str) -> bool {
    let Ok(name) = CString::new(name) else { return false };
    unsafe { !ffmpeg_sys_next::avcodec_find_encoder_by_name(name.as_ptr()).is_null() }
}

/// Whether opening `enc` for `codec` on a small frame and encoding one leaves a process alive,
/// tried in a forked child: a library that faults on this machine, with an instruction the CPU
/// lacks or a register the kernel does not emulate, takes the child down and not a session.
fn encodes_in_child(codec: Codec, enc: SoftwareEncoder) -> bool {
    survives_in_child(|| {
        // The child converts on a pool of its own: the parent's rayon workers do not exist in it.
        let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(1).build() else { return };
        pool.install(|| {
            let settings = RustCaptureSettings { width: 256, height: 128, ..Default::default() };
            let Ok(mut encoder) = AvcodecEncoder::open(
                &settings, codec, Backend::Software, enc.library, enc.avcodec, Input::Host { rgba: false },
            ) else {
                return;
            };
            let frame = vec![0u8; 256 * 128 * 4];
            for n in 0..64 {
                let packet = encoder.encode_host(&frame, 256 * 4, n, settings.video_crf as u32, n == 0);
                if !matches!(packet, Ok(p) if p.is_empty()) {
                    break;
                }
            }
        })
    })
}

/// Whether a forked child outlives `f`: a fatal signal in it is the finding, an exit is not, and
/// neither reaches the caller. The child has a minute, and one that neither returns nor dies
/// in it has stalled, which is as final. One a sandbox refuses to fork counts as alive.
fn survives_in_child(f: impl FnOnce()) -> bool {
    match unsafe { libc::fork() } {
        0 => {
            unsafe {
                libc::signal(libc::SIGALRM, libc::SIG_DFL);
                libc::alarm(60);
            }
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            unsafe { libc::_exit(0) }
        }
        pid if pid > 0 => {
            let mut status = 0;
            while unsafe { libc::waitpid(pid, &mut status, 0) } < 0 {
                if std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
                    return true;
                }
            }
            !libc::WIFSIGNALED(status)
        }
        _ => true,
    }
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
/// The 1.5-frame default holds for hardware rate control too: a one-frame buffer measured on
/// NVENC (`gpu_bench_cbr_policy`, a V100) ran an 8 Mbit/s H.264 session at 13.3 to 13.7 Mbit/s
/// on scene cuts at the same PSNR, made the frame after a single cut larger (408 against 288
/// kbit) and bought nothing on steady content. The overshoot is the quarter-resolution first
/// pass misjudging a scene cut with no buffer left to absorb the miss
/// (`gpu_bench_cbr_rate_control`): full-resolution two-pass holds a one-frame buffer at
/// 1.5 ms more per frame, single-pass halves the overshoot at a lower PSNR, and the GOP target
/// and quantizer ceiling change nothing. HEVC holds either buffer.
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
/// what was asked for: a hardware encoder can refuse 4:4:4, software 4:4:4 of x264's kind
/// carries full range, and a device converting in fixed function picks the range itself.
/// Shared so the X11 and Wayland logs describe an identical session identically.
pub fn colorspace_desc(fullcolor: bool, full_range: bool) -> &'static str {
    match (fullcolor, full_range) {
        (true, true) => "I444 (Full Range)",
        (true, false) => "I444 (Limited Range)",
        (false, true) => "I420 (Full Range)",
        (false, false) => "I420 (Limited Range)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The build always serves H.264 in software, JPEG never, and whatever the linked FFmpeg
    /// carries for the rest is reported by a library name with a libavcodec encoder behind it;
    /// a GPL-free build never names x265.
    /// A session lands on the codec it asked for, or on the next video codec this host serves;
    /// JPEG is the ladder's last leg, not its answer to one missing encoder.
    #[test]
    fn a_session_lands_on_a_video_codec_the_host_serves() {
        let mut settings = RustCaptureSettings {
            width: 256,
            height: 128,
            codec: Codec::Av1,
            use_cpu: true,
            ..Default::default()
        };
        let encoder = select_frame_encoder(&mut settings, FrameSource::Host { rgba: false }, None, "test");
        println!("asked for av1, landed on {}", settings.codec.display());
        assert_ne!(settings.codec, Codec::Jpeg, "this build encodes video in software");
        assert!(software_encoder(settings.codec).is_some());
        // H.264 comes back as the caller's own striped path rather than a full-frame session.
        assert!(encoder.is_some() || settings.codec == Codec::H264);
    }

    /// A codec with no path on this host falls through the video codecs it does serve, the
    /// encode node's hardware ones first and most efficient first, the software ones by encode
    /// time, with software H.264 among them for a full-frame session alone.
    #[test]
    fn a_codec_without_a_path_falls_through_the_served_video_codecs() {
        let software: Vec<Codec> = SOFTWARE_ORDER
            .iter()
            .copied()
            .filter(|&codec| {
                codec != Codec::Av1 && codec != Codec::H264 && software_encoder(codec).is_some()
            })
            .collect();
        assert_eq!(fallback_codecs(Codec::Av1, &[], false), software);
        let mut h264_first = vec![Codec::H264];
        h264_first.extend(software.iter().copied());
        assert_eq!(fallback_codecs(Codec::Av1, &[], true), h264_first, "one x264 session leads the software rungs");
        assert_eq!(fallback_codecs(Codec::Av1, &[Codec::H264], false), h264_first, "an engine's H.264 leads either way");
        assert!(!fallback_codecs(Codec::Av1, &[Codec::Av1], true).contains(&Codec::Av1));
        // An engine's codecs go most efficient first, whatever the build encodes in software.
        let engine = [Codec::H264, Codec::H265, Codec::Av1];
        assert_eq!(&fallback_codecs(Codec::Vp8, &engine, true)[..3], &[Codec::Av1, Codec::H265, Codec::H264]);
    }

    /// The software encoders that take a 4:4:4 session: x264 and x265, and libvpx for VP9 alone;
    /// AV1 and VP8 encode 4:2:0 whatever is asked.
    #[test]
    fn software_fullcolor_follows_the_library() {
        assert_eq!(software_fullcolor(Codec::H264), software_library(Codec::H264) == "x264");
        assert_eq!(software_fullcolor(Codec::H265), software_library(Codec::H265) == "x265");
        assert_eq!(software_fullcolor(Codec::Vp9), software_library(Codec::Vp9) == "libvpx");
        assert!(!software_fullcolor(Codec::Av1) && !software_fullcolor(Codec::Vp8));
    }

    /// A child's fate is the finding: a fatal signal, and only that, reads as not surviving.
    #[test]
    fn a_child_survives_unless_a_signal_takes_it() {
        assert!(survives_in_child(|| {}));
        assert!(survives_in_child(|| unsafe { libc::_exit(3) }));
        assert!(!survives_in_child(|| unsafe {
            libc::raise(libc::SIGILL);
        }));
    }

    #[test]
    fn software_encoders_follow_the_build() {
        use rayon::prelude::*;
        let _: u32 = (0..8u32).into_par_iter().sum();
        let h264 = software_encoder(Codec::H264).expect("H.264 is always served");
        assert_eq!(h264.library, if cfg!(feature = "gpl") { "x264" } else { "openh264" });
        assert_eq!(software_library(Codec::H264), h264.library);
        assert_eq!(software_library(Codec::Jpeg), "none");
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
        assert_eq!(software_fullcolor(Codec::H264), cfg!(feature = "gpl"));
        assert!(!software_fullcolor(Codec::Vp8));
        assert_eq!(software_fullcolor(Codec::Vp9), software_encoder(Codec::Vp9).is_some());
    }

    /// A session's range is the session's to report, not something read off whether its
    /// encoder is hardware. The two part on a striped 4:2:0 software session, which is
    /// limited range while the encoder is software, so a description taking the second for
    /// the first names a range the bitstream does not carry. All four pairings have a name
    /// of their own for that reason, rather than 4:2:0 falling through to one.
    #[test]
    fn a_session_reports_the_range_it_converted_at() {
        assert_eq!(colorspace_desc(true, true), "I444 (Full Range)");
        assert_eq!(colorspace_desc(true, false), "I444 (Limited Range)");
        assert_eq!(colorspace_desc(false, true), "I420 (Full Range)");
        assert_eq!(colorspace_desc(false, false), "I420 (Limited Range)");

        let mut settings = RustCaptureSettings { codec: Codec::H264, video_fullcolor: false, ..Default::default() };
        assert!(!session_full_range(None, &settings), "striped 4:2:0 converts at limited range");
        assert_eq!(colorspace_desc(session_fullcolor(None, &settings),
                                   session_full_range(None, &settings)),
                   "I420 (Limited Range)");

        settings.video_fullcolor = true;
        assert_eq!(session_full_range(None, &settings), software_fullcolor(Codec::H264),
                   "striped 4:4:4 converts at full range where the build carries it");
    }
}

/// One full-frame encoder session, whichever backend produced it, so the render and delivery
/// code passes "the frame encoder" around without caring which vendor path or library
/// produced the frames.
#[allow(clippy::large_enum_variant)]
pub enum FrameEncoder {
    Nvenc(NvencEncoder),
    Avcodec(AvcodecEncoder),
    /// Tegra's encoder, reached through the vendor V4L2 library.
    #[cfg(target_arch = "aarch64")]
    Tegra(tegra::TegraEncoder),
    /// A stateful V4L2 M2M encoder, driven through the kernel interface directly.
    V4l2m2m(v4l2m2m::V4l2M2mEncoder),
}

impl FrameEncoder {
    /// The codec the session emits.
    pub fn codec(&self) -> Codec {
        match self {
            FrameEncoder::Nvenc(enc) => enc.codec(),
            FrameEncoder::Avcodec(enc) => enc.codec(),
            #[cfg(target_arch = "aarch64")]
            FrameEncoder::Tegra(enc) => enc.codec(),
            FrameEncoder::V4l2m2m(enc) => enc.codec(),
        }
    }

    /// Whether the session encodes on a GPU.
    pub fn is_hardware(&self) -> bool {
        match self {
            FrameEncoder::Nvenc(_) => true,
            FrameEncoder::Avcodec(enc) => enc.backend() == Backend::Vaapi,
            #[cfg(target_arch = "aarch64")]
            FrameEncoder::Tegra(_) => true,
            FrameEncoder::V4l2m2m(_) => true,
        }
    }

    /// The backend as the logs name it: `NVENC`, `VAAPI`, or the software library.
    pub fn backend_name(&self) -> &'static str {
        match self {
            FrameEncoder::Nvenc(_) => "NVENC",
            FrameEncoder::Avcodec(enc) if enc.backend() == Backend::Vaapi => "VAAPI",
            FrameEncoder::Avcodec(enc) => enc.library(),
            #[cfg(target_arch = "aarch64")]
            FrameEncoder::Tegra(_) => "TEGRA",
            FrameEncoder::V4l2m2m(_) => "V4L2M2M",
        }
    }

    /// Whether the session negotiated 4:4:4 chroma.
    pub fn is_fullcolor(&self) -> bool {
        match self {
            FrameEncoder::Nvenc(enc) => enc.is_fullcolor(),
            FrameEncoder::Avcodec(enc) => enc.is_fullcolor(),
            #[cfg(target_arch = "aarch64")]
            FrameEncoder::Tegra(enc) => enc.is_fullcolor(),
            FrameEncoder::V4l2m2m(enc) => enc.is_fullcolor(),
        }
    }

    /// Whether the session signals full range: a software 4:4:4 of x264's kind, and the V4L2
    /// M2M sessions whose firmware converts at full range and offers no way to ask for another.
    pub fn is_full_range(&self) -> bool {
        match self {
            FrameEncoder::Nvenc(_) => false,
            FrameEncoder::Avcodec(enc) => enc.is_full_range(),
            #[cfg(target_arch = "aarch64")]
            FrameEncoder::Tegra(_) => false,
            FrameEncoder::V4l2m2m(enc) => enc.is_full_range(),
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
            #[cfg(target_arch = "aarch64")]
            FrameEncoder::Tegra(enc) => enc.reconfigure_rate(settings),
            FrameEncoder::V4l2m2m(enc) => enc.reconfigure_rate(settings),
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
            #[cfg(target_arch = "aarch64")]
            FrameEncoder::Tegra(enc) => enc.encode_host(pixels, stride, rgba, frame_number, qp, force_idr),
            FrameEncoder::V4l2m2m(enc) => enc.encode_host(pixels, stride, rgba, frame_number, qp, force_idr),
        }
    }

    /// The frame the last delivered frame predicted from.
    pub fn last_reference(&self) -> reference::Reference {
        match self {
            FrameEncoder::Nvenc(enc) => enc.last_reference(),
            FrameEncoder::Avcodec(_) => reference::Reference::Untracked,
            #[cfg(target_arch = "aarch64")]
            FrameEncoder::Tegra(_) => reference::Reference::Untracked,
            FrameEncoder::V4l2m2m(_) => reference::Reference::Untracked,
        }
    }

    /// Leave frame `frame_id` and every frame after it out of the predictions, so the next
    /// frame decodes for a client that lost it. False when the session cannot, and the caller
    /// codes a key frame instead.
    pub fn invalidate_reference(&mut self, frame_id: u16) -> bool {
        match self {
            FrameEncoder::Nvenc(enc) => enc.invalidate_reference(frame_id),
            FrameEncoder::Avcodec(_) => false,
            #[cfg(target_arch = "aarch64")]
            FrameEncoder::Tegra(_) => false,
            FrameEncoder::V4l2m2m(_) => false,
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
            #[cfg(target_arch = "aarch64")]
            FrameEncoder::Tegra(_) => Err("the Tegra session takes host frames".into()),
            FrameEncoder::V4l2m2m(_) => Err("the V4L2 M2M session takes host frames".into()),
        }
    }
}

/// The chroma sampling of a session as the logs name it.
pub fn chroma_name(fullcolor: bool) -> &'static str {
    if fullcolor { "4:4:4" } else { "4:2:0" }
}

/// The kernel driver's name out of the sysfs link `get_gpu_driver` read, `unknown` without one.
pub fn driver_name(driver: &str) -> &str {
    match driver.rsplit('/').next() {
        Some(name) if !name.is_empty() => name,
        _ => "unknown",
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
///    backend the encode node's driver selects — NVENC on the NVIDIA driver, VA-API otherwise,
///    and on a Jetson the vendor V4L2 encoder before either, since that board publishes no
///    render node driver to select on. A compatible NVENC session handed over in `prior` is
///    reconfigured in place instead of rebuilt. A hardware refusal is logged and falls through.
/// 2. A dmabuf source stops here: software cannot read dmabufs, and the caller's readback path
///    then runs this ladder again with host frames.
/// 3. The software encoder of the codec, except JPEG and H.264, whose software path is the
///    striped one.
/// 4. Where the codec has no path at all, the other video codecs this host serves, the encode
///    node's hardware ones before the build's software ones, software H.264 among the latter
///    for a full-frame session, then the striped H.264 path, and JPEG only past all of them.
///    `settings.codec` names what came up.
pub fn select_frame_encoder(
    settings: &mut RustCaptureSettings,
    source: FrameSource,
    prior: Option<FrameEncoder>,
    tag: &str,
) -> Option<FrameEncoder> {
    let requested = settings.codec;
    if !requested.is_video() {
        return None;
    }
    if let Some(enc) = select_for_codec(settings, source, prior, tag) {
        return Some(enc);
    }
    // A dmabuf source stops here, its readback path running the ladder again on host frames,
    // and H.264 has come up on the striped path the caller encodes itself.
    let FrameSource::Host { rgba } = source else {
        return None;
    };
    if requested == Codec::H264 {
        return None;
    }
    let hardware: Vec<Codec> = if settings.use_cpu || settings.encode_node_index == -1 {
        Vec::new()
    } else {
        hardware_encoders(settings.encode_node_index).into_iter().map(|(codec, ..)| codec).collect()
    };
    eprintln!("[{tag}] No {} path on this host; trying the video codecs it serves.", requested.display());
    for codec in fallback_codecs(requested, &hardware, settings.video_fullframe) {
        settings.codec = codec;
        if let Some(enc) = select_for_codec(settings, source, None, tag) {
            return Some(enc);
        }
        // Software H.264 is the caller's own path, one x264 session for a full-frame capture.
        if codec == Codec::H264 && settings.video_fullframe && software_encoder(Codec::H264).is_some() {
            return None;
        }
    }
    // No full-frame codec came up: H.264's software path, the striped one, is the last video
    // rung, and JPEG the leg past it.
    settings.codec = Codec::H264;
    if software_encoder(Codec::H264).is_some() {
        return software_fallback(settings, rgba, tag);
    }
    eprintln!("[{tag}] No video encoder on this host. Encoding JPEG instead.");
    settings.codec = Codec::Jpeg;
    None
}

/// The video codecs by compression efficiency: the order a fallthrough tries the ones an engine
/// on the encode node carries, where CPU is no constraint.
const HARDWARE_ORDER: [Codec; 5] = [Codec::Av1, Codec::H265, Codec::Vp9, Codec::H264, Codec::Vp8];

/// The video codecs by the measured time per frame of their software encoders, x264 to libvpx
/// VP9: the order a fallthrough tries the ones the build encodes in software.
const SOFTWARE_ORDER: [Codec; 5] = [Codec::H264, Codec::Av1, Codec::Vp8, Codec::H265, Codec::Vp9];

/// The video codecs a capture falls through to where the one it asked for has no path on this
/// host: those an engine on the encode node carries, named in `hardware`, in `HARDWARE_ORDER`,
/// then those the build encodes in software, in `SOFTWARE_ORDER`. Software H.264 joins a
/// full-frame session, where it is one x264 session; otherwise its software path is the
/// striped one, which the ladder reaches only past every full-frame rung.
fn fallback_codecs(requested: Codec, hardware: &[Codec], fullframe: bool) -> Vec<Codec> {
    let mut codecs: Vec<Codec> = HARDWARE_ORDER
        .iter()
        .copied()
        .filter(|&codec| codec != requested && hardware.contains(&codec))
        .collect();
    codecs.extend(SOFTWARE_ORDER.iter().copied().filter(|&codec| {
        codec != requested
            && (codec != Codec::H264 || fullframe)
            && !hardware.contains(&codec)
            && software_encoder(codec).is_some()
    }));
    codecs
}

/// The ladder for the one codec `settings` names; `None` where no backend of it opened, and for
/// H.264 on host frames, whose software path the caller encodes itself.
fn select_for_codec(
    settings: &mut RustCaptureSettings,
    source: FrameSource,
    prior: Option<FrameEncoder>,
    tag: &str,
) -> Option<FrameEncoder> {
    let codec = settings.codec;
    let software_forced = settings.use_cpu || settings.encode_node_index == -1;
    #[cfg(target_arch = "aarch64")]
    if let (false, Some(_), FrameSource::Host { rgba }) =
        (software_forced, tegra::coded_fourcc(codec), source)
    {
        // Tegra publishes no render node driver to probe and carries no libnvidia-encode, so the
        // vendor library is the only way to its encoder and this step comes before both; a
        // codec it does not serve, or refuses, goes straight to software, past the two backends
        // the board lacks.
        if tegra::available() {
            drop(prior);
            if tegra::served().contains(&codec) {
                match tegra::TegraEncoder::new(codec, settings, rgba) {
                    Ok(enc) => {
                        println!(
                            "[{tag}] Encoder: TEGRA {} {} on the vendor V4L2 encoder.",
                            codec.display(),
                            chroma_name(enc.is_fullcolor())
                        );
                        return Some(FrameEncoder::Tegra(enc));
                    }
                    Err(e) => {
                        eprintln!("[{tag}] Failed to init the Tegra encoder: {e}");
                        crate::report::encoder_reason(&format!("Tegra {} did not open: {e}", codec.display()));
                    }
                }
            } else {
                crate::report::encoder_reason(&format!("the Tegra encoder has no {} engine", codec.display()));
            }
            return software_fallback(settings, rgba, tag);
        }
    }
    if !software_forced {
        let node = settings.encode_node_index.max(0);
        let driver = crate::get_gpu_driver(node);
        crate::log::debug!("[{tag}] Encode node {node}, driver {driver}.");
        if crate::driver_selects_nvenc(&driver) {
            if let Some(FrameEncoder::Nvenc(mut enc)) = prior {
                match enc.reconfigure_resolution(settings) {
                    Ok(resized) => {
                        if resized {
                            crate::log::debug!("[{tag}] NVENC session reconfigured in place.");
                        }
                        crate::report::hardware_encoder(enc.device_name(), driver_name(&driver), node);
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
                    println!(
                        "[{tag}] Encoder: NVENC {} {} on {} (render node {node}, {} driver).",
                        codec.display(),
                        chroma_name(enc.is_fullcolor()),
                        enc.device_name(),
                        driver_name(&driver)
                    );
                    crate::report::hardware_encoder(enc.device_name(), driver_name(&driver), node);
                    return Some(FrameEncoder::Nvenc(enc));
                }
                Err(e) => {
                    eprintln!("[{tag}] Failed to init NVENC {}: {e}", codec.display());
                    crate::report::encoder_reason(&format!("NVENC {} did not open: {e}", codec.display()));
                }
            }
        } else {
            // Nothing below reconfigures a session in place, so the previous one is released
            // here: a single-context M2M node refuses to open while its own last session is
            // alive, and a window resize would then demote a working hardware path.
            drop(prior);
            let input = match source {
                FrameSource::Dmabuf { .. } => Input::Dmabuf,
                FrameSource::Host { rgba } => Input::Host { rgba },
            };
            match AvcodecEncoder::new(settings, codec, Backend::Vaapi, input) {
                Ok(enc) => {
                    println!(
                        "[{tag}] Encoder: VAAPI {} {} on {} surfaces (render node {node}, {} driver).",
                        codec.display(),
                        chroma_name(enc.is_fullcolor()),
                        enc.sw_format_name(),
                        driver_name(&driver)
                    );
                    crate::report::hardware_encoder("", driver_name(&driver), node);
                    return Some(FrameEncoder::Avcodec(enc));
                }
                Err(e) => {
                    eprintln!("[{tag}] Failed to init VAAPI {}: {e}", codec.display());
                    crate::report::encoder_reason(&format!("VAAPI {} did not open: {e}", codec.display()));
                }
            }
        }
        // A stateful M2M encoder publishes no render node driver to select on, so it is
        // reached only once the two backends that do have refused. The size is checked before
        // the node is opened: a refusal here falls through to software, a refusal later would
        // leave a session that came up and produces nothing.
        if let (Some(_), FrameSource::Host { rgba }) = (v4l2m2m::coded_fourcc(codec), source)
            && v4l2m2m::encodes(codec, settings.width, settings.height)
        {
            match v4l2m2m::V4l2M2mEncoder::new(codec, settings, rgba) {
                Ok(enc) => {
                    println!(
                        "[{tag}] Encoder: V4L2M2M {} {} on a stateful M2M node.",
                        codec.display(),
                        chroma_name(enc.is_fullcolor())
                    );
                    return Some(FrameEncoder::V4l2m2m(enc));
                }
                Err(e) => eprintln!("[{tag}] Failed to init the V4L2 M2M encoder: {e}"),
            }
        }
    } else {
        crate::log::debug!("[{tag}] Software encoding selected (use_cpu or encode_node_index -1).");
        crate::report::encoder_reason("software encoding selected");
    }
    let FrameSource::Host { rgba } = source else {
        return None;
    };
    software_fallback(settings, rgba, tag)
}

/// The codec's software encoder; `None` for H.264, whose software path is the striped one, and
/// where the build carries none for the codec, which the ladder then falls through on.
fn software_fallback(settings: &mut RustCaptureSettings, rgba: bool, tag: &str) -> Option<FrameEncoder> {
    let codec = settings.codec;
    if codec == Codec::H264 {
        println!("[{tag}] Encoder: software {} ({}).", codec.display(), software_library(Codec::H264));
        return None;
    }
    match AvcodecEncoder::new(settings, codec, Backend::Software, Input::Host { rgba }) {
        Ok(enc) => {
            println!("[{tag}] Encoder: software {} ({}).", codec.display(), enc.library());
            Some(FrameEncoder::Avcodec(enc))
        }
        Err(e) => {
            eprintln!("[{tag}] No {} encoder available: {e}.", codec.display());
            None
        }
    }
}


/// The fixture the chroma-siting checks of every backend share.
#[cfg(test)]
pub(crate) mod chroma_siting {
    /// Four colors averaging to gray, of which no pixel, row pair or column pair does: the
    /// chroma of a block comes out neutral only where all four were averaged. A 4:2:0 convert
    /// that keeps one pixel of the block, or one row or column of it, leaves the saturation
    /// subpixel-antialiased text carries on its glyph edges in the picture as visible color.
    pub const TILE: [[u8; 3]; 4] = [[0, 0, 128], [0, 255, 0], [128, 128, 255], [255, 0, 0]];

    /// `TILE` laid out as a `w`x`h` BGRA frame.
    pub fn bgra(w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let p = TILE[(y % 2) * 2 + (x % 2)];
                buf[(y * w + x) * 4..][..3].copy_from_slice(&[p[2], p[1], p[0]]);
            }
        }
        buf
    }

    /// The luma weights `(Kr, Kb)` of the matrix every session converts with and declares.
    pub const BT709: (f64, f64) = (0.2126, 0.0722);

    /// BT.601's weights, the other matrix a receiver might invert, which the checks use as the
    /// contrast: a mis-declared stream lands tens of levels away under them.
    pub const BT601: (f64, f64) = (0.299, 0.114);

    /// Limited-range Y/Cb/Cr of an RGB triple under the matrix `k` names.
    pub fn ycbcr(rgb: [f64; 3], k: (f64, f64)) -> [f64; 3] {
        let ([r, g, b], (kr, kb)) = (rgb, k);
        let y = kr * r + (1.0 - kr - kb) * g + kb * b;
        [
            16.0 + 219.0 * y / 255.0,
            128.0 + 224.0 * (b - y) / (2.0 * (1.0 - kb) * 255.0),
            128.0 + 224.0 * (r - y) / (2.0 * (1.0 - kr) * 255.0),
        ]
    }

    /// Its chroma pair alone.
    pub fn chroma(rgb: [f64; 3], k: (f64, f64)) -> (f64, f64) {
        let c = ycbcr(rgb, k);
        (c[1], c[2])
    }

    /// The inverse: the RGB a receiver paints from a limited-range Y/Cb/Cr under the matrix `k`
    /// names, which is what a client's presentation path computes.
    pub fn rgb(ycc: [f64; 3], k: (f64, f64)) -> [f64; 3] {
        let (kr, kb) = k;
        let (y, cb, cr) = ((ycc[0] - 16.0) / 219.0, (ycc[1] - 128.0) / 224.0, (ycc[2] - 128.0) / 224.0);
        let r = y + 2.0 * (1.0 - kr) * cr;
        let b = y + 2.0 * (1.0 - kb) * cb;
        let g = (y - kr * r - kb * b) / (1.0 - kr - kb);
        [r, g, b].map(|c| (c * 255.0).clamp(0.0, 255.0))
    }

    /// The eight-patch color chart the matrix checks paint: the neutrals, whose chroma a wrong
    /// matrix leaves alone, and the saturated corners, which it moves by tens of levels.
    pub const CHART: [[u8; 3]; 8] = [
        [255, 255, 255], [128, 128, 128], [0, 0, 0], [255, 0, 0],
        [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
    ];

    /// `CHART` as a `w`x`h` BGRA frame of eight columns.
    pub fn chart_bgra(w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let p = CHART[(x * CHART.len() / w).min(CHART.len() - 1)];
                buf[(y * w + x) * 4..][..3].copy_from_slice(&[p[2], p[1], p[0]]);
            }
        }
        buf
    }

    /// The worst channel error, over `CHART`'s patches, between the RGB a receiver paints from a
    /// decoded frame — inverting the matrix `k` names — and the RGB that was painted. Each
    /// patch is sampled well inside its column, so neither the 4:2:0 chroma edges nor the
    /// encoder's ringing at the boundaries counts.
    pub fn chart_error(f: &crate::webcam::convert::I420View<'_>, k: (f64, f64)) -> f64 {
        let cols = CHART.len();
        let mut worst = 0.0f64;
        for (i, want) in CHART.iter().enumerate() {
            let (x0, x1) = (f.width * i / cols, f.width * (i + 1) / cols);
            let (lo, hi) = (x0 + (x1 - x0) / 4, x1 - (x1 - x0) / 4);
            let (mut acc, mut n) = ([0.0f64; 3], 0.0f64);
            for y in f.height / 4..f.height * 3 / 4 {
                for x in lo..hi {
                    let ycc = [
                        f64::from(f.y[y * f.y_stride + x]),
                        f64::from(f.u[(y / 2) * f.uv_stride + x / 2]),
                        f64::from(f.v[(y / 2) * f.uv_stride + x / 2]),
                    ];
                    let got = rgb(ycc, k);
                    for c in 0..3 {
                        acc[c] += got[c];
                    }
                    n += 1.0;
                }
            }
            for c in 0..3 {
                worst = worst.max((acc[c] / n - f64::from(want[c])).abs());
            }
        }
        worst
    }

    /// The worst distance from neutral chroma over a decoded frame's chroma planes.
    pub fn worst(f: &crate::webcam::convert::I420View<'_>) -> f64 {
        let mut worst = 0.0f64;
        for r in 0..f.chroma_height() {
            for c in 0..f.chroma_width() {
                let i = r * f.uv_stride + c;
                worst = worst.max((f64::from(f.u[i]) - 128.0).hypot(f64::from(f.v[i]) - 128.0));
            }
        }
        worst
    }

    /// The tile is a fixture, so its premise is checked where it lives: the four chromas cancel,
    /// and every partial average a wrong siting would take is far from neutral.
    #[test]
    fn the_tile_separates_the_sitings() {
        let c: Vec<(f64, f64)> = TILE.iter().map(|&p| chroma(p.map(f64::from), BT709)).collect();
        let mean = |of: &[usize]| {
            let (u, v) = of.iter().fold((0.0, 0.0), |(u, v), &i| (u + c[i].0, v + c[i].1));
            let n = of.len() as f64;
            (u / n - 128.0).hypot(v / n - 128.0)
        };
        let all = mean(&[0, 1, 2, 3]);
        assert!(all < 0.5, "the whole tile must average to neutral chroma, off by {all:.1}");
        for part in [vec![0], vec![1], vec![2], vec![3], vec![0, 1], vec![2, 3], vec![0, 2], vec![1, 3]] {
            let d = mean(&part);
            assert!(d > 40.0, "pixels {part:?} average to chroma only {d:.1} from neutral");
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

/// Whether a session signals full range: a software 4:4:4 session of x264's kind, which the
/// striped path (`None`) is whenever it carries 4:4:4, or a device that converted in fixed
/// function at a range it chose. The one answer every description of a session reads, so no
/// two of them can disagree.
pub fn session_full_range(encoder: Option<&FrameEncoder>, settings: &RustCaptureSettings) -> bool {
    match encoder {
        Some(enc) => enc.is_full_range(),
        None => session_fullcolor(None, settings),
    }
}

#[cfg(test)]
mod hardware_encoder_tests {
    //! The hardware encoder table of a render node, on a host with an engine behind its first
    //! node. Ignored by default; run with `cargo test gpu_ -- --ignored --test-threads=1`.
    use super::*;

    /// The first node serves H.264 on the backend its driver selects, every entry names one of
    /// the two backends, and the second read is the remembered first.
    #[test]
    #[ignore]
    fn gpu_hardware_encoders_serve_h264_once_probed() {
        let served = hardware_encoders(0);
        assert!(served.iter().any(|(codec, ..)| *codec == Codec::H264), "node 0 serves {served:?}");
        assert!(served.iter().all(|(_, backend, _)| matches!(*backend, "nvenc" | "vaapi")));
        assert_eq!(hardware_encoders(0), served);
    }
}
