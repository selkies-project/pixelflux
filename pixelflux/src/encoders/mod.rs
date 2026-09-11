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

/// A software encoder the build can run for one codec: the library's name as reported to
/// Python and the logs, and the libavcodec encoder that reaches it (empty for H.264, whose
/// software encoders are linked directly).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftwareEncoder {
    pub library: &'static str,
    pub avcodec: &'static str,
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

/// The software encoder this build runs for `codec`, or `None` when the linked FFmpeg
/// carries none of the encoders the codec is served by.
///
/// H.264 is fixed by the crate features: libx264 whenever `gpl` is on (it wins even if
/// `openh264` is also enabled), Cisco OpenH264 for a GPL-free build; it is what the striped
/// software path and the full-frame software fallback under NVENC/VA-API both encode with.
/// The other codecs are probed once against the linked libavcodec: HEVC through x265 (GPL,
/// so only with the `gpl` feature) or kvazaar, VP8 and VP9 through libvpx, AV1 through
/// SVT-AV1. A build without `gpl` never picks x265 even from a system FFmpeg that has it,
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
        let first_linked = |candidates: &[SoftwareEncoder]| {
            candidates.iter().copied().find(|c| avcodec_has_encoder(c.avcodec))
        };
        [
            Some(SoftwareEncoder {
                library: if cfg!(feature = "gpl") { "x264" } else { "openh264" },
                avcodec: "",
            }),
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
/// what was asked for: a hardware encoder can refuse 4:4:4, and only some software encoders
/// carry it at full range. Shared so the X11 and Wayland logs describe an identical session
/// identically.
pub fn colorspace_desc(fullcolor: bool, full_range: bool) -> &'static str {
    match (fullcolor, full_range) {
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

    /// Whether the session signals full range, which only a software 4:4:4 of x264's kind does.
    pub fn is_full_range(&self) -> bool {
        match self {
            FrameEncoder::Nvenc(_) => false,
            FrameEncoder::Avcodec(enc) => enc.is_full_range(),
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
                    Ok(resized) => {
                        if resized {
                            println!("[{tag}] NVENC session reconfigured in place.");
                        }
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
        println!("[{tag}] Software H.264 ({}).", software_library(Codec::H264));
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
                software_library(Codec::H264)
            );
            settings.codec = Codec::H264;
            None
        }
    }
}

/// The fixture the chroma-siting checks of every backend share.
#[cfg(test)]
pub(crate) mod chroma_siting {
    /// Four colours averaging to grey, of which no pixel, row pair or column pair does: the
    /// chroma of a block comes out neutral only where all four were averaged. A 4:2:0 convert
    /// that keeps one pixel of the block, or one row or column of it, leaves the saturation
    /// subpixel-antialiased text carries on its glyph edges in the picture as visible colour.
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

    /// The eight-patch colour chart the matrix checks paint: the neutrals, whose chroma a wrong
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
/// striped path (`None`) is whenever it carries 4:4:4.
pub fn session_full_range(encoder: Option<&FrameEncoder>, settings: &RustCaptureSettings) -> bool {
    match encoder {
        Some(enc) => enc.is_full_range(),
        None => session_fullcolor(None, settings),
    }
}
