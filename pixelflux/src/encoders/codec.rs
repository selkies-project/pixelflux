/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! What every encoder backend shares about a codec: its name and wire identity, the
//! quantizer domain the session-wide quality index maps onto, the level a stream of a
//! given geometry advertises, and the bitstream reads that label a frame on the wire.

/// The picture codec of a capture session, and of a virtual-camera uplink. `Jpeg` is the
/// striped still-image mode; the rest are video codecs, which stream as whole frames except
/// H.264, whose software path may also stripe. The discriminants are the codec ids the
/// Python API speaks (`VirtualCamera.CODEC_*`) and the video wire ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Codec {
    Jpeg = 0,
    H264 = 1,
    Vp8 = 2,
    Vp9 = 3,
    Av1 = 4,
    H265 = 5,
}

/// Wire tag of a JPEG stripe: `u8 reserved`, `u16 frame id`, `u16 stripe Y`, JPEG data.
pub const WIRE_JPEG: u8 = 0x03;
/// Wire tag of an encoded video frame or stripe: a type byte whose low nibble is the
/// frame kind and whose high nibble is the codec's [`Codec::wire_id`], then `u16 frame
/// id`, `u16 stripe Y`, `u16 width`, `u16 height`, and the codec's own bitstream.
pub const WIRE_VIDEO: u8 = 0x04;
/// Frame kinds in the low nibble of the video type byte: a decode entry point, an
/// intra-coded picture that is not one, and a predicted picture.
pub const FRAME_KEY: u8 = 0x01;
pub const FRAME_INTRA: u8 = 0x02;
pub const FRAME_DELTA: u8 = 0x00;
/// Bytes of the two stripe headers.
pub const JPEG_HEADER_LEN: usize = 6;
pub const VIDEO_HEADER_LEN: usize = 10;

impl Codec {
    /// The video codecs, in wire-id order.
    pub const VIDEO: [Codec; 5] = [Codec::H264, Codec::Vp8, Codec::Vp9, Codec::Av1, Codec::H265];

    /// The codec's numeric id, as the Python API and the video wire carry it.
    pub const fn id(self) -> u32 {
        self as u32
    }

    /// The codec a numeric id names.
    pub fn from_id(id: u32) -> Option<Codec> {
        [Codec::Jpeg, Codec::H264, Codec::Vp8, Codec::Vp9, Codec::Av1, Codec::H265]
            .into_iter()
            .find(|c| c.id() == id)
    }

    /// The codec a settings name selects; `None` for a name no backend serves.
    pub fn parse(name: &str) -> Option<Codec> {
        match name.trim().to_ascii_lowercase().as_str() {
            "jpeg" | "mjpeg" => Some(Codec::Jpeg),
            "h264" | "avc" => Some(Codec::H264),
            "h265" | "hevc" => Some(Codec::H265),
            "vp8" => Some(Codec::Vp8),
            "vp9" => Some(Codec::Vp9),
            "av1" => Some(Codec::Av1),
            _ => None,
        }
    }

    /// The settings and log name of the codec.
    pub fn name(self) -> &'static str {
        match self {
            Codec::Jpeg => "jpeg",
            Codec::H264 => "h264",
            Codec::H265 => "h265",
            Codec::Vp8 => "vp8",
            Codec::Vp9 => "vp9",
            Codec::Av1 => "av1",
        }
    }

    /// The name a stream line prints.
    pub fn display(self) -> &'static str {
        match self {
            Codec::Jpeg => "JPEG",
            Codec::H264 => "H264",
            Codec::H265 => "H265",
            Codec::Vp8 => "VP8",
            Codec::Vp9 => "VP9",
            Codec::Av1 => "AV1",
        }
    }

    /// Whether the session is a video stream rather than striped stills.
    pub fn is_video(self) -> bool {
        self != Codec::Jpeg
    }

    /// Whether a frame of this codec may be cut into independently encoded stripes: the
    /// still-image mode always is, H.264 is on its software path, and every other codec
    /// streams whole frames.
    pub fn stripes(self) -> bool {
        matches!(self, Codec::Jpeg | Codec::H264)
    }

    /// The codec's identity in the high nibble of the video type byte: its id.
    pub fn wire_id(self) -> u8 {
        self.id() as u8
    }

    /// The video codec a wire id names.
    pub fn from_wire_id(id: u8) -> Option<Codec> {
        Codec::from_id(id as u32).filter(|c| c.is_video())
    }

    /// The codec tag the Python `StripeFrame` carries: the codec id.
    pub fn data_type(self) -> i32 {
        self.id() as i32
    }

    /// Whether frames depend on earlier ones, so a dropped frame forces a wait for a key
    /// frame.
    pub fn is_inter_coded(self) -> bool {
        self.is_video()
    }

    /// Whether the codec can carry 4:4:4 chroma at all; whether a given backend does is
    /// negotiated by that backend.
    pub fn fullcolor(self) -> bool {
        matches!(self, Codec::H264 | Codec::H265)
    }

    /// Whether a forced key frame of this codec repeats the stream's parameter sets, so
    /// a joining decoder can start there. Every backend is configured to, so the answer
    /// is the same for all of them.
    pub fn keyframes_self_contained(self) -> bool {
        self.is_video()
    }

    /// The top of the codec's quantizer domain: H.26x QP, the VP8 quantizer index, or
    /// the VP9 / AV1 `base_q_idx`. The value a backend programs never exceeds it.
    pub fn quantizer_max(self) -> u32 {
        match self {
            Codec::Jpeg => 100,
            Codec::H264 | Codec::H265 => 51,
            Codec::Vp8 => 127,
            Codec::Vp9 | Codec::Av1 => 255,
        }
    }

    /// The quantizer, in this codec's own domain, that a session quality index (`video_crf`,
    /// on the H.26x QP scale) selects.
    ///
    /// H.26x takes the index as its QP. The VP8, VP9 and AV1 quantizer indices are looked
    /// up in tables measured against x264 at the same settings pixelflux runs it with,
    /// matching on SSIM of a scrolling text desktop, and interpolated between the
    /// measured points. An AV1 index of zero is refused by NVENC and is lossless on the
    /// others, so its floor is one.
    pub fn quantizer(self, crf: i32) -> u32 {
        let crf = crf.clamp(0, 51) as u32;
        match self {
            Codec::Jpeg => crf,
            Codec::H264 | Codec::H265 => crf,
            Codec::Vp8 => interpolate(&VP8_QINDEX, crf),
            Codec::Vp9 => interpolate(&VP9_QINDEX, crf),
            Codec::Av1 => interpolate(&AV1_QINDEX, crf).max(1),
        }
    }

    /// The quantizer a CBR QP clamp (`video_min_qp` / `video_max_qp`, on the H.26x scale)
    /// becomes in this codec's domain; zero stays zero, meaning "the encoder's default".
    pub fn quantizer_bound(self, qp: i32) -> u32 {
        if qp <= 0 { 0 } else { self.quantizer(qp) }
    }
}

/// Session quality index → VP8 quantizer index (0..=127) breakpoints.
const VP8_QINDEX: [(u32, u32); 8] =
    [(0, 0), (10, 3), (15, 14), (20, 29), (25, 52), (30, 78), (35, 111), (40, 127)];
/// Session quality index → VP9 `base_q_idx` (0..=255) breakpoints.
const VP9_QINDEX: [(u32, u32); 9] =
    [(0, 0), (10, 25), (15, 70), (20, 122), (25, 158), (30, 190), (35, 216), (40, 249), (45, 255)];
/// Session quality index → AV1 `base_q_idx` (0..=255) breakpoints.
const AV1_QINDEX: [(u32, u32); 9] =
    [(0, 0), (10, 9), (15, 45), (20, 119), (25, 166), (30, 195), (35, 223), (40, 254), (45, 255)];

/// Piecewise-linear lookup of `x` in ascending `(x, y)` breakpoints, clamped at both ends.
fn interpolate(points: &[(u32, u32)], x: u32) -> u32 {
    let (first, last) = (points[0], points[points.len() - 1]);
    if x <= first.0 {
        return first.1;
    }
    if x >= last.0 {
        return last.1;
    }
    for pair in points.windows(2) {
        let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
        if x >= x0 && x <= x1 {
            let span = (x1 - x0) as i64;
            let along = (x - x0) as i64;
            let delta = y1 as i64 - y0 as i64;
            return (y0 as i64 + (delta * along + span / 2) / span) as u32;
        }
    }
    last.1
}

/// `base_q_idx` of each libvpx / libaom quantizer level 0..=63, as those libraries map
/// their `--min-q` / `--max-q` / `--cq-level` arguments.
pub const VPX_QINDEX: [u8; 64] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92,
    96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168,
    172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244,
    249, 255,
];
/// VP8 quantizer index of each libvpx VP8 quantizer level 0..=63.
pub const VP8_LEVEL_QINDEX: [u8; 64] = [
    0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 64, 67, 70, 73, 76, 79, 82,
    85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127,
];

/// The libvpx / SVT-AV1 quantizer level (0..=63) closest to a quantizer index of
/// `codec`'s domain: the software encoders take the level, the hardware ones the index.
pub fn vpx_level(codec: Codec, qindex: u32) -> u32 {
    let table: &[u8; 64] = if codec == Codec::Vp8 { &VP8_LEVEL_QINDEX } else { &VPX_QINDEX };
    let target = qindex.min(codec.quantizer_max()) as i64;
    let mut best = 0usize;
    for (level, &q) in table.iter().enumerate() {
        if (q as i64 - target).abs() < (table[best] as i64 - target).abs() {
            best = level;
        }
    }
    best as u32
}

/// Append the header of one encoded video frame or stripe.
pub fn push_video_header(
    out: &mut Vec<u8>,
    codec: Codec,
    frame_type: u8,
    frame_id: u16,
    y_start: u16,
    width: u16,
    height: u16,
) {
    out.push(WIRE_VIDEO);
    out.push((frame_type & 0x0f) | (codec.wire_id() << 4));
    out.extend_from_slice(&frame_id.to_be_bytes());
    out.extend_from_slice(&y_start.to_be_bytes());
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&height.to_be_bytes());
}

/// Append the header of one JPEG stripe.
pub fn push_jpeg_header(out: &mut Vec<u8>, frame_id: u16, y_start: u16) {
    out.push(WIRE_JPEG);
    out.push(0);
    out.extend_from_slice(&frame_id.to_be_bytes());
    out.extend_from_slice(&y_start.to_be_bytes());
}

/// The codec and frame kind a video type byte carries.
pub fn parse_video_type(type_byte: u8) -> Option<(Codec, u8)> {
    Codec::from_wire_id(type_byte >> 4).map(|c| (c, type_byte & 0x0f))
}

/// The frame kind of a key/delta answer an encoder reports without a bitstream parse.
pub fn frame_type_from_key(key: bool) -> u8 {
    if key { FRAME_KEY } else { FRAME_DELTA }
}

/// Lowest H.264 level whose Annex-A Table A-1 limits admit a `width` x `height` stream at
/// `fps`, as level_idc (41 = 4.1, 52 = 5.2, 62 = 6.2).
///
/// A frame is charged two ways: **MaxFS**, its size in macroblocks, and **MaxMBPS**, that
/// size times the frame rate. Advertising the lowest fitting level asks the least of a
/// decoder, so clients that gate hardware decode on the level accept the widest range of
/// streams. One shared ladder keeps every backend agreeing on the same geometry; it starts
/// at 4.1 and a caller needing more raises the floor itself. Above 6.2's limits there is
/// no higher level, so it returns 62 as a best effort.
pub fn h264_level(width: u32, height: u32, fps: u32) -> u32 {
    let mbs = (width as u64).div_ceil(16) * (height as u64).div_ceil(16);
    let mbps = mbs * fps.max(1) as u64;
    const LEVELS: [(u32, u64, u64); 8] = [
        (41, 8192, 245760),
        (42, 8704, 522240),
        (50, 22080, 589824),
        (51, 36864, 983040),
        (52, 36864, 2073600),
        (60, 139264, 4177920),
        (61, 139264, 8355840),
        (62, 139264, 16711680),
    ];
    for &(level, max_fs, max_mbps) in &LEVELS {
        if mbs <= max_fs && mbps <= max_mbps {
            return level;
        }
    }
    62
}

/// Lowest H.265 level whose Annex-A Tables A.8 and A.9 (Main tier) limits admit a `width` x
/// `height` stream at `fps`, as general_level_idc (123 = 4.1, 156 = 5.2, 186 = 6.2).
///
/// A frame is charged by **MaxLumaPs**, its luma samples, and **MaxLumaSr**, those times the
/// frame rate. The ladder starts at 4.1 like the H.264 one and answers 6.2 for anything
/// beyond its limits.
pub fn h265_level(width: u32, height: u32, fps: u32) -> u32 {
    let ps = width as u64 * height as u64;
    let sr = ps * fps.max(1) as u64;
    const LEVELS: [(u32, u64, u64); 7] = [
        (123, 2_228_224, 133_693_440),
        (150, 8_912_896, 267_386_880),
        (153, 8_912_896, 534_773_760),
        (156, 8_912_896, 1_069_547_520),
        (180, 35_651_584, 1_069_547_520),
        (183, 35_651_584, 2_139_095_040),
        (186, 35_651_584, 4_278_190_080),
    ];
    for &(level, max_ps, max_sr) in &LEVELS {
        if ps <= max_ps && sr <= max_sr {
            return level;
        }
    }
    186
}

/// Lowest AV1 level whose Annex A limits admit a `width` x `height` stream at `fps`, as
/// seq_level_idx (8 = 4.0, 13 = 5.1, 19 = 6.3).
///
/// A frame is charged by **MaxPicSize** and the per-axis **MaxHSize** / **MaxVSize**, and
/// its rate by **MaxDisplayRate**. The ladder starts at 4.0 and answers 6.3 for anything
/// beyond its limits.
pub fn av1_level(width: u32, height: u32, fps: u32) -> u32 {
    let ps = width as u64 * height as u64;
    let rate = ps * fps.max(1) as u64;
    const LEVELS: [(u32, u64, u32, u32, u64); 10] = [
        (8, 8_912_896, 4096, 2176, 267_386_880),
        (9, 8_912_896, 4096, 2176, 534_773_760),
        (12, 35_651_584, 8192, 4352, 1_069_547_520),
        (13, 35_651_584, 8192, 4352, 2_139_095_040),
        (14, 35_651_584, 8192, 4352, 4_278_190_080),
        (15, 35_651_584, 8192, 4352, 4_278_190_080),
        (16, 142_606_336, 16384, 8704, 4_278_190_080),
        (17, 142_606_336, 16384, 8704, 8_556_380_160),
        (18, 142_606_336, 16384, 8704, 17_112_760_320),
        (19, 142_606_336, 16384, 8704, 17_112_760_320),
    ];
    for &(level, max_ps, max_w, max_h, max_rate) in &LEVELS {
        if ps <= max_ps && width <= max_w && height <= max_h && rate <= max_rate {
            return level;
        }
    }
    19
}

/// Wire frame kind of one encoded H.264 access unit, read from the bitstream itself:
/// IDR = key, intra-only non-IDR = intra, predicted = delta. The stripe header labels
/// real decode entry points, so the NAL and slice types the encoder actually produced
/// decide, never what was requested of it. An access unit whose slices cannot be parsed
/// is labeled delta: a false delta costs one skipped entry point, a false key breaks the
/// decoder.
pub fn h264_frame_type(au: &[u8]) -> u8 {
    let mut intra = false;
    for nal in annexb_nals(au) {
        match nal[0] & 0x1f {
            5 => return FRAME_KEY,
            1 => match h264_slice_is_intra(&nal[1..]) {
                Some(true) => intra = true,
                _ => return FRAME_DELTA,
            },
            _ => {}
        }
    }
    if intra { FRAME_INTRA } else { FRAME_DELTA }
}

/// Wire frame kind of one encoded H.265 access unit: an IRAP picture (BLA, IDR or CRA,
/// which every backend here emits as an IDR with its parameter sets) is a key, anything
/// else a delta.
pub fn h265_frame_type(au: &[u8]) -> u8 {
    for nal in annexb_nals(au) {
        let nal_type = (nal[0] >> 1) & 0x3f;
        if (16..=23).contains(&nal_type) {
            return FRAME_KEY;
        }
    }
    FRAME_DELTA
}

/// Whether a VP8 frame is a key frame: the frame tag's lowest bit is clear.
pub fn vp8_is_key(frame: &[u8]) -> bool {
    frame.first().is_some_and(|b| b & 1 == 0)
}

/// Whether a VP9 frame is a key frame, read from its uncompressed header: not a shown existing
/// frame, and `frame_type` KEY_FRAME. Profile 3 carries one reserved bit ahead of those two.
pub fn vp9_is_key(frame: &[u8]) -> bool {
    let Some(&b) = frame.first() else { return false };
    if b >> 6 != 0b10 {
        return false;
    }
    let profile = ((b >> 5) & 1) | (((b >> 4) & 1) << 1);
    let (show_existing, frame_type) = if profile == 3 { (b & 0x04, b & 0x02) } else { (b & 0x08, b & 0x04) };
    show_existing == 0 && frame_type == 0
}

/// Whether an AV1 temporal unit opens with a key frame: the first frame header or frame
/// OBU's `frame_type`, read past any temporal delimiter and sequence header. A show
/// existing frame, or a unit without a frame, is a delta.
pub fn av1_is_key(tu: &[u8]) -> bool {
    let mut pos = 0usize;
    while pos < tu.len() {
        let header = tu[pos];
        let obu_type = (header >> 3) & 0x0f;
        let has_extension = header & 0x04 != 0;
        let has_size = header & 0x02 != 0;
        let mut i = pos + 1 + has_extension as usize;
        let size = if has_size {
            let mut value = 0u64;
            let mut shift = 0u32;
            loop {
                if i >= tu.len() || shift > 56 {
                    return false;
                }
                let byte = tu[i];
                i += 1;
                value |= ((byte & 0x7f) as u64) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            value as usize
        } else {
            tu.len().saturating_sub(i)
        };
        if matches!(obu_type, 3 | 6) {
            let Some(&first) = tu.get(i) else { return false };
            let show_existing_frame = first & 0x80 != 0;
            let frame_type = (first >> 5) & 0x03;
            return !show_existing_frame && frame_type == 0;
        }
        pos = i.saturating_add(size);
    }
    false
}

/// The NAL units of an Annex-B stream, each without its start code and never empty.
fn annexb_nals(au: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + 2 < au.len() {
        if au[i] == 0 && au[i + 1] == 0 && au[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::with_capacity(starts.len());
    for (n, &start) in starts.iter().enumerate() {
        let mut end = starts.get(n + 1).map_or(au.len(), |&next| next - 3);
        while end > start && au[end - 1] == 0 {
            end -= 1;
        }
        if end > start {
            nals.push(&au[start..end]);
        }
    }
    nals.into_iter()
}

/// Whether a non-IDR H.264 slice header opens an I/SI slice. Reads `first_mb_in_slice`
/// and `slice_type` (both ue(v)) from the start of the slice RBSP, undoing emulation
/// prevention bytes; `None` when the header is truncated or malformed.
fn h264_slice_is_intra(payload: &[u8]) -> Option<bool> {
    let mut rbsp = [0u8; 12];
    let mut n = 0usize;
    let mut zeros = 0usize;
    for &b in payload {
        if n == rbsp.len() {
            break;
        }
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        rbsp[n] = b;
        n += 1;
    }
    let mut pos = 0usize;
    let bit = |p: &mut usize| -> Option<u8> {
        if *p / 8 >= n {
            return None;
        }
        let b = (rbsp[*p / 8] >> (7 - *p % 8)) & 1;
        *p += 1;
        Some(b)
    };
    let ue = |p: &mut usize| -> Option<u32> {
        let mut leading = 0u32;
        while bit(p)? == 0 {
            leading += 1;
            if leading > 24 {
                return None;
            }
        }
        let mut val = 0u32;
        for _ in 0..leading {
            val = (val << 1) | bit(p)? as u32;
        }
        Some((1u32 << leading) - 1 + val)
    };
    ue(&mut pos)?;
    let slice_type = ue(&mut pos)?;
    if slice_type > 9 {
        return None;
    }
    Some(matches!(slice_type % 5, 2 | 4))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Names round-trip through the parser, and the aliases land on the same codec.
    #[test]
    fn names_round_trip() {
        for codec in [Codec::Jpeg, Codec::H264, Codec::H265, Codec::Vp8, Codec::Vp9, Codec::Av1] {
            assert_eq!(Codec::parse(codec.name()), Some(codec));
        }
        assert_eq!(Codec::parse("HEVC"), Some(Codec::H265));
        assert_eq!(Codec::parse(" avc "), Some(Codec::H264));
        assert_eq!(Codec::parse("mjpeg"), Some(Codec::Jpeg));
        assert_eq!(Codec::parse("h266"), None);
    }

    /// Every video codec has a distinct wire id that reads back, and the type byte packs
    /// the frame kind beside it.
    #[test]
    fn wire_ids_are_distinct_and_packed() {
        let mut seen = std::collections::HashSet::new();
        for codec in Codec::VIDEO {
            assert!(seen.insert(codec.wire_id()));
            assert_eq!(Codec::from_wire_id(codec.wire_id()), Some(codec));
            let mut out = Vec::new();
            push_video_header(&mut out, codec, FRAME_KEY, 0x1234, 7, 1920, 1080);
            assert_eq!(out.len(), VIDEO_HEADER_LEN);
            assert_eq!(out[0], WIRE_VIDEO);
            assert_eq!(parse_video_type(out[1]), Some((codec, FRAME_KEY)));
            assert_eq!(&out[2..], &[0x12, 0x34, 0, 7, 0x07, 0x80, 0x04, 0x38]);
        }
        assert_eq!(Codec::from_wire_id(0), None, "JPEG never rides the video tag");
        assert_eq!(Codec::from_wire_id(9), None);
        for id in 0..6 {
            assert_eq!(Codec::from_id(id).map(|c| c.id()), Some(id));
        }
        assert_eq!(Codec::from_id(6), None);
        assert_eq!(Codec::H264.data_type(), 1);
        assert_eq!(Codec::Jpeg.data_type(), 0);
        let mut jpeg = Vec::new();
        push_jpeg_header(&mut jpeg, 3, 64);
        assert_eq!(jpeg, [WIRE_JPEG, 0, 0, 3, 0, 64]);
    }

    /// Only the striped codecs stripe; every other codec is a whole-frame stream.
    #[test]
    fn striping_and_chroma_by_codec() {
        assert!(Codec::Jpeg.stripes() && Codec::H264.stripes());
        for codec in [Codec::H265, Codec::Vp8, Codec::Vp9, Codec::Av1] {
            assert!(!codec.stripes(), "{codec:?}");
            assert!(codec.is_video());
        }
        assert!(Codec::H264.fullcolor() && Codec::H265.fullcolor());
        assert!(!Codec::Vp9.fullcolor() && !Codec::Av1.fullcolor());
    }

    /// The quantizer maps are monotonic, span their domain, clamp the index, and keep AV1
    /// off the zero NVENC refuses; H.26x takes the index as it is.
    #[test]
    fn quantizer_maps_are_monotonic_and_bounded() {
        for codec in Codec::VIDEO {
            let mut last = 0;
            for crf in 0..=51 {
                let q = codec.quantizer(crf);
                assert!(q >= last, "{codec:?} crf {crf}: {q} < {last}");
                assert!(q <= codec.quantizer_max());
                last = q;
            }
            assert_eq!(codec.quantizer(51), codec.quantizer_max());
            assert_eq!(codec.quantizer(99), codec.quantizer(51));
            assert_eq!(codec.quantizer(-4), codec.quantizer(0));
            assert_eq!(codec.quantizer_bound(0), 0);
            assert_eq!(codec.quantizer_bound(30), codec.quantizer(30));
        }
        assert_eq!(Codec::H264.quantizer(25), 25);
        assert_eq!(Codec::H265.quantizer(25), 25);
        assert_eq!(Codec::Av1.quantizer(0), 1);
        assert_eq!(interpolate(&[(0, 0), (10, 100)], 5), 50);
        assert_eq!(interpolate(&[(0, 0), (10, 100)], 3), 30);
    }

    /// The libvpx quantizer level found for an index is the closest table entry, at both
    /// ends of both tables.
    #[test]
    fn vpx_levels_invert_the_tables() {
        assert_eq!(vpx_level(Codec::Vp9, 0), 0);
        assert_eq!(vpx_level(Codec::Vp9, 255), 63);
        assert_eq!(vpx_level(Codec::Vp9, 104), 26);
        assert_eq!(vpx_level(Codec::Av1, 249), 62);
        assert_eq!(vpx_level(Codec::Vp8, 0), 0);
        assert_eq!(vpx_level(Codec::Vp8, 127), 63);
        assert_eq!(vpx_level(Codec::Vp8, 52), 36, "a tie resolves to the lower level");
        for level in 0..64u32 {
            assert_eq!(vpx_level(Codec::Vp9, VPX_QINDEX[level as usize] as u32), level);
            assert_eq!(vpx_level(Codec::Vp8, VP8_LEVEL_QINDEX[level as usize] as u32), level);
        }
    }

    /// The level ladders start at their floors, step through the standard limits, and
    /// saturate rather than fail past the top.
    #[test]
    fn level_ladders() {
        assert_eq!(h264_level(1920, 1080, 60), 42);
        assert_eq!(h264_level(1280, 720, 30), 41);
        assert_eq!(h264_level(3840, 2160, 60), 52);
        assert_eq!(h264_level(7680, 4320, 60), 61);
        assert_eq!(h264_level(16384, 16384, 240), 62);

        assert_eq!(h265_level(1920, 1080, 60), 123);
        assert_eq!(h265_level(1920, 1080, 65), 150);
        assert_eq!(h265_level(3840, 2160, 60), 153);
        assert_eq!(h265_level(3840, 2160, 120), 156);
        assert_eq!(h265_level(7680, 4320, 30), 180);
        assert_eq!(h265_level(7680, 4320, 60), 183);
        assert_eq!(h265_level(7680, 4320, 120), 186);
        assert_eq!(h265_level(16384, 16384, 240), 186);

        assert_eq!(av1_level(1920, 1080, 60), 8);
        assert_eq!(av1_level(3840, 2160, 60), 9);
        assert_eq!(av1_level(4096, 2304, 30), 12, "taller than 4.x's MaxVSize");
        assert_eq!(av1_level(3840, 2160, 120), 12);
        assert_eq!(av1_level(7680, 4320, 60), 13);
        assert_eq!(av1_level(7680, 4320, 120), 14);
        assert_eq!(av1_level(16384, 8704, 60), 17);
        assert_eq!(av1_level(16384, 16384, 240), 19);
    }

    /// H.264 labels follow the NAL and slice types: an IDR is a key, an I slice in a
    /// non-IDR is intra, a P slice is delta, and a truncated slice is delta.
    #[test]
    fn h264_labels_from_the_bitstream() {
        let idr = [0, 0, 0, 1, 0x67, 0x64, 0, 0x1f, 0, 0, 1, 0x65, 0x88, 0x84, 0];
        assert_eq!(h264_frame_type(&idr), FRAME_KEY);
        // slice_type 7 (I, all slices) encoded ue: first_mb=0 -> '1', slice_type 7 -> '0001000'
        let intra = [0, 0, 1, 0x41, 0b1000_1000, 0b0000_0000];
        assert_eq!(h264_frame_type(&intra), FRAME_INTRA);
        // slice_type 0 (P): first_mb '1', slice_type '1'
        let p = [0, 0, 1, 0x41, 0b1100_0000];
        assert_eq!(h264_frame_type(&p), FRAME_DELTA);
        assert_eq!(h264_frame_type(&[0, 0, 1, 0x41]), FRAME_DELTA);
        assert_eq!(h264_frame_type(&[]), FRAME_DELTA);
    }

    /// H.265 labels follow the NAL type: an IDR (19) or CRA (21) is a key, a TRAIL_R (1)
    /// picture behind parameter sets is a delta, and a stream of parameter sets alone is
    /// a delta.
    #[test]
    fn h265_labels_from_the_bitstream() {
        let idr = [0, 0, 0, 1, 0x40, 0x01, 0x0c, 0, 0, 1, 0x26, 0x01, 0xaf];
        assert_eq!(h265_frame_type(&idr), FRAME_KEY);
        let cra = [0, 0, 1, 0x2a, 0x01, 0xaf];
        assert_eq!(h265_frame_type(&cra), FRAME_KEY);
        let trail = [0, 0, 1, 0x02, 0x01, 0xd0];
        assert_eq!(h265_frame_type(&trail), FRAME_DELTA);
        let headers_only = [0, 0, 1, 0x40, 0x01, 0, 0, 1, 0x42, 0x01];
        assert_eq!(h265_frame_type(&headers_only), FRAME_DELTA);
    }

    /// VP8 keys are read from the frame tag, VP9 keys from the uncompressed header in every
    /// profile, with a shown existing frame never a key.
    #[test]
    fn vpx_labels_from_the_bitstream() {
        assert!(vp8_is_key(&[0x50, 0x42, 0x00]));
        assert!(!vp8_is_key(&[0x51, 0x42, 0x00]));
        assert!(!vp8_is_key(&[]));
        assert!(vp9_is_key(&[0x82, 0x49, 0x83]));
        assert!(!vp9_is_key(&[0x86, 0x00]));
        assert!(!vp9_is_key(&[0x8a, 0x00]));
        assert!(vp9_is_key(&[0xb0, 0x00]), "profile 3");
        assert!(!vp9_is_key(&[0xb2, 0x00]), "profile 3 inter");
        assert!(!vp9_is_key(&[0x42, 0x00]), "bad frame marker");
    }

    /// AV1 labels come from the first frame's header past the temporal delimiter and
    /// sequence header: frame_type 0 is a key, 1 (inter) or a shown existing frame a
    /// delta, and a unit that ends before a frame, or a malformed size, is a delta.
    #[test]
    fn av1_labels_from_the_temporal_unit() {
        // TD (type 2, has_size, size 0), SEQ (type 1, size 1), FRAME (type 6, size 2) key.
        let key = [0x12, 0x00, 0x0a, 0x01, 0x00, 0x32, 0x02, 0x10, 0x00];
        assert!(av1_is_key(&key));
        let inter = [0x12, 0x00, 0x32, 0x02, 0x30, 0x00];
        assert!(!av1_is_key(&inter));
        let existing = [0x32, 0x01, 0x80];
        assert!(!av1_is_key(&existing));
        let frame_header_only = [0x1a, 0x01, 0x00];
        assert!(av1_is_key(&frame_header_only));
        assert!(!av1_is_key(&[0x12, 0x00]));
        assert!(!av1_is_key(&[0x32, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]));
        assert!(!av1_is_key(&[]));
    }
}
