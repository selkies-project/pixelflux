//! The color a stream declares, read from an H.264 sequence parameter set and written into one.
//!
//! An encoder that converts RGB itself decides the matrix and the range, and the stream is the
//! only place a decoder learns which it was: with no `video_signal_type` in the VUI a browser
//! guesses from the frame size, so one session is colored one way at 1080p and another at
//! 800x600. Devices differ in whether they say: a Raspberry Pi with firmware from August 2024
//! or later converts and declares it, and the same board on an older firmware converts to full
//! range BT.601 and declares nothing at all.
//!
//! So this reads first. A stream that declares its color is left alone, whatever it declares,
//! because the device knows what it did and we do not. Only a stream that declares nothing is
//! written into, and then only with a value the caller can justify.
//!
//! The VUI sits at the end of the SPS, ahead of the trailing bits, and the fields after the
//! insertion point are copied as opaque bits rather than parsed: an encoder's HRD parameters are
//! no business of ours, and a writer that re-emits what it does not understand is a writer that
//! corrupts streams on devices it was never run against.

/// What a stream says about the color it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorSignal {
    pub full_range: bool,
    pub primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
}

impl ColorSignal {
    /// BT.601 at full range, which is what a VideoCore firmware older than August 2024 converts
    /// RGB with, both by measurement against a decoded chart and by the driver author's account.
    pub const BT601_FULL: Self =
        Self { full_range: true, primaries: 6, transfer: 6, matrix: 6 };

    /// BT.709 at limited range, the crate's own convention for 4:2:0.
    pub const BT709_LIMITED: Self =
        Self { full_range: false, primaries: 1, transfer: 1, matrix: 1 };
}

struct Reader<'a> {
    bits: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bits: &'a [u8]) -> Self {
        Self { bits, pos: 0 }
    }

    fn bit(&mut self) -> Result<u32, String> {
        let byte = self.bits.get(self.pos >> 3).ok_or("the SPS ends mid-field")?;
        let value = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(value as u32)
    }

    fn bits(&mut self, count: usize) -> Result<u32, String> {
        let mut value = 0;
        for _ in 0..count {
            value = (value << 1) | self.bit()?;
        }
        Ok(value)
    }

    /// Exp-Golomb, the coding every length in an SPS is written with.
    fn ue(&mut self) -> Result<u32, String> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err("an exp-Golomb code longer than the field it names".into());
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        Ok((1 << zeros) - 1 + self.bits(zeros)?)
    }

    fn se(&mut self) -> Result<i32, String> {
        let value = self.ue()?;
        Ok(if value.is_multiple_of(2) { -((value / 2) as i32) } else { value.div_ceil(2) as i32 })
    }
}

struct Writer {
    bytes: Vec<u8>,
    pos: usize,
}

impl Writer {
    fn new() -> Self {
        Self { bytes: Vec::new(), pos: 0 }
    }

    fn bit(&mut self, value: u32) {
        if self.pos.is_multiple_of(8) {
            self.bytes.push(0);
        }
        if value & 1 != 0 {
            let index = self.pos >> 3;
            self.bytes[index] |= 1 << (7 - (self.pos & 7));
        }
        self.pos += 1;
    }

    fn bits(&mut self, value: u32, count: usize) {
        for shift in (0..count).rev() {
            self.bit((value >> shift) & 1);
        }
    }

    /// Copy bits `[from, to)` of `source` verbatim, which is how everything this code does not
    /// interpret survives an insertion that shifts it.
    fn copy_from(&mut self, source: &[u8], from: usize, to: usize) {
        for index in from..to {
            let byte = source[index >> 3];
            self.bit(((byte >> (7 - (index & 7))) & 1) as u32);
        }
    }

    fn trailing_bits(&mut self) {
        self.bit(1);
        while !self.pos.is_multiple_of(8) {
            self.bit(0);
        }
    }
}

/// Where the video signal type sits in a VUI, and what it says today.
struct Located {
    /// The bit the block starts at: the `video_signal_type_present_flag` itself.
    start: usize,
    /// The bit after the block, where the rest of the VUI resumes.
    end: usize,
    /// Present when the VUI exists at all; a VUI has to be created otherwise.
    vui_present: bool,
    /// The bit the `vui_parameters_present_flag` sits at, for a stream that carries no VUI.
    vui_flag: usize,
    declared: Option<ColorSignal>,
}

fn unescape(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut index = 0;
    while index < nal.len() {
        if index + 2 < nal.len() && nal[index] == 0 && nal[index + 1] == 0 && nal[index + 2] == 3 {
            out.push(0);
            out.push(0);
            index += 3;
        } else {
            out.push(nal[index]);
            index += 1;
        }
    }
    out
}

fn escape(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 4);
    let mut zeros = 0;
    for &byte in rbsp {
        if zeros >= 2 && byte <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
    out
}

/// The last bit that carries meaning: the RBSP stop bit, with alignment zeros after it.
fn stop_bit(rbsp: &[u8]) -> Result<usize, String> {
    for index in (0..rbsp.len() * 8).rev() {
        let byte = rbsp[index >> 3];
        if (byte >> (7 - (index & 7))) & 1 == 1 {
            return Ok(index);
        }
    }
    Err("the SPS carries no stop bit".into())
}

fn locate(rbsp: &[u8]) -> Result<Located, String> {
    let mut r = Reader::new(rbsp);
    let profile = r.bits(8)?;
    r.bits(8)?;
    r.bits(8)?;
    r.ue()?;
    if matches!(profile, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135) {
        let chroma = r.ue()?;
        if chroma == 3 {
            r.bit()?;
        }
        r.ue()?;
        r.ue()?;
        r.bit()?;
        if r.bit()? == 1 {
            // Scaling lists are the one part an insertion cannot step over blindly, so a stream
            // carrying them is refused rather than guessed at.
            return Err("the SPS carries scaling lists".into());
        }
    }
    r.ue()?;
    let poc_type = r.ue()?;
    if poc_type == 0 {
        r.ue()?;
    } else if poc_type == 1 {
        r.bit()?;
        r.se()?;
        r.se()?;
        let cycle = r.ue()?;
        for _ in 0..cycle {
            r.se()?;
        }
    }
    r.ue()?;
    r.bit()?;
    r.ue()?;
    r.ue()?;
    if r.bit()? == 0 {
        r.bit()?;
    }
    r.bit()?;
    if r.bit()? == 1 {
        r.ue()?;
        r.ue()?;
        r.ue()?;
        r.ue()?;
    }
    let vui_flag = r.pos;
    if r.bit()? == 0 {
        return Ok(Located { start: vui_flag, end: vui_flag, vui_present: false, vui_flag, declared: None });
    }
    if r.bit()? == 1 && r.bits(8)? == 255 {
        r.bits(16)?;
        r.bits(16)?;
    }
    if r.bit()? == 1 {
        r.bit()?;
    }
    let start = r.pos;
    let declared = if r.bit()? == 1 {
        r.bits(3)?;
        let full_range = r.bit()? == 1;
        if r.bit()? == 1 {
            Some(ColorSignal {
                full_range,
                primaries: r.bits(8)? as u8,
                transfer: r.bits(8)? as u8,
                matrix: r.bits(8)? as u8,
            })
        } else {
            // A range without a matrix says half of what a decoder needs, and the half it leaves
            // out is the one that colors the picture, so this counts as undeclared.
            None
        }
    } else {
        None
    };
    Ok(Located { start, end: r.pos, vui_present: true, vui_flag, declared })
}

/// What the stream says about its color, or `None` when it says nothing a decoder can use.
pub fn read_color(nal: &[u8]) -> Option<ColorSignal> {
    let rbsp = unescape(nal.get(1..)?);
    locate(&rbsp).ok()?.declared
}

/// The same SPS with `signal` declared in its VUI, replacing whatever it declared before.
pub fn write_color(nal: &[u8], signal: ColorSignal) -> Result<Vec<u8>, String> {
    let header = *nal.first().ok_or("an empty NAL unit")?;
    let rbsp = unescape(&nal[1..]);
    let at = locate(&rbsp)?;
    let last = stop_bit(&rbsp)?;

    let mut w = Writer::new();
    if at.vui_present {
        w.copy_from(&rbsp, 0, at.start);
        write_signal(&mut w, signal);
        w.copy_from(&rbsp, at.end, last);
    } else {
        w.copy_from(&rbsp, 0, at.vui_flag);
        w.bit(1);
        w.bit(0);
        w.bit(0);
        write_signal(&mut w, signal);
        // The rest of a VUI this stream never had: chroma location, timing, both HRDs, picture
        // structure, and bitstream restriction, each absent.
        for _ in 0..6 {
            w.bit(0);
        }
    }
    w.trailing_bits();

    let mut out = Vec::with_capacity(w.bytes.len() + 8);
    out.push(header);
    out.extend_from_slice(&escape(&w.bytes));
    Ok(out)
}

fn write_signal(w: &mut Writer, signal: ColorSignal) {
    w.bit(1);
    w.bits(5, 3);
    w.bit(u32::from(signal.full_range));
    w.bit(1);
    w.bits(signal.primaries as u32, 8);
    w.bits(signal.transfer as u32, 8);
    w.bits(signal.matrix as u32, 8);
}

/// The sequence parameter set read an H.264 session makes of its own key frames, and the
/// reference checks of every backend share.
mod dpb {
    struct Bits<'a> {
        rbsp: &'a [u8],
        pos: usize,
    }

    impl Bits<'_> {
        fn bits(&mut self, n: u32) -> u32 {
            (0..n).fold(0, |acc, _| {
                let bit = self.rbsp.get(self.pos / 8).map_or(0, |b| (b >> (7 - self.pos % 8)) & 1);
                self.pos += 1;
                (acc << 1) | bit as u32
            })
        }

        fn ue(&mut self) -> u32 {
            let mut zeros = 0;
            while self.bits(1) == 0 && zeros < 32 {
                zeros += 1;
            }
            (1 << zeros) - 1 + self.bits(zeros)
        }

        fn se(&mut self) -> i32 {
            let k = self.ue() as i32;
            if k % 2 == 1 { (k + 1) / 2 } else { -(k / 2) }
        }
    }

    /// `(log2_max_frame_num, max_num_ref_frames, chroma_format_idc)` of the first SPS in an
    /// Annex-B H.264 stream.
    fn h264_sps(stream: &[u8]) -> Option<(u32, u32, u32)> {
        let nal = crate::encoders::codec::annexb_nals(stream).find(|n| n[0] & 0x1f == 7)?;
        let mut rbsp = Vec::with_capacity(nal.len());
        let mut zeros = 0;
        for &b in &nal[1..] {
            if zeros >= 2 && b == 3 {
                zeros = 0;
                continue;
            }
            zeros = if b == 0 { zeros + 1 } else { 0 };
            rbsp.push(b);
        }
        let mut r = Bits { rbsp: &rbsp, pos: 0 };
        let profile = r.bits(8);
        r.bits(16);
        r.ue();
        let mut chroma = 1;
        if matches!(profile, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135) {
            chroma = r.ue();
            if chroma == 3 {
                r.bits(1);
            }
            r.ue();
            r.ue();
            r.bits(1);
            if r.bits(1) == 1 {
                for list in 0..(if chroma == 3 { 12 } else { 8 }) {
                    if r.bits(1) == 1 {
                        let (mut last, mut next) = (8i32, 8i32);
                        for _ in 0..(if list < 6 { 16 } else { 64 }) {
                            if next != 0 {
                                next = (last + r.se()).rem_euclid(256);
                            }
                            if next != 0 {
                                last = next;
                            }
                        }
                    }
                }
            }
        }
        let log2_max_frame_num = r.ue() + 4;
        match r.ue() {
            0 => {
                r.ue();
            }
            1 => {
                r.bits(1);
                r.se();
                r.se();
                for _ in 0..r.ue() {
                    r.se();
                }
            }
            _ => {}
        }
        Some((log2_max_frame_num, r.ue(), chroma))
    }

    /// How many values `frame_num` takes before it wraps, from the first SPS of an Annex-B
    /// H.264 stream.
    pub fn h264_frame_num_range(stream: &[u8]) -> Option<u32> {
        h264_sps(stream).map(|(log2, _, _)| 1 << log2)
    }

    /// `max_num_ref_frames` of the first SPS in an Annex-B H.264 stream.
    #[cfg(test)]
    pub fn h264_max_num_ref_frames(stream: &[u8]) -> Option<u32> {
        h264_sps(stream).map(|(_, refs, _)| refs)
    }

    /// `chroma_format_idc` of the first SPS in an Annex-B H.264 stream: 3 for 4:4:4.
    #[cfg(test)]
    pub fn h264_chroma_format_idc(stream: &[u8]) -> Option<u32> {
        h264_sps(stream).map(|(_, _, chroma)| chroma)
    }
}

pub use dpb::h264_frame_num_range;
#[cfg(test)]
pub use dpb::{h264_chroma_format_idc, h264_max_num_ref_frames};

#[cfg(test)]
mod tests {
    use super::*;

    /// An SPS a Raspberry Pi 4 produced at 1280x720: a VUI with timing and no color at all.
    const PI4_SPS: &[u8] = &[
        0x27, 0x64, 0x00, 0x28, 0xac, 0x2b, 0x40, 0x28, 0x02, 0xdd, 0x08, 0x00, 0x00, 0x03, 0x00,
        0x08, 0x00, 0x00, 0x03, 0x01, 0xe7, 0x15, 0x00, 0x01, 0xe8, 0x48, 0x00, 0x02, 0xfa, 0xf3,
        0x7b, 0xdc, 0x03, 0xc4, 0x89, 0xa8,
    ];

    #[test]
    fn a_stream_that_declares_nothing_reads_as_nothing() {
        assert_eq!(read_color(PI4_SPS), None);
    }

    #[test]
    fn what_is_written_reads_back() {
        for signal in [ColorSignal::BT601_FULL, ColorSignal::BT709_LIMITED] {
            let patched = write_color(PI4_SPS, signal).expect("the SPS takes a color");
            assert_eq!(read_color(&patched), Some(signal), "what was written did not read back");
        }
    }

    /// The fields after the insertion point are what the encoder said about timing and delay, and
    /// a stream whose insertion trod on them decodes at the wrong rate rather than failing.
    #[test]
    fn the_bits_after_the_insertion_survive_it() {
        let patched = write_color(PI4_SPS, ColorSignal::BT601_FULL).expect("patched");
        let (before, after) = (unescape(&PI4_SPS[1..]), unescape(&patched[1..]));
        let at_before = locate(&before).expect("located");
        let at_after = locate(&after).expect("located");
        let tail_before = stop_bit(&before).expect("stop") - at_before.end;
        let tail_after = stop_bit(&after).expect("stop") - at_after.end;
        assert_eq!(tail_before, tail_after, "the tail changed length");
        for index in 0..tail_before {
            let bit = |data: &[u8], at: usize| (data[at >> 3] >> (7 - (at & 7))) & 1;
            assert_eq!(
                bit(&before, at_before.end + index),
                bit(&after, at_after.end + index),
                "bit {index} of the tail changed"
            );
        }
    }

    #[test]
    fn a_declared_color_reads_back_as_declared() {
        let patched = write_color(PI4_SPS, ColorSignal::BT709_LIMITED).expect("patched");
        assert_eq!(read_color(&patched), Some(ColorSignal::BT709_LIMITED));
        let again = write_color(&patched, ColorSignal::BT601_FULL).expect("rewritten");
        assert_eq!(read_color(&again), Some(ColorSignal::BT601_FULL), "a rewrite did not replace");
        assert_eq!(again.len(), patched.len(), "replacing a color changed the length");
    }

    /// Emulation prevention is not decoration: a byte pair of zeros followed by a small byte is
    /// a start code to a decoder unless it is escaped.
    #[test]
    fn escaping_round_trips() {
        let raw = vec![0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x03, 0xff, 0x00, 0x00];
        assert_eq!(unescape(&escape(&raw)), raw);
        assert!(escape(&raw).len() > raw.len(), "nothing was escaped");
    }
}
