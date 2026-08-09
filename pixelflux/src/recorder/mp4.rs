/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Pure-Rust fragmented-MP4 (fMP4) muxer for the built-in recorder.
//!
//! Hand-rolled rather than pulled in as a dependency for two reasons: `ffmpeg-sys-next`'s
//! `avformat` feature would add libavformat as a hard runtime dependency of every build, and the
//! pure-Rust mp4 crates only write moov-trailing progressive files, which lose everything on a
//! crash. Fragmented MP4 needs no trailer and no seeking — each `moof`+`mdat` pair is
//! self-contained — so a file truncated by a crash or SIGKILL stays playable up to the last
//! fragment, and the writer works on any `Write` sink.
//!
//! Timestamps are caller-supplied wall-clock microseconds (damage-driven capture emits sparse,
//! irregular frames), carried at a 90 kHz track timescale with one sample per fragment: `tfdt`
//! anchors every sample at its true capture time, so variable framerate needs no constant-rate
//! lie. The sample duration is only known once the NEXT frame arrives, so one sample is always
//! buffered and flushed a frame behind (a clean stop closes it with the median observed
//! duration); on SIGKILL at most that one buffered frame is lost — every fragment already
//! written remains playable.
//!
//! Codec support is deliberately split: [`annexb_to_sample`] and the H.264-specific parameter-set
//! capture live in [`H264SampleBuilder`], while the fragment/box writer below is codec-agnostic
//! (bytes + sync flag + timestamps + a ready-made `stsd` sample entry). HEVC or AV1 recording
//! later means a new sample builder emitting an `hvc1`/`av01` entry, not a new muxer.

use std::io::Write;

/// 90 kHz: the conventional H.264 track timescale, exactly representing common frame intervals.
const TIMESCALE: u32 = 90_000;

/// Fallback duration for the final buffered sample when only one frame was ever written
/// (no observed inter-frame delta to take a median of): 1/30 s.
const DEFAULT_LAST_DURATION: u32 = TIMESCALE / 30;

/// Split an Annex-B elementary stream into NAL payloads (start codes removed, emulation
/// prevention bytes kept — the RBSP layer is only unescaped where a parser needs it).
pub fn split_annexb(data: &[u8]) -> Vec<&[u8]> {
    let mut nals = Vec::new();
    let mut i = 0usize;
    let mut nal_start: Option<usize> = None;
    while i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let code_start = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
            if let Some(s) = nal_start
                && code_start > s {
                    nals.push(&data[s..code_start]);
                }
            i += 3;
            nal_start = Some(i);
        } else if data[i + 2] == 0 {
            // A zero at i+2 can begin the next start code; only advance one byte.
            i += 1;
        } else {
            i += 3;
        }
    }
    if let Some(s) = nal_start
        && data.len() > s {
            nals.push(&data[s..]);
        }
    nals
}

/// Unescape an H.264 RBSP: drop the emulation-prevention byte from every `00 00 03` run.
fn unescape_rbsp(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut zeros = 0u32;
    for &b in data {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        out.push(b);
    }
    out
}

/// MSB-first bit reader over an unescaped RBSP, with Exp-Golomb decode for SPS parsing.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn bit(&mut self) -> Option<u32> {
        let byte = *self.data.get(self.pos / 8)?;
        let bit = (byte >> (7 - (self.pos % 8))) & 1;
        self.pos += 1;
        Some(bit as u32)
    }

    fn bits(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Some(v)
    }

    /// ue(v): count leading zeros, then read that many bits after the marker one.
    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0u32;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        let rest = self.bits(zeros)?;
        Some((1u32 << zeros) - 1 + rest)
    }

    fn se(&mut self) -> Option<i32> {
        let k = self.ue()? as i64;
        Some(if k % 2 == 0 { -(k / 2) as i32 } else { ((k + 1) / 2) as i32 })
    }
}

/// Coded frame dimensions parsed out of an H.264 SPS NAL (with its NAL header byte).
///
/// Walks every field ahead of `pic_width_in_mbs_minus1` — including the high-profile
/// chroma/bit-depth/scaling-list block — and applies the frame-cropping rectangle with the
/// chroma-format-dependent crop units, so 4:2:0, 4:2:2 and 4:4:4 streams from any of the
/// project's encoders all report their true display size.
pub fn parse_sps_dimensions(sps_nal: &[u8]) -> Option<(u32, u32)> {
    if sps_nal.len() < 4 || sps_nal[0] & 0x1f != 7 {
        return None;
    }
    let profile_idc = sps_nal[1];
    let rbsp = unescape_rbsp(&sps_nal[4..]);
    let mut r = BitReader::new(&rbsp);
    // seq_parameter_set_id
    r.ue()?;

    let mut chroma_format_idc = 1u32;
    if matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        chroma_format_idc = r.ue()?;
        if chroma_format_idc == 3 {
            // separate_colour_plane_flag
            r.bit()?;
        }
        // bit_depth_luma_minus8, bit_depth_chroma_minus8, qpprime_y_zero_transform_bypass_flag
        r.ue()?;
        r.ue()?;
        r.bit()?;
        if r.bit()? == 1 {
            // seq_scaling_matrix_present_flag
            let lists = if chroma_format_idc == 3 { 12 } else { 8 };
            for i in 0..lists {
                if r.bit()? == 1 {
                    let size = if i < 6 { 16 } else { 64 };
                    let mut next_scale = 8i32;
                    let mut last_scale = 8i32;
                    for _ in 0..size {
                        if next_scale != 0 {
                            let delta = r.se()?;
                            next_scale = (last_scale + delta + 256) % 256;
                        }
                        if next_scale != 0 {
                            last_scale = next_scale;
                        }
                    }
                }
            }
        }
    }

    // log2_max_frame_num_minus4
    r.ue()?;
    let pic_order_cnt_type = r.ue()?;
    if pic_order_cnt_type == 0 {
        // log2_max_pic_order_cnt_lsb_minus4
        r.ue()?;
    } else if pic_order_cnt_type == 1 {
        // delta_pic_order_always_zero_flag, offset_for_non_ref_pic,
        // offset_for_top_to_bottom_field, then the offset_for_ref_frame list
        r.bit()?;
        r.se()?;
        r.se()?;
        let n = r.ue()?;
        for _ in 0..n {
            r.se()?;
        }
    }
    // max_num_ref_frames, gaps_in_frame_num_value_allowed_flag
    r.ue()?;
    r.bit()?;
    let pic_width_in_mbs = r.ue()? + 1;
    let pic_height_in_map_units = r.ue()? + 1;
    let frame_mbs_only = r.bit()?;
    if frame_mbs_only == 0 {
        // mb_adaptive_frame_field_flag
        r.bit()?;
    }
    // direct_8x8_inference_flag
    r.bit()?;

    let (mut crop_l, mut crop_r, mut crop_t, mut crop_b) = (0u32, 0u32, 0u32, 0u32);
    if r.bit()? == 1 {
        crop_l = r.ue()?;
        crop_r = r.ue()?;
        crop_t = r.ue()?;
        crop_b = r.ue()?;
    }

    let (sub_w, sub_h) = match chroma_format_idc {
        0 | 3 => (1u32, 1u32),
        2 => (2, 1),
        _ => (2, 2),
    };
    let crop_unit_x = sub_w;
    let crop_unit_y = sub_h * (2 - frame_mbs_only);
    let width = pic_width_in_mbs * 16 - crop_unit_x * (crop_l + crop_r);
    let height = (2 - frame_mbs_only) * pic_height_in_map_units * 16 - crop_unit_y * (crop_t + crop_b);
    Some((width, height))
}

fn mk_box(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + payload.len());
    b.extend_from_slice(&((8 + payload.len()) as u32).to_be_bytes());
    b.extend_from_slice(fourcc);
    b.extend_from_slice(payload);
    b
}

fn mk_full_box(fourcc: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(4 + payload.len());
    p.push(version);
    p.extend_from_slice(&flags.to_be_bytes()[1..]);
    p.extend_from_slice(payload);
    mk_box(fourcc, &p)
}

/// The unity transformation matrix `mvhd` and `tkhd` carry: 0x00010000, 0x00010000 and
/// 0x40000000 on the diagonal (16.16 fixed point for the first two, 2.30 for the last).
const MATRIX_IDENTITY: [u8; 36] = {
    let mut m = [0u8; 36];
    m[1] = 0x01;
    m[17] = 0x01;
    m[32] = 0x40;
    m
};

/// A codec's contribution to the init segment: its `stsd` sample entry (with decoder
/// configuration record) plus the display dimensions for `tkhd`.
pub struct TrackConfig {
    pub sample_entry: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// One buffered access unit awaiting its duration (known when the next one arrives).
struct PendingSample {
    data: Vec<u8>,
    sync: bool,
    dts: u64,
}

/// Aggregate counters reported when the writer finishes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mp4Stats {
    pub samples: u64,
    pub sync_samples: u64,
    pub bytes: u64,
    pub duration_us: u64,
}

/// Codec-agnostic fMP4 fragment writer: `ftyp`+`moov` once, then one `moof`+`mdat` pair
/// per sample, each anchored at its wall-clock decode time via `tfdt`.
pub struct FragmentWriter<W: Write> {
    out: W,
    seq: u32,
    wrote_init: bool,
    pending: Option<PendingSample>,
    last_dts: Option<u64>,
    /// Every flushed inter-frame duration, kept so the final buffered sample (whose
    /// successor never arrives) can close on the MEDIAN — damage-driven capture makes the
    /// last observed delta an outlier as often as not (a long static gap, a burst pair).
    durations: Vec<u32>,
    stats: Mp4Stats,
}

impl<W: Write> FragmentWriter<W> {
    pub fn new(out: W) -> Self {
        Self {
            out,
            seq: 0,
            wrote_init: false,
            pending: None,
            last_dts: None,
            durations: Vec::new(),
            stats: Mp4Stats::default(),
        }
    }

    pub fn init_written(&self) -> bool {
        self.wrote_init
    }

    /// Counters for the fragments written so far (the buffered pending sample is not yet
    /// included; `finish` folds it in).
    pub fn stats(&self) -> Mp4Stats {
        self.stats
    }

    /// Write `ftyp` + `moov` (track 1, `mvex`/`trex` marking the movie fragmented). Must be
    /// called once, before the first sample.
    pub fn write_init(&mut self, cfg: &TrackConfig) -> std::io::Result<()> {
        let mut ftyp_p = Vec::new();
        ftyp_p.extend_from_slice(b"isom");
        ftyp_p.extend_from_slice(&0x200u32.to_be_bytes());
        for brand in [b"isom", b"iso5", b"iso6", b"avc1", b"mp41"] {
            ftyp_p.extend_from_slice(brand);
        }
        let ftyp = mk_box(b"ftyp", &ftyp_p);

        // mvhd payload in field order: creation/modification time, timescale, duration
        // (0 = unknown, as it must be in a fragmented movie), rate 1.0, volume 1.0, reserved,
        // the unity matrix, pre_defined, next_track_ID.
        let mut mvhd_p = Vec::new();
        mvhd_p.extend_from_slice(&[0u8; 8]);
        mvhd_p.extend_from_slice(&TIMESCALE.to_be_bytes());
        mvhd_p.extend_from_slice(&0u32.to_be_bytes());
        mvhd_p.extend_from_slice(&0x00010000u32.to_be_bytes());
        mvhd_p.extend_from_slice(&0x0100u16.to_be_bytes());
        mvhd_p.extend_from_slice(&[0u8; 10]);
        mvhd_p.extend_from_slice(&MATRIX_IDENTITY);
        mvhd_p.extend_from_slice(&[0u8; 24]);
        mvhd_p.extend_from_slice(&2u32.to_be_bytes());
        let mvhd = mk_full_box(b"mvhd", 0, 0, &mvhd_p);

        // tkhd payload in field order: creation/modification time, track_ID, reserved, duration,
        // the reserved/layer/alternate_group/volume block, the unity matrix, then the 16.16
        // display width and height. Its flags mark the track enabled and in_movie.
        let mut tkhd_p = Vec::new();
        tkhd_p.extend_from_slice(&[0u8; 8]);
        tkhd_p.extend_from_slice(&1u32.to_be_bytes());
        tkhd_p.extend_from_slice(&[0u8; 4]);
        tkhd_p.extend_from_slice(&0u32.to_be_bytes());
        tkhd_p.extend_from_slice(&[0u8; 16]);
        tkhd_p.extend_from_slice(&MATRIX_IDENTITY);
        tkhd_p.extend_from_slice(&(cfg.width << 16).to_be_bytes());
        tkhd_p.extend_from_slice(&(cfg.height << 16).to_be_bytes());
        let tkhd = mk_full_box(b"tkhd", 0, 3, &tkhd_p);

        // mdhd payload in field order: creation/modification time, timescale, duration,
        // language (0x55c4 = "und"), pre_defined.
        let mut mdhd_p = Vec::new();
        mdhd_p.extend_from_slice(&[0u8; 8]);
        mdhd_p.extend_from_slice(&TIMESCALE.to_be_bytes());
        mdhd_p.extend_from_slice(&0u32.to_be_bytes());
        mdhd_p.extend_from_slice(&0x55c4u16.to_be_bytes());
        mdhd_p.extend_from_slice(&[0u8; 2]);
        let mdhd = mk_full_box(b"mdhd", 0, 0, &mdhd_p);

        // hdlr payload in field order: pre_defined, the 'vide' handler type, reserved, and the
        // handler name.
        let mut hdlr_p = Vec::new();
        hdlr_p.extend_from_slice(&[0u8; 4]);
        hdlr_p.extend_from_slice(b"vide");
        hdlr_p.extend_from_slice(&[0u8; 12]);
        hdlr_p.extend_from_slice(b"pixelflux\0");
        let hdlr = mk_full_box(b"hdlr", 0, 0, &hdlr_p);

        let vmhd = mk_full_box(b"vmhd", 0, 1, &[0u8; 8]);
        // An empty 'url ' entry with flag 1 declares the media self-contained.
        let url = mk_full_box(b"url ", 0, 1, &[]);
        let mut dref_p = 1u32.to_be_bytes().to_vec();
        dref_p.extend_from_slice(&url);
        let dref = mk_full_box(b"dref", 0, 0, &dref_p);
        let dinf = mk_box(b"dinf", &dref);

        let mut stsd_p = 1u32.to_be_bytes().to_vec();
        stsd_p.extend_from_slice(&cfg.sample_entry);
        let stsd = mk_full_box(b"stsd", 0, 0, &stsd_p);
        let stts = mk_full_box(b"stts", 0, 0, &0u32.to_be_bytes());
        let stsc = mk_full_box(b"stsc", 0, 0, &0u32.to_be_bytes());
        let stsz = mk_full_box(b"stsz", 0, 0, &[0u8; 8]);
        let stco = mk_full_box(b"stco", 0, 0, &0u32.to_be_bytes());
        let stbl = mk_box(
            b"stbl",
            &[stsd, stts, stsc, stsz, stco].concat(),
        );

        let minf = mk_box(b"minf", &[vmhd, dinf, stbl].concat());
        let mdia = mk_box(b"mdia", &[mdhd, hdlr, minf].concat());
        let trak = mk_box(b"trak", &[tkhd, mdia].concat());

        // trex payload in field order: track_ID, default_sample_description_index, then zeroed
        // default sample duration, size and flags (each fragment states its own).
        let mut trex_p = Vec::new();
        trex_p.extend_from_slice(&1u32.to_be_bytes());
        trex_p.extend_from_slice(&1u32.to_be_bytes());
        trex_p.extend_from_slice(&[0u8; 12]);
        let trex = mk_full_box(b"trex", 0, 0, &trex_p);
        let mvex = mk_box(b"mvex", &trex);

        let moov = mk_box(b"moov", &[mvhd, trak, mvex].concat());

        self.out.write_all(&ftyp)?;
        self.out.write_all(&moov)?;
        self.stats.bytes += (ftyp.len() + moov.len()) as u64;
        self.wrote_init = true;
        Ok(())
    }

    /// Queue one sample at `pts_us` (wall-clock microseconds since recording start),
    /// flushing the previously buffered sample with its now-known duration. Timestamps are
    /// clamped strictly monotonic so a repeated or reordered clock can never emit a
    /// zero/negative duration.
    pub fn push_sample(&mut self, data: Vec<u8>, sync: bool, pts_us: u64) -> std::io::Result<()> {
        let mut dts = pts_us * (TIMESCALE as u64 / 1000) / 1000;
        if let Some(last) = self.last_dts
            && dts <= last {
                dts = last + 1;
            }
        self.last_dts = Some(dts);
        if let Some(prev) = self.pending.take() {
            let duration = (dts - prev.dts).min(u32::MAX as u64) as u32;
            self.durations.push(duration);
            self.write_fragment(&prev, duration)?;
        }
        self.pending = Some(PendingSample { data, sync, dts });
        Ok(())
    }

    /// Median of the observed inter-frame durations (default 1/30 s when fewer than two
    /// frames were pushed), used to close the final sample.
    fn median_duration(&self) -> u32 {
        if self.durations.is_empty() {
            return DEFAULT_LAST_DURATION;
        }
        let mut sorted = self.durations.clone();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }

    fn write_fragment(&mut self, s: &PendingSample, duration: u32) -> std::io::Result<()> {
        self.seq += 1;

        // tfhd carries only the track_ID; its flags set default-base-is-moof, so sample offsets
        // are relative to the start of this moof.
        let mut tfhd_p = Vec::new();
        tfhd_p.extend_from_slice(&1u32.to_be_bytes());
        let tfhd = mk_full_box(b"tfhd", 0, 0x020000, &tfhd_p);

        let mut tfdt_p = Vec::new();
        tfdt_p.extend_from_slice(&s.dts.to_be_bytes());
        let tfdt = mk_full_box(b"tfdt", 1, 0, &tfdt_p);

        // sample flags: sync = "depends on nothing"; non-sync also sets the non-sync bit.
        let sample_flags: u32 = if s.sync { 0x0200_0000 } else { 0x0101_0000 };
        // trun payload in field order: sample_count, a data_offset placeholder patched in below,
        // then this sample's duration, size and flags — exactly the fields its flags select
        // (data-offset | sample-duration | sample-size | sample-flags).
        let mut trun_p = Vec::new();
        trun_p.extend_from_slice(&1u32.to_be_bytes());
        trun_p.extend_from_slice(&0i32.to_be_bytes());
        trun_p.extend_from_slice(&duration.to_be_bytes());
        trun_p.extend_from_slice(&(s.data.len() as u32).to_be_bytes());
        trun_p.extend_from_slice(&sample_flags.to_be_bytes());
        let mut trun = mk_full_box(b"trun", 0, 0x000701, &trun_p);

        let traf_len = 8 + tfhd.len() + tfdt.len() + trun.len();
        // A moof box header is 8 bytes and the mfhd inside it 16.
        let moof_len = 8 + 16 + traf_len;
        // First sample byte sits just past the mdat header, relative to moof start.
        let data_offset = (moof_len + 8) as i32;
        // The data_offset field sits in the trun payload directly after sample_count.
        let off_pos = trun.len() - trun_p.len() + 4;
        trun[off_pos..off_pos + 4].copy_from_slice(&data_offset.to_be_bytes());

        let mfhd = mk_full_box(b"mfhd", 0, 0, &self.seq.to_be_bytes());
        let traf = mk_box(b"traf", &[tfhd, tfdt, trun].concat());
        let moof = mk_box(b"moof", &[mfhd, traf].concat());
        debug_assert_eq!(moof.len(), moof_len);

        self.out.write_all(&moof)?;
        self.out.write_all(&((8 + s.data.len()) as u32).to_be_bytes())?;
        self.out.write_all(b"mdat")?;
        self.out.write_all(&s.data)?;
        self.out.flush()?;

        self.stats.samples += 1;
        if s.sync {
            self.stats.sync_samples += 1;
        }
        self.stats.bytes += (moof.len() + 8 + s.data.len()) as u64;
        self.stats.duration_us = (s.dts + duration as u64) * 1000 / (TIMESCALE as u64 / 1000);
        Ok(())
    }

    /// Flush the final buffered sample, closed with the MEDIAN observed inter-frame
    /// duration (its successor never arrives), and return the aggregate counters.
    pub fn finish(mut self) -> std::io::Result<Mp4Stats> {
        if let Some(prev) = self.pending.take() {
            let d = self.median_duration();
            self.write_fragment(&prev, d)?;
        }
        self.out.flush()?;
        Ok(self.stats)
    }
}

/// H.264-specific front end: captures SPS/PPS from the stream, gates output on the first
/// IDR, converts Annex-B access units to AVCC samples, and builds the `avc1` sample entry.
pub struct H264SampleBuilder {
    sps: Option<Vec<u8>>,
    pps: Option<Vec<u8>>,
}

/// One converted access unit ready for the fragment writer.
pub struct BuiltSample {
    pub data: Vec<u8>,
    pub sync: bool,
}

impl Default for H264SampleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl H264SampleBuilder {
    pub fn new() -> Self {
        Self { sps: None, pps: None }
    }

    pub fn have_parameter_sets(&self) -> bool {
        self.sps.is_some() && self.pps.is_some()
    }

    /// Convert one Annex-B access unit into a length-prefixed AVCC sample, harvesting
    /// SPS/PPS on the way. Returns `None` for an AU with no slice data (e.g. bare parameter
    /// sets). `sync` is true when the AU contains an IDR slice.
    pub fn build_sample(&mut self, annexb: &[u8]) -> Option<BuiltSample> {
        let nals = split_annexb(annexb);
        let mut data = Vec::with_capacity(annexb.len() + 8);
        let mut sync = false;
        let mut has_slice = false;
        for nal in nals {
            if nal.is_empty() {
                continue;
            }
            match nal[0] & 0x1f {
                7 => {
                    if self.sps.as_deref() != Some(nal) {
                        self.sps = Some(nal.to_vec());
                    }
                }
                8 => {
                    if self.pps.as_deref() != Some(nal) {
                        self.pps = Some(nal.to_vec());
                    }
                }
                5 => {
                    sync = true;
                    has_slice = true;
                }
                1 => has_slice = true,
                _ => {}
            }
            data.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            data.extend_from_slice(nal);
        }
        if !has_slice {
            return None;
        }
        Some(BuiltSample { data, sync })
    }

    /// Build the `avc1` sample entry + `avcC` record from the captured parameter sets, with
    /// the display dimensions parsed from the SPS.
    pub fn track_config(&self) -> Option<TrackConfig> {
        let sps = self.sps.as_deref()?;
        let pps = self.pps.as_deref()?;
        let (width, height) = parse_sps_dimensions(sps)?;

        // avcC header in field order: configurationVersion, AVCProfileIndication,
        // profile_compatibility and AVCLevelIndication taken straight from the SPS,
        // lengthSizeMinusOne = 3 (the 4-byte NAL lengths this muxer writes), and
        // numOfSequenceParameterSets = 1. The SPS then the PPS follow, each length-prefixed.
        let mut avcc_p = vec![1, sps[1], sps[2], sps[3], 0xff, 0xe1];
        avcc_p.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        avcc_p.extend_from_slice(sps);
        // numOfPictureParameterSets
        avcc_p.push(1);
        avcc_p.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        avcc_p.extend_from_slice(pps);
        let avcc = mk_box(b"avcC", &avcc_p);

        // avc1 sample entry in field order: reserved, data_reference_index, the
        // pre_defined/reserved block, width and height, horizontal and vertical resolution
        // (72 dpi), reserved, frame_count, compressorname, depth 24, pre_defined.
        let mut entry_p = Vec::new();
        entry_p.extend_from_slice(&[0u8; 6]);
        entry_p.extend_from_slice(&1u16.to_be_bytes());
        entry_p.extend_from_slice(&[0u8; 16]);
        entry_p.extend_from_slice(&(width as u16).to_be_bytes());
        entry_p.extend_from_slice(&(height as u16).to_be_bytes());
        entry_p.extend_from_slice(&0x0048_0000u32.to_be_bytes());
        entry_p.extend_from_slice(&0x0048_0000u32.to_be_bytes());
        entry_p.extend_from_slice(&[0u8; 4]);
        entry_p.extend_from_slice(&1u16.to_be_bytes());
        entry_p.extend_from_slice(&[0u8; 32]);
        entry_p.extend_from_slice(&0x0018u16.to_be_bytes());
        entry_p.extend_from_slice(&(-1i16).to_be_bytes());
        entry_p.extend_from_slice(&avcc);
        let entry = mk_box(b"avc1", &entry_p);

        Some(TrackConfig { sample_entry: entry, width, height })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // x264 SPS at 1284x722 (High 4:2:0: cropping in both axes on odd-macroblock dims).
    const SPS_HIGH_1284X722: &str = "67640020acd9405105de788c0440000003004000000f03c60c6580";
    // x264 SPS at 640x360 (Constrained Baseline).
    const SPS_BASE_640X360: &str = "6742c01ed900a02ff970110000030001000003003c0f162e48";
    // x264 SPS at 1920x1080 (High 4:4:4 Predictive: chroma_format_idc == 3 path).
    const SPS_444_1920X1080: &str = "67f40028919b280f0044fc4e0220000003002000000781e30632c0";

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// Both start-code lengths delimit NALs; payloads come back exactly, with no framing.
    #[test]
    fn split_annexb_handles_mixed_start_codes() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 1, 0x67, 0xAA, 0xBB]);
        data.extend_from_slice(&[0, 0, 1, 0x68, 0xCC]);
        data.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x00, 0x00, 0x03, 0x01, 0xDD]);
        let nals = split_annexb(&data);
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0], &[0x67, 0xAA, 0xBB]);
        assert_eq!(nals[1], &[0x68, 0xCC]);
        assert_eq!(nals[2], &[0x65, 0x00, 0x00, 0x03, 0x01, 0xDD]);
    }

    /// Dimensions from real x264 SPS across the profiles the project's encoders emit,
    /// including the frame-cropping and 4:4:4 chroma paths.
    #[test]
    fn sps_dimensions_across_profiles() {
        assert_eq!(parse_sps_dimensions(&hex(SPS_HIGH_1284X722)), Some((1284, 722)));
        assert_eq!(parse_sps_dimensions(&hex(SPS_BASE_640X360)), Some((640, 360)));
        assert_eq!(parse_sps_dimensions(&hex(SPS_444_1920X1080)), Some((1920, 1080)));
    }

    /// Annex-B -> AVCC: every NAL is length-prefixed, IDR marks sync, parameter sets are
    /// harvested, and a parameter-set-only AU yields no sample.
    #[test]
    fn annexb_to_avcc_sample() {
        let sps = hex(SPS_BASE_640X360);
        let mut au = Vec::new();
        au.extend_from_slice(&[0, 0, 0, 1]);
        au.extend_from_slice(&sps);
        au.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x38, 0x80]);
        au.extend_from_slice(&[0, 0, 0, 1, 0x65, 1, 2, 3, 4]);

        let mut b = H264SampleBuilder::new();
        assert!(b.build_sample(&[0u8, 0, 0, 1, 0x67, 0x42, 0xc0, 0x1e, 0xd9]).is_none());
        let s = b.build_sample(&au).expect("IDR AU builds a sample");
        assert!(s.sync);
        assert!(b.have_parameter_sets());
        // Sample = 3 length-prefixed NALs, sizes preserved.
        let mut off = 0usize;
        let mut sizes = Vec::new();
        while off < s.data.len() {
            let n = u32::from_be_bytes(s.data[off..off + 4].try_into().unwrap()) as usize;
            sizes.push(n);
            off += 4 + n;
        }
        assert_eq!(off, s.data.len());
        assert_eq!(sizes, vec![sps.len(), 4, 5]);

        let p = b.build_sample(&[0u8, 0, 0, 1, 0x41, 9, 9]).unwrap();
        assert!(!p.sync);
    }

    /// Walk the top-level boxes of a finished two-sample stream: init once, then one
    /// moof+mdat pair per sample, with sizes that exactly tile the buffer.
    #[test]
    fn fragment_stream_box_layout() {
        let mut b = H264SampleBuilder::new();
        let mut au = Vec::new();
        au.extend_from_slice(&[0, 0, 0, 1]);
        au.extend_from_slice(&hex(SPS_BASE_640X360));
        au.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x38, 0x80]);
        au.extend_from_slice(&[0, 0, 0, 1, 0x65, 1, 2, 3, 4]);
        let s1 = b.build_sample(&au).unwrap();
        let s2 = b.build_sample(&[0u8, 0, 0, 1, 0x41, 5, 6, 7]).unwrap();

        let mut buf = Vec::new();
        let mut w = FragmentWriter::new(&mut buf);
        w.write_init(&b.track_config().unwrap()).unwrap();
        w.push_sample(s1.data, s1.sync, 0).unwrap();
        w.push_sample(s2.data, s2.sync, 33_000).unwrap();
        let stats = w.finish().unwrap();
        assert_eq!(stats.samples, 2);
        assert_eq!(stats.sync_samples, 1);
        assert_eq!(stats.bytes as usize, buf.len());

        let mut kinds = Vec::new();
        let mut off = 0usize;
        while off < buf.len() {
            let size = u32::from_be_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
            kinds.push(buf[off + 4..off + 8].to_vec());
            assert!(size >= 8 && off + size <= buf.len());
            off += size;
        }
        assert_eq!(off, buf.len());
        let names: Vec<&str> = kinds.iter().map(|k| std::str::from_utf8(k).unwrap()).collect();
        assert_eq!(names, vec!["ftyp", "moov", "moof", "mdat", "moof", "mdat"]);
    }

    /// Every trun sample_duration in stream order (the writer emits one sample per trun).
    fn trun_durations(buf: &[u8]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut off = 0usize;
        while off + 24 <= buf.len() {
            if &buf[off + 4..off + 8] == b"trun" {
                // [size][fourcc][ver+flags][sample_count][data_offset][duration]
                out.push(u32::from_be_bytes(buf[off + 20..off + 24].try_into().unwrap()));
            }
            off += 1;
        }
        out
    }

    /// Each flushed sample's duration is the pts delta to its successor, and the final
    /// buffered sample closes with the MEDIAN observed duration — not the last delta, which
    /// under damage-driven capture is an outlier as often as not.
    #[test]
    fn sample_durations_are_pts_deltas_with_median_tail() {
        let mut buf = Vec::new();
        let mut w = FragmentWriter::new(&mut buf);
        let cfg = {
            let mut b = H264SampleBuilder::new();
            let mut au = vec![0, 0, 0, 1];
            au.extend_from_slice(&hex(SPS_BASE_640X360));
            au.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x38, 0x80]);
            au.extend_from_slice(&[0, 0, 0, 1, 0x65, 1]);
            b.build_sample(&au);
            b.track_config().unwrap()
        };
        w.write_init(&cfg).unwrap();
        // 33 ms, 33 ms, then a 300 ms static gap before the final frame.
        w.push_sample(vec![0, 0, 0, 1, 0x65], true, 0).unwrap();
        w.push_sample(vec![0, 0, 0, 1, 0x41], false, 33_000).unwrap();
        w.push_sample(vec![0, 0, 0, 1, 0x41], false, 66_000).unwrap();
        w.push_sample(vec![0, 0, 0, 1, 0x41], false, 366_000).unwrap();
        let stats = w.finish().unwrap();
        assert_eq!(stats.samples, 4);
        // 90 kHz ticks: 33 ms = 2970. The tail closes at the median (2970), not 27000.
        assert_eq!(trun_durations(&buf), vec![2970, 2970, 27_000, 2970]);
    }

    /// A single-frame recording still closes with a sane nonzero duration.
    #[test]
    fn single_sample_uses_default_duration() {
        let mut buf = Vec::new();
        let mut w = FragmentWriter::new(&mut buf);
        let cfg = {
            let mut b = H264SampleBuilder::new();
            let mut au = vec![0, 0, 0, 1];
            au.extend_from_slice(&hex(SPS_BASE_640X360));
            au.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x38, 0x80]);
            au.extend_from_slice(&[0, 0, 0, 1, 0x65, 1]);
            b.build_sample(&au);
            b.track_config().unwrap()
        };
        w.write_init(&cfg).unwrap();
        w.push_sample(vec![0, 0, 0, 1, 0x65], true, 0).unwrap();
        let stats = w.finish().unwrap();
        assert_eq!(stats.samples, 1);
        assert_eq!(trun_durations(&buf), vec![DEFAULT_LAST_DURATION]);
    }

    /// Non-monotonic wall-clock input is clamped to strictly increasing decode times, so no
    /// fragment can carry a zero or negative duration.
    #[test]
    fn pts_clamped_strictly_monotonic() {
        let mut buf = Vec::new();
        let mut w = FragmentWriter::new(&mut buf);
        let cfg = {
            let mut b = H264SampleBuilder::new();
            let mut au = vec![0, 0, 0, 1];
            au.extend_from_slice(&hex(SPS_BASE_640X360));
            au.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x38, 0x80]);
            au.extend_from_slice(&[0, 0, 0, 1, 0x65, 1]);
            b.build_sample(&au);
            b.track_config().unwrap()
        };
        w.write_init(&cfg).unwrap();
        w.push_sample(vec![0, 0, 0, 1, 0x65], true, 1000).unwrap();
        // The second sample repeats the clock and the third winds it backwards.
        w.push_sample(vec![0, 0, 0, 1, 0x41], false, 1000).unwrap();
        w.push_sample(vec![0, 0, 0, 1, 0x41], false, 500).unwrap();
        let stats = w.finish().unwrap();
        assert_eq!(stats.samples, 3);

        // Extract each tfdt baseMediaDecodeTime and check strict monotonicity.
        let mut times = Vec::new();
        let mut off = 0usize;
        while off + 8 <= buf.len() {
            if &buf[off + 4..off + 8] == b"tfdt" {
                let t = u64::from_be_bytes(buf[off + 12..off + 20].try_into().unwrap());
                times.push(t);
            }
            off += 1;
        }
        assert_eq!(times.len(), 3);
        assert!(times.windows(2).all(|w| w[1] > w[0]), "tfdt times: {times:?}");
    }
}
