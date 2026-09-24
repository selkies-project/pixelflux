//! The session against the stood-in driver: what it asks the driver for, what it renders
//! for a key frame and a delta, how each codec names the frame a picture predicts from once
//! a client has lost one, and what the video processor is told to convert to.

use super::mock::{self, Driver};
use super::*;
use crate::encoders::codec::{parse_video_type, FRAME_DELTA, FRAME_KEY};
use crate::encoders::sps::{h264_frame_num_range, h264_max_num_ref_frames, read_color, ColorSignal};

const W: i32 = 320;
const H: i32 = 240;

fn settings(codec: Codec, cbr: bool) -> RustCaptureSettings {
    RustCaptureSettings {
        width: W,
        height: H,
        target_fps: 30.0,
        codec,
        video_crf: 25,
        video_cbr_mode: cbr,
        video_bitrate_kbps: 4000,
        ..Default::default()
    }
}

/// The stood-in driver on a placeholder node.
fn device() -> Arc<Device> {
    let node = std::fs::File::open("/dev/null").unwrap();
    Arc::new(Device::on(mock::api(), node.into(), "stand-in").unwrap())
}

fn open(codec: Codec, settings: &RustCaptureSettings) -> Result<VaapiEncoder, String> {
    VaapiEncoder::on_device(device(), settings, codec, Input::Host { rgba: false })
}

fn session(codec: Codec, cbr: bool) -> VaapiEncoder {
    open(codec, &settings(codec, cbr)).unwrap_or_else(|e| panic!("{codec:?}: {e}"))
}

fn frame() -> Vec<u8> {
    vec![0x40; (W * H * 4) as usize]
}

fn encode(enc: &mut VaapiEncoder, t: u64, key: bool) -> Vec<u8> {
    enc.encode_host(&frame(), (W * 4) as usize, false, t, 25, key).unwrap_or_else(|e| panic!("frame {t}: {e}"))
}

/// A most-significant-bit-first reader for the headers the checks parse back.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn u(&mut self, count: u32) -> u32 {
        (0..count).fold(0, |acc, _| {
            let bit = (self.bytes[self.pos / 8] >> (7 - self.pos % 8)) & 1;
            self.pos += 1;
            (acc << 1) | bit as u32
        })
    }

    fn ue(&mut self) -> u32 {
        let mut zeros = 0;
        while self.u(1) == 0 {
            zeros += 1;
        }
        (1 << zeros) - 1 + self.u(zeros)
    }

    fn se(&mut self) -> i32 {
        let k = self.ue() as i32;
        if k % 2 == 1 { (k + 1) / 2 } else { -(k / 2) }
    }
}

/// The RBSP of one NAL unit of `kind` in an Annex-B stream, unescaped.
fn nal(stream: &[u8], kind: u8, h265: bool) -> Vec<u8> {
    let unit = crate::encoders::codec::annexb_nals(stream)
        .find(|n| (if h265 { (n[0] >> 1) & 0x3f } else { n[0] & 0x1f }) == kind)
        .unwrap_or_else(|| panic!("no NAL unit of type {kind}"));
    let payload = &unit[if h265 { 2 } else { 1 }..];
    let mut out = Vec::new();
    let mut zeros = 0;
    for &b in payload {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

/// Every codec comes up on the stood-in driver, is listed by the probe, and asks the driver
/// for what its arm needs: the surface format, the rate-control mode, the packed headers it
/// writes itself, and contexts over the reconstruction surfaces and the converted one.
#[test]
fn every_codec_comes_up_and_asks_for_what_it_needs() {
    for codec in Codec::VIDEO {
        mock::reset(Driver::generous());
        let (served, fullcolor): (Vec<Codec>, Vec<bool>) = probe_codecs_on(&device()).unwrap().into_iter().unzip();
        assert_eq!(served, Codec::VIDEO.to_vec());
        assert_eq!(fullcolor, [false, false, true, false, true], "HEVC Main 4:4:4 and VP9 profile 1 carry 4:4:4");
        let enc = session(codec, false);
        assert_eq!(enc.codec(), codec);
        assert!(!enc.is_fullcolor() && !enc.is_full_range());
        assert!(enc.low_power(), "{codec:?} takes the low-power entry point offered");
        mock::with(|d| {
            let (profile, entrypoint, attribs) = &d.configs[0];
            assert_eq!(*entrypoint, VAEntrypointEncSliceLP);
            assert_eq!(*profile, profile_ladder(codec, false)[0], "{codec:?}");
            let value = |kind| attribs.iter().find(|a| a.type_ == kind).map(|a| a.value);
            assert_eq!(value(VAConfigAttribRTFormat), Some(VA_RT_FORMAT_YUV420));
            assert_eq!(value(VAConfigAttribRateControl), Some(VA_RC_CQP));
            let packed = match codec {
                Codec::H264 | Codec::H265 => Some(VA_ENC_PACKED_HEADER_SEQUENCE | VA_ENC_PACKED_HEADER_SLICE),
                Codec::Av1 => Some(VA_ENC_PACKED_HEADER_SEQUENCE | VA_ENC_PACKED_HEADER_PICTURE),
                _ => None,
            };
            assert_eq!(value(VAConfigAttribEncPackedHeaders), packed, "{codec:?}");
            assert_eq!(d.configs[1].0, VAProfileNone, "the video processor's configuration");
            let recon = if codec == Codec::Vp8 { 4 } else { REFERENCE_FRAMES as usize + 1 };
            assert_eq!(d.contexts[0].1.len(), recon + 1, "{codec:?}: the encode context over the reconstruction and converted surfaces");
            assert_eq!(d.contexts[1].1.len(), 1, "the processing context over the converted surface");
        });
    }
}

/// A key frame renders the sequence parameters, the rate control the session holds, and the
/// packed sequence header; a delta renders the picture alone. Every frame converts first.
#[test]
fn a_key_frame_carries_the_sequence_and_a_delta_does_not() {
    for cbr in [false, true] {
        for codec in Codec::VIDEO {
            mock::reset(Driver::generous());
            let mut enc = session(codec, cbr);
            let first = encode(&mut enc, 0, true);
            assert_eq!(parse_video_type(first[1]), Some((codec, FRAME_KEY)), "{codec:?}");
            mock::with(|d| {
                assert_eq!(d.pictures.len(), 2, "{codec:?}: a convert and an encode");
                assert_ne!(d.pictures[0].0, d.pictures[1].0, "the convert and the encode render on their own contexts");
                assert!(d.contexts[1].1.contains(&d.pictures[0].1), "the convert targets the converted surface");
                assert!(d.contexts[0].1.contains(&d.pictures[1].1), "the encode targets a reconstruction surface");
                assert_eq!(d.last_buffers(VAEncSequenceParameterBufferType).len(), 1, "{codec:?}");
                let misc: Vec<u32> = d.last_misc().into_iter().map(|m| m.0).collect();
                let mut wanted = if cbr { vec![VAEncMiscParameterTypeRateControl, VAEncMiscParameterTypeHRD] } else { vec![] };
                wanted.extend([VAEncMiscParameterTypeFrameRate, VAEncMiscParameterTypeQualityLevel]);
                assert_eq!(misc, wanted, "{codec:?} cbr={cbr}");
                for (kind, bytes) in d.last_misc() {
                    if kind == VAEncMiscParameterTypeRateControl {
                        let rc: VAEncMiscParameterRateControl = unsafe { ptr::read_unaligned(bytes.as_ptr() as *const _) };
                        assert_eq!((rc.bits_per_second, rc.target_percentage), (4_000_000, 100));
                        assert_eq!(rc.window_size, 50, "1.5 frames of VBV at 30 fps, in ms");
                        assert_eq!(unsafe { rc.rc_flags.bits.mb_rate_control() }, 2);
                    }
                    if kind == VAEncMiscParameterTypeHRD {
                        let hrd: VAEncMiscParameterHRD = unsafe { ptr::read_unaligned(bytes.as_ptr() as *const _) };
                        assert_eq!((hrd.buffer_size, hrd.initial_buffer_fullness), (200_000, 200_000));
                    }
                    if kind == VAEncMiscParameterTypeFrameRate {
                        let fr: VAEncMiscParameterFrameRate = unsafe { ptr::read_unaligned(bytes.as_ptr() as *const _) };
                        assert_eq!(fr.framerate, (1 << 16) | 30);
                    }
                    if kind == VAEncMiscParameterTypeQualityLevel {
                        let q: VAEncMiscParameterBufferQualityLevel = unsafe { ptr::read_unaligned(bytes.as_ptr() as *const _) };
                        assert_eq!(q.quality_level, 6);
                    }
                }
                let packed: Vec<u32> = d.last_packed().into_iter().map(|p| p.0).collect();
                let wanted: Vec<u32> = match codec {
                    Codec::H264 | Codec::H265 => [vec![VAEncPackedHeaderSequence], vec![VAEncPackedHeaderSlice; 4]].concat(),
                    Codec::Av1 => vec![VAEncPackedHeaderSequence, VAEncPackedHeaderPicture],
                    _ => vec![],
                };
                assert_eq!(packed, wanted, "{codec:?}");
                assert_eq!(d.last_buffers(VAEncPictureParameterBufferType).len(), 1);
                let slices = d.last_buffers(VAEncSliceParameterBufferType).len();
                assert_eq!(slices, match codec { Codec::H264 | Codec::H265 => 4, Codec::Av1 => 1, _ => 0 }, "{codec:?}");
            });
            let second = encode(&mut enc, 1, false);
            assert_eq!(parse_video_type(second[1]), Some((codec, FRAME_DELTA)), "{codec:?}");
            mock::with(|d| {
                assert_eq!(d.pictures.len(), 4);
                assert!(d.last_buffers(VAEncSequenceParameterBufferType).is_empty(), "{codec:?}: a delta repeats no sequence");
                assert!(d.last_misc().is_empty(), "{codec:?}: a delta repeats no rate control");
                assert_eq!(d.last_buffers(VAEncPictureParameterBufferType).len(), 1);
            });
        }
    }
}

/// H.264 keeps the decoded picture buffer the level admits, names the newest surviving frame
/// in the slice header once a client lost one, and codes a key frame once the loss reaches
/// past the buffer. What the session writes into the SPS is what the crate's own reader and
/// the loss bookkeeping read back.
#[test]
fn h264_names_the_newest_surviving_frame_in_its_slice_header() {
    mock::reset(Driver::generous());
    let mut enc = session(Codec::H264, false);
    let first = encode(&mut enc, 0, true);
    let stream = &first[VIDEO_HEADER_LEN..];
    assert_eq!(h264_max_num_ref_frames(stream), Some(REFERENCE_FRAMES));
    assert_eq!(h264_frame_num_range(stream), Some(65536));
    assert_eq!(read_color(crate::encoders::codec::annexb_nals(stream).find(|n| n[0] & 0x1f == 7).unwrap()), Some(ColorSignal::BT709_LIMITED));
    assert_eq!(enc.last_reference(), Reference::None);
    for t in 1..8u64 {
        encode(&mut enc, t, false);
        assert_eq!(enc.last_reference(), Reference::Frame(t as u16 - 1));
    }
    assert!(enc.invalidate_reference(5));
    let out = encode(&mut enc, 8, false);
    assert_eq!(parse_video_type(out[1]), Some((Codec::H264, FRAME_DELTA)));
    assert_eq!(enc.last_reference(), Reference::Frame(4));
    mock::with(|d| {
        let pic: VAEncPictureParameterBufferH264 = d.last_param(VAEncPictureParameterBufferType).unwrap();
        assert_eq!(pic.CurrPic.frame_idx, 8);
        let held: Vec<u32> = pic.ReferenceFrames.iter().filter(|r| r.flags != VA_PICTURE_H264_INVALID).map(|r| r.frame_idx).collect();
        assert_eq!(held, [7, 6, 5, 4, 3, 2, 1, 0], "every frame the decoder holds, newest first");
        let slices = d.last_buffers(VAEncSliceParameterBufferType);
        assert_eq!(slices.len(), 4);
        let slice: VAEncSliceParameterBufferH264 = unsafe { ptr::read_unaligned(slices[0].as_ptr() as *const _) };
        assert_eq!(slice.RefPicList0[0].frame_idx, 4, "the slice predicts from frame 4");
        assert_eq!(slice.slice_type, 0);
        let header = d.last_packed().into_iter().find(|p| p.0 == VAEncPackedHeaderSlice).unwrap().1;
        let rbsp = nal(&header, 1, false);
        let mut r = Reader { bytes: &rbsp, pos: 0 };
        assert_eq!(r.ue(), 0, "first_mb_in_slice");
        assert_eq!(r.ue(), 5, "slice_type P");
        assert_eq!(r.ue(), 0, "pps id");
        assert_eq!(r.u(16), 8, "frame_num");
        assert_eq!(r.u(1), 0, "num_ref_idx_active_override_flag");
        assert_eq!(r.u(1), 1, "ref_pic_list_modification_flag_l0");
        assert_eq!(r.ue(), 0, "modification_of_pic_nums_idc: subtract");
        assert_eq!(r.ue(), 3, "abs_diff_pic_num_minus1: 8 - 4 - 1");
        assert_eq!(r.ue(), 3, "end of the modification");
        assert_eq!(r.u(1), 0, "adaptive_ref_pic_marking_mode_flag");
        assert_eq!(r.ue(), 0, "cabac_init_idc");
        assert_eq!(r.se(), 25 - 26, "slice_qp_delta");
    });
    encode(&mut enc, 9, false);
    assert_eq!(enc.last_reference(), Reference::Frame(8));
    mock::with(|d| {
        let header = d.last_packed().into_iter().find(|p| p.0 == VAEncPackedHeaderSlice).unwrap().1;
        let rbsp = nal(&header, 1, false);
        let mut r = Reader { bytes: &rbsp, pos: 0 };
        r.ue();
        r.ue();
        r.ue();
        r.u(16);
        r.u(1);
        assert_eq!(r.u(1), 0, "the previous frame is the default reference");
    });
    for t in 10..20u64 {
        encode(&mut enc, t, false);
    }
    assert!(enc.invalidate_reference(9));
    let out = encode(&mut enc, 20, false);
    assert_eq!(parse_video_type(out[1]), Some((Codec::H264, FRAME_KEY)), "a loss past the buffer costs a key frame");
    assert_eq!(enc.last_reference(), Reference::None);
    mock::with(|d| assert_eq!(d.last_buffers(VAEncSequenceParameterBufferType).len(), 1, "the key frame repeats the sequence"));
}

/// HEVC lists the frames the decoder keeps in every slice header's reference picture set,
/// the one it predicts from marked as used, and drops a lost frame from the set.
#[test]
fn hevc_lists_the_kept_frames_in_its_reference_picture_set() {
    mock::reset(Driver::generous());
    let mut enc = session(Codec::H265, false);
    for t in 0..8u64 {
        let out = encode(&mut enc, t, t == 0);
        assert_eq!(parse_video_type(out[1]), Some((Codec::H265, if t == 0 { FRAME_KEY } else { FRAME_DELTA })));
    }
    assert!(enc.invalidate_reference(5));
    encode(&mut enc, 8, false);
    assert_eq!(enc.last_reference(), Reference::Frame(4));
    mock::with(|d| {
        let pic: VAEncPictureParameterBufferHEVC = d.last_param(VAEncPictureParameterBufferType).unwrap();
        assert_eq!(pic.decoded_curr_pic.pic_order_cnt, 8);
        let kept: Vec<(i32, u32)> = pic.reference_frames.iter().filter(|r| r.flags != VA_PICTURE_HEVC_INVALID).map(|r| (r.pic_order_cnt, r.flags)).collect();
        assert_eq!(kept, [(4, VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE), (3, 0), (2, 0), (1, 0), (0, 0)], "the lost frames are not kept");
        let slices = d.last_buffers(VAEncSliceParameterBufferType);
        let slice: VAEncSliceParameterBufferHEVC = unsafe { ptr::read_unaligned(slices[0].as_ptr() as *const _) };
        assert_eq!(slice.ref_pic_list0[0].pic_order_cnt, 4);
        assert_eq!(slice.slice_type, 1);
        let header = d.last_packed().into_iter().find(|p| p.0 == VAEncPackedHeaderSlice).unwrap().1;
        let rbsp = nal(&header, 1, true);
        let mut r = Reader { bytes: &rbsp, pos: 0 };
        assert_eq!(r.u(1), 1, "first_slice_segment_in_pic_flag");
        assert_eq!(r.ue(), 0, "pps id");
        assert_eq!(r.ue(), 1, "slice_type P");
        assert_eq!(r.u(12), 8, "poc lsb");
        assert_eq!(r.u(1), 0, "short_term_ref_pic_set_sps_flag");
        assert_eq!(r.ue(), 5, "num_negative_pics");
        assert_eq!(r.ue(), 0, "num_positive_pics");
        let mut previous = 8;
        for (poc, used) in [(4, 1), (3, 0), (2, 0), (1, 0), (0, 0)] {
            assert_eq!(r.ue(), (previous - poc - 1) as u32, "delta_poc_s0_minus1 of {poc}");
            assert_eq!(r.u(1), used, "used_by_curr_pic_s0_flag of {poc}");
            previous = poc;
        }
    });
}

/// VP9 refreshes the slot of each frame's timestamp and predicts from the slot of the newest
/// surviving frame, every frame error-resilient.
#[test]
fn vp9_addresses_its_slots_by_timestamp() {
    mock::reset(Driver::generous());
    let mut enc = session(Codec::Vp9, false);
    for t in 0..8u64 {
        encode(&mut enc, t, t == 0);
        mock::with(|d| {
            let pic: VAEncPictureParameterBufferVP9 = d.last_param(VAEncPictureParameterBufferType).unwrap();
            assert_eq!(pic.refresh_frame_flags, if t == 0 { 0xff } else { 1 << t }, "frame {t}");
            assert_eq!(unsafe { pic.pic_flags.bits.error_resilient_mode() }, 1);
            if t > 0 {
                assert_eq!(unsafe { pic.ref_flags.bits.ref_last_idx() }, t as u32 - 1);
                assert_ne!(pic.reference_frames[t as usize - 1], VA_INVALID_SURFACE);
            }
        });
    }
    assert!(enc.invalidate_reference(5));
    encode(&mut enc, 8, false);
    assert_eq!(enc.last_reference(), Reference::Frame(4));
    mock::with(|d| {
        let pic: VAEncPictureParameterBufferVP9 = d.last_param(VAEncPictureParameterBufferType).unwrap();
        assert_eq!(unsafe { pic.ref_flags.bits.ref_last_idx() }, 4);
        assert_eq!(pic.refresh_frame_flags, 1 << 0, "timestamp 8 takes slot 0");
        assert_eq!(pic.luma_ac_qindex, Codec::Vp9.quantizer(25) as u8);
    });
}

/// VP8 follows the buffer plan: LAST every frame, the golden and altref anchors on their
/// schedules, no entropy update ever, and a recovery from an anchor once LAST is lost.
#[test]
fn vp8_follows_the_slot_plan() {
    mock::reset(Driver::generous());
    let mut enc = session(Codec::Vp8, false);
    let mut refreshes = Vec::new();
    for t in 0..9u64 {
        encode(&mut enc, t, t == 0);
        mock::with(|d| {
            let pic: VAEncPictureParameterBufferVP8 = d.last_param(VAEncPictureParameterBufferType).unwrap();
            let p = unsafe { pic.pic_flags.bits };
            assert_eq!(p.refresh_entropy_probs(), 0, "frame {t}");
            refreshes.push((p.refresh_last(), p.refresh_golden_frame(), p.refresh_alternate_frame()));
            if t > 0 {
                let r = unsafe { pic.ref_flags.bits };
                assert_eq!((r.no_ref_last(), r.no_ref_gf(), r.no_ref_arf()), (0, 1, 1), "frame {t} predicts from LAST");
            }
        });
    }
    assert_eq!(refreshes[0], (1, 1, 1), "the key frame refreshes every buffer");
    assert_eq!(refreshes[1], (1, 0, 0));
    assert_eq!(refreshes[2], (1, 1, 0), "frame 2 is a golden anchor");
    assert_eq!(refreshes[8], (1, 0, 1), "frame 8 is an altref anchor");
    // Frames 5 and 6 lost: LAST holds 8, GOLDEN 6, ALTREF 8, so nothing survives but the key
    // frame's golden... which frame 6 replaced; the next frame is a key frame.
    assert!(enc.invalidate_reference(5));
    let out = encode(&mut enc, 9, false);
    assert_eq!(parse_video_type(out[1]), Some((Codec::Vp8, FRAME_KEY)));
    // With a golden anchor older than the loss, the recovery frame predicts from it and
    // refreshes every buffer.
    for t in 10..14u64 {
        encode(&mut enc, t, false);
    }
    assert!(enc.invalidate_reference(12));
    encode(&mut enc, 14, false);
    assert_eq!(enc.last_reference(), Reference::Frame(11), "the golden anchor of frame 11");
    mock::with(|d| {
        let pic: VAEncPictureParameterBufferVP8 = d.last_param(VAEncPictureParameterBufferType).unwrap();
        let r = unsafe { pic.ref_flags.bits };
        let p = unsafe { pic.pic_flags.bits };
        assert_eq!((r.no_ref_last(), r.no_ref_gf(), r.no_ref_arf()), (1, 0, 1));
        assert_eq!((p.refresh_last(), p.refresh_golden_frame(), p.refresh_alternate_frame()), (1, 1, 1));
    });
}

/// AV1 names the slot of the frame it predicts from as every reference and as the source of
/// its probability contexts, in the picture parameters and in the frame header it packs.
#[test]
fn av1_names_its_reference_slot_in_the_frame_header() {
    mock::reset(Driver::generous());
    let mut enc = session(Codec::Av1, true);
    for t in 0..8u64 {
        let out = encode(&mut enc, t, t == 0);
        assert_eq!(parse_video_type(out[1]), Some((Codec::Av1, if t == 0 { FRAME_KEY } else { FRAME_DELTA })), "frame {t}");
    }
    assert!(enc.invalidate_reference(5));
    encode(&mut enc, 8, false);
    assert_eq!(enc.last_reference(), Reference::Frame(4));
    mock::with(|d| {
        let pic: VAEncPictureParameterBufferAV1 = d.last_param(VAEncPictureParameterBufferType).unwrap();
        assert_eq!(pic.ref_frame_idx, [4; 7]);
        assert_eq!(pic.primary_ref_frame, 0);
        assert_eq!(pic.refresh_frame_flags, 1 << 0);
        assert_eq!(pic.order_hint, 8);
        assert_ne!(pic.reference_frames[4], VA_INVALID_SURFACE);
        assert_eq!(pic.byte_offset_frame_hdr_obu_size, 1, "a delta's frame header is first in its picture");
        let header = d.last_packed().into_iter().find(|p| p.0 == VAEncPackedHeaderPicture).unwrap().1;
        assert_eq!(header[0], 0x1a, "a frame header OBU with a size field");
        let mut r = Reader { bytes: &header, pos: 8 * (1 + 4) };
        assert_eq!(r.u(1), 0, "show_existing_frame");
        assert_eq!(r.u(2), 1, "frame_type inter");
        assert_eq!(r.u(1), 1, "show_frame");
        assert_eq!(r.u(1), 0, "error_resilient_mode");
        assert_eq!(r.u(1), 0, "disable_cdf_update");
        assert_eq!(r.u(1), 0, "frame_size_override_flag");
        assert_eq!(r.u(8), 8, "order_hint");
        assert_eq!(r.u(3), 0, "primary_ref_frame");
        assert_eq!(r.u(8), 1, "refresh_frame_flags");
        assert_eq!(r.u(1), 0, "frame_refs_short_signaling");
        for i in 0..7 {
            assert_eq!(r.u(3), 4, "ref_frame_idx[{i}]");
        }
        // render_and_frame_size_different, allow_high_precision_mv, is_filter_switchable,
        // interpolation_filter, is_motion_mode_switchable, disable_frame_end_update_cdf,
        // uniform_tile_spacing_flag, and one increment bit each for columns and rows; no
        // use_ref_frame_mvs, the sequence having no reference motion vectors.
        assert_eq!(pic.bit_offset_qindex as usize, r.pos + 1 + 1 + 1 + 2 + 1 + 1 + 1 + 1 + 1, "the quantizer follows the render and filter bits and the tiling");
        r.pos = pic.bit_offset_qindex as usize;
        assert_eq!(r.u(8), 128, "a constant-rate session's quantizer, which the driver rewrites");
        assert_eq!(pic.size_in_bits_frame_hdr_obu as usize, 8 * header.len());
    });
    mock::reset(Driver::generous());
    let mut enc = session(Codec::Av1, true);
    encode(&mut enc, 0, true);
    mock::with(|d| {
        let pic: VAEncPictureParameterBufferAV1 = d.last_param(VAEncPictureParameterBufferType).unwrap();
        let sequence = d.last_packed().into_iter().find(|p| p.0 == VAEncPackedHeaderSequence).unwrap().1;
        assert_eq!(pic.byte_offset_frame_hdr_obu_size as usize, sequence.len() + 1, "a key frame's header follows the sequence header");
        assert_eq!(sequence[0], 0x0a, "a sequence header OBU with a size field");
        assert_eq!(pic.refresh_frame_flags, 0xff);
        assert_eq!(pic.primary_ref_frame, 7);
    });
}

/// The video processor is told the sRGB source it converts from and the matrix and range the
/// session declares, explicitly where the driver takes it, with chroma sited at the block
/// center.
#[test]
fn the_video_processor_is_told_the_declared_color() {
    for codec in [Codec::H264, Codec::Vp8] {
        mock::reset(Driver::generous());
        let mut enc = session(codec, false);
        encode(&mut enc, 0, true);
        mock::with(|d| {
            let convert = &d.pictures[0];
            let bytes = &d.buffers[convert.2[0]];
            assert_eq!(bytes.1, VAProcPipelineParameterBufferType);
            let p: VAProcPipelineParameterBuffer = unsafe { ptr::read_unaligned(bytes.2.as_ptr() as *const _) };
            assert_eq!((p.surface_color_standard, p.output_color_standard), (VAProcColorStandardExplicit, VAProcColorStandardExplicit));
            assert_eq!((p.input_color_properties.color_range, p.input_color_properties.matrix_coefficients), (VA_SOURCE_RANGE_FULL as u8, 0));
            assert_eq!(p.input_color_properties.colour_primaries, 1);
            assert_eq!(p.output_color_properties.color_range, VA_SOURCE_RANGE_REDUCED as u8);
            assert_eq!(p.output_color_properties.matrix_coefficients, if codec == Codec::Vp8 { 6 } else { 1 }, "{codec:?}");
            assert_eq!(p.output_color_properties.chroma_sample_location, (VA_CHROMA_SITING_VERTICAL_CENTER | VA_CHROMA_SITING_HORIZONTAL_CENTER) as u8);
            assert_eq!(p.filter_flags, VA_FRAME_PICTURE);
        });
    }
    let mut classic = Driver::generous();
    classic.color_standards = vec![VAProcColorStandardBT601, VAProcColorStandardBT709, VAProcColorStandardSMPTE170M];
    mock::reset(classic);
    let mut enc = session(Codec::Vp8, false);
    encode(&mut enc, 0, true);
    mock::with(|d| {
        let bytes = &d.buffers[d.pictures[0].2[0]];
        let p: VAProcPipelineParameterBuffer = unsafe { ptr::read_unaligned(bytes.2.as_ptr() as *const _) };
        assert_eq!((p.surface_color_standard, p.output_color_standard), (VAProcColorStandardBT709, VAProcColorStandardBT601));
    });
}

/// A rate change opens a new sequence: the next frame is a key frame carrying the new rate
/// control; a quality change in constant-quantizer mode reaches the next frame as it is.
#[test]
fn a_rate_change_restarts_the_sequence_and_a_quality_change_does_not() {
    mock::reset(Driver::generous());
    let mut enc = session(Codec::H264, true);
    encode(&mut enc, 0, true);
    encode(&mut enc, 1, false);
    let mut s = settings(Codec::H264, true);
    s.video_bitrate_kbps = 8000;
    enc.reconfigure_rate(&s).unwrap();
    let out = encode(&mut enc, 2, false);
    assert_eq!(parse_video_type(out[1]), Some((Codec::H264, FRAME_KEY)));
    mock::with(|d| {
        let rc = d.last_misc().into_iter().find(|m| m.0 == VAEncMiscParameterTypeRateControl).unwrap().1;
        let rc: VAEncMiscParameterRateControl = unsafe { ptr::read_unaligned(rc.as_ptr() as *const _) };
        assert_eq!(rc.bits_per_second, 8_000_000);
    });
    mock::reset(Driver::generous());
    let mut enc = session(Codec::H264, false);
    encode(&mut enc, 0, true);
    let out = enc.encode_host(&frame(), (W * 4) as usize, false, 1, 40, false).unwrap();
    assert_eq!(parse_video_type(out[1]), Some((Codec::H264, FRAME_DELTA)));
    mock::with(|d| {
        let header = d.last_packed().into_iter().find(|p| p.0 == VAEncPackedHeaderSlice).unwrap().1;
        let rbsp = nal(&header, 1, false);
        let mut r = Reader { bytes: &rbsp, pos: 0 };
        r.ue();
        r.ue();
        r.ue();
        r.u(16);
        r.u(1);
        r.u(1);
        r.u(1);
        r.ue();
        assert_eq!(r.se(), 40 - 26, "the new quantizer reaches the slice");
    });
}

/// A host frame lands on the surface through its derived image where the driver derives one,
/// else through an image the driver copies in.
#[test]
fn host_frames_upload_through_the_derived_image_or_a_put() {
    mock::reset(Driver::generous());
    let mut enc = session(Codec::H264, false);
    encode(&mut enc, 0, true);
    encode(&mut enc, 1, false);
    mock::with(|d| {
        assert_eq!(d.puts, 0);
        assert_eq!(d.images.len(), 3, "the derive probe and one derived image per frame");
    });
    let mut copying = Driver::generous();
    copying.derive = false;
    mock::reset(copying);
    let mut enc = session(Codec::H264, false);
    encode(&mut enc, 0, true);
    encode(&mut enc, 1, false);
    mock::with(|d| {
        assert_eq!(d.puts, 2);
        assert_eq!(d.images.len(), 1, "one image created and put twice");
    });
}

/// A driver that writes its own slice headers names the references it chooses, so the
/// session tracks none and refuses an invalidation; one without the codec refuses the
/// session.
#[test]
fn a_driver_writing_its_own_headers_tracks_no_reference() {
    let mut own_headers = Driver::generous();
    own_headers.attributes.retain(|a| a.0 != VAConfigAttribEncPackedHeaders);
    mock::reset(own_headers);
    for codec in [Codec::H264, Codec::H265] {
        let mut enc = session(codec, false);
        let first = encode(&mut enc, 0, true);
        assert_eq!(parse_video_type(first[1]), Some((codec, FRAME_KEY)));
        let second = encode(&mut enc, 1, false);
        assert_eq!(parse_video_type(second[1]), Some((codec, FRAME_DELTA)));
        assert_eq!(enc.last_reference(), Reference::Untracked);
        assert!(!enc.invalidate_reference(0));
        mock::with(|d| assert!(d.last_packed().is_empty(), "no packed header goes to a driver that takes none"));
    }
    let mut enc = session(Codec::Vp9, false);
    encode(&mut enc, 0, true);
    encode(&mut enc, 1, false);
    assert_eq!(enc.last_reference(), Reference::Frame(0), "VP9's references need no packed header");

    let mut no_av1 = Driver::generous();
    no_av1.profiles.retain(|&p| p != VAProfileAV1Profile0);
    mock::reset(no_av1);
    let refused = open(Codec::Av1, &settings(Codec::Av1, false));
    assert!(refused.err().unwrap().contains("encodes no AV1"));
    let served: Vec<Codec> = probe_codecs_on(&device()).unwrap().into_iter().map(|(codec, _)| codec).collect();
    assert_eq!(served, [Codec::H264, Codec::Vp8, Codec::Vp9, Codec::H265]);
}

/// A 4:4:4 session takes the planar surface where the driver renders it, the packed one
/// where that is all it renders, and refuses where it renders neither.
#[test]
fn fullcolor_takes_the_surface_the_driver_renders() {
    for (rendered, wanted) in [
        (vec![VA_FOURCC_NV12, VA_FOURCC_444P, VA_FOURCC_XYUV, VA_FOURCC_BGRA], Some("yuv444p")),
        (vec![VA_FOURCC_NV12, VA_FOURCC_XYUV, VA_FOURCC_BGRA], Some("vuyx")),
        (vec![VA_FOURCC_NV12, VA_FOURCC_BGRA], None),
    ] {
        let mut driver = Driver::generous();
        driver.surface_fourccs = rendered;
        mock::reset(driver);
        let mut s = settings(Codec::H265, false);
        s.video_fullcolor = true;
        match open(Codec::H265, &s) {
            Ok(enc) => {
                assert!(enc.is_fullcolor());
                assert_eq!(Some(enc.surface_format_name().as_str()), wanted);
                mock::with(|d| assert_eq!(d.configs[0].0, VAProfileHEVCMain444));
            }
            Err(e) => assert!(wanted.is_none(), "{e}"),
        }
    }
}
