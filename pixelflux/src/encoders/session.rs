/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! What the full-frame software sessions share: the planar picture a packed host frame is
//! converted into, the constant quantizer that follows the session quality index, the
//! rate-control settings a live change is compared against, and the frames an encoder has
//! been handed but not yet answered.

use std::collections::VecDeque;

use super::software::convert_to_yuv_mt;
use super::{vbv_bits, QP_HYSTERESIS_LIMIT};
use crate::RustCaptureSettings;

/// A planar 8-bit picture, 4:2:0 or 4:4:4, with tightly packed rows.
pub struct Planes {
    pub width: usize,
    pub height: usize,
    pub i444: bool,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl Planes {
    pub fn new(width: usize, height: usize, i444: bool) -> Self {
        let (cw, ch) = if i444 { (width, height) } else { (width.div_ceil(2), height.div_ceil(2)) };
        Self { width, height, i444, y: vec![0; width * height], u: vec![0; cw * ch], v: vec![0; cw * ch] }
    }

    pub fn chroma_width(&self) -> usize {
        if self.i444 { self.width } else { self.width.div_ceil(2) }
    }

    /// Convert a packed host frame (`stride` bytes per row, R,G,B,A when `rgba`, else B,G,R,A)
    /// into the planes across `threads` bands, with the BT.709 matrix at `full_range`, or BT.601
    /// where `bt601` names a codec whose bitstream can carry no other.
    pub fn convert(
        &mut self,
        pixels: &[u8],
        stride: usize,
        rgba: bool,
        full_range: bool,
        bt601: bool,
        threads: usize,
    ) -> Result<(), String> {
        let cw = self.chroma_width();
        convert_into(
            pixels, stride, self.width, self.height, rgba, self.i444, full_range, bt601, threads,
            &mut self.y, &mut self.u, &mut self.v, (self.width, cw),
        )
    }
}

/// Convert a packed host frame into planes a library owns, at `strides` (luma, chroma) bytes
/// per row, as `Planes::convert` does into its own.
#[allow(clippy::too_many_arguments)]
pub fn convert_into(
    pixels: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    rgba: bool,
    i444: bool,
    full_range: bool,
    bt601: bool,
    threads: usize,
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    strides: (usize, usize),
) -> Result<(), String> {
    check_host_frame(pixels, stride, width, height)?;
    convert_to_yuv_mt(pixels, stride as u32, width, height, rgba, i444, full_range, bt601, y, u, v, strides, threads)
        .map_err(|e| format!("rgb-to-yuv conversion failed: {e:?}"))
}

/// Whether `pixels` holds a `width` x `height` packed picture at `stride` bytes per row.
pub fn check_host_frame(pixels: &[u8], stride: usize, width: usize, height: usize) -> Result<(), String> {
    let row_bytes = width * 4;
    let needed = if height == 0 { 0 } else { stride.checked_mul(height - 1).ok_or("stride overflow")? + row_bytes };
    if stride < row_bytes || pixels.len() < needed {
        return Err("Input buffer too small".into());
    }
    Ok(())
}

/// The constant quantizer of a session, moved toward the one the quality index selects: a
/// decrease sharpens the picture and applies at once, an increase waits out
/// `QP_HYSTERESIS_LIMIT` consecutive requests so transient motion does not make quality blink.
/// The domain is the codec's own (`Codec::quantizer`).
pub struct Quality {
    pub current: u32,
    counter: u32,
}

impl Quality {
    pub fn new(quantizer: u32) -> Self {
        Self { current: quantizer, counter: 0 }
    }

    /// The quantizer to program now for `target`, once the hysteresis admits the change.
    pub fn update(&mut self, target: u32) -> Option<u32> {
        if target == self.current {
            self.counter = 0;
            return None;
        }
        if target > self.current {
            self.counter += 1;
            if self.counter <= QP_HYSTERESIS_LIMIT {
                return None;
            }
        }
        self.counter = 0;
        self.current = target;
        Some(target)
    }
}

/// The rate-control and frame-rate settings a session was configured with, so a live change
/// re-programs the encoder only when a value it reads actually moved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateSettings {
    pub cbr: bool,
    pub bitrate_kbps: i32,
    pub vbv_multiplier: f64,
    pub keyframe_interval_s: f64,
    pub fps: i32,
    pub min_qp: i32,
    pub max_qp: i32,
}

impl RateSettings {
    pub fn new(settings: &RustCaptureSettings) -> Self {
        Self {
            cbr: settings.video_cbr_mode,
            bitrate_kbps: settings.video_bitrate_kbps,
            vbv_multiplier: settings.video_vbv_multiplier,
            keyframe_interval_s: settings.keyframe_interval_s,
            fps: (settings.target_fps.max(1.0)) as i32,
            min_qp: settings.video_min_qp,
            max_qp: settings.video_max_qp,
        }
    }

    /// The settings as they stand now, if a rate or frame-rate value the encoder was programmed
    /// with changed: in CBR a bitrate or VBV multiplier, in any mode the frame rate.
    pub fn changed(&self, settings: &RustCaptureSettings) -> Option<Self> {
        let next = Self { cbr: self.cbr, min_qp: self.min_qp, max_qp: self.max_qp, ..Self::new(settings) };
        let rate_moved = self.cbr && (next.bitrate_kbps != self.bitrate_kbps || next.vbv_multiplier != self.vbv_multiplier);
        (rate_moved || next.fps != self.fps).then_some(next)
    }

    /// The constant-rate target in bits per second.
    pub fn bps(&self) -> u64 {
        (self.bitrate_kbps.max(0) as u64) * 1000
    }

    /// The VBV buffer, in bits, the constant-rate target is held to.
    pub fn vbv(&self) -> u32 {
        vbv_bits(self.bps().min(u32::MAX as u64) as u32, self.fps as f64, self.keyframe_interval_s, self.vbv_multiplier)
    }
}

/// The frames handed to an encoder and not yet answered, by the timestamp each was submitted
/// with, so a packet names the frame it encodes rather than the one submitted alongside it.
/// An encoder that pipelines answers a frame several later, and one that drops a frame never
/// answers it at all, so taking a timestamp discards every submission it passed over.
#[derive(Default)]
pub struct Pending(VecDeque<(u64, u16)>);

impl Pending {
    /// Record that the frame with wire id `frame_id` went in at `pts`.
    pub fn push(&mut self, pts: u64, frame_id: u16) {
        self.0.push_back((pts, frame_id));
    }

    /// Drop the last submission, for a frame the encoder refused outright.
    pub fn undo(&mut self) {
        self.0.pop_back();
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The wire id submitted at `pts`, dropping it and everything older; `None` where no
    /// submission still held carries that timestamp.
    pub fn take(&mut self, pts: u64) -> Option<u16> {
        let at = self.0.iter().position(|&(p, _)| p == pts)?;
        let id = self.0[at].1;
        self.0.drain(..=at);
        Some(id)
    }
}

/// The encode threads a software session spreads across: one less than the host's cores,
/// between one and eight.
pub fn encode_threads() -> i32 {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).saturating_sub(1).clamp(1, 8) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quality_increase_applies_at_once_and_a_drop_waits() {
        let mut q = Quality::new(30);
        assert_eq!(q.update(30), None);
        assert_eq!(q.update(20), Some(20), "a lower quantizer applies at once");
        for _ in 0..QP_HYSTERESIS_LIMIT {
            assert_eq!(q.update(40), None);
        }
        assert_eq!(q.update(40), Some(40), "a higher one waits out the hysteresis");
        assert_eq!(q.update(45), None);
        assert_eq!(q.update(40), None, "a request back at the current value resets the count");
        assert_eq!(q.current, 40);
    }

    #[test]
    fn a_rate_change_is_noticed_only_where_the_encoder_reads_it() {
        let mut settings = RustCaptureSettings { target_fps: 30.0, video_bitrate_kbps: 4000, ..Default::default() };
        let rate = RateSettings::new(&settings);
        assert_eq!(rate.changed(&settings), None);
        settings.video_bitrate_kbps = 8000;
        assert_eq!(rate.changed(&settings), None, "a bitrate moves nothing in constant-quality mode");
        settings.target_fps = 60.0;
        assert_eq!(rate.changed(&settings).map(|r| r.fps), Some(60));
        settings.video_cbr_mode = true;
        let cbr = RateSettings::new(&settings);
        settings.video_bitrate_kbps = 9000;
        assert_eq!(cbr.changed(&settings).map(|r| r.bitrate_kbps), Some(9000));
        assert_eq!(cbr.bps(), 8_000_000);
    }

    /// A packet names the frame it encodes however deep the encoder pipelines, the frames it
    /// passed over go with it, and a timestamp never submitted names nothing.
    #[test]
    fn a_packet_names_the_frame_it_encodes() {
        let mut pending = Pending::default();
        assert!(pending.is_empty());
        for (pts, id) in [(0u64, 7u16), (1, 8), (2, 9)] {
            pending.push(pts, id);
        }
        assert_eq!(pending.take(99), None, "a timestamp never submitted");
        assert_eq!(pending.take(1), Some(8), "the encoder answered two frames in");
        assert_eq!(pending.take(0), None, "the frame it passed over went with it");
        assert_eq!(pending.take(2), Some(9));
        assert!(pending.is_empty());
        pending.push(3, 10);
        pending.undo();
        assert_eq!(pending.take(3), None, "a refused frame was never submitted");
    }

    #[test]
    fn a_frame_shorter_than_its_stride_says_so() {
        assert!(check_host_frame(&[0; 16 * 4 * 4], 64, 16, 4).is_ok());
        assert!(check_host_frame(&[0; 16 * 4 * 4 - 1], 64, 16, 4).is_err());
        assert!(check_host_frame(&[0; 16 * 4 * 4], 60, 16, 4).is_err(), "a stride shorter than a row");
        let mut planes = Planes::new(4, 2, false);
        assert_eq!((planes.u.len(), planes.chroma_width()), (2, 2));
        assert!(planes.convert(&[0; 4 * 4 * 2], 16, false, false, false, 1).is_ok());
        assert!(Planes::new(4, 2, true).convert(&[0; 4 * 4], 16, false, true, false, 1).is_err());
    }
}
