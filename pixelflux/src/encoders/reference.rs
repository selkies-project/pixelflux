//! Which frame a delivered frame predicts from, kept true across the frames a client reports
//! lost.

use std::collections::VecDeque;

/// How many frames back a prediction may reach: the decoded picture buffer every session asks
/// for, where its level admits it.
pub const REFERENCE_FRAMES: u32 = 8;

/// The frame a delivered frame predicts from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reference {
    /// A session that cannot say which frame it predicted from, or leave a lost one out.
    #[default]
    Untracked,
    /// A frame that decodes on its own.
    None,
    /// The id of the frame predicted from.
    Frame(u16),
}

impl Reference {
    /// The wire form: the frame id, -1 for a frame that decodes on its own, -2 where the
    /// session does not track its references.
    pub fn frame_id(self) -> i32 {
        match self {
            Reference::Untracked => -2,
            Reference::None => -1,
            Reference::Frame(id) => id as i32,
        }
    }
}

/// What an invalidation asks of the encoder.
#[derive(Debug, PartialEq, Eq)]
pub enum Invalidation {
    /// The frame predates the last key frame or was never a reference: nothing to do.
    Ignored,
    /// Forget the frame carrying this timestamp and every frame after it.
    Forget(u64),
    /// The frame has left the window, so every reference still held predicts through it: the
    /// next frame has to be a key frame.
    KeyFrame,
}

/// The reference frames a session's decoder holds, in encode order, with the ones a client
/// reported lost marked.
///
/// An encoder told to forget a frame leaves it and every frame after it out of its predictions,
/// predicts the next frame from the newest frame before them, and codes a key frame when none is
/// left. The window mirrors that, so the reference a frame carries on the wire is the one its
/// bitstream uses, and it asks for the key frame where the encoder would. Frames are addressed
/// by the capture's frame id, which wraps; the timestamp the encoder is told to forget is the
/// session's own count of encoded frames, which does not.
///
/// An H.264 session also names how many values its `frame_num` takes before wrapping
/// (`set_frame_num_range`). A decoder that never receives the frame carrying `frame_num` 0 sees
/// a gap across that wrap, and FFmpeg's H.264 decoder derives the picture order past such a gap
/// wrongly and withholds every picture after it until a key frame, so a loss covering that frame
/// is answered with a key frame rather than a prediction past it.
pub struct ReferenceWindow {
    frames: VecDeque<(u16, u64, bool)>,
    capacity: usize,
    next_pts: u64,
    key: Option<u16>,
    key_pts: u64,
    frame_num_range: u64,
}

/// Whether frame id `a` came before `b`, across the wrap.
fn before(a: u16, b: u16) -> bool {
    a != b && b.wrapping_sub(a) < 0x8000
}

impl ReferenceWindow {
    pub fn new(capacity: u32) -> Self {
        Self {
            frames: VecDeque::new(),
            capacity: capacity.max(1) as usize,
            next_pts: 0,
            key: None,
            key_pts: 0,
            frame_num_range: 0,
        }
    }

    /// How many values the stream's `frame_num` takes before it wraps; 0 for a codec without
    /// that counter.
    pub fn set_frame_num_range(&mut self, range: u32) {
        self.frame_num_range = range as u64;
    }

    /// The decoded picture buffer the session has now; frames past it are let go.
    pub fn set_capacity(&mut self, capacity: u32) {
        self.capacity = capacity.max(1) as usize;
        while self.frames.len() > self.capacity {
            self.frames.pop_front();
        }
    }

    /// The timestamp the next frame is encoded with.
    pub fn next_pts(&self) -> u64 {
        self.next_pts
    }

    /// Whether a frame not coded as a key frame has a reference left to predict from.
    pub fn has_reference(&self) -> bool {
        self.frames.iter().any(|f| !f.2)
    }

    /// Record the frame just encoded with `next_pts` and answer what it predicted from.
    pub fn record(&mut self, frame_id: u16, key: bool) -> Reference {
        let pts = self.next_pts;
        self.next_pts += 1;
        let reference = if key {
            self.frames.clear();
            self.key = Some(frame_id);
            self.key_pts = pts;
            Reference::None
        } else {
            self.frames.iter().rev().find(|f| !f.2).map_or(Reference::None, |f| Reference::Frame(f.0))
        };
        self.frames.push_back((frame_id, pts, false));
        if self.frames.len() > self.capacity {
            self.frames.pop_front();
        }
        reference
    }

    /// Leave `frame_id` and every frame encoded after it out of the references.
    pub fn invalidate(&mut self, frame_id: u16) -> Invalidation {
        let Some(&(oldest, _, _)) = self.frames.front() else {
            return Invalidation::Ignored;
        };
        if self.key.is_some_and(|key| before(frame_id, key)) {
            return Invalidation::Ignored;
        }
        match self.frames.iter().position(|f| f.0 == frame_id) {
            Some(at) => {
                let pts = self.frames[at].1;
                let wraps = self.frame_num_range > 0
                    && self.frames.iter().skip(at).any(|f| (f.1 - self.key_pts).is_multiple_of(self.frame_num_range));
                for f in self.frames.iter_mut().skip(if wraps { 0 } else { at }) {
                    f.2 = true;
                }
                if wraps { Invalidation::KeyFrame } else { Invalidation::Forget(pts) }
            }
            None if before(frame_id, oldest) => {
                for f in self.frames.iter_mut() {
                    f.2 = true;
                }
                Invalidation::KeyFrame
            }
            None => Invalidation::Ignored,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_predicts_from_the_newest_one_the_client_still_has() {
        let mut w = ReferenceWindow::new(4);
        assert!(!w.has_reference(), "an empty window has nothing to predict from");
        assert_eq!(w.record(10, true), Reference::None);
        assert_eq!(w.record(11, false), Reference::Frame(10));
        assert_eq!(w.record(12, false), Reference::Frame(11));
        assert_eq!(w.invalidate(11), Invalidation::Forget(1));
        assert!(w.has_reference(), "frame 10 is still there to predict from");
        assert_eq!(w.record(13, false), Reference::Frame(10));
        assert_eq!(w.record(14, false), Reference::Frame(13));
        assert_eq!(w.next_pts(), 5);
    }

    #[test]
    fn losing_every_reference_asks_for_a_key_frame() {
        let mut w = ReferenceWindow::new(3);
        w.record(0, true);
        for id in 1..6u16 {
            w.record(id, false);
        }
        // The window holds 3, 4, and 5; frame 2 has left it, and everything held predicts
        // through it.
        assert_eq!(w.invalidate(2), Invalidation::KeyFrame);
        assert!(!w.has_reference());
        assert_eq!(w.record(6, true), Reference::None);
        assert!(w.has_reference());
        // Forgetting the only frame held leaves nothing either.
        assert_eq!(w.invalidate(6), Invalidation::Forget(6));
        assert!(!w.has_reference());
    }

    #[test]
    fn a_frame_older_than_the_key_frame_is_ignored() {
        let mut w = ReferenceWindow::new(4);
        w.record(100, true);
        w.record(101, false);
        assert_eq!(w.invalidate(99), Invalidation::Ignored);
        assert_eq!(w.invalidate(105), Invalidation::Ignored, "a frame never sent is nothing to forget");
        assert!(w.has_reference());
    }

    #[test]
    fn frame_ids_wrap_and_timestamps_do_not() {
        let mut w = ReferenceWindow::new(4);
        w.record(65534, true);
        w.record(65535, false);
        w.record(0, false);
        assert_eq!(w.record(1, false), Reference::Frame(0));
        assert_eq!(w.invalidate(65535), Invalidation::Forget(1));
        assert_eq!(w.record(2, false), Reference::Frame(65534));
        assert_eq!(w.invalidate(65533), Invalidation::Ignored, "before the key frame");
        w.set_capacity(2);
        assert_eq!(w.invalidate(65534), Invalidation::KeyFrame, "left the window, and everything held predicts through it");
    }

    #[test]
    fn a_loss_covering_the_frame_num_wrap_costs_a_key_frame() {
        let mut w = ReferenceWindow::new(8);
        w.set_frame_num_range(16);
        w.record(0, true);
        for id in 1..=17u16 {
            w.record(id, false);
        }
        // Frame 16 carries frame_num 0 again; a loss that leaves it out cannot be predicted past.
        assert_eq!(w.invalidate(17), Invalidation::Forget(17));
        assert_eq!(w.record(18, false), Reference::Frame(16));
        assert_eq!(w.invalidate(15), Invalidation::KeyFrame, "15, 16, and 18 go, and 16 is the wrap");
        assert!(!w.has_reference());
        assert_eq!(w.record(19, true), Reference::None);
        assert_eq!(w.record(20, false), Reference::Frame(19));
        assert_eq!(w.invalidate(20), Invalidation::Forget(20), "the count restarts at the key frame");
        let mut w = ReferenceWindow::new(8);
        w.record(0, true);
        for id in 1..=17u16 {
            w.record(id, false);
        }
        assert_eq!(w.invalidate(16), Invalidation::Forget(16), "a codec without the counter predicts past it");
    }

    #[test]
    fn the_wire_form_names_the_frame() {
        assert_eq!(Reference::Untracked.frame_id(), -2);
        assert_eq!(Reference::None.frame_id(), -1);
        assert_eq!(Reference::Frame(7).frame_id(), 7);
    }
}
