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
    age: KeyAge,
    key_pts: u64,
    frame_num_range: u64,
}

/// How far a frame id lies behind the newest frame recorded, in capture frames counted across
/// the id's wrap, and whether it went out at or after the last key frame. A session can run far
/// longer than the id takes to wrap between key frames, so the ids alone cannot order a frame
/// against the key.
#[derive(Default)]
struct KeyAge {
    newest: u16,
    since_key: u64,
}

impl KeyAge {
    fn record(&mut self, frame_id: u16, key: bool) {
        self.since_key = if key { 0 } else { self.since_key + frame_id.wrapping_sub(self.newest) as u64 };
        self.newest = frame_id;
    }

    /// How many frames `frame_id` went out before the newest one; `None` for a frame before the
    /// last key frame or one never sent.
    fn behind(&self, frame_id: u16) -> Option<u64> {
        let behind = self.newest.wrapping_sub(frame_id) as u64;
        (behind < 0x8000 && behind <= self.since_key).then_some(behind)
    }
}

impl ReferenceWindow {
    pub fn new(capacity: u32) -> Self {
        Self {
            frames: VecDeque::new(),
            capacity: capacity.max(1) as usize,
            next_pts: 0,
            age: KeyAge::default(),
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

    /// The newest frame the client still has, as its id and timestamp: the one the next frame
    /// predicts from.
    pub fn newest_valid(&self) -> Option<(u16, u64)> {
        self.frames.iter().rev().find(|f| !f.2).map(|f| (f.0, f.1))
    }

    /// The frames the decoder holds, oldest first: each frame's id, timestamp, and whether
    /// a client reported it lost.
    pub fn held(&self) -> impl Iterator<Item = (u16, u64, bool)> + '_ {
        self.frames.iter().copied()
    }

    /// The timestamp of the last key frame, where the counts a codec keeps restart.
    pub fn key_pts(&self) -> u64 {
        self.key_pts
    }

    /// Record the frame just encoded with `next_pts` and answer what it predicted from.
    pub fn record(&mut self, frame_id: u16, key: bool) -> Reference {
        let pts = self.next_pts;
        self.next_pts += 1;
        self.age.record(frame_id, key);
        let reference = if key {
            self.frames.clear();
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
            None if self.age.behind(frame_id).is_some_and(|b| b > self.age.newest.wrapping_sub(oldest) as u64) => {
                for f in self.frames.iter_mut() {
                    f.2 = true;
                }
                Invalidation::KeyFrame
            }
            None => Invalidation::Ignored,
        }
    }
}

/// The buffers a VP8 or VP9 encoder predicts from, with the frame each holds and whether a
/// client reported it lost.
///
/// Some buffers are spent on anchors older than the last few frames, so a loss up to twelve
/// frames deep still finds a frame older than it. VP8 has three, LAST, GOLDEN, and ALTREF: every
/// frame lands in LAST, and every twelfth in GOLDEN and ALTREF by turns, so one anchor is always
/// twelve to twenty-three frames old. VP9 has eight: the last four frames by turns in the first
/// four, and every fourth frame by turns in the other four, the oldest twelve to fifteen frames
/// back. A frame predicts from the newest buffer whose frame the client still has, the first
/// such buffer when others are as new, and a VP8 frame coded from an anchor rather than LAST
/// refreshes all three, since it is the newest picture both sides hold. The frame ids wrap; the
/// timestamps are the session's own count of encoded frames and do not, and the recent ones are
/// kept so a lost frame no buffer holds still dates the buffers coded after it.
pub struct ReferenceSlots {
    slots: [Option<(u16, u64, bool)>; 8],
    count: usize,
    recent: VecDeque<(u16, u64)>,
    next_pts: u64,
    age: KeyAge,
    key_pts: u64,
}

/// How many recent frames are remembered by id; the anchors reach twenty-three back.
const RECENT_FRAMES: usize = 64;

/// VP8's GOLDEN and ALTREF each take every `ANCHOR_PERIOD`th frame, half a period apart.
const ANCHOR_PERIOD: u64 = 24;

/// VP9 keeps the last `RING` frames, and every `RING`th frame in as many buffers again.
const RING: u64 = 4;

/// The buffers a frame refreshes, as a bit per slot: LAST, GOLDEN, ALTREF for VP8, the eight in
/// order for VP9.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotRefresh(pub u8);

impl SlotRefresh {
    pub const LAST: u8 = 1;
    pub const GOLDEN: u8 = 2;
    pub const ALTREF: u8 = 4;
    pub const ALL: u8 = 7;

    pub fn refreshes(self, slot: u8) -> bool {
        self.0 & slot != 0
    }
}

/// How the next frame predicts and which buffers it refreshes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotPlan {
    /// The buffer predicted from, as a `SlotRefresh` bit; zero for a key frame.
    pub predict_from: u8,
    pub refresh: SlotRefresh,
}

impl SlotPlan {
    /// A VP8 key frame: it predicts from nothing and re-anchors every buffer.
    pub const KEY: Self = Self { predict_from: 0, refresh: SlotRefresh(SlotRefresh::ALL) };
}

impl Default for ReferenceSlots {
    fn default() -> Self {
        Self::new()
    }
}

impl ReferenceSlots {
    /// VP8's three buffers.
    pub fn new() -> Self {
        Self::with(3)
    }

    /// VP9's eight buffers.
    pub fn vp9() -> Self {
        Self::with(8)
    }

    fn with(count: usize) -> Self {
        Self { slots: [None; 8], count, recent: VecDeque::new(), next_pts: 0, age: KeyAge::default(), key_pts: 0 }
    }

    /// The timestamp the next frame is encoded with.
    pub fn next_pts(&self) -> u64 {
        self.next_pts
    }

    /// Whether a frame not coded as a key frame has a buffer left to predict from.
    pub fn has_reference(&self) -> bool {
        self.slots[..self.count].iter().any(|s| s.is_some_and(|(_, _, lost)| !lost))
    }

    /// What each buffer holds: the frame's id and timestamp and whether it was reported lost.
    pub fn slot(&self, slot: u8) -> Option<(u16, u64, bool)> {
        self.slots[slot.trailing_zeros() as usize]
    }

    /// The buffer the next frame predicts from (the newest one the client still has) and the
    /// buffers it refreshes; a key frame refreshes all of them and predicts from none.
    pub fn plan(&self, key: bool) -> SlotPlan {
        if key || !self.has_reference() {
            return SlotPlan { predict_from: 0, refresh: SlotRefresh(((1u16 << self.count) - 1) as u8) };
        }
        let mut newest = 0;
        for i in 0..self.count {
            if let Some((_, pts, false)) = self.slots[i]
                && self.slots[newest].is_none_or(|(_, held, lost)| lost || pts > held)
            {
                newest = i;
            }
        }
        let predict_from = 1 << newest;
        let since_key = self.next_pts - self.key_pts;
        let refresh = if self.count > 3 {
            1 << (since_key % RING) | if since_key % RING == 0 { 1 << (RING + since_key / RING % RING) } else { 0 }
        } else if predict_from != SlotRefresh::LAST {
            SlotRefresh::ALL
        } else {
            SlotRefresh::LAST
                | if since_key % ANCHOR_PERIOD == 0 { SlotRefresh::GOLDEN } else { 0 }
                | if since_key % ANCHOR_PERIOD == ANCHOR_PERIOD / 2 { SlotRefresh::ALTREF } else { 0 }
        };
        SlotPlan { predict_from, refresh: SlotRefresh(refresh) }
    }

    /// Record the frame just encoded under `plan` and answer what it predicted from.
    pub fn record(&mut self, frame_id: u16, plan: SlotPlan) -> Reference {
        let pts = self.next_pts;
        self.next_pts += 1;
        self.age.record(frame_id, plan.predict_from == 0);
        let reference = match plan.predict_from {
            0 => {
                self.key_pts = pts;
                Reference::None
            }
            from => self.slots[from.trailing_zeros() as usize].map_or(Reference::None, |(id, _, _)| Reference::Frame(id)),
        };
        for (i, slot) in self.slots[..self.count].iter_mut().enumerate() {
            if plan.predict_from == 0 || plan.refresh.refreshes(1 << i) {
                *slot = Some((frame_id, pts, false));
            }
        }
        self.recent.push_back((frame_id, pts));
        if self.recent.len() > RECENT_FRAMES {
            self.recent.pop_front();
        }
        reference
    }

    /// Leave `frame_id` and every frame encoded after it out of the references.
    pub fn invalidate(&mut self, frame_id: u16) -> Invalidation {
        let lost_pts = match self.recent.iter().find(|r| r.0 == frame_id) {
            Some(&(_, pts)) if pts >= self.key_pts => pts,
            Some(_) => return Invalidation::Ignored,
            None => {
                // A frame older than what is remembered was sent before every buffered one,
                // so everything held predicts through it; one before the key frame, or a
                // newer one, was never a reference.
                let Some(&(oldest, _)) = self.recent.front() else { return Invalidation::Ignored };
                if !self.age.behind(frame_id).is_some_and(|b| b > self.age.newest.wrapping_sub(oldest) as u64) {
                    return Invalidation::Ignored;
                }
                0
            }
        };
        for slot in self.slots[..self.count].iter_mut().flatten() {
            if slot.1 >= lost_pts {
                slot.2 = true;
            }
        }
        if self.has_reference() { Invalidation::Forget(lost_pts) } else { Invalidation::KeyFrame }
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
    fn three_buffers_anchor_older_frames() {
        let mut s = ReferenceSlots::new();
        assert!(!s.has_reference());
        let key = s.plan(true);
        assert_eq!(key.predict_from, 0);
        assert_eq!(s.record(0, key), Reference::None);
        for id in 1..=30u16 {
            let plan = s.plan(false);
            assert_eq!(plan.predict_from, SlotRefresh::LAST, "frame {id}");
            assert_eq!(s.record(id, plan), Reference::Frame(id - 1));
        }
        // Frames 25 to 30 are reported lost: LAST holds 30, GOLDEN 24, ALTREF 12.
        assert_eq!(s.invalidate(25), Invalidation::Forget(25));
        let plan = s.plan(false);
        assert_eq!(plan.predict_from, SlotRefresh::GOLDEN);
        assert_eq!(plan.refresh, SlotRefresh(SlotRefresh::ALL), "a recovery frame re-anchors every buffer");
        assert_eq!(s.record(31, plan), Reference::Frame(24));
        assert_eq!(s.record(32, s.plan(false)), Reference::Frame(31));
        // Losing the re-anchoring frame leaves nothing older than it.
        assert_eq!(s.invalidate(31), Invalidation::KeyFrame);
        assert!(!s.has_reference());
        assert_eq!(s.record(33, s.plan(false)), Reference::None);
        assert_eq!(s.invalidate(3), Invalidation::Ignored, "before the key frame");
        assert_eq!(s.next_pts(), 34);
    }

    #[test]
    fn golden_and_altref_follow_their_periods() {
        let mut s = ReferenceSlots::new();
        s.record(0, s.plan(true));
        let mut refreshes = Vec::new();
        for id in 1..=48u16 {
            let plan = s.plan(false);
            refreshes.push(plan.refresh.0);
            s.record(id, plan);
        }
        for (i, &r) in refreshes.iter().enumerate() {
            let frame = i + 1;
            let want = SlotRefresh::LAST
                | if frame % 24 == 0 { SlotRefresh::GOLDEN } else { 0 }
                | if frame % 24 == 12 { SlotRefresh::ALTREF } else { 0 };
            assert_eq!(r, want, "frame {frame}");
        }
        // A loss of frames 37 to 48 takes the golden of frame 48; the altref of frame 36 is
        // older than it, though no buffer holds frame 37 itself.
        assert_eq!(s.invalidate(37), Invalidation::Forget(37));
        assert_eq!(s.plan(false).predict_from, SlotRefresh::ALTREF);
        assert_eq!(s.record(49, s.plan(false)), Reference::Frame(36));
        assert_eq!(s.invalidate(100), Invalidation::Ignored, "a frame never sent is nothing to forget");
    }

    #[test]
    fn eight_buffers_keep_four_recent_frames_and_every_fourth() {
        let mut s = ReferenceSlots::vp9();
        let key = s.plan(true);
        assert_eq!(key.refresh, SlotRefresh(0xff), "a key frame refreshes all eight");
        s.record(0, key);
        for id in 1..=19u16 {
            let plan = s.plan(false);
            let n = id as u8;
            let want = 1 << (n % 4) | if n % 4 == 0 { 1 << (4 + n / 4 % 4) } else { 0 };
            assert_eq!(plan.refresh.0, want, "frame {id}");
            assert_eq!(s.record(id, plan), Reference::Frame(id - 1));
        }
        let held: Vec<u16> = (0..8).map(|i| s.slot(1 << i).unwrap().0).collect();
        assert_eq!(held, [16, 17, 18, 19, 16, 4, 8, 12]);
        // Frames 9 to 19 are reported lost: the anchor of frame 8 is the newest older than it.
        assert_eq!(s.invalidate(9), Invalidation::Forget(9));
        let plan = s.plan(false);
        assert_eq!(plan.predict_from, 1 << 6);
        assert_eq!(s.record(20, plan), Reference::Frame(8));
        assert_eq!(s.record(21, s.plan(false)), Reference::Frame(20));
    }

    /// A loss up to twelve frames deep, at any point of the schedule, still finds a buffer
    /// holding a frame older than it.
    #[test]
    fn a_loss_up_to_twelve_frames_deep_finds_an_anchor() {
        for make in [ReferenceSlots::new as fn() -> ReferenceSlots, ReferenceSlots::vp9] {
            for sent in 1..=60u16 {
                for depth in 1..=sent.min(12) {
                    let mut s = make();
                    s.record(0, s.plan(true));
                    for id in 1..=sent {
                        let plan = s.plan(false);
                        s.record(id, plan);
                    }
                    let lost = sent + 1 - depth;
                    assert!(matches!(s.invalidate(lost), Invalidation::Forget(_)), "{} buffers, frame {lost} of {sent} lost", s.count);
                    let plan = s.plan(false);
                    assert!(s.slot(plan.predict_from).is_some_and(|(id, _, lost_too)| id < lost && !lost_too));
                }
            }
        }
    }

    #[test]
    fn the_set_held_leaves_out_what_a_client_lost() {
        let held = |w: &ReferenceWindow| w.held().filter(|f| !f.2).map(|f| f.1).collect::<Vec<_>>();
        let mut w = ReferenceWindow::new(4);
        assert!(held(&w).is_empty(), "nothing is held before the first frame");
        w.record(10, true);
        w.record(11, false);
        w.record(12, false);
        assert_eq!(held(&w), [0, 1, 2]);
        assert_eq!(w.invalidate(11), Invalidation::Forget(1));
        assert_eq!(held(&w), [0], "11 and every frame after it are out");
        // The frame recorded next predicts from the newest one held, which is the set's last.
        assert_eq!(w.record(13, false), Reference::Frame(10));
        assert_eq!(held(&w), [0, 3]);
        // A lost frame still takes its place in the window until it is let go.
        w.record(14, false);
        assert_eq!(held(&w), [3, 4], "10 left the window of four, behind the two lost ones");
        assert_eq!(w.invalidate(10), Invalidation::KeyFrame);
        assert!(held(&w).is_empty(), "the next frame has to be a key frame");
        w.record(15, true);
        assert_eq!(held(&w), [5]);
    }

    /// A loss is answered however long the stream has run since its key frame: the ids wrap
    /// every 65536 frames, so past half of that an id no longer orders a frame against the key,
    /// and a capture that skips unchanged frames leaves gaps between the ids it encodes.
    #[test]
    fn a_loss_long_after_the_key_frame_is_still_answered() {
        let mut w = ReferenceWindow::new(8);
        let mut s = ReferenceSlots::new();
        let id = |n: u32| (n * 3) as u16;
        w.record(id(0), true);
        s.record(id(0), SlotPlan::KEY);
        for n in 1..=40_000u32 {
            w.record(id(n), false);
            let plan = s.plan(false);
            s.record(id(n), plan);
        }
        assert_eq!(w.invalidate(id(39_998)), Invalidation::Forget(39_998));
        assert_eq!(s.invalidate(id(39_998)), Invalidation::Forget(39_998));
        assert_eq!(w.invalidate(id(39_900)), Invalidation::KeyFrame, "sent since the key, and older than the window");
        assert_eq!(w.invalidate(id(39_999) + 1), Invalidation::Ignored, "a frame the capture skipped was never sent");
        let mut w = ReferenceWindow::new(8);
        w.record(id(0), false);
        for n in 1..=40_000u32 {
            w.record(id(n), n == 39_990);
        }
        assert_eq!(w.invalidate(id(39_989)), Invalidation::Ignored, "before the key frame");
        assert_eq!(w.invalidate(id(39_995)), Invalidation::Forget(39_995));
    }

    #[test]
    fn the_wire_form_names_the_frame() {
        assert_eq!(Reference::Untracked.frame_id(), -2);
        assert_eq!(Reference::None.frame_id(), -1);
        assert_eq!(Reference::Frame(7).frame_id(), 7);
    }
}
