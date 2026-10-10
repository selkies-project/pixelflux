/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Captured-sample provenance, independent of scene continuity and wire frame IDs.

/// Process-local identity of captured pixels, not a scene or presentation serial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleStamp {
    pub run_id: u64,
    pub sample_seq: u64,
    pub captured_ns: i64,
}

/// Keeps delayed submissions associated with the pixels in the encoder's staging buffer.
#[cfg(any(target_arch = "aarch64", test))]
pub(crate) struct SubmissionSamples {
    staged: Option<SampleStamp>,
    next: u64,
    capacity: usize,
    pending: std::collections::VecDeque<(u64, Option<SampleStamp>)>,
}

#[cfg(any(target_arch = "aarch64", test))]
impl SubmissionSamples {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            staged: None,
            next: 0,
            capacity,
            pending: std::collections::VecDeque::new(),
        }
    }

    pub(crate) fn stage(&mut self, sample: Option<SampleStamp>) {
        self.staged = sample;
    }

    /// Unique microseconds within a signed nanosecond clock, retaining the wire ID's low bits.
    pub(crate) fn reserve(&mut self, frame_id: u16) -> Result<u64, String> {
        let limit = i64::MAX as u64 / 1000;
        if self.next > limit >> 16 {
            return Err("encoder submission identity exhausted".into());
        }
        let id = (self.next << 16) | u64::from(frame_id);
        if id > limit {
            return Err("encoder submission identity exhausted".into());
        }
        self.next += 1;
        Ok(id)
    }

    /// Record only after the device accepted the submission, including staging repeats.
    pub(crate) fn submitted(&mut self, id: u64) {
        if self.capacity == 0 {
            return;
        }
        if self.pending.len() == self.capacity {
            self.pending.pop_front();
        }
        self.pending.push_back((id, self.staged));
    }

    pub(crate) fn take(&mut self, id: u64) -> Option<SampleStamp> {
        let index = self.pending.iter().position(|(queued, _)| *queued == id)?;
        self.pending.remove(index).and_then(|(_, sample)| sample)
    }

    pub(crate) fn timestamp(id: u64) -> [i64; 2] {
        [(id / 1_000_000) as i64, (id % 1_000_000) as i64]
    }

    pub(crate) fn from_timestamp(timestamp: [i64; 2]) -> Option<u64> {
        let [seconds, micros] = timestamp;
        if seconds < 0 || !(0..1_000_000).contains(&micros) {
            return None;
        }
        let id = (seconds as u64)
            .checked_mul(1_000_000)?
            .checked_add(micros as u64)?;
        (id <= i64::MAX as u64 / 1000).then_some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(run_id: u64, sample_seq: u64) -> SampleStamp {
        SampleStamp {
            run_id,
            sample_seq,
            captured_ns: (run_id * 100 + sample_seq) as i64,
        }
    }

    fn submit(queue: &mut SubmissionSamples, frame_id: u16) -> u64 {
        let id = queue.reserve(frame_id).unwrap();
        queue.submitted(id);
        id
    }

    #[test]
    fn delayed_and_out_of_order_units_keep_their_samples() {
        let mut queue = SubmissionSamples::new(4);
        queue.stage(Some(sample(1, 1)));
        let first = submit(&mut queue, 7);
        queue.stage(Some(sample(1, 2)));
        let second = submit(&mut queue, 8);
        queue.stage(Some(sample(1, 3)));
        let third = submit(&mut queue, 9);
        assert_eq!(queue.take(second), Some(sample(1, 2)));
        assert_eq!(queue.take(first), Some(sample(1, 1)));
        assert_eq!(queue.take(third), Some(sample(1, 3)));
    }

    #[test]
    fn held_repeats_name_staging_not_a_new_capture() {
        let mut queue = SubmissionSamples::new(4);
        queue.stage(Some(sample(1, 1)));
        let first = submit(&mut queue, 7);
        let repeat = submit(&mut queue, 8);
        assert_ne!(first, repeat);
        assert_eq!(queue.take(first), Some(sample(1, 1)));
        assert_eq!(queue.take(repeat), Some(sample(1, 1)));
    }

    #[test]
    fn reused_encoder_keeps_pending_samples_from_the_previous_run() {
        let mut queue = SubmissionSamples::new(4);
        queue.stage(Some(sample(1, 1)));
        let first = submit(&mut queue, 0);
        queue.stage(Some(sample(2, 1)));
        let second = submit(&mut queue, 0);
        assert_eq!(queue.take(first), Some(sample(1, 1)));
        assert_eq!(queue.take(second), Some(sample(2, 1)));
    }

    #[test]
    fn wire_id_wrap_cannot_alias_a_submission() {
        let mut queue = SubmissionSamples::new(4);
        queue.stage(Some(sample(1, 1)));
        let first = submit(&mut queue, 0);
        let end = submit(&mut queue, u16::MAX);
        queue.stage(Some(sample(1, 65537)));
        let wrapped = submit(&mut queue, 0);
        assert_ne!(first, wrapped);
        assert_eq!(first as u16, wrapped as u16);
        assert_eq!(end as u16, u16::MAX);
        assert_eq!(queue.take(first), Some(sample(1, 1)));
        assert_eq!(queue.take(wrapped), Some(sample(1, 65537)));
    }

    #[test]
    fn missing_and_evicted_units_never_borrow_staging_identity() {
        let mut queue = SubmissionSamples::new(2);
        queue.stage(Some(sample(1, 1)));
        let first = submit(&mut queue, 0);
        let second = submit(&mut queue, 1);
        queue.stage(Some(sample(2, 1)));
        let third = submit(&mut queue, 0);
        assert_eq!(queue.pending.len(), 2);
        assert_eq!(queue.take(first), None);
        assert_eq!(queue.take(u64::MAX), None);
        assert_eq!(queue.take(second), Some(sample(1, 1)));
        assert_eq!(queue.take(third), Some(sample(2, 1)));
        assert_eq!(queue.take(third), None);
    }

    #[test]
    fn untagged_or_failed_staging_clears_the_previous_identity() {
        let mut queue = SubmissionSamples::new(4);
        queue.stage(Some(sample(1, 1)));
        let first = submit(&mut queue, 0);
        queue.stage(None);
        let untagged = submit(&mut queue, 1);
        let repeat = submit(&mut queue, 2);
        assert_eq!(queue.take(first), Some(sample(1, 1)));
        assert_eq!(queue.take(untagged), None);
        assert_eq!(queue.take(repeat), None);
    }

    #[test]
    fn failed_submission_does_not_enter_the_pending_queue() {
        let mut queue = SubmissionSamples::new(4);
        queue.stage(Some(sample(1, 1)));
        let failed = queue.reserve(0).unwrap();
        let accepted = submit(&mut queue, 0);
        assert_eq!(queue.take(failed), None);
        assert_eq!(queue.take(accepted), Some(sample(1, 1)));
    }

    #[test]
    fn timestamp_exhaustion_is_explicit() {
        let mut queue = SubmissionSamples::new(1);
        queue.next = (i64::MAX as u64 / 1000 >> 16) - 1;
        assert!(queue.reserve(u16::MAX).unwrap() <= i64::MAX as u64 / 1000);
        queue.next += 1;
        assert!(queue.reserve(0).is_err());
    }

    #[test]
    fn timestamps_round_trip_without_overflowing_the_driver_nanosecond_clock() {
        for id in [0, 65535, 1_000_001, i64::MAX as u64 / 1000] {
            assert_eq!(
                SubmissionSamples::from_timestamp(SubmissionSamples::timestamp(id)),
                Some(id)
            );
        }
        for value in [[-1, 0], [0, -1], [0, 1_000_000], [i64::MAX, 0]] {
            assert_eq!(SubmissionSamples::from_timestamp(value), None);
        }
    }
}
