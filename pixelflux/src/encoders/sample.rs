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
