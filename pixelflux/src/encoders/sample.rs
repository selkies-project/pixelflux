/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Captured-sample provenance, frozen before encoding and independent of wire frame IDs.

/// Process-local captured pixels and optional conservative scene continuity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleStamp {
    pub run_id: u64,
    pub sample_seq: u64,
    pub captured_ns: i64,
    pub source_id: Option<u64>,
    pub scene_id: Option<u64>,
}
