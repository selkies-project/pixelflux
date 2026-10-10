/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Opt-in continuity of successfully rendered local Wayland scenes.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SceneIdentity {
    pub source_id: u64,
    pub scene_id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SceneLayout {
    pub output_id: u32,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub scale_bits: u64,
    pub format: u32,
    pub cursor: bool,
}

#[derive(Default)]
struct State {
    supported: bool,
    encoder_supported: bool,
    stopped: bool,
    exhausted: bool,
    enabled: bool,
    source: u64,
    scene: u64,
    current: Option<(SceneLayout, SceneIdentity)>,
}

#[derive(Default)]
pub(crate) struct SceneTracker {
    enabled: AtomicBool,
    state: Mutex<State>,
}

impl SceneTracker {
    pub fn set_supported(&self, supported: bool) {
        let mut state = self.state.lock().unwrap();
        state.supported = supported && !state.stopped;
        if !state.supported {
            state.enabled = false;
            state.current = None;
            self.enabled.store(false, Ordering::Release);
        }
    }

    pub fn set_encoder_supported(&self, supported: bool) {
        let mut state = self.state.lock().unwrap();
        state.encoder_supported = supported && !state.stopped;
        if !state.encoder_supported {
            state.enabled = false;
            state.current = None;
            self.enabled.store(false, Ordering::Release);
        }
    }

    pub fn supported(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.supported && state.encoder_supported && !state.stopped && !state.exhausted
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    pub fn set_enabled(&self, enabled: bool) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();
        if state.stopped {
            return Err("Scene tracking inactive".into());
        }
        if enabled && (!state.supported || !state.encoder_supported) {
            return Err("Scene tracking unsupported; requires local Wayland and an encoder with sample identity".into());
        }
        if enabled && state.exhausted {
            return Err("Scene tracking identity exhausted".into());
        }
        if state.enabled != enabled {
            state.current = None;
            state.enabled = enabled;
            self.enabled.store(enabled, Ordering::Release);
        }
        Ok(())
    }

    pub fn stop(&self) {
        let mut state = self.state.lock().unwrap();
        state.stopped = true;
        state.supported = false;
        state.enabled = false;
        state.current = None;
        self.enabled.store(false, Ordering::Release);
    }

    pub fn invalidate(&self) {
        if !self.enabled() {
            return;
        }
        self.state.lock().unwrap().current = None;
    }

    pub fn observe(&self, layout: SceneLayout, changed: bool) -> Option<SceneIdentity> {
        if !self.enabled() {
            return None;
        }
        let mut state = self.state.lock().unwrap();
        if !state.enabled {
            return None;
        }
        let scale = f64::from_bits(layout.scale_bits);
        if layout.width <= 0 || layout.height <= 0 || !scale.is_finite() || scale <= 0.0 {
            state.current = None;
            return None;
        }
        let source_changed = state.current.is_none_or(|(old, _)| old != layout);
        if source_changed || changed {
            let source = if source_changed {
                state.source.checked_add(1)
            } else {
                Some(state.source)
            };
            let scene = state.scene.checked_add(1);
            let (Some(source), Some(scene)) = (source, scene) else {
                state.exhausted = true;
                state.enabled = false;
                state.current = None;
                self.enabled.store(false, Ordering::Release);
                return None;
            };
            state.source = source;
            state.scene = scene;
            state.current = Some((
                layout,
                SceneIdentity {
                    source_id: source,
                    scene_id: scene,
                },
            ));
        }
        state.current.map(|(_, identity)| identity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    fn layout() -> SceneLayout {
        SceneLayout {
            output_id: 1,
            x: 0,
            y: 0,
            width: 320,
            height: 180,
            scale_bits: 1.0_f64.to_bits(),
            format: 1,
            cursor: false,
        }
    }

    fn active() -> SceneTracker {
        let tracker = SceneTracker::default();
        tracker.set_supported(true);
        tracker.set_encoder_supported(true);
        tracker.set_enabled(true).unwrap();
        tracker
    }

    #[test]
    fn disabled_render_never_locks_or_allocates_an_identity() {
        let tracker = Arc::new(SceneTracker::default());
        let state = tracker.state.lock().unwrap();
        let other = tracker.clone();
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let value = other.observe(layout(), true);
            other.invalidate();
            tx.send(value).unwrap();
        });
        let observed = rx.recv_timeout(Duration::from_secs(1));
        assert_eq!((state.source, state.scene), (0, 0));
        drop(state);
        worker.join().unwrap();
        assert_eq!(observed.unwrap(), None);
    }

    #[test]
    fn stable_render_keeps_scene_and_damage_advances_without_replacing_source() {
        let tracker = active();
        let first = tracker.observe(layout(), false).unwrap();
        assert_eq!(tracker.observe(layout(), false), Some(first));
        let repaint = tracker.observe(layout(), true).unwrap();
        assert_eq!(first.source_id, repaint.source_id);
        assert!(repaint.scene_id > first.scene_id);
    }

    #[test]
    fn every_layout_component_retires_source() {
        let mut variants = [layout(); 8];
        variants[0].output_id += 1;
        variants[1].x += 1;
        variants[2].y += 1;
        variants[3].width += 1;
        variants[4].height += 1;
        variants[5].scale_bits = 1.5_f64.to_bits();
        variants[6].format += 1;
        variants[7].cursor = true;
        for variant in variants {
            let tracker = active();
            let first = tracker.observe(layout(), false).unwrap();
            let next = tracker.observe(variant, false).unwrap();
            assert!(next.source_id > first.source_id && next.scene_id > first.scene_id);
        }
    }

    #[test]
    fn disable_and_render_failure_never_reuse_accepted_identity() {
        let tracker = active();
        let first = tracker.observe(layout(), false).unwrap();
        tracker.set_enabled(true).unwrap();
        assert_eq!(tracker.observe(layout(), false), Some(first));
        tracker.set_enabled(false).unwrap();
        assert_eq!(tracker.observe(layout(), true), None);
        tracker.set_enabled(true).unwrap();
        let second = tracker.observe(layout(), false).unwrap();
        tracker.invalidate();
        let third = tracker.observe(layout(), false).unwrap();
        assert!(first.source_id < second.source_id && second.source_id < third.source_id);
        assert!(first.scene_id < second.scene_id && second.scene_id < third.scene_id);
    }

    #[test]
    fn unsupported_and_stopped_capture_cannot_enable_or_recover() {
        let tracker = SceneTracker::default();
        assert!(!tracker.supported());
        assert!(
            tracker
                .set_enabled(true)
                .unwrap_err()
                .contains("unsupported")
        );
        tracker.set_supported(true);
        tracker.set_encoder_supported(true);
        tracker.set_enabled(true).unwrap();
        tracker.stop();
        tracker.set_supported(true);
        assert!(!tracker.supported() && !tracker.enabled());
        assert!(tracker.set_enabled(true).unwrap_err().contains("inactive"));
        assert_eq!(tracker.observe(layout(), true), None);
    }

    #[test]
    fn realized_encoder_capability_is_required_and_demotions_retire_tracking() {
        for encoder_first in [false, true] {
            let tracker = SceneTracker::default();
            if encoder_first {
                tracker.set_encoder_supported(true);
            } else {
                tracker.set_supported(true);
            }
            assert!(!tracker.supported());
            assert!(tracker.set_enabled(true).is_err());
            tracker.set_supported(true);
            tracker.set_encoder_supported(true);
            assert!(tracker.supported() && !tracker.enabled());
            tracker.set_enabled(true).unwrap();
            let first = tracker.observe(layout(), false).unwrap();
            tracker.set_encoder_supported(false);
            assert!(!tracker.supported() && !tracker.enabled());
            assert_eq!(tracker.observe(layout(), true), None);
            tracker.set_encoder_supported(true);
            assert!(!tracker.enabled());
            tracker.set_enabled(true).unwrap();
            let recovered = tracker.observe(layout(), false).unwrap();
            assert!(recovered.source_id > first.source_id && recovered.scene_id > first.scene_id);
        }
    }

    #[test]
    fn unknown_layout_retires_identity_before_recovery() {
        let tracker = active();
        let first = tracker.observe(layout(), false).unwrap();
        let mut invalid = layout();
        invalid.scale_bits = f64::NAN.to_bits();
        assert_eq!(tracker.observe(invalid, false), None);
        let next = tracker.observe(layout(), false).unwrap();
        assert!(next.source_id > first.source_id && next.scene_id > first.scene_id);
    }

    #[test]
    fn exhausted_ids_fail_closed_without_wrapping_or_reactivation() {
        let tracker = active();
        tracker.state.lock().unwrap().scene = u64::MAX;
        assert_eq!(tracker.observe(layout(), true), None);
        assert!(!tracker.enabled() && !tracker.supported());
        assert!(tracker.set_enabled(true).unwrap_err().contains("exhausted"));
    }
}
