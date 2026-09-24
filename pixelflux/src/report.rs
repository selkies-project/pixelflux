/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! What a capture streams and how it got there, for the caller to show its user: the capture
//! path and whether it is zero-copy, the encoder and whether it is hardware, the GPU behind
//! it, and the reason a faster path was declined. The same facts the tagged log lines state,
//! kept as values so nobody has to read a log to learn them.
//!
//! A capture's decisions are spread over three backends and the shared encoder ladder, each
//! made on the thread that owns the capture being built, so the report a decision lands in is
//! the one bound to the deciding thread ([`enter`]) rather than an argument threaded through
//! every site. A decision made with no report bound, as in a test or a probe, records nothing.
//!
//! Counters are relaxed atomics tallied once per delivered frame and read only when the
//! caller asks, so a capture nobody inspects pays for a few additions and nothing else.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::encoders::software::EncodedStripe;
use crate::RustCaptureSettings;

/// The description half of a report. `capture_reason` says why the capture is not zero-copy
/// and `encoder_reason` why the encoder is not hardware; both are empty where there is
/// nothing to explain.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamInfo {
    pub backend: &'static str,
    pub capture: &'static str,
    pub zero_copy: bool,
    pub capture_reason: String,
    pub encoder: String,
    pub hardware: bool,
    pub encoder_reason: String,
    pub codec: &'static str,
    pub fullcolor: bool,
    pub full_range: bool,
    pub stripes: usize,
    pub gpu: String,
    pub driver: String,
    pub encode_node: i32,
}

/// Cumulative counters of a capture since it started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamTotals {
    pub frames: u64,
    pub bytes: u64,
    pub encode_ns: u64,
    pub pipeline_ns: u64,
}

/// One capture's description and counters.
#[derive(Default)]
pub struct StreamReport {
    info: Mutex<StreamInfo>,
    frames: AtomicU64,
    bytes: AtomicU64,
    encode_ns: AtomicU64,
    pipeline_ns: AtomicU64,
}

impl StreamReport {
    pub fn new(backend: &'static str) -> Arc<Self> {
        let report = Self::default();
        report.info.lock().unwrap().backend = backend;
        Arc::new(report)
    }

    pub fn info(&self) -> StreamInfo {
        self.info.lock().unwrap().clone()
    }

    pub fn totals(&self) -> StreamTotals {
        StreamTotals {
            frames: self.frames.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            encode_ns: self.encode_ns.load(Ordering::Relaxed),
            pipeline_ns: self.pipeline_ns.load(Ordering::Relaxed),
        }
    }

    /// Count one delivered frame: its bytes over every stripe, the time its encode took, and
    /// the time from capture to the end of the encode.
    pub fn tally(&self, stripes: &[EncodedStripe]) {
        let Some(first) = stripes.first() else { return };
        let bytes: usize = stripes.iter().map(|s| s.data.len()).sum();
        let timing = first.timing;
        self.frames.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        if timing.encode_end_ns > 0 {
            let encode = (timing.encode_end_ns - timing.encode_start_ns).max(0) as u64;
            let pipeline = (timing.encode_end_ns - timing.capture_ns).max(0) as u64;
            self.encode_ns.fetch_add(encode, Ordering::Relaxed);
            self.pipeline_ns.fetch_add(pipeline, Ordering::Relaxed);
        }
    }

    fn update(&self, change: impl FnOnce(&mut StreamInfo)) {
        change(&mut self.info.lock().unwrap());
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<StreamReport>>> = const { RefCell::new(None) };
}

/// Restores the report that was bound before [`enter`].
pub struct Scope(Option<Arc<StreamReport>>);

impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|current| *current.borrow_mut() = self.0.take());
    }
}

/// Bind `report` to this thread until the scope drops: every decision recorded meanwhile
/// lands in it. The Wayland event loop serves every display, so it binds per capture start
/// and per recovery; a capture's own threads bind once.
pub fn enter(report: &Arc<StreamReport>) -> Scope {
    Scope(CURRENT.with(|current| current.borrow_mut().replace(report.clone())))
}

fn record(change: impl FnOnce(&mut StreamInfo)) {
    CURRENT.with(|current| {
        if let Some(report) = current.borrow().as_ref() {
            report.update(change);
        }
    });
}

/// The capture path taken. A zero-copy path needs no reason, so the ones collected from the
/// paths declined ahead of it are dropped.
pub fn capture(path: &'static str, zero_copy: bool) {
    record(|info| {
        info.capture = path;
        info.zero_copy = zero_copy;
        if zero_copy {
            info.capture_reason.clear();
        }
    });
}

/// Why a zero-copy path was not taken. A capture that declines several keeps each reason.
pub fn capture_declined(path: &str, reason: &str) {
    record(|info| {
        if !info.capture_reason.is_empty() {
            info.capture_reason.push_str("; ");
        }
        info.capture_reason.push_str(&format!("{path}: {reason}"));
    });
}

/// Why the capture is not zero-copy, replacing whatever was collected.
pub fn capture_reason(reason: &str) {
    record(|info| info.capture_reason = reason.to_string());
}

/// The hardware session the ladder opened, and the device it runs on.
pub fn hardware_encoder(gpu: &str, driver: &str, node: i32) {
    record(|info| {
        info.gpu = gpu.to_string();
        info.driver = driver.to_string();
        info.encode_node = node;
        info.encoder_reason.clear();
    });
}

/// Why the session does not encode in hardware.
pub fn encoder_reason(reason: &str) {
    record(|info| info.encoder_reason = reason.to_string());
}

/// What the stream settled on, from the same inputs as the `Stream settings active` line.
/// `backend` is the full-frame session as `(name, hardware)`, `None` the striped software path.
pub fn stream(
    settings: &RustCaptureSettings,
    stripes: usize,
    backend: Option<(&str, bool)>,
    fullcolor: bool,
    full_range: bool,
) {
    record(|info| {
        let (name, hardware) = match backend {
            Some((name, hardware)) => (name, hardware),
            None if settings.codec.is_video() => (crate::encoders::software_library(settings.codec), false),
            None => ("turbojpeg", false),
        };
        info.encoder = name.to_string();
        info.hardware = hardware;
        if !hardware {
            info.gpu.clear();
            info.driver.clear();
            info.encode_node = -1;
        }
        info.codec = settings.codec.name();
        info.fullcolor = fullcolor;
        info.full_range = full_range;
        info.stripes = stripes;
    });
}

/// How the Wayland compositor renders, which every capture of it shares: `gl` on a render
/// node with the renderer's own name for the GPU, or `pixman` in software with the reason.
#[derive(Clone, Debug, Default)]
pub struct Renderer {
    pub kind: &'static str,
    pub node: String,
    pub gpu: String,
    pub reason: String,
}

static RENDERER: OnceLock<Mutex<Renderer>> = OnceLock::new();

pub fn set_renderer(renderer: Renderer) {
    *RENDERER.get_or_init(Default::default).lock().unwrap() = renderer;
}

pub fn renderer() -> Renderer {
    RENDERER.get_or_init(Default::default).lock().unwrap().clone()
}

static WAYLAND: OnceLock<Mutex<HashMap<u32, Arc<StreamReport>>>> = OnceLock::new();

/// The report of each Wayland display's capture, read by `ScreenCapture::stream_info`
/// without a round trip to the compositor thread.
pub fn wayland_reports() -> &'static Mutex<HashMap<u32, Arc<StreamReport>>> {
    WAYLAND.get_or_init(Default::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A decision lands in the report bound to the thread and nowhere without one, and a
    /// zero-copy path drops the reasons collected ahead of it.
    #[test]
    fn decisions_land_in_the_bound_report() {
        capture_declined("NvFBC", "unbound");
        let report = StreamReport::new("x11");
        {
            let _scope = enter(&report);
            capture_declined("NvFBC", "the codec is JPEG");
            capture_declined("DRI3", "no DRI3");
            capture("XShm", false);
        }
        capture_declined("NvFBC", "unbound");
        let info = report.info();
        assert_eq!(info.backend, "x11");
        assert_eq!(info.capture, "XShm");
        assert_eq!(info.capture_reason, "NvFBC: the codec is JPEG; DRI3: no DRI3");
        {
            let _scope = enter(&report);
            capture("DRI3", true);
        }
        assert!(report.info().zero_copy);
        assert!(report.info().capture_reason.is_empty());
    }

    /// Scopes nest, so a capture start on a thread already serving another restores it.
    #[test]
    fn scopes_restore_the_outer_report() {
        let outer = StreamReport::new("wayland");
        let inner = StreamReport::new("wayland");
        let _outer_scope = enter(&outer);
        {
            let _inner_scope = enter(&inner);
            encoder_reason("inner");
        }
        encoder_reason("outer");
        assert_eq!(inner.info().encoder_reason, "inner");
        assert_eq!(outer.info().encoder_reason, "outer");
    }
}
