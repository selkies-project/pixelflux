/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Process-local identities for captured samples and bounded copies of active buffers.
//! A sample identifies pixels, not scene continuity or delivery to a consumer.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use crate::computer_use::{ScreenshotFrame, ScreenshotPixelFormat};
use crate::encoders::sample::SampleStamp;

thread_local! {
    static DELIVERY_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) struct DeliveryThreadGuard {
    previous: bool,
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl DeliveryThreadGuard {
    pub(crate) fn enter() -> Self {
        Self {
            previous: DELIVERY_THREAD.replace(true),
            _thread_bound: std::marker::PhantomData,
        }
    }
}

impl Drop for DeliveryThreadGuard {
    fn drop(&mut self) {
        DELIVERY_THREAD.set(self.previous);
    }
}

pub(crate) fn check_snapshot_caller() -> Result<(), String> {
    if DELIVERY_THREAD.get() {
        Err(
            "Capture snapshot cannot wait on a delivery thread; request it from another thread"
                .into(),
        )
    } else {
        Ok(())
    }
}

const STARTING: u8 = 0;
const SUPPORTED: u8 = 1;
const UNSUPPORTED: u8 = 2;
const STOPPED: u8 = 3;
const MAX_REQUESTS: usize = 4;
const MAX_RAW_BYTES: usize = 128 * 1024 * 1024;
static NEXT_RUN: AtomicU64 = AtomicU64::new(1);
static REQUESTS: AtomicUsize = AtomicUsize::new(0);
static RAW_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Realized buffer geometry. The origin is in root pixels on X11 and layout logical
/// coordinates on Wayland; scale maps logical coordinates to the returned pixels.
#[derive(Clone, Copy, Debug)]
pub struct SampleLayout {
    pub x: i32,
    pub y: i32,
    pub scale: f64,
    pub cursor_composited: Option<bool>,
    pub coordinate_space: &'static str,
}

struct Pending {
    reply: mpsc::SyncSender<Result<RawSnapshot, String>>,
    canceled: Arc<AtomicBool>,
    deadline: Instant,
    min_sample_seq: u64,
}

impl Pending {
    fn check(&self) -> Result<(), String> {
        if self.canceled.load(Ordering::Acquire) {
            Err("Capture snapshot canceled".into())
        } else if Instant::now() >= self.deadline {
            Err("Capture snapshot timed out".into())
        } else {
            Ok(())
        }
    }
}

/// One active capture run. Requests never retain a borrowed streaming buffer.
pub struct CaptureSamples {
    pub run_id: u64,
    sequence: AtomicU64,
    status: AtomicU8,
    pending: AtomicBool,
    outstanding: AtomicBool,
    request: Mutex<Option<Pending>>,
}

impl Default for CaptureSamples {
    #[allow(
        deprecated,
        reason = "Atomic::try_update requires a newer compiler than Rust 1.89."
    )]
    fn default() -> Self {
        Self {
            run_id: NEXT_RUN
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .expect("capture run identity exhausted"),
            sequence: AtomicU64::new(1),
            status: AtomicU8::new(STARTING),
            pending: AtomicBool::new(false),
            outstanding: AtomicBool::new(false),
            request: Mutex::new(None),
        }
    }
}

impl CaptureSamples {
    #[allow(
        deprecated,
        reason = "Atomic::try_update requires a newer compiler than Rust 1.89."
    )]
    pub fn supported(&self, supported: bool) {
        let state = if supported { SUPPORTED } else { UNSUPPORTED };
        let _ = self
            .status
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |s| {
                (s != STOPPED).then_some(state)
            });
        if !supported {
            self.reject("Capture snapshot unsupported by this capture path");
        }
    }

    pub fn stop(&self) {
        self.status.store(STOPPED, Ordering::Release);
        self.reject("Capture snapshot inactive");
    }

    pub fn guard(self: &Arc<Self>) -> CaptureRunGuard {
        CaptureRunGuard(self.clone())
    }

    #[allow(
        deprecated,
        reason = "Atomic::try_update requires a newer compiler than Rust 1.89."
    )]
    pub fn next(&self, captured_ns: i64) -> Option<SampleStamp> {
        if self.status.load(Ordering::Acquire) == STOPPED {
            return None;
        }
        let sample_seq = self
            .sequence
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .ok()?;
        Some(SampleStamp {
            run_id: self.run_id,
            sample_seq,
            captured_ns,
        })
    }

    pub fn has_pending(&self) -> bool {
        self.pending.load(Ordering::Acquire)
    }

    pub(crate) fn check(&self, expected_run: u64) -> Result<(), String> {
        if expected_run != self.run_id {
            return Err("Capture snapshot stale run".into());
        }
        match self.status.load(Ordering::Acquire) {
            SUPPORTED => Ok(()),
            STARTING => Err("Capture snapshot not ready".into()),
            UNSUPPORTED => Err("Capture snapshot unsupported by this capture path".into()),
            _ => Err("Capture snapshot inactive".into()),
        }
    }

    #[allow(
        deprecated,
        reason = "Atomic::try_update requires a newer compiler than Rust 1.89."
    )]
    pub fn begin(
        self: &Arc<Self>,
        expected_run: u64,
        timeout: Duration,
    ) -> Result<SnapshotTicket, String> {
        self.check(expected_run)?;
        if timeout.is_zero() || timeout > Duration::from_secs(30) {
            return Err(
                "Capture snapshot timeout must be greater than zero and at most 30 seconds".into(),
            );
        }
        let deadline = Instant::now() + timeout;
        let mut request = self.request.lock().unwrap();
        self.check(expected_run)?;
        if self.outstanding.swap(true, Ordering::AcqRel) {
            return Err("Capture snapshot busy".into());
        }
        if REQUESTS
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < MAX_REQUESTS).then_some(n + 1)
            })
            .is_err()
        {
            self.outstanding.store(false, Ordering::Release);
            return Err("Capture snapshot busy".into());
        }
        let canceled = Arc::new(AtomicBool::new(false));
        let (reply, rx) = mpsc::sync_channel(1);
        *request = Some(Pending {
            reply,
            canceled: canceled.clone(),
            deadline,
            min_sample_seq: self.sequence.load(Ordering::Acquire),
        });
        self.pending.store(true, Ordering::Release);
        Ok(SnapshotTicket {
            run: self.clone(),
            rx,
            canceled,
            deadline,
        })
    }

    pub fn reject(&self, error: &str) {
        let mut request = self.request.lock().unwrap();
        if let Some(pending) = request.take() {
            let _ = pending.reply.try_send(Err(error.to_string()));
        }
        self.pending.store(false, Ordering::Release);
    }

    /// Copy only an admitted request, after normal streaming publication where possible.
    /// `stamp` was allocated for this buffer before it crossed any encode handoff.
    pub fn copy_requested(
        &self,
        stamp: SampleStamp,
        pixels: &[u8],
        stride: usize,
        size: (u32, u32),
        format: ScreenshotPixelFormat,
        layout: SampleLayout,
    ) {
        self.fulfill_requested(stamp, size, format, layout, |bytes| {
            let (width, height) = size;
            let row_bytes = width as usize * 4;
            let required = stride
                .checked_mul(height.saturating_sub(1) as usize)
                .and_then(|n| n.checked_add(row_bytes))
                .ok_or("Capture snapshot size overflow")?;
            if stride < row_bytes || pixels.len() < required {
                return Err("Capture snapshot invalid buffer".into());
            }
            let mut owned = Vec::with_capacity(bytes);
            for row in pixels.chunks(stride).take(height as usize) {
                owned.extend_from_slice(&row[..row_bytes]);
            }
            Ok(owned)
        });
    }

    /// Fill an admitted snapshot directly without retaining a streaming buffer.
    pub fn fill_requested(
        &self,
        stamp: SampleStamp,
        size: (u32, u32),
        format: ScreenshotPixelFormat,
        layout: SampleLayout,
        fill: impl FnOnce(&mut [u8]) -> Result<(), String>,
    ) {
        self.fulfill_requested(stamp, size, format, layout, |bytes| {
            let mut pixels = vec![0; bytes];
            fill(&mut pixels)?;
            Ok(pixels)
        });
    }

    fn fulfill_requested(
        &self,
        stamp: SampleStamp,
        size: (u32, u32),
        format: ScreenshotPixelFormat,
        layout: SampleLayout,
        pixels: impl FnOnce(usize) -> Result<Vec<u8>, String>,
    ) {
        if !self.has_pending() {
            return;
        }
        let request = {
            let mut slot = self.request.lock().unwrap();
            if slot.as_ref().is_some_and(|r| {
                r.check().is_ok()
                    && stamp.run_id == self.run_id
                    && stamp.sample_seq < r.min_sample_seq
            }) {
                return;
            }
            let request = slot.take();
            self.pending.store(false, Ordering::Release);
            request
        };
        let Some(request) = request else { return };
        let result = (|| {
            self.check(stamp.run_id)?;
            request.check()?;
            let (width, height) = size;
            let row_bytes = (width as usize)
                .checked_mul(4)
                .ok_or("Capture snapshot size overflow")?;
            let bytes = row_bytes
                .checked_mul(height as usize)
                .ok_or("Capture snapshot size overflow")?;
            if width == 0 || height == 0 {
                return Err("Capture snapshot invalid buffer".into());
            }
            let budget = RawBudget::acquire(bytes)?;
            let owned = pixels(bytes)?;
            request.check()?;
            self.check(stamp.run_id)?;
            Ok(RawSnapshot {
                frame: ScreenshotFrame {
                    pixels: owned,
                    width,
                    height,
                    format,
                },
                stamp,
                layout,
                _budget: budget,
            })
        })();
        let _ = request.reply.try_send(result);
    }
}

pub struct CaptureRunGuard(Arc<CaptureSamples>);

impl Drop for CaptureRunGuard {
    fn drop(&mut self) {
        self.0.stop();
    }
}

struct RawBudget(usize);

impl RawBudget {
    #[allow(
        deprecated,
        reason = "Atomic::try_update requires a newer compiler than Rust 1.89."
    )]
    fn acquire(bytes: usize) -> Result<Self, String> {
        RAW_BYTES
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes)
                    .filter(|&total| total <= MAX_RAW_BYTES)
            })
            .map_err(|_| "Capture snapshot raw byte budget exceeded")?;
        Ok(Self(bytes))
    }
}

impl Drop for RawBudget {
    fn drop(&mut self) {
        RAW_BYTES.fetch_sub(self.0, Ordering::AcqRel);
    }
}

pub struct RawSnapshot {
    frame: ScreenshotFrame,
    stamp: SampleStamp,
    layout: SampleLayout,
    _budget: RawBudget,
}

pub struct PngSnapshot {
    pub png: Vec<u8>,
    pub stamp: SampleStamp,
    pub layout: SampleLayout,
    pub width: u32,
    pub height: u32,
}

/// Admission remains held during PNG compression and is released even on cancellation.
pub struct SnapshotTicket {
    run: Arc<CaptureSamples>,
    rx: mpsc::Receiver<Result<RawSnapshot, String>>,
    canceled: Arc<AtomicBool>,
    deadline: Instant,
}

impl SnapshotTicket {
    pub fn finish(self) -> Result<PngSnapshot, String> {
        self.finish_with(ScreenshotFrame::encode_png)
    }

    fn finish_with(
        self,
        encode: impl FnOnce(ScreenshotFrame) -> Result<Vec<u8>, String>,
    ) -> Result<PngSnapshot, String> {
        let raw = self
            .rx
            .recv_timeout(self.deadline.saturating_duration_since(Instant::now()))
            .map_err(|e| match e {
                mpsc::RecvTimeoutError::Timeout => "Capture snapshot timed out",
                mpsc::RecvTimeoutError::Disconnected => "Capture snapshot canceled",
            })??;
        self.check()?;
        let width = raw.frame.width;
        let height = raw.frame.height;
        let png = encode(raw.frame)?;
        self.check()?;
        Ok(PngSnapshot {
            png,
            stamp: raw.stamp,
            layout: raw.layout,
            width,
            height,
        })
    }

    fn check(&self) -> Result<(), String> {
        self.run.check(self.run.run_id)?;
        if Instant::now() >= self.deadline {
            return Err("Capture snapshot timed out".into());
        }
        Ok(())
    }
}

impl Drop for SnapshotTicket {
    fn drop(&mut self) {
        self.canceled.store(true, Ordering::Release);
        self.run.outstanding.store(false, Ordering::Release);
        REQUESTS.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static ADMISSION_TEST: Mutex<()> = Mutex::new(());

    #[test]
    fn delivery_thread_rejects_waiting_and_restores_callers_after_drop() {
        assert!(check_snapshot_caller().is_ok());
        {
            let _delivery = DeliveryThreadGuard::enter();
            assert!(
                check_snapshot_caller()
                    .unwrap_err()
                    .contains("delivery thread")
            );
            {
                let _nested = DeliveryThreadGuard::enter();
                assert!(check_snapshot_caller().is_err());
            }
            assert!(check_snapshot_caller().is_err());
            assert!(
                std::thread::spawn(check_snapshot_caller)
                    .join()
                    .unwrap()
                    .is_ok()
            );
        }
        assert!(check_snapshot_caller().is_ok());
    }

    fn active() -> Arc<CaptureSamples> {
        let run = Arc::new(CaptureSamples::default());
        run.supported(true);
        run
    }

    fn layout() -> SampleLayout {
        SampleLayout {
            x: 120,
            y: 37,
            scale: 1.0,
            cursor_composited: Some(false),
            coordinate_space: "x11-root-pixels",
        }
    }

    #[test]
    fn snapshot_owns_exact_pixels_and_their_sample() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        let run = active();
        let ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
        let sample = run.next(123).unwrap();
        let mut pixels = [3, 2, 1, 0, 99, 99, 99, 99, 6, 5, 4, 0];
        run.copy_requested(
            sample,
            &pixels,
            8,
            (1, 2),
            ScreenshotPixelFormat::Bgrx,
            layout(),
        );
        pixels.fill(0);
        let result = ticket.finish().unwrap();
        assert_eq!(result.stamp, sample);
        assert_eq!(result.layout.x, 120);
        assert_eq!(result.layout.y, 37);
        let decoded = image::load_from_memory(&result.png).unwrap().to_rgba8();
        assert_eq!(decoded.as_raw(), &[1, 2, 3, 255, 4, 5, 6, 255]);
        assert_eq!(RAW_BYTES.load(Ordering::Acquire), 0);
        assert_eq!(REQUESTS.load(Ordering::Acquire), 0);
    }

    #[test]
    fn direct_fill_owns_exact_pixels_and_their_sample() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        let run = active();
        let ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
        let sample = run.next(123).unwrap();
        run.fill_requested(
            sample,
            (2, 1),
            ScreenshotPixelFormat::Rgba,
            layout(),
            |pixels| {
                assert_eq!(pixels.len(), 8);
                assert_eq!(RAW_BYTES.load(Ordering::Acquire), 8);
                pixels.copy_from_slice(&[1, 2, 3, 255, 4, 5, 6, 127]);
                Ok(())
            },
        );
        let result = ticket.finish().unwrap();
        assert_eq!(result.stamp, sample);
        assert_eq!((result.width, result.height), (2, 1));
        assert_eq!(result.layout.x, 120);
        let decoded = image::load_from_memory(&result.png).unwrap().to_rgba8();
        assert_eq!(decoded.as_raw(), &[1, 2, 3, 255, 4, 5, 6, 127]);
        assert_eq!(RAW_BYTES.load(Ordering::Acquire), 0);
        assert_eq!(REQUESTS.load(Ordering::Acquire), 0);
    }

    #[test]
    fn direct_fill_skips_unadmitted_or_invalid_requests() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        let run = active();
        let no_fill = |_: &mut [u8]| -> Result<(), String> {
            panic!("an invalid request must not run the writer")
        };
        run.fill_requested(
            run.next(1).unwrap(),
            (1, 1),
            ScreenshotPixelFormat::Rgba,
            layout(),
            no_fill,
        );
        for error in ["canceled", "timed out", "stale", "invalid buffer"] {
            let ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
            let mut sample = run.next(2).unwrap();
            let mut size = (1, 1);
            match error {
                "canceled" => ticket.canceled.store(true, Ordering::Release),
                "timed out" => {
                    run.request.lock().unwrap().as_mut().unwrap().deadline = Instant::now();
                }
                "stale" => sample.run_id = active().run_id,
                _ => size = (0, 1),
            }
            run.fill_requested(sample, size, ScreenshotPixelFormat::Rgba, layout(), no_fill);
            assert!(ticket.finish().err().unwrap().contains(error));
            assert!(!run.has_pending());
            assert_eq!(RAW_BYTES.load(Ordering::Acquire), 0);
            assert_eq!(REQUESTS.load(Ordering::Acquire), 0);
        }
        let ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
        let budget = RawBudget::acquire(MAX_RAW_BYTES).unwrap();
        run.fill_requested(
            run.next(3).unwrap(),
            (1, 1),
            ScreenshotPixelFormat::Rgba,
            layout(),
            no_fill,
        );
        assert!(ticket.finish().err().unwrap().contains("raw byte budget"));
        assert_eq!(RAW_BYTES.load(Ordering::Acquire), MAX_RAW_BYTES);
        assert_eq!(REQUESTS.load(Ordering::Acquire), 0);
        drop(budget);
        assert_eq!(RAW_BYTES.load(Ordering::Acquire), 0);
    }

    #[test]
    fn direct_fill_waits_past_samples_already_in_flight() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        let run = active();
        let old = run.next(1).unwrap();
        let ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
        run.fill_requested(old, (1, 1), ScreenshotPixelFormat::Rgba, layout(), |_| {
            panic!("an old sample must leave the request pending")
        });
        assert!(run.has_pending());
        let new = run.next(2).unwrap();
        run.fill_requested(
            new,
            (1, 1),
            ScreenshotPixelFormat::Rgba,
            layout(),
            |pixels| {
                pixels.fill(255);
                Ok(())
            },
        );
        assert_eq!(ticket.finish().unwrap().stamp, new);
        assert_eq!(RAW_BYTES.load(Ordering::Acquire), 0);
        assert_eq!(REQUESTS.load(Ordering::Acquire), 0);
    }

    #[test]
    fn direct_fill_errors_and_stop_release_resources() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        for stop in [false, true] {
            let run = active();
            let ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
            run.fill_requested(
                run.next(1).unwrap(),
                (1, 1),
                ScreenshotPixelFormat::Rgba,
                layout(),
                |pixels| {
                    assert_eq!(RAW_BYTES.load(Ordering::Acquire), 4);
                    assert!(
                        run.begin(run.run_id, Duration::from_secs(1))
                            .err()
                            .unwrap()
                            .contains("busy")
                    );
                    pixels.fill(255);
                    if stop {
                        run.stop();
                        Ok(())
                    } else {
                        Err("readback failed".into())
                    }
                },
            );
            let error = ticket.finish().err().unwrap();
            assert!(error.contains(if stop { "inactive" } else { "readback failed" }));
            assert!(!run.has_pending());
            assert_eq!(RAW_BYTES.load(Ordering::Acquire), 0);
            assert_eq!(REQUESTS.load(Ordering::Acquire), 0);
        }
    }

    #[test]
    fn stop_rejects_queued_and_already_copied_snapshots() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        for copy in [false, true] {
            let run = active();
            let ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
            if copy {
                run.copy_requested(
                    run.next(1).unwrap(),
                    &[0; 4],
                    4,
                    (1, 1),
                    ScreenshotPixelFormat::Rgba,
                    layout(),
                );
            }
            run.stop();
            assert!(ticket.finish().err().unwrap().contains("inactive"));
            assert!(!run.has_pending());
            assert!(run.next(2).is_none());
            run.supported(true);
            assert!(run.begin(run.run_id, Duration::from_secs(1)).is_err());
        }
        assert_eq!(RAW_BYTES.load(Ordering::Acquire), 0);
    }

    #[test]
    fn old_run_and_other_output_are_rejected() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        let old = active();
        let new = active();
        assert_ne!(old.run_id, new.run_id);
        assert!(
            new.begin(old.run_id, Duration::from_secs(1))
                .err()
                .unwrap()
                .contains("stale")
        );
        let ticket = new.begin(new.run_id, Duration::from_secs(1)).unwrap();
        new.copy_requested(
            old.next(1).unwrap(),
            &[0; 4],
            4,
            (1, 1),
            ScreenshotPixelFormat::Rgba,
            layout(),
        );
        assert!(ticket.finish().err().unwrap().contains("stale"));
    }

    #[test]
    fn admission_covers_handoff_and_compression() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        let run = active();
        let ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
        run.copy_requested(
            run.next(1).unwrap(),
            &[0; 4],
            4,
            (1, 1),
            ScreenshotPixelFormat::Rgba,
            layout(),
        );
        assert!(!run.has_pending());
        assert!(
            run.begin(run.run_id, Duration::from_secs(1))
                .err()
                .unwrap()
                .contains("busy")
        );
        ticket.finish().unwrap();
        assert!(run.begin(run.run_id, Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn cancellation_timeout_and_invalid_buffers_release_resources() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        let run = active();
        drop(run.begin(run.run_id, Duration::from_secs(1)).unwrap());
        run.copy_requested(
            run.next(1).unwrap(),
            &[0; 4],
            4,
            (1, 1),
            ScreenshotPixelFormat::Rgba,
            layout(),
        );
        let mut ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
        ticket.deadline = Instant::now();
        assert!(ticket.finish().err().unwrap().contains("timed out"));
        let ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
        run.copy_requested(
            run.next(2).unwrap(),
            &[0; 3],
            4,
            (1, 1),
            ScreenshotPixelFormat::Rgba,
            layout(),
        );
        assert!(ticket.finish().err().unwrap().contains("invalid buffer"));
        assert_eq!(RAW_BYTES.load(Ordering::Acquire), 0);
        assert_eq!(REQUESTS.load(Ordering::Acquire), 0);
    }

    #[test]
    fn global_request_and_raw_byte_limits_are_independent() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        let runs: Vec<_> = (0..=MAX_REQUESTS).map(|_| active()).collect();
        let tickets: Vec<_> = runs[..MAX_REQUESTS]
            .iter()
            .map(|run| run.begin(run.run_id, Duration::from_secs(1)).unwrap())
            .collect();
        let last = &runs[MAX_REQUESTS];
        assert!(last.begin(last.run_id, Duration::from_secs(1)).is_err());
        let budget = RawBudget::acquire(MAX_RAW_BYTES).unwrap();
        assert!(RawBudget::acquire(1).is_err());
        drop(budget);
        drop(tickets);
        assert!(last.begin(last.run_id, Duration::from_secs(1)).is_ok());
        assert_eq!(RAW_BYTES.load(Ordering::Acquire), 0);
    }

    #[test]
    fn sample_counter_never_wraps() {
        let run = active();
        run.sequence.store(u64::MAX - 1, Ordering::Relaxed);
        assert_eq!(run.next(1).unwrap().sample_seq, u64::MAX - 1);
        assert!(run.next(2).is_none());
    }

    #[test]
    fn snapshot_waits_past_samples_already_in_flight() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        let run = active();
        let old = run.next(1).unwrap();
        let ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
        run.copy_requested(
            old,
            &[0; 4],
            4,
            (1, 1),
            ScreenshotPixelFormat::Rgba,
            layout(),
        );
        assert!(run.has_pending());
        let new = run.next(2).unwrap();
        run.copy_requested(
            new,
            &[255; 4],
            4,
            (1, 1),
            ScreenshotPixelFormat::Rgba,
            layout(),
        );
        assert_eq!(ticket.finish().unwrap().stamp, new);
    }

    #[test]
    fn run_guard_invalidates_waiters() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        let run = active();
        let guard = run.guard();
        let ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
        drop(guard);
        assert!(ticket.finish().err().unwrap().contains("inactive"));
    }

    #[test]
    fn stop_during_compression_discards_result_and_releases_admission() {
        let _serial = ADMISSION_TEST.lock().unwrap();
        let run = active();
        let ticket = run.begin(run.run_id, Duration::from_secs(1)).unwrap();
        run.copy_requested(
            run.next(1).unwrap(),
            &[0; 4],
            4,
            (1, 1),
            ScreenshotPixelFormat::Rgba,
            layout(),
        );
        let result = ticket.finish_with(|frame| {
            assert!(
                run.begin(run.run_id, Duration::from_secs(1))
                    .err()
                    .unwrap()
                    .contains("busy")
            );
            assert_eq!(RAW_BYTES.load(Ordering::Acquire), 4);
            run.stop();
            frame.encode_png()
        });
        assert!(result.err().unwrap().contains("inactive"));
        assert_eq!(RAW_BYTES.load(Ordering::Acquire), 0);
        assert_eq!(REQUESTS.load(Ordering::Acquire), 0);
    }
}
