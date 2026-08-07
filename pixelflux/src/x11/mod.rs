/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! X11 host capture: grab the root window into host memory as BGRA, composite the XFixes hardware
//! cursor and the watermark on the CPU, and feed each frame to [`pipeline::X11Pipeline`], which owns
//! damage/stripe/encode. The grab goes through a shared-memory segment (XShm via x11rb) rather than
//! a plain `GetImage` for one reason: a full-screen frame is far too large to copy through the X
//! protocol socket every tick, so XShm has the server write the pixels straight into memory this
//! process already has mapped.
//!
//! `run_capture` splits the work across two threads because grabbing the next frame and encoding
//! the previous one have no reason to wait on each other: the caller's thread grabs frames and owns
//! the x11rb connection and the pool of shm surfaces, while a spawned encode thread owns the
//! [`pipeline::X11Pipeline`] (and thus the encoder) and runs the delivery callback, so the two overlap for
//! throughput. Frames hand off through a bounded `FramePool` that carries only a raw pointer and
//! geometry — never an X object, none of which is safe to share — so nothing X-related ever crosses
//! the thread boundary and the encoder never has to touch X. Multi-instance safety for the encoders
//! is handled inside them (e.g. the libx264 open/close lock); each capture owns its own private xcb
//! connection, so there is no shared X state to serialize here.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use x11rb::connection::Connection;
use x11rb::protocol::shm::ConnectionExt as ShmExt;
use x11rb::protocol::xfixes::ConnectionExt as XfixesExt;
use x11rb::protocol::xproto::{ConnectionExt as XprotoExt, ImageFormat};
use x11rb::rust_connection::RustConnection;

use crate::encoders::overlay::blend_pixel_premultiplied;
use crate::encoders::software::EncodedStripe;
use crate::pipeline::X11Pipeline;
use crate::recording_sink::RecordingSink;
use crate::RustCaptureSettings;

pub mod computer_use;
pub mod cursor;

/// Cross-thread controls for a running capture: a bag of atomics (plus two mutex-guarded
/// payloads) the owning `ScreenCapture` pyclass flips from the Python thread and the capture thread
/// reads at the top of each iteration.
///
/// It exists so Python never has to reach into the pipeline. The `X11Pipeline` and its encoder are
/// only safe to touch from the capture and encode threads, so `request_idr` / rate / fps / region
/// changes are posted here as atomic flags and applied later on the thread that owns the pipeline,
/// rather than mutating encoder state across the thread boundary from Python.
///
/// 1. **Lifecycle**: `stop` ends the capture loop; `force_idr` requests an on-demand keyframe on the
///    next processed frame.
/// 2. **Rate control** (gated by `rate_dirty`): `bitrate_kbps`, `vbv_mult_milli` (the VBV frame-time
///    multiplier * 1000, held as an integer for atomics; `<= 0` selects the policy default), and
///    `fps_milli` (target fps * 1000, re-read every frame for dynamic pacing and rate control). These
///    atomics always hold the CURRENT values, so a pipeline rebuild can carry live rates forward.
/// 3. **Tunables** (gated by `tunables_dirty`): one `LiveTunables` struct behind `tunables`, set
///    rarely, carrying the per-frame quality knobs for the encode thread.
/// 4. **Capture geometry**: `capture_cursor` toggles the cursor overlay (read on every grab);
///    `region_dirty` guards `region` = `(x, y, w, h)` (w/h `<= 0` extends to the root edge), applied
///    by the capture thread with the same drain/recreate machinery as auto-adjust.
pub struct Controls {
    pub stop: AtomicBool,
    pub force_idr: AtomicBool,
    pub rate_dirty: AtomicBool,
    pub bitrate_kbps: AtomicI32,
    pub vbv_mult_milli: AtomicI32,
    pub fps_milli: AtomicU64,
    pub tunables_dirty: AtomicBool,
    pub tunables: Mutex<Option<crate::LiveTunables>>,
    pub capture_cursor: AtomicBool,
    pub region_dirty: AtomicBool,
    pub region: Mutex<(i32, i32, i32, i32)>,
}

impl Controls {
    /// Seed the controls from the initial settings so the first loop iteration reads the
    /// configured bitrate / VBV / fps / region / cursor state rather than defaults.
    pub fn new(s: &RustCaptureSettings) -> Self {
        Self {
            stop: AtomicBool::new(false),
            force_idr: AtomicBool::new(false),
            rate_dirty: AtomicBool::new(false),
            bitrate_kbps: AtomicI32::new(s.video_bitrate_kbps),
            vbv_mult_milli: AtomicI32::new((s.video_vbv_multiplier * 1000.0).round() as i32),
            fps_milli: AtomicU64::new((s.target_fps.max(1.0) * 1000.0) as u64),
            tunables_dirty: AtomicBool::new(false),
            tunables: Mutex::new(None),
            capture_cursor: AtomicBool::new(s.capture_cursor),
            region_dirty: AtomicBool::new(false),
            region: Mutex::new((s.capture_x, s.capture_y, s.width, s.height)),
        }
    }
}

/// A shared-memory image surface: a POSIX shm segment mapped into both this process and the
/// X server, into which `shm_get_image` writes one BGRA frame.
///
/// Both ends map the same physical pages, which is the whole point: a full-screen grab then costs no
/// per-frame copy across the X socket, because the server's blit lands directly in memory the
/// capture thread already reads. The surface owns the server-side segment id (`shmseg`), the local
/// mapping (`addr` / `size`), and the geometry (`width` / `height` / `stride`) needed to interpret
/// the bytes.
struct ShmSurface {
    shmseg: u32,
    addr: *mut u8,
    size: usize,
    width: u16,
    height: u16,
    stride: usize,
}

impl ShmSurface {
    /// Allocate a shm segment of `width*height*4` bytes, attach it both locally and to the X
    /// server, and hand back a surface both ends can read.
    ///
    /// The segment is created with `shmget(IPC_PRIVATE)` and mapped locally via `shmat`, then
    /// attached on the server (`shm_attach`). That attach is confirmed with a round-trip `check()`
    /// BEFORE the segment is marked `IPC_RMID`: deleting only after both ends hold a reference means
    /// the kernel frees the segment once this process and the server have both detached, never while
    /// either still needs it. Every failure path unwinds the partial state (local detach and/or
    /// `IPC_RMID`) so a failed allocation leaks neither memory nor a server attachment; a zero-sized
    /// request is rejected up front.
    fn create(conn: &RustConnection, width: u16, height: u16) -> Result<Self, String> {
        let stride = width as usize * 4;
        let size = stride * height as usize;
        if size == 0 {
            return Err("zero-sized capture surface".into());
        }
        unsafe {
            let shmid = libc::shmget(libc::IPC_PRIVATE, size, libc::IPC_CREAT | 0o600);
            if shmid < 0 {
                return Err("shmget failed".into());
            }
            let addr = libc::shmat(shmid, std::ptr::null(), 0);
            if addr == (-1isize) as *mut libc::c_void {
                libc::shmctl(shmid, libc::IPC_RMID, std::ptr::null_mut());
                return Err("shmat failed".into());
            }
            let shmseg = match conn.generate_id() {
                Ok(id) => id,
                Err(e) => {
                    libc::shmdt(addr);
                    libc::shmctl(shmid, libc::IPC_RMID, std::ptr::null_mut());
                    return Err(format!("generate_id: {e}"));
                }
            };
            let attach = conn
                .shm_attach(shmseg, shmid as u32, false)
                .map_err(|e| format!("shm_attach: {e}"))
                .and_then(|c| c.check().map_err(|e| format!("shm_attach check: {e}")));
            libc::shmctl(shmid, libc::IPC_RMID, std::ptr::null_mut());
            if let Err(e) = attach {
                libc::shmdt(addr);
                return Err(e);
            }
            Ok(Self {
                shmseg,
                addr: addr as *mut u8,
                size,
                width,
                height,
                stride,
            })
        }
    }

    /// Borrow the mapped segment as a mutable byte slice for the capture blit and overlays.
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.addr, self.size) }
    }

    /// Borrow the mapped segment read-only, for the stability comparison.
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.addr, self.size) }
    }

    /// Detach the segment from the X server and then locally, clearing the mapped pointer.
    ///
    /// This is a manual method rather than a `Drop` impl because releasing the server-side
    /// attachment is an X protocol request that needs the live connection, which a `Drop` could not
    /// reach; the owner of the connection must call it before the surface is dropped. The order —
    /// server `shm_detach` first, then the local `shmdt` — completes the release of the segment that
    /// was marked `IPC_RMID` at creation, so the kernel reclaims it once neither end holds it.
    fn destroy(&mut self, conn: &RustConnection) {
        let _ = conn.shm_detach(self.shmseg);
        let _ = conn.flush();
        unsafe {
            libc::shmdt(self.addr as *mut libc::c_void);
        }
        self.addr = std::ptr::null_mut();
    }
}

/// Reject a server whose root visual is not 32 bits per pixel.
///
/// Every `ShmSurface` is sized at `width * 4` bytes per row and the capture is read as packed
/// BGRA, so a root of any other depth would be grabbed as garbage rather than failing. Checked
/// on the initial connection and again on every reconnect, since a restarted server can come
/// back with a different visual.
fn require_32bpp(conn: &RustConnection, screen: usize) -> Result<(), String> {
    let root_depth = conn.setup().roots[screen].root_depth;
    let bpp = conn
        .setup()
        .pixmap_formats
        .iter()
        .find(|f| f.depth == root_depth)
        .map(|f| f.bits_per_pixel)
        .unwrap_or(32);
    if bpp != 32 {
        return Err(format!(
            "unsupported root depth {root_depth} ({bpp} bpp); only 32-bpp BGRA is supported"
        ));
    }
    Ok(())
}

/// Clamp a capture offset to the live root so a grab always has at least one root pixel in
/// range: a past-end offset fails every `shm_get_image` with BadMatch and would burn out the
/// failure budget with no chance of recovery. The upper clamp also keeps protocol-range roots
/// (INT16 coordinates) from truncating a wider value through the `as i16` cast.
fn clamp_offset(x: i32, root: u16) -> i16 {
    let upper = (root as i32).saturating_sub(1).min(i16::MAX as i32 - 1);
    x.max(0).min(upper) as i16
}

/// Resolve the capture dimensions `shm_get_image` will read, bounded so the region can never
/// run past the live root from the (clamped) capture offset: with auto-adjust (or unset
/// width/height `<= 0`) the capture tracks the full root minus the offset, otherwise the
/// requested size is clamped to what is available from it. Saturating subtraction plus a final
/// `u16` clamp (minimum 2) keep pathological settings from overflowing or collapsing to an
/// unusable surface. H.264 (`output_mode == 1`) requires even dimensions, so both are then
/// rounded down to a multiple of two.
fn resolve_dims(root_w: u16, root_h: u16, s: &RustCaptureSettings) -> (u16, u16) {
    let cap_x = clamp_offset(s.capture_x, root_w) as i32;
    let cap_y = clamp_offset(s.capture_y, root_h) as i32;
    let avail_w = (root_w as i32).saturating_sub(cap_x).max(2);
    let avail_h = (root_h as i32).saturating_sub(cap_y).max(2);
    let mut w = if s.auto_adjust_screen_capture_size || s.width <= 0 {
        avail_w
    } else {
        s.width.min(avail_w)
    };
    let mut h = if s.auto_adjust_screen_capture_size || s.height <= 0 {
        avail_h
    } else {
        s.height.min(avail_h)
    };
    w = w.clamp(2, u16::MAX as i32);
    h = h.clamp(2, u16::MAX as i32);
    if s.output_mode == 1 {
        w &= !1;
        h &= !1;
    }
    (w as u16, h as u16)
}

/// Rebuild the X channel + capture surfaces after a connection-level failure (Xorg restart,
/// VT switch), refreshing the offset/dims state against the live root. The pool is drained
/// first so the encode thread can never be reading a surface as it is destroyed, and its
/// generation is bumped so the encoder rebuilds against the new layout. Every fallible step
/// completes before the swap, and the old channel/surfaces are destroyed best-effort after
/// it (their server is likely gone; the local detaches still run, and the kernel frees
/// each old segment once nothing maps it).
#[allow(clippy::too_many_arguments)]
fn try_rebuild_channel(
    conn: &mut RustConnection,
    root: &mut u32,
    rsettings: &RustCaptureSettings,
    cap_x: &mut i16,
    cap_y: &mut i16,
    cap_w: &mut u16,
    cap_h: &mut u16,
    root_w: &mut u16,
    root_h: &mut u16,
    pool_n: usize,
    pool: &FramePool,
    surfaces: &mut Vec<ShmSurface>,
    stop: &AtomicBool,
) -> Result<(), String> {
    let (new_conn, new_screen) =
        x11rb::connect(None).map_err(|e| format!("X11 reconnect failed: {e}"))?;
    require_32bpp(&new_conn, new_screen)?;
    new_conn
        .shm_query_version()
        .map_err(|e| format!("shm_query_version: {e}"))?
        .reply()
        .map_err(|e| format!("XShm unavailable on reconnect: {e}"))?;
    new_conn
        .xfixes_query_version(5, 0)
        .map_err(|e| format!("xfixes_query_version: {e}"))?
        .reply()
        .map_err(|e| format!("XFixes unavailable on reconnect: {e}"))?;
    let new_root = new_conn.setup().roots[new_screen].root;
    let geo = new_conn
        .get_geometry(new_root)
        .map_err(|e| format!("get_geometry: {e}"))?
        .reply()
        .map_err(|e| format!("get_geometry reply on reconnect: {e}"))?;
    if !pool.drain_for_resize(pool_n, stop) {
        return Err("pool drain interrupted by stop during reconnect".to_string());
    }
    let (fw, fh) = resolve_dims(geo.width, geo.height, rsettings);
    let mut fresh: Vec<ShmSurface> = Vec::with_capacity(pool_n);
    for _ in 0..pool_n {
        match ShmSurface::create(&new_conn, fw, fh) {
            Ok(s) => fresh.push(s),
            Err(e) => {
                for s in fresh.iter_mut() {
                    s.destroy(&new_conn);
                }
                return Err(e);
            }
        }
    }
    for s in surfaces.iter_mut() {
        s.destroy(conn);
    }
    *conn = new_conn;
    *root = new_root;
    *cap_x = clamp_offset(rsettings.capture_x, geo.width);
    *cap_y = clamp_offset(rsettings.capture_y, geo.height);
    *cap_w = fw;
    *cap_h = fh;
    *root_w = geo.width;
    *root_h = geo.height;
    *surfaces = fresh;
    pool.bump_generation();
    Ok(())
}

/// Grab one frame of the capture region into `surface` with a single XShm round-trip.
///
/// One `shm_get_image` request is atomic — the server never interleaves another client
/// inside it — so the grab is a coherent snapshot of whatever the screen held at that
/// instant. Every grabbed frame is published; whether any of it changed is decided
/// downstream by the encoder's own per-stripe damage detection.
fn grab_frame(
    conn: &RustConnection,
    root: u32,
    surface: &ShmSurface,
    cap_x: i16,
    cap_y: i16,
    cap_w: u16,
    cap_h: u16,
) -> Result<(), String> {
    conn.shm_get_image(
        root,
        cap_x,
        cap_y,
        cap_w,
        cap_h,
        !0u32,
        ImageFormat::Z_PIXMAP.into(),
        surface.shmseg,
        0,
    )
    .map_err(|e| format!("shm_get_image: {e}"))?
    .reply()
    .map_err(|e| format!("shm_get_image reply: {e}"))?;
    Ok(())
}

/// Frame-space top-left of the cursor image, given the XFixes hotspot position.
///
/// XFixes reports the cursor position at its HOTSPOT; X draws the image with its top-left at
/// `(pos - hot)`, and the capture origin `(cap_x, cap_y)` is subtracted to move it into frame space.
/// The result may go negative near the frame edges, which `overlay_cursor` clips per pixel to match
/// the server's own edge clipping.
#[inline]
pub(crate) fn cursor_image_origin(x: i16, y: i16, xhot: u16, yhot: u16, cap_x: i32, cap_y: i32) -> (i32, i32) {
    (x as i32 - xhot as i32 - cap_x, y as i32 - yhot as i32 - cap_y)
}

/// Composite the XFixes cursor (ARGB `u32` per pixel, premultiplied by the XFixes
/// format definition) onto the BGRA frame with its top-left at `(img_x, img_y)`,
/// blending each pixel through `blend_pixel_premultiplied` with per-pixel bounds
/// clipping so an image straddling a frame edge writes only its in-frame portion.
#[allow(clippy::too_many_arguments)]
pub(crate) fn overlay_cursor(
    frame: &mut [u8],
    stride: usize,
    frame_w: i32,
    frame_h: i32,
    cur_w: i32,
    cur_h: i32,
    pixels: &[u32],
    img_x: i32,
    img_y: i32,
) {
    for y in 0..cur_h {
        let ty = img_y + y;
        if ty < 0 || ty >= frame_h {
            continue;
        }
        for x in 0..cur_w {
            let tx = img_x + x;
            if tx < 0 || tx >= frame_w {
                continue;
            }
            let px = pixels[(y * cur_w + x) as usize];
            let a = ((px >> 24) & 0xFF) as u8;
            let r = ((px >> 16) & 0xFF) as u8;
            let g = ((px >> 8) & 0xFF) as u8;
            let b = (px & 0xFF) as u8;
            let off = ty as usize * stride + tx as usize * 4;
            blend_pixel_premultiplied(&mut frame[off..off + 4], r, g, b, a);
        }
    }
}

/// A captured raw BGRA frame held in a pooled shm surface, ready to encode.
///
/// It carries the surface pointer plus geometry so the encode thread reads the pixels directly (no
/// copy) and can rebuild its pipeline when the capture size changed (auto-adjust). It also carries
/// the pool's surface `generation`, so a rebuild is still triggered when the surfaces were recreated
/// at the SAME size (a resize flap) and the new segments happen to reuse the old virtual addresses.
struct RawFrame {
    idx: usize,
    ptr: *mut u8,
    len: usize,
    width: u16,
    height: u16,
    stride: usize,
    generation: u64,
}
/// `RawFrame` is `Send`: its raw pointer addresses a pooled shm surface that the pool
/// guarantees is not reused until the encode thread recycles this frame, so the handle is safe to
/// move across the capture -> encode thread boundary.
unsafe impl Send for RawFrame {}

/// Mutex-guarded interior of `FramePool`: the free-surface index list and the single
/// capture -> encode handoff slot, under one lock so moving a surface between them — acquire,
/// publish, take, recycle — is always a single atomic step.
struct PoolInner {
    free: Vec<usize>,
    slot: Option<RawFrame>,
}

/// Demand-driven capture -> encode handoff: a bounded single-slot channel over a fixed set of
/// pooled shm surfaces.
///
/// The capture thread writes into a pooled surface and `publish`es it into the single `slot`; the
/// encode thread `take`s it. Capture stays at most one frame ahead of encode because `acquire` and
/// `publish` BLOCK (bounded) until the encoder frees a surface / drains the slot, throttling capture
/// to the encode rate. Since X11 capture is pull-based (no backlog), throttling — rather than
/// capturing-then-dropping — means a full-resolution shm round-trip is never spent on a frame that
/// would be discarded, while the next capture still overlaps the current encode (the throughput win).
/// The encoder therefore only ever sees a contiguous frame stream, keeping the H.264 reference chain
/// valid.
///
/// `generation` is bumped by the capture thread each time the backing shm surfaces are destroyed and
/// recreated. Published frames carry it so the encode thread rebuilds its pipeline — dropping encoder
/// state keyed to surface base pointers (e.g. NVENC's pinned-host registrations) — even when a resize
/// flap lands back on the old dimensions and the recreated segments reuse the old virtual addresses.
struct FramePool {
    inner: Mutex<PoolInner>,
    cv: Condvar,
    stop: AtomicBool,
    generation: AtomicU64,
}

impl FramePool {
    /// Create a pool with `n` free surfaces, an empty handoff slot, and generation 0.
    fn new(n: usize) -> Self {
        Self {
            inner: Mutex::new(PoolInner { free: (0..n).collect(), slot: None }),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
            generation: AtomicU64::new(0),
        }
    }

    /// Capture: record that the surfaces were recreated by advancing the generation. Only
    /// called after `drain_for_resize` succeeded, so no frame from the previous generation is still
    /// in flight.
    fn bump_generation(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Read the current surface generation, stamped onto each published frame.
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Capture: claim a free surface to write the next frame into, or `None` on stop.
    ///
    /// Blocks (bounded, re-checking `stop` every 20ms) until a surface is free, which throttles
    /// capture to the encode rate so a full-resolution capture is never spent on a frame that would
    /// be dropped. The bounded re-check means a stop that races the wait cannot hang the thread.
    fn acquire(&self, stop: &AtomicBool) -> Option<usize> {
        let mut g = self.inner.lock().unwrap();
        loop {
            if let Some(idx) = g.free.pop() {
                return Some(idx);
            }
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let (gg, _) = self.cv.wait_timeout(g, Duration::from_millis(20)).unwrap();
            g = gg;
        }
    }

    /// Capture: publish the just-captured frame into the single slot, or `false` (frame
    /// discarded) on stop.
    ///
    /// Blocks (bounded, re-checking `stop` every 20ms) until the encode thread has taken the previous
    /// frame, so capture stays at most one frame ahead and never drops under normal flow.
    fn publish(&self, frame: RawFrame, stop: &AtomicBool) -> bool {
        let mut g = self.inner.lock().unwrap();
        loop {
            if g.slot.is_none() {
                g.slot = Some(frame);
                drop(g);
                self.cv.notify_all();
                return true;
            }
            if stop.load(Ordering::Relaxed) {
                return false;
            }
            let (gg, _) = self.cv.wait_timeout(g, Duration::from_millis(20)).unwrap();
            g = gg;
        }
    }

    /// Encode: block until a frame is available (`Some`) or stop is signalled (`None`).
    ///
    /// The wait is bounded (re-checking `stop` every 20ms) as defense-in-depth against a lost wakeup,
    /// so a stop that races the park can never leave the encode thread blocked forever.
    fn take(&self) -> Option<RawFrame> {
        let mut g = self.inner.lock().unwrap();
        loop {
            if let Some(f) = g.slot.take() {
                return Some(f);
            }
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            let (gg, _) = self.cv.wait_timeout(g, Duration::from_millis(20)).unwrap();
            g = gg;
        }
    }

    /// Encode: return a surface to the free list after it has been encoded, waking any waiter
    /// parked in `acquire` / `publish` / `drain_for_resize`.
    fn recycle(&self, idx: usize) {
        self.inner.lock().unwrap().free.push(idx);
        self.cv.notify_all();
    }

    /// Capture: before recreating surfaces (auto-adjust / region resize), reclaim the pending
    /// slot and wait until every surface is back in the free list, so no surface is destroyed while
    /// the encode thread is still reading it.
    ///
    /// Reclaiming the slot returns any un-taken frame to the free list; the loop then waits until
    /// `free.len() == n`, i.e. the encode thread has finished any in-flight frame. The wait is bounded
    /// and also breaks on stop — pool shutdown OR the external `stop` — so a panicked or dead encode
    /// thread that never recycles cannot wedge the capture thread here forever; a requested stop
    /// unblocks it. Returns `true` if it fully drained (safe to recreate surfaces), or `false` if it
    /// aborted on stop — in which case the caller tears down, which joins the encode thread before
    /// destroying surfaces, so the resize-safety guarantee still holds.
    fn drain_for_resize(&self, n: usize, stop: &AtomicBool) -> bool {
        let mut g = self.inner.lock().unwrap();
        if let Some(old) = g.slot.take() {
            g.free.push(old.idx);
        }
        while g.free.len() < n {
            if self.stop.load(Ordering::Acquire) || stop.load(Ordering::Relaxed) {
                return false;
            }
            let (gg, _) = self.cv.wait_timeout(g, Duration::from_millis(20)).unwrap();
            g = gg;
        }
        true
    }

    /// Signal every waiter to stop and wake them, ending the encode thread's `take` loop.
    ///
    /// The inner mutex is acquired BEFORE storing `stop` and notifying, which closes the lost-wakeup
    /// window: with the lock held, `take` is either still before its `stop` check or already parked on
    /// the condvar (and will receive the notify), never in the gap between the two. The notify is
    /// issued after the guard is dropped so the woken thread does not immediately re-block on the lock
    /// this call holds.
    fn shutdown(&self) {
        let g = self.inner.lock().unwrap();
        self.stop.store(true, Ordering::Release);
        drop(g);
        self.cv.notify_all();
    }
}

/// Encode thread body: consume captured frames from the pool, keep the [`pipeline::X11Pipeline`] in
/// step with the capture size and cross-thread controls, encode, recycle, and deliver.
///
/// The loop runs until `FramePool::take` returns `None` (stop). For each frame:
///
/// 1. **(Re)build the pipeline** when it is missing, the frame size changed, or the surface
///    generation changed. A generation change alone forces the rebuild path because recreated shm
///    segments often reuse the old virtual base addresses, so encoder state keyed to base pointers
///    (NVENC's pinned-host cache) must be dropped even at identical dimensions. An in-place
///    [`X11Pipeline::reshape`] is preferred — NVENC reconfigures its live session and the striped
///    software path just re-derives stripe state — so a resize never stalls on a full encoder
///    re-init; encoders that cannot follow in place (VAAPI/OpenH264) report `false` and the pipeline
///    is rebuilt. On a rebuild the old pipeline (with its GPU session/surfaces) is dropped BEFORE the
///    new one is built, so an auto-adjust resize never holds two full encoder allocations at once
///    (transient 2x GPU memory). The rebuilt settings pull the CURRENT bitrate / VBV / fps from the
///    controls atomics, carrying live rate changes forward instead of reverting to the capture-start
///    values.
/// 2. **Apply cross-thread controls** here, on the thread that owns the pipeline: a pending
///    `force_idr` requests a keyframe; a `rate_dirty` swap (Acquire, pairing with the setters' Release
///    stores so the bitrate / VBV / fps payload is never seen half-applied) pushes a live rate change;
///    a `tunables_dirty` swap applies per-frame tunables into both the local settings copy and the
///    pipeline.
/// 3. **Encode and hand off**: read the pooled surface directly through its raw pointer — sound
///    because the pool guarantees the surface is not reused until it is recycled — run
///    [`X11Pipeline::process`], recycle the surface BEFORE delivering so a slow consumer never holds a
///    capture surface, then deliver any encoded stripes through `on_frame`.
///
/// The optional Unix-socket recording sink (parity with the Wayland path) is bound ONCE here and
/// owned outside the pipeline, so pipeline rebuilds on resize keep the socket listener and any
/// attached recorders alive. It can only carry a single full-frame H.264 stream, so configurations
/// that cannot produce one (JPEG output, or a striped CPU encoder) are warned about up front.
fn encode_loop<F>(pool: &FramePool, controls: &Controls, settings: &RustCaptureSettings, on_frame: &mut F)
where
    F: FnMut(Vec<EncodedStripe>),
{
    let mut psettings = settings.clone();
    let recording_sink = RecordingSink::try_bind(&settings.recording_socket);
    if recording_sink.is_some() {
        if settings.output_mode == 0 {
            eprintln!("[recording_sink] recording_socket set but output_mode is JPEG; no recordable H.264 stream.");
        } else if settings.use_cpu && !settings.use_openh264 && !settings.video_fullframe {
            eprintln!("[recording_sink] recording_socket set but the CPU encoder is striped; set video_fullframe=true for a recordable stream.");
        }
    }
    let mut pipeline: Option<X11Pipeline> = None;
    let (mut pw, mut ph) = (0i32, 0i32);
    let mut pgen = 0u64;
    let mut last_log_time = Instant::now();
    let mut frame_count: u64 = 0;
    let mut stripe_count: u64 = 0;

    while let Some(frame) = pool.take() {
        let (fw, fh) = (frame.width as i32, frame.height as i32);
        if pipeline.is_none() || pw != fw || ph != fh || pgen != frame.generation {
            let size_changed = pw != fw || ph != fh;
            psettings.width = fw;
            psettings.height = fh;
            psettings.video_bitrate_kbps = controls.bitrate_kbps.load(Ordering::Relaxed);
            psettings.video_vbv_multiplier =
                controls.vbv_mult_milli.load(Ordering::Relaxed) as f64 / 1000.0;
            psettings.target_fps =
                (controls.fps_milli.load(Ordering::Relaxed).max(1) as f64) / 1000.0;
            let reshaped = pipeline
                .as_mut()
                .is_some_and(|pl| pl.reshape(&psettings, size_changed));
            if !reshaped {
                drop(pipeline.take());
                pipeline = Some(X11Pipeline::new(psettings.clone()));
                if let Some(pl) = &pipeline {
                    let enc_name = pl.encoder_name();
                    let mut log_msg = format!(
                        "[x11] Stream settings active -> Res: {}x{} | FPS: {:.1} | Encoder: {}",
                        psettings.width, psettings.height, psettings.target_fps, enc_name
                    );
                    if psettings.output_mode == 0 {
                        log_msg.push_str(&format!(" | Mode: JPEG | Quality: {}", psettings.jpeg_quality));
                    } else {
                        if psettings.video_cbr_mode {
                            log_msg.push_str(&format!(" | Mode: H264 CBR {}", psettings.video_bitrate_kbps));
                        } else {
                            log_msg.push_str(&format!(" | Mode: H264 | CRF: {}", psettings.video_crf));
                            if psettings.video_bitrate_kbps > 0 {
                                log_msg.push_str(&format!(" | VBV: {} kbps", psettings.video_bitrate_kbps));
                            }
                        }
                        if psettings.video_fullcolor {
                            log_msg.push_str(" | Colorspace: I444 (Full Range)");
                        } else {
                            log_msg.push_str(" | Colorspace: I420 (Limited Range)");
                        }
                    }
                    log_msg.push_str(&format!(
                        " | Damage Thresh: {}f | Damage Dur: {}f",
                        psettings.damage_block_threshold, psettings.damage_block_duration
                    ));
                    println!("{}", log_msg);
                }
            }
            pw = fw;
            ph = fh;
            pgen = frame.generation;
        }
        let pl = pipeline.as_mut().unwrap();

        // A recorder connecting to the socket sink needs a fresh decode entry point; it is
        // folded into the same standard request-IDR path as a client-driven force_idr.
        let sink_idr = recording_sink
            .as_ref()
            .map(|s| s.should_force_idr())
            .unwrap_or(false);
        if controls.force_idr.swap(false, Ordering::Relaxed) || sink_idr {
            pl.request_idr();
        }
        if controls.rate_dirty.swap(false, Ordering::Acquire) {
            let b = controls.bitrate_kbps.load(Ordering::Relaxed);
            let v = controls.vbv_mult_milli.load(Ordering::Relaxed) as f64 / 1000.0;
            let fps = (controls.fps_milli.load(Ordering::Relaxed).max(1) as f64) / 1000.0;
            pl.update_rate(b, v, fps);
        }
        if controls.tunables_dirty.swap(false, Ordering::Acquire)
            && let Some(t) = controls.tunables.lock().unwrap().take() {
                t.apply_to(&mut psettings);
                pl.update_tunables(&t);
            }

        let buf = unsafe { std::slice::from_raw_parts(frame.ptr, frame.len) };
        let stripes = pl.process(buf, frame.stride);
        pool.recycle(frame.idx);
        if !stripes.is_empty() {
            frame_count += 1;
            stripe_count += stripes.len() as u64;
            // IDR arming is consumed inside X11Pipeline::process; this tap only fans out.
            if let Some(ref socket) = recording_sink {
                socket.write_frame(&stripes, psettings.height);
            }
            on_frame(stripes);
        }

        let now = Instant::now();
        let elapsed = now.duration_since(last_log_time).as_secs_f64();
        if elapsed >= 1.0 {
            if settings.debug_logging {
                let actual_fps = frame_count as f64 / elapsed;
                let stripes_per_sec = stripe_count as f64 / elapsed;
                println!(
                    "[x11] Res: {}x{} Encoder: {} EncFPS: {:.2} EncStripes/s: {:.2}",
                    psettings.width, psettings.height, pl.encoder_name(), actual_fps, stripes_per_sec
                );
            }
            frame_count = 0;
            stripe_count = 0;
            last_log_time = now;
        }
    }
}

/// Run the X11 capture pipeline until `stop` is set, splitting capture and encode across two
/// threads that overlap for throughput.
///
/// This (the caller's) thread performs setup and then the capture loop; a spawned encode thread runs
/// `encode_loop` and invokes `on_frame(stripes)` once per encoded frame. The split lets capture and
/// encode overlap, and because frames are throttled/dropped as RAW frames before encode, the delivered
/// H.264 stays a valid contiguous reference chain.
///
/// 1. **Setup**: connect to X (a private connection) and require a 32-bpp BGRA root — the Z-pixmap
///    byte depth must be 4, which modern servers use for depth 24/32 — then negotiate XShm and
///    XFixes. XFixes is always negotiated (one round-trip) so the cursor overlay can be toggled on
///    live even when capture started without it. Initial dimensions and origin are resolved from the
///    settings and the live root geometry, and a `FramePool` of `POOL_N` = 3 shm surfaces is
///    allocated — the working set (one in-capture, one in-slot, one in-encode) that a one-frame-ahead
///    demand-driven capture needs, which also keeps the memory cost (`3 * W*H*4`, significant at 4K)
///    bounded.
/// 2. **Encode thread**: spawned with a raised scheduling priority; it reports its thread id back
///    through `encode_tid_tx` so the caller can detect a re-entrant stop issued from inside the
///    delivery callback.
/// 3. **Capture loop** (until `stop`): fps is re-read each iteration for live pacing — sleep to the
///    next frame deadline, or `yield_now` when already behind instead of busy-spinning.
///    - **Live region change** (`region_dirty`): re-target the grab origin immediately (an x/y pan
///      needs no surface work); a size change reuses the drain/recreate path below.
///    - **Auto-adjust**: on a root geometry change, drain in-flight frames then recreate the surfaces
///      and bump the generation (which the encode thread turns into a reshape or rebuild). The geometry
///      is re-resolved AFTER the drain, because a slow drain can outlast another geometry change (a
///      fast flap) and recreating at a stale size would make the next `shm_get_image` exceed the root;
///      a fully-reverted flap keeps its surfaces as-is. A stop that races the drain breaks out to
///      teardown.
///    - **Grab**: acquire a pooled surface (blocks until the encoder frees one), then `shm_get_image`
///      the region into it synchronously (`reply()` waits).
///    - **Overlays**: composite the XFixes cursor (drawn at its hotspot-offset origin; live-toggleable)
///      and the watermark onto the BGRA pixels on the CPU.
///    - **Publish**: hand the finished frame to the encode thread (blocks until the slot is free;
///      never drops). A stop observed while waiting in acquire/publish exits the loop.
/// 4. **Teardown**: stop and JOIN the encode thread BEFORE destroying the shm surfaces it may still be
///    reading, preserving the resize-safety guarantee.
///
/// Blocking; intended to run on a dedicated thread. The X connection and shm surfaces live on this
/// thread and the encoder lives on the encode thread — nothing X-related crosses the boundary.
pub fn run_capture<F>(
    settings: RustCaptureSettings,
    controls: Arc<Controls>,
    encode_tid_tx: Sender<thread::ThreadId>,
    on_frame: F,
) -> Result<(), String>
where
    F: FnMut(Vec<EncodedStripe>) + Send + 'static,
{
    let (mut conn, screen_num) =
        x11rb::connect(None).map_err(|e| format!("X11 connect failed: {e}"))?;
    let mut root = conn.setup().roots[screen_num].root;
    require_32bpp(&conn, screen_num)?;

    conn.shm_query_version()
        .map_err(|e| format!("shm_query_version: {e}"))?
        .reply()
        .map_err(|e| format!("XShm unavailable: {e}"))?;
    conn.xfixes_query_version(5, 0)
        .map_err(|e| format!("xfixes_query_version: {e}"))?
        .reply()
        .map_err(|e| format!("XFixes unavailable: {e}"))?;

    let geo = conn
        .get_geometry(root)
        .map_err(|e| format!("get_geometry: {e}"))?
        .reply()
        .map_err(|e| format!("get_geometry reply: {e}"))?;

    let mut rsettings = settings.clone();
    let (mut cap_w, mut cap_h) = resolve_dims(geo.width, geo.height, &rsettings);
    let mut cap_x = clamp_offset(rsettings.capture_x, geo.width);
    let mut cap_y = clamp_offset(rsettings.capture_y, geo.height);
    // Last root size seen. A live pan has to clamp against something even when the geometry
    // round-trip fails, or the requested origin would be dropped instead of applied.
    let (mut root_w, mut root_h) = (geo.width, geo.height);

    const POOL_N: usize = 3;
    let mut surfaces: Vec<ShmSurface> = Vec::with_capacity(POOL_N);
    for _ in 0..POOL_N {
        surfaces.push(ShmSurface::create(&conn, cap_w, cap_h)?);
    }
    // Consecutive channel rebuilds without one good grab in between: geometry
    // errors can burn out the grab budget repeatedly under a dead region, so
    // recovery only gets a bounded number of channel rebuilds.
    let mut reconnect_streak = 0u32;
    let pool = Arc::new(FramePool::new(POOL_N));

    let mut watermark = crate::encoders::overlay::OverlayState::default();
    if !settings.watermark_path.is_empty() {
        watermark.load_watermark(&settings.watermark_path, 1.0);
    }

    let enc_pool = pool.clone();
    let enc_controls = controls.clone();
    let enc_settings = settings.clone();
    let encode_panicked = Arc::new(AtomicBool::new(false));
    let guard_panicked = encode_panicked.clone();
    let encode_thread = thread::spawn(move || {
        crate::boost_thread_priority(-10);
        let _ = encode_tid_tx.send(thread::current().id());
        let mut on_frame = on_frame;
        // An encoder panic must never wedge the capture: without the pool shutdown
        // the capture thread would block in publish forever while is_capturing still
        // reports true on this thread's handle. Shutting the pool unblocks it through
        // the same path a clean stop takes, so the panic is also recorded — otherwise
        // the capture would return Ok and Python could not tell the two apart.
        let guard_pool = enc_pool.clone();
        let guard_controls = enc_controls.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            encode_loop(&enc_pool, &enc_controls, &enc_settings, &mut on_frame);
        }));
        if result.is_err() {
            eprintln!("[pixelflux x11] encode thread panicked; shutting the pool to fail the capture");
            guard_panicked.store(true, Ordering::Release);
            guard_pool.shutdown();
            guard_controls.stop.store(true, Ordering::Relaxed);
        }
    });

    let mut next_frame = Instant::now();
    // Frames between root-geometry polls (auto-adjust); ~0.5s at 60fps.
    const GEOMETRY_POLL_FRAMES: i32 = 30;
    let mut geometry_check = 0i32;
    let mut grab_failures = 0u32;

    let result = (|| -> Result<(), String> {
        while !controls.stop.load(Ordering::Relaxed) {
            let fps = (controls.fps_milli.load(Ordering::Relaxed).max(1) as f64) / 1000.0;
            let frame_dur = Duration::from_secs_f64(1.0 / fps.max(1.0));
            let now = Instant::now();
            if now < next_frame {
                std::thread::sleep(next_frame - now);
            } else {
                std::thread::yield_now();
            }
            next_frame += frame_dur;
            let now = Instant::now();
            if next_frame < now {
                next_frame = now;
            }
            if controls.stop.load(Ordering::Relaxed) {
                break;
            }

            if controls.region_dirty.swap(false, Ordering::Acquire) {
                let (nx, ny, nw, nh) = *controls.region.lock().unwrap();
                rsettings.capture_x = nx;
                rsettings.capture_y = ny;
                rsettings.width = nw;
                rsettings.height = nh;
                // An explicit live size pins the region: auto-adjust (from the start
                // settings) would glue resolve_dims back to the ROOT size, undoing the
                // re-target on the next geometry poll. w/h <= 0 keep following the root.
                rsettings.auto_adjust_screen_capture_size = nw <= 0 || nh <= 0;
                cap_x = clamp_offset(nx, root_w);
                cap_y = clamp_offset(ny, root_h);
                if let Some(g) = conn.get_geometry(root).ok().and_then(|c| c.reply().ok()) {
                    root_w = g.width;
                    root_h = g.height;
                    cap_x = clamp_offset(nx, root_w);
                    cap_y = clamp_offset(ny, root_h);
                    let (fw, fh) = resolve_dims(g.width, g.height, &rsettings);
                    if fw != cap_w || fh != cap_h {
                        if !pool.drain_for_resize(POOL_N, &controls.stop) {
                            break;
                        }
                        for s in surfaces.iter_mut() {
                            s.destroy(&conn);
                        }
                        surfaces.clear();
                        cap_w = fw;
                        cap_h = fh;
                        for _ in 0..POOL_N {
                            surfaces.push(ShmSurface::create(&conn, cap_w, cap_h)?);
                        }
                        pool.bump_generation();
                    }
                }
            }

            // Root geometry is polled on a cadence, not per frame: the reply costs a
            // round-trip that would otherwise precede every grab, and external size
            // changes are rare (live resizes arrive through region_dirty instead).
            // Grab-failure recovery enters regardless of auto_adjust: a fixed-size
            // region has no other re-clamp path after an external root shrink, and
            // without one the capture dies at the failure cap instead of recovering.
            geometry_check = geometry_check.saturating_sub(1);
            if (settings.auto_adjust_screen_capture_size || grab_failures > 0) && geometry_check <= 0 {
                geometry_check = GEOMETRY_POLL_FRAMES;
                if let Some(g) = conn.get_geometry(root).ok().and_then(|c| c.reply().ok()) {
                    // A root shrink can strand even a previously-valid offset past
                    // the new edge; re-clamp it against the live root.
                    root_w = g.width;
                    root_h = g.height;
                    cap_x = clamp_offset(rsettings.capture_x, root_w);
                    cap_y = clamp_offset(rsettings.capture_y, root_h);
                    let (nw, nh) = resolve_dims(g.width, g.height, &rsettings);
                    if nw != cap_w || nh != cap_h {
                        if !pool.drain_for_resize(POOL_N, &controls.stop) {
                            break;
                        }
                        let (fw, fh) = conn
                            .get_geometry(root)
                            .ok()
                            .and_then(|c| c.reply().ok())
                            .map(|g| resolve_dims(g.width, g.height, &rsettings))
                            .unwrap_or((nw, nh));
                        if fw != cap_w || fh != cap_h {
                            for s in surfaces.iter_mut() {
                                s.destroy(&conn);
                            }
                            surfaces.clear();
                            cap_w = fw;
                            cap_h = fh;
                            for _ in 0..POOL_N {
                                surfaces.push(ShmSurface::create(&conn, cap_w, cap_h)?);
                            }
                            pool.bump_generation();
                        }
                    }
                }
            }

            let idx = match pool.acquire(&controls.stop) {
                Some(i) => i,
                None => break,
            };
            let grab_result = {
                let surface = &mut surfaces[idx];
                grab_frame(&conn, root, surface, cap_x, cap_y, cap_w, cap_h)
            };
            if let Err(e) = grab_result {
                // Most likely an external root resize between geometry polls made
                // the grab run past the root edge: return the surface, re-poll
                // geometry immediately, and only give up if it never recovers.
                pool.recycle(idx);
                grab_failures += 1;
                geometry_check = 0;
                if grab_failures > 120 {
                    // A burn-out usually means the X connection itself is gone
                    // (Xorg restart, VT switch): every request fails until the
                    // server is back. Rebuild the channel a few times before
                    // declaring the capture dead — a recovered server resumes the
                    // stream where a dead thread would leave a black screen until
                    // the Python watchdog restarts everything.
                    let mut recovered = false;
                    for _attempt in 0..5 {
                        if controls.stop.load(Ordering::Relaxed) {
                            break;
                        }
                        std::thread::sleep(Duration::from_secs(1));
                        match try_rebuild_channel(
                            &mut conn, &mut root, &rsettings,
                            &mut cap_x, &mut cap_y, &mut cap_w, &mut cap_h,
                            &mut root_w, &mut root_h,
                            POOL_N, &pool, &mut surfaces, &controls.stop,
                        ) {
                            Ok(()) => { recovered = true; break; }
                            Err(re) => {
                                eprintln!("[pixelflux x11] reconnect attempt failed: {re}");
                            }
                        }
                    }
                    if !recovered {
                        return Err(format!("capture could not recover: {e}"));
                    }
                    grab_failures = 0;
                    geometry_check = 0;
                    reconnect_streak += 1;
                    if reconnect_streak >= 4 {
                        return Err("capture recovered its channel but the region never grabbed again".to_string());
                    }
                }
                continue;
            }
            grab_failures = 0;
            reconnect_streak = 0;
            let surface = &mut surfaces[idx];

            let frame_w = cap_w as i32;
            let frame_h = cap_h as i32;
            let stride = surface.stride;
            let buf = surface.as_mut_slice();

            if controls.capture_cursor.load(Ordering::Relaxed)
                && let Some(c) = conn
                    .xfixes_get_cursor_image()
                    .ok()
                    .and_then(|c| c.reply().ok())
                    && c.width > 0 && c.height > 0 {
                        // Translate from root coordinates using the LIVE grab origin
                        // (cap_x/cap_y follow region_dirty pans and clamp negatives
                        // exactly like the grab itself), never the immutable startup
                        // settings — those go stale on the first live region move.
                        let (img_x, img_y) = cursor_image_origin(
                            c.x,
                            c.y,
                            c.xhot,
                            c.yhot,
                            cap_x as i32,
                            cap_y as i32,
                        );
                        overlay_cursor(
                            buf,
                            stride,
                            frame_w,
                            frame_h,
                            c.width as i32,
                            c.height as i32,
                            &c.cursor_image,
                            img_x,
                            img_y,
                        );
                    }

            if watermark.is_active() {
                watermark.update_position(frame_w, frame_h, settings.watermark_location_enum);
                watermark.blend_bgra(buf, stride, frame_w, frame_h);
            }

            let published = pool.publish(
                RawFrame {
                    idx,
                    ptr: surface.addr,
                    len: surface.size,
                    width: cap_w,
                    height: cap_h,
                    stride,
                    generation: pool.generation(),
                },
                &controls.stop,
            );
            if !published {
                break;
            }
        }
        Ok(())
    })();

    pool.shutdown();
    let _ = encode_thread.join();
    for s in surfaces.iter_mut() {
        s.destroy(&conn);
    }
    match result {
        // A real capture error is the more specific diagnosis, so it outranks the panic flag.
        Err(e) => Err(e),
        Ok(()) if encode_panicked.load(Ordering::Acquire) => Err("encode thread panicked".into()),
        Ok(()) => Ok(()),
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    fn dummy(idx: usize) -> RawFrame {
        RawFrame {
            idx,
            ptr: std::ptr::null_mut(),
            len: 0,
            width: 0,
            height: 0,
            stride: 0,
            generation: 0,
        }
    }

    /// A full acquire -> publish -> take -> recycle round-trip returns the same surface, and
    /// afterwards all three pool surfaces are acquirable again and distinct.
    #[test]
    fn roundtrip_then_recycle_returns_all_surfaces() {
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        let a = p.acquire(&stop).unwrap();
        assert!(p.publish(dummy(a), &stop));
        let f = p.take().unwrap();
        assert_eq!(f.idx, a);
        p.recycle(f.idx);
        let (x, y, z) = (
            p.acquire(&stop).unwrap(),
            p.acquire(&stop).unwrap(),
            p.acquire(&stop).unwrap(),
        );
        assert!(x != y && y != z && x != z);
    }

    /// With every surface held and `stop` set, `acquire` returns `None` after its bounded wait
    /// rather than blocking forever.
    #[test]
    fn acquire_returns_none_when_exhausted_and_stopped() {
        let p = FramePool::new(2);
        let stop = AtomicBool::new(false);
        let _a = p.acquire(&stop).unwrap();
        let _b = p.acquire(&stop).unwrap();
        stop.store(true, Ordering::Relaxed);
        assert!(p.acquire(&stop).is_none());
    }

    /// With the handoff slot already occupied and `stop` set, `publish` returns `false` (frame
    /// discarded) instead of blocking.
    #[test]
    fn publish_returns_false_when_slot_full_and_stopped() {
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        let a = p.acquire(&stop).unwrap();
        assert!(p.publish(dummy(a), &stop));
        let b = p.acquire(&stop).unwrap();
        stop.store(true, Ordering::Relaxed);
        assert!(!p.publish(dummy(b), &stop));
    }

    /// `drain_for_resize` blocks until every held surface has been recycled (`free == n`), then
    /// reports a full drain and leaves the pool whole; a helper thread recycles the three held surfaces
    /// shortly after the drain begins.
    #[test]
    fn drain_for_resize_waits_until_all_free() {
        let p = Arc::new(FramePool::new(3));
        let stop = AtomicBool::new(false);
        let held = [
            p.acquire(&stop).unwrap(),
            p.acquire(&stop).unwrap(),
            p.acquire(&stop).unwrap(),
        ];
        let p2 = p.clone();
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            for idx in held {
                p2.recycle(idx);
            }
        });
        assert!(p.drain_for_resize(3, &stop));
        t.join().unwrap();
        assert!(p.acquire(&stop).is_some());
    }

    /// A surface is held and never recycled (as a dead encode thread would leave it): setting
    /// `stop` unblocks the bounded drain, which reports it did NOT fully drain instead of hanging.
    #[test]
    fn drain_for_resize_aborts_on_stop() {
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        let _held = p.acquire(&stop).unwrap();
        stop.store(true, Ordering::Relaxed);
        assert!(!p.drain_for_resize(3, &stop));
    }

    /// A W1 -> W2 -> W1 resize flap where the encoder never takes the W2 frame:
    /// `drain_for_resize` reclaims the un-taken frame from the slot, the surfaces recreate twice, and
    /// the next taken frame lands back on the ORIGINAL dimensions — so the bumped generation is the
    /// only rebuild signal that invalidates encoder state keyed to the reused surface addresses.
    #[test]
    fn resize_flap_reclaim_changes_generation_at_identical_dims() {
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        let a = p.acquire(&stop).unwrap();
        assert!(p.publish(
            RawFrame { generation: p.generation(), width: 1920, height: 1080, ..dummy(a) },
            &stop
        ));
        let f = p.take().unwrap();
        let (last_gen, last_dims) = (f.generation, (f.width, f.height));
        p.recycle(f.idx);
        assert!(p.drain_for_resize(3, &stop));
        p.bump_generation();
        let b = p.acquire(&stop).unwrap();
        assert!(p.publish(
            RawFrame { generation: p.generation(), width: 2560, height: 1600, ..dummy(b) },
            &stop
        ));
        assert!(p.drain_for_resize(3, &stop));
        p.bump_generation();
        let c = p.acquire(&stop).unwrap();
        assert!(p.publish(
            RawFrame { generation: p.generation(), width: 1920, height: 1080, ..dummy(c) },
            &stop
        ));
        let g = p.take().unwrap();
        assert_eq!((g.width, g.height), last_dims, "flap lands on identical dimensions");
        assert_eq!(g.generation, 2);
        assert_ne!(g.generation, last_gen, "generation is the only rebuild signal");
    }

    /// Recreating the surfaces bumps the generation, and frames published afterwards carry the
    /// new value — the signal the encode thread uses to trigger a rebuild at identical dimensions.
    #[test]
    fn generation_bumps_on_recreate_and_rides_published_frames() {
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        assert_eq!(p.generation(), 0);
        let a = p.acquire(&stop).unwrap();
        assert!(p.publish(RawFrame { generation: p.generation(), ..dummy(a) }, &stop));
        let f = p.take().unwrap();
        assert_eq!(f.generation, 0);
        p.recycle(f.idx);
        p.bump_generation();
        let b = p.acquire(&stop).unwrap();
        assert!(p.publish(RawFrame { generation: p.generation(), ..dummy(b) }, &stop));
        assert_eq!(p.take().unwrap().generation, 1);
    }
}

#[cfg(test)]
mod cursor_tests {
    use super::*;

    /// `cursor_image_origin` subtracts both the hotspot and the capture-region offset from the
    /// reported pointer position, and goes negative when the hotspot sits near the frame origin (the
    /// result is clipped when drawn).
    #[test]
    fn origin_subtracts_hotspot_and_capture_offset() {
        assert_eq!(cursor_image_origin(100, 80, 4, 6, 0, 0), (96, 74));
        assert_eq!(cursor_image_origin(100, 80, 4, 6, 10, 20), (86, 54));
        assert_eq!(cursor_image_origin(1, 1, 8, 8, 0, 0), (-7, -7));
    }

    /// `overlay_cursor` blits at the hotspot-offset origin, not the raw pointer position: a
    /// 2x2 cursor with hotspot (1,1) at pointer (4,4) lands at (3,3)..(4,4), leaving the un-offset
    /// (hotspot) corner untouched.
    #[test]
    fn overlay_blits_at_hotspot_offset_origin() {
        let stride = 8 * 4;
        let mut frame = vec![0u8; stride * 8];
        let pixels = [0xFFFF_FFFFu32; 4];
        let (ox, oy) = cursor_image_origin(4, 4, 1, 1, 0, 0);
        overlay_cursor(&mut frame, stride, 8, 8, 2, 2, &pixels, ox, oy);
        let px = |x: usize, y: usize| frame[y * stride + x * 4];
        assert_eq!(px(3, 3), 255);
        assert_eq!(px(4, 4), 255);
        assert_eq!(px(2, 2), 0);
        assert_eq!(px(5, 5), 0, "no pixel at the un-offset (hotspot) corner");
    }

    #[test]
    fn offset_clamps_into_root() {
        assert_eq!(clamp_offset(100, 1920), 100);
        assert_eq!(clamp_offset(-50, 1920), 0);
        // Past-end offsets keep one root pixel instead of failing forever.
        assert_eq!(clamp_offset(100000, 1920), 1919);
        // A protocol-max root still maps into i16 X coordinates.
        assert_eq!(clamp_offset(40000, u16::MAX), 32766);
    }

    #[test]
    fn resolve_dims_survives_past_end_offset() {
        let s = RustCaptureSettings {
            capture_x: 100000,
            capture_y: 100000,
            width: 1920,
            height: 1080,
            output_mode: 1,
            ..Default::default()
        };
        let (w, h) = resolve_dims(1920, 1080, &s);
        assert!(w >= 2 && h >= 2, "a past-end offset now degrades to a small grab, not death");
        assert_eq!(w % 2, 0);
        assert_eq!(h % 2, 0);
    }

    /// `overlay_cursor` treats XFixes pixels as premultiplied: half-alpha at half
    /// intensity stays at that intensity over black (the straight-alpha formula
    /// would multiply alpha twice and darken it).
    #[test]
    fn overlay_blends_premultiplied_pixels() {
        let stride = 4 * 4;
        let mut frame = vec![0u8; stride * 4];
        // a=128, g=128 (premultiplied half-intensity), b=64.
        let pixels = [0x8000_8040u32; 4];
        overlay_cursor(&mut frame, stride, 4, 4, 2, 2, &pixels, 0, 0);
        assert_eq!(frame[0], 64, "blue channel: dst = 64 over black");
        assert_eq!(frame[1], 128, "premultiplied half-alpha stays 128, not darkened to 64");
    }

    /// `overlay_cursor` clips a negative origin at the frame edge: with the origin at (-1,-1)
    /// only the in-frame quadrant is written, landing the cursor's bottom-right pixel at (0,0).
    #[test]
    fn overlay_clips_negative_origin_at_frame_edge() {
        let stride = 4 * 4;
        let mut frame = vec![0u8; stride * 4];
        let pixels = [0xFFFF_FFFFu32; 4];
        let (ox, oy) = cursor_image_origin(0, 0, 1, 1, 0, 0);
        overlay_cursor(&mut frame, stride, 4, 4, 2, 2, &pixels, ox, oy);
        assert_eq!(frame[0], 255);
        assert!(frame[4..].iter().all(|&b| b == 0), "no writes outside (0,0)");
    }
}
