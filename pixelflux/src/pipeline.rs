/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Frame-processing policy shared by the two capture backends. It lives in its own module for one
//! reason: the Wayland path (dmabuf, compositor damage) and the X11 path (host-ARGB, stripe-hash
//! damage) capture pixels in completely different ways, but a viewer must never be able to tell
//! which one produced a frame — a paint-over refresh or a recovery keyframe has to behave
//! identically either way. Keeping the decision logic here, source-agnostic, is what guarantees it.

use crate::RustCaptureSettings;
use crate::encoders::software::{
    EncodedStripe, StripeState, encode_cpu, invalidate_reference, stripes_held_still,
};
use crate::encoders::{self, Codec, FrameEncoder, FrameSource};
use smithay::utils::{Physical, Rectangle};
use std::sync::Arc;

/// How much of a frame changed since the frame before it, as far as its capture can tell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Damage {
    /// Nothing changed.
    None,
    /// This fraction of the frame's area changed, in `(0, 1]`.
    Area(f32),
    /// Something changed, and the capture cannot say how much (NvFBC's new-frame report).
    Unknown,
}

/// A change covering at most this fraction of the frame is small: it does not count as motion
/// toward the low-motion cleanup.
const SMALL_AREA: f32 = 1.0 / 16.0;

/// How much of a screen a change of unknown extent counts for toward a key-frame cleanup, from
/// the third frame of a run of changed frames on: sustained motion, where an isolated change
/// (a caret, a clock, a keystroke) counts for nothing.
const UNKNOWN_MASS: f32 = 1.0 / 16.0;
const UNKNOWN_RUN: u32 = 3;

/// The change since the last key-frame cleanup, in screens, past which a region still for
/// `KEY_STILL_TRIGGERS` trigger periods is cleaned up by a key frame.
const KEY_CLEANUP_MASS: f32 = 0.5;

/// The trigger periods a region holds still before its key-frame cleanup.
const KEY_STILL_TRIGGERS: u32 = 4;

/// A region that keeps changing is still due a cleanup once this many trigger periods passed
/// since it first changed, where at most a quarter of its frames carry more than a small change,
/// on an average over as many frames (`StripeState::motion`), so the motion that went before
/// (a scroll) ages out of it while a caret blinks.
const LOW_MOTION_TRIGGERS: u32 = 4;
const LOW_MOTION_SHARE: f32 = 0.25;

/// A constant-rate cleanup that runs through the rate control keeps the frames flowing until the
/// rate control codes the region at the paint-over quantizer or finer in a frame under this share
/// of the frame budget: it has nothing left to refine.
const CONVERGED_SHARE: f64 = 0.25;

/// Trigger periods of such small frames at a quantizer still coarser than the paint-over one after
/// which the rate control counts as stalled: it stopped refining short of the paint-over quality,
/// as NVENC's does at 2 Mbit/s at 1080p.
const STALL_TRIGGERS: u32 = 2;

/// Trigger periods a constant-rate cleanup through the rate control has to converge before it
/// falls back to a held refresh: at 8 Mbit/s NVENC converges on a 1080p screen of text within
/// four, and where the rate is low for the resolution it crawls for seconds.
const FALLBACK_TRIGGERS: u32 = 6;

/// Seconds a constant-rate cleanup through the rate control flows at most.
const CONVERGE_S: f64 = 10.0;

/// Frame budgets the rate control's frames of a still screen exceed for a trigger period when it
/// is pinned at its coarsest quantizer, refining nothing within the rate: NVENC's H.264 codes a
/// still screen of texture at 0.1 Mbit/s at 1080p in 3.3 kB frames, 16 budgets, at quantizer 51,
/// where at 2 Mbit/s single frames after a change reach 4.5 budgets.
const OVERSHOOT_BUDGETS: f64 = 2.0;

/// Bytes a slice that refines nothing comes to at most: a frame of x264's four slices skipping
/// every block takes 61 to 66 at 720p and 1080p with the wire header, a third of the budget at
/// 0.1 Mbit/s, so never a small frame.
pub(crate) const EMPTY_SLICE_BYTES: usize = 32;

/// Seconds of such frames after which a constant-rate cleanup through x264 ends: x264 keeps its
/// quantizer on a screen that does not change, and at 0.1 Mbit/s refines one in frames as far
/// as 2.4 s apart. A session that holds no quantizer and measures nothing ends the same way
/// once its frames at its coarsest have kept the size of the one before as long
/// (`SAME_FRAME_SHARE`).
pub(crate) const EMPTY_S: f64 = 3.0;

/// The share of the frame before it within which a frame of a still screen codes nothing new,
/// and the quality index from which its rate control sits at its coarsest: at 0.1 Mbit/s at
/// 1080p libvpx's VP9 frames are 266 bytes each at index 51, or 220-221 at 46-47, and x265's
/// 224 at 50-51, its strict constant rate padding them with filler, while the picture stays
/// level, so never small against the 208-byte budget. From 1 Mbit/s their rate controls code a
/// still screen at 36 and finer, x265's padded frames again of one size while it refines.
const SAME_FRAME_SHARE: f64 = 0.125;
const SAME_FRAME_QUALITY: u32 = 45;

/// The same for a session that holds no quantizer but says the one it codes at or measures its
/// pictures, which has no refresh to fall back to and ends its cleanup at the paint-over
/// quantizer or once the picture stops improving: at 2 Mbit/s at 1080p x265's rate control
/// refines a screen of text by 0.7 dB a second, still short of that quality after ten. A band
/// sweep runs this long at most too.
const REFINE_S: f64 = 30.0;

/// Seconds the finest quantizer such a session has coded the still screen at stands before its
/// rate control counts as done with the picture, where the session says its quantizer and does
/// not measure: at 0.25 Mbit/s at 1080p x265 takes a step finer every 2 to 8 s for a tenth of
/// a dB a second, and libvpx's VP9 one every 6 s after its first ten, each for all of `REFINE_S`
/// at the full rate, where at 2 Mbit/s x265 takes one every 2 s for its first twenty.
const LEVEL_S: f64 = 4.0;

/// Where the encoder holds a band of the picture (`EncoderQuality::band`), the refresh that
/// falls back from a constant-rate cleanup sweeps the picture a band a frame in raster order,
/// the first `FIRST_BAND` of it and each next sized from the bytes of the last to
/// `BAND_BUDGETS` frame budgets, so the stream keeps its rate: a whole refresh at the
/// paint-over quantizer is 86 budgets of text at 2 Mbit/s at 1080p and queues for 240 ms on a
/// 12 Mbit/s link.
const FIRST_BAND: f64 = 1.0 / 64.0;
const BAND_BUDGETS: f64 = 1.0;

impl Damage {
    /// The damage a set of rectangles on a `width` x `height` frame reports: their summed area,
    /// clipped to the frame, overlaps counted twice.
    pub fn of_rects<'a>(
        rects: impl IntoIterator<Item = &'a smithay::utils::Rectangle<i32, smithay::utils::Physical>>,
        width: i32,
        height: i32,
    ) -> Self {
        let frame = (width.max(1) as f64) * (height.max(1) as f64);
        let mut area = 0f64;
        for r in rects {
            let w = (r.loc.x + r.size.w).min(width) - r.loc.x.max(0);
            let h = (r.loc.y + r.size.h).min(height) - r.loc.y.max(0);
            if w > 0 && h > 0 {
                area += w as f64 * h as f64;
            }
        }
        if area > 0.0 {
            Damage::Area((area / frame).min(1.0) as f32)
        } else {
            Damage::None
        }
    }

    /// Whether anything changed.
    pub fn is_dirty(self) -> bool {
        self != Damage::None
    }

    /// Whether the change is more than a small one, so the frame counts as motion.
    pub(crate) fn is_motion(self) -> bool {
        match self {
            Damage::None => false,
            Damage::Area(f) => f > SMALL_AREA,
            Damage::Unknown => true,
        }
    }
}

/// What the cleanup policy decided for one region's frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cleanup {
    None,
    /// Refresh predicted from the picture the client holds, at the paint-over quantizer.
    Refresh,
    /// Key frame at the paint-over quantizer.
    Key,
}

/// The still-screen cleanup of one region (a whole frame, or a stripe): track how long and how
/// much it has been changing, and say when it is due a cleanup and of which kind. `keys` false
/// (JPEG, whose every stripe is a picture of its own, and a constant-rate session cleaned up
/// through its rate control) never answers `Cleanup::Key`.
///
/// Static is read from the content (`damage`), never from what was sent, so a stream that
/// encodes every frame (Turbo) sees the screen go still exactly like one that sends only changes.
/// The cleanup comes in two steps. A region that changed is refreshed once it has held still for
/// `trigger` frames: a frame predicted from the picture the client holds, which re-codes only
/// what changed. Once the change since the last key-frame cleanup adds up to
/// `KEY_CLEANUP_MASS` of a screen and the region has held still for `KEY_STILL_TRIGGERS` trigger
/// periods, it is cleaned up by a key frame, which leaves nothing of the motion behind and
/// resynchronizes every viewer, but costs a whole picture: so it waits for a screen still long
/// enough that the user is reading it, and a caret or a clock, which change little, never pays
/// for one. A region that keeps changing never holds still, so once `LOW_MOTION_TRIGGERS`
/// trigger periods have passed since it first changed and at most a quarter of its recent frames
/// carried more than a small change (a blinking caret, a ticking clock) it is cleaned up then,
/// by a key frame when the change adds up to one.
///
/// `improves` false (the paint-over is off, or a constant rate already codes the region finer)
/// keeps the bookkeeping and answers `Cleanup::None`, and a cleanup that falls due then restarts
/// the count of frames since the region first changed: the region is clean, so those frames are
/// no low motion, and counted, they would have the first frame of the next large change keyed,
/// the whole picture at the paint-over quantizer the moment a window opens. `now` false (a
/// stripe whose cleanup is staggered to a later frame) answers `Cleanup::None` and leaves the
/// cleanup pending.
pub fn cleanup_due(
    st: &mut StripeState,
    trigger: u32,
    improves: bool,
    now: bool,
    keys: bool,
    damage: Damage,
) -> Cleanup {
    if damage.is_dirty() {
        st.dirty_run = st.dirty_run.saturating_add(1);
        st.no_motion_frame_count = 0;
        st.paint_over_sent = false;
        st.change_mass += match damage {
            Damage::Area(f) => f,
            _ if st.dirty_run >= UNKNOWN_RUN => UNKNOWN_MASS,
            _ => 0.0,
        };
    } else {
        st.dirty_run = 0;
        st.no_motion_frame_count = st.no_motion_frame_count.saturating_add(1);
    }
    if !st.paint_over_sent {
        st.unclean_frames = st.unclean_frames.saturating_add(1);
    }
    st.motion = recent_motion(st.motion, trigger.max(1), damage.is_motion());
    let due = due_cleanup(
        st,
        trigger.max(1),
        keys,
        st.no_motion_frame_count,
        st.unclean_frames,
        st.motion,
    );
    if due == Cleanup::None {
        return Cleanup::None;
    }
    if !improves {
        st.unclean_frames = 0;
        return Cleanup::None;
    }
    if !now {
        return Cleanup::None;
    }
    st.paint_over_sent = true;
    st.unclean_frames = 0;
    if due == Cleanup::Key {
        st.change_mass = 0.0;
    }
    due
}

/// The share of a region's recent frames that carried more than a small change, `motion` taking
/// in one more frame: an average over `LOW_MOTION_TRIGGERS` trigger periods.
fn recent_motion(motion: f32, trigger: u32, moved: bool) -> f32 {
    motion + (moved as u8 as f32 - motion) / (trigger * LOW_MOTION_TRIGGERS) as f32
}

/// The cleanup a region is due with `still` frames held still, `unclean` frames since it first
/// changed after its last cleanup, and `motion` the share of its recent frames in motion.
fn due_cleanup(
    st: &StripeState,
    trigger: u32,
    keys: bool,
    still: u32,
    unclean: u32,
    motion: f32,
) -> Cleanup {
    let low_motion = !st.paint_over_sent
        && unclean >= trigger * LOW_MOTION_TRIGGERS
        && motion <= LOW_MOTION_SHARE;
    if keys
        && st.change_mass >= KEY_CLEANUP_MASS
        && (still >= trigger * KEY_STILL_TRIGGERS || low_motion)
    {
        Cleanup::Key
    } else if !st.paint_over_sent && (still >= trigger || low_motion) {
        Cleanup::Refresh
    } else {
        Cleanup::None
    }
}

/// Whether `cleanup_due` would answer a cleanup for a frame that changed nothing, without moving
/// the bookkeeping: what an idle frame has to know before it skips a region.
pub fn cleanup_pending(st: &StripeState, trigger: u32, enabled: bool, keys: bool) -> bool {
    let unclean = st.unclean_frames.saturating_add(!st.paint_over_sent as u32);
    let motion = recent_motion(st.motion, trigger.max(1), false);
    enabled
        && due_cleanup(
            st,
            trigger.max(1),
            keys,
            st.no_motion_frame_count.saturating_add(1),
            unclean,
            motion,
        ) != Cleanup::None
}

/// What an encoder says about the frames it codes, which the cleanup reads: the quality index
/// its rate control last coded a frame at, where it reports one (`FrameEncoder::last_quality`),
/// the bytes of that frame, where it reports them too (`FrameEncoder::last_size`), whether it
/// holds a frame at a quantizer it is asked for (`FrameEncoder::holds_quantizer`), whether a
/// change of its constant quality re-opens it on a key frame (`FrameEncoder::reopens_on_quality`),
/// whether its constant-rate cleanup ends on a key frame (`FrameEncoder::cleans_up_with_key`),
/// where it holds a band of a frame at that quantizer (`FrameEncoder::band_size`), the
/// bytes of its last held frame, and, where it measures a frame's reconstruction against its
/// source when asked (`FrameEncoder::measures`), the last measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncoderQuality {
    pub last: Option<u32>,
    pub bytes: Option<usize>,
    pub holds: bool,
    pub reopens: bool,
    pub keys: bool,
    pub band: Option<usize>,
    pub measures: bool,
    pub psnr: Option<f32>,
}

impl EncoderQuality {
    /// What `encoder` says.
    pub fn of(encoder: &FrameEncoder) -> Self {
        Self {
            last: encoder.last_quality(),
            bytes: encoder.last_size(),
            holds: encoder.holds_quantizer(),
            reopens: encoder.reopens_on_quality(),
            keys: encoder.cleans_up_with_key(),
            band: encoder.band_size(),
            measures: encoder.measures(),
            psnr: encoder.last_psnr(),
        }
    }

    /// Whether a constant-rate cleanup runs through this encoder's rate control: it says the
    /// bytes of its frames, so the cleanup can tell when the rate control has stopped refining
    /// a still screen, and with their quantizer whether that is at the paint-over quality.
    pub fn converges(&self) -> bool {
        self.bytes.is_some()
    }
}

/// Where a rate control that the cleanup runs through stands after its last frame, of `bytes`
/// at quality index `last`, against the paint-over quantizer and `budget` bytes a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Convergence {
    /// Still refining: its last frame carried a share of the budget.
    Refining,
    /// A small frame at the paint-over quantizer or finer: nothing left to refine.
    Converged,
    /// A small frame at a coarser quantizer, or at one the encoder does not say: refining
    /// nothing more this frame.
    Idle,
}

/// Where the rate control stands (`Convergence`).
pub fn convergence(
    last: Option<u32>,
    bytes: Option<usize>,
    paint: u32,
    budget: f64,
) -> Convergence {
    match bytes {
        Some(b) if (b as f64) <= budget * CONVERGED_SHARE => {
            if last.is_some_and(|q| q <= paint) {
                Convergence::Converged
            } else {
                Convergence::Idle
            }
        }
        _ => Convergence::Refining,
    }
}

/// Trigger periods between two measurements of a picture a constant-rate cleanup is refining
/// (`FrameEncoder::last_psnr`), the gain in dB under which one finds the picture level with
/// the one before, and the level measurements in a row after which the rate control is done
/// with it. A rate control that has a whole screen of dense text to refine at 2 Mbit/s at
/// 1080p gains 0.2 dB a second at first, coding a few blocks of it a frame, and one that is
/// done with a screen under 0.06.
const PLATEAU_TRIGGERS: u32 = 4;
const PLATEAU_GAIN_DB: f32 = 0.05;
const PLATEAU_CHECKS: u32 = 2;

/// Bytes a frame of the constant-rate target, at `kbps` and `fps`.
pub fn frame_budget(kbps: i32, fps: f64) -> f64 {
    kbps.max(1) as f64 * 125.0 / fps.max(1.0)
}

/// The share of the picture the next band of a sweep covers, after one of `size` whose held frame
/// came out at `bytes` (`FIRST_BAND` onward): `BAND_BUDGETS` of `budget`, at most doubling or
/// halving a frame.
fn band_share(size: f64, bytes: Option<usize>, budget: f64) -> f64 {
    let next = match bytes {
        Some(b) if b > 0 => (size * BAND_BUDGETS * budget / b as f64).clamp(size / 2.0, size * 2.0),
        _ => size,
    };
    next.clamp(1.0 / 4096.0, 1.0)
}

/// One frame of a cleanup that measures the picture it is refining, where the encoder can
/// (`FrameEncoder::measure`): whether this frame is to be measured, one every `period`, and
/// whether the measurement of the last such frame, `psnr`, was the `PLATEAU_CHECKS`th in a
/// row to gain less than `PLATEAU_GAIN_DB` on the one before, the rate control having
/// nothing left to give the picture.
fn plateau(st: &mut StripeState, psnr: Option<f32>, period: u32) -> (bool, bool) {
    if std::mem::take(&mut st.measuring)
        && let Some(now) = psnr
    {
        let level = st
            .measured
            .is_some_and(|before| now - before < PLATEAU_GAIN_DB);
        st.level_checks = if level { st.level_checks + 1 } else { 0 };
        st.measured = Some(now);
    }
    if st.measure_in == 0 {
        st.measure_in = period;
        st.measuring = true;
    } else {
        st.measure_in -= 1;
    }
    (st.measuring, st.level_checks >= PLATEAU_CHECKS)
}

/// Frames a constant-rate cleanup through the rate control flows at most (`CONVERGE_S`).
pub fn converge_frames(settings: &RustCaptureSettings) -> i32 {
    (CONVERGE_S * settings.target_fps.max(1.0)).round() as i32
}

/// `converge_frames` for the session `encoder` describes (`REFINE_S`).
fn cleanup_frames(settings: &RustCaptureSettings, encoder: EncoderQuality) -> i32 {
    if encoder.holds || (encoder.last.is_none() && !encoder.measures) {
        converge_frames(settings)
    } else {
        (REFINE_S * settings.target_fps.max(1.0)).round() as i32
    }
}

/// Whether a cleanup improves the picture at all: at a constant quality where the paint-over
/// quality is finer than the session's (the cleanup then moves the session's quality, see
/// `decide_constant_quality`); at a constant rate where it is finer than the quality index the
/// rate control last coded at, or the encoder does not say. A rate control that has already
/// refined a still screen past it (a constant rate refines every frame it sends) is left alone:
/// holding the paint-over quantizer there would coarsen the picture. A constant-rate encoder
/// that holds no quantizer is cleaned up by the rate control's own frames, a refresh and its
/// burst, which it refines; a key frame there would come out at the rate control's quantizer,
/// which a small buffer starves.
pub fn paint_over_improves(settings: &RustCaptureSettings, encoder: EncoderQuality) -> bool {
    let paint = settings.video_paintover_crf.max(0) as u32;
    settings.use_paint_over_quality
        && if settings.video_cbr_mode {
            encoder.last.is_none_or(|q| q > paint)
        } else {
            settings.video_paintover_crf < settings.video_crf
        }
}

/// Outcome of the full-frame send decision produced by `decide_hw_fullframe`.
pub struct HwFrameDecision {
    pub send: bool,
    pub force_idr: bool,
    /// The session quality index the frame is encoded at.
    pub target_qp: u32,
    /// The quality index this frame is held at whatever the rate control: the cleanup of a
    /// still screen, handed to the encoder's `hold_quantizer`.
    pub hold_qp: Option<u32>,
    /// The share of the picture, from and to in raster order, `hold_qp` covers, the rest of the
    /// frame held at the coarsest quantizer; `None` for the whole picture.
    pub hold_band: Option<(f64, f64)>,
    /// Whether the encoder is asked to measure this frame's reconstruction
    /// (`FrameEncoder::measure`).
    pub measure: bool,
}

impl HwFrameDecision {
    /// The quality the whole frame is coded at in place of the session's `normal` one: a
    /// quantizer held for it, or a constant quality's paint-over one.
    pub(crate) fn cleanup_quality(&self, normal: u32) -> Option<u32> {
        self.hold_qp
            .filter(|_| self.hold_band.is_none())
            .or((self.target_qp != normal).then_some(self.target_qp))
    }

    /// Tell `encoder` what the decision asks of the frame it is about to code: the quantizer
    /// it is held at and whether its reconstruction is measured.
    pub fn prepare(&self, encoder: &mut FrameEncoder) {
        if let Some(q) = self.hold_qp {
            encoder.hold_quantizer(q, self.hold_band);
        }
        if self.measure {
            encoder.measure();
        }
    }
}

/// Whether a scheduled keyframe is due this tick.
///
/// The default (`keyframe_interval_s <= 0`) is an infinite GOP with no scheduled IDRs. A positive
/// interval buys a fixed ~N-second recovery cadence for consumers that cannot request one on demand.
///
/// # Arguments
///
/// * `settings` - Capture settings; reads `keyframe_interval_s` and `target_fps`.
/// * `frame_counter` - Current frame number (wrapping `u16`).
///
/// # Returns
///
/// `true` if a periodic IDR should be forced on this frame.
pub fn periodic_idr_due(settings: &RustCaptureSettings, frame_counter: u16) -> bool {
    let secs = settings.keyframe_interval_s;
    if secs <= 0.0 {
        return false;
    }
    let safe_fps = settings.target_fps.max(1.0);
    let interval = ((safe_fps * secs).round() as u64).max(1);
    (frame_counter as u64).is_multiple_of(interval)
}

/// The quality index a held refresh (a predicted frame) is coded at: the paint-over one, or,
/// where a constant rate last coded far coarser, as much finer than that as
/// `HELD_KEY_BUDGET_S` of the target allows at six steps a doubling, since a refresh of a whole
/// screen at the paint-over quantizer after heavy motion at a low rate would cost many frames of
/// budget at once.
pub fn held_refresh_quality(settings: &RustCaptureSettings, encoder: EncoderQuality) -> u32 {
    refresh_quality(settings, encoder, 1.0)
}

/// `held_refresh_quality` from a rate control's last frame of `spent` frame budgets rather than
/// one: a rate control pinned at its coarsest quantizer (`OVERSHOOT_BUDGETS`) refreshed in one
/// held frame, which then costs that much more.
fn refresh_quality(settings: &RustCaptureSettings, encoder: EncoderQuality, spent: f64) -> u32 {
    let paint = settings.video_paintover_crf.max(0) as u32;
    match (settings.video_cbr_mode, encoder.last) {
        (true, Some(last)) => {
            let frames = (crate::encoders::HELD_KEY_BUDGET_S * settings.target_fps
                / spent.max(1.0))
            .max(1.0);
            paint.max(last.saturating_sub((6.0 * frames.log2()).floor() as u32))
        }
        _ => paint,
    }
}

/// The send / quality / keyframe policy every full-frame encoder obeys. A constant-quality
/// session takes `decide_constant_quality`; what follows is the constant-rate one.
///
/// The GOP is left infinite and an IDR is forced only when a consumer genuinely needs a fresh
/// decode entry point or a still screen is cleaned up. Every full-frame encoder shares this one
/// function so they cannot drift apart; the striped software path applies the same policy per
/// stripe inside `encode_cpu`.
///
/// 1. **Cleanup** (`cleanup_due`): once the content stops changing, whatever Turbo sends, a
///    refresh and, after a large change and a longer stillness, a key frame where the encoder's
///    cleanup ends on one (`EncoderQuality::keys`), each encoded at the paint-over quantizer held
///    for it (`hold_qp`, which the encoder applies under any rate control; a refresh of a
///    constant-rate session coarsened to `held_refresh_quality`, a key frame capped by the encoder
///    at `HELD_KEY_BUDGET_S`) and followed by a recovery burst. A constant-rate session whose rate
///    control reports its frames' bytes (`EncoderQuality::converges`) is cleaned up through that
///    rate control instead, NVENC's like this:
///    the frames keep flowing, each within the rate control's budget, until it codes the screen
///    at the paint-over quantizer or finer in a frame under `CONVERGED_SHARE` of the budget
///    (`convergence`), for `CONVERGE_S` at most. A frame held at the paint-over quantizer the
///    size of a whole refresh would otherwise queue for up to half a second on a link near the
///    stream's rate; the rate control reaches the same picture in about a second at the
///    default rate with every frame within its budget. Only where it has not converged within
///    `FALLBACK_TRIGGERS` periods, or stalls short of the paint-over quality (`STALL_TRIGGERS`
///    periods of small frames at a coarser quantizer), as where the rate is low for the
///    resolution, is the picture refreshed at `held_refresh_quality`, which ends the cleanup: a
///    band a frame where the encoder holds one (`FIRST_BAND`, `BAND_BUDGETS`) until the sweep
///    has covered it or motion ends it, else one held frame. A sweep whose rest would take it
///    past `REFINE_S` at its bands' pace, once they are sized to the budget, is coded a step
///    coarser a band until the pace fits, and ends where that reaches the rate control's own
///    quantizer: dense text at 0.1 and 0.25 Mbit/s never finished in 40 s at the paint-over
///    one. At 2 Mbit/s at 1080p a key frame
///    capped to `HELD_KEY_BUDGET_S` of the target came out coarser than the picture it cleaned
///    (35.9 dB), where the refresh reached 44.2. A rate control whose frames of the still screen
///    run over `OVERSHOOT_BUDGETS` for a trigger period is pinned at its coarsest quantizer and
///    refines nothing within the rate, and every frame of a sweep would carry the rest of the
///    picture at that cost: NVENC's at 0.1 Mbit/s codes a still 1080p screen of texture in
///    16-budget frames, and its sweep sent 4.7 MB in 40 s without ending. Such a screen is
///    refreshed at once in one held frame, coarsened to what `HELD_KEY_BUDGET_S` buys at those
///    frames' size and coded again to fit it where it comes out past `HELD_REFRESH_LIMIT_S`, as
///    dense text does, and a burst that has no refresh to fall back to ends. A session that reports
///    its frames' bytes but holds no quantizer at a constant rate (VA-API, x265, kvazaar, libvpx's
///    VP9) has no refresh to fall back to, so its frames flow, each the rate control's, until the
///    picture is done: where the session measures its reconstruction against its source (VA-API,
///    `plateau`) until that stops improving, else after a trigger period of frames at the
///    paint-over quantizer or once the finest quantizer it has coded at has stood for
///    `LEVEL_S`; and in either case after `STALL_TRIGGERS` periods of small frames or
///    `REFINE_S`. A fixed burst left such a screen where the motion did (19 to 27 dB at
///    2 Mbit/s at 1080p on iHD, x265, and SVT-AV1, each frame within its budget), and a
///    reported quantizer alone ends it early or never: iHD's H.264 average reads 26 through
///    a refinement from 23 to 50 dB. That cleanup then counts as settled
///    (`StripeState::settled`), and only a change of more than a small area arms another, so
///    a caret blinking on a clean screen costs its own frames and no more. A session that
///    reports no bytes (Tegra, a stateful V4L2 device) gets the refresh and burst at its rate
///    control's own quality, and no key frame. A change of more than a small area ends a
///    cleanup through the rate control; a caret blinking or a clock ticking on the screen
///    does not, so constant low motion is cleaned up the same way.
/// 2. **Recovery keyframe**: a requested or scheduled IDR, at the rate control's own quality: it
///    answers a join or a loss, often on a link that just fell behind, where a key frame the
///    size of a cleanup would only fall behind again. On a still screen it opens the burst,
///    which lasts until the rate control converges where the cleanup runs through it, or is
///    pinned (`OVERSHOOT_BUDGETS`).
/// 3. **Recovery burst**: the frames after a keyframe or cleanup on a still screen keep flowing
///    so rate control settles, until motion resumes: after a cleanup held at the refresh's
///    quantizer where the paint-over one still improves on the rate control's, so they go on
///    refining the picture (a key frame capped to its budget below the paint-over quality
///    included), and so after any keyframe of a constant-quality session; after a recovery key
///    frame of a constant-rate session they are the rate control's.
/// 4. **Motion**, Turbo (`video_streaming_mode`), and animated overlays send at the session's
///    quality; a still screen outside all of the above sends nothing.
///
/// # Arguments
///
/// * `st` - Per-region state carrying the cleanup and burst bookkeeping.
/// * `settings` - Capture settings; reads CRF, paint-over CRF, burst frames, trigger frames,
///   streaming mode, and keyframe interval.
/// * `frame_counter` - Current frame number (wrapping `u16`).
/// * `damage` - What changed, from compositor damage, the X server, the driver, or a content hash.
/// * `is_animated` - Forces a send for animated overlays.
/// * `requested_idr` - On-demand IDR request (client join / reset / recording cadence).
/// * `encoder` - What the encoder says of its frames (`EncoderQuality`).
///
/// # Returns
///
/// [`HwFrameDecision`] with `send`, `force_idr`, `target_qp` (the session quality index),
/// `hold_qp` (the quantizer a cleanup holds its frame at), and `hold_band` (the share of the
/// picture it covers).
pub fn decide_hw_fullframe(
    st: &mut StripeState,
    settings: &RustCaptureSettings,
    frame_counter: u16,
    damage: Damage,
    is_animated: bool,
    requested_idr: bool,
    encoder: EncoderQuality,
) -> HwFrameDecision {
    if !settings.video_cbr_mode {
        return decide_constant_quality(
            st,
            settings,
            frame_counter,
            damage,
            is_animated,
            requested_idr,
            encoder,
        );
    }
    let normal_qp = settings.video_crf as u32;
    let paint_qp = settings.video_paintover_crf as u32;
    let burst = settings.video_paintover_burst_frames;
    let improves = paint_over_improves(settings, encoder);
    let recovery_idr = requested_idr || periodic_idr_due(settings, frame_counter);
    let converges = encoder.converges();
    let holds = encoder.holds && !converges;
    // A session that holds its refresh and a band of a frame (libvpx's VP8) sweeps the refresh
    // a band a frame, as a converging one's fallback does, rather than holding it whole.
    let sweeps = holds && encoder.band.is_some();
    let cleanup = cleanup_due(
        st,
        settings.paint_over_trigger_frames,
        improves,
        true,
        holds && encoder.keys,
        damage,
    );
    let refresh_qp = held_refresh_quality(settings, encoder);
    let cleanup_qp = match cleanup {
        Cleanup::Refresh if holds => Some(refresh_qp),
        Cleanup::Key => Some(paint_qp),
        _ => None,
    };
    let mut d = HwFrameDecision {
        send: false,
        force_idr: false,
        target_qp: normal_qp,
        hold_qp: None,
        hold_band: None,
        measure: false,
    };
    if damage.is_dirty() {
        if !converges || damage.is_motion() {
            st.h264_burst_frames_remaining = 0;
        }
        // Motion ends a sweep. A small change (a caret) leaves it to finish: one that started
        // over at every blink would never cover the picture.
        if damage.is_motion() {
            st.sweep = None;
            st.settled = false;
        }
        d.send = true;
        d.force_idr = recovery_idr || cleanup == Cleanup::Key;
        d.hold_qp = cleanup_qp;
        if sweeps && cleanup == Cleanup::Refresh && !d.force_idr {
            sweep_refresh(st, &mut d);
        }
        return d;
    }
    if cleanup != Cleanup::None || recovery_idr {
        d.send = true;
        d.force_idr = recovery_idr || cleanup == Cleanup::Key;
        d.hold_qp = cleanup_qp;
        if recovery_idr {
            st.sweep = None;
        } else if sweeps && cleanup == Cleanup::Refresh && !d.force_idr {
            sweep_refresh(st, &mut d);
        }
        if converges && (cleanup != Cleanup::None || (burst > 0 && recovery_idr)) {
            if (st.h264_burst_frames_remaining <= 0 && !st.settled) || recovery_idr {
                st.h264_burst_frames_remaining = cleanup_frames(settings, encoder);
                st.idle_frames = 0;
                st.over_frames = 0;
                st.same_frames = 0;
                st.rc_bytes = 0;
                st.finest = None;
                st.finest_frames = 0;
                st.fine_frames = 0;
                st.measured = None;
                st.level_checks = 0;
                st.measuring = false;
                st.measure_in = 0;
            }
            st.burst_held = false;
        } else if burst > 0 && (d.force_idr || cleanup != Cleanup::None) && st.sweep.is_none() {
            // A sweep spends a budget a frame and leaves the rate control nothing to settle, so no
            // burst follows it, which would hold whole frames again once it ends.
            st.h264_burst_frames_remaining = burst;
            st.burst_held =
                holds && (cleanup != Cleanup::None || (improves && !settings.video_cbr_mode));
        }
        return d;
    }
    if let Some((from, size)) = st.sweep {
        let budget = frame_budget(settings.video_bitrate_kbps, settings.target_fps);
        let size = band_share(size, encoder.band, budget);
        let to = (from + size).min(1.0);
        st.sweep_frames = st.sweep_frames.saturating_add(1);
        let left = (REFINE_S * settings.target_fps.max(1.0)).round() - st.sweep_frames as f64;
        let paced = encoder
            .band
            .is_some_and(|b| b as f64 <= 2.0 * BAND_BUDGETS * budget);
        if paced && (1.0 - to) / size > left.max(1.0) {
            st.sweep_coarser += 1;
        }
        let q = refresh_qp + st.sweep_coarser;
        if encoder.last.is_some_and(|last| q >= last) {
            st.sweep = None;
            return d;
        }
        st.sweep = (to < 1.0).then_some((to, size));
        d.send = true;
        d.hold_qp = Some(q);
        d.hold_band = Some((from, to));
        return d;
    }
    let budget = frame_budget(settings.video_bitrate_kbps, settings.target_fps);
    let trigger = settings.paint_over_trigger_frames.max(1);
    if converges && st.h264_burst_frames_remaining > 0 {
        st.over_frames = if encoder
            .bytes
            .is_some_and(|b| b as f64 > budget * OVERSHOOT_BUDGETS)
        {
            st.over_frames.saturating_add(1)
        } else {
            0
        };
    }
    let pinned = converges && st.h264_burst_frames_remaining > 0 && st.over_frames >= trigger;
    if pinned && !(improves && encoder.holds) {
        st.h264_burst_frames_remaining = 0;
        st.settled = !encoder.holds;
    }
    if converges && st.h264_burst_frames_remaining > 0 {
        match convergence(encoder.last, encoder.bytes, paint_qp, budget) {
            Convergence::Converged if !encoder.measures => {
                st.h264_burst_frames_remaining = 0;
                st.change_mass = 0.0;
                st.settled = !encoder.holds;
            }
            Convergence::Converged | Convergence::Idle => {
                st.idle_frames = st.idle_frames.saturating_add(1)
            }
            Convergence::Refining => st.idle_frames = 0,
        }
        st.fine_frames = if encoder.last.is_some_and(|q| q <= paint_qp) {
            st.fine_frames.saturating_add(1)
        } else {
            0
        };
    }
    if st.h264_burst_frames_remaining > 0 {
        st.h264_burst_frames_remaining -= 1;
        d.send = true;
        d.hold_qp = (st.burst_held && improves).then_some(refresh_qp);
        let elapsed = (converge_frames(settings) - st.h264_burst_frames_remaining).max(0) as u32;
        let stalled = st.idle_frames >= STALL_TRIGGERS * trigger;
        if converges && !encoder.holds {
            let reached = settings.use_paint_over_quality && st.fine_frames >= trigger;
            let (measure, level) = plateau(st, encoder.psnr, PLATEAU_TRIGGERS * trigger);
            d.measure = measure;
            let done = if encoder.measures { level } else { reached };
            let bytes = encoder.bytes.unwrap_or(0);
            st.same_frames = if bytes > 0
                && bytes.abs_diff(st.rc_bytes) as f64 <= st.rc_bytes as f64 * SAME_FRAME_SHARE
            {
                st.same_frames.saturating_add(1)
            } else {
                0
            };
            st.rc_bytes = bytes;
            let same = st.same_frames as f64 >= EMPTY_S * settings.target_fps
                && encoder.last.is_some_and(|q| q >= SAME_FRAME_QUALITY);
            if let Some(q) = encoder.last {
                if st.finest.is_none_or(|finest| q < finest) {
                    st.finest = Some(q);
                    st.finest_frames = 0;
                } else {
                    st.finest_frames = st.finest_frames.saturating_add(1);
                }
            }
            let level = st.finest_frames as f64 >= LEVEL_S * settings.target_fps;
            if ((stalled || same || level) && !encoder.measures)
                || done
                || st.h264_burst_frames_remaining == 0
            {
                st.h264_burst_frames_remaining = 0;
                st.settled = true;
            }
        } else if converges
            && improves
            && (stalled || pinned || elapsed >= FALLBACK_TRIGGERS * trigger)
        {
            d.hold_qp = Some(if pinned {
                refresh_quality(
                    settings,
                    encoder,
                    encoder.bytes.unwrap_or(0) as f64 / budget,
                )
            } else {
                refresh_qp
            });
            st.h264_burst_frames_remaining = 0;
            if encoder.band.is_some() && !pinned {
                start_sweep(st, &mut d);
            }
        }
        return d;
    }
    d.send = settings.video_streaming_mode || is_animated;
    d
}

/// Start a refresh sweep at the top of the picture: its first band in this frame, the rest a band
/// a frame after it (`FIRST_BAND`, `band_share`).
fn start_sweep(st: &mut StripeState, d: &mut HwFrameDecision) {
    d.hold_band = Some((0.0, FIRST_BAND));
    st.sweep = Some((FIRST_BAND, FIRST_BAND));
    st.sweep_frames = 0;
    st.sweep_coarser = 0;
}

/// A refresh due on a session that sweeps it (`decide_hw_fullframe`'s `sweeps`): a sweep from the
/// top, or, with one under way, which is already refreshing the picture, nothing held, the frame
/// going at the rate control's quality: a sweep started over at every change of a blinking caret
/// would never cover the picture, and a whole held frame is what the sweep spreads.
fn sweep_refresh(st: &mut StripeState, d: &mut HwFrameDecision) {
    if st.sweep.is_some() {
        d.hold_qp = None;
    } else {
        start_sweep(st, d);
    }
}

/// `decide_hw_fullframe` for a constant-quality session: the cleanup moves the session's own
/// quality to the paint-over one, as main's paint-over did, rather than holding a frame at a
/// quantizer. Each encoder then codes the cleanup the way it codes any frame at that quality:
/// x264's and x265's rate factor give a key frame their intra offset (about three steps finer
/// than the paint-over index), SVT-AV1 its key-frame qindex, and the encoders whose quality
/// change re-opens the session (x265, SVT-AV1, kvazaar) open on a key frame at it. A frame held
/// at the paint-over quantizer itself came out coarser than main's (SVT-AV1 49.6 against 61.0 dB,
/// x264 53.7 against 56.5 dB).
///
/// The session stays at the paint-over quality while the region stays clean: from its cleanup
/// until motion (more than a small change), so the frames Turbo sends of a still screen, or a
/// caret blinking on it, do not take a re-opening encoder back to the session's quality, a key
/// frame at it, after `QP_HYSTERESIS_LIMIT` frames. The refresh, the key frame, and the bursts
/// come when `cleanup_due` says, as at a constant rate; a requested or scheduled key frame is
/// coded at the session's own quality, as on main, and its burst at the paint-over one. An
/// encoder whose quality change re-opens it (`EncoderQuality::reopens`: x265, kvazaar, SVT-AV1)
/// codes the refresh that moves it as a key frame, as main's paint-over did, so that refresh is
/// its key-frame cleanup and no second key frame follows for the same change.
fn decide_constant_quality(
    st: &mut StripeState,
    settings: &RustCaptureSettings,
    frame_counter: u16,
    damage: Damage,
    is_animated: bool,
    requested_idr: bool,
    encoder: EncoderQuality,
) -> HwFrameDecision {
    let normal_qp = settings.video_crf as u32;
    let paint_qp = settings.video_paintover_crf as u32;
    let burst = settings.video_paintover_burst_frames;
    let improves =
        settings.use_paint_over_quality && settings.video_paintover_crf < settings.video_crf;
    let recovery_idr = requested_idr || periodic_idr_due(settings, frame_counter);
    let mut cleanup = cleanup_due(
        st,
        settings.paint_over_trigger_frames,
        improves,
        true,
        true,
        damage,
    );
    if damage.is_motion() || !improves {
        st.clean_quality = false;
    }
    if cleanup == Cleanup::Refresh && encoder.reopens && !st.clean_quality {
        cleanup = Cleanup::Key;
        st.change_mass = 0.0;
    }
    if cleanup != Cleanup::None {
        st.clean_quality = true;
    }
    let quality = if st.clean_quality {
        paint_qp
    } else {
        normal_qp
    };
    let mut d = HwFrameDecision {
        send: false,
        force_idr: false,
        target_qp: quality,
        hold_qp: None,
        hold_band: None,
        measure: false,
    };
    if recovery_idr && cleanup == Cleanup::None {
        d.target_qp = normal_qp;
    }
    if damage.is_dirty() {
        st.h264_burst_frames_remaining = 0;
        d.send = true;
        d.force_idr = recovery_idr || cleanup == Cleanup::Key;
        return d;
    }
    if cleanup != Cleanup::None || recovery_idr {
        d.send = true;
        d.force_idr = recovery_idr || cleanup == Cleanup::Key;
        if burst > 0 {
            st.h264_burst_frames_remaining = burst;
            st.burst_held = improves;
        }
        return d;
    }
    if st.h264_burst_frames_remaining > 0 {
        st.h264_burst_frames_remaining -= 1;
        d.send = true;
        if st.burst_held {
            d.target_qp = paint_qp;
        }
        return d;
    }
    d.send = settings.video_streaming_mode || is_animated;
    d
}

/// Rows per band of the content hash a full-frame X11 session reads its damage from: one band is
/// under `SMALL_AREA` of a screen from 720 rows up, so a caret or a clock reads as the small
/// change it is.
pub(crate) const DAMAGE_BAND_ROWS: usize = 32;

/// Which bands of `DAMAGE_BAND_ROWS` rows of a host frame (`stride` bytes per row, `height` rows)
/// changed against the frame before, read from a content hash per band
/// (`StripeState::content_dirty`, one state per band in `bands`, with its damage blocks). The
/// bands hash in turn on the calling thread: a band takes tens of microseconds, less than waking
/// the rayon pool for it costs, which on a many-core host spent several times the hash in CPU
/// and added milliseconds waiting on the slowest worker.
fn hash_bands(
    bands: &mut Vec<StripeState>,
    pixels: &[u8],
    stride: usize,
    height: usize,
    threshold: u32,
    duration: i32,
) -> Vec<bool> {
    let n = height.div_ceil(DAMAGE_BAND_ROWS).max(1);
    if bands.len() != n {
        bands.clear();
        bands.resize_with(n, StripeState::default);
    }
    bands
        .iter_mut()
        .enumerate()
        .map(|(i, band)| {
            let end = ((i + 1) * DAMAGE_BAND_ROWS).min(height) * stride;
            let bytes =
                &pixels[(i * DAMAGE_BAND_ROWS * stride).min(pixels.len())..end.min(pixels.len())];
            band.content_dirty(bytes, threshold, duration)
        })
        .collect()
}

/// The damage a band hash (`hash_bands`) reads: the fraction of the bands that changed. No bands
/// read yet, before a capture's first hash, is all new.
fn band_damage(dirty: &[bool]) -> Damage {
    let changed = dirty.iter().filter(|&&d| d).count();
    if dirty.is_empty() {
        Damage::Area(1.0)
    } else if changed == 0 {
        Damage::None
    } else {
        Damage::Area(changed as f32 / dirty.len() as f32)
    }
}

/// The damage of a frame whose bands were hashed where it lies (NvFBC's, on the GPU) against the
/// hashes of the frame before, which `last` keeps: the share of the bands that changed. No hashes
/// before is all new.
pub(crate) fn hashed_damage(last: &mut Vec<u64>, hashes: Vec<u64>) -> Damage {
    let dirty: Vec<bool> = if last.len() == hashes.len() {
        hashes
            .iter()
            .zip(last.iter())
            .map(|(a, b)| a != b)
            .collect()
    } else {
        Vec::new()
    };
    *last = hashes;
    band_damage(&dirty)
}

/// The damage of a host frame against the one before it, from its band hash (`hash_bands`).
fn hash_damage(
    bands: &mut Vec<StripeState>,
    pixels: &[u8],
    stride: usize,
    height: usize,
    threshold: u32,
    duration: i32,
) -> Damage {
    band_damage(&hash_bands(
        bands, pixels, stride, height, threshold, duration,
    ))
}

/// The rows a band hash (`hash_bands`) found changed in a `width` x `height` frame, as
/// full-width rectangles, one per run of changed bands: the striped path's dirty map. No bands
/// read yet is the whole frame.
fn band_rects(dirty: &[bool], width: i32, height: i32) -> Vec<Rectangle<i32, Physical>> {
    if dirty.is_empty() {
        return vec![Rectangle::new((0, 0).into(), (width, height).into())];
    }
    let mut rects = Vec::new();
    let mut run = None;
    for (i, &changed) in dirty.iter().chain(std::iter::once(&false)).enumerate() {
        match (changed, run) {
            (true, None) => run = Some(i),
            (false, Some(first)) => {
                let top = (first * DAMAGE_BAND_ROWS) as i32;
                let bottom = ((i * DAMAGE_BAND_ROWS) as i32).min(height);
                rects.push(Rectangle::new(
                    (0, top).into(),
                    (width, bottom - top).into(),
                ));
                run = None;
            }
            _ => {}
        }
    }
    rects
}

/// Run `encode` on this thread while a thread of its own hashes the frame's bands (`hash_bands`),
/// and return what `encode` returned with the bands that changed: a Turbo frame's hash kept off
/// its encode's path. A scoped thread per frame rather than the rayon pool, whose idle workers
/// spin after every wake; where no thread starts, the hash runs here after the encode.
#[allow(clippy::too_many_arguments)]
fn hash_beside<R>(
    bands: &mut Vec<StripeState>,
    pixels: &[u8],
    stride: usize,
    height: usize,
    threshold: u32,
    duration: i32,
    encode: impl FnOnce() -> R,
) -> (R, Vec<bool>) {
    let (out, hashed) = std::thread::scope(|s| {
        let hashing = std::thread::Builder::new()
            .name("pxf-x11-hash".into())
            .spawn_scoped(s, || {
                hash_bands(bands, pixels, stride, height, threshold, duration)
            });
        let out = encode();
        (out, hashing.ok().map(|h| h.join().unwrap_or_default()))
    });
    let dirty =
        hashed.unwrap_or_else(|| hash_bands(bands, pixels, stride, height, threshold, duration));
    (out, dirty)
}

/// Everything the X11 host-ARGB path has to remember between frames.
///
/// Unlike the Wayland backend, X11 capture has no compositor to report what changed, so this
/// context exists to hold the state that stands in for that missing damage signal: the per-stripe
/// hashes and the persistent encoder session that let `process()` discover damage by comparing
/// content frame-to-frame. A full-frame encoder runs through `decide_hw_fullframe`; the striped
/// software path (JPEG, striped or full-frame software H.264) runs through `encode_cpu` with
/// `hash_damage=true`.
///
/// Recording fan-out (socket sink and MP4 recorder) is handled at the delivery layer; a
/// consumer needing a keyframe goes through [`X11Pipeline::request_idr`] like everyone else.
pub struct X11Pipeline {
    settings: RustCaptureSettings,
    stripes: Vec<StripeState>,
    /// Smoothed number of stripes carrying the encode budget (see `stripe_rate_control`).
    stripes_carrying: f32,
    /// The full-frame encoder, or `None` for the striped software path.
    hw: Option<FrameEncoder>,
    hw_state: StripeState,
    /// The content hashes of a full-frame session, one per band of `DAMAGE_BAND_ROWS` rows.
    bands: Vec<StripeState>,
    /// The bands the content hash found changed in the frame before this one, which a Turbo
    /// frame reads (`process`); empty until the first hash.
    turbo_dirty: Vec<bool>,
    frame_counter: u16,
    pending_force_idr: bool,
    /// Consecutive hardware encode failures and whether this pipeline already spent its one
    /// rebuild; together they drive `recover_hw`.
    hw_error_streak: u32,
    hw_rebuilt: bool,
    /// The cleanup quality (`HwFrameDecision::cleanup_quality`) and key frame the last full frame
    /// was decided on, and the ones it was coded with.
    #[cfg(test)]
    hold: [(Option<u32>, bool); 2],
}

impl X11Pipeline {
    /// Build the context, choosing the full-frame encoder for the X11 host-BGRA path through
    /// the shared ladder (`select_frame_encoder`): the hardware backend the encode node's driver
    /// selects, then the codec's software encoder, or the striped software path for JPEG and
    /// H.264. A codec no backend serves falls through to the video codecs the host does serve,
    /// JPEG last.
    pub fn new(mut settings: RustCaptureSettings) -> Self {
        let hw = encoders::select_frame_encoder(
            &mut settings,
            FrameSource::Host { rgba: false },
            None,
            "X11",
        );
        let pipeline = Self {
            settings,
            stripes: Vec::new(),
            stripes_carrying: 1.0,
            hw,
            hw_state: StripeState::default(),
            bands: Vec::new(),
            turbo_dirty: Vec::new(),
            frame_counter: 0,
            pending_force_idr: false,
            hw_error_streak: 0,
            hw_rebuilt: false,
            #[cfg(test)]
            hold: [(None, false); 2],
        };
        pipeline.record_stream();
        pipeline
    }

    /// Put the session this pipeline settled on into the capture's report.
    fn record_stream(&self) {
        let fullframe = self.hw.is_some() || self.settings.video_fullframe;
        crate::report::stream(
            &self.settings,
            encoders::software::stripe_count(self.settings.height, self.settings.codec, fullframe),
            self.hw
                .as_ref()
                .map(|enc| (enc.backend_name(), enc.is_hardware())),
            encoders::session_fullcolor(self.hw.as_ref(), &self.settings),
            encoders::session_full_range(self.hw.as_ref(), &self.settings),
        );
        crate::report::bit_depth(encoders::session_bit_depth(
            self.hw.as_ref(),
            &self.settings,
        ));
    }

    /// React to a streak of hardware encode failures: rebuild the session once with the startup
    /// selection, and demote to the software encoder when a fresh session fails the same way.
    /// A session whose encodes keep failing still constructs, so the rebuild only counts as
    /// recovery until an encode succeeds; otherwise the stream would rebuild in a loop and never
    /// demote. Streaming nothing forever is not an option.
    ///
    /// The demote is the last rung: the software path has no encode-error streak of its own, so
    /// once the pipeline lands there it is never re-entered.
    fn recover_hw(&mut self) {
        self.hw_error_streak = 0;
        if self.hw_rebuilt {
            eprintln!(
                "[X11] HW encoder unrecoverable; demoting to software encoding ({}).",
                crate::encoders::software_library(self.settings.codec)
            );
            // The broken session is released before its replacement is built: these failures
            // are usually device memory pressure, and holding both at once is what would make
            // the replacement fail too.
            self.hw = None;
            self.settings.use_cpu = true;
            self.hw = encoders::select_frame_encoder(
                &mut self.settings,
                FrameSource::Host { rgba: false },
                None,
                "X11",
            );
            crate::report::encoder_reason(
                "the hardware encoder failed repeatedly and was given up",
            );
        } else {
            eprintln!("[X11] rebuilding HW encoder after repeated encode errors.");
            self.hw = None;
            self.hw = encoders::select_frame_encoder(
                &mut self.settings,
                FrameSource::Host { rgba: false },
                None,
                "X11",
            );
            self.hw_rebuilt = true;
        }
        self.record_stream();
        self.hw_state = StripeState::default();
        self.stripes.clear();
        self.pending_force_idr = true;
    }

    /// Request an on-demand keyframe on the next processed frame.
    pub fn request_idr(&mut self) {
        self.pending_force_idr = true;
    }

    /// Leave frame `frame_id` and every frame after it out of the predictions, for a client that
    /// lost it; an encoder that cannot codes a keyframe instead.
    pub fn invalidate_reference(&mut self, frame_id: u16) {
        let forgotten = match self.hw.as_mut() {
            Some(enc) => enc.invalidate_reference(frame_id),
            None => invalidate_reference(&mut self.stripes, frame_id),
        };
        if !forgotten {
            self.pending_force_idr = true;
        }
    }

    /// The encoder's name for the stream log: the hardware backend, the software library of a
    /// full-frame session, or `CPU` with the library of the striped software path.
    pub fn encoder_name(&self) -> String {
        match &self.hw {
            Some(enc) if enc.is_hardware() => enc.backend_name().to_string(),
            Some(enc) => format!("CPU ({})", enc.backend_name()),
            None if !self.settings.codec.is_video() => "CPU (turbojpeg)".to_string(),
            None => format!("CPU ({})", crate::encoders::software_library(Codec::H264)),
        }
    }

    /// The codec the pipeline actually emits, after any demotion.
    pub fn codec(&self) -> Codec {
        self.settings.codec
    }

    /// Whether the pipeline encodes on a GPU.
    pub fn is_hardware(&self) -> bool {
        self.hw.as_ref().is_some_and(|enc| enc.is_hardware())
    }

    /// The encoder's one rate control where its backend has one alone
    /// (`FrameEncoder::fixed_rate_control`), for this pipeline's stream log.
    pub fn fixed_rate_control(&self) -> Option<&'static str> {
        self.hw.as_ref().and_then(FrameEncoder::fixed_rate_control)
    }

    /// Whether the encoder holds a cleanup at the quantizer asked for
    /// (`FrameEncoder::holds_quantizer`, `software::stripes_hold_quantizer`), for this
    /// pipeline's stream log.
    pub fn holds_quantizer(&self) -> bool {
        self.hw.as_ref().map_or_else(
            || crate::encoders::software::stripes_hold_quantizer(&self.settings),
            FrameEncoder::holds_quantizer,
        )
    }

    /// The `Colorspace:` field for this pipeline's stream log, describing what its encoder settled
    /// on: a hardware session only reaches 4:4:4 when the device carries it, the software path
    /// only when the build's encoder for the codec does.
    pub fn colorspace_desc(&self) -> &'static str {
        let fullcolor = encoders::session_fullcolor(self.hw.as_ref(), &self.settings);
        crate::encoders::colorspace_desc(
            fullcolor,
            encoders::session_full_range(self.hw.as_ref(), &self.settings),
        )
    }

    /// Adapt the live pipeline to recreated capture surfaces without rebuilding it.
    ///
    /// # Arguments
    ///
    /// * `settings` - New geometry plus current live rates.
    /// * `size_changed` - Whether the capture dimensions changed.
    ///
    /// # Returns
    ///
    /// `true` if the pipeline was successfully adapted in place; `false` when the active encoder
    /// cannot follow (VAAPI on resize) and the caller must rebuild.
    pub fn reshape(&mut self, settings: &RustCaptureSettings, size_changed: bool) -> bool {
        if !size_changed {
            if let Some(FrameEncoder::Nvenc(enc)) = &mut self.hw {
                enc.release_pinned_hosts();
            }
            self.settings = settings.clone();
            return true;
        }
        match &mut self.hw {
            Some(FrameEncoder::Nvenc(enc)) => {
                if let Err(e) = enc.reconfigure_resolution(settings) {
                    eprintln!("[X11] NVENC in-place resize unavailable ({e}); rebuilding");
                    return false;
                }
            }
            None => {}
            _ => return false,
        }
        let codec = self.settings.codec;
        self.settings = settings.clone();
        self.settings.codec = codec;
        self.stripes.clear();
        self.hw_state = StripeState::default();
        self.turbo_dirty.clear();
        true
    }

    /// Apply a runtime rate-control / framerate change: the CBR target bitrate + VBV (kbps /
    /// kb; ignored unless CBR is active) and the target fps. NVENC and libvpx reconfigure their
    /// live session, as x265 does unless the frame rate moved; x265 then re-opens, as kvazaar
    /// and SVT-AV1 always do; a VA-API session starts a new sequence; the striped software path
    /// picks the new values up on the next `process()` (encode_cpu reads the updated settings
    /// and reconfigures each stripe's encoder).
    pub fn update_rate(&mut self, bitrate_kbps: i32, vbv_multiplier: f64, fps: f64) {
        self.settings.video_bitrate_kbps = bitrate_kbps;
        self.settings.video_vbv_multiplier = vbv_multiplier;
        if fps > 0.0 {
            self.settings.target_fps = fps;
        }
        if let Some(enc) = self.hw.as_mut()
            && let Err(e) = enc.reconfigure_rate(&self.settings)
        {
            // The failed re-open left the session without a codec context, so it goes
            // through the rebuild-or-demote ladder instead of being encoded into.
            eprintln!("[X11] rate reconfigure failed: {e}");
            self.recover_hw();
        }
    }

    /// Apply live per-frame tunables (quality, paint-over, streaming mode, keyframe
    /// cadence); every encoder re-reads them from the settings on the next process().
    pub fn update_tunables(&mut self, t: &crate::LiveTunables) {
        t.apply_to(&mut self.settings);
    }

    /// Encode one host-ARGB frame and return the encoded stripes.
    ///
    /// Damage is read from the content hash (`hash_bands`). Without Turbo the hash decides whether
    /// a frame (or a stripe) is sent, so it runs first. Turbo sends every video frame, and only
    /// its cleanup reads the damage, so there the frame's hash runs beside its encode
    /// (`hash_beside`) and the cleanup reads the frame before's (`turbo_dirty`; the striped path
    /// as the rows it maps onto its stripes): the hash, a millisecond and a half at 1080p and six
    /// at 4K, stays off the path to the client. A frame the cleanup codes at its own quality is
    /// hashed before its encode instead, so a change landing on it is coded at the session's.
    /// With the paint-over off nothing reads it, and Turbo hashes nothing.
    ///
    /// # Arguments
    ///
    /// * `argb` - Packed BGRA pixel buffer (B,G,R,A byte order, `stride` bytes per row).
    /// * `stride` - Bytes per row (must equal `width * 4` for the software path).
    ///
    /// # Returns
    ///
    /// Vec of [`EncodedStripe`] — empty when nothing changed.
    pub fn process(&mut self, argb: &[u8], stride: usize) -> Vec<EncodedStripe> {
        let width = self.settings.width;
        let height = self.settings.height;
        let requested = self.pending_force_idr;
        let threshold = self.settings.damage_block_threshold;
        let duration = self.settings.damage_block_duration as i32;

        let out = if self.hw.is_some() {
            let turbo = self.settings.video_streaming_mode;
            let damage = if turbo {
                band_damage(&self.turbo_dirty)
            } else {
                hash_damage(
                    &mut self.bands,
                    argb,
                    stride,
                    height as usize,
                    threshold,
                    duration,
                )
            };
            let quality = EncoderQuality::of(self.hw.as_ref().unwrap());
            let mut d = decide_hw_fullframe(
                &mut self.hw_state,
                &self.settings,
                self.frame_counter,
                damage,
                false,
                requested,
                quality,
            );
            // A Turbo frame is decided on the hash of the one before, so a frame coded whole at
            // a cleanup's quality could carry a change it was not decided for, coded at that
            // quality. Such a frame is hashed before it is encoded instead of beside it, and one
            // that moved is coded as motion is: at the session's quality, and a key frame only
            // where one is due. A held band needs none of this: the rest of its frame is held at
            // the coarsest quantizer.
            let normal = self.settings.video_crf as u32;
            #[cfg(test)]
            let decided = (d.cleanup_quality(normal), d.force_idr);
            let mut hashed = None;
            if turbo
                && self.settings.use_paint_over_quality
                && d.send
                && d.cleanup_quality(normal).is_some()
            {
                let dirty = hash_bands(
                    &mut self.bands,
                    argb,
                    stride,
                    height as usize,
                    threshold,
                    duration,
                );
                if band_damage(&dirty).is_motion() {
                    d.hold_qp = None;
                    d.target_qp = normal;
                    d.force_idr = requested || periodic_idr_due(&self.settings, self.frame_counter);
                }
                hashed = Some(dirty);
            }
            #[cfg(test)]
            {
                self.hold = [decided, (d.cleanup_quality(normal), d.force_idr)];
            }
            let fc = self.frame_counter as u64;
            if d.send || self.hw.as_ref().unwrap().holds_frame() {
                let force_idr = d.force_idr;
                let enc = self.hw.as_mut().unwrap();
                d.prepare(enc);
                let mut encode = || {
                    if d.send {
                        enc.encode_host(argb, stride, false, fc, d.target_qp, force_idr)
                    } else {
                        enc.push_held(fc)
                    }
                };
                let res = if let Some(dirty) = hashed {
                    self.turbo_dirty = dirty;
                    encode()
                } else if turbo && self.settings.use_paint_over_quality {
                    let (res, dirty) = hash_beside(
                        &mut self.bands,
                        argb,
                        stride,
                        height as usize,
                        threshold,
                        duration,
                        encode,
                    );
                    self.turbo_dirty = dirty;
                    res
                } else {
                    self.turbo_dirty.clear();
                    encode()
                };
                match res {
                    Ok(data) if !data.is_empty() => {
                        self.hw_error_streak = 0;
                        self.hw_rebuilt = false;
                        let codec = self.settings.codec;
                        enc.delivered_units(data, self.frame_counter)
                            .into_iter()
                            .map(|(data, id, reference)| EncodedStripe {
                                data: Arc::new(data),
                                codec,
                                stripe_y_start: 0,
                                stripe_height: height,
                                frame_id: id as i32,
                                timing: Default::default(),
                                reference,
                            })
                            .collect()
                    }
                    Ok(_) => {
                        self.hw_error_streak = 0;
                        self.hw_rebuilt = false;
                        Vec::new()
                    }
                    Err(e) => {
                        // One line per recovery window: a session failing at frame rate would
                        // otherwise write a line per frame for the life of the capture.
                        if self
                            .hw_error_streak
                            .is_multiple_of(crate::HW_ERROR_RECOVERY_THRESHOLD)
                        {
                            eprintln!("[X11] HW encode error: {e}");
                        }
                        self.hw_error_streak = self.hw_error_streak.saturating_add(1);
                        if self.hw_error_streak >= crate::HW_ERROR_RECOVERY_THRESHOLD {
                            self.recover_hw();
                        }
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            }
        } else {
            debug_assert_eq!(
                stride,
                width as usize * 4,
                "software encode path assumes tightly-packed rows (stride == width*4)"
            );
            let force_idr_all = requested
                || (self.settings.codec.is_video()
                    && periodic_idr_due(&self.settings, self.frame_counter));
            if self.settings.codec.is_video() && self.settings.video_streaming_mode {
                // As a full frame above: a frame that codes a stripe at the paint-over quality
                // while it holds still is hashed before it is encoded, and decided on its own
                // change as well as the one before.
                let hashed = (self.settings.use_paint_over_quality
                    && stripes_held_still(&self.stripes, &self.settings))
                .then(|| {
                    hash_bands(
                        &mut self.bands,
                        argb,
                        stride,
                        height as usize,
                        threshold,
                        duration,
                    )
                });
                let rects = match &hashed {
                    Some(dirty) => {
                        let seen: Vec<bool> = dirty
                            .iter()
                            .enumerate()
                            .map(|(i, &d)| d || self.turbo_dirty.get(i).is_none_or(|&b| b))
                            .collect();
                        band_rects(&seen, width, height)
                    }
                    None => band_rects(&self.turbo_dirty, width, height),
                };
                let (stripes, carrying, settings) = (
                    &mut self.stripes,
                    &mut self.stripes_carrying,
                    &self.settings,
                );
                let mut encode = || {
                    encode_cpu(
                        stripes,
                        carrying,
                        argb,
                        width,
                        height,
                        &rects,
                        settings,
                        self.frame_counter,
                        false,
                        false,
                        force_idr_all,
                    )
                };
                if let Some(dirty) = hashed {
                    self.turbo_dirty = dirty;
                    encode()
                } else if self.settings.use_paint_over_quality {
                    let (out, dirty) = hash_beside(
                        &mut self.bands,
                        argb,
                        stride,
                        height as usize,
                        threshold,
                        duration,
                        encode,
                    );
                    self.turbo_dirty = dirty;
                    out
                } else {
                    self.turbo_dirty.clear();
                    encode()
                }
            } else {
                encode_cpu(
                    &mut self.stripes,
                    &mut self.stripes_carrying,
                    argb,
                    width,
                    height,
                    &[],
                    &self.settings,
                    self.frame_counter,
                    false,
                    true,
                    force_idr_all,
                )
            }
        };

        // An unserved request stays armed: on an infinite GOP an IDR lost to an encode
        // error or skip would never self-heal, leaving a joining consumer with an
        // undecodable stream. A rebuilt or demoted encoder arms one the same way, which
        // is what the second read of the flag picks up.
        self.pending_force_idr = (requested || self.pending_force_idr) && out.is_empty();
        self.frame_counter = self.frame_counter.wrapping_add(1);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An encoder that holds a quantizer and reports no quality of its own.
    const HOLDS: EncoderQuality = EncoderQuality {
        last: None,
        bytes: None,
        holds: true,
        reopens: false,
        keys: true,
        band: None,
        measures: false,
        psnr: None,
    };

    /// A settings block for the hardware full-frame policy: a constant rate, whose cleanup holds
    /// its frames (`HOLDS` reports no quality of its own, so the cleanup always improves),
    /// paint-over on and clearly finer than normal, a short burst, a short trigger, and no
    /// scheduled IDR. `crf_settings` is the constant-quality one.
    fn hw_settings() -> RustCaptureSettings {
        RustCaptureSettings {
            codec: Codec::H264,
            video_cbr_mode: true,
            video_crf: 25,
            video_paintover_crf: 10,
            video_paintover_burst_frames: 3,
            paint_over_trigger_frames: 2,
            use_paint_over_quality: true,
            video_streaming_mode: false,
            keyframe_interval_s: 0.0,
            target_fps: 60.0,
            ..Default::default()
        }
    }

    /// `hw_settings` at a constant quality.
    fn crf_settings() -> RustCaptureSettings {
        RustCaptureSettings {
            video_cbr_mode: false,
            ..hw_settings()
        }
    }

    /// Drive the policy through `frames` frames of `damage`, returning the decisions.
    fn run(
        st: &mut StripeState,
        s: &RustCaptureSettings,
        damage: Damage,
        frames: usize,
    ) -> Vec<HwFrameDecision> {
        (0..frames)
            .map(|i| decide_hw_fullframe(st, s, i as u16 + 1, damage, false, false, HOLDS))
            .collect()
    }

    /// Once motion stops a constant-rate session's screen is cleaned up whether Turbo sends every
    /// frame or not: a refresh held at the paint-over quantizer after the trigger, a key frame
    /// held there once it has held still four trigger periods, each followed by the recovery
    /// burst, then silence (or Turbo's frames at the session quality).
    #[test]
    fn a_still_screen_is_cleaned_up_whatever_turbo_sends() {
        for turbo in [false, true] {
            let s = RustCaptureSettings {
                video_streaming_mode: turbo,
                ..hw_settings()
            };
            let paint = Some(s.video_paintover_crf as u32);
            let mut st = StripeState::default();
            let moving = run(&mut st, &s, Damage::Area(1.0), 3);
            assert!(
                moving
                    .iter()
                    .all(|d| d.send && !d.force_idr && d.hold_qp.is_none()),
                "motion sends at the session quality"
            );
            let still = run(&mut st, &s, Damage::None, 16);
            let sent: Vec<(usize, bool, Option<u32>)> = still
                .iter()
                .enumerate()
                .filter(|(_, d)| d.send)
                .map(|(i, d)| (i + 1, d.force_idr, d.hold_qp))
                .collect();
            let cleanups = vec![
                (2, false, paint),
                (3, false, paint),
                (4, false, paint),
                (5, false, paint),
                (8, true, paint),
                (9, false, paint),
                (10, false, paint),
                (11, false, paint),
            ];
            if !turbo {
                assert_eq!(
                    sent, cleanups,
                    "a refresh at the trigger and a key four triggers in, each with its burst held"
                );
            } else {
                assert_eq!(sent.len(), 16, "Turbo sends every frame");
                let held: Vec<(usize, bool, Option<u32>)> =
                    sent.into_iter().filter(|r| r.2.is_some()).collect();
                assert_eq!(held, cleanups, "the same cleanups under Turbo");
            }
            assert!(
                still.iter().all(|d| d.target_qp == s.video_crf as u32),
                "the session's own quality is left alone"
            );
        }
    }

    /// Under a constant rate the refresh and its burst are held at what the budget allows below the
    /// rate control's last quantizer, the key frame at the paint-over one (the encoder caps it to
    /// its budget), and the burst behind the key frame at the refresh's again, so a key frame the
    /// cap coarsened is refined back.
    #[test]
    fn a_constant_rate_cleanup_keeps_to_its_budget() {
        let s = RustCaptureSettings {
            video_cbr_mode: true,
            ..hw_settings()
        };
        let coarse = EncoderQuality {
            last: Some(40),
            bytes: None,
            holds: true,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        for i in 0..3u16 {
            decide_hw_fullframe(&mut st, &s, i, Damage::Area(1.0), false, false, coarse);
        }
        let still: Vec<HwFrameDecision> = (0..12u16)
            .map(|i| decide_hw_fullframe(&mut st, &s, 10 + i, Damage::None, false, false, coarse))
            .collect();
        let refresh_qp = held_refresh_quality(&s, coarse);
        assert_eq!(
            refresh_qp, 10,
            "35 steps under the rate control's 40 is past the paint-over quality"
        );
        assert_eq!(
            still[1].hold_qp,
            Some(refresh_qp),
            "the refresh at the budget's quality"
        );
        assert!(
            (2..5).all(|i| still[i].hold_qp == Some(refresh_qp)),
            "its burst at the same"
        );
        assert!(
            still[7].force_idr && still[7].hold_qp == Some(s.video_paintover_crf as u32),
            "the key at the paint-over one"
        );
        assert!(
            (8..11).all(|i| still[i].send && still[i].hold_qp == Some(refresh_qp)),
            "its burst at the refresh's"
        );
    }

    /// A small change is cleaned up by a refresh alone; one covering half a screen since the last
    /// key frame earns a key frame once the screen holds still long enough.
    #[test]
    fn a_small_change_is_refreshed_and_a_large_one_keyed() {
        let s = RustCaptureSettings {
            video_paintover_burst_frames: 0,
            ..hw_settings()
        };
        let keys = |st: &mut StripeState, damage: Damage, frames: usize| -> (usize, usize) {
            run(st, &s, damage, frames);
            let still = run(st, &s, Damage::None, 12);
            (
                still
                    .iter()
                    .filter(|d| d.hold_qp.is_some() && !d.force_idr)
                    .count(),
                still.iter().filter(|d| d.force_idr).count(),
            )
        };
        let mut st = StripeState::default();
        assert_eq!(
            keys(&mut st, Damage::Area(1.0), 2),
            (1, 1),
            "a scroll is refreshed, then keyed"
        );
        assert_eq!(
            keys(&mut st, Damage::Area(0.001), 1),
            (1, 0),
            "a caret is refreshed, never keyed"
        );
        assert_eq!(
            keys(&mut st, Damage::Area(0.3), 2),
            (1, 1),
            "0.6 of a screen changed since the last key frame"
        );
    }

    /// An encoder whose held refresh restores the picture whole (libvpx's VP8) is never keyed:
    /// after a large change and a long stillness its refresh and burst are the whole cleanup,
    /// where one that holds only key frames (SVT-AV1) ends on a key frame.
    #[test]
    fn a_refresh_that_restores_the_picture_is_not_followed_by_a_key_frame() {
        let s = RustCaptureSettings {
            video_paintover_burst_frames: 0,
            ..hw_settings()
        };
        for (encoder, keyed) in [
            (
                EncoderQuality {
                    keys: false,
                    ..HOLDS
                },
                0,
            ),
            (HOLDS, 1),
        ] {
            let mut st = StripeState::default();
            for i in 0..2u16 {
                decide_hw_fullframe(&mut st, &s, i, Damage::Area(1.0), false, false, encoder);
            }
            let still: Vec<HwFrameDecision> = (0..40u16)
                .map(|i| {
                    decide_hw_fullframe(&mut st, &s, 10 + i, Damage::None, false, false, encoder)
                })
                .collect();
            assert_eq!(
                still
                    .iter()
                    .filter(|d| d.hold_qp.is_some() && !d.force_idr)
                    .count(),
                1,
                "keys {}: one refresh",
                encoder.keys
            );
            assert_eq!(
                still.iter().filter(|d| d.force_idr).count(),
                keyed,
                "keys {}: the key frames",
                encoder.keys
            );
        }
    }

    /// A still screen the rate control codes finer than the paint-over quantizer owes no cleanup,
    /// so its still frames are no low motion: the first frames of the next large change (a
    /// window opening) go out at the session quality, and the key frame waits for the screen to
    /// hold still after it.
    #[test]
    fn a_constant_rate_cleanup_runs_through_a_rate_control_that_converges() {
        let s = hw_settings();
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps) as usize;
        let refining = EncoderQuality {
            last: Some(40),
            bytes: Some(budget),
            holds: true,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let converged = EncoderQuality {
            last: Some(8),
            bytes: Some(budget / 8),
            holds: true,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let trigger = s.paint_over_trigger_frames as u16;
        let window = trigger + (FALLBACK_TRIGGERS as u16) * trigger;
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, refining);
        let still: Vec<HwFrameDecision> = (1..window)
            .map(|i| decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, refining))
            .collect();
        assert!(
            still.iter().all(|d| d.hold_qp.is_none() && !d.force_idr),
            "no frame held, no key frame"
        );
        let first = still.iter().position(|d| d.send).expect("a cleanup");
        assert_eq!(first + 1, trigger as usize, "due at the trigger");
        assert!(
            still[first..].iter().all(|d| d.send),
            "the frames keep flowing while it refines"
        );
        let d = decide_hw_fullframe(&mut st, &s, window, Damage::None, false, false, converged);
        assert!(
            !d.send,
            "and stop once it codes a small frame at the paint-over quality"
        );
        assert!((window + 1..window + 100).all(|i| {
            !decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, converged).send
        }));
    }

    /// A session that holds no quantizer has no refresh to fall back to: its cleanup flows
    /// through the rate control past the window a holding one is given, and ends after a
    /// trigger period of frames coded at the paint-over quantizer, whatever their size, or
    /// once its finest quantizer has stood for `LEVEL_S`.
    #[test]
    fn a_session_that_holds_no_quantizer_is_refined_until_it_reaches_the_paint_over_quality() {
        let s = hw_settings();
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps) as usize;
        let at = |last: u32| EncoderQuality {
            last: Some(last),
            bytes: Some(budget),
            holds: false,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let trigger = s.paint_over_trigger_frames as u16;
        let past = trigger + 2 * (FALLBACK_TRIGGERS as u16) * trigger;
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, at(40));
        let still: Vec<HwFrameDecision> = (1..past)
            .map(|i| decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, at(40)))
            .collect();
        assert!(still.iter().all(|d| d.hold_qp.is_none() && !d.force_idr));
        assert!(still[trigger as usize - 1..].iter().all(|d| d.send));
        let fine = s.video_paintover_crf as u32;
        assert!(
            (past..past + trigger).all(|i| {
                decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, at(fine)).send
            }),
            "a trigger period of frames at it"
        );
        assert!((past + trigger..past + 100).all(|i| {
            !decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, at(fine)).send
        }));
        let window = |encoder: EncoderQuality| {
            let mut st = StripeState::default();
            decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, encoder);
            (1..u16::MAX)
                .filter(|&i| {
                    decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, encoder).send
                })
                .count() as f64
                / s.target_fps
        };
        assert!(
            (window(at(40)) - LEVEL_S).abs() < 1.0,
            "a quantizer that stands ends it"
        );
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, at(44));
        let step = (2.0 * s.target_fps) as u32;
        let refining = (1..u16::MAX)
            .filter(|&i| {
                let q = 44 - (i as u32 / step).min(20);
                decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, at(q)).send
            })
            .count() as f64
            / s.target_fps;
        assert!(
            (refining - REFINE_S).abs() < 1.0,
            "one a step finer every two seconds runs the window"
        );
        let silent = EncoderQuality {
            last: None,
            ..at(40)
        };
        assert!(
            (window(silent) - CONVERGE_S).abs() < 1.0,
            "one that says no quantizer gets the shorter window"
        );
    }

    /// Such a session whose frames run small has nothing left to refine: the cleanup ends, and
    /// a caret blinking on the settled screen starts no other; motion does.
    #[test]
    fn a_settled_cleanup_is_not_restarted_by_a_caret() {
        let s = hw_settings();
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps) as usize;
        let small = EncoderQuality {
            last: None,
            bytes: Some(budget / 8),
            holds: false,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let trigger = s.paint_over_trigger_frames;
        let mut st = StripeState::default();
        let mut frame = 0u16;
        let mut run = |st: &mut StripeState, damage: Damage, frames: u32| {
            (0..frames)
                .filter(|_| {
                    frame += 1;
                    decide_hw_fullframe(st, &s, frame, damage, false, false, small).send
                })
                .count() as u32
        };
        run(&mut st, Damage::Area(1.0), 1);
        let cleanup = run(&mut st, Damage::None, 20 * trigger);
        assert_eq!(cleanup, STALL_TRIGGERS * trigger + 1);
        let mut blinking = 0;
        for _ in 0..10 {
            blinking += run(&mut st, Damage::Area(0.001), 1);
            blinking += run(&mut st, Damage::None, 2 * trigger);
        }
        assert_eq!(blinking, 20, "each blink and its refresh, and no more");
        run(&mut st, Damage::Area(1.0), 1);
        assert_eq!(run(&mut st, Damage::None, 20 * trigger), cleanup);
    }

    #[test]
    fn a_rate_control_that_does_not_converge_gets_one_held_refresh() {
        let s = hw_settings();
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps) as usize;
        let crawling = EncoderQuality {
            last: Some(38),
            bytes: Some(budget / 2),
            holds: true,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, crawling);
        let still: Vec<HwFrameDecision> = (1..=200)
            .map(|i| decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, crawling))
            .collect();
        assert!(still.iter().all(|d| !d.force_idr), "no key frame");
        let held: Vec<usize> = still
            .iter()
            .enumerate()
            .filter(|(_, d)| d.hold_qp.is_some())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(held.len(), 1, "{held:?}");
        let trigger = s.paint_over_trigger_frames as usize;
        assert_eq!(
            held[0] + 1,
            trigger + (FALLBACK_TRIGGERS as usize) * trigger,
            "after the window"
        );
        assert_eq!(
            still[held[0]].hold_qp,
            Some(held_refresh_quality(&s, crawling))
        );
        assert!(still[held[0] + 1..].iter().all(|d| !d.send));
    }

    /// A rate control pinned at its coarsest quantizer on a still screen, its frames many budgets
    /// each, is refreshed by one held frame at once, coarsened to what a second of the target buys
    /// at the bytes its frames take, and the cleanup ends: a band sweep would carry the rest of
    /// every frame at that quantizer.
    #[test]
    fn a_rate_control_pinned_at_its_coarsest_quantizer_is_refreshed_once() {
        let s = RustCaptureSettings {
            video_bitrate_kbps: 100,
            ..hw_settings()
        };
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps);
        let pinned = EncoderQuality {
            last: Some(51),
            bytes: Some((budget * 16.0) as usize),
            holds: true,
            reopens: false,
            keys: true,
            band: Some(0),
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, pinned);
        let still: Vec<HwFrameDecision> = (1..=200)
            .map(|i| decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, pinned))
            .collect();
        let sent: Vec<usize> = still
            .iter()
            .enumerate()
            .filter(|(_, d)| d.send)
            .map(|(i, _)| i)
            .collect();
        let trigger = s.paint_over_trigger_frames as usize;
        let refresh = 2 * trigger - 1;
        assert_eq!(
            sent,
            (trigger - 1..=refresh).collect::<Vec<_>>(),
            "a trigger period through the rate control, then the refresh"
        );
        assert_eq!(
            (still[refresh].hold_qp, still[refresh].hold_band),
            (Some(40), None),
            "frames of 16 budgets leave 11 steps of the 35 a second buys"
        );
    }

    /// A key frame asked for on a still screen with the paint-over off opens a burst through the
    /// rate control, which ends at once where the rate control is pinned at its coarsest quantizer.
    #[test]
    fn a_pinned_rate_control_ends_a_recovery_burst() {
        let s = RustCaptureSettings {
            video_bitrate_kbps: 100,
            use_paint_over_quality: false,
            ..hw_settings()
        };
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps);
        let pinned = EncoderQuality {
            last: Some(51),
            bytes: Some((budget * 16.0) as usize),
            holds: true,
            reopens: false,
            keys: true,
            band: Some(0),
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, pinned);
        assert!((1..=20).all(|i| {
            !decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, pinned).send
        }));
        let key = decide_hw_fullframe(&mut st, &s, 21, Damage::None, false, true, pinned);
        assert!(key.send && key.force_idr);
        let trigger = s.paint_over_trigger_frames as u16;
        assert!(
            (22..21 + trigger).all(|i| decide_hw_fullframe(
                &mut st,
                &s,
                i,
                Damage::None,
                false,
                false,
                pinned
            )
            .send),
            "its burst until a trigger period over the budget"
        );
        assert!(
            (21 + trigger..=200).all(|i| !decide_hw_fullframe(
                &mut st,
                &s,
                i,
                Damage::None,
                false,
                false,
                pinned
            )
            .send),
            "and no more"
        );
    }

    #[test]
    fn a_blinking_caret_does_not_hold_off_a_converging_cleanup() {
        let s = hw_settings();
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps) as usize;
        let crawling = EncoderQuality {
            last: Some(38),
            bytes: Some(budget / 2),
            holds: true,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, crawling);
        let trigger = s.paint_over_trigger_frames as u16;
        let decisions: Vec<HwFrameDecision> = (1..300)
            .map(|i| {
                let damage = if i % 3 == 0 {
                    Damage::Area(0.001)
                } else {
                    Damage::None
                };
                decide_hw_fullframe(&mut st, &s, i, damage, false, false, crawling)
            })
            .collect();
        assert!(decisions.iter().all(|d| !d.force_idr), "no key frame");
        let first = decisions
            .iter()
            .position(|d| d.hold_qp.is_some())
            .expect("a held refresh") as u16
            + 1;
        let bound = trigger * (LOW_MOTION_TRIGGERS + 2 * FALLBACK_TRIGGERS) as u16 * 3 / 2;
        assert!(
            first <= bound,
            "the caret does not hold it off: {first} > {bound}"
        );
        let held = decisions.iter().filter(|d| d.hold_qp.is_some()).count();
        assert!(
            held <= 300 / (trigger as usize * LOW_MOTION_TRIGGERS as usize),
            "{held} held refreshes"
        );
    }

    #[test]
    fn a_stalled_rate_control_is_held_once_at_the_paint_over_quantizer() {
        let s = hw_settings();
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps) as usize;
        let idle = EncoderQuality {
            last: Some(38),
            bytes: Some(budget / 10),
            holds: true,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, idle);
        let still: Vec<HwFrameDecision> = (1..=200)
            .map(|i| decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, idle))
            .collect();
        let held: Vec<usize> = still
            .iter()
            .enumerate()
            .filter(|(_, d)| d.hold_qp.is_some())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(held.len(), 1, "one held frame: {held:?}");
        assert_eq!(still[held[0]].hold_qp, Some(held_refresh_quality(&s, idle)));
        assert!(!still[held[0]].force_idr, "a refresh, not a key frame");
        let stall = (STALL_TRIGGERS * s.paint_over_trigger_frames) as usize;
        assert!(
            held[0] + 1 >= s.paint_over_trigger_frames as usize + stall,
            "after the stall, not at the trigger"
        );
        assert!(
            still[held[0] + 1..].iter().all(|d| !d.send),
            "which ends the cleanup"
        );
    }

    #[test]
    fn a_stalled_rate_control_that_holds_a_band_sweeps_the_picture_within_the_budget() {
        let s = hw_settings();
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps);
        let idle = |held: usize| EncoderQuality {
            last: Some(38),
            bytes: Some(budget as usize / 10),
            holds: true,
            reopens: false,
            keys: true,
            band: Some(held),
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, idle(0));
        let mut bands: Vec<(f64, f64)> = Vec::new();
        let mut sent_after = 0;
        for i in 1..=600u16 {
            let held = bands
                .last()
                .map_or(0, |(a, b)| ((b - a) * 128.0 * budget) as usize);
            let d = decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, idle(held));
            assert!(!d.force_idr, "no key frame");
            match d.hold_band {
                Some(band) => {
                    assert!(d.send && d.hold_qp == Some(held_refresh_quality(&s, idle(0))));
                    bands.push(band);
                }
                None => {
                    assert!(d.hold_qp.is_none(), "every held frame is a band");
                    sent_after += (!bands.is_empty() && d.send) as u32;
                }
            }
        }
        assert_eq!(
            bands[0],
            (0.0, FIRST_BAND),
            "the sweep starts at the stall, from the top"
        );
        assert!(
            bands.windows(2).all(|w| w[0].1 == w[1].0),
            "contiguous bands: {bands:?}"
        );
        assert_eq!(bands.last().unwrap().1, 1.0, "that cover the picture");
        assert!(
            bands[2..bands.len() - 1]
                .iter()
                .all(|(a, b)| ((b - a) * 128.0 - 1.0).abs() < 1e-3),
            "each a budget: {bands:?}"
        );
        assert_eq!(sent_after, 0, "which ends the cleanup");
    }

    /// A session that holds its refresh rather than converging to it, and holds a band of a frame
    /// (libvpx's VP8), sweeps that refresh a band a frame from the top, each a budget, until it
    /// covers the picture, with no whole held frame; a caret blinking meanwhile leaves the sweep
    /// to finish.
    #[test]
    fn a_held_refresh_that_holds_a_band_sweeps_the_picture() {
        let s = hw_settings();
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps);
        let vp8 = |held: usize| EncoderQuality {
            last: Some(38),
            bytes: None,
            holds: true,
            reopens: false,
            keys: false,
            band: Some(held),
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, vp8(0));
        let mut bands: Vec<(f64, f64)> = Vec::new();
        let mut caret_in_sweep = false;
        let mut blinked = false;
        for i in 1..=600u16 {
            let held = bands
                .last()
                .map_or(0, |(a, b)| ((b - a) * 128.0 * budget) as usize);
            let caret = !blinked && bands.len() == 20;
            blinked |= caret;
            caret_in_sweep |= caret && st.sweep.is_some();
            let damage = if caret {
                Damage::Area(0.001)
            } else {
                Damage::None
            };
            let d = decide_hw_fullframe(&mut st, &s, i, damage, false, false, vp8(held));
            assert!(!d.force_idr, "no key frame");
            match d.hold_band {
                Some(band) => {
                    assert!(d.send && d.hold_qp == Some(held_refresh_quality(&s, vp8(0))));
                    bands.push(band);
                }
                None => assert!(d.hold_qp.is_none(), "no whole held frame, frame {i}"),
            }
        }
        assert!(caret_in_sweep, "the caret blinked mid-sweep");
        assert_eq!(bands[0], (0.0, FIRST_BAND), "the sweep starts from the top");
        assert!(
            bands.windows(2).all(|w| w[0].1 == w[1].0),
            "contiguous bands, the caret's frame between two: {bands:?}"
        );
        assert_eq!(bands.last().unwrap().1, 1.0, "that cover the picture");
    }

    /// A sweep whose bands, each a budget, would take longer than `REFINE_S` to cover the picture
    /// is coded a step coarser a band until its pace fits: here a picture of four times as many
    /// budgets at the refresh's quantizer, half that six steps coarser.
    #[test]
    fn a_sweep_that_would_run_past_refine_s_is_coded_coarser() {
        let s = hw_settings();
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps);
        let idle = |held: usize| EncoderQuality {
            last: Some(38),
            bytes: Some(budget as usize / 10),
            holds: true,
            reopens: false,
            keys: true,
            band: Some(held),
            measures: false,
            psnr: None,
        };
        let refresh = held_refresh_quality(&s, idle(0));
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, idle(0));
        let window = (REFINE_S * s.target_fps).round() as usize;
        let mut bands: Vec<(f64, f64, u32)> = Vec::new();
        for i in 1..=(2 * window) as u16 {
            let held = bands.last().map_or(0, |(a, b, q)| {
                let cost = 4.0 * window as f64 * 2f64.powf(-((q - refresh) as f64) / 6.0);
                ((b - a) * cost * budget) as usize
            });
            let d = decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, idle(held));
            if let (Some((a, b)), Some(q)) = (d.hold_band, d.hold_qp) {
                bands.push((a, b, q));
            }
        }
        let coarser = bands.last().unwrap().2 - refresh;
        assert_eq!(bands.last().unwrap().1, 1.0, "the sweep covers the picture");
        assert!(
            bands.len() <= window + window / 10,
            "within REFINE_S: {} bands",
            bands.len()
        );
        assert!(
            bands.windows(2).all(|w| w[0].2 <= w[1].2),
            "never finer again"
        );
        assert!(
            (12..=15).contains(&coarser),
            "about the 12 steps that quarter the picture's cost: {coarser}"
        );
    }

    /// A session that holds no quantizer and measures nothing, whose frames of a still screen keep
    /// the size of the one before (libvpx's VP9 at 0.1 Mbit/s, over the budget and so never
    /// small), ends its cleanup after `EMPTY_S` of them rather than `REFINE_S`.
    #[test]
    fn a_cleanup_whose_frames_keep_one_size_ends() {
        let s = RustCaptureSettings {
            video_bitrate_kbps: 100,
            ..hw_settings()
        };
        let level = EncoderQuality {
            last: Some(51),
            bytes: Some(266),
            holds: false,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, level);
        let sent = (1..=2400u16)
            .filter(|&i| {
                decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, level).send
            })
            .count();
        let run = (EMPTY_S * s.target_fps) as usize;
        assert!(
            (run..=run + 4).contains(&sent),
            "{sent} frames, not the {} of REFINE_S",
            (REFINE_S * s.target_fps) as usize
        );
    }

    #[test]
    fn motion_ends_a_band_sweep() {
        let s = hw_settings();
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps) as usize;
        let idle = EncoderQuality {
            last: Some(38),
            bytes: Some(budget / 10),
            holds: true,
            reopens: false,
            keys: true,
            band: Some(budget),
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &s, 0, Damage::Area(1.0), false, false, idle);
        let first = (1..=200u16)
            .find(|&i| {
                decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, idle)
                    .hold_band
                    .is_some()
            })
            .expect("a sweep");
        let moving = decide_hw_fullframe(
            &mut st,
            &s,
            first + 1,
            Damage::Area(0.5),
            false,
            false,
            idle,
        );
        assert!(moving.send && moving.hold_qp.is_none() && moving.hold_band.is_none());
        assert!(st.sweep.is_none());
        let caret = decide_hw_fullframe(
            &mut st,
            &s,
            first + 2,
            Damage::Area(0.001),
            false,
            false,
            idle,
        );
        assert!(caret.hold_band.is_none());
    }

    #[test]
    fn a_recovery_key_frame_is_refined_until_the_rate_control_converges() {
        let s = hw_settings();
        let budget = frame_budget(s.video_bitrate_kbps, s.target_fps) as usize;
        let coarse = EncoderQuality {
            last: Some(40),
            bytes: Some(budget),
            holds: true,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let clean = EncoderQuality {
            last: Some(8),
            bytes: Some(budget / 8),
            holds: true,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        assert!(
            (0..200).all(|i| !decide_hw_fullframe(
                &mut st,
                &s,
                i,
                Damage::None,
                false,
                false,
                clean
            )
            .send)
        );
        let join = decide_hw_fullframe(&mut st, &s, 200, Damage::None, false, true, clean);
        assert!(join.send && join.force_idr && join.hold_qp.is_none());
        let past_burst = 201 + s.video_paintover_burst_frames as u16 + 3;
        assert!(
            (201..past_burst).all(|i| {
                let d = decide_hw_fullframe(&mut st, &s, i, Damage::None, false, false, coarse);
                d.send && !d.force_idr && d.hold_qp.is_none()
            }),
            "the rate control keeps refining the key frame past the configured burst"
        );
        assert!(
            !decide_hw_fullframe(&mut st, &s, past_burst, Damage::None, false, false, clean).send,
            "until it converges"
        );
    }

    #[test]
    fn a_large_change_after_a_clean_still_screen_is_not_keyed_as_it_lands() {
        let s = RustCaptureSettings {
            video_paintover_burst_frames: 0,
            ..hw_settings()
        };
        let fine = EncoderQuality {
            last: Some(5),
            bytes: None,
            holds: true,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let coarse = EncoderQuality {
            last: Some(40),
            bytes: None,
            holds: true,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        let mut frame = 0u16;
        let mut step = |st: &mut StripeState, damage: Damage, q: EncoderQuality| {
            frame = frame.wrapping_add(1);
            decide_hw_fullframe(st, &s, frame, damage, false, false, q)
        };
        step(&mut st, Damage::Area(1.0), coarse);
        assert!(
            (0..100).all(|_| step(&mut st, Damage::None, fine).hold_qp.is_none()),
            "a finer rate control is left alone"
        );
        let landing = [
            step(&mut st, Damage::Area(1.0), fine),
            step(&mut st, Damage::Area(1.0), coarse),
        ];
        assert!(
            landing
                .iter()
                .all(|d| d.send && !d.force_idr && d.hold_qp.is_none()),
            "the change goes out at the session quality"
        );
        assert!(
            (0..12).any(|_| step(&mut st, Damage::None, coarse).force_idr),
            "and is keyed once the screen holds still"
        );
    }

    /// A screen that never holds still for the trigger (a caret blinking every other frame)
    /// still gets its cleanup, once the low-motion window passes, and the rest of the screen
    /// with it.
    #[test]
    fn low_motion_does_not_hold_off_the_cleanup_forever() {
        let s = RustCaptureSettings {
            paint_over_trigger_frames: 4,
            ..hw_settings()
        };
        let mut st = StripeState::default();
        run(&mut st, &s, Damage::Area(1.0), 3);
        let mut fired = None;
        for i in 0..40u16 {
            let damage = if i % 2 == 0 {
                Damage::Area(0.0005)
            } else {
                Damage::None
            };
            let d = decide_hw_fullframe(&mut st, &s, 100 + i, damage, false, false, HOLDS);
            if d.hold_qp.is_some() {
                fired = Some((i, d.force_idr));
                break;
            }
        }
        let (at, key) = fired.expect("a cleanup under low motion");
        assert_eq!(
            at as u32 + 4,
            s.paint_over_trigger_frames * LOW_MOTION_TRIGGERS,
            "at the end of the window since the scroll began"
        );
        assert!(key, "the scroll before it makes it a key frame");

        // After a long scroll the caret gets the cleanup once the scroll has aged out of the
        // window, not only after a quarter of every frame since it began.
        let mut st = StripeState::default();
        run(&mut st, &s, Damage::Area(1.0), 200);
        let window = s.paint_over_trigger_frames * LOW_MOTION_TRIGGERS;
        let fired = (0..200u16).find(|&i| {
            let damage = if i % 2 == 0 {
                Damage::Area(0.0005)
            } else {
                Damage::None
            };
            decide_hw_fullframe(&mut st, &s, 300 + i, damage, false, false, HOLDS).force_idr
        });
        let at = fired.expect("a key frame under low motion after a long scroll") as u32;
        assert!(
            at >= window && at <= window * 2,
            "{at} frames after the scroll stopped, window {window}"
        );
    }

    /// Sustained motion is not low motion: a screen changing every frame gets no cleanup until
    /// it stops.
    #[test]
    fn sustained_motion_gets_no_cleanup_until_it_stops() {
        let s = hw_settings();
        let mut st = StripeState::default();
        assert!(
            run(&mut st, &s, Damage::Area(0.5), 60)
                .iter()
                .all(|d| d.hold_qp.is_none())
        );
        assert!(
            run(&mut st, &s, Damage::Unknown, 60)
                .iter()
                .all(|d| d.hold_qp.is_none())
        );
    }

    /// A change of unknown extent counts toward a key frame only in runs: isolated ones (a
    /// caret on a capture that cannot say where it changed) are refreshed, never keyed.
    #[test]
    fn unknown_changes_count_toward_a_key_frame_only_in_runs() {
        let s = hw_settings();
        let mut st = StripeState::default();
        for _ in 0..20 {
            run(&mut st, &s, Damage::Unknown, 1);
            let still = run(&mut st, &s, Damage::None, 10);
            assert!(
                still[1].hold_qp.is_some() && !still[1].force_idr,
                "isolated changes are refreshed"
            );
            assert!(still.iter().all(|d| !d.force_idr), "and never keyed");
        }
        run(&mut st, &s, Damage::Unknown, 12);
        assert!(
            run(&mut st, &s, Damage::None, 10)
                .iter()
                .any(|d| d.force_idr),
            "a run of ten counts for more than half a screen"
        );
    }

    /// Motion outranks the burst: it sends at normal quality, drops a burst in flight, and forces
    /// no keyframe of its own.
    #[test]
    fn motion_sends_at_normal_quality_and_cancels_the_burst() {
        let s = hw_settings();
        let mut st = StripeState {
            h264_burst_frames_remaining: 3,
            paint_over_sent: true,
            no_motion_frame_count: 9,
            ..Default::default()
        };
        let d = decide_hw_fullframe(&mut st, &s, 1, Damage::Area(0.2), false, false, HOLDS);
        assert!(d.send && !d.force_idr && d.hold_qp.is_none());
        assert_eq!(d.target_qp, s.video_crf as u32);
        assert_eq!(st.h264_burst_frames_remaining, 0, "the burst is dropped");
        assert_eq!(st.no_motion_frame_count, 0);
        assert!(!st.paint_over_sent);
    }

    /// A requested key frame is coded at the rate control's own quality whatever the screen does:
    /// it answers a join or a loss, often on a link that fell behind, where a larger key frame
    /// would only fall behind again. On a still screen of a constant-quality session the burst
    /// behind it refines at the paint-over quality, a clean screen's included; under a constant
    /// rate it is not held.
    #[test]
    fn a_requested_key_frame_is_the_rate_controls_own() {
        let s = RustCaptureSettings {
            paint_over_trigger_frames: 30,
            ..crf_settings()
        };
        let (normal, paint) = (s.video_crf as u32, s.video_paintover_crf as u32);
        let mut st = StripeState::default();
        run(&mut st, &s, Damage::Area(1.0), 1);
        let join = decide_hw_fullframe(&mut st, &s, 2, Damage::None, false, true, HOLDS);
        assert!(join.send && join.force_idr && join.hold_qp.is_none() && join.target_qp == normal);
        assert_eq!(
            st.h264_burst_frames_remaining,
            s.video_paintover_burst_frames
        );
        let burst = decide_hw_fullframe(&mut st, &s, 3, Damage::None, false, false, HOLDS);
        assert!(
            burst.send && burst.hold_qp.is_none() && burst.target_qp == paint,
            "the burst refines at the paint-over quality"
        );

        let s = crf_settings();
        let mut st = StripeState::default();
        run(&mut st, &s, Damage::Area(1.0), 1);
        run(&mut st, &s, Damage::None, 12);
        assert!(st.clean_quality, "cleaned up");
        let join = decide_hw_fullframe(&mut st, &s, 30, Damage::None, false, true, HOLDS);
        assert!(
            join.force_idr && join.target_qp == normal,
            "a key frame on a clean screen at the session's quality"
        );
        let burst = decide_hw_fullframe(&mut st, &s, 31, Damage::None, false, false, HOLDS);
        assert_eq!(burst.target_qp, paint);

        let cbr = RustCaptureSettings {
            video_cbr_mode: true,
            paint_over_trigger_frames: 30,
            ..hw_settings()
        };
        let coarse = EncoderQuality {
            last: Some(40),
            bytes: None,
            holds: true,
            reopens: false,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        decide_hw_fullframe(&mut st, &cbr, 1, Damage::Area(1.0), false, false, coarse);
        let join = decide_hw_fullframe(&mut st, &cbr, 2, Damage::None, false, true, coarse);
        assert!(join.force_idr && join.hold_qp.is_none());
        let burst = decide_hw_fullframe(&mut st, &cbr, 3, Damage::None, false, false, coarse);
        assert!(
            burst.send && burst.hold_qp.is_none(),
            "a constant rate refines the burst itself"
        );

        let mut moving = StripeState::default();
        let dirty = decide_hw_fullframe(&mut moving, &s, 1, Damage::Area(1.0), false, true, HOLDS);
        assert!(dirty.send && dirty.force_idr && dirty.hold_qp.is_none());
        assert_eq!(
            moving.h264_burst_frames_remaining, 0,
            "motion carries the refinement, so no burst is opened"
        );
    }

    /// At a constant quality the cleanup applies where the paint-over quality is finer than the
    /// session's, through the session's quality, whatever the encoder holds. At a constant rate
    /// it applies where the paint-over quality is finer than what the rate control last coded
    /// at, or the encoder does not say: held where the encoder holds a quantizer, and otherwise a
    /// refresh and its burst at the rate control's own quality, never a key frame. Never with
    /// paint-over off.
    #[test]
    fn the_cleanup_follows_the_rate_control_and_the_switch() {
        #[derive(Debug, PartialEq)]
        enum Seen {
            Held,
            Session,
            Unheld,
            Nothing,
        }
        for (cbr, paint, on, last, holds, seen) in [
            (true, 25, true, None, true, Seen::Held),
            (true, 18, true, Some(30), true, Seen::Held),
            (true, 18, true, Some(9), true, Seen::Nothing),
            (true, 18, true, Some(18), true, Seen::Nothing),
            (true, 18, true, Some(30), false, Seen::Unheld),
            (true, 18, true, None, false, Seen::Unheld),
            (true, 18, true, Some(9), false, Seen::Nothing),
            (false, 25, true, None, true, Seen::Nothing),
            (false, 18, true, None, true, Seen::Session),
            (false, 18, true, None, false, Seen::Session),
            (false, 18, false, None, true, Seen::Nothing),
            (true, 18, false, None, true, Seen::Nothing),
        ] {
            let s = RustCaptureSettings {
                video_cbr_mode: cbr,
                video_paintover_crf: paint,
                use_paint_over_quality: on,
                ..hw_settings()
            };
            let mut st = StripeState::default();
            let mut got = Seen::Nothing;
            for i in 0..40u16 {
                let damage = if i == 0 {
                    Damage::Area(1.0)
                } else {
                    Damage::None
                };
                let d = decide_hw_fullframe(
                    &mut st,
                    &s,
                    i,
                    damage,
                    false,
                    false,
                    EncoderQuality {
                        last,
                        bytes: None,
                        holds,
                        reopens: false,
                        keys: true,
                        band: None,
                        measures: false,
                        psnr: None,
                    },
                );
                if i > 0 && d.send {
                    assert!(
                        !cbr || holds || !d.force_idr,
                        "a key frame where no quantizer is held: last={last:?}"
                    );
                    if d.hold_qp.is_some() {
                        got = Seen::Held;
                    } else if d.target_qp != s.video_crf as u32 {
                        assert!(!cbr, "a constant rate keeps its session quality");
                        got = Seen::Session;
                    } else if got == Seen::Nothing {
                        got = Seen::Unheld;
                    }
                }
            }
            assert_eq!(
                got, seen,
                "cbr={cbr} paint={paint} on={on} last={last:?} holds={holds}"
            );
        }
    }

    /// At a constant quality the cleanup moves the session's quality to the paint-over one,
    /// holding no frame: the refresh at the trigger and its burst, the key frame four trigger
    /// periods in and its burst, and under Turbo every frame from the refresh on, since a region
    /// stays at the paint-over quality until it moves again.
    #[test]
    fn a_constant_quality_cleanup_moves_the_session_quality() {
        for turbo in [false, true] {
            let s = RustCaptureSettings {
                video_streaming_mode: turbo,
                ..crf_settings()
            };
            let (normal, paint) = (s.video_crf as u32, s.video_paintover_crf as u32);
            let mut st = StripeState::default();
            let moving = run(&mut st, &s, Damage::Area(1.0), 3);
            assert!(
                moving.iter().all(|d| d.send
                    && !d.force_idr
                    && d.hold_qp.is_none()
                    && d.target_qp == normal)
            );
            let still = run(&mut st, &s, Damage::None, 16);
            assert!(
                still.iter().all(|d| d.hold_qp.is_none()),
                "no frame is held"
            );
            let sent: Vec<(usize, bool, u32)> = still
                .iter()
                .enumerate()
                .filter(|(_, d)| d.send)
                .map(|(i, d)| (i + 1, d.force_idr, d.target_qp))
                .collect();
            if !turbo {
                let cleanups = vec![
                    (2, false, paint),
                    (3, false, paint),
                    (4, false, paint),
                    (5, false, paint),
                    (8, true, paint),
                    (9, false, paint),
                    (10, false, paint),
                    (11, false, paint),
                ];
                assert_eq!(
                    sent, cleanups,
                    "a refresh at the trigger and a key four triggers in, each with its burst"
                );
            } else {
                assert_eq!(sent.len(), 16, "Turbo sends every frame");
                assert_eq!(
                    sent[0],
                    (1, false, normal),
                    "the session's quality until the cleanup"
                );
                assert!(
                    sent[1..].iter().all(|r| r.2 == paint),
                    "the paint-over quality from the refresh on: {sent:?}"
                );
                assert_eq!(
                    sent.iter().filter(|r| r.1).map(|r| r.0).collect::<Vec<_>>(),
                    vec![8],
                    "one key frame"
                );
            }
        }
    }

    /// An encoder whose quality change re-opens it codes the refresh that moves it as a key
    /// frame, which then counts as the key-frame cleanup: one key frame for one change, as main's
    /// paint-over sent, where the four-trigger key frame would have been a second.
    #[test]
    fn a_reopening_encoder_counts_its_refresh_as_the_key_frame() {
        let s = crf_settings();
        let reopens = EncoderQuality {
            last: None,
            bytes: None,
            holds: true,
            reopens: true,
            keys: true,
            band: None,
            measures: false,
            psnr: None,
        };
        let mut st = StripeState::default();
        for i in 0..3u16 {
            decide_hw_fullframe(&mut st, &s, i, Damage::Area(1.0), false, false, reopens);
        }
        let keys: Vec<usize> = (0..16u16)
            .map(|i| decide_hw_fullframe(&mut st, &s, 10 + i, Damage::None, false, false, reopens))
            .enumerate()
            .filter(|(_, d)| d.force_idr)
            .map(|(i, _)| i + 1)
            .collect();
        assert_eq!(
            keys,
            vec![2],
            "the refresh at the trigger is the key frame, and the only one"
        );
    }

    /// A clean region keeps the paint-over quality through a small change (a caret), so an
    /// encoder whose quality change re-opens it does not go back and forth; motion, or a change
    /// of unknown extent, takes it back to the session's quality until its next cleanup.
    #[test]
    fn a_clean_region_keeps_its_quality_until_it_moves() {
        let s = RustCaptureSettings {
            video_streaming_mode: true,
            ..crf_settings()
        };
        let (normal, paint) = (s.video_crf as u32, s.video_paintover_crf as u32);
        let mut st = StripeState::default();
        run(&mut st, &s, Damage::Area(1.0), 3);
        run(&mut st, &s, Damage::None, 12);
        let caret = decide_hw_fullframe(&mut st, &s, 100, Damage::Area(0.001), false, false, HOLDS);
        assert_eq!(caret.target_qp, paint, "a caret keeps it");
        assert_eq!(
            decide_hw_fullframe(&mut st, &s, 101, Damage::None, false, false, HOLDS).target_qp,
            paint
        );
        let scroll = decide_hw_fullframe(&mut st, &s, 102, Damage::Area(0.5), false, false, HOLDS);
        assert_eq!(scroll.target_qp, normal, "motion takes it back");
        assert_eq!(
            decide_hw_fullframe(&mut st, &s, 103, Damage::None, false, false, HOLDS).target_qp,
            normal,
            "until the next cleanup"
        );
        run(&mut st, &s, Damage::None, 12);
        assert_eq!(
            decide_hw_fullframe(&mut st, &s, 120, Damage::Unknown, false, false, HOLDS).target_qp,
            normal
        );

        let off = RustCaptureSettings {
            use_paint_over_quality: false,
            ..s
        };
        let mut st = StripeState::default();
        run(&mut st, &off, Damage::Area(1.0), 3);
        assert!(
            run(&mut st, &off, Damage::None, 20)
                .iter()
                .all(|d| d.target_qp == normal),
            "paint-over off"
        );
    }

    /// Streaming and animated modes send every frame, and a scheduled interval forces the
    /// keyframe `periodic_idr_due` picks even with nothing else happening.
    #[test]
    fn hw_fullframe_streaming_and_scheduled_keyframes_send_unconditionally() {
        let mut s = hw_settings();
        s.video_streaming_mode = true;
        let mut st = StripeState::default();
        assert!(
            decide_hw_fullframe(&mut st, &s, 1, Damage::None, false, false, HOLDS).send,
            "streaming mode sends a static frame"
        );

        s.video_streaming_mode = false;
        let mut animated = StripeState::default();
        assert!(
            decide_hw_fullframe(&mut animated, &s, 1, Damage::None, true, false, HOLDS).send,
            "an animated overlay sends a static frame"
        );

        // 1 s at 60 fps: frame 60 is due, 61 is not.
        s.keyframe_interval_s = 1.0;
        let mut scheduled = StripeState::default();
        assert!(
            decide_hw_fullframe(&mut scheduled, &s, 60, Damage::None, false, false, HOLDS)
                .force_idr
        );
        let mut off_beat = StripeState::default();
        assert!(
            !decide_hw_fullframe(&mut off_beat, &s, 61, Damage::None, false, false, HOLDS)
                .force_idr
        );
    }

    /// Software JPEG path emits on change and stays silent while static: the first frame
    /// sends every stripe (all dirty vs init), an identical static frame sends nothing, and a
    /// frame with changed top rows re-sends the dirty stripes. `use_cpu` forces the software path
    /// and paint-over is disabled to keep the static-frame assertions clean.
    #[test]
    fn x11_software_emits_on_change_and_stays_quiet_when_static() {
        let s = RustCaptureSettings {
            width: 128,
            height: 128,
            codec: Codec::Jpeg,
            use_cpu: true,
            jpeg_quality: 60,
            use_paint_over_quality: false,
            ..Default::default()
        };
        let mut p = X11Pipeline::new(s);
        let stride = 128 * 4;
        let frame_a = vec![10u8; stride * 128];
        let mut frame_b = frame_a.clone();
        for px in frame_b.iter_mut().take(stride * 40) {
            *px = 200;
        }
        let n1 = p.process(&frame_a, stride).len();
        let n2 = p.process(&frame_a, stride).len();
        let n3 = p.process(&frame_b, stride).len();
        assert!(
            n1 > 0,
            "first frame should emit (all stripes dirty vs init)"
        );
        assert_eq!(n2, 0, "identical static frame should emit nothing");
        assert!(n3 > 0, "changed frame should emit dirty stripes");
    }

    /// Software H.264 (the build's encoder), paint-over off, non-streaming: a requested IDR on
    /// a static screen is followed by a short recovery burst so rate control can refine the
    /// keyframe, instead of the stream going silent and stranding an unrefined keyframe. The
    /// stream goes quiet again once the burst ends.
    #[test]
    fn x11_software_h264_streams_recovery_burst_after_requested_idr() {
        let s = RustCaptureSettings {
            width: 128,
            height: 128,
            codec: Codec::H264,
            use_cpu: true,
            video_crf: 25,
            video_paintover_burst_frames: 5,
            use_paint_over_quality: false,
            video_streaming_mode: false,
            target_fps: 60.0,
            ..Default::default()
        };
        let mut p = X11Pipeline::new(s);
        let stride = 128 * 4;
        let frame = vec![10u8; stride * 128];
        assert!(!p.process(&frame, stride).is_empty(), "first frame emits");
        for _ in 0..4 {
            let _ = p.process(&frame, stride);
        }
        assert!(
            p.process(&frame, stride).is_empty(),
            "static screen is quiet before the request"
        );
        p.request_idr();
        assert!(
            !p.process(&frame, stride).is_empty(),
            "requested IDR emits on a static screen"
        );
        for i in 0..5 {
            assert!(
                !p.process(&frame, stride).is_empty(),
                "recovery burst frame {i} streams while static"
            );
        }
        assert!(
            p.process(&frame, stride).is_empty(),
            "stream goes quiet again after the recovery burst"
        );
    }

    /// A noise screen of 640x384 at `seed`: a rate control codes it still coarser than a cleanup's
    /// quantizer.
    fn noise(seed: u8) -> Vec<u8> {
        (0..640 * 4 * 384)
            .map(|i| {
                (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed as u32 * 97) as u8
            })
            .collect()
    }

    /// A still screen x264 refines nothing more of at a constant rate, its frames slice headers
    /// alone, ends its cleanup after `EMPTY_S` of them rather than `CONVERGE_S`. The quantizer
    /// floor holds x264 at its coarsest, as at 0.1 Mbit/s, on every run and thread count.
    #[cfg(feature = "gpl")]
    #[test]
    fn an_x264_cleanup_whose_frames_refine_nothing_ends() {
        let mut p = X11Pipeline::new(RustCaptureSettings {
            width: 640,
            height: 384,
            codec: Codec::H264,
            use_cpu: true,
            video_fullframe: true,
            video_cbr_mode: true,
            video_bitrate_kbps: 100,
            video_min_qp: 51,
            video_paintover_crf: 18,
            paint_over_trigger_frames: 15,
            use_paint_over_quality: true,
            target_fps: 60.0,
            ..Default::default()
        });
        for i in 0..12u8 {
            p.process(&noise(i), 640 * 4);
        }
        let window = converge_frames(&p.settings) as usize;
        let frames: Vec<Vec<usize>> = (0..window + 60)
            .map(|_| {
                p.process(&noise(42), 640 * 4)
                    .iter()
                    .map(|e| e.data.len())
                    .collect()
            })
            .collect();
        let last = frames
            .iter()
            .rposition(|f| !f.is_empty())
            .expect("a cleanup");
        let empty = crate::encoders::codec::VIDEO_HEADER_LEN + 4 * EMPTY_SLICE_BYTES;
        let run = (EMPTY_S * 60.0) as usize;
        assert!(last < window, "the cleanup ends at {last} of {window}");
        assert!(
            frames[last + 1 - run..=last]
                .iter()
                .flatten()
                .all(|&b| b <= empty),
            "on frames that carry nothing: {:?}",
            &frames[last + 1 - run..=last]
        );
    }

    /// The probes (`X11Pipeline::hold`) of a Turbo session's first still frame coded at a
    /// cleanup's quality, as a key frame where `key`, and of the same frame in a second session
    /// where the screen moves on it.
    fn cleanup_frame_that_moved(
        s: &RustCaptureSettings,
        key: bool,
    ) -> [[(Option<u32>, bool); 2]; 2] {
        let still = noise(42);
        let lead = |p: &mut X11Pipeline| {
            for i in 0..3u8 {
                p.process(&noise(i), 640 * 4);
            }
        };
        let mut p = X11Pipeline::new(s.clone());
        lead(&mut p);
        let at = (0..300)
            .find(|_| {
                p.process(&still, 640 * 4);
                p.hold[0].0.is_some() && p.hold[0].1 == key
            })
            .expect("a still screen is cleaned up");
        let mut q = X11Pipeline::new(s.clone());
        lead(&mut q);
        for _ in 0..at {
            q.process(&still, 640 * 4);
        }
        assert!(
            !q.process(&noise(9), 640 * 4).is_empty(),
            "the moved frame is sent"
        );
        [p.hold, q.hold]
    }

    /// A Turbo frame is decided on the hash of the frame before it, so a change can land on a
    /// frame decided to be held whole at the cleanup's quantizer (SVT-AV1's held frames, from the
    /// release that takes a new target with a picture). That frame is hashed before it is
    /// encoded and goes to the rate control, a refresh or a key frame alike.
    #[test]
    fn a_turbo_frame_decided_held_that_moved_goes_to_the_rate_control() {
        let s = RustCaptureSettings {
            width: 640,
            height: 384,
            codec: Codec::Av1,
            use_cpu: true,
            video_cbr_mode: true,
            video_bitrate_kbps: 100,
            video_crf: 25,
            video_paintover_crf: 18,
            paint_over_trigger_frames: 15,
            use_paint_over_quality: true,
            video_streaming_mode: true,
            target_fps: 60.0,
            ..Default::default()
        };
        let p = X11Pipeline::new(s.clone());
        assert!(p.hw.is_some(), "AV1 runs as a full-frame session");
        if !codec_sys::svtav1::HAS_EVENTS {
            assert!(!EncoderQuality::of(p.hw.as_ref().unwrap()).holds);
            return;
        }
        for key in [false, true] {
            let [still, moved] = cleanup_frame_that_moved(&s, key);
            assert_eq!(
                still[1], still[0],
                "a frame that stood still keeps its hold"
            );
            assert!(moved[0].0.is_some() && moved[0].1 == key, "{moved:?}");
            assert_eq!(
                moved[1],
                (None, false),
                "a held frame that moved is coded under the rate control"
            );
        }
    }

    /// The same at a constant quality, whose cleanup moves the session's quality to the
    /// paint-over one while the screen stays clean (`decide_constant_quality`): a frame that
    /// moved is coded at the session's own.
    #[test]
    fn a_turbo_frame_at_the_paint_over_quality_that_moved_is_coded_at_the_session_quality() {
        let s = RustCaptureSettings {
            width: 640,
            height: 384,
            codec: Codec::Vp9,
            use_cpu: true,
            video_cbr_mode: false,
            video_crf: 25,
            video_paintover_crf: 18,
            paint_over_trigger_frames: 15,
            use_paint_over_quality: true,
            video_streaming_mode: true,
            target_fps: 60.0,
            ..Default::default()
        };
        assert!(
            X11Pipeline::new(s.clone()).hw.is_some(),
            "VP9 runs as a full-frame session"
        );
        for key in [false, true] {
            let [still, moved] = cleanup_frame_that_moved(&s, key);
            assert_eq!(
                still,
                [(Some(18), key); 2],
                "a frame that stood still keeps it"
            );
            assert_eq!(
                moved,
                [(Some(18), key), (None, false)],
                "a frame that moved does not"
            );
        }
    }

    /// The striped path decides each stripe's cleanup inside its encode (`encode_cpu`), so a
    /// Turbo frame that codes a stripe of a constant-quality session at the paint-over quality
    /// while it holds still (`software::stripes_held_still`) is hashed before it is encoded: a
    /// change landing on its refresh, on a frame after it, or on its key frame is coded at the
    /// session's quality, as without Turbo.
    #[test]
    fn a_turbo_stripe_at_the_paint_over_quality_codes_a_change_at_the_session_quality() {
        use crate::encoders::codec::{FRAME_KEY, parse_video_type};
        let session = |turbo: bool| {
            let mut p = X11Pipeline::new(RustCaptureSettings {
                width: 640,
                height: 384,
                codec: Codec::H264,
                use_cpu: true,
                video_fullframe: true,
                video_cbr_mode: false,
                video_crf: 25,
                video_paintover_crf: 18,
                paint_over_trigger_frames: 15,
                use_paint_over_quality: true,
                video_streaming_mode: turbo,
                target_fps: 60.0,
                ..Default::default()
            });
            assert!(p.hw.is_none(), "software H.264 runs striped");
            for i in 0..12u8 {
                p.process(&noise(i), 640 * 4);
            }
            p
        };
        let mut p = session(true);
        let (mut held, mut key) = (None, None);
        for i in 0..300 {
            if held.is_none() && stripes_held_still(&p.stripes, &p.settings) {
                held = Some(i);
            }
            let out = p.process(&noise(42), 640 * 4);
            let keyed = out
                .iter()
                .any(|e| parse_video_type(e.data[1]).is_some_and(|(_, k)| k == FRAME_KEY));
            if held.is_some() && keyed {
                key = Some(i);
                break;
            }
        }
        let held = held.expect("a still stripe is cleaned up");
        let key = key.expect("a large change is cleaned up by a key frame");
        for at in [held, held + 15, key] {
            let jump = |turbo: bool| -> usize {
                let mut p = session(turbo);
                for _ in 0..at {
                    p.process(&noise(42), 640 * 4);
                }
                p.process(&noise(9), 640 * 4)
                    .iter()
                    .map(|e| e.data.len())
                    .sum()
            };
            let (turbo, plain) = (jump(true), jump(false));
            assert!(
                turbo as f64 <= plain as f64 * 1.2,
                "a change {at} frames still is coded at the session's quality: {turbo} bytes \
                 against {plain}"
            );
        }
    }

    /// The software path carries a 4:4:4 request exactly when the build's encoder does —
    /// libx264 at full range, so its log line says so; OpenH264 never, reporting the 4:2:0 it
    /// encodes — and without the request it reports 4:2:0. This is the same string the Wayland log
    /// builds from the same shared helper, so an identical session reads identically on both
    /// backends.
    #[test]
    fn x11_colorspace_desc_reports_what_the_software_encoder_carries() {
        let carries_444 = crate::encoders::software_fullcolor(Codec::H264);
        assert_eq!(
            carries_444,
            crate::encoders::software_library(Codec::H264) == "x264"
        );
        let i444 = if carries_444 {
            "I444 (Full Range)"
        } else {
            "I420 (Limited Range)"
        };
        for (fullcolor, expected) in [(true, i444), (false, "I420 (Limited Range)")] {
            let settings = RustCaptureSettings {
                width: 64,
                height: 64,
                codec: Codec::H264,
                use_cpu: true,
                video_fullcolor: fullcolor,
                ..Default::default()
            };
            let p = X11Pipeline::new(settings.clone());
            assert_eq!(
                p.encoder_name(),
                format!("CPU ({})", crate::encoders::software_library(Codec::H264))
            );
            assert_eq!(p.colorspace_desc(), expected);
            // The chroma and the range are each read from the session rather than assumed
            // equal: a 4:2:0 session converted at full range exists, so the two are no longer
            // the same answer.
            assert_eq!(
                p.colorspace_desc(),
                crate::encoders::colorspace_desc(
                    crate::encoders::session_fullcolor(None, &settings),
                    crate::encoders::session_full_range(None, &settings),
                ),
                "X11 and Wayland must describe the same session identically"
            );
        }
    }

    /// A pipeline built with a report bound describes itself in it, and a frame it encodes is
    /// tallied with its bytes and its encode time.
    #[test]
    fn x11_pipeline_reports_the_session_it_settled_on() {
        let report = crate::report::StreamReport::new("x11");
        let _scope = crate::report::enter(&report);
        let mut p = X11Pipeline::new(RustCaptureSettings {
            width: 64,
            height: 64,
            codec: Codec::H264,
            use_cpu: true,
            video_fullframe: true,
            ..Default::default()
        });
        let info = report.info();
        assert_eq!(info.encoder, crate::encoders::software_library(Codec::H264));
        assert!(!info.hardware);
        assert_eq!(info.encoder_reason, "software encoding selected");
        assert_eq!(info.codec, "h264");
        assert_eq!(info.stripes, 1);
        assert!(info.gpu.is_empty());

        let pixels = vec![0x80u8; 64 * 64 * 4];
        let mut stripes = p.process(&pixels, 64 * 4);
        assert!(!stripes.is_empty(), "the first frame of a pipeline is sent");
        let start = crate::wayland::host::now_ns();
        crate::encoders::software::FrameTiming::stamp(
            &mut stripes,
            start - 2_000_000,
            start - 1_000_000,
        );
        report.tally(&stripes);
        let totals = report.totals();
        assert_eq!(totals.frames, 1);
        assert_eq!(
            totals.bytes,
            stripes.iter().map(|s| s.data.len() as u64).sum::<u64>()
        );
        assert!(totals.encode_ns >= 1_000_000 && totals.pipeline_ns >= 2_000_000);
    }

    /// A JPEG stream's log line names the library that encodes it, as its report does.
    #[test]
    fn x11_jpeg_pipeline_names_its_encoder() {
        let report = crate::report::StreamReport::new("x11");
        let _scope = crate::report::enter(&report);
        let p = X11Pipeline::new(RustCaptureSettings {
            width: 64,
            height: 64,
            codec: Codec::Jpeg,
            use_cpu: true,
            ..Default::default()
        });
        assert_eq!(p.encoder_name(), format!("CPU ({})", report.info().encoder));
    }

    /// On a host with a hardware encoder the report names it, the GPU it runs on, and the node.
    #[test]
    #[ignore]
    fn gpu_x11_pipeline_reports_the_hardware_session() {
        let report = crate::report::StreamReport::new("x11");
        let _scope = crate::report::enter(&report);
        let p = X11Pipeline::new(RustCaptureSettings {
            width: 1280,
            height: 720,
            codec: Codec::H264,
            target_fps: 60.0,
            video_crf: 25,
            ..Default::default()
        });
        assert!(p.is_hardware(), "needs a hardware encoder on render node 0");
        let info = report.info();
        println!("{info:?}");
        assert!(info.hardware);
        assert_eq!(info.encoder, p.encoder_name());
        assert!(info.encoder_reason.is_empty());
        assert_eq!(info.encode_node, 0);
        assert!(!info.driver.is_empty());
    }

    /// With no motion, no request, and paint-over off, nothing is emitted at any frame-counter
    /// position: the GOP is infinite, so there is no scheduled IDR to break the silence.
    #[test]
    fn static_frames_stay_silent_without_request() {
        let s = RustCaptureSettings {
            use_paint_over_quality: false,
            ..hw_settings()
        };
        let mut st = StripeState::default();
        for fc in [0u16, 1, 120, 240] {
            let d = decide_hw_fullframe(&mut st, &s, fc, Damage::None, false, false, HOLDS);
            assert!(!d.send && !d.force_idr, "frame {fc} should stay idle");
        }
    }

    /// With paint-over off, a forced keyframe on a still screen still opens the recovery burst,
    /// so a constant-rate session can refine it.
    #[test]
    fn requested_idr_recovers_even_without_paint_over() {
        let s = RustCaptureSettings {
            use_paint_over_quality: false,
            video_paintover_burst_frames: 5,
            ..hw_settings()
        };
        let mut st = StripeState::default();
        let d = decide_hw_fullframe(&mut st, &s, 5, Damage::None, false, true, HOLDS);
        assert!(d.send && d.force_idr && d.hold_qp.is_none());
        assert_eq!(
            st.h264_burst_frames_remaining, 5,
            "recovery burst armed without paint-over"
        );
        let d = decide_hw_fullframe(&mut st, &s, 6, Damage::None, false, false, HOLDS);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 25);
    }

    #[test]
    fn configured_interval_restores_scheduled_keyframes() {
        let mut s = hw_settings();
        s.keyframe_interval_s = 2.0;
        assert!(periodic_idr_due(&s, 0));
        assert!(!periodic_idr_due(&s, 1));
        assert!(periodic_idr_due(&s, 120));
        let mut st = StripeState::default();
        let d = decide_hw_fullframe(&mut st, &s, 120, Damage::None, false, false, HOLDS);
        assert!(
            d.send && d.force_idr,
            "interval keyframe fires on a static screen"
        );
        s.keyframe_interval_s = 0.0;
        assert!(!periodic_idr_due(&s, 0) && !periodic_idr_due(&s, 120));
    }

    /// A held refresh of a constant-rate session is coarsened from the rate control's last
    /// quantizer to what a second of the target buys; a constant quality, or a rate control
    /// that coded near the paint-over quantizer, gets the paint-over one.
    #[test]
    fn a_held_refresh_fits_the_budget_a_constant_rate_leaves() {
        let cbr = RustCaptureSettings {
            video_cbr_mode: true,
            video_paintover_crf: 18,
            target_fps: 30.0,
            ..hw_settings()
        };
        let q = |s: &RustCaptureSettings, last| {
            held_refresh_quality(
                s,
                EncoderQuality {
                    last,
                    bytes: None,
                    holds: true,
                    reopens: false,
                    keys: true,
                    band: None,
                    measures: false,
                    psnr: None,
                },
            )
        };
        assert_eq!(
            q(&cbr, Some(51)),
            22,
            "thirty frames of budget are 29 steps finer than the rate control"
        );
        assert_eq!(q(&cbr, Some(45)), 18);
        assert_eq!(q(&cbr, Some(30)), 18);
        assert_eq!(q(&cbr, None), 18);
        let crf = RustCaptureSettings {
            video_cbr_mode: false,
            ..cbr.clone()
        };
        assert_eq!(
            q(&crf, Some(45)),
            18,
            "a constant quality has no budget to fit"
        );
    }

    /// Damage rectangles report the share of the frame they cover, clipped to it.
    #[test]
    fn damage_rectangles_report_their_share_of_the_frame() {
        use smithay::utils::Rectangle;
        assert_eq!(Damage::of_rects(&[], 100, 100), Damage::None);
        let quarter = [Rectangle::new((0, 0).into(), (50, 50).into())];
        assert_eq!(Damage::of_rects(&quarter, 100, 100), Damage::Area(0.25));
        let past = [Rectangle::new((80, 80).into(), (50, 50).into())];
        assert_eq!(
            Damage::of_rects(&past, 100, 100),
            Damage::Area(0.04),
            "clipped to the frame"
        );
        let whole = [Rectangle::new((0, 0).into(), (200, 200).into())];
        assert_eq!(Damage::of_rects(&whole, 100, 100), Damage::Area(1.0));
    }

    /// The banded content hash reports the share of bands that changed, and nothing when the
    /// frame is the same.
    #[test]
    fn hash_damage_reports_the_bands_that_changed() {
        let (w, h) = (32usize, 256usize);
        let mut bands = Vec::new();
        let frame = vec![7u8; w * 4 * h];
        assert_eq!(
            hash_damage(&mut bands, &frame, w * 4, h, 10, 20),
            Damage::Area(1.0),
            "the first frame is all new"
        );
        assert_eq!(
            hash_damage(&mut bands, &frame, w * 4, h, 10, 20),
            Damage::None
        );
        let mut caret = frame.clone();
        caret[w * 4 * (DAMAGE_BAND_ROWS + 1)] = 0;
        let n = h.div_ceil(DAMAGE_BAND_ROWS);
        assert_eq!(
            hash_damage(&mut bands, &caret, w * 4, h, 10, 20),
            Damage::Area(1.0 / n as f32),
            "one band of {n}"
        );
    }

    /// The rows a band hash found changed map onto full-width rectangles, one per run of bands,
    /// the last clipped to the frame; before any hash the whole frame is dirty.
    #[test]
    fn band_rects_cover_the_runs_of_changed_bands() {
        let r = |y: i32, h: i32| Rectangle::new((0, y).into(), (64, h).into());
        assert_eq!(band_rects(&[], 64, 100), vec![r(0, 100)]);
        assert_eq!(band_rects(&[false, false, false, false], 64, 100), vec![]);
        assert_eq!(
            band_rects(&[true, true, false, true], 64, 100),
            vec![r(0, 64), r(96, 4)]
        );
        assert_eq!(band_damage(&[true, true, false, true]), Damage::Area(0.75));
        assert_eq!(band_damage(&[]), Damage::Area(1.0));
    }

    /// Band hashes taken elsewhere read as the share of the bands that changed: a caret's band of
    /// a 1080p screen is a small change and three are motion; a first frame, or one of another
    /// height, is all new.
    #[test]
    fn hashed_damage_reports_the_bands_that_changed() {
        let n = 1080usize.div_ceil(DAMAGE_BAND_ROWS);
        let frame: Vec<u64> = (0..n as u64).collect();
        let mut last = Vec::new();
        assert_eq!(hashed_damage(&mut last, frame.clone()), Damage::Area(1.0));
        assert_eq!(hashed_damage(&mut last, frame.clone()), Damage::None);
        let mut caret = frame.clone();
        caret[6] ^= 1;
        let d = hashed_damage(&mut last, caret.clone());
        assert_eq!(d, Damage::Area(1.0 / n as f32));
        assert!(!d.is_motion());
        let mut window = caret.clone();
        for band in &mut window[10..13] {
            *band ^= 1;
        }
        assert!(hashed_damage(&mut last, window).is_motion());
        assert_eq!(
            hashed_damage(&mut last, frame[..n - 1].to_vec()),
            Damage::Area(1.0)
        );
    }
}

#[cfg(test)]
mod vbv_tests {
    /// VBV sizing policy: an infinite GOP uses 1.5 frames of headroom, scheduled keyframes
    /// relax to 3 frames, and an explicit multiplier overrides both and rescales with bitrate.
    #[test]
    fn vbv_policy() {
        use crate::encoders::vbv_bits;
        let frame = 4_000_000f64 / 60.0;
        assert_eq!(
            vbv_bits(4_000_000, 60.0, 0.0, 0.0),
            (frame * 1.5).round() as u32
        );
        assert_eq!(
            vbv_bits(4_000_000, 60.0, 2.0, 0.0),
            (frame * 3.0).round() as u32
        );
        assert_eq!(vbv_bits(4_000_000, 60.0, 2.0, 1.0), frame.round() as u32);
        assert_eq!(
            vbv_bits(8_000_000, 60.0, 0.0, 1.0),
            (2.0 * frame).round() as u32
        );
    }
}
