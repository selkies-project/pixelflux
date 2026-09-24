//! The frame pacing every capture backend shares: when a display is due a frame, and how far
//! a change that just happened may bring the next one forward.
//!
//! A capture renders on its frame timer, and something that actually happened -- input the
//! compositor processed, a frame a host delivered, a region the X server reports damaged --
//! may render one ahead of that timer. The pull spends from a budget that refills slowly, so a
//! screen changing faster than the cadence raises the output rate by no more than the refill
//! while a change arriving on its own schedule is published as it lands rather than up to a
//! period later.

use std::time::{Duration, Instant};

/// What asks a display for a frame: its frame timer, fresh input, a frame the host just
/// delivered under host capture, or a region the X server reports damaged. All but the timer
/// say something has actually changed, and so may pull the cadence's phase forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TickTrigger {
    Timer,
    Input,
    HostFrame,
    Damage,
}

impl TickTrigger {
    /// Whether this trigger may render ahead of the timer, moving the cadence to its phase.
    fn pulls_forward(self) -> bool {
        !matches!(self, TickTrigger::Timer)
    }
}

/// The share of a frame period the timer's tick may run early by, absorbing its wakeup jitter.
const TIMER_TICK_MIN_FRACTION: f64 = 0.9;
/// The share of a frame period that has to pass before input or a host frame may pull the
/// next frame forward.
const INPUT_TICK_MIN_FRACTION: f64 = 0.5;
/// The share of wall time that accrues as time a pull may bring frames forward by. A frame
/// pulled forward by some time is one the cadence earns that much later, so the sustained
/// rate can rise above the configured one by at most this share.
const INPUT_BORROW_REFILL: f64 = 1.0 / 32.0;
/// The share of a frame period the timer waits beyond a whole one after an input frame while
/// the input paces itself at the cadence, so input arriving a little slower than the timer
/// still renders its own frames instead of landing just behind a timer frame and waiting a
/// period.
const TIMER_GRACE_AFTER_PULL: f64 = 0.25;
/// The input spacings, as shares of the frame period, that count as pacing at the cadence:
/// a client at the cadence give or take its jitter. A faster client is borrowing frames and a
/// slower one is content the timer paces, and the timer keeps its period behind either.
const PACED_INPUT_MIN_FRACTION: f64 = 0.75;
const PACED_INPUT_MAX_FRACTION: f64 = 1.25;

/// A capture's frame pacing: the last frame it rendered, and how far a pull may still bring
/// the next one forward.
///
/// The shared frame timer fires for the earliest due capture; each capture renders on it once
/// its own period has (nearly) passed. Fresh input may also render a frame, and so may a frame
/// the host compositor delivers under host capture: a whole period after the last one the pull
/// renders at once, and from half a period on it brings the frame forward, spending the time it
/// comes early by from a budget that refills at `INPUT_BORROW_REFILL`, is paid back by a pull
/// that arrives late, and holds half a period at most. A pointer moving at the client's refresh
/// rate is thus captured as it lands rather than up to a period later, and a host's frames are
/// published as they arrive rather than on the next timer tick, while either at a rate the
/// cadence cannot follow raises the output rate by no more than the refill.
///
/// After an input frame, while the input's own spacing sits at the cadence
/// (`PACED_INPUT_MIN_FRACTION` to `PACED_INPUT_MAX_FRACTION` of a period), the timer stands
/// back for `TIMER_GRACE_AFTER_PULL` beyond a period, so such input keeps rendering its own
/// frames whether it runs a hair faster or slower than the timer, or jitters around it: a timer
/// that resumed exactly a period after the frame would render just ahead of a move landing a
/// little late, which then could not pull and waited a period. When the input stops, the timer
/// renders after the grace and carries on from there. A faster client, a slower one, and a
/// host's frames leave the timer its period as before.
#[derive(Debug, Default)]
pub struct FramePace {
    /// Last tick this capture actually rendered.
    pub last_tick: Option<Instant>,
    /// Whether that frame was input's, with the input pacing itself at the cadence, so the
    /// timer stands back for the next move.
    last_paced: bool,
    /// The budget left when it was last spent, and when that was; a fresh capture holds the cap.
    borrow_budget: Option<(Duration, Instant)>,
    /// Where the timer was held off to by a tick that rendered nothing (host capture with no
    /// fresh frame), so it neither spins nor counts against the next frame.
    deferred_until: Option<Instant>,
}

impl FramePace {
    /// Whether a capture rendering every `period` is due a frame at `now`, given what asks.
    pub fn due(&self, trigger: TickTrigger, period: Duration, now: Instant) -> bool {
        let Some(last) = self.last_tick else { return true };
        let elapsed = now.saturating_duration_since(last);
        if !trigger.pulls_forward() {
            return elapsed >= self.timer_wait(period);
        }
        elapsed >= period
            || (elapsed >= period.mul_f64(INPUT_TICK_MIN_FRACTION)
                && self.budget(period, now) >= period - elapsed)
    }

    /// Whether input spaced `interval` apart paces itself at the cadence of `period`.
    pub fn input_paced(interval: Option<Duration>, period: Duration) -> bool {
        interval.is_some_and(|i| {
            i >= period.mul_f64(PACED_INPUT_MIN_FRACTION) && i <= period.mul_f64(PACED_INPUT_MAX_FRACTION)
        })
    }

    /// Record the frame rendered at `now`. A pull ahead of the period spends the budget by
    /// what it came early; one behind the period pays back what it came late. `input_paced`
    /// says whether the input driving the capture paces itself at the cadence
    /// (`Self::input_paced`), which is what lets the timer stand back after an input frame.
    pub fn ticked(&mut self, trigger: TickTrigger, period: Duration, now: Instant, input_paced: bool) {
        if trigger.pulls_forward()
            && let Some(last) = self.last_tick
        {
            let elapsed = now.saturating_duration_since(last);
            let budget = self.budget(period, now);
            let left = if elapsed < period {
                budget.saturating_sub(period - elapsed)
            } else {
                (budget + (elapsed - period)).min(period.mul_f64(INPUT_TICK_MIN_FRACTION))
            };
            self.borrow_budget = Some((left, now));
        }
        self.last_paced = trigger == TickTrigger::Input && input_paced;
        self.last_tick = Some(now);
        self.deferred_until = None;
    }

    /// How long after the last frame the timer's own tick renders again.
    fn timer_wait(&self, period: Duration) -> Duration {
        if self.last_paced {
            period.mul_f64(1.0 + TIMER_GRACE_AFTER_PULL)
        } else {
            period.mul_f64(TIMER_TICK_MIN_FRACTION)
        }
    }

    /// Hold the timer off this capture until `until` without a frame counting: a tick that
    /// published nothing must neither spin the timer nor move the cadence a fresh frame is
    /// measured against.
    pub fn defer(&mut self, until: Instant) {
        self.deferred_until = Some(until);
    }

    /// Time since this capture last rendered, if it has.
    pub fn since_last_tick(&self, now: Instant) -> Option<Duration> {
        self.last_tick.map(|last| now.saturating_duration_since(last))
    }

    /// How far a pull may bring the next frame forward at `now`.
    fn budget(&self, period: Duration, now: Instant) -> Duration {
        let cap = period.mul_f64(INPUT_TICK_MIN_FRACTION);
        match self.borrow_budget {
            None => cap,
            Some((left, at)) => {
                (left + now.saturating_duration_since(at).mul_f64(INPUT_BORROW_REFILL)).min(cap)
            }
        }
    }

    /// The earliest a trigger that pulls forward may render: a whole period after the last
    /// frame, less what the budget still allows it to borrow. A caller that waits until then
    /// need not test `due` again, since the budget only grows with time.
    pub fn pull_at(&self, period: Duration, now: Instant) -> Instant {
        self.last_tick
            .map_or(now, |last| last + period.saturating_sub(self.budget(period, now)))
    }

    /// When the timer is next due for this capture: a period after the last frame, and the
    /// grace beyond that after an input frame the input paced.
    pub fn next_due(&self, period: Duration, now: Instant) -> Instant {
        let due = self.last_tick.map_or(now, |last| {
            last + if self.last_paced { period.mul_f64(1.0 + TIMER_GRACE_AFTER_PULL) } else { period }
        });
        self.deferred_until.map_or(due, |held| held.max(due))
    }
}

#[cfg(test)]
mod pacing_tests {
    use super::*;

    const PERIOD: Duration = Duration::from_millis(20);

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    /// The earliest pull and the pull gate agree, so a caller may wait for the one and skip
    /// the other: a fresh capture pulls at once, a full budget buys half a period, and a spent
    /// one leaves the timer its whole period.
    #[test]
    fn the_earliest_pull_is_the_instant_the_gate_opens() {
        let base = Instant::now();
        let mut pace = FramePace::default();
        assert_eq!(pace.pull_at(PERIOD, base), base, "a fresh capture waits for nothing");
        pace.ticked(TickTrigger::Timer, PERIOD, base, false);
        let earliest = pace.pull_at(PERIOD, base);
        assert_eq!(earliest, at(base, 10), "a whole budget buys half a period");
        assert!(pace.due(TickTrigger::Damage, PERIOD, earliest));
        assert!(!pace.due(TickTrigger::Damage, PERIOD, earliest - Duration::from_millis(1)));
        pace.ticked(TickTrigger::Damage, PERIOD, earliest, false);
        assert_eq!(pace.pull_at(PERIOD, earliest), at(base, 30), "a spent budget waits the period out");
    }

    #[test]
    fn a_fresh_capture_is_due_at_once() {
        let pace = FramePace::default();
        let now = Instant::now();
        assert!(pace.due(TickTrigger::Timer, PERIOD, now));
        assert!(pace.due(TickTrigger::Input, PERIOD, now));
        assert_eq!(pace.next_due(PERIOD, now), now);
    }

    #[test]
    fn the_timer_ticks_at_its_period_with_jitter_slack() {
        let base = Instant::now();
        let mut pace = FramePace::default();
        pace.ticked(TickTrigger::Timer, PERIOD, base, false);
        assert!(!pace.due(TickTrigger::Timer, PERIOD, at(base, 17)));
        assert!(pace.due(TickTrigger::Timer, PERIOD, at(base, 18)));
        assert_eq!(pace.next_due(PERIOD, at(base, 5)), at(base, 20));
    }

    #[test]
    fn input_pulls_a_frame_forward_from_a_budget() {
        let base = Instant::now();
        let mut pace = FramePace::default();
        pace.ticked(TickTrigger::Timer, PERIOD, base, false);
        assert!(!pace.due(TickTrigger::Input, PERIOD, at(base, 9)), "under half a period waits for the timer");
        assert!(pace.due(TickTrigger::Input, PERIOD, at(base, 10)));
        pace.ticked(TickTrigger::Input, PERIOD, at(base, 10), false);
        assert_eq!(pace.next_due(PERIOD, at(base, 10)), at(base, 30), "a pull off the cadence leaves the timer its period");
        assert!(pace.due(TickTrigger::Timer, PERIOD, at(base, 30)));
        assert!(!pace.due(TickTrigger::Input, PERIOD, at(base, 25)), "the budget is spent for a second pull");
        assert!(pace.due(TickTrigger::Input, PERIOD, at(base, 30)), "a whole period on, input takes the due frame");
        pace.ticked(TickTrigger::Input, PERIOD, at(base, 30), false);
        // About 4 ms refilled by now: a pull of 10 ms does not fit, one of 3 ms does.
        pace.ticked(TickTrigger::Timer, PERIOD, at(base, 120), false);
        assert!(!pace.due(TickTrigger::Input, PERIOD, at(base, 130)), "a whole pull needs the whole budget back");
        assert!(pace.due(TickTrigger::Input, PERIOD, at(base, 137)), "a small pull fits what refilled");
        pace.ticked(TickTrigger::Input, PERIOD, at(base, 137), false);
        pace.ticked(TickTrigger::Timer, PERIOD, at(base, 500), false);
        assert!(pace.due(TickTrigger::Input, PERIOD, at(base, 510)), "the budget is whole again");
    }

    #[test]
    fn input_a_hair_slower_than_the_cadence_renders_every_move() {
        let base = Instant::now();
        let mut pace = FramePace::default();
        pace.ticked(TickTrigger::Timer, PERIOD, base, false);
        // A client pacing at 20.4 ms against a 20 ms period: its first move lands late in the
        // period and pulls its frame; each later move lands after a whole period and renders
        // at once, and no timer tick falls between two of them.
        let spacing = Duration::from_micros(20_400);
        assert!(FramePace::input_paced(Some(spacing), PERIOD));
        let mut t = 16.0;
        pace.ticked(TickTrigger::Input, PERIOD, at(base, 16), true);
        for _ in 0..50 {
            t += 20.4;
            let now = base + Duration::from_micros((t * 1000.0) as u64);
            assert!(pace.next_due(PERIOD, now) > now, "no timer frame slipped in ahead of the move");
            assert!(pace.due(TickTrigger::Input, PERIOD, now), "the move renders its own frame");
            pace.ticked(TickTrigger::Input, PERIOD, now, true);
        }
        let stopped = base + Duration::from_micros((t * 1000.0) as u64);
        assert_eq!(pace.next_due(PERIOD, stopped), stopped + Duration::from_millis(25),
                   "once the moves stop, the timer resumes after the grace");
    }

    #[test]
    fn a_pull_that_arrives_late_pays_the_budget_back() {
        let base = Instant::now();
        let mut pace = FramePace::default();
        pace.ticked(TickTrigger::Timer, PERIOD, base, false);
        pace.ticked(TickTrigger::Input, PERIOD, at(base, 10), true);
        assert!(!pace.due(TickTrigger::Input, PERIOD, at(base, 25)), "the budget is spent");
        pace.ticked(TickTrigger::Input, PERIOD, at(base, 40), true);
        assert!(pace.due(TickTrigger::Input, PERIOD, at(base, 50)), "a pull 10 ms late earned a 10 ms pull back");
    }

    #[test]
    fn only_input_at_the_cadence_holds_the_timer_back() {
        let base = Instant::now();
        let mut pace = FramePace::default();
        pace.ticked(TickTrigger::Timer, PERIOD, base, false);
        assert!(!FramePace::input_paced(Some(Duration::from_millis(7)), PERIOD), "a 143 Hz client is not paced at 50 Hz");
        assert!(!FramePace::input_paced(Some(Duration::from_millis(40)), PERIOD), "nor is one at half the rate");
        assert!(!FramePace::input_paced(None, PERIOD), "nor no input at all");
        pace.ticked(TickTrigger::Input, PERIOD, at(base, 12), false);
        assert_eq!(pace.next_due(PERIOD, at(base, 12)), at(base, 32), "an unpaced pull leaves the timer its period");
        pace.ticked(TickTrigger::Input, PERIOD, at(base, 32), true);
        assert_eq!(pace.next_due(PERIOD, at(base, 32)), at(base, 57), "a paced one adds the grace");
        pace.ticked(TickTrigger::HostFrame, PERIOD, at(base, 57), true);
        assert_eq!(pace.next_due(PERIOD, at(base, 57)), at(base, 77), "a host frame never does");
    }

    #[test]
    fn a_host_frame_pulls_forward_like_input() {
        let base = Instant::now();
        let mut pace = FramePace::default();
        pace.ticked(TickTrigger::Timer, PERIOD, base, false);
        assert!(!pace.due(TickTrigger::HostFrame, PERIOD, at(base, 9)), "under half a period waits for the timer");
        assert!(pace.due(TickTrigger::HostFrame, PERIOD, at(base, 10)));
        pace.ticked(TickTrigger::HostFrame, PERIOD, at(base, 10), false);
        assert_eq!(pace.next_due(PERIOD, at(base, 10)), at(base, 30), "the cadence follows the host's frames");
        assert!(!pace.due(TickTrigger::Timer, PERIOD, at(base, 20)), "the old phase's tick is skipped");
        assert!(!pace.due(TickTrigger::HostFrame, PERIOD, at(base, 25)), "the budget is spent for a second pull");
        assert!(pace.due(TickTrigger::HostFrame, PERIOD, at(base, 30)), "a whole period on, the frame is due");
    }

    #[test]
    fn a_deferred_tick_holds_the_timer_without_counting_a_frame() {
        let base = Instant::now();
        let mut pace = FramePace::default();
        pace.ticked(TickTrigger::Timer, PERIOD, base, false);
        pace.defer(at(base, 30));
        assert_eq!(pace.next_due(PERIOD, at(base, 20)), at(base, 30), "the timer waits where it was held to");
        assert!(pace.due(TickTrigger::HostFrame, PERIOD, at(base, 21)), "a frame a period on is still due at once");
        pace.ticked(TickTrigger::HostFrame, PERIOD, at(base, 21), false);
        assert_eq!(pace.next_due(PERIOD, at(base, 21)), at(base, 41), "a rendered frame lifts the hold");
    }

    #[test]
    fn input_faster_than_the_period_raises_the_rate_by_the_refill_at_most() {
        let base = Instant::now();
        let mut pace = FramePace::default();
        let end = base + Duration::from_secs(2);
        let (mut frames, mut next_input, mut next_timer) = (0, base, base);
        loop {
            let t = next_input.min(next_timer);
            if t >= end {
                break;
            }
            if t == next_timer {
                if pace.due(TickTrigger::Timer, PERIOD, t) {
                    pace.ticked(TickTrigger::Timer, PERIOD, t, false);
                    frames += 1;
                }
                next_timer = pace.next_due(PERIOD, t);
            }
            if t == next_input {
                if pace.due(TickTrigger::Input, PERIOD, t) {
                    pace.ticked(TickTrigger::Input, PERIOD, t, false);
                    frames += 1;
                    next_timer = pace.next_due(PERIOD, t);
                }
                next_input = t + Duration::from_millis(7);
            }
        }
        assert!((100..=104).contains(&frames), "{frames} frames in two seconds at 50 fps");
    }
}
