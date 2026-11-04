//! The one monotonic clock every backend stamps its frames with.
//!
//! Capture and encode timestamps, A/V pairing, and the latency figures the
//! report carries all subtract across sources, so they must come from this one
//! function: Linux reads `CLOCK_MONOTONIC` as it always has, and the other
//! operating systems read their own monotonic clock through the standard
//! library. Values are process-relative and only their differences mean
//! anything.

/// Nanoseconds on the process's monotonic clock.
#[cfg(target_os = "linux")]
pub(crate) fn now_ns() -> i64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    t.tv_sec * 1_000_000_000 + t.tv_nsec
}

/// Nanoseconds since some fixed point in this process's life, monotonic.
#[cfg(not(target_os = "linux"))]
pub(crate) fn now_ns() -> i64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = EPOCH.get_or_init(Instant::now);
    epoch.elapsed().as_nanos() as i64
}
