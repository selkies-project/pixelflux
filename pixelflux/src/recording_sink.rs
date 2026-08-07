//! Unix-socket H.264 fan-out for external recording.
//!
//! Frames are intercepted at the delivery layer (not inside each encoder), so the tap works
//! uniformly for every full-frame encoder. The 10-byte pixelflux wire header is skipped so
//! consumers receive a plain Annex-B elementary stream that is directly muxable.
//!
//! The tap must never perturb the live viewer transport, and it never copies frame bytes:
//! stripe payloads are `Arc`-shared, so the encode thread only clones a handle into a bounded
//! per-client channel drained by a dedicated writer thread. A slow or stalled recorder blocks
//! nothing but itself and is dropped once its queue overflows. A newly connected client arms
//! [`RecordingSink::should_force_idr`] so the next encode emits an IDR it can decode from.

use std::fs;
use std::io::{ErrorKind, Write};
use std::os::unix::net::UnixListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{bounded, Sender, TrySendError};

use crate::encoders::software::EncodedStripe;

/// Per-write timeout on a client stream; a stalled write surfaces as a soft error that
/// [`write_all_frame`] retries, keeping the writer thread responsive to teardown.
const WRITE_TIMEOUT: Duration = Duration::from_millis(100);

/// How often the non-blocking accept loop retries when no client is waiting.
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Per-client backlog bound. A recorder that falls this far behind is dropped rather than
/// allowed to grow memory or push back on the shared encode thread.
const CLIENT_QUEUE_CAP: usize = 256;

/// A queued frame: the `Arc`-shared payload plus the byte offset where the recordable
/// Annex-B stream starts (past the wire header, or `0` when the payload is bare).
type QueuedFrame = (Arc<Vec<u8>>, usize);

/// The sink's handle to one connected recorder: the feed end of its bounded queue and a kill
/// switch for its writer thread. The socket itself is owned solely by that writer thread.
struct ClientHandle {
    tx: Sender<QueuedFrame>,
    stop: Arc<AtomicBool>,
}

/// Unix-socket fan-out that broadcasts every encoded H.264 frame to connected consumers.
///
/// A listener thread accepts connections and gives each its own bounded queue and writer thread
/// (see [`ClientHandle`]) so one slow reader cannot stall the others or the encode thread.
pub struct RecordingSink {
    /// Filesystem path of the Unix socket; removed on drop.
    path: String,
    /// Feed handles for the connected clients, shared with the accept thread.
    clients: Arc<Mutex<Vec<ClientHandle>>>,
    /// Signals the accept thread to exit; set in [`Drop`].
    shutdown: Arc<AtomicBool>,
    /// Flipped to `true` each time a new client connects; consumed by [`should_force_idr`].
    ///
    /// [`should_force_idr`]: RecordingSink::should_force_idr
    client_connected: Arc<AtomicBool>,
    /// One-time notice that the session's H.264 frames are striped and unrecordable.
    warned_unrecordable: AtomicBool,
}

impl RecordingSink {
    /// Bind a Unix socket at `settings_path`, or return `None` when no path is configured or the
    /// bind fails. Recording is an optional tap that must never take the pipeline down, so a bind
    /// error is logged and swallowed.
    pub fn try_bind(settings_path: &str) -> Option<Arc<Self>> {
        if settings_path.is_empty() {
            return None;
        }
        match Self::bind(settings_path.to_string()) {
            Ok(sink) => Some(Arc::new(sink)),
            Err(e) => {
                eprintln!("[recording_sink] bind failed: {:?}", e);
                None
            }
        }
    }

    /// Create the socket and spawn the accept thread. Each accepted connection gets a write
    /// timeout, a bounded queue, and a writer thread; the sink keeps only the feed handle.
    fn bind(path: String) -> std::io::Result<Self> {
        let _ = fs::remove_file(&path);
        let listener = UnixListener::bind(&path)?;
        listener.set_nonblocking(true)?;

        let clients: Arc<Mutex<Vec<ClientHandle>>> = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let client_connected = Arc::new(AtomicBool::new(false));

        let clients_acc = clients.clone();
        let shutdown_acc = shutdown.clone();
        let client_connected_acc = client_connected.clone();
        let path_log = path.clone();

        thread::spawn(move || {
            eprintln!("[recording_sink] listening on {}", path_log);
            while !shutdown_acc.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        if let Err(e) = stream.set_write_timeout(Some(WRITE_TIMEOUT)) {
                            eprintln!("[recording_sink] set_write_timeout failed: {:?}", e);
                            continue;
                        }

                        let (tx, rx) = bounded::<QueuedFrame>(CLIENT_QUEUE_CAP);
                        let stop = Arc::new(AtomicBool::new(false));
                        let stop_writer = stop.clone();
                        thread::spawn(move || {
                            let mut stream = stream;
                            for (buf, offset) in rx.iter() {
                                if stop_writer.load(Ordering::Relaxed) {
                                    break;
                                }
                                if let Err(e) =
                                    write_all_frame(&mut stream, &buf[offset..], &stop_writer)
                                {
                                    eprintln!(
                                        "[recording_sink] writer thread exiting; write failed: {:?}",
                                        e
                                    );
                                    break;
                                }
                            }
                        });

                        let mut guard = clients_acc.lock().unwrap();
                        guard.push(ClientHandle { tx, stop });
                        client_connected_acc.store(true, Ordering::Relaxed);
                        eprintln!("[recording_sink] client connected; total {}", guard.len());
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(ACCEPT_POLL_INTERVAL);
                    }
                    Err(e) => {
                        eprintln!("[recording_sink] accept error: {:?}", e);
                        thread::sleep(Duration::from_millis(500));
                    }
                }
            }
            eprintln!("[recording_sink] listener thread exiting");
        });

        Ok(Self {
            path,
            clients,
            shutdown,
            client_connected,
            warned_unrecordable: AtomicBool::new(false),
        })
    }

    /// Returns `true` exactly once after a new client connects, signalling that the next encode
    /// should produce an IDR so the consumer starts from a clean reference frame.
    pub fn should_force_idr(&self) -> bool {
        self.client_connected.swap(false, Ordering::Relaxed)
    }

    /// Delivery-layer tap for one encoded frame. The socket carries a single H.264
    /// elementary stream, so only a lone full-height stripe (`data_type == 2`) is
    /// recordable: striped CPU encodes are N independent per-stripe streams, and
    /// interleaving them would produce an undecodable file — those are skipped with a
    /// one-time notice (live streaming is unaffected). The 10-byte wire header
    /// (`0x04` tag) is skipped via the queued offset so consumers receive plain
    /// Annex-B.
    ///
    /// Never blocks and never copies: the `Arc` payload is cloned into each client's
    /// bounded queue with `try_send`, and a client whose queue is full or whose
    /// writer died is dropped.
    pub fn write_frame(&self, stripes: &[EncodedStripe], full_height: i32) {
        let mut h264 = stripes
            .iter()
            .filter(|s| s.data_type == 2 && !s.data.is_empty());
        let Some(stripe) = h264.next() else { return };
        if h264.next().is_some() || stripe.stripe_y_start != 0 || stripe.stripe_height != full_height
        {
            if !self.warned_unrecordable.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[recording_sink] WARNING: striped H.264 frames are not recordable \
                     (the socket carries one elementary stream); use a full-frame encoder \
                     to record this session"
                );
            }
            return;
        }
        let offset = if stripe.data.len() >= 10 && stripe.data[0] == 0x04 {
            10
        } else {
            0
        };
        if stripe.data.len() == offset {
            return;
        }

        let mut clients = self.clients.lock().unwrap();
        if clients.is_empty() {
            return;
        }
        let mut to_remove: Vec<usize> = Vec::new();
        for (idx, client) in clients.iter().enumerate() {
            match client.tx.try_send((stripe.data.clone(), offset)) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    eprintln!("[recording_sink] dropping slow client (idx {})", idx);
                    to_remove.push(idx);
                }
                Err(TrySendError::Disconnected(_)) => {
                    to_remove.push(idx);
                }
            }
        }
        for idx in to_remove.into_iter().rev() {
            let removed = clients.swap_remove(idx);
            removed.stop.store(true, Ordering::Relaxed);
        }
    }
}

impl Drop for RecordingSink {
    /// Stop accepting, release every writer thread (set each `stop`, then drop its sender so an
    /// idle writer parked on `rx.iter()` wakes), and remove the socket file.
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Ok(mut clients) = self.clients.lock() {
            for client in clients.iter() {
                client.stop.store(true, Ordering::Relaxed);
            }
            clients.clear();
        }
        let _ = fs::remove_file(&self.path);
    }
}

/// Write one whole frame to a recorder's socket, resuming across the soft timeouts a slow reader
/// induces so a partial Annex-B NAL is never left behind. Aborts if `stop` is set (the client was
/// dropped by [`RecordingSink::write_encoded_frame`]) or a hard error occurs.
fn write_all_frame<W: Write>(stream: &mut W, buf: &[u8], stop: &AtomicBool) -> std::io::Result<()> {
    let mut written = 0usize;
    while written < buf.len() {
        if stop.load(Ordering::Relaxed) {
            return Err(std::io::Error::other("writer stopped (client dropped)"));
        }
        match stream.write(&buf[written..]) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    ErrorKind::WriteZero,
                    "failed to write whole frame",
                ));
            }
            Ok(n) => written += n,
            Err(ref e) if e.kind() == ErrorKind::TimedOut => {}
            Err(ref e) if e.kind() == ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod cost_tests {
    //! The sink's isolation contract, measured: feeding a frame must cost nothing when no
    //! recorder is connected (empty-clients early return), microseconds when a healthy
    //! recorder drains its queue, and stay bounded (a lock + `try_send`, never a blocking
    //! write) when a recorder stalls completely — until the bounded queue overflows and the
    //! client is dropped, returning the tap to idle cost.

    use super::*;
    use std::io::Read;
    use std::os::unix::net::UnixStream;
    use std::time::Instant;

    fn frame(len: usize) -> EncodedStripe {
        let mut data = vec![0u8; len];
        data[0] = 0x04; // wire-header tag so the 10-byte strip path runs
        EncodedStripe {
            data: Arc::new(data),
            data_type: 2,
            stripe_y_start: 0,
            stripe_height: 720,
            frame_id: 0,
        }
    }

    fn feed_timed(sink: &RecordingSink, n: usize, len: usize) -> (f64, f64) {
        let f = frame(len);
        let mut max_us = 0f64;
        let mut total_us = 0f64;
        for _ in 0..n {
            let t = Instant::now();
            sink.write_frame(std::slice::from_ref(&f), 720);
            let us = t.elapsed().as_secs_f64() * 1e6;
            total_us += us;
            max_us = max_us.max(us);
            thread::sleep(Duration::from_micros(200));
        }
        (total_us / n as f64, max_us)
    }

    #[test]
    fn stalled_recorder_isolation_cost() {
        let path = format!("/tmp/pf-sink-cost-{}.sock", std::process::id());
        let sink = RecordingSink::try_bind(&path).expect("bind");

        // Idle: no client connected.
        let (idle_mean, idle_max) = feed_timed(&sink, 500, 100_000);

        // Healthy: a client draining as fast as it can.
        let mut healthy = UnixStream::connect(&path).expect("connect");
        thread::sleep(Duration::from_millis(200));
        let drain = thread::spawn(move || {
            let mut buf = vec![0u8; 1 << 20];
            while healthy.read(&mut buf).map(|n| n > 0).unwrap_or(false) {}
        });
        let (healthy_mean, healthy_max) = feed_timed(&sink, 500, 100_000);

        // Stalled: a connected client that never reads. The socket buffer fills, then the
        // bounded queue fills, then the client is dropped (~256 frames later).
        let stalled = UnixStream::connect(&path).expect("connect");
        thread::sleep(Duration::from_millis(200));
        let (stalled_mean, stalled_max) = feed_timed(&sink, 500, 100_000);
        drop(stalled);

        println!(
            "[sink-cost] idle    mean {idle_mean:.3}us max {idle_max:.3}us\n\
             [sink-cost] healthy mean {healthy_mean:.3}us max {healthy_max:.3}us\n\
             [sink-cost] stalled mean {stalled_mean:.3}us max {stalled_max:.3}us"
        );
        drop(sink);
        let _ = drain.join();

        assert!(idle_mean < 5.0, "idle feed should be sub-5us, was {idle_mean:.3}us");
        assert!(healthy_mean < 100.0, "healthy feed should be tens of us, was {healthy_mean:.3}us");
        assert!(
            stalled_max < 10_000.0,
            "a stalled recorder must never block the tap >10ms, was {stalled_max:.3}us"
        );
    }
}
