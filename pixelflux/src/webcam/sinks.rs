//! The camera's consumers, one interface on every OS.
//!
//! On Linux a camera serves three sinks: the shared-memory ring over the control socket
//! the Selkies V4L2 interposer reads, a v4l2loopback kernel device, and a PipeWire
//! `Video/Source` node. The control socket is the piece the camera reports on and holds
//! past the worker's lifetime ([`Shared`]); the kernel and PipeWire sinks belong to the
//! decoder thread, which retires them on failure ([`Sinks`]). Elsewhere none exists yet,
//! and every accessor reports that absence rather than pretending a sink is up, so the
//! camera core in [`super`] stays single-source across the operating systems.

#[cfg(target_os = "linux")]
mod imp {
    use std::sync::{Arc, Mutex};

    use super::super::Stats;
    use super::super::pipewire::PipeWireSink;
    use super::super::ring::Ring;
    use super::super::server::Server;
    use super::super::v4l2out::V4l2Output;

    /// The control socket, held by both the decoder thread and the camera object: the
    /// clients it serves and the path it binds are what the camera reports.
    pub struct Shared {
        server: Arc<Server>,
    }

    impl Shared {
        /// Interposer clients that completed the handshake.
        pub(crate) fn clients(&self) -> usize {
            self.server.client_count()
        }

        /// The control socket's path.
        pub(crate) fn socket_path(&self) -> Option<String> {
            Some(self.server.path().to_string())
        }
    }

    /// The sinks the decoder thread publishes through.
    pub struct Sinks {
        server: Arc<Server>,
        v4l2out: Option<V4l2Output>,
        pipewire: Option<PipeWireSink>,
        device_path: Arc<Mutex<String>>,
    }

    impl Sinks {
        /// Bind the control socket for the ring's consumers, open the kernel device the
        /// setting names ("" or a false value none, "auto" the first loopback device, else
        /// that path), and connect the PipeWire node when asked. Each sink degrades to
        /// absent with a line on its own failure; the socket failing the camera fails.
        pub(crate) fn start(
            socket_path: &str,
            ring: &Ring,
            device_setting: &str,
            want_pipewire: bool,
            pipewire_node_name: &str,
        ) -> Result<(Shared, Self), String> {
            let server = Arc::new(
                Server::bind(socket_path, ring.config_bytes(), ring.fd())
                    .map_err(|e| format!("bind({}) failed: {}", socket_path, e))?,
            );
            let v4l2out = match device_setting {
                "" | "false" | "no" | "off" | "none" => None,
                "auto" | "true" | "yes" | "on" => {
                    V4l2Output::find_loopback_device().and_then(|p| {
                        match V4l2Output::open(&p, ring.format()) {
                            Ok(o) => Some(o),
                            Err(e) => {
                                eprintln!("[webcam] {}; kernel device sink disabled", e);
                                None
                            }
                        }
                    })
                }
                path => match V4l2Output::open(path, ring.format()) {
                    Ok(o) => Some(o),
                    Err(e) => {
                        eprintln!("[webcam] {}; kernel device sink disabled", e);
                        None
                    }
                },
            };
            let pipewire = if want_pipewire {
                match PipeWireSink::connect(
                    pipewire_node_name,
                    "Selkies Virtual Camera",
                    ring.format(),
                ) {
                    Ok(sink) => Some(sink),
                    Err(e) => {
                        eprintln!("[webcam] PipeWire sink unavailable: {}", e);
                        None
                    }
                }
            } else {
                None
            };
            let device_path = Arc::new(Mutex::new(
                v4l2out
                    .as_ref()
                    .map(|o| o.path().to_string())
                    .unwrap_or_default(),
            ));
            let shared = Shared {
                server: server.clone(),
            };
            Ok((
                shared,
                Sinks {
                    server,
                    v4l2out,
                    pipewire,
                    device_path,
                },
            ))
        }

        /// After a publish: wake the socket clients and mirror the frame into the kernel
        /// device and the PipeWire node where those sinks are up, retiring a sink that
        /// failed.
        pub(crate) fn fan_out(&mut self, ring: &Ring, ts: u64, stats: &Stats) {
            use std::sync::atomic::Ordering;

            self.server.ring_doorbell();
            if let Some(frame) = ring.latest_frame() {
                if let Some(out) = self.v4l2out.as_mut() {
                    out.write_frame(frame);
                    if out.is_failed() {
                        self.device_path
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .clear();
                        self.v4l2out = None;
                    }
                }
                if let Some(pw) = self.pipewire.as_mut() {
                    pw.publish(frame, ts);
                    stats
                        .pipewire_streaming
                        .store(pw.is_streaming(), Ordering::Relaxed);
                    if pw.is_failed() {
                        eprintln!("[webcam] PipeWire node stopped; PipeWire sink disabled");
                        self.pipewire = None;
                        stats.pipewire_streaming.store(false, Ordering::Relaxed);
                    }
                }
            }
        }

        /// The kernel device path, shared with the caller that reports it.
        pub(crate) fn device_path(&self) -> Arc<Mutex<String>> {
            self.device_path.clone()
        }

        /// Whether a PipeWire node is connected.
        pub(crate) fn pipewire_active(&self) -> bool {
            self.pipewire.is_some()
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use std::sync::{Arc, Mutex};

    use super::super::Stats;
    use super::super::ring::Ring;

    /// No control socket exists on this OS yet: the camera reports no clients and no path.
    pub struct Shared;

    impl Shared {
        pub(crate) fn clients(&self) -> usize {
            0
        }

        pub(crate) fn socket_path(&self) -> Option<String> {
            None
        }
    }

    /// No camera sink exists on this OS yet: frames are published into the ring and the
    /// publish-side accessors report the absence.
    pub struct Sinks {
        device_path: Arc<Mutex<String>>,
    }

    impl Sinks {
        pub(crate) fn start(
            _socket_path: &str,
            _ring: &Ring,
            _device_setting: &str,
            _want_pipewire: bool,
            _pipewire_node_name: &str,
        ) -> Result<(Shared, Self), String> {
            Ok((
                Shared,
                Sinks {
                    device_path: Arc::new(Mutex::new(String::new())),
                },
            ))
        }

        pub(crate) fn fan_out(&mut self, _ring: &Ring, _ts: u64, _stats: &Stats) {}

        pub(crate) fn device_path(&self) -> Arc<Mutex<String>> {
            self.device_path.clone()
        }

        pub(crate) fn pipewire_active(&self) -> bool {
            false
        }
    }
}

pub use imp::{Shared, Sinks};
