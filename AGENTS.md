# Working on this repository

pixelflux is the screen-capture and video-encode library behind
[selkies](https://github.com/selkies-project/selkies), and is developed together with it and with
[pcmflux](https://github.com/selkies-project/pcmflux) (audio capture and encode). A change in one often belongs in
another; coordinate across all three.

Use web search, web fetch, and other available tools as necessary. Make sure that the comments or documentation are
not too verbose (do not add comments more fit for a PR summary than a comment). Do not leave arbitrary numbers (such
as issue or task numbers) in the code or documentation. Do not use inline comments. Do not use comments or
documentation that describe arbitrary code changes of previous states compared to the current code that do not need
explanation. The code commenting should reflect the current state of the codebase and be used to convey information
to an LLM bot or developer.

Empirical testing is a very useful way to develop this project, and empirical testing is possible for EVERYTHING,
including implementation, auditing, validation, or verification. A few such options are by utilizing the currently
installed Firefox and Chrome, as well as the WebKit engine provided by Playwright/Selenium/Puppeteer/Cypress in place
of Safari, for end-to-end tests. HOWEVER, ask the user for permission to create a test environment (possibly using
Miniforge; but note that it is likely the system `libgbm.so` should be used for GBM support on NVIDIA and other GPUs)
and receive directives from the user on how the environment should be constructed and constrained.

Note that parity between X11 and Wayland, as well as between WebSockets and WebRTC, or between the default dashboard
and the wish dashboard, is considered a key focus (things that were not wired up correctly on either side, and similar
discrepancies, are subject to fixes or deduplication). I prefer deduplicating code that performs similar purposes
across different modes over keeping duplicate code for no reason and more fragility. Refactor through deduplication if
you are confident there will be no regressions (or able to validate regressions). Screen coroutine usage in both
Python and JavaScript, as well as thread usage in all languages, so that everything is performant and does not lead to
hanging or lagging. Performance preservation or improvements such as zero-copy and latency-reducing measures are
always important. Note that compatibility should be ensured for Python 3.9 to 3.14 or even higher, and CUDA/NVENC 11
to 13 or higher. Protocol clients form fallback ladders that bind the newest architecture first (ext- before
zwlr-data-control in dcclient) and exist to keep selkies' Wayland path subprocess-free — they replace wtype/wl-copy
style forks, so extend them in-process rather than shelling out. Update the translations as well (and write/update additional entries if necessary) as necessary.
A defect that predates the change you are making is still in scope: finding it does not make it someone else's,
and "pre-existing" is not a reason to leave it. Fix it, or say precisely what is broken, what you ruled out, and
what you would do next. The same applies to a failure you cannot reproduce yet -- narrow it until it is either
fixed or precisely described, and never let a test that fails for an unknown reason pass unremarked.

Software H.264 is resolved at build time, never by a setting: the default `gpl` feature makes libx264 the
encoder behind every CPU H.264 session (striped and full-frame), and a build without it
(`PIXELFLUX_ENABLE_GPL=0` → `--no-default-features --features openh264`) puts Cisco OpenH264 behind the same
striped path (`encoders/oh264.rs`, one instance per stripe) with the same wire framing; `SOFTWARE_H264_ENCODER`
/ `SOFTWARE_H264_FULLCOLOR` in `lib.rs` (exported to Python as `pixelflux.SOFTWARE_H264_ENCODER`) are the only
places the choice is read, and selkies derives its rate-control default from the exported name. Test both
configurations (`cargo test --lib` and `cargo test --lib --no-default-features --features openh264`); the
OpenH264 crates are also dev-dependencies so its tests run under the default build.

The virtual camera (`pixelflux/src/webcam/`, Python class `VirtualCamera`) is the webcam counterpart of pcmflux's
`AudioPlayback`: selkies only gates and hands encoded frames over; decoding (libavcodec, TurboJPEG), fitting into the
fixed device format, and publishing happen on the camera's own thread. `push` takes the frame's upright transform as
optional arguments (`rotation` in clockwise degrees, then a horizontal `flip`); `convert::orient_i420` bakes it
right after decode, ahead of the fit, and an upright MJPEG frame keeps its pass-through. Sinks are the shared-memory ring served to the
Selkies V4L2 interposer (`ring.rs`/`server.rs`; the layout is mirrored byte-for-byte by
`selkies/addons/v4l2-interposer/v4l2_interposer.c` and checked by selkies' `tests/unit/test_webcam_abi.py`), a
v4l2loopback output device (`v4l2out.rs`), and a PipeWire `Video/Source` node (`pipewire.rs`, `libpipewire-0.3`
loaded at run time, pods built by hand — never add a build-time PipeWire dependency). `cargo test --lib webcam`
covers the ring, decoders (including an OpenH264→avcodec round trip) and pod layouts; the device-level and browser
checks live in selkies (`tests/integration/test_webcam_device.py`, `tests/e2e/test_webcam.py`).

Licensing is part of the build matrix: `LICENSES.md` inventories every crate and native library of the default
(`gpl`, libx264) and `PIXELFLUX_ENABLE_GPL=0` (`openh264`) builds, `scripts/check-licenses.py` and
`pixelflux/deny.toml` gate both (the `Licenses` workflow runs them), and a new crate that links, loads or vendors
native code has to be described in the script's `NATIVE` table and in `LICENSES.md` before the check passes.
Copyleft stays confined to the `gpl` feature.

Update this file when certain details change.
