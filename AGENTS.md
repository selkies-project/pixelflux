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
to an LLM bot or developer. Write American English -- color, behavior, center, initialize, canceled -- except
where a name belongs to something upstream, such as a Wayland `Cancelled` event or an NVENC `colourMatrix` field.

Empirical testing is possible for everything here, including implementation, auditing, validation, and verification,
and every change is validated before it is reported. `cargo test --lib` in both feature configurations is the floor;
the `#[ignore]`d `gpu_` tests need an NVIDIA GPU (`cargo test gpu_ -- --ignored --nocapture --test-threads=1`,
serially, since concurrent session builds fault in the driver), the `gpu_dmabuf_` ones a render node as well, and the
`gpu_bench_` ones print measurements to quote rather than assert. The VA-API surface probe (`vpp_sw_formats`)
runs only where a VA-API device opens, so the CI runners and NVIDIA hosts never test it; it is proven on Intel
hardware in the selkies sandbox. End to end, a change is a wheel
(`pip wheel . --no-deps`) installed into a selkies sandbox as the Agentic Development section of that repository's
`docs/development.md` describes, driven by its suites over both transports on X11 and Wayland with the installed
Firefox and Chrome and Playwright/Selenium/Puppeteer/Cypress WebKit in place of Safari; the `.devcontainer` here
builds the extension with its native dependencies. Ask before building an environment on a machine that was not set
up for one (Miniforge serves a host with a closed package manager; keep the system `libgbm.so` for GBM on NVIDIA and
other GPUs) and take the operator's directives on how it is constructed and constrained. Say which checks could not
run where the hardware for them was not available.

Note that parity between X11 and Wayland, as well as between WebSockets and WebRTC, or between the default dashboard
and the wish dashboard, is considered a key focus (things that were not wired up correctly on either side, and similar
discrepancies, are subject to fixes or deduplication). I prefer deduplicating code that performs similar purposes
across different modes over keeping duplicate code for no reason and more fragility. Refactor through deduplication if
you are confident there will be no regressions (or able to validate regressions). Screen coroutine usage in both
Python and JavaScript, as well as thread usage in all languages, so that everything is performant and does not lead to
hanging or lagging. Performance preservation or improvements such as zero-copy and latency-reducing measures are
always important, and the GIL is held no longer than the work needs. End-to-end latency and an unrestricted frame
rate are separate goals rather than two ends of one dial: neither is spent to buy the other. A change never drops a
capability or falls back to an older implementation to make itself simpler; where one seems to be in the way, say
what it is rather than removing it. Note that compatibility should be ensured for Python 3.9 to 3.14 or even higher, and CUDA/NVENC 11
to 13 or higher. Protocol clients form fallback ladders that bind the newest architecture first (ext- before
zwlr-data-control in dcclient) and exist to keep selkies' Wayland path subprocess-free — they replace wtype/wl-copy
style forks, so extend them in-process rather than shelling out. A nested KWin session forwards no delta from its host
seat, so relative pointer motion reaches it through `org_kde_kwin_fake_input` on the app compositor socket
(`wayland/ficlient.rs`); wlroots sessions take the seat's `zwp_relative_pointer_v1` as before. Update the translations as well (and write/update additional entries if necessary) as necessary.
A defect that predates the change you are making is still in scope: finding it does not make it someone else's,
and "pre-existing" is not a reason to leave it. Fix it, or say precisely what is broken, what you ruled out, and
what you would do next. The same applies to a failure you cannot reproduce yet -- narrow it until it is either
fixed or precisely described, and never let a test that fails for an unknown reason pass unremarked.

A change is ready when four questions have answers, and the commit or pull request gives them to the reviewer:
was the defect, or the missing behavior, reproduced on the code before the change (a failing check or a measurement
on the old tree, not an argument from the source); is it gone, or present, on the exact code being committed, through
the path a user takes rather than a switch a user would never flip (a developer toggle, a debug key, a knob of the
rig); can the change affect behavior it was not aimed at, and what was run to know; and is the change stripped to what
makes it work, since every line the first two answers do not need is noise the maintainers have to sift. A change in an
area a maintainer has said they are working on goes to a branch and a pull request carrying those answers, never
straight to `main`, whatever standing permission to push `main` exists. An issue is closed by a
maintainer, never by you. A pull request's `Closes` keyword is not you closing it; the maintainer's
merge is. An optional path another component may offer
(a protocol a compositor advertises, a driver feature, a device) is taken only when its presence is detected and never
as the default: that it is exposed is not proof it works, and a reviewer has to be able to tell what runs where.

The codec of a capture is `CaptureSettings.codec` (`jpeg`, `h264`, `h265`, `vp8`, `vp9`, `av1`), the
`encoders::Codec` enum in Rust: it carries the wire id (the high nibble of a `0x04` frame's type byte, the
low nibble being the frame kind), the per-codec quantizer domain the shared `video_crf` index maps onto, the
level ladders and the bitstream reads that label frames. A session advertises its stream's level from the
shared ladder at the current geometry (`codec.rs`), the lowest a decoder is asked to accept, so a hardware
decoder that gates on the level — older Apple and Intel parts refuse a level above their ceiling even for a
picture they could hold — takes the stream; NVENC re-declares it with a forced IDR on each in-place resize, and
holds AV1 alone at the resize headroom's level, which NVENC validates its session against at init. JPEG and
H.264 may stripe (`encoders/software.rs`);
every other codec streams whole frames. Every session that can name what a frame predicts from
does (`encoders/reference.rs`, `StripeFrame.reference_frame_id`), and `invalidate_reference`
leaves a frame a consumer lost out of the predictions so recovery costs no keyframe: NVENC where
the device reports reference-picture invalidation, libx264 always, and the stream declares the
decoded picture buffer its level admits, or the eight AV1 fixes whatever the level. A session that cannot refuses, and the caller forces an
IDR instead; an H.264 session answers a loss covering the frame at its `frame_num` wrap with a key frame itself,
since FFmpeg's decoder derives the picture order past that gap wrongly and withholds every picture after it. Every full-frame session is chosen by one ladder,
`encoders::select_frame_encoder` (Tegra's vendor encoder where its library answers, then NVENC on the NVIDIA
driver, VA-API otherwise, then a stateful V4L2 memory-to-memory device, then the codec's software encoder, then
a demotion to H.264), shared by X11, Wayland zero-copy, and Wayland readback.
The V4L2 step comes after the render-node probes, since a machine carrying either backend never reaches it, and
the boards it serves -- the Raspberry Pi's `bcm2835-codec`, RK356x's hantro, i.MX8M's VPU -- publish no driver
for those probes to select. It is reached by asking for the interface (`V4L2_CAP_VIDEO_M2M` and the codec's
fourcc on the capture queue) rather than by naming a board, so a device nobody here has is served on the same
path, and the size is checked before the node is opened because a refusal afterwards leaves a session that
produces nothing. Both backends take the codec as
a parameter rather than carrying a second copy of the interface: the queues, controls and surface formats are
the same whichever coded format the capture queue is set to, so a device that advertises H.265 serves it there.
Only the picture type is codec-specific, because an H.265 NAL header is two bytes where H.264's is one. `encoders/nvenc.rs`
is codec-parameterized (H.264, HEVC, AV1; a codec the GPU lacks is refused at open). `encoders/avcodec.rs` is
the libavcodec session: VA-API for all five codecs (a 4:4:4 session tries the surface formats the
driver allocates and its video processor renders, read through libva's `VAProfileNone`
configuration, until one survives the surface pool, the convert, and the codec open: Intel's iHD
allocates planar 444P but its VPP writes 4:4:4 only packed, as XYUV, and its HEVC 4:4:4 entry point
takes only what the VPP writes), and the software HEVC (x265 with the `gpl` feature, else
kvazaar), VP8/VP9 (libvpx) and AV1 (SVT-AV1) encoders the linked FFmpeg carries — probed once
(`encoders::software_encoder`, exported as `pixelflux.SOFTWARE_ENCODERS`), never assumed. Software H.264 is
resolved at build time, never by a setting: the default `gpl` feature makes libx264 the encoder behind every
CPU H.264 session (striped and full-frame), and a build without it (`PIXELFLUX_ENABLE_GPL=0` →
`--no-default-features --features openh264`) puts Cisco OpenH264 behind the same striped path
(`encoders/oh264.rs`, one instance per stripe) with the same wire framing; selkies derives its rate-control
default from the exported names. Every session, on every backend, converts with the BT.709 matrix — the sRGB
desktop's own primaries and transfer — at limited range for 4:2:0 and full range for the software
4:4:4 sessions, and declares it; VP8 is the exception its bitstream forces, one color-space bit
whose only defined value is BT.601 — told BT.709 out of band, Firefox reads neither the decoder
configuration nor the RTP color-space extension and inverts the other matrix. A device that
converts in fixed function is the other exception, and only where it is probed and found to
discard the colorimetry it is handed: that session declares the matrix and range the firmware
produced rather than the one asked for, and one that honors the fields converts BT.709 limited
like every other. What a session says of itself therefore comes from `session_full_range`, never
from whether its encoder is hardware, so no two descriptions of one session can disagree. Chroma sits at
the center of each 2x2 block, which NVENC's own conversion does not: it weights the two columns
of a block 3:1 (its matrix follows what the session declares, measured on Volta and Pascal, so
only the siting is at stake). A 4:2:0 session therefore converts with `ChromaConvert`, a PTX
kernel the driver JIT-compiles (`encoders/argb_to_nv12.cu`, `scripts/build-ptx.sh`) that reads
the packed surface, a pitch-linear dmabuf import or a texture over an array-typed one and writes
the NV12 NVENC encodes; 4:4:4 subsamples nothing and keeps the hardware conversion, as does a
driver that refuses the kernel.
`AvDecoder::color_tags` reads what a stream declares, and the unit tests hold each
encoder to it; the sequence headers travel with every IDR so a client joining or resynchronizing on any key
frame can decode, which `encoders/v4l2m2m.rs` keeps true itself for the devices whose drivers will not
(`REPEAT_SEQ_HEADER` is asked for and the parameter sets are put back where it is refused), since neither
FFmpeg's nor GStreamer's M2M encoder guarantees it. Encoder settings are chosen by measured latency first, frame rate second, quality third and
bitrate last: every software encoder runs at the fastest setting its library offers in real time (x264
ultrafast, VP8 speed 16, VP9 speed 8 with screen tuning, SVT-AV1 preset 11 in its real-time mode, x265
ultrafast with wavefront threads) and NVENC at preset P3 with two-pass quarter-resolution rate control
(`gpu_bench_tuning` measures the alternatives); the VP8, VP9 and AV1 quantizer tables in `codec.rs` were
measured at those settings and must be re-measured whenever they change (`encoders::codec` documents the
method). VP9 carries 4:4:4 as profile 1 at the same limited range as its 4:2:0, so the decoder hint the
client sends for it stays true. The CBR
sessions of x264 and x265 cap the quantizer at 51: both libraries default to an out-of-spec range above
it that forces macroblock skips on a VBV underflow, which freezes rows of a screen for a few frames, so a
budget the content cannot meet overshoots instead, as NVENC and libvpx do. Test both configurations (`cargo test --lib` and
`cargo test --lib --no-default-features --features openh264`, the latter against an FFmpeg carrying
`libkvazaar`); the OpenH264 crates are also dev-dependencies so its tests run under the default build. The
wheel recipe (`pyproject.toml`) builds kvazaar, libvpx, SVT-AV1, dav1d and, for the GPL wheel, x264 and x265
from source ahead of FFmpeg.
The crate's `Cargo.toml` is the one place the version lives: `setup.py` reads it, spelling a semver pre-release
the PEP 440 way (`2.1.0-rc.1` is `2.1.0rc1` to pip), and the release workflow stamps the tag into the manifest
and the lock, so a build ahead of a release carries the series version and a release the tag's.

Host capture of an external Wayland compositor (`wayland/host.rs`, `wayland_host_display`) picks each
rung per capability from the host's registry, never by a setting: frames through
`ext-image-copy-capture` or `wlr-screencopy` into pixelflux-allocated dmabufs, else through an
xdg-desktop-portal RemoteDesktop/ScreenCast session whose PipeWire streams are imported where they lie
(`wayland/portal.rs` over the pure-Rust zbus client, `wayland/pwcapture.rs` over the run-time
libpipewire binding shared with the webcam sink in `pipewire.rs`); keyboard and pointer over libei
(`wayland/eiclient.rs`, the pure-Rust `reis` client) where the backend answers `ConnectToEIS`, since
it carries keyboard, pointer, and touch on one socket, else the virtual-keyboard and virtual-pointer
protocols where the host offers them, else kernel uinput devices where `/dev/uinput` is writable, and
the portal's own `Notify*` methods by keysym last. That order is measured rather than assumed: a
socket write to the compositor runs about 4 us, a uinput write reaching its evdev node about 12 us,
and a `Notify*` D-Bus call about 2 ms. A successful `ConnectToEIS` makes the session refuse `Notify*`, so
libei is committed to only once its handshake binds a device, and the compositor's own keymap
(delivered over EIS, read-only) resolves each key's base keysym with a raw-keycode fallback. A portal
that refuses those devices is asked again for capture alone. The KDE 5.27 session the sandbox can run (`kwin_wayland --virtual` with
`xdg-desktop-portal-kde` on a private bus) is the real non-wlroots target for the portal rung; it shows
no consent dialog to an unsandboxed app and offers memfd frames only, so the dmabuf import of a portal
stream is verified against GNOME or KDE 6 on a GPU host.

X11 capture has three backends and picks between them itself (`x11::run_capture`): NvFBC
(`x11/nvfbc.rs`) where the NVIDIA driver composites the screen into video memory and the buffer is
registered with NVENC in place, which is zero-copy and the lower-latency path; DRI3
(`x11/dri3.rs`) on any server whose screen lives on the GPU (XLibre's Xvfb with glamor, an Xorg
on a DRM driver), where the server blits the root into dmabufs pixelflux allocated through GBM on
the server's own render node and wrapped as pixmaps, the blit is waited for with a one-pixel
`GetImage`, the XFixes cursor is composited by the server through Render, and the hardware session
imports each dmabuf in place through the same `encode_dmabuf` the Wayland zero-copy path uses; and
the general XShm path otherwise. Each zero-copy backend is declined -- with one line saying why --
for a codec its engine does not serve, software encoding, a server or device that does not qualify
(DRI3 also asks that the server draw on the encode node and that the encoder read the first frame),
and NvFBC for a non-NVIDIA encode node, a watermark or a driver without it; DRI3 composites a
watermark through Render as it does the cursor, so that one costs it no readback. There is no
setting either way: the server's and the driver's own answers decide, and a host without NvFBC
pays about 7 ms once per capture start to find that out.
All three publish a change as it lands rather than on the next tick: NvFBC because the driver
generates a frame on damage and the grab waits for it, DRI3 and XShm because the X server's
DAMAGE reports end the wait early. How far any of them may come early is the frame pacing the
Wayland backend also keeps (`pace.rs`), so the rate stays the configured one. The NvFBC
structures are hand-written FFI checked against the SDK by the layout and version assertions
in that module, and `libnvidia-fbc.so.1` is
loaded at run time like NVENC's library. The hardware checks are `#[ignore]`d
(`cargo test gpu_nvfbc -- --ignored --nocapture --test-threads=1` with `DISPLAY` on an NVIDIA X
server; `cargo test gpu_dri3 -- --ignored --nocapture --test-threads=1` with `DISPLAY` on a
server whose screen lives on the GPU and a hardware encoder on the same device). They capture and
paint the root of whatever `DISPLAY` names, so on a host where that is a live desktop a `gpu_`
sweep runs with `--skip nvfbc --skip dri3` and those checks run only against an X server of their
own; `x11::gpu_test_support` holds the painting and decoding helpers they share.

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

Logging follows one rule on every backend, because the combined selkies log is what a remote user pastes
back: a plain `println!` tagged `[X11]`, `[Wayland]`, `[HostCapture]` or `[pixelflux]` for what an operator
reads (the encoder chosen and the GPU it runs on, the zero-copy or readback decision and why, each fallback,
the `Stream settings active` line that names the mode), `eprintln!` for warnings and errors, and
`crate::log::debug!` (`src/log.rs`) for everything behind those lines — device enumeration, each dmabuf
import, the per-second rate counters, an in-place reconfigure — which prints only while the switch a
capture's `debug_logging` sets is on (or `PIXELFLUX_DEBUG=1`). A line says what was decided, not which
function ran, names the output and the size it acts on, and appears once per decision, never per frame or
per buffer; the `[X11]` and `[Wayland]` lines of the same decision read the same apart from the tag, and the
one `Stream settings active` builder (`log_stream_settings`) serves every path. The strings selkies' suites
wait on (`Stream settings active`, `Socket listening on:`, `Configuring Output`, `[Wayland] Output`) are
contracts; a change to one changes the test with it.

What those lines tell an operator, a caller reads as values (`src/report.rs`, `ScreenCapture.stream_info` and
`stream_stats`), which is what selkies shows a user who will never see the log. Each decision is recorded where
it is made and logged, into the report bound to the deciding thread (`report::enter`), so a new capture path, a
new fallback or a new reason a path is declined records itself beside its log line, and the counters stay one
tally per delivered frame, never a callback into Python.

Update this file when certain details change.
