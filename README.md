# pixelflux

[![PyPI version](https://badge.fury.io/py/pixelflux.svg)](https://badge.fury.io/py/pixelflux)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://selkies-project.github.io/pixelflux/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/selkies-project/pixelflux)

**A performant pixel delivery pipeline for diverse sources, blending flexible and high-performance modern encoding formats.**

This module provides a Python interface to a high-performance capture library supporting both **X11** and **Wayland** environments. It captures pixel data, detects changes, and encodes modified stripes into JPEG or H.264.

It encodes JPEG, H.264, H.265, VP8, VP9 and AV1. Every video codec runs on NVIDIA's NVENC (H.264, H.265, AV1) or on VA-API for Intel/AMD GPUs (all five) where the GPU carries it, and otherwise on the software encoder the build resolves for it: x264 or, in a GPL-free build, the BSD-licensed OpenH264 for H.264; x265 or kvazaar for H.265; libvpx for VP8 and VP9; SVT-AV1 for AV1. JPEG and H.264 can be cut into stripes encoded in parallel; the other codecs stream whole frames. **About "zero copy":** the Wayland GPU path is truly zero-copy (dmabuf frames flow GBM → encoder without touching system RAM). The X11 path copies **exactly once**: the X server renders each frame into a shared-memory surface (`XShmGetImage`); the encoder threads then read that mapped surface **in place** and pass the encoded bytes to Python through the buffer protocol without any further copies.

## Installation

pixelflux is a single self-contained **Rust** extension compiled during installation. Both the X11 and Wayland backends, all encoders, and the Python API live in it. (`libjpeg-turbo` — and, in a GPL-free build, `openh264` — C sources are vendored and built by their Rust `-sys` crates, so `cmake` and `nasm` are required, but no system copies of those libraries are used.)

### 1. Prerequisites

Ensure you have the Rust toolchain (`cargo`), Python development files, and the development libraries below.

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build dependencies (Debian/Ubuntu)
sudo apt-get update && \
sudo apt-get install -y \
  git \
  curl \
  python3-dev \
  cmake \
  nasm \
  libclang-dev \
  libavcodec-dev \
  libavutil-dev \
  libx264-dev \
  libgbm-dev \
  libdrm-dev \
  libwayland-dev \
  libinput-dev \
  libxkbcommon-dev \
  libva-dev
```

> **Notes:** the FFmpeg bindings (`ffmpeg-sys-next` 9.0) work with any system **FFmpeg 6.0–9.0** (only `avcodec`/`avfilter` are used: the VA-API encoders, and the software HEVC/VP8/VP9/AV1 encoders that FFmpeg build carries — `libx265` or `libkvazaar`, `libvpx`, `libsvtav1`; a codec whose encoder the build lacks has no software path, which `pixelflux.SOFTWARE_ENCODERS` reports); on distros shipping an older FFmpeg, install a newer build and point `PKG_CONFIG_PATH` at it. `libjpeg-turbo` is vendored and built statically by its crate — **no `libturbojpeg` system package is needed** (only `cmake` + `nasm`). X11 capture uses pure-Rust XCB; colorspace conversion is pure-Rust and the NVENC/CUDA libraries are loaded at runtime (no compile-time NVIDIA packages).
>
> **GPL component (`libx264`):** software H.264 uses the system `libx264` (GPL-2.0+), which is the only GPL-licensed dependency **of pixelflux itself**. It is enabled **by default**; to build without it, set `PIXELFLUX_ENABLE_GPL=0` (or `=false`) before `pip install`. The build then substitutes the BSD-licensed Cisco OpenH264 (vendored, built from source) as the software H.264 encoder behind the very same API and wire format — striped and full-frame sessions, CRF and CBR, live bitrate/quality changes all keep working, and `libx264-dev` is not required. What you lose is 4:4:4 software H.264: OpenH264 is 4:2:0-only, so `video_fullcolor` is encoded 4:2:0 on the CPU (NVENC still carries it). Software H.265 follows the same switch through the linked FFmpeg: x265 (GPL) in the default build, kvazaar (BSD) otherwise. JPEG, NVENC and VA-API are unaffected. `pixelflux.SOFTWARE_ENCODERS` maps each codec to the software encoder the build in use carries (`{"h264": "x264", "h265": "x265", "vp8": "libvpx", "vp9": "libvpx", "av1": "svt-av1"}` for the wheels), and a notice is printed at install time whether GPL components are enabled or not.
>
> **Caveat (transitively-linked x264):** the extension links the *system* FFmpeg (`libavcodec`/`libavfilter`) for VA-API, and many distro FFmpeg builds (e.g. Ubuntu/Debian's) are themselves compiled with `--enable-libx264`, so their `libavcodec` drags `libx264` in as a transitive shared-library dependency even when pixelflux was built GPL-free. pixelflux contains no x264 code in that case (verified: no `x264` symbols or `NEEDED` entries), for a deployment that must be x264-free end to end, use an FFmpeg built without `--enable-libx264` (the project's non-GPL wheel builds FFmpeg n8.1 LGPL-only with kvazaar, libvpx, SVT-AV1 and dav1d; the GPL wheel's FFmpeg adds x264 and x265 under `--enable-gpl`).
>
> **Official wheels are always GPL-enabled** (x264 and x265 as the software H.264 and H.265 encoders, a GPL-built FFmpeg); the `PIXELFLUX_ENABLE_GPL=0` path is for verified license-minimal source builds. The AppImage distribution bundles the LGPL-only FFmpeg variant so the optional GPL-free posture holds end-to-end.

### 2. Hardware Acceleration (Optional but Recommended)
*   **NVIDIA (NVENC):** The library detects the NVIDIA driver at runtime. No extra compile-time packages are needed.
*   **Intel/AMD (VA-API):** Ensure `libva-dev` and `libdrm-dev` are installed. You must also have the correct drivers (e.g., `intel-media-va-driver-non-free` or `mesa-va-drivers`).

### 3. Install the Package

**Option A: Install from PyPI**
```bash
pip install pixelflux
```

Prebuilt wheels are published on the GitHub Releases page (`manylinux_2_28` and `musllinux`, x86_64 and aarch64, CPython 3.9–3.14). PyPI serves them automatically on supported platforms; on other platforms pip builds from source with the prerequisites above.

**Option B: Install from local source**
```bash
# From the root of the project repository
pip install .
```

## Usage

### Backend Selection

`pixelflux` supports both an X11 and a **Wayland** backend (the latter built on [Smithay](https://github.com/Smithay/smithay)), selected per capture by the `use_wayland` attribute on `CaptureSettings`:

- `settings.use_wayland = True` — force the Wayland backend
- `settings.use_wayland = False` — force the X11 backend
- `settings.use_wayland = None` (default) — use Wayland when the session exposes a `WAYLAND_DISPLAY`, otherwise X11

To test launching programs into this backend simply add `WAYLAND_DISPLAY=wayland-1` before launching them: 

```bash
WAYLAND_DISPLAY=wayland-1 glmark2-es2-wayland -s 1920x1080
```

### Host capture (external compositors)

Setting `wayland_host_display` to another compositor's socket captures **that** session instead
of the built-in one, with input injected through the virtual keyboard/pointer protocols. Frames
are fetched with `ext-image-copy-capture-v1` when the host offers it (wlroots 0.19+, KWin 6.2+,
COSMIC) and `zwlr-screencopy-v1` (v3) otherwise, so any wlroots-era or KDE compositor works;
`PIXELFLUX_HOST_CAPTURE=zwlr` forces the fallback for triage. Both protocols share the same
buffer plan: with a GPU the host blits into GBM dmabufs that the encoder imports directly (no
CPU copy anywhere), a host that cannot import our dmabufs (other GPU, software renderer) is
captured through shm instead, and both are damage-gated so a static screen costs nothing.

Displays map onto host outputs by rank, so multi-display capture needs the host to expose that
many outputs: `ScreenCapture.output_capacity()` reports the host's output count (`-1` when
self-compositing, where outputs are created on demand), and `create_output` refuses ids the
host cannot back rather than minting an output that would never receive a frame.

The built-in compositor also **serves** `ext-image-copy-capture-v1` (with per-output sources),
so standard capture tools — or another pixelflux — can record a pixelflux session: dmabuf
clients are filled by one GPU blit from the composited frame, shm clients by one readback, and
an unchanged screen holds the frame instead of duplicating it.

### Automatic GPU Selection

Set the `auto_gpu` attribute on `CaptureSettings` to let pixelflux pick a render node
automatically instead of supplying one — `"true"` (or any truthy value) picks the first GPU, and
a token (a kernel driver name, PCI **vendor** id, or devicetree prefix) picks the first GPU that
matches. It enumerates `/sys/class/drm`, pairs each `cardN` with its `renderD*` node by PCI
device, and skips non-GPU cards (IPMI/VGA). Selection is **driver-aware**: NVIDIA nodes are routed
to NVENC, while Intel (`i915`) and AMD (`amdgpu`) nodes take the VA-API path. Both the X11 and
Wayland backends honor it. (Selkies fills this from its `--auto-gpu` / `SELKIES_AUTO_GPU` inputs.)

```python
settings.auto_gpu = "true"
```

When auto-selection is off, the encoder device is chosen by `encode_node_index` (default `-2` =
auto): `-1` forces software and `>= 0` selects `/dev/dri/renderD(128 + index)`; an explicit
`encode_node_path` / `render_node_path` takes precedence.

### Capture Settings

The `CaptureSettings` class configures both backends.

```python
from pixelflux import CaptureSettings, ScreenCapture

settings = CaptureSettings()

# --- Core Capture ---
settings.capture_width = 1920
settings.capture_height = 1080
settings.capture_x = 0
settings.capture_y = 0
settings.capture_cursor = True
settings.target_fps = 60.0
settings.scale = 1.0  # Fractional scaling (Wayland only)
settings.wayland_host_display = ""                  # Capture from an EXTERNAL compositor instead of the built-in one (host-capture mode)

# --- Codec ---
# "jpeg" (striped stills), "h264" (striped, or full-frame with video_fullframe), or a
# full-frame video codec: "h265", "vp8", "vp9", "av1"
settings.codec = "h264"
# Force CPU encoding and ignore hardware encoders. The software encoder behind each codec
# is the build's (pixelflux.SOFTWARE_ENCODERS): x264 or OpenH264 for H.264, x265 or kvazaar
# for H.265, libvpx for VP8/VP9, SVT-AV1 for AV1.
settings.use_cpu = False

# --- Debugging ---
settings.debug_logging = False # Enable/disable the continuous FPS and settings log to the console.

# --- JPEG Settings ---
settings.jpeg_quality = 75              # Quality for changed stripes (0-100)
settings.paint_over_jpeg_quality = 90   # Quality for static "paint-over" stripes (0-100)

# --- Video Settings ---
settings.video_crf = 25                            # Quality index on the H.264 QP scale (0-51, lower is better quality); mapped onto each codec's own quantizer range
settings.video_paintover_crf = 18                  # Quality index for the paintover on static content. Must be lower than video_crf to activate.
settings.video_paintover_burst_frames = 5          # Number of high-quality frames to send in a burst when a paintover is triggered.
settings.video_fullcolor = False                   # Use 4:4:4 chroma instead of 4:2:0 where the codec carries it (H.264 and H.265): software x264/x265 and NVENC take it, VA-API negotiates it per device.
settings.video_fullframe = True                    # H.264 only: encode full frames instead of changed stripes (every other video codec is full-frame)
settings.video_streaming_mode = False              # Bypass all VNC logic and work like a normal video encoder, higher constant CPU usage for fullscreen gaming/videos
settings.keyframe_interval_s = 0.0                   # Periodic keyframe interval in seconds (0 = keyframes only on demand/paint-over)
settings.video_cbr_mode = False                    # Switches to CBR mode and ignores CRF value. Used in conjunction with video_bitrate_kbps.
settings.video_bitrate_kbps = 4000                 # Target bitrate for CBR mode. Required when video_cbr_mode is enabled.
settings.video_vbv_multiplier = 1.5                # Optional CBR VBV size as a multiple of one frame's bit budget (0 = auto: 1.5, or 3 with periodic keyframes).
settings.auto_adjust_screen_capture_size = True   # Allow pixelflux to adjust its capture width and height.

# --- Hardware Acceleration ---
# Encoder device selection:
#   -2: Auto-detect (default; combine with auto_gpu — see Automatic GPU Selection)
#   -1: Force software encoding
#   >= 0: Use the GPU at /dev/dri/renderD(128 + index)
settings.encode_node_index = -2
# Explicit encoder device path; takes precedence over the index above. str or bytes,
# e.g. "/dev/dri/renderD128".
settings.encode_node_path = None
# Explicit compositor render node (Wayland); str or bytes.
settings.render_node_path = None

# --- Wire Format / Zero-Copy (X11) ---
# False (default): prepend the per-stripe header to each packet (the WebSocket path).
# True: emit the raw encoded payload with no header (for a WebRTC path that frames itself).
settings.omit_stripe_headers = False

# --- Change Detection & Optimization ---
settings.video_min_qp = 0                          # CBR QP clamps: 0 = encoder default; max bounds the quality floor, min bounds bit waste on easy content
settings.video_max_qp = 0
settings.use_paint_over_quality = True  # Enable paint-over/IDR requests for static regions
settings.paint_over_trigger_frames = 15 # Frames of no motion to trigger paint-over
settings.damage_block_threshold = 10    # Consecutive changes to trigger "damaged" state
settings.damage_block_duration = 30     # Frames a stripe stays "damaged"

# --- Watermarking ---
# Must be a bytes object. The path to your PNG image.
settings.watermark_path = b"/path/to/your/watermark.png" 
settings.cursor_size_cap = 128                     # Cap out-of-band hardware-cursor PNGs to this longest edge (<= 0 = uncapped)
# 0:None, 1:TopLeft, 2:TopRight, 3:BottomLeft, 4:BottomRight, 5:Middle, 6:Animated
settings.watermark_location_enum = 4 
```

### Input Injection (Wayland Only)

In Wayland mode, `pixelflux` acts as the compositor. You cannot use external tools like `xdotool`. Instead, use the input injection methods provided by the `ScreenCapture` instance:

```python
capture = ScreenCapture()
capture.start_capture(my_callback, settings)

# Inject Mouse Motion (Absolute coordinates)
capture.inject_mouse_move(x=500.0, y=300.0)

# Inject Mouse Button (evdev button codes: 272=Left, 273=Right, 274=Middle)
# State: 1 = Pressed, 0 = Released
capture.inject_mouse_button(btn=272, state=1) 

# Inject Scroll (Vertical/Horizontal)
capture.inject_mouse_scroll(x=0.0, y=10.0)

# Inject Keyboard Key
# scancode: Linux raw keycode (e.g., 17 for 'w')
# state: 1 = Pressed, 0 = Released
capture.inject_key(scancode=17, state=1)
```

### Stripe Callback

Your callback receives a single **`StripeFrame`** object (the same type on both the X11 and
Wayland backends). It supports the buffer protocol — `bytes(frame)` / `memoryview(frame)` /
`len(frame)` — and exposes the stripe metadata as attributes:

```python
def my_callback(frame):
    # frame.data_type      (the codec id: 0=JPEG, 1=H.264, 2=VP8, 3=VP9, 4=AV1, 5=H.265)
    # frame.frame_id
    # frame.stripe_y_start
    # frame.stripe_height
    encoded_data = bytes(frame)          # copy out, or use memoryview(frame) zero-copy (below)
    # Send encoded_data to the client...
```

### Zero-Copy Frames

`memoryview(frame)` aliases the native encoder buffer with **no copy**, on **every supported
Python version (3.9–3.14)**. The frame object owns its buffer and keeps it alive until every
consumer — including a transport that retained a slice during a partial write — has released its
view, so the hand-off is memory-safe. (The old `deferred_free` / `OwnedFrame` / PEP 688 /
Python-3.12-only path is gone; the native buffer protocol does this on all versions.) Hand the
view straight to an async socket; keep the frame referenced for the duration of the send.

```python
def my_callback(frame):
    if frame.data_type == 0 or len(frame) == 0:   # nothing to send
        return
    # Hand BOTH the view and the frame to your sender (e.g. an asyncio.Queue) so the buffer
    # outlives the send: the view pins the frame, which frees the buffer once the view drops.
    queue.put_nowait({"data": memoryview(frame), "owner": frame})
```

See `example/screen_to_browser.py` for a complete queue-based usage.

## Zero-Copy Pipeline (Wayland)

The Wayland backend implements a **Zero-Copy** architecture for hardware encoding.

1.  **Rendering:** The compositor renders the desktop to a GPU buffer (GBM).
2.  **Export:** This buffer is exported as a `Dmabuf` (file descriptor).
3.  **Encoding:** The `Dmabuf` is imported directly into the encoder context (NVENC or VA-API) without ever copying pixel data to system RAM (CPU).

**Performance Note:** Software (Pixman) rendering, the absence of a hardware encoder, or utilizing a render node different from the encoding node will force a "Readback" fallback, copying pixels to the CPU and breaking the zero-copy chain (higher latency and CPU load). A watermark does **not** force readback — on the GPU path it is composited into the frame before encoding.

## Built-in MP4 Recorder

For convenience, the extension ships its own fragmented-MP4 muxer (no `avformat` dependency) with the `start_recording(...)`, `stop_recording()`, and `recording_status()` Python functions, controllable through the `PIXELFLUX_RECORD*` environment variables. Recording taps the encoded full-frame H.264 stream, and HTTP endpoints allow remote trigger/stop/status.

## Recording Sink

The capture session can output the raw video stream directly to a Unix domain socket for external recording: Annex-B for H.264 and H.265, an OBU stream for AV1, and IVF for VP8 and VP9.

*Note: This feature requires full-frame video encoding and does not work with JPEG or striped H.264 modes.*

```python
# Enable the unix socket (forces IDR frames every 30 frames and on connect)
settings.recording_socket = "/tmp/pixelflux_record"
```

You can then capture the stream using `ffmpeg`:

```bash
# Raw copy
ffmpeg -f h264 -i unix:///tmp/pixelflux_record -c:v copy test.h264

# Re-encode for a clean MP4
ffmpeg -f h264 -framerate 60 -i unix:///tmp/pixelflux_record -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p test.mp4
```

## Virtual Camera

`VirtualCamera` turns a client's webcam uplink into a V4L2 capture device for applications. Encoded frames of any
browser codec — H.264, VP8, VP9, AV1, HEVC (WebCodecs or a WebRTC media track) and MJPEG (the canvas fallback) — are
pushed in; a worker thread decodes them (libavcodec, TurboJPEG), fits them into the device's fixed format (raw
I420 by default, NV12 or YUYV; or MJPEG, a compressed device that carries an MJPEG uplink's frames as received,
decoding nothing, and re-encodes only frames that must be fitted), and publishes every frame to the configured sinks at once:

- a shared-memory ring served over a Unix socket to the Selkies V4L2 interposer (`LD_PRELOAD`, no privileges, no
  kernel module), which presents `/dev/videoN` to the application;
- a v4l2loopback output device (`device_path`) on hosts and privileged containers that have the module, where
  applications need no preload at all;
- a PipeWire `Video/Source` node (`pipewire`) when a daemon is reachable, for PipeWire-native consumers and the
  `pipewire-v4l2` wrapper (`libpipewire-0.3` is loaded at run time; nothing is linked at build time).

```python
from pixelflux import VirtualCamera, VirtualCameraSettings

settings = VirtualCameraSettings()
settings.socket_path = "/tmp/selkies_webcam0.sock"   # what the interposer connects to
settings.width, settings.height = 1280, 720          # frames are scaled and letterboxed to fit
settings.pixel_format = "I420"                       # "I420", "NV12", "YUYV" or "MJPEG" (an MJPEG uplink passes through)
settings.device_path = "auto"                        # "", "auto", or a /dev/videoN to mirror into
settings.pipewire = True                             # publish the PipeWire node when a daemon is reachable

cam = VirtualCamera()
cam.start(settings)
flags = cam.push(h264_frame, VirtualCamera.CODEC_H264, keyframe=True)   # any buffer object; returns at once
if flags & VirtualCamera.KEYFRAME_WANTED:
    ...  # the decoder lost its reference (dropped frame, late start): ask the client for a keyframe
print(cam.stats())   # pushed/decoded/published/dropped/skipped/errors, geometry, clients, device_path
cam.stop()
```

`push(data, codec, keyframe=False, offset=0)` copies the encoded bytes out of the buffer and returns; decoding never
runs on the caller's thread and no worker ever calls back into Python. A full decoder queue drops the oldest frame
and, for inter-coded codecs, waits for the next keyframe rather than decoding against a missing reference.
`VirtualCamera.shm_layout()` reports the ring's byte layout for the interposer's ABI test.

## Computer Use Interface (X11 and Wayland)

Both backends implement the [Anthropic Computer Use specification](https://github.com/anthropics/claude-quickstarts/tree/main/computer-use-demo), providing an HTTP API for AI agents to control the desktop. On Wayland the compositor injects input natively; on X11 the existing display is driven through XTEST with root-window screenshots. Enable it by setting the `PIXELFLUX_CU` environment variable to the port the server should listen on:

```bash
export PIXELFLUX_CU=5000
```

A bare port is served on the loopback addresses only (`127.0.0.1,::1`); the API carries no authentication, so open it to other hosts deliberately by naming the addresses to listen on as comma-separated `host:port` entries, `PIXELFLUX_CU=0.0.0.0:5000,[::]:5000` for every interface. The same value is what `start_computer_use()` takes when a script starts the server itself.

When using Computer Use, call `ensure_wayland_display()` before starting a capture to bring the compositor socket up early — this lets apps launched alongside your script connect to `WAYLAND_DISPLAY` immediately. GPU auto-selection (`auto_gpu` on `CaptureSettings`) works normally; the screenshot path forces a single-frame CPU readback when the GPU is in zero-copy mode.

The Computer Use server listens for `POST` requests on `/computer-use` and responds with JSON. Unless otherwise noted, successful actions return:

```json
{"result":"ok"}
```

Coordinates are specified in absolute framebuffer pixels. Any coordinates outside the framebuffer are automatically clamped to the nearest valid pixel.

### Actions

All actions are `POST` requests to `/computer-use` with a JSON body.

**`screenshot`** - Capture the current display as a base64-encoded PNG:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"screenshot"}' | jq -r '.data' | base64 -d > screen.png
```

**`mouse_move`** - Move the cursor to absolute pixel coordinates:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"mouse_move","coordinate":[500,300]}'
```

**`left_click`** / **`right_click`** / **`middle_click`** - Click a mouse button, optionally at a coordinate and/or while holding a keyboard modifier:

```bash
# Simple click
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"left_click"}'

# Right click at a specific position while holding Shift
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"right_click","coordinate":[800,600],"text":"shift"}'
```

**`double_click`** / **`triple_click`** - Perform multiple left mouse clicks, optionally while holding a modifier:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"double_click","coordinate":[400,300]}'

curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"triple_click","text":"ctrl"}'
```

**`left_click_drag`** - Press the left mouse button at `start_coordinate`, drag to `coordinate`, then release:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"left_click_drag","start_coordinate":[100,100],"coordinate":[500,300]}'
```

**`left_mouse_down`** / **`left_mouse_up`** - Press or release the left mouse button without moving the pointer:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"left_mouse_down"}'
```

**`type`** - Type a string of text:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"type","text":"Hello, world!"}'
```

**`key`** - Press a key or key combination. Key combinations are specified using `+` separators:

```bash
# Single key
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"key","text":"Return"}'

# Key combination
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"key","text":"ctrl+s"}'

curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"key","text":"ctrl+alt+Delete"}'
```

**`hold_key`** - Hold a key for the specified duration (seconds). Durations are capped at 100 seconds.

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"hold_key","text":"ctrl","duration":2.0}'
```

**`scroll`** - Scroll vertically or horizontally, optionally at a coordinate and/or while holding a keyboard modifier:

```bash
# Scroll down 3 clicks
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"scroll","scroll_direction":"down","scroll_amount":3}'

# Scroll at a position while holding Shift
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"scroll","coordinate":[500,400],"scroll_direction":"up","scroll_amount":5,"text":"shift"}'
```

**`cursor_position`** - Return the current cursor position:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"cursor_position"}' | jq -r '.text'
# → X=500,Y=300
```

**`wait`** - Pause execution for the specified duration (seconds). Durations are capped at 100 seconds.

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"wait","duration":0.5}'
```

**`zoom`** - Capture and return a cropped base64-encoded PNG of the specified framebuffer region (`[left, top, right, bottom]`):

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"zoom","region":[100,200,400,350]}' | jq -r '.data' | base64 -d > zoomed.png
```

## NVIDIA NVENC (X11)

*   **Multi-GPU containers:** When several GPUs are exposed to a container, NVENC is filtered
    in-process to the GPU you selected (no separate `LD_PRELOAD` shim is required). Verified on
    NVIDIA drivers 570–595.
*   **4:4:4 (High 4:4:4):** Set `video_fullcolor = True` to encode full-chroma H.264 via NVENC
    (`video_fullcolor` codec), in addition to the software path. See
    [VA-API 4:4:4](#va-api-444) for how the same request is resolved on VA-API.
*   **Force a keyframe on demand:** `capture.request_idr_frame()` forces an IDR frame, e.g. when
    a client reconnects or its decoder is reset. It routes to whichever encoder is active
    (NVENC, VA-API, or software) and is a no-op while no capture is running.

### NVENC color conversion

NVENC encodes the captured ARGB directly, so there is **no CUDA Toolkit / NVRTC requirement** —
only the NVIDIA driver runtime (`libnvidia-encode`, `libcuda`), which is loaded at runtime.
The driver's ARGB→YUV hardware conversion is fixed at BT.601 limited range (no encode-session
flag retargets it), so pixelflux declares exactly that in the VUI — BT.709 primaries and
transfer for the sRGB desktop source, SMPTE 170M matrix, limited range — and uses the same
matrix when it converts on the host-planar path, so clients decode correct colour either way.
Nothing extra to install at build or runtime beyond the driver.

## VA-API 4:4:4

`video_fullcolor = True` is carried into the VA-API session rather than ruled out in advance. The
encoder asks the device which 4:4:4 surface format it holds (planar `yuv444p` is preferred, since
the readback path uploads its I444 buffer to that one untouched; packed `vuyx` is taken when it is
all a driver offers), builds the surface pool and the `scale_vaapi` convert around it, and lets
FFmpeg match a profile to that format instead of pinning `high`.

Three layers can refuse, and each says so in the log line that precedes the fallback: the driver
carrying no 4:4:4 surface format, the driver refusing to allocate one, and `h264_vaapi` having no
profile that matches it. **On every current driver the third is what answers**: H.264 4:4:4 has no
`VAProfile` in libva at all, so FFmpeg's `h264_vaapi` advertises only 4:2:0 profiles (plus 10-bit
4:2:0 from libva 1.18). A refusal falls back to the software path, where x264 does carry 4:4:4 —
the request is honoured, on the CPU, rather than silently downgraded to 4:2:0 (a GPL-free build's
OpenH264 is 4:2:0-only and says so in the log).

Nothing here is pinned to that state of affairs: a driver and FFmpeg build that gain H.264 4:4:4
start using it with no code change, and `Colorspace:` in the stream log always reports what the
session settled on rather than what was asked for.

## Features

*   **Dual Backend (one Rust extension):**
    *   **X11:** XShm capture via pure-Rust XCB, with XFixes cursor and watermark compositing.
    *   **Wayland:** Modern, secure, headless compositor based on [Smithay](https://github.com/Smithay/smithay).
*   **Flexible Encoding:**
    *   **Software:** H.264 through x264 (incl. 4:4:4 — GPL, the default) or, in a GPL-free build, the BSD-licensed OpenH264 (4:2:0), and JPEG — both with multi-threaded striping; full-frame H.265 through x265 (incl. 4:4:4) or kvazaar, VP8 and VP9 through libvpx, AV1 through SVT-AV1, all through the linked FFmpeg; `pixelflux.SOFTWARE_ENCODERS` names the build's encoder per codec.
    *   **Hardware:** NVIDIA NVENC (H.264, H.265 and AV1; incl. 4:4:4 for H.264 and H.265, ARGB-direct with matched VUI colour signaling, multi-GPU containers, API-version negotiation) and VA-API (Intel/AMD; H.264, H.265, VP8, VP9 and AV1, VA-VPP convert, per-device 4:4:4 negotiation, low-power entry points) with Zero-Copy support.
    *   **Driver-aware GPU auto-selection** via the `auto_gpu` setting.
*   **Zero-Copy Frames (X11 & Wayland):** the native frame object (buffer protocol) hands the encoded buffer to Python with no copy, on every supported Python version (3.9–3.14).
*   **Smart Bandwidth Management:**
    *   **Change Detection:** Encodes only changed stripes (Software/JPEG mode).
    *   **Paint-Over:** Automatically improves quality for static regions.
    *   **Damage Throttling:** Limits processing during high-motion scenes.
    *   **On-demand keyframes:** `request_idr_frame()` forces an IDR for reconnecting clients.
*   **Input Handling:** Built-in input injection for mouse and keyboard (Wayland; XTEST on X11 via Computer Use).
*   **Cursor Compositing:** Hardware cursor planes or software rendering options.
*   **Dynamic Watermarking:** Overlay PNGs with static positioning or DVD-screensaver style animation.
*   **Recording Sink:** Direct Unix socket output of full-frame video streams (Annex-B, OBU or IVF by codec) for local capture.
*   **Virtual Camera:** A client's webcam uplink (H.264/VP8/VP9/AV1/HEVC/MJPEG) decoded off the GIL into a V4L2 capture device (or passed through as an MJPEG device when the browser sends JPEG), served to the Selkies V4L2 interposer (no privileges) and mirrored into v4l2loopback and a PipeWire node where available.
*   **Built-in MP4 Recorder:** Crash-safe fragmented-MP4 recording without any FFmpeg `avformat` dependency.
*   **AI Agent Control:** Computer Use API to dump screenshots and drive all facets of a desktop environment.

## License

This project is licensed under the **Mozilla Public License Version 2.0**.
A copy of the MPL 2.0 can be found at https://mozilla.org/MPL/2.0/.

Note that the default build links the GPL-2.0+ `libx264` as its software H.264 encoder and reaches GPL-2.0+ x265 through FFmpeg for H.265; build with `PIXELFLUX_ENABLE_GPL=0` to exclude every GPL-licensed component (the BSD-licensed `openh264` and kvazaar then take their places, and the FFmpeg bindings are used LGPL-only).

[LICENSES.md](LICENSES.md) inventories every third-party component of both builds (crates, linked and vendored native libraries, what is loaded at run time) with its license, and describes the check (`scripts/check-licenses.py`, `pixelflux/deny.toml`, the `Licenses` workflow) that keeps the non-GPL build free of copyleft code.
