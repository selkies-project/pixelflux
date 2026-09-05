# Third-party licenses

pixelflux itself is licensed under the [Mozilla Public License 2.0](LICENSE).
This file inventories what a built `pixelflux` extension contains or loads, per
build configuration, and the license of each piece. It is enforced by
`scripts/check-licenses.py`, `pixelflux/deny.toml` and the `Licenses` workflow
(see [How this is enforced](#how-this-is-enforced)).

Categories used below:

| Category | Meaning | Licenses |
| --- | --- | --- |
| copyleft | the whole combined binary must be distributed under the license's terms | GPL-2.0-or-later (libx264) |
| weak copyleft | the library itself stays under its license and must remain replaceable (dynamic linking is fine) | LGPL-2.1-or-later (FFmpeg, glibc, libudev) |
| permissive | attribution only | MIT, BSD, Apache-2.0, ISC, Zlib, BSL-1.0, Unlicense, WTFPL, ... |

MPL-2.0 is file-level copyleft (modified MPL files stay MPL, the larger work
may be under any license) and is grouped with the permissive licenses here.

## Build configurations

| Configuration | How it is selected | Software H.264 encoder | Software H.265 encoder | Cargo features |
| --- | --- | --- | --- | --- |
| default (GPL) | `pip install pixelflux` / the published wheels | libx264 (GPL-2.0-or-later) | x265 (GPL-2.0-or-later), through FFmpeg | `gpl` (default) |
| non-GPL | `PIXELFLUX_ENABLE_GPL=0 pip install .` | Cisco OpenH264 (BSD-2-Clause, compiled from vendored source) | kvazaar (BSD-3-Clause), through FFmpeg | `--no-default-features --features openh264` |

Both configurations share everything else: JPEG, the VP8/VP9 (libvpx) and AV1
(SVT-AV1) software encoders reached through FFmpeg, NVENC, VA-API, capture,
compositor and virtual camera. A build with neither feature does not compile;
a build whose FFmpeg carries x265 but not the `gpl` feature never selects it.

## Native libraries and vendored code

Every component of either build that is not a Rust crate. "How used" is what
was observed on the built extension (`readelf -d`, `ldd`, the strings passed to
`dlopen`) and in the crates' build scripts; "Build" says which configuration
contains it.

| Component | License | Category | Build | How used | Notes |
| --- | --- | --- | --- | --- | --- |
| libx264 (via `x264-sys`) | GPL-2.0-or-later | copyleft | GPL only | linked shared library (`NEEDED libx264.so.*`); auditwheel bundles it into the manylinux wheel, the musllinux wheel takes Alpine's package | Striped software H.264. The only GPL component of pixelflux itself; the `x264-sys` crate is MIT but has no purpose without libx264. |
| Cisco OpenH264 2.6 (via `openh264-sys2`) | BSD-2-Clause | permissive | non-GPL only | compiled from the source vendored in the crate (needs a C++ toolchain and nasm) and linked statically; no binary download | Software H.264 without GPL. Cisco's royalty-covered binary module is irrelevant to a source build; the AVC patent pool applies to any H.264 encoder and is the deployer's concern. Pulls `libstdc++` in as the only C++ code. |
| FFmpeg libavcodec, libavfilter, libavutil (via `ffmpeg-sys-next`), plus libswresample, libswscale and libavformat they depend on | LGPL-2.1-or-later as built for the non-GPL wheels (n8.1, `--enable-shared --disable-static --disable-programs`, no `--enable-gpl`; the libraries report "LGPL version 2.1 or later"); GPL-2.0-or-later as built for the GPL wheels (`--enable-gpl --enable-libx265`); whatever the system FFmpeg is when building from source | weak copyleft (non-GPL wheel), copyleft (GPL wheel) | both | linked shared libraries; the wheels bundle them, a source build links the system FFmpeg | the VA-API encoders (`h264_vaapi`, `hevc_vaapi`, `vp8_vaapi`, `vp9_vaapi`, `av1_vaapi`) and filters, and the software encoders below. A GPL-built system FFmpeg (Debian/Ubuntu, Alpine, conda-forge's `gpl_*` variant) makes the linked set GPL: see [Distribution notes](#distribution-notes). |
| x265 | GPL-2.0-or-later | copyleft | GPL only | linked by libavcodec (`libx265`); bundled into the GPL wheels | software H.265 (incl. 4:4:4) |
| kvazaar | BSD-3-Clause | permissive | both | linked by libavcodec (`libkvazaar`); bundled into the wheels | software H.265 of a GPL-free build (4:2:0) |
| libvpx | BSD-3-Clause | permissive | both | linked by libavcodec (`libvpx`, `libvpx-vp9`); bundled into the wheels | software VP8 and VP9 |
| SVT-AV1 | BSD-3-Clause-Clear (with the Alliance for Open Media patent license) | permissive | both | linked by libavcodec (`libsvtav1`); bundled into the wheels | software AV1 |
| dav1d | BSD-2-Clause | permissive | both | linked by libavcodec (`libdav1d`); bundled into the wheels | the virtual camera's AV1 decoder |
| libva, libva-drm, libva-x11 | MIT | permissive | both | linked by libavutil/libavcodec, not by pixelflux; excluded from the wheel (`auditwheel --exclude`), the host's copy is used | VA-API |
| libdrm | MIT | permissive | both | linked by libavutil; excluded from the wheel. `drm-sys`/`drm-ffi` only carry bindings and issue the ioctls themselves, no libdrm symbol is linked by pixelflux | DRM/KMS |
| libgbm (Mesa) | MIT | permissive | both | linked shared library (`gbm-sys`); excluded from the wheel | GPU buffer allocation |
| libpixman-1 | MIT | permissive | both | linked shared library (`pixman-sys`); excluded from the wheel | software renderer of the compositor |
| libxkbcommon | MIT | permissive | both | linked shared library (`xkbcommon` crate); excluded from the wheel | keymaps |
| libwayland-server | MIT | permissive | both | `dlopen` at run time (`wayland-sys` `dlopen` feature), never linked | capture compositor. libwayland-client is not used: the Wayland client side is the pure-Rust `wayland-backend`. |
| libEGL (Mesa / vendor) | MIT (Mesa), Khronos headers | permissive | both | `dlopen("libEGL.so.1")` at run time for the dmabuf import path, never linked; excluded from the wheel | GL entry points are resolved through `eglGetProcAddress` |
| libpipewire-0.3 | MIT | permissive | both | `dlopen("libpipewire-0.3.so.0")` at run time for the virtual-camera PipeWire sink, never linked | no build-time PipeWire dependency |
| libinput | MIT | permissive | both (graph only) | `input-sys` is in the crate graph through smithay's `backend_libinput`, but no symbol is referenced and the linker drops it: not in `NEEDED` | compositor backend not used by pixelflux |
| libudev (systemd) | LGPL-2.1-or-later | weak copyleft | both (graph only) | `libudev-sys` is in the crate graph through smithay's `backend_udev`; not referenced, not in `NEEDED` | compositor backend not used by pixelflux |
| libjpeg-turbo 3.1 (via `turbojpeg-sys`) | IJG AND BSD-3-Clause AND Zlib | permissive | both | compiled from the source vendored in the crate (cmake + nasm) and linked statically; no system `libturbojpeg` | JPEG encoder and the virtual-camera MJPEG decoder |
| libyuv | not used | - | - | - | `yuv` is a pure-Rust crate (BSD-3-Clause OR Apache-2.0), not a binding to Google's libyuv |
| libxcb, libX11 | not linked | - | - | `x11rb` is pure Rust and opens the X socket itself; no libxcb/libX11 symbol is linked (the `auditwheel --exclude` entries for them are harmless) | X11 capture, XTEST, XFixes cursor |
| NVIDIA NVENC (`libnvidia-encode.so.1`) and CUDA driver API (`libcuda.so.1`) | proprietary (NVIDIA driver) | - | both | `dlopen` at run time, never linked, never shipped; only present on hosts with the NVIDIA driver | NVENC encoder |
| NVIDIA SDK headers (`pixelflux/nvcodec-sys/headers/nvEncodeAPI.h`, NVENCAPI 13.0 from FFmpeg's nv-codec-headers) | MIT | permissive | both | vendored header; the committed `src/bindgen/nvenc.rs` is generated from it | the crate `nvcodec-sys` is MIT OR Apache-2.0 |
| KDE plasma-wayland-protocols XMLs (`pixelflux/protocols/kde-output-device-v2.xml`, `kde-output-management-v2.xml`) | MIT-CMU | permissive | both | vendored protocol descriptions compiled to Rust by `wayland-scanner` at build time; nothing is linked | newer than the copies `wayland-protocols-plasma` bundles, which KWin outgrew (`src/wayland/kdeproto.rs`) |
| CUDA driver API declarations (`pixelflux/nvcodec-sys/src/bindgen/cuda.rs`) | bindgen output of the CUDA toolkit `cuda.h` declarations (the 31 functions pixelflux calls); the toolkit header is under the NVIDIA CUDA Toolkit EULA | - | both | committed FFI declarations only, as cudarc/cuda-sys ship; the toolkit is not needed to build and nothing from it is redistributed | regenerated only with the `regen` feature |
| Linux kernel ABI (`linux-raw-sys`) | Linux-syscall-note (Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT for the crate) | permissive | both | syscall numbers and structs as Rust constants | rustix backend |
| CPython (`libpython`) | PSF-2.0 | permissive | both | extension module: symbols come from the hosting interpreter, nothing is linked | `pyo3` with `extension-module` |
| glibc (`libc`, `libm`, `libpthread`, `libdl`) | LGPL-2.1-or-later | weak copyleft | both | linked shared libraries, as for every program; musllinux wheels use musl (MIT) | C runtime |
| libgcc_s, libstdc++ | GPL-3.0-or-later WITH GCC-exception-3.1 | permissive in effect (the runtime library exception covers linked programs) | libgcc_s: both; libstdc++: whenever OpenH264 is compiled in, the only C++ code | linked shared libraries; excluded from the wheel | GCC runtime |
| smithay (git dependency, rev `5de53056`) | MIT | permissive | both | Rust source, compiled in | the only non-crates.io crate; `deny.toml` allows exactly that repository |

## Rust crates

The crate graph was resolved with `cargo metadata` (normal dependencies only,
Linux targets) for both configurations: 199 crates in the default (GPL) build,
202 in the non-GPL build, 203 distinct crates in total. Every one of them has a
permissive license (MPL-2.0 for `pixelflux` itself); no crate is GPL, LGPL,
AGPL or unlicensed. The only differences between the two sets:

| Crate | License | Build | Why |
| --- | --- | --- | --- |
| `x264-sys` 0.2.3 | MIT (links libx264, GPL-2.0-or-later) | GPL only | `gpl` feature |
| `openh264` 0.9.7, `openh264-sys2` 0.9.7 | BSD-2-Clause (vendors OpenH264, BSD-2-Clause) | non-GPL only | `openh264` feature; dev-dependencies in every build so the encoder is still tested |
| `safe_arch` 1.1.0, `wide` 1.6.0 | Zlib OR Apache-2.0 OR MIT | non-GPL only | dependencies of `openh264` |

License expressions as published by the crates (count of the 203):
MIT OR Apache-2.0 (and spellings of it) 98, MIT 57, MIT OR Apache-2.0 OR Zlib
(and spellings) 12, Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT 6,
BSD-2-Clause 5, BSD-3-Clause 4, Unlicense OR MIT 4, Apache-2.0 3, BSD-3-Clause
OR Apache-2.0 3, ISC 2, BSD-2-Clause OR Apache-2.0 OR MIT 2, and one each of
0BSD OR MIT OR Apache-2.0, Apache-2.0 OR MIT OR Unlicense, WTFPL
(`ffmpeg-sys-next`), CC0-1.0 OR Apache-2.0 (`imgref`), (MIT OR Apache-2.0) AND
Unicode-3.0 (`unicode-ident`), BSL-1.0 (`xxhash-rust`) and MPL-2.0
(`pixelflux`). `scripts/check-licenses.py --markdown` regenerates the full
table below.

<details>
<summary>All 203 crates (Build: both, GPL only, non-GPL only)</summary>

| Crate | Version | License (SPDX) | Category | Build | Native library / note |
| --- | --- | --- | --- | --- | --- |
| adler2 | 2.0.1 | 0BSD OR MIT OR Apache-2.0 | permissive | both |  |
| aligned | 0.4.3 | MIT OR Apache-2.0 | permissive | both |  |
| aligned-vec | 0.6.4 | MIT | permissive | both |  |
| anyhow | 1.0.104 | MIT OR Apache-2.0 | permissive | both |  |
| appendlist | 1.4.0 | MIT | permissive | both |  |
| approx | 0.4.0 | Apache-2.0 | permissive | both |  |
| arg_enum_proc_macro | 0.3.4 | MIT | permissive | both |  |
| arrayvec | 0.7.8 | MIT OR Apache-2.0 | permissive | both |  |
| as-slice | 0.2.1 | MIT OR Apache-2.0 | permissive | both |  |
| ascii | 1.1.0 | Apache-2.0 OR MIT | permissive | both |  |
| atomic_float | 1.1.0 | Apache-2.0 OR MIT OR Unlicense | permissive | both |  |
| av-scenechange | 0.14.1 | MIT | permissive | both |  |
| av1-grain | 0.2.5 | BSD-2-Clause | permissive | both |  |
| avif-serialize | 0.8.9 | BSD-3-Clause | permissive | both |  |
| base64 | 0.22.1 | MIT OR Apache-2.0 | permissive | both |  |
| bit_field | 0.10.3 | Apache-2.0/MIT | permissive | both |  |
| bitflags | 2.13.1 | MIT OR Apache-2.0 | permissive | both |  |
| bitstream-io | 4.10.0 | MIT/Apache-2.0 | permissive | both |  |
| block-buffer | 0.10.4 | MIT OR Apache-2.0 | permissive | both |  |
| bumpalo | 3.20.3 | MIT OR Apache-2.0 | permissive | both |  |
| bytemuck | 1.25.2 | Zlib OR Apache-2.0 OR MIT | permissive | both |  |
| bytemuck_derive | 1.11.0 | Zlib OR Apache-2.0 OR MIT | permissive | both |  |
| byteorder-lite | 0.1.0 | Unlicense OR MIT | permissive | both |  |
| calloop | 0.14.4 | MIT | permissive | both |  |
| cfg-if | 1.0.4 | MIT OR Apache-2.0 | permissive | both |  |
| cgmath | 0.18.0 | Apache-2.0 | permissive | both |  |
| chunked_transfer | 1.5.0 | MIT OR Apache-2.0 | permissive | both |  |
| color_quant | 1.1.0 | MIT | permissive | both |  |
| cpufeatures | 0.2.17 | MIT OR Apache-2.0 | permissive | both |  |
| crc32fast | 1.5.0 | MIT OR Apache-2.0 | permissive | both |  |
| crossbeam-channel | 0.5.16 | MIT OR Apache-2.0 | permissive | both |  |
| crossbeam-deque | 0.8.7 | MIT OR Apache-2.0 | permissive | both |  |
| crossbeam-epoch | 0.9.20 | MIT OR Apache-2.0 | permissive | both |  |
| crossbeam-utils | 0.8.22 | MIT OR Apache-2.0 | permissive | both |  |
| crypto-common | 0.1.7 | MIT OR Apache-2.0 | permissive | both |  |
| cursor-icon | 1.2.0 | MIT OR Apache-2.0 OR Zlib | permissive | both |  |
| digest | 0.10.7 | MIT OR Apache-2.0 | permissive | both |  |
| dlib | 0.5.3 | MIT | permissive | both |  |
| downcast-rs | 1.2.1 | MIT/Apache-2.0 | permissive | both |  |
| drm | 0.14.1 | MIT | permissive | both |  |
| drm-ffi | 0.9.1 | MIT | permissive | both | Linux DRM ioctls (no library): MIT (permissive) |
| drm-fourcc | 2.2.0 | MIT | permissive | both |  |
| drm-sys | 0.8.1 | MIT | permissive | both | libdrm headers (bindings only): MIT (permissive) |
| either | 1.17.0 | MIT OR Apache-2.0 | permissive | both |  |
| equator | 0.4.2 | MIT | permissive | both |  |
| equator-macro | 0.4.2 | MIT | permissive | both |  |
| equivalent | 1.0.2 | Apache-2.0 OR MIT | permissive | both |  |
| errno | 0.3.14 | MIT OR Apache-2.0 | permissive | both |  |
| exr | 1.74.2 | BSD-3-Clause | permissive | both |  |
| fastrand | 2.5.0 | Apache-2.0 OR MIT | permissive | both |  |
| fax | 0.2.7 | MIT | permissive | both |  |
| fdeflate | 0.3.7 | MIT OR Apache-2.0 | permissive | both |  |
| ffmpeg-sys-next | 9.0.0 | WTFPL | permissive | both | FFmpeg libavcodec, libavfilter, libavutil (plus the libswresample, libswscale, libavformat they pull in): LGPL-2.1-or-later (weak copyleft), and through them the codec libraries listed above |
| flate2 | 1.1.9 | MIT OR Apache-2.0 | permissive | both |  |
| gbm | 0.18.0 | MIT | permissive | both |  |
| gbm-sys | 0.4.0 | MIT | permissive | both | libgbm (Mesa): MIT (permissive) |
| gcd | 2.3.0 | MIT/Apache-2.0 | permissive | both |  |
| generic-array | 0.14.7 | MIT | permissive | both |  |
| gethostname | 1.1.0 | Apache-2.0 | permissive | both |  |
| getrandom | 0.3.4 | MIT OR Apache-2.0 | permissive | both |  |
| getrandom | 0.4.3 | MIT OR Apache-2.0 | permissive | both |  |
| gif | 0.14.2 | MIT OR Apache-2.0 | permissive | both |  |
| half | 2.7.1 | MIT OR Apache-2.0 | permissive | both |  |
| hashbrown | 0.17.1 | MIT OR Apache-2.0 | permissive | both |  |
| heck | 0.5.0 | MIT OR Apache-2.0 | permissive | both |  |
| httpdate | 1.0.3 | MIT OR Apache-2.0 | permissive | both |  |
| image | 0.25.9 | MIT OR Apache-2.0 | permissive | both |  |
| image-webp | 0.2.4 | MIT OR Apache-2.0 | permissive | both |  |
| imgref | 1.12.2 | CC0-1.0 OR Apache-2.0 | permissive | both |  |
| indexmap | 2.14.0 | Apache-2.0 OR MIT | permissive | both |  |
| input | 0.10.0 | MIT | permissive | both |  |
| input-sys | 1.19.0 | MIT | permissive | both | libinput: MIT (permissive) |
| io-lifetimes | 1.0.11 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | permissive | both |  |
| itertools | 0.14.0 | MIT OR Apache-2.0 | permissive | both |  |
| itoa | 1.0.18 | MIT OR Apache-2.0 | permissive | both |  |
| lebe | 0.5.3 | BSD-3-Clause | permissive | both |  |
| libc | 0.2.189 | MIT OR Apache-2.0 | permissive | both | C runtime (glibc, or musl on musllinux wheels): LGPL-2.1-or-later (glibc), MIT (musl) (weak copyleft) |
| libloading | 0.8.9 | ISC | permissive | both | libEGL.so.1 (Mesa/Khronos, MIT), libpipewire-0.3.so.0 (MIT), libwayland-server.so.0 (MIT), libcuda.so.1/libnvidia-encode.so.1 (proprietary): MIT and proprietary driver libraries (permissive) |
| libloading | 0.9.0 | ISC | permissive | both | libEGL.so.1 (Mesa/Khronos, MIT), libpipewire-0.3.so.0 (MIT), libwayland-server.so.0 (MIT), libcuda.so.1/libnvidia-encode.so.1 (proprietary): MIT and proprietary driver libraries (permissive) |
| libm | 0.2.16 | MIT | permissive | both |  |
| libudev-sys | 0.1.4 | MIT | permissive | both | libudev (systemd): LGPL-2.1-or-later (weak copyleft) |
| linux-raw-sys | 0.12.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | permissive | both | Linux kernel ABI (syscall numbers and structs): Linux-syscall-note (permissive) |
| linux-raw-sys | 0.4.15 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | permissive | both | Linux kernel ABI (syscall numbers and structs): Linux-syscall-note (permissive) |
| linux-raw-sys | 0.9.4 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | permissive | both | Linux kernel ABI (syscall numbers and structs): Linux-syscall-note (permissive) |
| log | 0.4.33 | MIT OR Apache-2.0 | permissive | both |  |
| loop9 | 0.1.5 | MIT | permissive | both |  |
| maybe-rayon | 0.1.1 | MIT | permissive | both |  |
| memchr | 2.8.3 | Unlicense OR MIT | permissive | both |  |
| memmap2 | 0.9.11 | MIT OR Apache-2.0 | permissive | both |  |
| memoffset | 0.9.1 | MIT | permissive | both |  |
| miniz_oxide | 0.8.9 | MIT OR Zlib OR Apache-2.0 | permissive | both |  |
| moxcms | 0.7.11 | BSD-3-Clause OR Apache-2.0 | permissive | both |  |
| new_debug_unreachable | 1.0.6 | MIT | permissive | both |  |
| no_std_io2 | 0.9.4 | Apache-2.0 OR MIT | permissive | both |  |
| nom | 8.0.0 | MIT | permissive | both |  |
| noop_proc_macro | 0.3.0 | MIT | permissive | both |  |
| num-bigint | 0.4.8 | MIT OR Apache-2.0 | permissive | both |  |
| num-complex | 0.4.6 | MIT OR Apache-2.0 | permissive | both |  |
| num-derive | 0.4.2 | MIT OR Apache-2.0 | permissive | both |  |
| num-integer | 0.1.46 | MIT OR Apache-2.0 | permissive | both |  |
| num-rational | 0.4.2 | MIT OR Apache-2.0 | permissive | both |  |
| num-traits | 0.2.19 | MIT OR Apache-2.0 | permissive | both |  |
| nvcodec-sys | 0.1.0 | MIT OR Apache-2.0 | permissive | both | NVIDIA NVENC (libnvidia-encode.so.1) and CUDA driver (libcuda.so.1): proprietary driver libraries; nvEncodeAPI.h is MIT, the CUDA bindings are declarations generated from the CUDA toolkit headers (permissive) |
| once_cell | 1.21.4 | MIT OR Apache-2.0 | permissive | both |  |
| openh264 | 0.9.7 | BSD-2-Clause | permissive | non-GPL only |  |
| openh264-sys2 | 0.9.7 | BSD-2-Clause | permissive | non-GPL only | Cisco OpenH264 2.6 (vendored source): BSD-2-Clause (permissive) |
| paste | 1.0.15 | MIT OR Apache-2.0 | permissive | both |  |
| pastey | 0.1.1 | MIT OR Apache-2.0 | permissive | both |  |
| pin-project-lite | 0.2.17 | Apache-2.0 OR MIT | permissive | both |  |
| pixelflux | 2.1.0 | MPL-2.0 | permissive (file-level copyleft) | both | repository LICENSE; Cargo.toml has no license field |
| pixman | 0.2.1 | MIT | permissive | both |  |
| pixman-sys | 0.1.0 | MIT | permissive | both | libpixman-1: MIT (permissive) |
| png | 0.18.1 | MIT OR Apache-2.0 | permissive | both |  |
| polling | 3.11.0 | Apache-2.0 OR MIT | permissive | both |  |
| ppv-lite86 | 0.2.21 | MIT OR Apache-2.0 | permissive | both |  |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 | permissive | both |  |
| profiling | 1.0.18 | MIT OR Apache-2.0 | permissive | both |  |
| profiling-procmacros | 1.0.18 | MIT OR Apache-2.0 | permissive | both |  |
| pulp | 0.22.3 | MIT | permissive | both |  |
| pulp-wasm-simd-flag | 0.1.1 | MIT | permissive | both |  |
| pxfm | 0.1.30 | BSD-3-Clause OR Apache-2.0 | permissive | both |  |
| pyo3 | 0.29.2 | MIT OR Apache-2.0 | permissive | both |  |
| pyo3-ffi | 0.29.2 | MIT OR Apache-2.0 | permissive | both | libpython (CPython): PSF-2.0 (permissive) |
| pyo3-macros | 0.29.2 | MIT OR Apache-2.0 | permissive | both |  |
| pyo3-macros-backend | 0.29.2 | MIT OR Apache-2.0 | permissive | both |  |
| qoi | 0.4.1 | MIT/Apache-2.0 | permissive | both |  |
| quick-error | 2.0.1 | MIT/Apache-2.0 | permissive | both |  |
| quick-xml | 0.41.0 | MIT | permissive | both |  |
| quote | 1.0.47 | MIT OR Apache-2.0 | permissive | both |  |
| rand | 0.9.5 | MIT OR Apache-2.0 | permissive | both |  |
| rand_chacha | 0.9.0 | MIT OR Apache-2.0 | permissive | both |  |
| rand_core | 0.9.5 | MIT OR Apache-2.0 | permissive | both |  |
| rav1e | 0.8.1 | BSD-2-Clause | permissive | both |  |
| ravif | 0.12.0 | BSD-3-Clause | permissive | both |  |
| raw-cpuid | 11.6.0 | MIT | permissive | both |  |
| rayon | 1.12.0 | MIT OR Apache-2.0 | permissive | both |  |
| rayon-core | 1.13.0 | MIT OR Apache-2.0 | permissive | both |  |
| reborrow | 0.5.5 | MIT | permissive | both |  |
| rgb | 0.8.53 | MIT | permissive | both |  |
| rustix | 0.38.44 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | permissive | both |  |
| rustix | 1.1.4 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | permissive | both |  |
| safe_arch | 1.1.0 | Zlib OR Apache-2.0 OR MIT | permissive | non-GPL only |  |
| scoped-tls | 1.0.1 | MIT/Apache-2.0 | permissive | both |  |
| serde | 1.0.229 | MIT OR Apache-2.0 | permissive | both |  |
| serde_core | 1.0.229 | MIT OR Apache-2.0 | permissive | both |  |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 | permissive | both |  |
| serde_json | 1.0.151 | MIT OR Apache-2.0 | permissive | both |  |
| sha2 | 0.10.9 | MIT OR Apache-2.0 | permissive | both |  |
| simd-adler32 | 0.3.10 | MIT | permissive | both |  |
| simd_helpers | 0.1.0 | MIT | permissive | both |  |
| slab | 0.4.12 | MIT | permissive | both |  |
| smallvec | 1.15.2 | MIT OR Apache-2.0 | permissive | both |  |
| smithay | 0.7.0 | MIT | permissive | both |  |
| stable_deref_trait | 1.2.1 | MIT OR Apache-2.0 | permissive | both |  |
| syn | 2.0.119 | MIT OR Apache-2.0 | permissive | both |  |
| syn | 3.0.3 | MIT OR Apache-2.0 | permissive | both |  |
| tempfile | 3.27.0 | MIT OR Apache-2.0 | permissive | both |  |
| thiserror | 1.0.69 | MIT OR Apache-2.0 | permissive | both |  |
| thiserror | 2.0.19 | MIT OR Apache-2.0 | permissive | both |  |
| thiserror-impl | 1.0.69 | MIT OR Apache-2.0 | permissive | both |  |
| thiserror-impl | 2.0.19 | MIT OR Apache-2.0 | permissive | both |  |
| tiff | 0.10.3 | MIT | permissive | both |  |
| tiny_http | 0.12.0 | MIT OR Apache-2.0 | permissive | both |  |
| tracing | 0.1.44 | MIT | permissive | both |  |
| tracing-attributes | 0.1.31 | MIT | permissive | both |  |
| tracing-core | 0.1.36 | MIT | permissive | both |  |
| turbojpeg | 1.5.1 | Unlicense OR MIT | permissive | both |  |
| turbojpeg-sys | 1.2.0 | Unlicense OR MIT | permissive | both | libjpeg-turbo 3.1 (vendored source): IJG AND BSD-3-Clause AND Zlib (permissive) |
| typenum | 1.20.1 | MIT OR Apache-2.0 | permissive | both |  |
| udev | 0.9.3 | MIT | permissive | both |  |
| unicode-ident | 1.0.24 | (MIT OR Apache-2.0) AND Unicode-3.0 | permissive | both |  |
| v_frame | 0.3.9 | BSD-2-Clause | permissive | both |  |
| wasm-bindgen | 0.2.126 | MIT OR Apache-2.0 | permissive | both |  |
| wasm-bindgen-macro | 0.2.126 | MIT OR Apache-2.0 | permissive | both |  |
| wasm-bindgen-macro-support | 0.2.126 | MIT OR Apache-2.0 | permissive | both |  |
| wasm-bindgen-shared | 0.2.126 | MIT OR Apache-2.0 | permissive | both |  |
| wayland-backend | 0.3.16 | MIT | permissive | both |  |
| wayland-client | 0.31.15 | MIT | permissive | both |  |
| wayland-protocols | 0.32.13 | MIT | permissive | both |  |
| wayland-protocols-misc | 0.3.12 | MIT | permissive | both |  |
| wayland-protocols-plasma | 0.3.12 | MIT | permissive | both |  |
| wayland-protocols-wlr | 0.3.12 | MIT | permissive | both |  |
| wayland-scanner | 0.31.11 | MIT | permissive | both |  |
| wayland-server | 0.31.14 | MIT | permissive | both |  |
| wayland-sys | 0.31.11 | MIT | permissive | both | libwayland-server: MIT (permissive) |
| weezl | 0.1.12 | MIT OR Apache-2.0 | permissive | both |  |
| wide | 1.6.0 | Zlib OR Apache-2.0 OR MIT | permissive | non-GPL only |  |
| x11rb | 0.13.2 | MIT OR Apache-2.0 | permissive | both |  |
| x11rb-protocol | 0.13.2 | MIT OR Apache-2.0 | permissive | both |  |
| x264-sys | 0.2.3 | MIT | permissive | GPL only | libx264: GPL-2.0-or-later (copyleft) |
| xcursor | 0.3.11 | MIT | permissive | both |  |
| xkbcommon | 0.9.0 | MIT | permissive | both | libxkbcommon: MIT (permissive) |
| xkeysym | 0.2.1 | MIT OR Apache-2.0 OR Zlib | permissive | both |  |
| xxhash-rust | 0.8.18 | BSL-1.0 | permissive | both |  |
| y4m | 0.8.0 | MIT | permissive | both |  |
| yuv | 0.8.16 | BSD-3-Clause OR Apache-2.0 | permissive | both |  |
| zerocopy | 0.8.55 | BSD-2-Clause OR Apache-2.0 OR MIT | permissive | both |  |
| zerocopy-derive | 0.8.55 | BSD-2-Clause OR Apache-2.0 OR MIT | permissive | both |  |
| zmij | 1.0.23 | MIT | permissive | both |  |
| zune-core | 0.4.12 | MIT OR Apache-2.0 OR Zlib | permissive | both |  |
| zune-core | 0.5.1 | MIT OR Apache-2.0 OR Zlib | permissive | both |  |
| zune-inflate | 0.2.54 | MIT OR Apache-2.0 OR Zlib | permissive | both |  |
| zune-jpeg | 0.4.21 | MIT OR Apache-2.0 OR Zlib | permissive | both |  |
| zune-jpeg | 0.5.15 | MIT OR Apache-2.0 OR Zlib | permissive | both |  |
</details>

## What a non-GPL build contains

`PIXELFLUX_ENABLE_GPL=0` (`--no-default-features --features openh264`):

- the 202 crates above, all permissive (pixelflux itself MPL-2.0), with
  OpenH264 and libjpeg-turbo compiled from vendored BSD/IJG source;
- linked: FFmpeg libavcodec/libavfilter/libavutil (+ swresample, swscale,
  avformat) under LGPL-2.1-or-later when FFmpeg is built without
  `--enable-gpl`, and through it kvazaar, libvpx, SVT-AV1 and dav1d (BSD),
  libgbm, libpixman-1, libxkbcommon (MIT), the C and C++
  runtimes (glibc LGPL-2.1-or-later or musl MIT; libgcc_s/libstdc++ with the
  GCC runtime exception), and through FFmpeg libva/libdrm/libX11 (MIT);
- loaded at run time only when present: libwayland-server, libEGL,
  libpipewire-0.3 (MIT), and the NVIDIA driver's libcuda/libnvidia-encode
  (proprietary, never shipped);
- no GPL code. The build is only as GPL-free as the FFmpeg it links: the
  project's non-GPL wheel recipe builds FFmpeg n8.1 without `--enable-gpl`,
  a source build against a distribution FFmpeg that was configured with
  `--enable-gpl` (Debian, Ubuntu, Alpine, conda-forge's `gpl_*` builds of `ffmpeg`, which is the default variant)
  links a GPL libavcodec even though pixelflux contains no x264 code, and a
  build without the `gpl` feature never selects that FFmpeg's `libx265`.

## What the GPL build adds

The default build (`gpl` feature, what the published wheels and
`pip install pixelflux` give you) swaps the software H.264 and H.265 encoders:

- adds `x264-sys` and links libx264 (GPL-2.0-or-later); the manylinux wheels
  bundle `libx264.so`;
- selects x265 (GPL-2.0-or-later) through FFmpeg for software H.265, in place
  of kvazaar; the wheels' FFmpeg is built `--enable-gpl --enable-libx265`;
- removes the OpenH264 crates and `safe_arch`/`wide` from the binary (they stay
  dev-dependencies for the tests);
- the resulting binary is a combination of MPL-2.0, permissive and GPL code and
  is therefore distributed under the GPL-2.0-or-later terms as a whole
  (MPL-2.0 is GPL-compatible through its secondary-license clause). setup.py
  prints which configuration is being built.

## Distribution notes

- manylinux and musllinux wheels (cibuildwheel, `pyproject.toml`): FFmpeg
  n8.1, kvazaar, libvpx, SVT-AV1 and dav1d — plus x264 and x265 for the GPL
  wheel — are built from source in the image; auditwheel bundles them
  (`libx264.so`, `libx265.so`, `libkvazaar.so`, `libvpx.so`, `libSvtAv1Enc.so`,
  `libdav1d.so`, `libavcodec`, `libavfilter`, `libavformat`, `libavutil`,
  `libswresample`, `libswscale`) into `pixelflux.libs/` and leaves libva,
  libdrm, libgbm, libEGL, libxkbcommon, libpixman-1, libX11/libxcb, zlib,
  liblzma and the GCC runtime to the host (`repair-wheel-command` excludes).
  The non-GPL wheel's FFmpeg reports `LGPL version 2.1 or later` and its
  configuration contains no `--enable-gpl`; the GPL wheel's is built
  `--enable-gpl --enable-libx265`.
- The wheels carry pixelflux's own LICENSE only. The statically linked OpenH264
  (BSD-2-Clause) and libjpeg-turbo (IJG/BSD-3-Clause/Zlib) and the bundled
  libx264/FFmpeg notices are not included in the wheel; this file is the
  inventory, their license texts live in the upstream sources named above.
- selkies' AppImage pairs pixelflux with conda-forge's `ffmpeg=*=*lgpl*`
  variant; its container images install the distribution `ffmpeg` and `x264`
  packages.

## How this is enforced

- `scripts/check-licenses.py` resolves both configurations with `cargo
  metadata --locked` (no build, normal dependencies only), classifies every
  crate's SPDX expression, overlays the native libraries behind the binding
  crates (the `NATIVE` table in the script, the source of the component table
  above) and fails when a copyleft component appears in a configuration not
  listed for it in `ALLOWED_COPYLEFT` (`x264-sys` → `gpl` only), when a crate
  has no usable license metadata, or when a crate named like a native binding
  (`-sys`, `_sys`, `-ffi`) is not described in `NATIVE`. Run it from the
  repository root: `python3 scripts/check-licenses.py` (`--set non-gpl`,
  `--markdown`, `--metadata FILE` for a saved `cargo metadata` JSON). It
  reports `PASS`/`FAIL` per configuration and exits non-zero on a failure.
- `pixelflux/deny.toml` is the [cargo-deny](https://embarkstudios.github.io/cargo-deny/)
  policy: its graph is the non-GPL configuration, the allow list is
  permissive-only, `x264-sys` is clarified as `MIT AND GPL-2.0-or-later`,
  banned from that graph, and accepted only through an exception when the graph
  is switched to the GPL build. From `pixelflux/`:
  `cargo deny --exclude-dev check licenses bans sources` (non-GPL gate) and
  `cargo deny --exclude-dev --features gpl check licenses sources` (GPL build;
  its `bans` check fails on purpose). `cargo deny list` ignores the `[graph]`
  section and needs `--no-default-features --features openh264` spelled out.
- `.github/workflows/licenses.yml` runs the script and both cargo-deny
  invocations on every push and pull request.
- Adding a dependency that links, loads or vendors native code means adding it
  to `NATIVE` in the script and to the table above; adding a copyleft
  dependency means an `ALLOWED_COPYLEFT` entry, a `deny.toml` clarification
  and exception, and a row here. To exercise the gate itself, save the GPL
  graph (`cargo metadata --format-version 1 --manifest-path pixelflux/Cargo.toml
  > gpl.json`) and audit it under the non-GPL policy
  (`scripts/check-licenses.py --metadata gpl.json --set non-gpl`): it must
  fail on `x264-sys`.

## Open items

- `pixelflux/Cargo.toml` has no `license = "MPL-2.0"` field (the script and
  `deny.toml` carry the clarification); adding it lets every tool see it.
- The musllinux wheels bundle Alpine's GPL FFmpeg build (above).
