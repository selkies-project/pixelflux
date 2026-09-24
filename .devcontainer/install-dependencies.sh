#!/bin/bash
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Dev setup for the pure-Rust pixelflux PyO3 extension (replaces the old C++ setup.py build).
set -euxo pipefail

sudo apt-get update
# System C libraries the crate links against (x264-sys -> libx264, codec-sys -> the
# software codec libraries it binds from their headers, x11rb -> libxcb + shm +
# xfixes, GBM/DRM, Wayland/xkb) plus the build toolchain (nasm is needed to build
# the vendored OpenH264 and libjpeg-turbo sources, which are statically linked and
# need no system copy). The VA-API session opens libva at run time, so only its
# runtime package is needed.
sudo apt-get install -y \
  build-essential pkg-config nasm clang libclang-dev curl ca-certificates \
  libx264-dev libx265-dev libvpx-dev libsvtav1enc-dev libdav1d-dev libde265-dev \
  libva2 libdrm-dev libgbm-dev \
  libwayland-dev libxkbcommon-dev \
  libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev \
  python3-dev python3-pip

# firefox-esr (for end-to-end testing the stream in a browser)
sudo apt install -y software-properties-common && sudo add-apt-repository ppa:mozillateam/ppa -y && sudo apt install -y firefox-esr

# Rust toolchain (the extension is built via setuptools-rust / cargo).
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
fi

# Build and install the extension from source.
pip3 install --upgrade pip setuptools-rust
pip3 install .
