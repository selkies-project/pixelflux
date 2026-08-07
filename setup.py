# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys

from setuptools import setup
from setuptools_rust import Binding, RustExtension, Strip

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# GPL components (GPL-2.0+ libx264, used for striped software H.264) are enabled by
# default. Set PIXELFLUX_ENABLE_GPL=0 (or "false"/"no") to build without them; the
# BSD-licensed OpenH264 encoder is then substituted for software H.264.
_enable_gpl = os.environ.get("PIXELFLUX_ENABLE_GPL", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
if _enable_gpl:
    print(
        "NOTICE: pixelflux is being built WITH GPL-licensed components "
        "(GPL-2.0+ libx264 for striped software H.264), which is the default. "
        "Set PIXELFLUX_ENABLE_GPL=0 to exclude every GPL-licensed component.",
        file=sys.stderr,
    )
else:
    print(
        "NOTICE: pixelflux is being built WITHOUT GPL-licensed components "
        "(PIXELFLUX_ENABLE_GPL=0): libx264 is excluded and OpenH264 (BSD) "
        "substitutes for software H.264.",
        file=sys.stderr,
    )

setup(
    name="pixelflux",
    version="2.1.0",
    author="Selkies Project",
    author_email="pypi@linuxserver.io",
    description="A performant web native pixel delivery pipeline for diverse sources, blending VNC-inspired parallel processing of pixel buffers with flexible modern encoding formats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MPL-2.0",
    url="https://github.com/selkies-project/pixelflux",

    # Single self-contained Rust extension: the top-level `pixelflux` module does X11 (XShm)
    # and Wayland capture plus all encoding/conversion. No C/C++ sources, no Python package
    # layer -- `import pixelflux` resolves directly to pixelflux.cpython-*.so.
    packages=[],
    rust_extensions=[
        RustExtension(
            "pixelflux",
            "pixelflux/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
            strip=Strip.All,
            args=([] if _enable_gpl else ["--no-default-features"]),
        )
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.9",
    zip_safe=False,
)
