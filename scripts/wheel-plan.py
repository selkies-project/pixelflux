#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Plan of the wheel build the ``Wheels`` workflow runs.

``matrix`` prints the two job matrices as ``name=json`` lines for
``$GITHUB_OUTPUT``: ``wheels``, one entry per wheel the installed cibuildwheel
selects from ``pyproject.toml``, and ``images``, one per platform those wheels
cover, each naming the base image that cibuildwheel release pins for it.

``recipe`` prints the ``before-all`` of ``pyproject.toml`` as a script, under
the environment cibuildwheel would run it in, so an image carrying the native
dependencies is built by the same recipe a plain cibuildwheel run uses.
"""

import configparser
import json
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
RUNNERS = {"x86_64": "ubuntu-latest", "aarch64": "ubuntu-24.04-arm"}
DEFAULT_IMAGES = {"manylinux": "manylinux_2_28", "musllinux": "musllinux_1_2"}


def config():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["cibuildwheel"]


def matrix():
    import cibuildwheel

    pins = configparser.ConfigParser()
    pins.read(Path(cibuildwheel.__file__).parent / "resources" / "pinned_docker_images.cfg")
    settings = config()
    identifiers = subprocess.run(
        [sys.executable, "-m", "cibuildwheel", "--print-build-identifiers", "--platform", "linux", str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if not identifiers:
        sys.exit("cibuildwheel selected no wheel")
    wheels, images = [], {}
    for identifier in identifiers:
        platform = identifier.split("-", 1)[1]
        libc, arch = platform.split("_", 1)
        wheels.append({"only": identifier, "platform": platform, "runner": RUNNERS[arch]})
        image = settings.get(f"{libc}-{arch}-image", DEFAULT_IMAGES[libc])
        images[platform] = {"platform": platform, "base": pins[arch].get(image, image).split("#")[0].strip(), "runner": RUNNERS[arch]}
    print(f"wheels={json.dumps(wheels)}")
    print(f"images={json.dumps(list(images.values()))}")


def recipe():
    settings = config()
    if any("before-all" in override for override in settings.get("overrides", [])):
        sys.exit("an override's before-all would be left out of the image")
    for name, value in settings.get("environment", {}).items():
        print(f'export {name}="{value}"')
    print(settings["linux"]["before-all"])


if __name__ == "__main__":
    {"matrix": matrix, "recipe": recipe}[sys.argv[1]]()
