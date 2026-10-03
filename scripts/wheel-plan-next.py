#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Plan of the wheel build the ``Wheels`` workflow runs.

``matrix IMAGE JOBS`` prints the two job matrices as ``name=json`` lines for
``$GITHUB_OUTPUT``: ``groups``, the wheels the installed cibuildwheel selects
from ``pyproject.toml`` dealt into JOBS jobs per platform, and ``images``, the
platforms whose image the Actions cache holds no copy of for this ref, each
naming the base image that cibuildwheel release pins for it. Both carry the
cache key of the platform's image, which changes with the recipe, the base, the
variables passed to the build and the ISO week, so a toolchain or package update
reaches the wheels within a week.

``recipe`` prints the ``before-all`` of ``pyproject.toml`` as a script, under
the environment cibuildwheel would run it in, so an image carrying the native
dependencies is built by the same recipe a plain cibuildwheel run uses.
"""

import configparser
import datetime
import hashlib
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
RUNNERS = {"x86_64": "ubuntu-latest", "aarch64": "ubuntu-24.04-arm"}
DEFAULT_IMAGES = {"manylinux": "manylinux_2_28", "musllinux": "musllinux_1_2"}


def config():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["cibuildwheel"]


def recipe_text():
    settings = config()
    if any("before-all" in override for override in settings.get("overrides", [])):
        sys.exit("an override's before-all would be left out of the image")
    exports = "".join(f'export {name}="{value}"\n' for name, value in settings.get("environment", {}).items())
    return exports + settings["linux"]["before-all"] + "\n"


def cached(keys):
    """The keys the Actions cache holds an entry of that this run can restore."""
    api, repo, token = (os.environ.get(v) for v in ("GITHUB_API_URL", "GITHUB_REPOSITORY", "GH_TOKEN"))
    if not (api and repo and token):
        return set()
    refs = {os.environ.get("GITHUB_REF"), os.environ.get("DEFAULT_REF")}
    found = set()
    for key in keys:
        request = urllib.request.Request(
            f"{api}/repos/{repo}/actions/caches?per_page=100&key={urllib.parse.quote(key)}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            entries = json.load(response)["actions_caches"]
        if any(entry["key"] == key and entry["ref"] in refs for entry in entries):
            found.add(key)
    return found


def matrix(image, jobs, abi3="", salt=""):
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
    if abi3:
        # The floor's abi3 wheel serves every later interpreter
        floor = int(abi3[3:])
        identifiers = [i for i in identifiers if int(i.split("-", 1)[0][3:]) <= floor]
    if not identifiers:
        sys.exit("cibuildwheel selected no wheel")
    year, week, _ = datetime.date.today().isocalendar()
    passed = {name: os.environ.get(name) for name in settings.get("linux", {}).get("environment-pass", [])}
    recipe = recipe_text()
    platforms = {}
    for identifier in identifiers:
        platform = identifier.split("-", 1)[1]
        platforms.setdefault(platform, []).append(identifier)
    images, groups = [], []
    for platform, members in platforms.items():
        libc, arch = platform.split("_", 1)
        name = settings.get(f"{libc}-{arch}-image", DEFAULT_IMAGES[libc])
        base = pins[arch].get(name, name).split("#")[0].strip()
        digest = hashlib.sha256(json.dumps([recipe, base, passed, f"{year}-W{week:02d}", salt]).encode()).hexdigest()
        key = f"{image}-{platform}-{digest[:16]}"
        images.append({"platform": platform, "base": base, "runner": RUNNERS[arch], "key": key})
        count = min(jobs, len(members))
        for n in range(count):
            share = members[n * len(members) // count : (n + 1) * len(members) // count]
            first, last = share[0].split("-", 1)[0], share[-1].split("-", 1)[0]
            groups.append({
                "name": share[0] if len(share) == 1 else f"{first}-{last}-{platform}",
                "build": " ".join(share),
                "platform": platform,
                "runner": RUNNERS[arch],
                "key": key,
            })
    held = cached([i["key"] for i in images])
    print(f"groups={json.dumps(groups)}")
    print(f"images={json.dumps([i for i in images if i['key'] not in held])}")


if __name__ == "__main__":
    if sys.argv[1] == "recipe":
        print(recipe_text(), end="")
    else:
        matrix(sys.argv[2], int(sys.argv[3]), *sys.argv[4:])
