"""The committed lock: uv refuses an edited copy, and each system has a wheel for each package it can need."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker

ROOT = Path(__file__).parents[2]
# The two supported systems as the lock's markers see them, and the wheel tag each one installs.
WINDOWS = {
    "sys_platform": "win32",
    "platform_machine": "AMD64",
    "platform_system": "Windows",
    "os_name": "nt",
}
LINUX = {
    "sys_platform": "linux",
    "platform_machine": "x86_64",
    "platform_system": "Linux",
    "os_name": "posix",
}
SYSTEMS = {"win_amd64": (WINDOWS, r"win_amd64"), "manylinux x86_64": (LINUX, r"manylinux[^-]*x86_64")}
PYTHON = {
    "python_version": "3.12", "python_full_version": platform.python_version(),
    "implementation_name": "cpython", "platform_python_implementation": "CPython",
}  # fmt: skip
PLANT = """
version = 1
[[package]]
name = "app"
version = "1.0"
source = {{ editable = "." }}
dependencies = [{{ name = "native"{marker} }}]

[[package]]
name = "native"
version = "2.0"
source = {{ registry = "https://pypi.org/simple" }}
wheels = [{{ url = "https://files.example/native-2.0-{tag}.whl" }}]
"""


def missing_wheels(text: str) -> list[str]:
    """Each package that one system can need but that has neither a wheel for it nor a pure-Python wheel.

    A package without a wheel for a system is fine when every lock edge into it has a marker that is false
    there, such as a Linux-only CUDA library.
    """
    lock = tomllib.loads(text)
    extras = [*sorted(set(re.findall(r"extra == '([^']+)'", text))), ""]
    into: dict[tuple[str, str | None], list[str]] = {}
    for package in lock["package"]:
        groups = [package.get("dependencies", []), *package.get("optional-dependencies", {}).values(),
                  *package.get("dev-dependencies", {}).values()]  # fmt: skip
        for edge in (e for group in groups for e in group):
            into.setdefault((edge["name"], edge.get("version")), []).append(edge.get("marker", ""))
    problems = []
    for package in lock["package"]:
        if {"editable", "virtual"} & set(package["source"]):
            continue  # the project itself, built from its source
        wheels = [w["url"].rsplit("/", 1)[-1] for w in package.get("wheels", [])]
        markers = into.get((package["name"], package["version"]), []) + into.get((package["name"], None), [])
        for system, (env, tag) in SYSTEMS.items():
            if any(re.search(rf"-(?:{tag}|any)\.whl$", name) for name in wheels):
                continue
            here = {**env, **PYTHON}
            if any(not m or Marker(m).evaluate({**here, "extra": e}) for m in markers for e in extras):
                problems.append(f"{package['name']} {package['version']}: no wheel for {system}")
    return problems


def test_each_system_has_a_wheel_for_every_package_it_can_need() -> None:
    assert missing_wheels((ROOT / "uv.lock").read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    ("marker", "tag", "expected"),
    [("", "cp312-cp312-manylinux_2_28_x86_64", ["native 2.0: no wheel for win_amd64"]),
     ("", "cp312-cp312-win_amd64", ["native 2.0: no wheel for manylinux x86_64"]),
     (", marker = \"sys_platform == 'linux'\"", "cp312-cp312-manylinux_2_28_x86_64", []),
     (", marker = \"sys_platform == 'win32'\"", "cp312-cp312-manylinux_2_28_x86_64",
      ["native 2.0: no wheel for win_amd64"]),
     (", marker = \"extra == 'extra-3-app-gpu'\"", "cp312-cp312-manylinux_2_28_x86_64",
      ["native 2.0: no wheel for win_amd64"]),
     ("", "py3-none-any", [])],
)  # fmt: skip
def test_a_package_without_a_wheel_counts_only_where_an_edge_needs_it(
    marker: str, tag: str, expected: list[str]
) -> None:
    assert missing_wheels(PLANT.format(marker=marker, tag=tag)) == expected


def locked_copy(folder: Path) -> Path:
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, folder / name)
    return folder


def sync(folder: Path, offline: bool = True) -> subprocess.CompletedProcess[str]:
    """`uv sync --locked` as the installer runs it, as a dry run that installs nothing, offline by default."""
    uv = os.environ.get("UV") or shutil.which("uv")
    assert uv, "the tests run under uv, so uv is on PATH"
    where = ["--offline"] if offline else []
    return subprocess.run(
        [uv, "sync", "--locked", "--dry-run", *where, "--directory", str(folder)],
        capture_output=True, text=True, encoding="utf-8", timeout=300, check=False,
    )  # fmt: skip


def test_an_untouched_copy_of_the_lock_syncs(tmp_path: Path) -> None:
    done = sync(locked_copy(tmp_path))
    assert done.returncode == 0, done.stderr
    assert not (tmp_path / ".venv").exists()  # a dry run makes no environment


def test_a_hand_edited_lock_is_refused(tmp_path: Path) -> None:
    lock = locked_copy(tmp_path) / "uv.lock"
    text = lock.read_text(encoding="utf-8")
    entry = re.search(r'\[\[package\]\]\nname = "click"\nversion = "([^"]+)"', text)
    assert entry is not None
    lock.write_text(text.replace(entry.group(0), entry.group(0).replace(entry[1], f"{entry[1]}.1")), "utf-8")
    done = sync(tmp_path)
    assert done.returncode != 0 and "Failed to parse `uv.lock`" in done.stderr, done.stderr


@pytest.mark.network  # uv resolves the edited project before it refuses, and that needs the package index
def test_a_dependency_changed_without_locking_again_is_refused(tmp_path: Path) -> None:
    project = locked_copy(tmp_path) / "pyproject.toml"
    text = project.read_text(encoding="utf-8")
    assert '    "click",\n' in text
    project.write_text(text.replace('    "click",\n', "", 1), encoding="utf-8")
    done = sync(tmp_path, offline=False)
    assert done.returncode != 0, done.stderr
    assert "needs to be updated, but `--locked` was provided" in done.stderr, done.stderr
