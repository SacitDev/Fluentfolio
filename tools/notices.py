"""Licence gate and generator of THIRD_PARTY_NOTICES.md.

Usage, from the repository root:  uv run --locked python tools/notices.py [--check]

Every package in uv.lock gets a licence: from tools/licences.toml when it is listed there, else from
the installed metadata (License-Expression, then the OSI classifiers, then the first License line).
A package that not every supported install has (Windows or Linux, with the cpu or the cu130 extra,
plus the dev group) must be listed in tools/licences.toml, so that every machine writes the same text.
A second table lists every entry of the model catalogue, src/fluentfolio/data/models.toml, and a third
the files copied into the web app ([[vendored]] in tools/licences.toml), with their SHA-256.
--check fails on an unknown licence, on a copyleft, proprietary or non-commercial package missing from
its named list, on a missing licence file of a copied file, and on any difference between the generated
text and THIRD_PARTY_NOTICES.md, so a changed copied file fails it too.
Without --check the file is written.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.markers import Marker

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "uv.lock"
TABLE = ROOT / "tools" / "licences.toml"
OUT = ROOT / "THIRD_PARTY_NOTICES.md"
MODELS = ROOT / "src" / "fluentfolio" / "data" / "models.toml"

PERMISSIVE = {
    "0BSD", "Apache-2.0", "BSD", "BSD-2-Clause", "BSD-3-Clause", "BSL-1.0", "CC0-1.0", "CNRI-Python", "HPND",
    "ISC", "MIT", "MIT-0", "MIT-CMU", "MPL-2.0", "PSF-2.0", "Python-2.0", "Unlicense", "Zlib",
}  # fmt: skip
# Share-alike data licences count here too: a copy must keep the same licence.
COPYLEFT = {
    "CC-BY-SA-4.0", "GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later", "LGPL-2.1-only",
    "LGPL-2.1-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later",
}  # fmt: skip
# Licence names that metadata uses instead of SPDX identifiers.
ALIASES = {
    "MIT License": "MIT",
    "Apache Software License": "Apache-2.0",
    "Apache 2.0": "Apache-2.0",
    "Apache License 2.0": "Apache-2.0",
    "BSD License": "BSD",
    "3-Clause BSD License": "BSD-3-Clause",
    "ISC License (ISCL)": "ISC",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "Python Software Foundation License": "PSF-2.0",
    "The Unlicense (Unlicense)": "Unlicense",
    "GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
    "GNU General Public License v3 or later (GPLv3+)": "GPL-3.0-or-later",
    "GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
}
RANK = ["permissive", "gpl", "proprietary", "nc"]
NAMED = ("copyleft", "proprietary", "noncommercial")  # table sections, in RANK order after "permissive"
BACKENDS = ("cpu", "cu130")
PLATFORMS = [
    {"sys_platform": "win32", "platform_machine": "AMD64", "platform_system": "Windows", "os_name": "nt"},
    {"sys_platform": "linux", "platform_machine": "x86_64", "platform_system": "Linux", "os_name": "posix"},
]
PYTHON = {
    "python_version": "3.12",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
}


class LicenceError(Exception):
    """A package whose licence is unknown or not allowed."""


def from_metadata(name: str, version: str) -> str | None:
    """The licence an installed package declares, or None when this version is not installed."""
    try:
        meta = metadata.metadata(name)
    except metadata.PackageNotFoundError:
        return None
    if meta["Version"] != version:
        return None
    if meta.get("License-Expression"):
        return meta["License-Expression"].strip()
    osi = "License :: OSI Approved :: "
    found = {
        ALIASES.get(c[len(osi) :].strip(), c[len(osi) :].strip())
        for c in meta.get_all("Classifier") or []
        if c.startswith(osi)
    }
    if found:
        return " OR ".join(sorted(found))
    first = (meta.get("License") or "").strip().splitlines()
    return ALIASES.get(first[0].strip(), first[0].strip()) if first else ""


def licence_class(expr: str, table: dict) -> str:
    """Classify an SPDX-style expression; OR takes the most permissive choice, AND the strictest."""
    proprietary = set(table.get("proprietary", {}).get("licences", []))
    nc = set(table.get("noncommercial", {}).get("licences", []))
    choices = []
    for alternative in re.split(r"\s+OR\s+", expr.replace("(", " ").replace(")", " ").strip()):
        classes = []
        for part in re.split(r"\s+AND\s+", alternative.strip()):
            ident = part.split(" WITH ")[0].strip()
            if ident in PERMISSIVE:
                classes.append("permissive")
            elif ident in COPYLEFT:
                classes.append("gpl")
            elif ident in proprietary:
                classes.append("proprietary")
            elif ident in nc:
                classes.append("nc")
            else:
                raise LicenceError(f"unknown licence {ident!r}")
        choices.append(max(classes, key=RANK.index))
    return min(choices, key=RANK.index)


def lock_graph(lock: dict) -> tuple[list[dict], set[tuple[str, str]], set[tuple[str, str]]]:
    """The locked packages, those every supported install has, and those the app (not only dev) uses."""
    packages = lock["package"]
    root = next(p for p in packages if "editable" in p["source"] or "virtual" in p["source"])
    by_name: dict[str, list[dict]] = {}
    for p in packages:
        by_name.setdefault(p["name"], []).append(p)

    def reach(edges: list[dict], env: dict[str, str]) -> set[tuple[str, str]]:
        done: set[tuple[str, str, str]] = set()  # (name, version, extra or "")
        todo = list(edges)
        while todo:
            edge = todo.pop()
            if "marker" in edge and not Marker(edge["marker"]).evaluate(env):
                continue
            for p in by_name.get(edge["name"], []):
                if edge.get("version", p["version"]) != p["version"]:
                    continue
                for extra in ["", *edge.get("extra", [])]:
                    if (p["name"], p["version"], extra) not in done:
                        done.add((p["name"], p["version"], extra))
                        todo += (
                            p.get("optional-dependencies", {}).get(extra, [])
                            if extra
                            else p.get("dependencies", [])
                        )
        return {(name, version) for name, version, _ in done}

    runtime = list(root.get("dependencies", []))
    optional = root.get("optional-dependencies", {})
    dev = [d for group in root.get("dev-dependencies", {}).values() for d in group]
    others = [d for name, group in optional.items() if name not in BACKENDS for d in group]
    everywhere: set[tuple[str, str]] | None = None
    app: set[tuple[str, str]] = set()
    development: set[tuple[str, str]] = set()
    # Every install is one supported platform with exactly one backend extra; uv names that extra in markers.
    for platform, backend in itertools.product(PLATFORMS, BACKENDS):
        env = {**platform, **PYTHON, "extra": f"extra-{len(root['name'])}-{root['name']}-{backend}"}
        installed = reach(runtime + optional.get(backend, []) + dev, env)
        everywhere = installed if everywhere is None else everywhere & installed
        app |= reach(runtime + optional.get(backend, []) + others, env)
        development |= installed
    used_by = {key: "app" for key in app} | {key: "development" for key in development - app}
    return [p for p in packages if p is not root], everywhere or set(), used_by


def build(table: dict) -> tuple[str, list[str]]:
    """The notices text and the list of problems."""
    packages, everywhere, used_by = lock_graph(tomllib.loads(LOCK.read_text(encoding="utf-8")))
    overrides = table.get("override", {})
    named = {
        klass: set(table.get(section, {}).get("names", []))
        for klass, section in zip(RANK[1:], NAMED, strict=True)
    }
    rows, problems = [], []
    for p in sorted(packages, key=lambda p: (p["name"], p["version"])):
        name, version = p["name"], p["version"]
        key = (name, version)
        expr = overrides.get(name)
        if expr is None and key not in everywhere:
            problems.append(
                f"{name} {version}: not installed on every machine; list it in tools/licences.toml"
            )
            continue
        expr = expr if expr is not None else from_metadata(name, version)
        if not expr:
            problems.append(f"{name} {version}: no licence found; list it in tools/licences.toml")
            continue
        try:
            klass = licence_class(expr, table)
        except LicenceError as exc:
            problems.append(f"{name} {version}: {exc}")
            continue
        if klass != "permissive" and name not in named[klass]:
            problems.append(f"{name} {version}: {klass} licence {expr!r} but not on its named list")
        rows.append(f"| {name} | {version} | {expr} | {klass} | {used_by.get(key, 'none')} |")
    text = "\n".join([
        "# Third-party notices",
        "",
        "Made by tools/notices.py from uv.lock, the installed package metadata and tools/licences.toml.",
        "Do not edit this file by hand; run `uv run --locked python tools/notices.py` instead.",
        "",
        "Fluentfolio's own code is MIT licensed. The packages below keep their own licences.",
        "Class `gpl` marks copyleft packages, `proprietary` free but closed-source ones (the NVIDIA CUDA",
        "libraries of a GPU install) and `nc` non-commercial or more restricted ones (openSMILE and a",
        "package only it installs, both added only after you accept openSMILE's licence). `development`",
        "packages are used only to build and test Fluentfolio; `none` marks builds for other systems that",
        "Windows and Linux never install.",
        "",
        "## Python packages",
        "",
        "| Package | Version | Licence | Class | Used by |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "## Models and tools",
        "",
        "The models and tools that Fluentfolio downloads or uses, from src/fluentfolio/data/models.toml.",
        "Each keeps its own licence. A `cc-by` entry asks for the credit shown. An `nc` entry is installed",
        "only after you accept its licence.",
        "",
        "| Component | Source | Licence | Class | Credit |",
        "|---|---|---|---|---|",
        *model_rows(),
        "",
        "## Files inside the web app",
        "",
        "Copied from their projects into src/fluentfolio/web/, each beside its licence file. A SHA-256 marks",
        "a file copied unchanged.",
        "",
        "| Name | Version | Source | Licence | Licence file | Files |",
        "|---|---|---|---|---|---|",
        *vendored_rows(table, problems),
        "",
    ])  # fmt: skip
    return text, problems


def vendored_rows(table: dict, problems: list[str]) -> list[str]:
    """One row per project whose files the web app carries, with the SHA-256 of each unchanged file."""
    rows = []
    for v in table.get("vendored", []):
        licence_file = v["licence_file"]
        if not (ROOT / licence_file).is_file():
            problems.append(f"{v['name']}: licence file {licence_file} is missing")
        found = [
            f"{Path(f).name} sha256 {hashlib.sha256((ROOT / f).read_bytes()).hexdigest()}"
            for f in v.get("files", [])
        ]
        files = "<br>".join(found) or v["changed"]
        rows.append(
            f"| {v['name']} | {v['version']} | {v['source']} | {v['licence']} | {licence_file} | {files} |"
        )
    return rows


def model_rows() -> list[str]:
    """One row per entry of the model catalogue, in its order."""
    rows = []
    for c in tomllib.loads(MODELS.read_text(encoding="utf-8"))["component"]:
        source = f"huggingface.co/{c['repo']}" if c["host"] == "huggingface.co" else c["host"]
        rows.append(
            f"| {c['title']} | {source} | {c['licence']} | {c['licence_class']} | {c['attribution']} |"
        )
    return rows


def main(argv: list[str]) -> int:
    if argv not in ([], ["--check"]):
        print("usage: python tools/notices.py [--check]", file=sys.stderr)
        return 2
    text, problems = build(tomllib.loads(TABLE.read_text(encoding="utf-8")))
    if argv == ["--check"]:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            problems.append("THIRD_PARTY_NOTICES.md differs from the generated text; run tools/notices.py")
    elif not problems:
        OUT.write_text(text, encoding="utf-8", newline="\n")
    for line in problems:
        print(f"notices: {line}")
    print(f"notices: {'FAIL' if problems else 'ok'} ({text.count(chr(10) + '| ') - 1} packages)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
