"""Download the public test data listed in tests/data_manifest.toml into the FF_TEST_DATA folder.

Each file is checked against its size and SHA-256 before it is kept; archives are unpacked with the safe
"data" filter. A file already present with the right hash is not downloaded again.
Usage: python tools/fetch_test_data.py [--dest DIR] [--manifest FILE] [--only NAME ...]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import tomllib
import urllib.request
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

MANIFEST = Path(__file__).resolve().parents[1] / "tests" / "data_manifest.toml"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
CHUNK = 1 << 20


def safe_relative(text: str) -> Path:
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or ":" in text:
        raise ValueError(f"not a safe relative path: {text}")
    return Path(*path.parts)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, part: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in LOOPBACK):
        raise ValueError(f"only https downloads are allowed: {url}")
    with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:  # noqa: S310
        while block := response.read(CHUNK):
            out.write(block)


def fetch(entry: dict, dest: Path) -> str:
    target = dest / safe_relative(entry["save_as"])
    if target.exists() and target.stat().st_size == entry["size"] and sha256_of(target) == entry["sha256"]:
        state = "present"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)  # a damaged earlier copy
        part = target.with_name(target.name + ".part")
        errors = []
        for url in entry["urls"]:
            try:
                download(url, part)
            except (OSError, ValueError) as exc:
                errors.append(f"{url}: {exc}")
                continue
            size, digest = part.stat().st_size, sha256_of(part)
            if size == entry["size"] and digest == entry["sha256"]:
                os.replace(part, target)
                break
            errors.append(f"{url}: got {size} bytes with SHA-256 {digest}")
        part.unlink(missing_ok=True)
        if not target.exists():
            raise RuntimeError("; ".join(errors))
        state = "downloaded"
    if "unpack_to" in entry:
        folder = dest / safe_relative(entry["unpack_to"])
        done = folder / f".{entry['name']}.unpacked"
        if not done.exists() or done.read_text(encoding="utf-8") != entry["sha256"]:
            folder.mkdir(parents=True, exist_ok=True)
            with tarfile.open(target) as archive:
                archive.extractall(folder, filter="data")
            done.write_text(entry["sha256"], encoding="utf-8")
            state += ", unpacked"
    return state


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=os.environ.get("FF_TEST_DATA"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args(argv)
    if args.dest is None:
        print("Set FF_TEST_DATA or pass --dest.", file=sys.stderr)
        return 2
    entries = tomllib.loads(args.manifest.read_text(encoding="utf-8"))["file"]
    failed = 0
    for entry in entries:
        if args.only and entry["name"] not in args.only:
            continue
        try:
            print(f"{entry['name']}: {fetch(entry, Path(args.dest))}")
        except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
            failed += 1
            print(f"{entry['name']}: FAILED: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
