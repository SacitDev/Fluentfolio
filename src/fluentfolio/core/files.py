"""State files written whole, folder copies checked before deleting, and downloads unpacked safely."""

from __future__ import annotations

import errno
import json
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

from fluentfolio.core.errors import SetupError

REPLACE_TRIES = 20  # about 1 s: antivirus and indexers hold new files open for a moment on Windows
REPLACE_WAIT_S = 0.05
MAX_ZIP_RATIO = 100


def size_text(n: int) -> str:
    """A size for people: 2.1 GB, 37 MB, 96 KB or 512 bytes, so a small file never reads as 1 MB."""
    # Each unit starts where the smaller one would round to 1000.
    if n >= 999_500_000:
        return f"{n / 1e9:.1f} GB"
    if n >= 999_500:
        return f"{n / 1e6:.0f} MB"
    if n >= 1000:  # noqa: PLR2004
        return f"{n / 1e3:.0f} KB"
    return "1 byte" if n == 1 else f"{n} bytes"


def atomic_write(path: Path, data: str | bytes) -> None:
    """Write a temporary file next to path, then replace path, so a reader never sees half a file."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        with part.open("wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
    except OSError as exc:
        part.unlink(missing_ok=True)
        if exc.errno == errno.ENOSPC:
            free = shutil.disk_usage(path.parent).free
            raise SetupError("E_DISK_FULL", f"need {size_text(len(raw))}, have {size_text(free)}") from exc
        raise
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(part, path)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                part.unlink(missing_ok=True)
                raise
            time.sleep(REPLACE_WAIT_S)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=1, ensure_ascii=False))


def read_replaced(path: Path) -> str:
    """Read a state file that another process may be replacing at this moment."""
    # On Windows the file cannot be opened while it is being replaced, or while antivirus holds it.
    for _ in range(REPLACE_TRIES - 1):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            time.sleep(REPLACE_WAIT_S)
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a JSON state file, trying once more after a short wait if it does not parse."""
    try:
        return json.loads(read_replaced(path))
    except json.JSONDecodeError:
        time.sleep(REPLACE_WAIT_S)
        return json.loads(read_replaced(path))


def tree_size(folder: Path, skip: tuple[str, ...] = ()) -> tuple[int, int]:
    """The number of files under folder and their total size in bytes, leaving out the names in skip."""
    files = [p for p in folder.rglob("*") if p.is_file() and p.name not in skip]
    return len(files), sum(p.stat().st_size for p in files)


def copy_verified(src: Path, dst: Path, skip: tuple[str, ...] = ()) -> None:
    """Copy a folder and check the copy's file count and size; a bad copy is removed and src is untouched."""
    if dst.exists() and any(dst.iterdir()):
        raise SetupError("E_MOVE_FAILED", "the new folder is not empty")
    need = tree_size(src, skip)[1]
    dst.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dst).free
    if need > free:
        raise SetupError("E_DISK_FULL", f"need {size_text(need)}, have {size_text(free)}")
    try:
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*skip), dirs_exist_ok=True)
        if tree_size(src, skip) != tree_size(dst, skip):
            raise SetupError("E_MOVE_FAILED", "file count or size differs")
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise


def _within(root: Path, name: str) -> bool:
    return (root / name).resolve().is_relative_to(root)


def _over_budget(total: int, budget: int) -> None:
    if total > budget:
        raise SetupError("E_DOWNLOAD_UNSAFE", f"unpacks to over {size_text(budget)}")


def extract(archive: Path, dest: Path, budget: int) -> None:
    """Unpack a downloaded, hash-checked archive into dest, refusing it whole when any member is unsafe.

    Unsafe: a path or link that leaves dest, over budget bytes in all, or a ZIP member packed over 100 to 1.
    """
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            _over_budget(sum(m.file_size for m in z.infolist()), budget)
            for m in z.infolist():
                if not _within(root, m.filename):
                    raise SetupError("E_DOWNLOAD_UNSAFE", "a member outside the folder")
                if m.file_size > MAX_ZIP_RATIO * max(m.compress_size, 1):
                    raise SetupError("E_DOWNLOAD_UNSAFE", f"a member packed over {MAX_ZIP_RATIO} to 1")
            z.extractall(root)  # noqa: S202 - every member was checked above
        return
    with tarfile.open(archive) as t:
        _over_budget(sum(m.size for m in t.getmembers()), budget)
        try:
            for member in t.getmembers():  # every member is checked before the first one is written
                tarfile.data_filter(member, str(root))
        except tarfile.FilterError as exc:
            raise SetupError("E_DOWNLOAD_UNSAFE", "a member or link outside the folder") from exc
        t.extractall(root, filter="data")
