"""Uploads: the browser sends a recording as the raw request body, saved in the data folder as it arrives.

The file goes to `<data>/tmp/<id>.part` while it arrives and becomes `<id>.upload` once complete, never
into the system temp folder or into memory. The bytes are counted as they arrive, so the size limit
holds whatever the request's Content-Length says. A failed upload leaves no file behind, and uploads
that a stopped app left behind are deleted when the app starts.
"""

from __future__ import annotations

import errno
import os
import re
import secrets
import shutil
from pathlib import Path

from starlette.requests import Request

from fluentfolio.core.errors import SetupError
from fluentfolio.core.files import size_text
from fluentfolio.ingest.upload import check_size
from fluentfolio.settings.model import Limits

PART, DONE = ".part", ".upload"
UPLOAD_ID = re.compile(r"[0-9a-f]{32}")


def folder(data: Path) -> Path:
    return data / "tmp"


def clear(data: Path) -> None:
    """Delete uploads left by an app that stopped; the jobs' own folders in tmp/ are not touched."""
    for leftover in [*folder(data).glob(f"*{PART}"), *folder(data).glob(f"*{DONE}")]:
        leftover.unlink(missing_ok=True)


def find(data: Path, upload_id: str) -> Path | None:
    """A finished upload by its ID; an ID of any other shape never reaches the file system."""
    path = folder(data) / f"{upload_id}{DONE}"
    return path if UPLOAD_ID.fullmatch(upload_id) and path.is_file() else None


def _check_space(size: int, where: Path) -> None:
    free = shutil.disk_usage(where).free
    if size > free:
        raise SetupError("E_DISK_FULL", f"need {size_text(size)}, have {size_text(free)}")


async def receive(request: Request, data: Path, limits: Limits) -> tuple[str, Path]:
    """Stream the body to disk, counting bytes; the ID and the finished file."""
    tmp = folder(data)
    tmp.mkdir(parents=True, exist_ok=True)
    declared = request.headers.get("content-length", "")
    if declared.isdigit():  # the early answer; the count below is what holds
        check_size(int(declared), limits)
        _check_space(int(declared), tmp)
    upload_id = secrets.token_hex(16)
    part = tmp / f"{upload_id}{PART}"
    received = 0
    done = False
    try:
        with part.open("wb") as out:
            async for chunk in request.stream():
                received += len(chunk)
                check_size(received, limits)
                out.write(chunk)
        target = part.with_suffix(DONE)
        os.replace(part, target)
        done = True
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise SetupError("E_DISK_FULL", f"need more than {size_text(received)}") from exc
        raise
    finally:
        if not done:  # refused, cut off by the browser, or a disk error: nothing half-written stays
            part.unlink(missing_ok=True)
    return upload_id, target
