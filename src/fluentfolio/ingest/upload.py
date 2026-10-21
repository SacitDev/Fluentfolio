"""Taking a recording in: its type from its first bytes, size and length limits, storage under the job ID."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from fluentfolio.core.errors import InputError, SetupError
from fluentfolio.core.files import size_text
from fluentfolio.settings.model import Limits

HEAD_BYTES = 12
CHUNK = 1 << 20
MAX_NAME = 120
# Text playlists make a decoder open other files or URLs; they are refused before any decoder sees them.
PLAYLISTS = (b"#EXTM3U", b"ffconcat")
QUICKTIME_ATOMS = (b"ftyp", b"moov", b"mdat", b"wide", b"free")


@dataclass(frozen=True)
class Kind:
    name: str  # the container, as job.json records it
    ext: str  # the stored file's extension
    demuxer: str  # the decoder's container reader, always forced


WAV = Kind("wav", "wav", "wav")
MP3 = Kind("mp3", "mp3", "mp3")
MP4 = Kind("mp4", "mp4", "mov")  # MP4, M4A and MOV share one reader
MATROSKA = Kind("matroska", "mkv", "matroska")  # WebM is a kind of Matroska
OGG = Kind("ogg", "ogg", "ogg")
FLAC = Kind("flac", "flac", "flac")
KINDS = {k.name: k for k in (WAV, MP3, MP4, MATROSKA, OGG, FLAC)}


def sniff(head: bytes) -> Kind:
    """The container, from the file's first bytes only; the file name and extension are never trusted."""
    if head.removeprefix(b"\xef\xbb\xbf").startswith(PLAYLISTS):
        raise InputError("E_INPUT_UNSUPPORTED", "a playlist")
    if head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE":
        return WAV
    if head[4:8] in QUICKTIME_ATOMS:
        return MP4
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return MATROSKA
    if head[:4] == b"OggS":
        return OGG
    if head[:4] == b"fLaC":
        return FLAC
    # An ID3 tag, or an MPEG audio frame header for Layer III (11 sync bits, then the layer bits 01).
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE6 == 0xE2):
        return MP3
    raise InputError("E_INPUT_UNSUPPORTED")


def sniff_file(path: Path) -> Kind:
    try:
        with path.open("rb") as f:
            head = f.read(HEAD_BYTES)
    except OSError as exc:  # a recorder still holds it open, or this user may not read it
        raise InputError("E_INPUT_UNREADABLE", type(exc).__name__) from exc
    return sniff(head)


def _shown(c: str) -> bool:
    # Categories C* hold control, zero-width and direction marks; Zl and Zp break lines.
    category = unicodedata.category(c)
    return not category.startswith("C") and category not in ("Zl", "Zp")


def display_name(name: str) -> str:
    """The name as display text only: no control, zero-width or direction marks, at most 120 characters.

    A longer name ends in "..." so that a reader can see it was shortened.
    """
    shown = "".join(filter(_shown, name)).strip()
    return shown if len(shown) <= MAX_NAME else shown[: MAX_NAME - 3].rstrip() + "..."


def check_size(size: int, limits: Limits) -> None:
    most = limits.max_upload_mb * 1_000_000  # decimal MB, as every size the app shows
    if size > most:
        raise InputError("E_INPUT_TOO_LARGE", f"{size_text(size)}, at most {size_text(most)}")


def check_duration(seconds: float, limits: Limits, *, confirmed: bool) -> list[str]:
    """Refuse a recording outside the length limits; a long one needs confirming. Returns warning codes."""
    if seconds < limits.min_s:
        raise InputError("E_INPUT_TOO_SHORT", f"{seconds:.1f} s, at least {limits.min_s} s")
    if seconds > limits.max_min * 60:
        raise InputError("E_INPUT_TOO_LONG", f"{seconds / 60:.1f} min, at most {limits.max_min} min")
    if seconds > limits.confirm_min * 60 and not confirmed:
        raise InputError("E_CONFIRM_LONG", f"{seconds / 60:.1f} min")
    return ["W_INPUT_SHORT"] if seconds < limits.short_s else []


def _read(src: BinaryIO) -> bytes:
    try:
        return src.read(CHUNK)
    except OSError as exc:  # another program locked a later part of the file, or its drive was removed
        raise InputError("E_INPUT_UNREADABLE", type(exc).__name__) from exc


def store_input(
    job: Path, source: Path, limits: Limits, stem: str = "source", name: str | None = None
) -> dict[str, Any]:
    """Copy a recording into the job as input/<stem>.<ext>; its own name survives only as display text.

    `name` is that name when the file was saved under another one, as an upload is.
    """
    size = source.stat().st_size
    check_size(size, limits)
    kind = sniff_file(source)
    target = job / "input" / f"{stem}.{kind.ext}"
    part = target.with_name(target.name + ".part")
    free = shutil.disk_usage(target.parent).free
    if size > free:
        raise SetupError("E_DISK_FULL", f"need {size_text(size)}, have {size_text(free)}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as src, part.open("wb") as dst:
            for block in iter(lambda: _read(src), b""):
                digest.update(block)
                dst.write(block)
        os.replace(part, target)
    except (OSError, InputError) as exc:
        part.unlink(missing_ok=True)
        if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
            free = shutil.disk_usage(target.parent).free
            raise SetupError("E_DISK_FULL", f"need {size_text(size)}, have {size_text(free)}") from exc
        raise
    return {
        "sha256": digest.hexdigest(),
        "container": kind.name,
        "bytes": size,
        "display_name": display_name(source.name if name is None else name),
    }
