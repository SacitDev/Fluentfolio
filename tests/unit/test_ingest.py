"""Taking a recording in: the type from its bytes, display names, the size and length limits, stored names."""

import hashlib
import sys
from collections import namedtuple
from collections.abc import Callable
from pathlib import Path

import pytest

from fluentfolio.core.errors import AppError
from fluentfolio.ingest import secondary, upload
from fluentfolio.settings.model import Limits

# First bytes of each accepted type, written out from the published file formats.
HEADS = {
    b"RIFF\x24\x00\x00\x00WAVEfmt ": "wav",
    b"RF64\xff\xff\xff\xffWAVEds64": "wav",
    b"\x00\x00\x00\x20ftypM4A \x00\x00": "mp4",
    b"\x00\x00\x00\x18ftypisom\x00\x00": "mp4",
    b"\x00\x00\x00\x14ftypqt  \x00\x00": "mp4",
    b"\x00\x00\x00\x08wide\x00\x00\x00\x00": "mp4",
    b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81": "matroska",
    b"OggS\x00\x02\x00\x00\x00\x00\x00\x00": "ogg",
    b"fLaC\x00\x00\x00\x22\x10\x00\x10\x00": "flac",
    b"ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00": "mp3",
    b"\xff\xfb\x90\x64\x00\x00\x00\x00\x00\x00\x00\x00": "mp3",  # MPEG-1 Layer III frame, no tag
}
REFUSED = [
    b"#EXTM3U\nhttp://example.com/a.wav\n",
    b"\xef\xbb\xbf#EXTM3U\n",
    b"ffconcat version 1.0\nfile a.wav\n",
    b"\xff\xf1\x50\x80\x00\x1f\xfc\x00\x00\x00\x00\x00",  # AAC in ADTS: same sync bits, layer bits 00
    b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n",
    b"",
    hashlib.sha256(b"not a recording").digest()[:12],
]


def test_the_type_comes_from_the_first_bytes_and_playlists_are_out() -> None:
    for head, name in HEADS.items():
        assert upload.sniff(head).name == name, head
    for head in REFUSED:
        with pytest.raises(AppError) as caught:
            upload.sniff(head)
        assert caught.value.code == "E_INPUT_UNSUPPORTED"
        assert caught.value.detail == ("a playlist" if b"EXTM3U" in head or b"ffconcat" in head else "")
    assert {k.demuxer for k in upload.KINDS.values()} == {"wav", "mp3", "mov", "matroska", "ogg", "flac"}


def test_a_file_name_is_display_text_only() -> None:
    assert upload.display_name("evil\u202ewav.exe") == "evilwav.exe"  # a right-to-left override
    assert upload.display_name("a\u200bb\u200dc\ufeff.m4a") == "abc.m4a"  # zero-width marks
    assert upload.display_name(" line\nbreak\ttab\u2028\x00.wav ") == "linebreaktab.wav"
    assert upload.display_name("Zoë practice – Part 2.m4a") == "Zoë practice – Part 2.m4a"
    assert upload.display_name("x" * 119) == "x" * 119
    assert upload.display_name("x" * 120) == "x" * 120
    assert upload.display_name("x" * 119 + "yz") == "x" * 117 + "..."  # shortened, and it says so
    assert upload.display_name("a" * 116 + " " + "b" * 10) == "a" * 116 + "..."  # no space before it
    assert len(upload.display_name("é" * 500)) == 120


def test_the_size_limit_at_its_edge() -> None:
    limits = Limits(max_upload_mb=3)
    most = 3_000_000  # decimal MB, as the app prints sizes
    upload.check_size(most - 1, limits)
    upload.check_size(most, limits)
    with pytest.raises(AppError) as caught:
        upload.check_size(most + 1, limits)
    assert caught.value.code == "E_INPUT_TOO_LARGE"
    assert caught.value.detail == "3 MB, at most 3 MB"


def test_the_default_size_limit_is_two_gb_as_the_pages_count_them() -> None:
    upload.check_size(2 * 10**9, Limits())
    with pytest.raises(AppError) as caught:
        upload.check_size(2 * 10**9 + 1, Limits())
    assert caught.value.detail == "2.0 GB, at most 2.0 GB"


def test_the_length_limits_at_each_edge() -> None:
    limits = Limits()  # 10 s, 60 s, 60 min, 180 min

    def outcome(seconds: float, confirmed: bool = False) -> str | list[str]:
        try:
            return upload.check_duration(seconds, limits, confirmed=confirmed)
        except AppError as error:
            return error.code

    assert [outcome(s) for s in (0.0, 9.99, 10.0, 10.01)] == [
        "E_INPUT_TOO_SHORT", "E_INPUT_TOO_SHORT", ["W_INPUT_SHORT"], ["W_INPUT_SHORT"],
    ]  # fmt: skip
    assert [outcome(s) for s in (59.99, 60.0, 60.01)] == [["W_INPUT_SHORT"], [], []]
    assert [outcome(s) for s in (3599.99, 3600.0, 3600.01)] == [[], [], "E_CONFIRM_LONG"]
    assert [outcome(s, confirmed=True) for s in (10799.99, 10800.0, 10800.01)] == [[], [], "E_INPUT_TOO_LONG"]
    assert outcome(10800.01) == "E_INPUT_TOO_LONG"  # too long is refused even before anyone confirms


def test_a_recording_is_stored_under_the_job_and_never_under_its_name(tmp_path: Path) -> None:
    job = tmp_path / "jobs" / "20261013-170509-abcd"
    (job / "input").mkdir(parents=True)
    raw = b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(range(256)) * 40
    for name in ("..\\..\\evil\u202ewav.exe", "Part 2 (final).mp4", "Zoë's answer.wav"):
        source = tmp_path / "upload" / name.replace("\\", "_")
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(raw)
        info = upload.store_input(job, source, Limits())
        assert sorted(p.name for p in (job / "input").iterdir()) == ["source.wav"]
        assert (job / "input" / "source.wav").read_bytes() == raw
        assert info == {
            "sha256": hashlib.sha256(raw).hexdigest(), "container": "wav", "bytes": len(raw),
            "display_name": upload.display_name(source.name),
        }  # fmt: skip


def test_a_recording_that_does_not_fit_on_the_disk_is_not_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job"
    (job / "input").mkdir(parents=True)
    source = tmp_path / "a.flac"
    source.write_bytes(b"fLaC" + bytes(5_000_000))
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(upload.shutil, "disk_usage", lambda _: usage(10**9, 10**9 - 10**6, 10**6))
    with pytest.raises(AppError) as caught:
        upload.store_input(job, source, Limits())
    assert (caught.value.code, caught.value.detail) == ("E_DISK_FULL", "need 5 MB, have 1 MB")
    assert list((job / "input").iterdir()) == []


def held_without_sharing(path: Path) -> Callable[[], None]:
    """Open a file as a recorder that is still writing it does on Windows: no other program may read it."""
    assert sys.platform == "win32"
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    # GENERIC_READ, share mode 0 (none), OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL
    handle = kernel32.CreateFileW(str(path), 0x80000000, 0, None, 3, 0x80, None)
    assert handle not in (None, ctypes.c_void_p(-1).value), ctypes.get_last_error()
    return lambda: kernel32.CloseHandle(ctypes.c_void_p(handle))


def locked_from(path: Path, offset: int) -> Callable[[], None]:
    """Lock the bytes from `offset` to the end, as another program may on Windows; the start stays open."""
    assert sys.platform == "win32"
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    # GENERIC_READ, share mode 3 (others may read and write), OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL
    handle = kernel32.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
    assert handle not in (None, ctypes.c_void_p(-1).value), ctypes.get_last_error()
    rest = path.stat().st_size - offset
    assert kernel32.LockFile(ctypes.c_void_p(handle), offset, 0, rest, 0), ctypes.get_last_error()
    return lambda: kernel32.CloseHandle(ctypes.c_void_p(handle))  # closing the handle ends the lock


def refused_as_unreadable(read: Callable[[Path], object], path: Path) -> None:
    with pytest.raises(AppError) as caught:
        read(path)
    assert caught.value.code == "E_INPUT_UNREADABLE" and str(path) not in str(caught.value)


def test_a_file_that_cannot_be_opened_is_refused_without_its_path(tmp_path: Path) -> None:
    job = tmp_path / "job"
    (job / "input").mkdir(parents=True)
    folder = tmp_path / "answer.wav"
    folder.mkdir()  # it exists, but no system opens it as a file
    refused_as_unreadable(upload.sniff_file, folder)
    refused_as_unreadable(lambda p: secondary.read_text(p, (".wav",)), folder)
    if sys.platform == "win32":
        recording, questions = tmp_path / "held.wav", tmp_path / "held.txt"
        recording.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(64))
        questions.write_text("Do you work or study?\n", encoding="utf-8")
        for path, read in ((recording, lambda p: upload.store_input(job, p, Limits())),
                           (questions, lambda p: secondary.read_text(p, (".txt",)))):  # fmt: skip
            release = held_without_sharing(path)
            try:
                refused_as_unreadable(read, path)
            finally:
                release()
        long = tmp_path / "long.wav"
        long.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(2_000_000))
        release = locked_from(long, 1_000_000)  # the format check reads the start; the copy meets the lock
        try:
            refused_as_unreadable(lambda p: upload.store_input(job, p, Limits()), long)
        finally:
            release()
        assert list((job / "input").iterdir()) == []
