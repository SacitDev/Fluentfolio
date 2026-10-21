"""State files are written whole or not at all, a folder copy is checked before anything is deleted, and a
downloaded archive unpacks only inside its folder."""

import errno
import io
import json
import os
import re
import shutil
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest

from fluentfolio.core import files, paths
from fluentfolio.core.errors import SetupError
from fluentfolio.settings import store as settings_store


def test_a_write_replaces_the_file_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "state" / "job.json"
    files.write_json(target, {"state": "running"})
    files.write_json(target, {"state": "completed", "name": "Zoë"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"state": "completed", "name": "Zoë"}
    assert [p.name for p in target.parent.iterdir()] == ["job.json"]


@pytest.mark.parametrize(
    ("size", "shown"),
    [
        (0, "0 bytes"), (1, "1 byte"), (999, "999 bytes"), (1000, "1 KB"),
        (96_044, "96 KB"),  # a 3 s 16 kHz mono WAV: it once read "1 MB"
        (999_499, "999 KB"), (999_500, "1 MB"), (1_000_000, "1 MB"), (37_400_000, "37 MB"),
        (999_499_999, "999 MB"), (999_500_000, "1.0 GB"), (2**31, "2.1 GB"),
    ],
)  # fmt: skip
def test_a_size_is_shown_in_the_unit_that_fits_it(size: int, shown: str) -> None:
    assert files.size_text(size) == shown


def test_a_full_disk_leaves_no_half_file_and_says_how_much_space_is_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "evidence.json"
    target.write_text("old", encoding="utf-8")

    def full(fd: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "fsync", full)
    with pytest.raises(SetupError) as caught:
        files.atomic_write(target, "x" * 3_000_000)
    assert caught.value.code == "E_DISK_FULL"
    assert re.fullmatch(r"need 3 MB, have \d+(\.\d)? (GB|MB)", caught.value.detail)
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["evidence.json"]


def test_a_reader_holding_the_file_for_a_moment_does_not_break_a_write(tmp_path: Path) -> None:
    target = tmp_path / "job.json"
    target.write_text("{}", encoding="utf-8")
    reader = target.open("rb")  # on Windows an open file cannot be replaced until it is closed
    threading.Timer(0.3, reader.close).start()
    files.write_json(target, {"state": "completed"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"state": "completed"}


def test_a_reader_tries_once_more_when_the_file_does_not_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "job.json"
    target.write_text('{"state": "running"}', encoding="utf-8")
    texts = iter(['{"state": "runn', '{"state": "running"}'])  # caught mid-write, then whole
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: next(texts))
    assert files.read_json(target) == {"state": "running"}


def test_a_reader_waits_while_windows_refuses_the_file_for_a_moment(tmp_path: Path) -> None:
    import msvcrt

    target = tmp_path / "job.json"
    target.write_text('{"state": "running"}', encoding="utf-8")
    fd = os.open(target, os.O_RDWR)
    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1024)
    with pytest.raises(PermissionError):
        target.read_text(encoding="utf-8")  # what a page saw while a job replaced its job.json

    def release() -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1024)
        os.close(fd)

    threading.Timer(0.3, release).start()
    assert files.read_json(target) == {"state": "running"}


def refused_twice(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Make the next two reads fail as Windows fails them while a file is replaced; returns the paths read."""
    real, tried = Path.read_text, []

    def read_text(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        tried.append(self)
        if len(tried) <= 2:  # noqa: PLR2004
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return real(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text)
    return tried


def test_each_reader_of_a_replaced_state_file_tries_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "job.json").write_text('{"state": "running"}', encoding="utf-8")
    (tmp_path / "state.json").write_text('{"backend": "cpu"}', encoding="utf-8")
    settings = 'schema_version = 1\n[review]\npolicy = "always"\n'
    (tmp_path / "settings.toml").write_text(settings, encoding="utf-8")
    tried = refused_twice(monkeypatch)
    assert files.read_json(tmp_path / "job.json") == {"state": "running"}
    assert len(tried) == 3  # noqa: PLR2004
    tried = refused_twice(monkeypatch)
    assert paths.install_state(tmp_path) == {"backend": "cpu"}
    assert len(tried) == 3  # noqa: PLR2004
    tried = refused_twice(monkeypatch)
    loaded = settings_store.load(tmp_path / "settings.toml")
    assert loaded.settings.review.policy == "always" and loaded.warnings == []
    assert len(tried) == 3  # noqa: PLR2004
    kept = ["job.json", "settings.toml", "state.json"]
    assert sorted(p.name for p in tmp_path.iterdir()) == kept  # settings.toml was not set aside as damaged


def test_a_file_that_stays_unreadable_still_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "job.json"
    target.write_text("{}", encoding="utf-8")

    def refuse(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        raise PermissionError(errno.EACCES, "Permission denied", str(self))

    monkeypatch.setattr(Path, "read_text", refuse)
    with pytest.raises(PermissionError):
        files.read_json(target)


def make_tree(root: Path) -> None:
    (root / "jobs" / "a").mkdir(parents=True)
    (root / "jobs" / "a" / "job.json").write_text("{}", encoding="utf-8")
    (root / "settings.toml").write_text("x = 1\n", encoding="utf-8")
    (root / "work.lock").write_text("", encoding="utf-8")


def test_a_verified_copy_has_every_file_except_the_skipped_ones(tmp_path: Path) -> None:
    make_tree(tmp_path / "old")
    files.copy_verified(tmp_path / "old", tmp_path / "new", skip=("work.lock",))
    assert sorted(p.relative_to(tmp_path / "new").as_posix() for p in (tmp_path / "new").rglob("*")) == [
        "jobs", "jobs/a", "jobs/a/job.json", "settings.toml",
    ]  # fmt: skip
    with pytest.raises(SetupError) as caught:
        files.copy_verified(tmp_path / "old", tmp_path / "new")
    assert caught.value.detail == "the new folder is not empty"


def test_a_copy_that_does_not_match_is_removed_and_the_source_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_tree(tmp_path / "old")
    real_copy = shutil.copytree

    def lossy(src: Path, dst: Path, *args: object, **kw: object) -> None:  # a copy that silently loses a file
        real_copy(
            src, dst, *args, **kw
        )  # copytree calls itself for subfolders, so only the top call loses one
        if Path(dst) == tmp_path / "new":
            (Path(dst) / "settings.toml").unlink()

    monkeypatch.setattr(files.shutil, "copytree", lossy)
    with pytest.raises(SetupError) as caught:
        files.copy_verified(tmp_path / "old", tmp_path / "new")
    assert (caught.value.code, caught.value.detail) == ("E_MOVE_FAILED", "file count or size differs")
    assert not (tmp_path / "new").exists() and (tmp_path / "old" / "settings.toml").is_file()


def make_zip(path: Path, members: dict[str, bytes], method: int = zipfile.ZIP_STORED) -> Path:
    with zipfile.ZipFile(path, "w", method) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def make_tar(path: Path, members: list[tarfile.TarInfo | tuple[str, bytes]]) -> Path:
    with tarfile.open(path, "w:gz") as t:
        for member in members:
            if isinstance(member, tarfile.TarInfo):
                t.addfile(member)
            else:
                info = tarfile.TarInfo(member[0])
                info.size = len(member[1])
                t.addfile(info, io.BytesIO(member[1]))
    return path


def link(name: str, target: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type, info.linkname = tarfile.SYMTYPE, target
    return info


def test_a_safe_archive_is_unpacked_inside_its_folder(tmp_path: Path) -> None:
    members = {"model/a.onnx": b"x" * 1000, "model/b.txt": b"hello"}
    files.extract(make_zip(tmp_path / "m.zip", members), tmp_path / "z", budget=1005)  # exactly the budget
    files.extract(make_tar(tmp_path / "m.tar.gz", list(members.items())), tmp_path / "t", budget=1006)
    for out in (tmp_path / "z", tmp_path / "t"):
        assert {
            p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()
        } == members


def test_an_unsafe_archive_is_refused_whole(tmp_path: Path) -> None:
    first = ("model/ok.txt", b"fine")
    cases = {
        "a member above the folder": make_zip(tmp_path / "1.zip", dict([first, ("../evil.txt", b"x")])),
        "an absolute member": make_zip(tmp_path / "2.zip", dict([first, ("/etc/evil.txt", b"x")])),
        "a tar member above the folder": make_tar(tmp_path / "3.tar.gz", [first, ("../evil.txt", b"x")]),
        "a tar link to a system folder": make_tar(tmp_path / "4.tar.gz", [first, link("model/etc", "/etc")]),
        "a tar link above the folder": make_tar(tmp_path / "5.tar.gz", [first, link("model/up", "../../..")]),
        "one byte over the budget, zip": make_zip(tmp_path / "6.zip", dict([first, ("model/b", b"x" * 7)])),
        "one byte over the budget, tar": make_tar(tmp_path / "7.tar.gz", [first, ("model/b", b"x" * 7)]),
        "a zip bomb": make_zip(tmp_path / "8.zip", {"model/zeros": bytes(1_000_000)}, zipfile.ZIP_DEFLATED),
    }
    for why, archive in cases.items():
        out = tmp_path / "out" / archive.name
        with pytest.raises(SetupError) as caught:
            files.extract(archive, out, budget=10 if "budget" in why else 10_000_000)
        assert caught.value.code == "E_DOWNLOAD_UNSAFE", why
        assert list(out.iterdir()) == [], why
    assert not (tmp_path / "out" / "evil.txt").exists() and not (tmp_path / "evil.txt").exists()
