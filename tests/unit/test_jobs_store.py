"""Job IDs, the job folder, liveness, resume and the clean-up of old files."""

import errno
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from fluentfolio.core.enums import JobState, Status
from fluentfolio.core.errors import AppError
from fluentfolio.jobs import store
from fluentfolio.jobs.locks import work_lock


def new(data: Path, job_id: str = "20261013-170509-abcd", state: JobState = JobState.RUNNING) -> Path:
    now = store.utc_now()
    record = store.JobRecord(job_id=job_id, created_utc=now, updated_utc=now, app_version="1.0", state=state)
    return store.create(data, record)


def test_a_job_id_is_the_utc_time_and_four_hex_digits() -> None:
    job_id = store.new_job_id(datetime(2026, 10, 13, 17, 5, 9, tzinfo=UTC))
    assert job_id.startswith("20261013-170509-") and store.JOB_ID.fullmatch(job_id)
    assert all(c in "0123456789abcdef" for c in job_id[-4:])


def test_a_new_job_gets_the_documented_folders_and_record(tmp_path: Path) -> None:
    folder = new(tmp_path)
    assert folder == tmp_path / "jobs" / "20261013-170509-abcd"
    assert sorted(p.name for p in folder.iterdir()) == [
        "analyzers",
        "export",
        "input",
        "job.json",
        "logs",
        "stages",
    ]
    record = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    assert (record["schema"], record["state"], record["job_id"]) == (1, "running", "20261013-170509-abcd")


@pytest.mark.parametrize(
    "job_id", ["20261013-170509-ffff", "../20261013-170509-abcd", "x", "20261013-170509-ABCD"]
)
def test_an_unknown_or_malformed_job_id_is_not_found(tmp_path: Path, job_id: str) -> None:
    new(tmp_path)
    with pytest.raises(AppError) as caught:
        store.job_dir(tmp_path, job_id)
    assert (caught.value.code, caught.value.entry.exit_code) == ("E_JOB_NOT_FOUND", 4)


def test_a_damaged_or_abandoned_job_reads_as_interrupted(tmp_path: Path) -> None:
    folder = new(tmp_path)
    assert store.read(tmp_path, "20261013-170509-abcd", lock_free=False).state is JobState.RUNNING
    abandoned = store.read(
        tmp_path, "20261013-170509-abcd", lock_free=True
    )  # running, and nobody holds work.lock
    assert (abandoned.state, abandoned.error) == (JobState.INTERRUPTED, {"code": "E_JOB_INTERRUPTED"})
    (folder / "job.json").write_text('{"state": "runn', encoding="utf-8")
    damaged = store.read(tmp_path, "20261013-170509-abcd", lock_free=False)
    assert (damaged.state, damaged.error, damaged.app_version) == (
        JobState.INTERRUPTED, {"code": "E_JOB_INTERRUPTED"}, "unknown",
    )  # fmt: skip


def test_only_jobs_still_marked_running_are_rewritten_as_interrupted(tmp_path: Path) -> None:
    new(tmp_path, "20261013-170509-0001")
    new(tmp_path, "20261013-170509-0002", JobState.COMPLETED)
    damaged = new(tmp_path, "20261013-170509-0003") / "job.json"
    damaged.write_text('{"state": "running", "from_a_newer_version": 1}', encoding="utf-8")
    assert store.mark_interrupted(tmp_path) == ["20261013-170509-0001"]
    on_disk = json.loads(
        (tmp_path / "jobs" / "20261013-170509-0001" / "job.json").read_text(encoding="utf-8")
    )
    assert (on_disk["state"], on_disk["error"]) == ("interrupted", {"code": "E_JOB_INTERRUPTED"})
    assert damaged.read_text(encoding="utf-8") == '{"state": "running", "from_a_newer_version": 1}'


def test_liveness_follows_the_work_lock(tmp_path: Path) -> None:
    new(tmp_path)
    with work_lock(tmp_path):
        assert not work_lock(tmp_path).is_free()
        assert (
            store.read(tmp_path, "20261013-170509-abcd", lock_free=work_lock(tmp_path).is_free()).state
            == "running"
        )
    assert work_lock(tmp_path).is_free()


def test_resume_finds_the_first_unfinished_result_and_deletes_every_later_one(tmp_path: Path) -> None:
    results = ["stages/media.json", "stages/asr.json", "stages/speakers.json", "stages/words.json"]
    (tmp_path / "stages").mkdir()
    for name, status in zip(results, ("ok", "ok", "failed", "ok"), strict=True):
        (tmp_path / name).write_text(json.dumps({"status": status}), encoding="utf-8")
    assert store.resume_point(tmp_path, results) == 2
    assert sorted(p.name for p in (tmp_path / "stages").iterdir()) == [
        "asr.json",
        "media.json",
        "speakers.json",
    ]
    (tmp_path / "stages" / "speakers.json").write_text('{"status": "disabled"}', encoding="utf-8")
    assert store.resume_point(tmp_path, results[:3], (Status.OK, Status.DISABLED)) is None
    assert store.resume_point(tmp_path, results) == 2  # by default only ok counts as finished
    (tmp_path / "stages" / "speakers.json").write_text('{"status": "ok"}', encoding="utf-8")
    assert store.resume_point(tmp_path, results) == 3  # words.json is missing


def test_deleting_a_job_removes_its_temp_and_alignment_folders(tmp_path: Path) -> None:
    data, app = tmp_path / "data", tmp_path / "app"
    new(data)
    for folder in (app / "tmp" / "20261013-170509-abcd", app / "tmp" / "mfa" / "20261013-170509-abcd"):
        folder.mkdir(parents=True)
        (folder / "x.wav").write_bytes(b"x")
    store.delete(data, app, "20261013-170509-abcd")
    assert not (data / "jobs" / "20261013-170509-abcd").exists()
    assert [p.name for p in (app / "tmp").iterdir()] == ["mfa"] and not any((app / "tmp" / "mfa").iterdir())


def test_a_held_file_keeps_the_record_marked_half_deleted_until_a_second_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, app, job_id = tmp_path / "data", tmp_path / "app", "20261013-170509-abcd"
    folder = new(data, state=JobState.COMPLETED)
    held = folder / "input" / "source.wav"
    held.write_bytes(b"RIFF")
    created = store.read(data, job_id, lock_free=True).created_utc
    real = os.unlink

    def unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        # Windows' answer while a media player has the recording open. By name, because on Linux rmtree
        # passes only the file's name with the folder's handle.
        if Path(path).name == held.name:
            raise PermissionError(32, "The process cannot access the file: another process is using it")
        real(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    assert store.delete(data, app, job_id) == [str(held), str(held.parent)]
    assert sorted(p.name for p in folder.iterdir()) == ["input", "job.json"]
    kept = store.read(data, job_id, lock_free=True)  # the same job, saying what happened, not a stopped one
    assert (kept.created_utc, kept.state) == (created, JobState.FAILED)
    assert kept.error == {"code": "E_DELETE_INCOMPLETE"}
    monkeypatch.setattr(os, "unlink", real)
    assert store.delete(data, app, job_id) == [] and not folder.exists()


def test_a_held_record_stops_the_delete_before_anything_goes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, app, job_id = tmp_path / "data", tmp_path / "app", "20261013-170509-abcd"
    folder = new(data, state=JobState.COMPLETED)
    (folder / "input" / "source.wav").write_bytes(b"RIFF")
    before = sorted(str(p.relative_to(folder)) for p in folder.rglob("*"))
    real = os.replace

    def replace(src: Any, dst: Any) -> None:
        if Path(dst) == folder / "job.json":  # Windows' answer while another program has the record open
            raise PermissionError(13, "Access is denied")
        real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr("fluentfolio.core.files.REPLACE_WAIT_S", 0)
    assert store.delete(data, app, job_id) == [str(folder / "job.json")]
    assert sorted(str(p.relative_to(folder)) for p in folder.rglob("*")) == before  # no .part file either
    assert store.read(data, job_id, lock_free=True).state is JobState.COMPLETED  # whole and finished


def test_a_full_disk_still_deletes_the_job_because_deleting_frees_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, app, job_id = tmp_path / "data", tmp_path / "app", "20261013-170509-abcd"
    folder = new(data, state=JobState.COMPLETED)
    (folder / "input" / "source.wav").write_bytes(b"RIFF")

    def fsync(_fd: int) -> None:  # the disk fills up while the mark is written
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "fsync", fsync)
    assert store.delete(data, app, job_id) == [] and not folder.exists()


def test_the_sweep_removes_leftovers_older_than_a_day_only(tmp_path: Path) -> None:
    data, app = tmp_path / "data", tmp_path / "app"
    old = time.time() - 25 * 3600
    paths = {
        "old job temp": app / "tmp" / "20261011-000000-aaaa",
        "new job temp": app / "tmp" / "20261013-000000-bbbb",
        "old alignment": app / "tmp" / "mfa" / "20261011-000000-aaaa",
        "shared temp": app / "tmp" / "misc",
    }
    for folder in paths.values():
        folder.mkdir(parents=True)
    (data / "tmp").mkdir(parents=True)
    for name in ("old.part", "new.part", "old.upload", "new.upload"):
        (data / "tmp" / name).write_bytes(b"x")
    for target in (
        paths["old job temp"],
        paths["old alignment"],
        paths["shared temp"],
        data / "tmp" / "old.part",
        data / "tmp" / "old.upload",
    ):
        os.utime(target, (old, old))
    assert store.sweep(data, app) == 4  # noqa: PLR2004
    assert sorted(p.name for p in (app / "tmp").iterdir()) == ["20261013-000000-bbbb", "mfa", "misc"]
    assert sorted(p.name for p in (data / "tmp").iterdir()) == ["new.part", "new.upload"]


def test_retention_lists_jobs_older_than_the_setting(tmp_path: Path) -> None:
    for job_id in ("20261001-120000-aaaa", "20261012-120000-bbbb"):
        new(tmp_path, job_id)
    now = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
    assert store.expired(tmp_path, 0, now) == []
    assert store.expired(tmp_path, 7, now) == ["20261001-120000-aaaa"]
    assert store.expired(tmp_path, 1, now) == ["20261001-120000-aaaa"]
    assert store.expired(tmp_path, 1, datetime(2026, 10, 13, 12, 0, 1, tzinfo=UTC)) == [
        "20261001-120000-aaaa", "20261012-120000-bbbb",
    ]  # fmt: skip
