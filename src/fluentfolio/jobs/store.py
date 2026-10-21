"""Job folders and job.json: IDs, layout, liveness, resume and clean-up. Only the work.lock holder writes."""

from __future__ import annotations

import re
import secrets
import shutil
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fluentfolio.core.enums import JobState, Status
from fluentfolio.core.errors import InputError, SetupError
from fluentfolio.core.files import read_json, write_json
from fluentfolio.core.paths import job_tmp, mfa_root, safe_join

JOB_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")
FOLDERS = ("input", "stages", "analyzers", "logs", "export")
SWEEP_AGE = timedelta(days=1)
CANCEL_FILE = "cancel.request"  # written by a cancel from any process, obeyed by the job's own loop


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    status: Status | None = None
    started_utc: str | None = None
    ended_utc: str | None = None
    device: str | None = None
    fallback: dict[str, str] | None = None  # {"to": what ran instead, "after": the first failure's code}
    error_code: str | None = None
    exit_code_hex: str | None = None
    wall_ms: int | None = None
    peak_ram_mb: int | None = None
    peak_vram_mb: int | None = None


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_name=True)
    schema_: int = Field(1, alias="schema")
    job_id: str
    created_utc: str
    updated_utc: str
    app_version: str
    mode: str | None = None
    layout: str | None = None
    swap_style: str | None = None
    offset_ms: int | None = None
    asr: str | None = None
    device: dict[str, str | None] = {}
    light_run: bool = False
    selftest: bool = False
    settings: dict[str, Any] = {}
    input: dict[str, Any] = {}
    consent: dict[str, Any] = {}
    state: JobState = JobState.RUNNING
    stage: str | None = None
    stages: list[StageRecord] = []
    error: dict[str, Any] | None = None
    rerun_of: str | None = None  # the finished job this one runs again in another mode
    rerun_from: str | None = None  # its first step; the steps before it are the other job's, copied


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_job_id(now: datetime | None = None) -> str:
    """`<YYYYMMDD-HHMMSS>-<4 hex>` in UTC; no user input ever enters a folder name."""
    return f"{(now or datetime.now(UTC)):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def job_dir(data: Path, job_id: str) -> Path:
    if not JOB_ID.fullmatch(job_id) or not (path := safe_join(data / "jobs", job_id)).is_dir():
        raise InputError("E_JOB_NOT_FOUND", job_id if JOB_ID.fullmatch(job_id) else "")
    return path


def create(data: Path, record: JobRecord) -> Path:
    path = safe_join(data / "jobs", record.job_id)
    for name in FOLDERS:
        (path / name).mkdir(parents=True, exist_ok=True)
    write(data, record)
    return path


def write(data: Path, record: JobRecord) -> None:
    record.updated_utc = utc_now()
    write_json(
        safe_join(data / "jobs", record.job_id) / "job.json", record.model_dump(mode="json", by_alias=True)
    )


def _load(data: Path, job_id: str) -> JobRecord | None:
    try:
        return JobRecord.model_validate(read_json(job_dir(data, job_id) / "job.json"))
    except (OSError, ValueError):  # pydantic's ValidationError is a ValueError
        return None


def _interrupted(record: JobRecord) -> JobRecord:
    return record.model_copy(update={"state": JobState.INTERRUPTED, "error": {"code": "E_JOB_INTERRUPTED"}})


def read(data: Path, job_id: str, *, lock_free: bool) -> JobRecord:
    """The job as a reader shows it: `running` with a free work.lock, or a damaged file, is interrupted."""
    record = _load(data, job_id)
    if record is None:
        stamp = utc_now()
        return _interrupted(
            JobRecord(job_id=job_id, created_utc=stamp, updated_utc=stamp, app_version="unknown")
        )
    return _interrupted(record) if record.state is JobState.RUNNING and lock_free else record


def list_ids(data: Path) -> list[str]:
    folder = data / "jobs"
    return sorted(p.name for p in folder.iterdir() if JOB_ID.fullmatch(p.name)) if folder.is_dir() else []


def mark_interrupted(data: Path) -> list[str]:
    """Called by the new work.lock holder only: every job still marked running stopped without finishing.

    A damaged or unknown record is left as it is, so nothing a newer version wrote is lost.
    """
    changed = []
    for job_id in list_ids(data):
        record = _load(data, job_id)
        if record is not None and record.state is JobState.RUNNING:
            write(data, _interrupted(record))
            # A cancel the dead job never saw must not stop the job when it is run again.
            (job_dir(data, job_id) / CANCEL_FILE).unlink(missing_ok=True)
            changed.append(job_id)
    return changed


def resume_point(path: Path, results: list[str], finished: tuple[Status, ...] = (Status.OK,)) -> int | None:
    """Index of the first result (in run order) that is missing or not finished; later results are deleted.

    None means every stage is finished, so there is nothing to run.
    """
    for index, name in enumerate(results):
        result = path / name
        if not result.is_file() or _status(result) not in finished:
            for later in results[index + 1 :]:
                (path / later).unlink(missing_ok=True)
            return index
    return None


def _status(result: Path) -> str | None:
    try:
        status = read_json(result).get("status")
    except (OSError, ValueError, AttributeError):
        return None
    return status if isinstance(status, str) else None


def remove(paths: Iterable[Path]) -> list[str]:
    """Delete folders and files, going on past any that another program holds; returns the ones that stayed.

    On Windows a virus scan, the search indexer or a child that has not exited can hold a file for a while.
    """
    left: list[str] = []
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, onexc=lambda _call, name, _exc: left.append(str(name)))
        else:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                left.append(str(path))
    return left


def delete(data: Path, app: Path, job_id: str) -> list[str]:
    """Remove a job with its temporary and MFA folders; the caller holds work.lock. Returns what stayed.

    The record is marked failed with E_DELETE_INCOMPLETE first and goes last, so a job that keeps a held
    file stays in History saying what happened, and can be deleted again. When the record itself is held,
    nothing is deleted: the job stays whole rather than a finished job without its results. When the mark
    cannot be written for another reason, above all a full disk, the files go without it: deleting is how
    the learner gets space back.
    """
    path = job_dir(data, job_id)
    record = read(data, job_id, lock_free=False)
    record.state, record.error = JobState.FAILED, {"code": "E_DELETE_INCOMPLETE"}
    try:
        write(data, record)
    except PermissionError:  # Windows' answer while another program has job.json open
        return [str(path / "job.json")]
    except (OSError, SetupError):  # SetupError is atomic_write's E_DISK_FULL
        pass  # Known limit: a held file on a full disk leaves the job unmarked; revisit if reported.
    inside = (p for p in path.iterdir() if p.name != "job.json")
    return remove((job_tmp(app, job_id), mfa_root(app, job_id), *inside)) or remove((path,))


def selftest_jobs(data: Path) -> list[str]:
    """Self-test jobs on disk, which only a self-test that stopped before deleting its job leaves behind."""
    return [job_id for job_id in list_ids(data) if (r := _load(data, job_id)) is not None and r.selftest]


def expired(data: Path, days: int, now: datetime | None = None) -> list[str]:
    """Jobs older than the retention setting; 0 keeps every job."""
    if days <= 0:
        return []
    cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
    stamp = cutoff.strftime("%Y%m%d-%H%M%S")
    return [job_id for job_id in list_ids(data) if job_id[:15] < stamp]


def sweep(data: Path, app: Path, now: float | None = None) -> int:
    """Delete leftover job temp folders and uploads older than a day; the caller holds work.lock.

    An upload is `<data>/tmp/<id>.part` while it arrives and `<id>.upload` until a job takes it in.
    """
    limit = (now or time.time()) - SWEEP_AGE.total_seconds()
    old = [p for p in (app / "tmp").glob("*") if p.name not in ("mfa", "misc") and p.stat().st_mtime < limit]
    old += [p for p in (app / "tmp" / "mfa").glob("*") if p.stat().st_mtime < limit]
    uploads = [*(data / "tmp").glob("*.part"), *(data / "tmp").glob("*.upload")]
    old += [p for p in uploads if p.stat().st_mtime < limit]
    remove(old)  # what another program holds goes at a later sweep
    return len(old)
