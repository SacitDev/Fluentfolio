"""The job loop: runs a job's stages in order, classifies each outcome and records it in job.json."""

from __future__ import annotations

import logging
import shutil
import sys
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fluentfolio import __version__
from fluentfolio.core.enums import Device, JobState, Status
from fluentfolio.core.errors import AppError, CancelledError, InputError, catalogue
from fluentfolio.core.files import read_json, write_json
from fluentfolio.core.logs import crash_lines, handler
from fluentfolio.core.paths import job_tmp, mfa_root
from fluentfolio.jobs import store
from fluentfolio.jobs.awake import KeepAwake
from fluentfolio.jobs.runner import Outcome, child_env, run

# Windows STATUS_NO_MEMORY, and SIGKILL on Linux when the runner did not send it (the kernel's OOM killer).
RAM_EXIT_CODES = (0xC0000017, -9)
RAM_TEXT = ("MemoryError", "WinError 1455")
GPU_RETRY = ("E_GPU_OOM", "E_STAGE_CRASHED", "E_CUDA_RUNTIME")
CPU_FACTOR = 3
FINISHED = (Status.OK, Status.DISABLED)
MEDIA_RESULT = "stages/media.json"  # holds the decoded recording's length
GPU_FAULTS = (*GPU_RETRY, "E_GPU_UNAVAILABLE")
GPU_LOST = "The GPU could not be used ({code}), so the CPU was used."
STEP_MOVED = "The {step} step ran {to} on the CPU after {code}."  # a cause other than the card
joblog = logging.getLogger("fluentfolio.job")
joblog.propagate = False  # child output goes to the job's own log only


class Program(Protocol):
    """An outside program's command: its argv, folders for PATH, its own variables, where it runs and what
    it leaves behind (removed once its output is read)."""

    @property
    def argv(self) -> list[str]: ...
    @property
    def path(self) -> list[Path]: ...
    @property
    def env(self) -> dict[str, str]: ...
    @property
    def cwd(self) -> Path: ...
    @property
    def scratch(self) -> Path: ...


@dataclass(frozen=True)
class Stage:
    """One step: a child process (`fluentfolio.jobs.stage <id> <job>`) or, with `inline`, a pure function.

    With `prepare` and `finish`, an outside program: `prepare` writes its input and returns its command,
    or why it does not run; `finish` gets how it ended and returns the step's result.
    """

    id: str
    result: str  # its result file, relative to the job folder
    required: bool = True
    gpu: bool = False  # only speech recognition and speaker separation may use the GPU
    inline: Callable[[Path], dict[str, Any]] | None = None
    base_s: float = 600.0  # backstop: base_s plus per_min_s per audio minute, three times that on the CPU
    per_min_s: float = 60.0
    child: str = ""  # the child to start, when the step has several (one per speaker separation backend)
    fallback: str = ""  # a child that runs on the CPU when `child` fails; recorded and warned, never silent
    # Both get the job, app and models folders.
    prepare: Callable[[Path, Path, Path], Program | dict[str, Any]] | None = None
    finish: Callable[[Path, Path, Path, dict[str, Any]], dict[str, Any]] | None = None


@dataclass(frozen=True)
class Roots:
    app: Path
    data: Path
    models: Path


def child_argv(stage_id: str, job_id: str) -> list[str]:
    return [sys.executable, "-I", "-X", "utf8", "-m", "fluentfolio.jobs.stage", stage_id, job_id]


def _result(path: Path) -> dict[str, Any] | None:
    try:
        result = read_json(path)
        Status(result["status"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return result if isinstance(result, dict) else None


def classify(outcome: Outcome, result: dict[str, Any] | None) -> tuple[Status, str | None]:
    """A child's status and code, from its exit code and its result file only."""
    if outcome.stopped:
        return Status.FAILED, "E_JOB_CANCELLED" if outcome.stopped == "cancel" else "E_STAGE_TIMEOUT"
    if result is not None:
        code = result.get("code")
        if code is not None and not (isinstance(code, str) and code in catalogue()):
            code = "E_INTERNAL"
        if outcome.exit_code == 0:
            return Status(result["status"]), code
        return Status.FAILED, code or "E_STAGE_CRASHED"
    out_of_memory = any(text in line for line in outcome.stderr_tail for text in RAM_TEXT)
    if out_of_memory or outcome.exit_code in RAM_EXIT_CODES:
        return Status.FAILED, "E_RAM_OOM"
    return Status.FAILED, "E_STAGE_CRASHED"


class _Run:
    """One run of one job; the caller holds work.lock, so this is the only writer of job.json."""

    def __init__(self, roots: Roots, record: store.JobRecord, argv_for: Callable[[str, str], list[str]],
                 audio_min: float, no_progress_s: float,
                 on_progress: Callable[[str, dict[str, Any]], None] | None) -> None:  # fmt: skip
        self.roots, self.record, self.argv_for = roots, record, argv_for
        self.audio_min, self.no_progress_s, self.on_progress = audio_min, no_progress_s, on_progress
        self.path = store.job_dir(roots.data, record.job_id)
        self.cancel = self.path / store.CANCEL_FILE

    def minutes(self) -> float:
        """The recording's length for the backstop: measured once media ran, else the caller's bound."""
        try:
            return int(read_json(self.path / MEDIA_RESULT)["duration_ms"]) / 60000
        except (OSError, ValueError, KeyError, TypeError):
            return self.audio_min

    def child(
        self, stage: Stage, child_id: str, cpu: bool
    ) -> tuple[Status, str | None, Outcome, dict[str, Any] | None]:
        (self.path / stage.result).unlink(missing_ok=True)
        env = child_env(self.roots.app, self.roots.data, self.roots.models, self.record.job_id, cpu=cpu)
        outcome = run(
            self.argv_for(child_id, self.record.job_id), env=env, cwd=self.path,
            timeout_s=(stage.base_s + stage.per_min_s * self.minutes()) * (CPU_FACTOR if cpu else 1),
            no_progress_s=self.no_progress_s, should_stop=self.cancel.exists,
            on_progress=(lambda event: self.on_progress(stage.id, event)) if self.on_progress else None,
        )  # fmt: skip
        for line in outcome.stderr_tail:
            joblog.info("[%s] %s", stage.id, line)
        result = _result(self.path / stage.result)
        return (*classify(outcome, result), outcome, result)

    def inline(self, stage: Stage, step: Callable[[Path], dict[str, Any]]) -> tuple[Status, str | None]:
        try:
            result = step(self.path)
        except AppError as exc:
            return Status.FAILED, exc.code
        except Exception as error:  # noqa: BLE001 - one of the few places that catch everything
            joblog.error("[%s] the step failed: %s", stage.id, crash_lines(error))
            return Status.FAILED, "E_INTERNAL"
        write_json(self.path / stage.result, result)
        return Status(result["status"]), result.get("code")

    def external(
        self, stage: Stage, prepare: Callable[[Path, Path, Path], Program | dict[str, Any]],
        finish: Callable[[Path, Path, Path, dict[str, Any]], dict[str, Any]],
    ) -> Callable[[Path], dict[str, Any]]:  # fmt: skip
        """An outside program's step as one function: write its input, run it, read its output, clean up."""
        app, models = self.roots.app, self.roots.models

        def step(job: Path) -> dict[str, Any]:
            tool = prepare(job, app, models)
            if isinstance(tool, dict):  # it does not run: switched off, not installed or nothing to do
                return finish(job, app, models, tool)
            try:
                job_id = self.record.job_id
                env = child_env(app, self.roots.data, models, job_id, cpu=True, extra_path=tool.path)
                timeout = (stage.base_s + stage.per_min_s * self.minutes()) * CPU_FACTOR
                # The program prints no progress lines, so only the backstop or a cancel stops it early.
                outcome = run(
                    tool.argv, env=env | tool.env, cwd=tool.cwd, timeout_s=timeout, no_progress_s=timeout,
                    should_stop=self.cancel.exists,
                )  # fmt: skip
                for line in outcome.stderr_tail:
                    joblog.info("[%s] %s", stage.id, line)
                if outcome.stopped == "cancel":
                    raise CancelledError("E_JOB_CANCELLED")
                ran: dict[str, Any] = {"status": "ok", "wall_ms": outcome.wall_ms}
                if outcome.exit_code or outcome.stopped:
                    code = classify(outcome, None)[1]
                    ran |= {"status": "failed", "code": code, "exit_code_hex": hex(outcome.exit_code)}
                return finish(job, app, models, ran)
            finally:
                shutil.rmtree(tool.scratch, ignore_errors=True)

        return step

    def stage(self, stage: Stage, rec: store.StageRecord) -> None:
        step = stage.inline
        if stage.prepare is not None and stage.finish is not None:
            step = self.external(stage, stage.prepare, stage.finish)
        if step is not None:
            started = time.monotonic()
            rec.status, rec.error_code = self.inline(stage, step)
            rec.device = Device.CPU
            rec.wall_ms = int((time.monotonic() - started) * 1000)
            return
        cpu = not (stage.gpu and self.record.device.get("resolved") == Device.GPU)
        primary = stage.child or stage.id
        status, code, outcome, result = self.child(stage, primary, cpu)
        wall = outcome.wall_ms
        if not cpu and status is Status.FAILED and code in GPU_RETRY:
            joblog.warning("W_CPU_FALLBACK [%s] %s on the GPU; running once more on the CPU", stage.id, code)
            rec.fallback = {"to": Device.CPU, "after": code}
            cpu = True
            status, code, outcome, result = self.child(stage, primary, cpu)
            wall += outcome.wall_ms
        # A GPU the person chose stays chosen: a step that finds no card fails with that code, never moves on.
        forced = self.record.device.get("requested") == Device.GPU and code == "E_GPU_UNAVAILABLE"
        if (
            stage.fallback
            and status in (Status.FAILED, Status.UNAVAILABLE)
            and code != "E_JOB_CANCELLED"
            and not forced
        ):
            joblog.warning(
                "[%s] %s ended with %s; running %s instead", stage.id, primary, code, stage.fallback
            )
            rec.fallback = {"to": stage.fallback, "after": str(code)}
            cpu = True
            status, code, outcome, result = self.child(stage, stage.fallback, cpu)
            wall += outcome.wall_ms
        rec.status, rec.error_code, rec.wall_ms = status, code, wall
        rec.device = Device.CPU if cpu else Device.GPU
        if outcome.exit_code and not outcome.stopped:
            rec.exit_code_hex = hex(outcome.exit_code)
        if result is not None:
            rec.peak_ram_mb, rec.peak_vram_mb = result.get("peak_ram_mb"), result.get("peak_vram_mb")

    def describe_input(self) -> None:
        """Copy the decoded recording's length and format into job.json, which no child may write."""
        media = _result(self.path / MEDIA_RESULT)
        if media is not None and media["status"] == Status.OK:
            source = media["source"]
            self.record.input.update(
                duration_ms=media["duration_ms"], codec=source["codec"], channels=source["channels"],
                sample_rate=source["sample_rate"],
            )  # fmt: skip

    def settle_device(self, gpu_steps: set[str]) -> None:
        """Say the CPU ran the job when every GPU-capable step that ran had to move to it, and why."""
        ran = [r for r in self.record.stages if r.id in gpu_steps]
        moved = [(r.id, r.fallback) for r in ran if r.device == Device.CPU and r.fallback]
        if self.record.device.get("resolved") != Device.GPU or not ran or len(moved) < len(ran):
            return
        faults = [f["after"] for _, f in moved if f["after"] in GPU_FAULTS]
        step, first = moved[0]
        reason = (
            GPU_LOST.format(code=faults[0]) if faults
            else STEP_MOVED.format(step=step, to=first["to"], code=first["after"])
        )  # fmt: skip
        self.record.device.update(resolved=Device.CPU.value, reason=reason)

    def walk(
        self, stages: Sequence[Stage], start: int, gpu_steps: set[str], only: Collection[str] | None = None
    ) -> None:
        record = self.record
        order = {s.id: k for k, s in enumerate(stages)}
        for number, stage in enumerate(stages[start:], start + 1):
            if only is not None and stage.id not in only:
                continue
            if self.cancel.exists():
                record.state = JobState.CANCELLED
                break
            if self.on_progress:
                self.on_progress(stage.id, {"step": number, "of": len(stages)})
            rec = store.StageRecord(id=stage.id, started_utc=store.utc_now())
            record.stage = stage.id
            record.stages.append(rec)
            record.stages.sort(key=lambda r: order[r.id])  # a step run again keeps its place in the plan
            store.write(self.roots.data, record)
            self.stage(stage, rec)
            if stage.result == MEDIA_RESULT:
                self.describe_input()
            if stage.gpu and gpu_steps <= {r.id for r in record.stages}:
                self.settle_device(gpu_steps)  # the later steps copy the device into the evidence and export
            rec.ended_utc = store.utc_now()
            store.write(self.roots.data, record)
            if rec.error_code == "E_JOB_CANCELLED":
                record.state = JobState.CANCELLED
                break
            if stage.required and rec.status in (Status.FAILED, Status.UNAVAILABLE):
                # A refused input or a missing part cannot pass on a retry, so the job names that cause.
                own = catalogue().get(str(rec.error_code))
                code = rec.error_code if own is not None and not own.retryable else "E_JOB_FAILED"
                record.state, record.error = JobState.FAILED, {"code": code, "stage": stage.id}
                return
        if record.state is JobState.CANCELLED:
            record.error = {"code": "E_JOB_CANCELLED"}
            return
        gaps = any(r.fallback or r.status not in FINISHED for r in record.stages)
        record.state = JobState.COMPLETED_WITH_GAPS if gaps else JobState.COMPLETED
        record.stage = None


def run_job(
    roots: Roots,
    job_id: str,
    stages: Sequence[Stage],
    *,
    audio_min: float,
    no_progress_s: float,
    argv_for: Callable[[str, str], list[str]] = child_argv,
    on_progress: Callable[[str, dict[str, Any]], None] | None = None,
    only: Collection[str] | None = None,
) -> store.JobRecord:
    """Run a job, or run it again from its first unfinished stage; the caller holds work.lock.

    With `only`, a finished job runs just those stages again and keeps every other stage as it was.
    """
    record = store.read(roots.data, job_id, lock_free=False)
    if record.app_version != __version__:
        raise InputError("E_RESUME_OTHER_VERSION", job_id)
    path = store.job_dir(roots.data, job_id)
    if only is not None:
        if record.state not in (JobState.COMPLETED, JobState.COMPLETED_WITH_GAPS):
            raise InputError("E_JOB_NOT_FINISHED", job_id)
        start, done = 0, {s.id for s in stages} - set(only)
    else:
        # A re-run in another mode keeps the steps before its first one as they were, gaps included.
        first = [s.id for s in stages].index(record.rerun_from) if record.rerun_from else 0
        found = store.resume_point(path, [s.result for s in stages[first:]], FINISHED)
        finished = record.state is JobState.COMPLETED_WITH_GAPS and found is None
        if record.state is JobState.COMPLETED or finished:
            return record
        # Every result is there but the job stopped before it was marked finished: only its state is set.
        start = len(stages) if found is None else first + found
        done = {s.id for s in stages[:start]}
    record.stages = [r for r in record.stages if r.id in done]
    record.state, record.error = JobState.RUNNING, None
    target = handler(logging.FileHandler(path / "logs" / "job.log", encoding="utf-8"))
    joblog.addHandler(target)
    joblog.setLevel(str(record.settings.get("general", {}).get("log_level", "info")).upper())
    job = _Run(roots, record, argv_for, audio_min, no_progress_s, on_progress)
    gpu_steps = {s.id for s in stages if s.gpu}
    try:
        with KeepAwake():
            job.walk(stages, start, gpu_steps, only)
    except KeyboardInterrupt as exc:
        record.state, record.error = JobState.INTERRUPTED, {"code": "E_INTERRUPTED"}
        raise CancelledError("E_INTERRUPTED", f"job {job_id}") from exc  # the job to run again
    finally:
        job.settle_device(gpu_steps)
        store.write(roots.data, record)
        # A cancel that came after the last step, or was obeyed, must not stop the next run.
        job.cancel.unlink(missing_ok=True)
        joblog.removeHandler(target)
        target.close()
        for folder in (job_tmp(roots.app, job_id), mfa_root(roots.app, job_id)):
            shutil.rmtree(folder, ignore_errors=True)
    return record


def housekeeping(roots: Roots, retention_days: int) -> None:
    """What the new work.lock holder does first: mark dead jobs, remove old temp files and expired jobs."""
    store.mark_interrupted(roots.data)
    store.sweep(roots.data, roots.app)
    for job_id in store.expired(roots.data, retention_days) + store.selftest_jobs(roots.data):
        store.delete(roots.data, roots.app, job_id)
