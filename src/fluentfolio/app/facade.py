"""The functions the command line and the web app both call."""

from __future__ import annotations

import math
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from itertools import islice
from pathlib import Path
from typing import Any, Literal, NamedTuple, get_args

import psutil

from fluentfolio import __version__
from fluentfolio.alignment import mfa
from fluentfolio.analyzers import registry
from fluentfolio.analyzers.measures import collect
from fluentfolio.analyzers.opensmile import consented
from fluentfolio.asr.base import choices
from fluentfolio.core import paths
from fluentfolio.core.enums import Device, JobState, Layout, Mode
from fluentfolio.core.errors import (
    AppError,
    CancelledError,
    InputError,
    ResourceError,
    SetupError,
    StageError,
)
from fluentfolio.core.errors import catalogue as codes
from fluentfolio.core.files import atomic_write, copy_verified, read_json, write_json
from fluentfolio.core.logs import close_app_log
from fluentfolio.doctor import checks, estimate, selftest
from fluentfolio.doctor import report as diagnostic
from fluentfolio.evidence import build as evidence
from fluentfolio.evidence.models import SCHEMA_VERSION
from fluentfolio.export import answer_check, audio_check, bundle, for_ai
from fluentfolio.ingest import secondary, upload
from fluentfolio.jobs import loop, runner, store, stub
from fluentfolio.jobs.locks import work_lock
from fluentfolio.models import catalogue
from fluentfolio.settings import store as settings_store
from fluentfolio.settings.model import Asr, Limits, Review, Settings
from fluentfolio.setup import components as installer
from fluentfolio.setup import device as gpu
from fluentfolio.setup import token as hf_token
from fluentfolio.setup import updates
from fluentfolio.speakers import ownership, tracks
from fluentfolio.structure import build as structure
from fluentfolio.structure import overrides

LOCK_FILES = ("work.lock", "app.lock")
ANSWER_TYPES = (".txt", ".md")  # an AI chat's answer, saved as text
DEFAULT_ASR = Asr().model
LAYOUTS = tuple(m.value for m in Layout)
SWAP_STYLES = ("once", "per-question")  # once or per Part (the default), or on each question
REVIEW_POLICIES = get_args(Review.model_fields["policy"].annotation)
Consent = Literal["ui", "cli-flag", "cli-tty"]  # where the other person's agreement was given
# How a job's runner process starts, from its command and environment: by default it outlives this process.
Starter = Callable[[list[str], dict[str, str]], object]
PROMPT_FILES = {"grading": "grader", "pronunciation": "pronunciation"}
# The speaker separation child for each Settings choice, and the child that takes over when it fails.
BACKENDS = {
    "auto": ("speakers-pyannote", "speakers-sherpa"),
    "pyannote": ("speakers-pyannote", ""),
    "sherpa": ("speakers-sherpa", ""),
}
DIARIZERS = {"speakers-pyannote": "pyannote-community-1", "speakers-sherpa": "sherpa-onnx-diarizer"}
PROBE_S = 120.0  # headers take a second; a file without its length in them is read packet by packet
DETACH_WAIT_S = 15.0  # how long --detach waits for its runner to take the job
POLL_S = 0.2
AUDIO_STEPS = frozenset({"ai_audio", "evidence", "export"})  # the steps the optional audio file changes
SWITCHABLE_ANALYZERS = registry.switchable()  # each needs an optional tool


def _speakers(mode: str, layout: str, backend: str) -> loop.Stage:
    """The speaker step: the second-voice check in solo, the tracks, or a speaker separation backend."""
    result = "stages/speakers.json"
    if mode == Mode.SOLO:
        return loop.Stage("speakers", result, required=False, child="speakers-sherpa")
    if layout != Layout.MIXED:
        return loop.Stage("speakers", result, required=False, inline=tracks.run)
    child, fallback = BACKENDS[backend]
    return loop.Stage("speakers", result, required=False, gpu=child == "speakers-pyannote", child=child,
                      fallback=fallback)  # fmt: skip


def plan(
    mode: str, layout: str = Layout.MIXED, backend: str = "auto", asr: str = DEFAULT_ASR
) -> list[loop.Stage]:
    """The steps of an analysis, in run order; with FF_TEST_STUB=1 the model steps are test stand-ins."""
    found = [
        loop.Stage("media", "stages/media.json"),
        loop.Stage("vad", "stages/vad.json"),
        loop.Stage("language", "stages/language.json", required=False),
        loop.Stage("asr", "stages/asr.json", gpu=catalogue.get(asr).device == "cpu+gpu"),
        _speakers(mode, layout, backend),
        loop.Stage("words", "stages/words.json", inline=ownership.run),
        loop.Stage("structure", "stages/structure.json", inline=structure.run),
        loop.Stage("candidate_audio", "stages/candidate_audio.json", required=False),
        loop.Stage("ai_audio", "stages/ai_audio.json", required=False),
        loop.Stage("phones", "stages/phones.json", required=False),
        loop.Stage(
            "alignment", "stages/alignment.json", required=False, prepare=mfa.prepare, finish=mfa.finish
        ),
        loop.Stage("praat", "analyzers/praat.json", required=False),
        loop.Stage("swiftf0", "analyzers/swiftf0.json", required=False),
        loop.Stage("opensmile", "analyzers/opensmile.json", required=False),
        loop.Stage("nlp", "analyzers/nlp.json", required=False),
        loop.Stage("measures", "analyzers/measures.json", inline=collect.run),
        loop.Stage("evidence", "stages/evidence.json", inline=evidence.run),
        loop.Stage("export", "stages/export.json", inline=bundle.run),
    ]
    return stub.plan(found) if stub.on() else found


def _job_plan(record: store.JobRecord) -> list[loop.Stage]:
    """A job's steps from the choices it was started with; a section its settings lack is the default."""
    backend = record.settings.get("speakers", {}).get("backend", "auto")
    return plan(str(record.mode), str(record.layout), backend, str(record.asr))


def steps(record: store.JobRecord) -> int:
    """How many steps the job's analysis has."""
    return len(_job_plan(record))


def job_steps(record: store.JobRecord) -> list[str]:
    """The job's step IDs in run order; a job of a speech model this version lacks lists the steps it ran."""
    if record.mode is None or record.asr not in choices():
        return [s.id for s in record.stages]
    return [s.id for s in _job_plan(record)]


@dataclass(frozen=True)
class Analysis:
    record: store.JobRecord
    job: Path
    warnings: list[tuple[str, str]]  # settings problems found on the way: (code, detail)


def check_layout(mode: str, layout: str, second: Path | None, offset_ms: int | None, limits: Limits) -> None:
    """Refuse a layout that does not fit the mode or the files given, before any job exists."""
    if layout not in LAYOUTS:
        raise InputError("E_INPUT_LAYOUT", f"unknown layout {layout!r}")
    if mode == Mode.SOLO and layout != Layout.MIXED:
        raise InputError("E_INPUT_LAYOUT", "a solo recording has one person, so its layout is mixed")
    if (layout == Layout.TWO_FILES) != (second is not None):
        raise InputError("E_INPUT_LAYOUT", "two-files needs a second file, and only two-files takes one")
    if offset_ms is not None and (layout != Layout.TWO_FILES or not 0 <= offset_ms <= limits.max_min * 60000):
        raise InputError(
            "E_INPUT_LAYOUT", "an offset belongs to two-files and is at most the longest recording"
        )


def _roots() -> loop.Roots:
    return loop.Roots(paths.app_root(), paths.data_root(), paths.models_dir())


def probe(files: list[Path]) -> list[dict[str, Any]]:
    """Each file's format, tracks and length, read in a child: the decoder never loads in this process."""
    roots = _roots()
    found: dict[str, Any] = {}
    files_given = [str(f.absolute()) for f in files]
    argv = [sys.executable, "-I", "-X", "utf8", "-m", "fluentfolio.media.probe", *files_given]
    done = runner.run(
        argv, env=runner.child_env(roots.app, roots.data, roots.models, None, cpu=True), timeout_s=PROBE_S,
        no_progress_s=PROBE_S, on_progress=found.update,
    )  # fmt: skip
    if isinstance(found.get("code"), str) and found["code"] in codes():
        raise InputError(found["code"], str(found.get("detail") or ""))
    if done.exit_code or done.stopped or "probe" not in found:
        raise InputError("E_INPUT_DAMAGED")
    return list(found["probe"])


def estimate_ms(
    stages: list[loop.Stage], record_settings: dict[str, Any], mode: str, asr: str, audio_ms: int, device: str
) -> int:
    """The run time the measured step times give for this plan, length and device."""
    return estimate.estimate_ms(
        stages, mode=mode, asr=asr, settings=record_settings,
        opensmile_consented=consented(paths.data_root()), audio_ms=audio_ms, device=device,
    )  # fmt: skip


def job_estimate_ms(record: store.JobRecord) -> int | None:
    """A job's run time estimate, or None before its length is known or for a plan this version lacks."""
    audio_ms = record.input.get("duration_ms")
    if audio_ms is None or record.mode is None or record.asr not in choices():
        return None
    where = str(record.device.get("resolved") or Device.CPU)
    stages = _job_plan(record)
    if record.rerun_from:  # only the steps this job runs itself
        stages = stages[[s.id for s in stages].index(record.rerun_from) :]
    return estimate_ms(stages, record.settings, str(record.mode), str(record.asr), audio_ms, where)


def _free_ram_mb() -> int:
    """The memory free for new programs now, in MB."""
    return int(psutil.virtual_memory().available / 2**20)


def _ram_short(
    stages: list[loop.Stage], record_settings: dict[str, Any], mode: str, asr: str, audio_ms: int, device: str
) -> tuple[str, int, int] | None:
    """The step needing the most free memory, if it needs more than is free: (step, needed, free) in MB."""
    needs = estimate.ram_needs_mb(
        stages, mode=mode, asr=asr, settings=record_settings,
        opensmile_consented=consented(paths.data_root()), audio_ms=audio_ms, device=device,
    )  # fmt: skip
    if not needs:
        return None
    step = max(needs, key=needs.__getitem__)
    free = _free_ram_mb()
    return (step, needs[step], free) if needs[step] > free else None


def _in_gb(short: tuple[str, int, int]) -> tuple[str, str, str]:
    """(step, needed GB, free GB) to 0.1 GB: the need rounded up and the free memory down, so they never read
    as equal."""
    step, need, free = short
    return step, f"{math.ceil(need / 102.4) / 10:.1f}", f"{math.floor(free / 102.4) / 10:.1f}"


def _ram_warning(short: tuple[str, int, int]) -> tuple[str, str]:
    step, need, free = _in_gb(short)
    return "W_FREE_RAM", f"the {step} step needs about {need} GB; {free} GB is free"


def _warn_before(
    warnings: list[tuple[str, str]], on_progress: Callable[[str, dict[str, Any]], None] | None
) -> list[tuple[str, str]]:
    """A plan's memory warning, shown before the run starts and kept for the job's result."""
    memory = [w for w in warnings if w[0] == "W_FREE_RAM"]
    for code, detail in memory:
        if on_progress:
            on_progress("", {"warning": code, "detail": detail})
    return memory


def estimates(duration_ms: int) -> tuple[dict[str, int], str, tuple[str, str, str] | None]:
    """The run time of each mode for a recording this long with the settings in force, where it runs, and the
    step of any mode that needs more memory than is free now, as (step, needed GB, free GB)."""
    settings = settings_store.load(paths.data_root() / "settings.toml").settings
    snapshot = settings_store.snapshot(settings)
    found, where, short = {}, Device.CPU.value, None
    for mode in Mode:
        stages = plan(mode, Layout.MIXED, settings.speakers.backend, settings.asr.model)
        where = gpu.resolve(settings.device.choice, any(s.gpu for s in stages))["resolved"]
        found[mode.value] = estimate_ms(stages, snapshot, mode, settings.asr.model, duration_ms, where)
        mode_short = _ram_short(stages, snapshot, mode, settings.asr.model, duration_ms, where)
        if mode_short and (short is None or mode_short[1] > short[1]):
            short = mode_short
    return found, where, _in_gb(short) if short else None


@dataclass(frozen=True)
class Planned:
    """What an analysis would do, worked out before any job exists (`analyze --dry-run`)."""

    probe: list[dict[str, Any]]
    duration_ms: int  # the session: with two files, from the first file's start to the later end
    stages: list[loop.Stage]
    device: dict[str, str]
    snapshot: dict[str, Any]
    estimate_ms: int
    will_run: list[str]
    missing: list[str]
    needs: list[str]  # decisions only a person makes: partner_consent, confirm_long
    warnings: list[tuple[str, str]]  # (code, detail)


def _plan_run(
    files: list[Path], mode: str, settings: Settings, *, layout: str, offset_ms: int | None,
    device: Device | None, asr: str | None, partner_consent: Consent | None, confirm_long: bool,
    light_run: bool,
) -> Planned:  # fmt: skip
    """The plan, the device, the estimate and the decisions still needed, from the files' probe."""
    second = files[1] if len(files) > 1 else None
    check_layout(mode, layout, second, offset_ms, settings.limits)
    model = asr or settings.asr.model
    if model not in choices():
        raise SetupError(
            "E_COMPONENT_UNAVAILABLE", "a speech model Fluentfolio does not know"
        )  # never the name typed
    for f in files:
        upload.check_size(f.stat().st_size, settings.limits)
    found = probe(files)
    lengths = [p["duration_ms"] for p in found]
    duration_ms = max(lengths[0], (offset_ms or 0) + lengths[1]) if second else lengths[0]
    limits = settings.limits
    if light_run and duration_ms <= limits.long_min * 60000:
        raise InputError("E_INPUT_OPTION", f"a light run is for recordings over {limits.long_min} min")
    needs = ["partner_consent"] if mode != Mode.SOLO and partner_consent is None else []
    try:
        warnings = [
            (code, "")
            for code in upload.check_duration(duration_ms / 1000, limits, confirmed=confirm_long or light_run)
        ]
    except InputError as exc:
        if exc.code != "E_CONFIRM_LONG":
            raise
        needs.append("confirm_long")
        warnings = []
    snapshot = settings_store.snapshot(settings, asr=asr, device=device, light_run=light_run)
    stages = plan(mode, layout, settings.speakers.backend, model)
    where = gpu.resolve(device or settings.device.choice, any(s.gpu for s in stages))
    took = estimate_ms(stages, snapshot, mode, model, duration_ms, where["resolved"])
    if short := _ram_short(stages, snapshot, mode, model, duration_ms, where["resolved"]):
        warnings.append(_ram_warning(short))
    off = estimate.off_steps(snapshot, consented(paths.data_root()))
    will_run = [s.id for s in stages if s.id not in off]
    # The parts the run would use: the chosen speech model, the voice detector, the speaker backend and
    # fallback of this plan, and every switched-on part of a step that runs.
    diarizers = {DIARIZERS[s.child] for s in stages if s.child} | {
        DIARIZERS[s.fallback] for s in stages if s.fallback
    }
    rows = checks.in_use(
        checks.components(Settings.model_validate(_settings_part(snapshot)), _roots()), model
    )
    parts = {c.id: c for c in catalogue.components().values()}
    missing = [
        r.id for r in rows if r.status != "ready" and any(s in will_run for s in parts[r.id].stages)
        and (r.kind != "diarizer" or r.id in diarizers)
    ]  # fmt: skip
    return Planned(found, duration_ms, stages, where, snapshot, took, will_run, missing, needs, warnings)


def _settings_part(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The Settings fields of a job's settings snapshot, which also records the light run."""
    return {k: v for k, v in snapshot.items() if k != "light_run"}


def dry_run(
    source: Path, mode: str, *, second: Path | None = None, layout: str = Layout.MIXED,
    offset_ms: int | None = None, device: Device | None = None, asr: str | None = None,
    partner_consent: Consent | None = None, confirm_long: bool = False, light_run: bool = False,
) -> Planned:  # fmt: skip
    """`analyze --dry-run`: the probe, the estimate, the steps, what is missing and what a person decides."""
    settings = settings_store.load(paths.data_root() / "settings.toml").settings
    return _plan_run(
        [source, *([second] if second else [])], mode, settings, layout=layout, offset_ms=offset_ms,
        device=device, asr=asr, partner_consent=partner_consent, confirm_long=confirm_long,
        light_run=light_run,
    )  # fmt: skip


def _run(
    roots: loop.Roots, job_id: str, stages: list[loop.Stage], limits: dict[str, Any],
    on_progress: Callable[[str, dict[str, Any]], None] | None, only: frozenset[str] | None = None,
) -> store.JobRecord:  # fmt: skip
    """Run a job under the caller's work.lock; Ctrl+C leaves it interrupted, to be run again later."""
    try:
        return loop.run_job(
            roots, job_id, stages, audio_min=limits["max_min"], no_progress_s=limits["no_progress_min"] * 60,
            on_progress=on_progress, only=only,
        )  # fmt: skip
    except CancelledError:
        return store.read(roots.data, job_id, lock_free=False)


def analyze(
    source: Path, mode: str, *, device: Device | None = None, asr: str | None = None,
    layout: str = Layout.MIXED, second: Path | None = None, offset_ms: int | None = None,
    swap_style: str | None = None,
    questions: Path | None = None, transcript: Path | None = None, partner_consent: Consent | None = None,
    confirm_long: bool = False, light_run: bool = False, review: str | None = None, detach: bool = False,
    start: Starter | None = None, on_progress: Callable[[str, dict[str, Any]], None] | None = None,
    names: tuple[str | None, str | None] = (None, None),
) -> Analysis:  # fmt: skip
    """Take a recording in and run every step while holding work.lock; the job keeps its own settings copy.

    The question list and the learner's own transcript are optional; the transcript is kept without names.
    A recording of two people needs the other person's agreement, which the job keeps without any name.
    A recording over the confirmation length needs `confirm_long`, or `light_run` over the light-run length.
    `review` is this run's review form policy, kept in its settings copy; by default the Settings one.
    With `detach` the job runs in a runner of its own, started by `start` if given, and this returns once
    that runner has taken it. `names` are the files' own names, for an upload saved under another name.
    """
    if swap_style is not None and (mode != Mode.ROLE_SWAP or swap_style not in SWAP_STYLES):
        raise InputError("E_INPUT_OPTION", f"swap style {swap_style!r} with mode {mode}")
    if review is not None and review not in REVIEW_POLICIES:
        raise InputError("E_INPUT_OPTION", f"review form {review!r}")
    if mode != Mode.SOLO and partner_consent is None:
        raise AppError("E_CONSENT_REQUIRED")
    roots = _roots()
    roots.data.mkdir(parents=True, exist_ok=True)
    loaded = settings_store.load(roots.data / "settings.toml")
    settings = loaded.settings
    planned = _plan_run(
        [source, *([second] if second else [])], mode, settings, layout=layout, offset_ms=offset_ms,
        device=device, asr=asr, partner_consent=partner_consent, confirm_long=confirm_long,
        light_run=light_run,
    )  # fmt: skip
    if "confirm_long" in planned.needs:
        minutes = max(1, round(planned.estimate_ms / 60000))  # as the dry run says it: never "about 0 min"
        raise InputError(
            "E_CONFIRM_LONG", f"{planned.duration_ms / 60000:.1f} min, about {minutes} min to analyse"
        )
    if review is not None:
        planned.snapshot["review"]["policy"] = review
    memory = _warn_before(planned.warnings, on_progress)
    with work_lock(roots.data):
        loop.housekeeping(roots, settings.general.retention_days)
        now = store.utc_now()
        record = store.JobRecord(
            job_id=store.new_job_id(), created_utc=now, updated_utc=now, app_version=__version__, mode=mode,
            layout=layout, offset_ms=offset_ms, asr=asr or settings.asr.model,
            swap_style=(swap_style or SWAP_STYLES[0]) if mode == Mode.ROLE_SWAP else None,
            device=planned.device, light_run=light_run, settings=planned.snapshot,
            consent={"utc": now, "source": partner_consent} if mode != Mode.SOLO else {},
            state=JobState.QUEUED if detach else JobState.RUNNING,
        )  # fmt: skip
        job = store.create(roots.data, record)
        try:
            record.input = upload.store_input(job, source, settings.limits, name=names[0])
            if second is not None:
                record.input["second"] = upload.store_input(
                    job, second, settings.limits, stem="source_2", name=names[1]
                )
            if questions is not None:
                secondary.store_questions(job, questions)
            if transcript is not None:
                record.input["user_transcript"] = {"speakers": secondary.store_transcript(job, transcript)}
        except (AppError, OSError):
            store.delete(roots.data, roots.app, record.job_id)  # a refused file leaves no job behind
            raise
        # The probed length, for the estimate while the job waits; the media step writes the decoded one.
        record.input["duration_ms"] = planned.duration_ms
        store.write(roots.data, record)
        if not detach:
            done = _run(roots, record.job_id, planned.stages, settings.limits.model_dump(), on_progress)
            refused = done.error if done.state is JobState.FAILED else None
            if refused and codes()[refused["code"]].category == "input":
                # A recording refused during the analysis leaves no job behind either, nor its decoded copy.
                step = next(s for s in planned.stages if s.id == refused["stage"])
                result = job / step.result
                detail = str(read_json(result).get("detail") or "") if result.is_file() else ""
                store.delete(roots.data, roots.app, record.job_id)
                raise InputError(refused["code"], detail)
    if detach:
        return Analysis(start_detached(record.job_id, start), job, loaded.warnings + memory)
    return Analysis(done, job, loaded.warnings + memory)


def start_detached(job_id: str, start: Starter | None = None) -> store.JobRecord:
    """Start a runner with no window, by default one that outlives this process, and return once it has
    taken the job.

    A job no runner took within the wait is marked interrupted, so `resume` runs it in the foreground.
    """
    data = paths.data_root()
    argv = [sys.executable, "-I", "-X", "utf8", "-m", "fluentfolio", "_run", job_id]
    if start is None:
        runner.detach(argv, env=runner.own_env())
    else:
        start(argv, runner.own_env())
    deadline = time.monotonic() + DETACH_WAIT_S
    while time.monotonic() < deadline:
        # job.json as written, read without trying work.lock, which the starting runner is about to take
        if (record := store.read(data, job_id, lock_free=False)).state is not JobState.QUEUED:
            return record
        time.sleep(POLL_S)
    try:
        with work_lock(data):
            record = store.read(data, job_id, lock_free=False)
            if record.state is not JobState.QUEUED:
                return record
            record.state, record.error = JobState.INTERRUPTED, {"code": "E_RUNNER_NOT_STARTED"}
            store.write(data, record)
    except AppError as exc:
        if exc.code != "E_BUSY":
            raise
        return status(job_id).record  # the runner took work.lock just now
    raise StageError("E_RUNNER_NOT_STARTED", job_id)


def _finished(job_id: str, candidate: str | None) -> tuple[Path, dict[str, Any], str | None]:
    """A finished job's folder and evidence, and the person asked for: the only one, or the one named."""
    job = store.job_dir(paths.data_root(), job_id)
    found = job / "evidence.json"
    if not found.is_file():
        raise InputError("E_JOB_NOT_FINISHED")
    data = read_json(found)
    ids = [c["id"] for c in data["candidates"]]
    if candidate is None and len(ids) > 1:
        raise InputError("E_INPUT_OPTION", f"say which person: {' or '.join(ids)}")
    if candidate is not None and candidate not in ids:
        raise InputError("E_INPUT_OPTION", f"this job has no {candidate}")
    return job, data, candidate or (ids[0] if ids else None)


def hidden_words(job_id: str, candidate: str | None = None) -> str | None:
    """The words of a person's audio-check span, shown only when the learner asks; None with no span."""
    job, _, chosen = _finished(job_id, candidate)
    found = job / audio_check.FILE
    return (read_json(found) if found.is_file() else {}).get(chosen, {}).get("words")


def check_answer(job_id: str, answer: Path, candidate: str | None = None) -> answer_check.Report:
    """Check an AI chat's answer to one person's text: its format and band rules, and the audio check."""
    job, data, chosen = _finished(job_id, candidate)
    text = secondary.read_text(answer, ANSWER_TYPES)
    # Only the chosen person's evidence, so an answer about the other person fails the format rule.
    mine = {**data, "candidates": [c for c in data["candidates"] if c["id"] == chosen]}
    return answer_check.check_answer(
        text, mine, hidden_words=hidden_words(job_id, chosen), removed_names=secondary.removed_names(job)
    )


def status(job_id: str) -> Analysis:
    """A job as any reader sees it: running only while its runner holds work.lock."""
    data = paths.data_root()
    job = store.job_dir(data, job_id)
    return Analysis(store.read(data, job_id, lock_free=work_lock(data).is_free()), job, [])


def wait(job_id: str, timeout_s: float, poll_s: float = 1.0) -> Analysis:
    """The job's status once it stops running, or as it is at the timeout."""
    deadline = time.monotonic() + timeout_s
    while (done := status(job_id)).record.state is JobState.RUNNING and time.monotonic() < deadline:
        time.sleep(max(0.0, min(poll_s, deadline - time.monotonic())))
    return done


def jobs(limit: int) -> list[store.JobRecord]:
    """The newest jobs first; a running self-test's job is not the learner's, so it is left out."""
    data = paths.data_root()
    ids = store.list_ids(data)[::-1]
    free = not ids or work_lock(data).is_free()
    found = (store.read(data, job_id, lock_free=free) for job_id in ids)
    return list(islice((r for r in found if not r.selftest), limit))


def _check_resumable(record: store.JobRecord) -> None:
    """Refuse a job that a delete left half done, or one that another version made, before it runs again."""
    if (record.error or {}).get("code") == "E_DELETE_INCOMPLETE":  # its recording and results are gone
        raise ResourceError("E_DELETE_INCOMPLETE", record.job_id)
    if record.app_version != __version__:  # before its plan, which an unknown record cannot give
        raise InputError("E_RESUME_OTHER_VERSION", record.job_id)


def resume(job_id: str, on_progress: Callable[[str, dict[str, Any]], None] | None = None) -> Analysis:
    """Run a job again from its first unfinished step with its own settings; a completed job is left as is."""
    roots = _roots()
    job = store.job_dir(roots.data, job_id)
    with work_lock(roots.data):
        store.mark_interrupted(roots.data)
        record = store.read(roots.data, job_id, lock_free=False)
        if record.state is JobState.COMPLETED:
            return Analysis(record, job, [])
        _check_resumable(record)
        limits = Limits.model_validate(record.settings.get("limits", {})).model_dump()
        return Analysis(_run(roots, job_id, _job_plan(record), limits, on_progress), job, [])


def resume_detached(job_id: str, start: Starter | None = None) -> Analysis:
    """`resume --detach`: queue the job, then run it in a runner of its own; a completed job is left as is."""
    roots = _roots()
    job = store.job_dir(roots.data, job_id)
    with work_lock(roots.data):
        store.mark_interrupted(roots.data)
        record = store.read(roots.data, job_id, lock_free=False)
        if record.state is JobState.COMPLETED:
            return Analysis(record, job, [])
        _check_resumable(record)
        record.state = JobState.QUEUED
        store.write(roots.data, record)
    return Analysis(start_detached(job_id, start), job, [])


@dataclass(frozen=True)
class Rerun:
    """What running a finished job again in another mode does, worked out before the new job exists."""

    source: store.JobRecord
    stages: list[loop.Stage]  # the new job's plan
    first: str  # its first step; the steps before it are copied
    device: dict[str, str]
    estimate_ms: int | None
    needs_consent: bool  # the new mode has two people and the job has no agreement on record


def _rerun_first(old: str, new: str) -> str:
    # Solo's speaker step only checks for a second voice, and solo gives every word to one person,
    # so a change to or from solo separates the speakers again; else the words keep their speakers.
    return "speakers" if Mode.SOLO in (old, new) else "structure"


def rerun_offer(job_id: str, mode: str) -> Rerun:
    """Check that a finished job can run again as `mode`, and give its first step and time estimate."""
    data = paths.data_root()
    record = store.read(data, job_id, lock_free=work_lock(data).is_free())
    if mode not in tuple(Mode) or mode == record.mode:
        raise InputError("E_INPUT_OPTION", f"mode {mode!r} for a {record.mode} job")
    if record.state not in (JobState.COMPLETED, JobState.COMPLETED_WITH_GAPS):
        raise InputError("E_JOB_NOT_FINISHED", job_id)
    if record.app_version != __version__:  # its results may not fit this version's steps
        raise InputError("E_RESUME_OTHER_VERSION", job_id)
    if mode == Mode.SOLO and record.layout != Layout.MIXED:
        raise InputError("E_INPUT_LAYOUT", "a solo recording has one person, so its layout is mixed")
    backend = record.settings.get("speakers", {}).get("backend", "auto")
    stages = plan(mode, str(record.layout), backend, str(record.asr))
    first = _rerun_first(str(record.mode), mode)
    runs = stages[[s.id for s in stages].index(first) :]
    where = gpu.resolve(Device(record.device.get("requested") or Device.AUTO), any(s.gpu for s in runs))
    audio_ms = record.input.get("duration_ms")
    took = None
    if audio_ms is not None:
        took = estimate_ms(runs, record.settings, mode, str(record.asr), audio_ms, where["resolved"])
    needs = mode != Mode.SOLO and not record.consent
    return Rerun(record, stages, first, where, took, needs)


def rerun(
    job_id: str, mode: str, *, partner_consent: Consent | None = None, detach: bool = False,
    start: Starter | None = None, on_progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> Analysis:  # fmt: skip
    """Run a finished job again as `mode`, in a new job; the recording and its transcription are copied.

    A two-person mode needs the other person's agreement: the job's own, or `partner_consent`.
    """
    roots = _roots()
    settings = settings_store.load(roots.data / "settings.toml").settings
    with work_lock(roots.data):
        loop.housekeeping(roots, settings.general.retention_days)
        offer = rerun_offer(job_id, mode)
        old = offer.source
        if offer.needs_consent and partner_consent is None:
            raise AppError("E_CONSENT_REQUIRED")
        now = store.utc_now()
        if mode == Mode.SOLO:
            consent = {}
        else:
            consent = old.consent or {"utc": now, "source": partner_consent}
        kept = {s.id for s in offer.stages[: [s.id for s in offer.stages].index(offer.first)]}
        record = store.JobRecord(
            job_id=store.new_job_id(), created_utc=now, updated_utc=now, app_version=__version__, mode=mode,
            layout=old.layout, offset_ms=old.offset_ms, asr=old.asr,
            swap_style=SWAP_STYLES[0] if mode == Mode.ROLE_SWAP else None, device=offer.device,
            light_run=old.light_run, settings=old.settings, input=old.input, consent=consent,
            state=JobState.QUEUED if detach else JobState.RUNNING,
            stages=[r for r in old.stages if r.id in kept], rerun_of=old.job_id, rerun_from=offer.first,
        )  # fmt: skip
        job = store.create(roots.data, record)
        source = store.job_dir(roots.data, job_id)
        try:
            copy_verified(source / "input", job / "input")
            for stage in offer.stages:
                if stage.id in kept and (source / stage.result).is_file():
                    shutil.copy2(source / stage.result, job / stage.result)
        except BaseException:  # a failed or interrupted copy leaves no half job; the old one is untouched
            store.delete(roots.data, roots.app, record.job_id)
            raise
        if not detach:
            limits = Limits.model_validate(old.settings.get("limits", {})).model_dump()
            done = _run(roots, record.job_id, offer.stages, limits, on_progress)
    if detach:
        return Analysis(start_detached(record.job_id, start), job, [])
    return Analysis(done, job, [])


def review(job_id: str, edits: dict[str, Any], start: Starter | None = None) -> Analysis:
    """Save the review form's corrections and analyse the job again from its structure step on, in a runner of
    its own; refused while a job runs, as it takes work.lock."""
    try:
        found = overrides.Overrides.model_validate(edits)
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        raise InputError("E_REVIEW_INVALID") from exc
    roots = _roots()
    job = store.job_dir(roots.data, job_id)
    with work_lock(roots.data):
        record = store.read(roots.data, job_id, lock_free=False)
        if record.state not in (JobState.COMPLETED, JobState.COMPLETED_WITH_GAPS):
            raise InputError("E_JOB_NOT_FINISHED", job_id)
        if record.app_version != __version__:
            raise InputError("E_RESUME_OTHER_VERSION", job_id)
        times = list(found.swap_times_ms or [])
        for edit in found.parts.values():
            times += [*(edit.part2 or []), *([] if edit.part3_start_ms is None else [edit.part3_start_ms])]
        if any(not 0 <= t <= record.input.get("duration_ms", 0) for t in times):
            raise InputError("E_REVIEW_INVALID", "a time outside the recording")
        write_json(job / overrides.FILE, found.model_dump(mode="json", exclude_defaults=True))
        # The structure step's result goes, so the job runs again from it; every later step follows it.
        (job / "stages" / "structure.json").unlink(missing_ok=True)
        record.state, record.error = JobState.QUEUED, None
        store.write(roots.data, record)
    return Analysis(start_detached(job_id, start), job, [])


def settings_show() -> tuple[settings_store.Loaded, Path]:
    """The settings in force (settings.toml over the defaults), the file's problems, and its path."""
    path = paths.data_root() / "settings.toml"
    return settings_store.load(path), path


def settings_set(key: str, value: str) -> tuple[Settings, Path]:
    """Set one `section.key` of settings.toml as the Settings page does; refused while a job runs.

    Only known components, switchable analyzers and installed-or-listed speech models are taken.
    """
    section, _, name = key.partition(".")
    known = {"components": set(catalogue.components()), "analyzers": set(SWITCHABLE_ANALYZERS)}
    if section in known and name not in known[section]:
        raise InputError("E_SETTINGS_INVALID", f"no such {section} setting")  # never the name typed
    path = paths.data_root() / "settings.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with work_lock(path.parent):  # read and written under one lock, so two changes never undo each other
        loaded = settings_store.load(path)
        if loaded.read_only:
            raise InputError("E_SETTINGS_READ_ONLY")
        new = settings_store.changed(loaded.settings, key, value)
        if key == "asr.model" and new.asr.model not in choices():
            raise InputError("E_SETTINGS_INVALID", key)
        settings_store.save(path, new)
    return new, path


def settings_restore() -> Path:
    """Restore defaults: settings.toml with every value at its default; refused while a job runs."""
    path = paths.data_root() / "settings.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with work_lock(path.parent):
        if settings_store.load(path).read_only:
            raise InputError("E_SETTINGS_READ_ONLY")
        settings_store.save(path, Settings())
    return path


def delete_job(job_id: str) -> None:
    """Delete one job with its temporary and MFA folders; never while a job runs, as it takes work.lock."""
    roots = _roots()
    store.job_dir(roots.data, job_id)  # an unknown job is refused before the lock is taken
    with work_lock(roots.data):
        if left := store.delete(roots.data, roots.app, job_id):
            raise ResourceError("E_DELETE_INCOMPLETE", f"{len(left)} files or folders")


def delete_all() -> int:
    """Delete all my data: every job, both temporary folders, the logs, open.html and the licence consent.

    Models, the token and the settings stay. Returns how many jobs were deleted; refused while a job runs.
    A file another program holds stays, the rest still goes, and then E_DELETE_INCOMPLETE says so.
    """
    roots = _roots()
    with work_lock(roots.data):
        ids = store.list_ids(roots.data)
        left = [name for job_id in ids for name in store.delete(roots.data, roots.app, job_id)]
        close_app_log()  # an open log file cannot be deleted on Windows; the next line logged opens it again
        folders = (roots.data / "tmp", roots.app / "tmp", roots.data / "logs")
        left += store.remove((*folders, roots.data / "open.html", roots.data / installer.CONSENT_FILE))
    if left:
        raise ResourceError("E_DELETE_INCOMPLETE", f"{len(left)} files or folders")
    return len(ids)


def cancel(job_id: str) -> tuple[store.JobRecord, bool]:
    """Ask a running job to stop, which its own loop does within seconds; any other job is left as it is."""
    done = status(job_id)
    if done.record.state is not JobState.RUNNING:
        return done.record, False
    (done.job / store.CANCEL_FILE).write_text("", encoding="utf-8")
    return done.record, True


def _export_files(job: Path) -> list[Path]:
    """The files a learner takes away: each for_ai text, its audio file and step, and the ZIP."""
    found = job / "export" / "for_ai"
    texts = sorted(found.glob("for_ai*.txt")) + sorted(found.glob("audio_step*.txt"))
    return texts + sorted(found.glob("for_ai_audio*")) + [job / "export" / f"fluentfolio_{job.name}.zip"]


def export_files(job_id: str) -> list[Path]:
    """The export files a finished job has now, as they are: the web page offers each and writes nothing."""
    return [p for p in _export_files(store.job_dir(paths.data_root(), job_id)) if p.is_file()]


def _evidence(job: Path) -> dict[str, Any]:
    found = job / "evidence.json"
    if not found.is_file():
        raise InputError("E_JOB_NOT_FINISHED", job.name)
    return read_json(found)


def job_evidence(job_id: str) -> dict[str, Any]:
    """A finished job's evidence, as its last analysis left it."""
    return _evidence(store.job_dir(paths.data_root(), job_id))


def corrections(job_id: str) -> dict[str, Any]:
    """The review form's corrections saved for a job, as review() wrote them; empty when none were saved."""
    found = overrides.load(store.job_dir(paths.data_root(), job_id))
    return found.model_dump(mode="json", exclude_defaults=True)


def results(job_id: str) -> Results:
    """A finished job's evidence, speech segments, who spoke when if no person was found, and its words."""
    job = store.job_dir(paths.data_root(), job_id)
    data = _evidence(job)
    structure = read_json(job / "stages" / "structure.json")
    words = read_json(job / "stages" / "words.json")["words"]
    return Results(data, structure["speech"], [] if data["candidates"] else structure["transcript"], words)


class Results(NamedTuple):
    """A finished job as the Results page shows it."""

    evidence: dict[str, Any]
    speech: dict[str, list[list[int]]]  # each person's own speech segments
    runs: list[dict[str, Any]]  # who spoke when, when no person was found
    words: list[dict[str, Any]]  # the speech model's words with their times


@dataclass(frozen=True)
class ForAi:
    person: str  # candidate, person1 or person2
    label: str
    text: str | None  # None when the file is missing; making the export again brings it back
    step: str | None  # the message to send with the audio file, once the audio file is made
    audio: str | None  # the audio file's name


def for_ai_files(job_id: str) -> tuple[str, list[ForAi]]:
    """The export ID and each person's text for an AI chat, read as written: nothing is written here."""
    job = store.job_dir(paths.data_root(), job_id)
    data = _evidence(job)
    people = [(c["id"], c["label"]) for c in data["candidates"]] or [("candidate", "the candidate")]
    folder = job / "export" / "for_ai"
    made = job / "stages" / "ai_audio.json"
    audio = read_json(made)["candidates"] if made.is_file() else {}

    def text(name: str) -> str | None:
        return (folder / name).read_text(encoding="utf-8") if (folder / name).is_file() else None

    found = [
        ForAi(p, label, text(bundle.NAMES[p]), text(bundle.STEPS[p]), (audio.get(p) or {}).get("file"))
        for p, label in people
    ]
    return data["manifest"]["export_id"], found


def _add_audio(job_id: str) -> None:
    """Make the optional audio for a finished job: that step, then the evidence and export that name it."""
    roots = _roots()
    with work_lock(roots.data):
        record = store.read(roots.data, job_id, lock_free=False)
        if record.app_version != __version__:  # before the job's settings change
            raise InputError("E_RESUME_OTHER_VERSION", job_id)
        record.settings["export"]["audio"] = True  # the learner asked for it, for this job only
        store.write(roots.data, record)
        limits = Limits.model_validate(record.settings.get("limits", {})).model_dump()
        _run(roots, job_id, _job_plan(record), limits, None, only=AUDIO_STEPS)


def export(job_id: str, out: Path | None = None, audio: bool = False) -> tuple[Analysis, list[Path], bool]:
    """Write a finished job's export again if a part of it is missing; copy it into `out` when asked.

    With `audio`, a job made without the optional audio file gets it first; nothing else runs again.
    Returns the job, the files where the learner finds them, and whether anything was written again.
    """
    done = status(job_id)
    if done.record.state is JobState.RUNNING or not (done.job / "evidence.json").is_file():
        raise InputError("E_JOB_NOT_FINISHED")
    added = audio and not done.record.settings["export"]["audio"]
    if added:
        _add_audio(job_id)
        done = status(job_id)
    people = [c["id"] for c in read_json(done.job / "evidence.json")["candidates"]] or ["candidate"]
    wanted = [done.job / "export" / "for_ai" / bundle.NAMES[c] for c in people]
    changed = added or not all(p.is_file() for p in [*wanted, _export_files(done.job)[-1]])
    if changed and not added:
        bundle.run(done.job)  # a new export draws a new boundary, so only a missing part rewrites it
    files = _export_files(done.job)
    if out is None:
        return done, files, changed
    out = out.absolute()
    out.mkdir(parents=True, exist_ok=True)
    for path in files:
        atomic_write(out / path.name, path.read_bytes())
    return done, [out / p.name for p in files], changed


def export_text(job_id: str, candidate: str | None = None) -> str:
    """One person's text for an AI chat, as "Copy for AI" gives it."""
    export(job_id)
    job, _, chosen = _finished(job_id, candidate)
    return (job / "export" / "for_ai" / bundle.NAMES[chosen or "candidate"]).read_text(encoding="utf-8")


def doctor(report_to: Path | None = None) -> tuple[checks.Report, Path | None]:
    """Every check; with a path, also the diagnostic report, written there."""
    roots = _roots()
    settings = settings_store.load(roots.data / "settings.toml").settings
    found = checks.run(settings, roots)
    if report_to is None:
        return found, None
    text = diagnostic.build(found, checks.components(settings, roots), roots)
    target = report_to.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(target, text)
    return found, target


def serve(port: int, browser: bool) -> int:
    """Run the web app in this window until it stops, as a process of its own (cli never imports web)."""
    argv = [sys.executable, "-I", "-X", "utf8", "-m", "fluentfolio.web", "--port", str(port)]
    return runner.foreground([*argv, *([] if browser else ["--no-browser"])], env=runner.own_env())


def diagnostic_report() -> str:
    """The diagnostic report's text, for a page that shows it before it is copied."""
    roots = _roots()
    settings = settings_store.load(roots.data / "settings.toml").settings
    return diagnostic.build(checks.run(settings, roots), checks.components(settings, roots), roots)


def self_test(
    device: Device | None = None, on_progress: Callable[[str, dict[str, Any]], None] | None = None
) -> dict[str, Any]:
    """The bundled clip through every installed and switched-on part, checked; state.json keeps the outcome.

    The clip's job is thrown away when the check ends, and the job list never shows it.
    """
    roots = _roots()
    roots.data.mkdir(parents=True, exist_ok=True)
    settings = settings_store.load(roots.data / "settings.toml").settings
    stages = plan(Mode.INTERVIEW, Layout.MIXED, settings.speakers.backend, settings.asr.model)
    started = time.monotonic()
    with work_lock(roots.data):
        loop.housekeeping(roots, settings.general.retention_days)
        if paths.install_state(roots.app).get("backend") == "cu130" and device != Device.CPU:
            gpu.check_gpu(roots.app)  # Auto uses the GPU only after this test has passed
        now = store.utc_now()
        record = store.JobRecord(
            job_id=store.new_job_id(), created_utc=now, updated_utc=now, app_version=__version__,
            mode=Mode.INTERVIEW, layout=Layout.MIXED, asr=settings.asr.model, selftest=True,
            device=gpu.resolve(device or settings.device.choice, any(s.gpu for s in stages)),
            settings=settings_store.snapshot(settings, device=device),
        )  # fmt: skip
        job = store.create(roots.data, record)
        try:
            record.input = upload.store_input(job, selftest.clip(), settings.limits)
            store.write(roots.data, record)
            done = _run(roots, record.job_id, stages, settings.limits.model_dump(), on_progress)
            rows = checks.in_use(checks.components(settings, roots), settings.asr.model)
            parts = selftest.evaluate(job, done, rows)
        finally:
            store.delete(roots.data, roots.app, record.job_id)
        return selftest.save(roots.app, parts, done, round((time.monotonic() - started) * 1000))


def components() -> list[checks.ComponentRow]:
    """Every part with its install status, as `components list` and the Components page show them."""
    roots = _roots()
    return checks.components(settings_store.load(roots.data / "settings.toml").settings, roots)


def in_use() -> list[checks.ComponentRow]:
    """The parts an analysis with the current settings uses."""
    settings = settings_store.load(paths.data_root() / "settings.toml").settings
    return checks.in_use(checks.components(settings, _roots()), settings.asr.model)


@dataclass(frozen=True)
class InstallPlan:
    parts: list[catalogue.Component]  # in install order: the parts a part needs come before it
    download_bytes: int  # what the parts still lack, judged by file size


def install_plan(ids: list[str], every: bool = False) -> InstallPlan:
    """What `components install` fetches or makes: the parts named, or with `every` each part in use that
    is not ready; with the parts they need that are not ready either."""
    if every:
        ids = [r.id for r in in_use() if r.status in ("needs-download", "self-test-failed")]
    ready = {r.id for r in components() if r.status == "ready"}
    wanted = [c for c in installer.with_needs(ids) if c.id in ids or c.id not in ready]
    roots = _roots()
    return InstallPlan(wanted, sum(installer.missing_bytes(c, roots) for c in wanted))


def install(
    planned: InstallPlan, on_part: Callable[[catalogue.Component], None] | None = None,
    on_file: Callable[[catalogue.ModelFile], None] | None = None,
    on_bytes: Callable[[int], None] | None = None,
) -> dict[str, AppError | None]:  # fmt: skip
    """Install each planned part under work.lock; a part that fails does not stop the others."""
    roots = _roots()
    done: dict[str, AppError | None] = {}
    with work_lock(roots.data):
        for c in planned.parts:
            if on_part:
                on_part(c)
            try:
                installer.install(c, roots, on_file, on_bytes)
                done[c.id] = None
            except AppError as exc:
                done[c.id] = exc
    for c in planned.parts:  # a part that is opt-in here is used once the learner has installed it
        if done[c.id] is None and catalogue.SYSTEM in c.experimental_on:
            settings_set(f"components.{c.id}", "true")
    return done


def consent(component_id: str, source: Literal["ui", "cli-tty"]) -> dict[str, str]:
    """Record the learner's own consent to a part's licence: the licence box, or the typed words."""
    return installer.consent(paths.data_root(), component_id, source)


def check_removable(ids: list[str]) -> list[catalogue.Component]:
    """The named parts, once each known part has been found removable: a download or MFA's environment."""
    found = [c for c in installer.with_needs(ids) if c.id in ids]
    for c in found:
        installer.check_removable(c)
    return found


def remove(ids: list[str]) -> None:
    """Delete the named parts' files under work.lock; nothing is deleted when one of them cannot be."""
    roots = _roots()
    found = check_removable(ids)
    with work_lock(roots.data):
        for c in found:
            installer.remove(c, roots)
    for c in found:  # and deleting an opt-in part takes that choice back
        if catalogue.SYSTEM in c.experimental_on:
            settings_set(f"components.{c.id}", "false")


def token_status() -> hf_token.TokenStatus:
    """Whether a Hugging Face token is saved, works, and has the pyannote terms accepted (tested online)."""
    return hf_token.status()


def token_set(value: str) -> tuple[hf_token.TokenStatus, bool]:
    """Save the learner's token unless one that works is there already; returns the status and if it wrote."""
    return hf_token.save(value, paths.app_root())


def token_clear() -> bool:
    """Remove the token only if Fluentfolio saved it; returns whether it removed one."""
    return hf_token.clear(paths.app_root())


def token_file_place() -> str | None:
    """The shared place the token file lies in, if it does: Doctor and the token commands warn of it."""
    return hf_token.file_place(paths.app_root(), paths.data_root())


RELEASES_PAGE = updates.RELEASES


def check_updates() -> updates.Release:
    """Ask GitHub for the newest release; only when the learner clicks Check for updates."""
    return updates.check()


def prompt(kind: str) -> str:
    """The grading or the pronunciation prompt alone ("Copy prompt only")."""
    return for_ai.prompt(PROMPT_FILES[kind], SCHEMA_VERSION)


def agent_guide() -> str:
    """The guide for AI agents that run the command line."""
    return (resources.files("fluentfolio") / "agent" / "agent-guide.md").read_text(encoding="utf-8")


def write_skill(folder: Path) -> Path:
    """Copy the Agent Skills file to `folder`/fluentfolio/SKILL.md; written only when the user asks."""
    target = folder.absolute() / "fluentfolio" / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(target, (resources.files("fluentfolio") / "agent" / "SKILL.md").read_bytes())
    return target


def _refuse_nesting(kind: str, target: Path, old: Path, others: dict[str, Path]) -> None:
    """Refuse a move that would put one Fluentfolio folder in another or delete one with the old folder."""
    new, gone = target.resolve(), old.resolve()
    if new.is_relative_to(gone) or gone.is_relative_to(new):
        raise SetupError("E_MOVE_FAILED", "one folder is inside the other")
    # A move deletes the old folder with all it holds, so neither folder may end up inside the other one.
    other_kind = "models" if kind == "data" else "data"
    if new.is_relative_to(others[other_kind].resolve()):
        raise SetupError("E_MOVE_FAILED", f"the {kind} folder would lie inside the {other_kind} folder")
    for name, folder in others.items():
        if folder.resolve().is_relative_to(gone):
            detail = f"the {kind} folder holds the {name} folder, which the move would delete"
            raise SetupError("E_MOVE_FAILED", detail)
    # The uninstaller deletes the app folder with all it holds, so the learner's recordings stay outside it.
    if kind == "data" and new.is_relative_to(others["app"].resolve()):
        raise SetupError("E_MOVE_FAILED", "the data folder would lie inside the app folder")


def move_folder(kind: Literal["data", "models"], target: Path, *, keep_old: bool = False) -> Path:
    """Move the data or the models folder: copy, verify, switch, and only then delete the old copy.

    Refused while a job runs, because the mover must hold work.lock. With keep_old the caller deletes the old
    copy: the web server keeps its lock, log and launch files in the data folder until it stops.
    """
    app = paths.app_root()
    if kind == "data":
        if os.environ.get("FLUENTFOLIO_HOME"):
            raise InputError("E_HOME_SET_BY_ENV")
        old, others = paths.data_root(), {"app": app, "models": paths.models_dir()}
    else:
        old, others = paths.models_dir(), {"app": app, "data": paths.data_root()}
    # Typed into a form: a relative path would land beside wherever the server started.
    if not target.is_absolute() or target.is_file():
        raise SetupError("E_MOVE_FAILED", "not the full path of a folder")
    _refuse_nesting(kind, target, old, others)
    # The models folder holds files that MFA and espeak open, so it follows the app folder's path rule.
    if kind == "models" and (problems := paths.path_problems(target)):
        raise SetupError("E_PATH_RULE", ", ".join(problems))
    with work_lock(paths.data_root()):
        if old.exists():  # the running server's launch files hold its token and go with it
            copy_verified(old, target, skip=(*LOCK_FILES, "server.json", "open.html"))
        else:
            target.mkdir(parents=True, exist_ok=True)
        state = paths.install_state(app)
        state["data_root" if kind == "data" else "models_dir"] = str(target)
        write_json(app / "state.json", state)
    if not keep_old:
        shutil.rmtree(old, ignore_errors=True)
    return target
