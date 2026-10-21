"""A finished job run again in another mode: a new job that copies the transcription and never redoes it."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fluentfolio import __version__
from fluentfolio.app import facade
from fluentfolio.core.enums import JobState
from fluentfolio.core.errors import AppError, SetupError
from fluentfolio.core.files import read_json, write_json
from fluentfolio.jobs import loop
from fluentfolio.jobs.locks import work_lock
from fluentfolio.speakers import checks, sherpa

# An examiner asks (S2), the learner answers (S1); every time is chosen here.
SPOKEN = [
    ("What", 1000, 1300, 0), ("do", 1350, 1500, 0), ("you", 1550, 1700, 0), ("study?", 1750, 2600, 0),
    ("I", 3000, 3200, 1), ("study", 3250, 3700, 1), ("biology", 3750, 4500, 1), ("because", 4550, 5100, 1),
    ("I", 5150, 5300, 1), ("love", 5350, 5800, 1), ("living", 5850, 6400, 1), ("things.", 6450, 7400, 1),
]  # fmt: skip
SEGMENTS = [(1000, 2600), (3000, 7400)]
TURNS = [(1000, 2600, "S2"), (3000, 7400, "S1")]
COPIED = ("media", "vad", "language", "asr")  # never run again, whatever the new mode


def two_voices(job: Path) -> dict[str, Any]:
    """A speaker step that finds the two voices of TURNS, in place of a model the test does not load."""
    mode = str(read_json(job / "job.json")["mode"])
    return {**checks.summarize(sherpa.BACKEND, TURNS, mode=mode), "version": "hand-made 1", "device": "cpu"}


@pytest.fixture
def ran() -> list[tuple[str, str]]:
    """(job ID, step) for each step that ran in this process, in order."""
    return []


@pytest.fixture
def home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_job: ModuleType, ran: list[tuple[str, str]]
) -> Path:
    """The data folder with a finished solo job; each mode's plan leaves out the steps that need audio."""
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(tmp_path / "data"))
    job = synthetic_job.make(tmp_path, SPOKEN, SEGMENTS, 8000)
    (job / "input" / "audio16k.wav").write_bytes(b"RIFF and the rest of a recording")
    record = read_json(job / "job.json")
    write_json(
        job / "job.json", {**record, "input": {**record["input"], "duration_ms": 8000}}
    )  # as analyze does
    assert synthetic_job.finish(tmp_path).state is JobState.COMPLETED

    def noted(stage: loop.Stage) -> loop.Stage:
        if stage.id == "speakers":
            stage = replace(stage, inline=two_voices, child="", gpu=False)
        if stage.inline is None:  # a child step; the hand-made job lacks its audio, so it would fail
            return stage
        inner = stage.inline

        def step(job: Path) -> dict[str, Any]:
            ran.append((job.name, stage.id))
            return inner(job)

        return replace(stage, inline=step)

    plans = {mode: synthetic_job.plan(mode) for mode in ("solo", "interview", "role-swap")}
    plans = {mode: plans[mode] if mode == "solo" else [noted(s) for s in plans[mode]] for mode in plans}
    monkeypatch.setattr(facade, "plan", lambda mode, *choices: plans[mode])
    return tmp_path / "data"


def files(job: Path) -> dict[str, bytes]:
    return {p.relative_to(job).as_posix(): p.read_bytes() for p in sorted(job.rglob("*")) if p.is_file()}


def stages(job: Path) -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in read_json(job / "job.json")["stages"]}


def only_job(home: Path) -> Path:
    (job,) = (home / "jobs").iterdir()
    return job


def test_solo_run_again_as_an_interview_separates_the_speakers_and_keeps_the_transcription(
    home: Path, ran: list[tuple[str, str]]
) -> None:
    old = only_job(home)
    before = files(old)
    offer = facade.rerun_offer(old.name, "interview")
    assert (offer.first, offer.needs_consent) == ("speakers", True)
    assert offer.estimate_ms is not None and offer.estimate_ms > 0
    with pytest.raises(AppError) as caught:
        facade.rerun(old.name, "interview")  # the other person's agreement is not on record yet
    assert caught.value.code == "E_CONSENT_REQUIRED" and only_job(home) == old
    done = facade.rerun(old.name, "interview", partner_consent="cli-flag")
    record = done.record
    assert record.state is JobState.COMPLETED and done.job != old, record.stages
    assert (record.mode, record.rerun_of, record.rerun_from) == ("interview", old.name, "speakers")
    assert record.consent["source"] == "cli-flag" and record.swap_style is None
    assert files(old) == before  # the solo job is as it was
    for name in ("input/audio16k.wav", *(f"stages/{step}.json" for step in COPIED)):
        assert (done.job / name).read_bytes() == before[name], name
    assert all(stages(done.job)[step] == stages(old)[step] for step in COPIED)  # the records too
    steps = ["speakers", "words", "structure", "measures", "evidence", "export"]
    assert [step for job_id, step in ran if job_id == record.job_id] == steps
    words = read_json(done.job / "stages" / "words.json")["words"]
    owners = {w["start_ms"]: w["speaker"] for w in words}
    assert {owners[start] for _, start, _, chunk in SPOKEN if chunk == 0} == {"S2"}
    assert {owners[start] for _, start, _, chunk in SPOKEN if chunk == 1} == {"S1"}
    assert read_json(done.job / "evidence.json")["manifest"]["mode"] == "interview"
    assert facade.job_estimate_ms(record) == offer.estimate_ms  # its own steps only, as offered


def test_an_interview_run_again_as_a_role_swap_keeps_the_speakers_and_the_agreement(
    home: Path, ran: list[tuple[str, str]]
) -> None:
    interview = facade.rerun(only_job(home).name, "interview", partner_consent="cli-flag")
    before = files(interview.job)
    offer = facade.rerun_offer(interview.record.job_id, "role-swap")
    assert (offer.first, offer.needs_consent) == ("structure", False)
    done = facade.rerun(interview.record.job_id, "role-swap")
    assert done.record.state is JobState.COMPLETED, done.record.stages
    record = done.record
    assert (record.mode, record.swap_style, record.rerun_from) == ("role-swap", "once", "structure")
    assert done.record.consent == interview.record.consent  # the agreement given for this recording
    for step in (*COPIED, "speakers", "words"):
        assert (done.job / "stages" / f"{step}.json").read_bytes() == before[f"stages/{step}.json"], step
        assert stages(done.job)[step] == stages(interview.job)[step]
    steps = ["structure", "measures", "evidence", "export"]
    assert [step for job_id, step in ran if job_id == record.job_id] == steps
    assert files(interview.job) == before


def test_a_rerun_job_resumed_never_runs_a_step_before_its_first_even_a_failed_one(
    home: Path, ran: list[tuple[str, str]]
) -> None:
    done = facade.rerun(only_job(home).name, "interview", partner_consent="cli-flag")
    job = done.job
    # The language check had failed in the first job: a gap, kept as it was.
    write_json(job / "stages" / "language.json", {"status": "failed", "error_code": "E_STAGE_CRASHED"})
    record = read_json(job / "job.json")
    # Stopped during the evidence step, which leaves only its start time and no result.
    stopped = {
        "language": {"status": "failed"},
        "evidence": {"status": None, "ended_utc": None, "wall_ms": None},
    }
    record["stages"] = [{**s, **stopped.get(s["id"], {})} for s in record["stages"]]
    write_json(job / "job.json", {**record, "state": "interrupted", "error": {"code": "E_INTERRUPTED"}})
    (job / "stages" / "evidence.json").unlink()
    before, kept = files(job), stages(job)
    ran.clear()
    again = facade.resume(job.name)
    assert ran == [(job.name, "evidence"), (job.name, "export")]
    assert again.record.state is JobState.COMPLETED_WITH_GAPS, again.record.stages
    for name in ("language", "asr", "speakers", "words", "structure"):
        assert (job / "stages" / f"{name}.json").read_bytes() == before[f"stages/{name}.json"], name
    assert all(stages(job)[step] == kept[step] for step in [*COPIED, "speakers", "words", "structure"])
    assert kept["evidence"]["status"] is None and kept["evidence"]["ended_utc"] is None
    assert (job / "stages" / "evidence.json").is_file() and stages(job)["evidence"]["status"] == "ok"
    assert stages(job)["evidence"]["ended_utc"] is not None


@pytest.mark.parametrize(
    ("change", "mode", "code"),
    [
        ({}, "solo", "E_INPUT_OPTION"),  # the mode it has
        ({}, "duet", "E_INPUT_OPTION"),
        ({"state": "interrupted"}, "interview", "E_JOB_NOT_FINISHED"),
        ({"state": "failed"}, "interview", "E_JOB_NOT_FINISHED"),
        ({"app_version": __version__ + ".old"}, "interview", "E_RESUME_OTHER_VERSION"),
        ({"mode": "interview", "layout": "per-track"}, "solo", "E_INPUT_LAYOUT"),
    ],
)
def test_a_job_that_cannot_run_again_in_that_mode_is_refused_and_no_job_is_made(
    home: Path, change: dict[str, Any], mode: str, code: str
) -> None:
    old = only_job(home)
    write_json(old / "job.json", {**read_json(old / "job.json"), **change})
    before = files(old)
    with pytest.raises(AppError) as caught:
        facade.rerun(old.name, mode, partner_consent="cli-flag")
    assert caught.value.code == code
    assert only_job(home) == old and files(old) == before


def test_a_busy_folder_or_an_unknown_job_is_refused(home: Path) -> None:
    old = only_job(home)
    with work_lock(home), pytest.raises(AppError) as caught:
        facade.rerun(old.name, "interview", partner_consent="cli-flag")
    assert caught.value.code == "E_BUSY"
    with pytest.raises(AppError) as caught:
        facade.rerun("20261013-120000-ffff", "interview", partner_consent="cli-flag")
    assert caught.value.code == "E_JOB_NOT_FOUND" and only_job(home) == old


def test_a_copy_that_fails_leaves_no_new_job_and_the_old_one_as_it_was(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = only_job(home)
    before = files(old)

    def full(src: Path, dst: Path) -> None:
        raise SetupError("E_DISK_FULL", "need 1 GB, have 0 B")

    monkeypatch.setattr(facade, "copy_verified", full)
    with pytest.raises(AppError) as caught:
        facade.rerun(old.name, "interview", partner_consent="cli-flag")
    assert caught.value.code == "E_DISK_FULL"
    assert only_job(home) == old and files(old) == before


def test_rerun_detach_queues_the_new_job_for_a_runner_that_keeps_the_copied_steps(
    home: Path, monkeypatch: pytest.MonkeyPatch, ran: list[tuple[str, str]]
) -> None:
    old = only_job(home)
    seen: list[tuple[str, str]] = []

    def runner_now(argv: list[str], env: dict[str, str]) -> None:
        seen.append((argv[-1], read_json(home / "jobs" / argv[-1] / "job.json")["state"]))
        facade.resume(argv[-1])  # what the hidden _run command does, done before the wait starts

    monkeypatch.setattr(facade.runner, "detach", runner_now)
    done = facade.rerun(old.name, "interview", partner_consent="cli-flag", detach=True)
    assert seen == [(done.record.job_id, "queued")] and done.record.state is JobState.COMPLETED
    assert [step for _, step in ran] == ["speakers", "words", "structure", "measures", "evidence", "export"]
    assert all(stages(done.job)[step] == stages(old)[step] for step in COPIED)
    assert (done.job / "stages" / "asr.json").read_bytes() == (old / "stages" / "asr.json").read_bytes()
