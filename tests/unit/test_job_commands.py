"""The facade's job actions: the other person's agreement, status, resume and cancel, on hand-made jobs."""

from __future__ import annotations

import re
import sys
import wave
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fluentfolio import __version__
from fluentfolio.app import facade
from fluentfolio.core.enums import JobState
from fluentfolio.core.errors import AppError
from fluentfolio.core.files import read_json, write_json
from fluentfolio.jobs import loop, store
from fluentfolio.jobs.locks import work_lock

SPOKEN = [
    ("Reading", 1000, 1500, 0), ("is", 1550, 1700, 0), ("useful", 1750, 2300, 0), ("because", 2350, 2900, 0),
    ("it", 2950, 3050, 0), ("helps", 3100, 3500, 0), ("me", 3550, 3700, 0), ("learn.", 3750, 4300, 0),
]  # fmt: skip
UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The data folder the hand-made jobs live in, which the facade reads too."""
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture
def finished(home: Path, synthetic_job: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A finished solo job; the facade's plan for it leaves out the steps that need audio or a model."""
    job = synthetic_job.make(home.parent, SPOKEN, [(1000, 4300)], 6000)
    assert synthetic_job.finish(home.parent).state is JobState.COMPLETED
    stages = synthetic_job.plan("solo")
    monkeypatch.setattr(facade, "plan", lambda *choices: stages)
    return job


def silence(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))
    return path


def edit(job: Path, **values: Any) -> None:
    write_json(job / "job.json", {**read_json(job / "job.json"), **values})


@pytest.mark.parametrize("mode", ["interview", "role-swap"])
def test_two_people_need_the_other_persons_agreement_before_any_job(
    home: Path, tmp_path: Path, mode: str
) -> None:
    clip = silence(tmp_path / "clip.wav", 12)
    with pytest.raises(AppError) as caught:
        facade.analyze(clip, mode)
    assert (caught.value.code, caught.value.entry.exit_code) == ("E_CONSENT_REQUIRED", 5)
    assert not (home / "jobs").exists()


def test_solo_needs_no_agreement(home: Path, tmp_path: Path) -> None:
    # The next check after the agreement is the model, so its error shows the agreement check was passed.
    with pytest.raises(AppError) as caught:
        facade.analyze(silence(tmp_path / "clip.wav", 12), "solo", asr="no-such-model")
    assert caught.value.code == "E_COMPONENT_UNAVAILABLE"


def test_the_agreement_is_kept_with_the_job_as_time_and_source_only(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        loop, "run_job", lambda roots, job_id, *a, **k: store.read(roots.data, job_id, lock_free=False)
    )
    done = facade.analyze(silence(tmp_path / "Anna and Ben.wav", 12), "interview", partner_consent="cli-flag")
    consent = read_json(done.job / "job.json")["consent"]
    assert set(consent) == {"utc", "source"} and consent["source"] == "cli-flag"
    assert UTC.fullmatch(consent["utc"])


@pytest.mark.parametrize(("free_mb", "warned"), [(1999, True), (2000, False), (2001, False)])
def test_less_free_memory_than_a_steps_minimum_warns_before_the_run_and_in_its_result(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free_mb: int, warned: bool
) -> None:
    # The steps' minimums are planted (the largest is 2000 MB), so this checks the comparison at the limit
    # and one MB on either side, and where the warning goes; the minimums themselves are in test_estimate.py.
    monkeypatch.setattr(facade.estimate, "ram_needs_mb", lambda *a, **k: {"vad": 401, "phones": 2000})
    monkeypatch.setattr(facade, "_free_ram_mb", lambda: free_mb)
    seen: list[tuple[str, dict[str, Any]]] = []

    def run_job(roots: loop.Roots, job_id: str, *a: Any, **k: Any) -> store.JobRecord:
        seen.append(("run", {}))
        return store.read(roots.data, job_id, lock_free=False)

    monkeypatch.setattr(loop, "run_job", run_job)
    clip = silence(tmp_path / "clip.wav", 12)
    # By hand: 2000 MB is 1.95 GB, shown rounded up; 1999 MB is 1.95 GB, shown rounded down.
    expected = [("W_FREE_RAM", "the phones step needs about 2.0 GB; 1.9 GB is free")] if warned else []
    planned = facade.dry_run(clip, "solo")
    assert [w for w in planned.warnings if w[0] == "W_FREE_RAM"] == expected
    done = facade.analyze(clip, "solo", on_progress=lambda stage, event: seen.append((stage, event)))
    assert [w for w in done.warnings if w[0] == "W_FREE_RAM"] == expected
    before = [("", {"warning": code, "detail": detail}) for code, detail in expected]
    assert seen == [*before, ("run", {})]  # the warning comes before the run starts


def test_status_reads_running_only_while_the_work_lock_is_held(home: Path, finished: Path) -> None:
    edit(finished, state="running")
    with work_lock(home):
        assert facade.status(finished.name).record.state is JobState.RUNNING
    assert facade.status(finished.name).record.state is JobState.INTERRUPTED


def test_wait_returns_at_the_timeout_while_the_job_runs(home: Path, finished: Path) -> None:
    edit(finished, state="running")
    with work_lock(home):
        assert facade.wait(finished.name, timeout_s=0.3, poll_s=0.1).record.state is JobState.RUNNING


def test_resume_runs_only_the_unfinished_steps_and_keeps_the_rest(home: Path, finished: Path) -> None:
    before = {s["id"]: s for s in read_json(finished / "job.json")["stages"]}
    (finished / "stages" / "evidence.json").unlink()
    edit(finished, state="interrupted", error={"code": "E_INTERRUPTED"})
    done = facade.resume(finished.name)
    assert done.record.state is JobState.COMPLETED and done.record.error is None
    after = {s.id: s.model_dump(mode="json") for s in done.record.stages}
    assert list(after) == list(before)
    assert after["structure"] == before["structure"]  # a finished step is not run again
    assert after["evidence"]["started_utc"] >= before["evidence"]["started_utc"]
    assert (finished / "stages" / "evidence.json").is_file() and (
        finished / "stages" / "export.json"
    ).is_file()


def test_resume_of_a_completed_job_changes_nothing(home: Path, finished: Path) -> None:
    text = (finished / "job.json").read_bytes()
    assert facade.resume(finished.name).record.state is JobState.COMPLETED
    assert (finished / "job.json").read_bytes() == text


def test_resume_sets_the_state_of_a_job_that_stopped_after_its_last_step(home: Path, finished: Path) -> None:
    stages = read_json(finished / "job.json")["stages"]
    edit(finished, state="interrupted", error={"code": "E_JOB_INTERRUPTED"})
    done = facade.resume(finished.name)
    assert done.record.state is JobState.COMPLETED
    assert [s.model_dump(mode="json") for s in done.record.stages] == stages  # nothing ran again


def test_a_detached_job_no_runner_took_is_left_interrupted_and_resume_runs_it(
    home: Path, finished: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (finished / "stages" / "evidence.json").unlink()
    edit(finished, state="queued", error=None)
    started: list[list[str]] = []
    monkeypatch.setattr(facade.runner, "detach", lambda argv, env: started.append(argv))
    monkeypatch.setattr(facade, "DETACH_WAIT_S", 0.3)
    with pytest.raises(AppError) as caught:
        facade.start_detached(finished.name)
    assert (caught.value.code, caught.value.entry.exit_code) == ("E_RUNNER_NOT_STARTED", 7)
    assert started == [[sys.executable, "-I", "-X", "utf8", "-m", "fluentfolio", "_run", finished.name]]
    record = facade.status(finished.name).record
    assert record.state is JobState.INTERRUPTED and record.error == {"code": "E_RUNNER_NOT_STARTED"}
    assert facade.resume(finished.name).record.state is JobState.COMPLETED


def test_the_wait_ends_as_soon_as_a_runner_takes_the_job(
    home: Path, finished: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edit(finished, state="queued", error=None)
    monkeypatch.setenv("PYTHONPATH", "planted")
    envs: list[dict[str, str]] = []
    # The runner holds work.lock and has set the job running, as the hidden _run command does.
    monkeypatch.setattr(
        facade.runner, "detach", lambda argv, env: envs.append(env) or edit(finished, state="running")
    )
    with work_lock(home):
        assert facade.start_detached(finished.name).state is JobState.RUNNING
    assert envs and "PYTHONPATH" not in {name.upper() for name in envs[0]}


def test_resume_detach_queues_the_job_for_a_runner_of_its_own(
    home: Path, finished: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (finished / "stages" / "evidence.json").unlink()
    edit(finished, state="interrupted", error={"code": "E_INTERRUPTED"})
    seen: list[str] = []

    def runner_now(argv: list[str], env: dict[str, str]) -> None:
        seen.append(read_json(finished / "job.json")["state"])
        facade.resume(argv[-1])  # the runner's whole run, done before the wait starts

    monkeypatch.setattr(facade.runner, "detach", runner_now)
    done = facade.resume_detached(finished.name)
    assert seen == ["queued"] and done.record.state is JobState.COMPLETED


def test_resume_refuses_a_job_of_another_version(home: Path, finished: Path) -> None:
    edit(finished, state="interrupted", app_version=__version__ + ".old")
    with pytest.raises(AppError) as caught:
        facade.resume(finished.name)
    assert caught.value.code == "E_RESUME_OTHER_VERSION"


@pytest.mark.parametrize("detach", [False, True])
def test_resume_refuses_a_job_whose_delete_stopped_at_a_held_file(
    home: Path, finished: Path, monkeypatch: pytest.MonkeyPatch, detach: bool
) -> None:
    edit(finished, state="failed", error={"code": "E_DELETE_INCOMPLETE"})
    monkeypatch.setattr(facade.runner, "detach", lambda argv, env: None)  # no runner starts, whatever happens
    with pytest.raises(AppError) as caught:
        (facade.resume_detached if detach else facade.resume)(finished.name)
    assert (caught.value.code, read_json(finished / "job.json")["state"]) == ("E_DELETE_INCOMPLETE", "failed")


def test_cancel_asks_only_a_running_job_and_a_dead_jobs_request_is_dropped(
    home: Path, finished: Path
) -> None:
    request = finished / store.CANCEL_FILE
    record, asked = facade.cancel(finished.name)
    assert (record.state, asked, request.exists()) == (JobState.COMPLETED, False, False)
    edit(finished, state="running")
    with work_lock(home):
        record, asked = facade.cancel(finished.name)
    assert (record.state, asked, request.exists()) == (JobState.RUNNING, True, True)
    # The runner died before it saw the request: the next run, which has a step to do, must not stop for it.
    (finished / "stages" / "export.json").unlink()
    done = facade.resume(finished.name)
    assert done.record.state is JobState.COMPLETED and not request.exists()


def test_jobs_lists_the_newest_first(home: Path, finished: Path) -> None:
    older = home / "jobs" / "20200101-000000-0001"
    older.mkdir()
    write_json(older / "job.json", {**read_json(finished / "job.json"), "job_id": older.name})
    assert [r.job_id for r in facade.jobs(10)] == [finished.name, older.name]
    assert [r.job_id for r in facade.jobs(1)] == [finished.name]
