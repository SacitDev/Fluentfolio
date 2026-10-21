"""status, jobs list, cancel, export and prompt in real processes; stop, resume and cancel a real job."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from fluentfolio.app import facade
from fluentfolio.core.enums import Device
from fluentfolio.core.files import read_json, write_json
from fluentfolio.export.for_ai import PROMPT_VERSION
from fluentfolio.ingest import secondary
from fluentfolio.jobs.locks import work_lock
from fluentfolio.settings.model import Settings
from fluentfolio.settings.store import snapshot
from fluentfolio.web.tasks import Tasks

Run = Callable[..., subprocess.CompletedProcess[bytes]]
Read = Callable[[subprocess.CompletedProcess[bytes]], Any]
Job = tuple[Path, str, Path]
PROMPTS = Path(__file__).resolve().parents[2] / "src" / "fluentfolio" / "export" / "prompts"
AUDIO = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
CLIPS = ("ls_4446_2273_0000", "ls_4446_2273_0001", "ls_4446_2273_0002")  # public-domain read speech
OLD_UTC = "2026-01-01T00:00:00Z"  # a step's time before any test runs, so a step run again gets a later one


def lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")  # stdout is text mode, so Windows writes CRLF


def test_status_reports_a_finished_job_its_outputs_and_each_step(
    cli: Run, envelope: Read, finished: Job
) -> None:
    home, job_id, job = finished
    data = envelope(cli("status", job_id, "--json", home=home))["data"]
    assert (data["job_id"], data["state"], data["mode"], data["stage"]) == (job_id, "completed", "solo", None)
    assert data["audio_duration_ms"] == 20_000  # the length the hand-made media result gives
    outputs = data["outputs"]
    assert outputs["job_dir"] == job.as_posix() and Path(outputs["zip"]).is_file()
    assert [(f["candidate"], Path(f["path"]).name) for f in outputs["for_ai"]] == [
        ("candidate", "for_ai.txt")
    ]
    assert [s["id"] for s in data["stages"]] == [s["id"] for s in read_json(job / "job.json")["stages"]]
    assert data["elapsed_ms"] >= 0 and all(s["wall_ms"] is not None for s in data["stages"])


@pytest.mark.parametrize(("limit", "rows"), [(1, 1), (2, 2), (3, 3), (4, 3)])
def test_jobs_list_gives_the_newest_first_up_to_the_limit(
    cli: Run, envelope: Read, finished: Job, limit: int, rows: int
) -> None:
    home, job_id, job = finished
    ids = ["20261012-080000-0001", "20261014-080000-0002"]
    for other in ids:
        shutil.copytree(job, home / "jobs" / other)
        write_json(home / "jobs" / other / "job.json", {**read_json(job / "job.json"), "job_id": other})
    found = envelope(cli("jobs", "list", "--limit", limit, "--json", home=home))["data"]["jobs"]
    newest_first = [ids[1], job_id, ids[0]]
    assert [j["job_id"] for j in found] == newest_first[:rows]
    assert set(found[0]) == {"job_id", "created_utc", "state", "mode", "audio_duration_ms"}


def test_jobs_delete_removes_the_job_its_temp_and_mfa_folders_and_nothing_else(
    tmp_path: Path, cli: Run, envelope: Read, finished: Job
) -> None:
    home, job_id, job = finished
    other = home / "jobs" / "20261012-080000-0001"
    shutil.copytree(job, other)
    write_json(other / "job.json", {**read_json(job / "job.json"), "job_id": other.name})
    app = tmp_path / "app"
    left = [app / "tmp" / job_id, app / "tmp" / "mfa" / job_id, app / "tmp" / other.name]
    for folder in left:
        folder.mkdir(parents=True)
        (folder / "corpus.txt").write_text("x", encoding="utf-8")
    env = {"FLUENTFOLIO_APP_ROOT": str(app)}
    found = envelope(cli("jobs", "delete", job_id, "--yes", "--json", home=home, env=env))
    assert (found["exit_code"], found["data"]) == (0, {"job_id": job_id, "deleted": True})
    assert [p.exists() for p in [job, *left]] == [False, False, False, True]
    assert other.is_dir()  # the other job stays
    listed = envelope(cli("jobs", "list", "--json", home=home, env=env))["data"]["jobs"]
    assert [j["job_id"] for j in listed] == [other.name]


def test_jobs_delete_asks_first_and_refuses_an_unknown_job_or_a_busy_folder(
    cli: Run, envelope: Read, finished: Job
) -> None:
    home, job_id, job = finished
    found = envelope(cli("jobs", "delete", job_id, "--json", home=home))  # no terminal to ask in
    assert (found["error"]["code"], found["exit_code"]) == ("E_CONFIRM_NEEDED", 5) and job.is_dir()
    found = envelope(cli("jobs", "delete", "20261013-120000-ffff", "--yes", "--json", home=home))
    assert (found["error"]["code"], found["exit_code"]) == ("E_JOB_NOT_FOUND", 4)
    with work_lock(home):  # a job runs
        found = envelope(cli("jobs", "delete", job_id, "--yes", "--json", home=home))
    assert (found["error"]["code"], found["exit_code"]) == ("E_BUSY", 6) and job.is_dir()
    shown = cli("jobs", "delete", job_id, "--yes", home=home)
    assert shown.returncode == 0 and b"skip the Recycle Bin" in shown.stdout and not job.exists()


def test_a_limit_under_1_is_a_usage_error(cli: Run, envelope: Read, finished: Job) -> None:
    assert (
        envelope(cli("jobs", "list", "--limit", 0, "--json", home=finished[0]))["error"]["code"] == "E_USAGE"
    )


def test_cancel_writes_the_request_only_for_a_running_job(cli: Run, envelope: Read, finished: Job) -> None:
    home, job_id, job = finished
    request = job / "cancel.request"
    found = envelope(cli("cancel", job_id, "--json", home=home))["data"]
    assert (found["requested"], found["state"], request.exists()) == (False, "completed", False)
    write_json(job / "job.json", {**read_json(job / "job.json"), "state": "running"})
    with work_lock(home):
        found = envelope(cli("cancel", job_id, "--json", home=home))["data"]
    assert (found["requested"], found["state"], request.exists()) == (True, "running", True)


def test_export_writes_again_only_what_is_missing_and_copies_on_request(
    tmp_path: Path, cli: Run, envelope: Read, finished: Job
) -> None:
    home, job_id, job = finished
    text = job / "export" / "for_ai" / "for_ai.txt"
    zipped = job / "export" / f"fluentfolio_{job_id}.zip"
    before = text.read_bytes()
    found = envelope(cli("export", job_id, "--json", home=home))["data"]
    assert found["changed"] is False and text.read_bytes() == before  # the same boundary, so the same bytes
    zipped.unlink()
    found = envelope(cli("export", job_id, "--json", home=home))["data"]
    assert found["changed"] is True and zipped.is_file() and found["outputs"]["zip"] == zipped.as_posix()
    out = tmp_path / "for my AI"
    found = envelope(cli("export", job_id, "--out", out, "--json", home=home))["data"]
    copied = Path(found["outputs"]["for_ai"][0]["path"])
    assert copied.parent == out and copied.read_bytes() == text.read_bytes()
    assert (out / zipped.name).read_bytes() == zipped.read_bytes()
    shown = cli("export", job_id, "--print", home=home)
    assert shown.returncode == 0 and lf(shown.stdout) == text.read_bytes()


def test_export_audio_runs_only_the_audio_step_and_the_steps_that_name_its_file(
    cli: Run, envelope: Read, finished: Job
) -> None:
    home, job_id, job = finished
    # A job made by analyze keeps its settings; this hand-made one gets the defaults, audio off. Its steps get
    # an older time, because a step run again within the same second can write the very same record.
    old = read_json(job / "job.json")
    stages = [{**s, "started_utc": OLD_UTC, "ended_utc": OLD_UTC} for s in old["stages"]]
    write_json(job / "job.json", {**old, "settings": snapshot(Settings()), "stages": stages})
    before = {s["id"]: s for s in read_json(job / "job.json")["stages"]}
    found = envelope(cli("export", job_id, "--audio", "--json", home=home))["data"]
    record = read_json(job / "job.json")
    after = {s["id"]: s for s in record["stages"]}
    assert record["settings"]["export"]["audio"] is True  # the job keeps the learner's choice
    assert {k for k, s in after.items() if s != before.get(k)} == {"ai_audio", "evidence", "export"}
    assert list(after) == [
        "media", "vad", "language", "asr", "speakers", "words", "structure", "ai_audio", "measures",
        "evidence", "export",
    ]  # fmt: skip
    # This hand-made job has no audio of the candidate's own, so the step has nothing to export.
    assert read_json(job / "stages" / "ai_audio.json")["status"] == "skipped"
    assert (found["audio"], found["changed"]) == ([], True)
    assert record["state"] == "completed_with_gaps"  # asked for and not made: a gap, as in a first run
    again = envelope(cli("export", job_id, "--audio", "--json", home=home))["data"]
    assert again["changed"] is False and read_json(job / "job.json")["stages"] == record["stages"]
    shown = cli("export", job_id, "--audio", home=home)
    assert shown.returncode == 0 and b"No audio file: no candidate has speech of their own." in shown.stdout


def test_rerun_needs_the_other_persons_agreement_and_a_mode_the_job_does_not_have(
    cli: Run, envelope: Read, finished: Job
) -> None:
    home, job_id, _ = finished
    found = envelope(cli("rerun", job_id, "--mode", "interview", "--json", home=home))
    assert (found["error"]["code"], found["exit_code"]) == ("E_CONSENT_REQUIRED", 5)  # no terminal to ask at
    found = envelope(cli("rerun", job_id, "--mode", "solo", "--json", home=home))
    assert (found["error"]["code"], found["exit_code"]) == ("E_INPUT_OPTION", 4)
    assert [p.name for p in (home / "jobs").iterdir()] == [job_id]  # no new job either time


def test_export_print_takes_neither_json_nor_out(cli: Run, envelope: Read, finished: Job) -> None:
    found = envelope(cli("export", finished[1], "--print", "--json", home=finished[0]))
    assert found["error"]["code"] == "E_USAGE" and "--print" in found["error"]["message"]


def test_export_of_a_running_job_waits_for_its_end(cli: Run, envelope: Read, finished: Job) -> None:
    home, job_id, job = finished
    write_json(job / "job.json", {**read_json(job / "job.json"), "state": "running"})
    with work_lock(home):
        found = envelope(cli("export", job_id, "--json", home=home))
    assert found["error"]["code"] == "E_JOB_NOT_FINISHED" and found["exit_code"] == 4


@pytest.mark.parametrize(("kind", "file"), [("grading", "grader"), ("pronunciation", "pronunciation")])
def test_prompt_prints_the_prompt_file_alone(cli: Run, envelope: Read, kind: str, file: str) -> None:
    expected = (PROMPTS / f"{file}_v{PROMPT_VERSION}.txt").read_bytes()
    shown = cli("prompt", "--kind", kind)
    assert shown.returncode == 0 and lf(shown.stdout) == expected
    found = envelope(cli("prompt", "--kind", kind, "--json"))["data"]
    assert found["text"].encode("utf-8") == expected and found["prompt_version"] == PROMPT_VERSION


def test_check_answer_finds_a_removed_name_and_never_prints_it(
    tmp_path: Path, cli: Run, envelope: Read, finished: Job
) -> None:
    home, job_id, job = finished
    own = tmp_path / "named.txt"
    own.write_text("Ilsabet Varga: Well, I think reading is really useful.\n", encoding="utf-8")
    secondary.store_transcript(job, own)  # a made-up speaker label, taken out as the transcript is stored
    answer = tmp_path / "answer.txt"
    answer.write_text(
        "audio_access: none\naudio_check: not done\ninputs: for_ai.txt\nYour results: Ilsabet keeps going.\n",
        encoding="utf-8",
    )
    done = cli("check-answer", job_id, answer, "--json", home=home)
    results = {r["rule"]: r for r in envelope(done)["data"]["results"]}
    assert results["AC-09"]["verdict"] == "FAIL" and results["AC-09"]["detail"] == "1 removed name(s) appear"
    human = cli("check-answer", job_id, answer, home=home)
    for shown in (done, human):
        assert b"Ilsabet" not in shown.stdout + shown.stderr and b"Varga" not in shown.stdout + shown.stderr


@pytest.fixture
def speech(tmp_path: Path, session_builder: ModuleType) -> Path:
    """About half a minute of read speech in one file, in a folder whose name has non-ASCII letters."""
    folder = tmp_path / "kayıt"
    folder.mkdir()
    gap = np.zeros(8000, dtype=np.float32)
    parts = [x for name in CLIPS for x in (session_builder.load_clip(AUDIO / f"{name}.opus"), gap)]
    sf.write(folder / "practice.wav", np.concatenate(parts), 16000, subtype="PCM_16")
    return folder / "practice.wav"


def start(argv: list[object], home: Path) -> subprocess.Popen[bytes]:
    windows = sys.platform == "win32"
    return subprocess.Popen(
        [sys.executable, "-m", "fluentfolio", *map(str, argv)],
        env={**os.environ, "FLUENTFOLIO_HOME": str(home)}, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if windows else 0,
    )  # fmt: skip


def at_step(home: Path, step: str, proc: subprocess.Popen[bytes], limit_s: float = 300) -> tuple[str, dict]:
    """Wait until the job's runner is in `step`; returns the job ID and its record then."""
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline and proc.poll() is None:
        for path in (home / "jobs").glob("*/job.json") if (home / "jobs").is_dir() else []:
            try:
                record = read_json(path)
            except (OSError, ValueError):  # caught while the runner replaced it
                continue
            if record.get("stage") == step and record.get("state") == "running":
                return path.parent.name, record
        time.sleep(0.2)
    raise AssertionError(f"the job never reached {step}")


def test_a_real_job_that_stops_for_a_missing_model_exits_7_and_names_the_setup_error(
    tmp_path: Path, cli: Run, envelope: Read, speech: Path
) -> None:
    home, empty = tmp_path / "home", tmp_path / "no-models"
    done = cli("analyze", speech, "--mode", "solo", "--device", "cpu", "--json", home=home,
               env={"FLUENTFOLIO_APP_ROOT": str(empty)}, timeout=300)  # fmt: skip
    found = envelope(done)
    # A job's outcome is 0, 7 or 8; the error's category tells an agent that setup, not a step, failed.
    assert done.returncode == 7 and found["error"]["code"] == "E_MODEL_MISSING", done.stderr[-2000:]
    assert found["error"]["category"] == "setup" and found["data"]["state"] == "failed"
    record = read_json(home / "jobs" / found["data"]["job_id"] / "job.json")
    assert record["error"]["code"] == "E_MODEL_MISSING"  # the job keeps the reason that resume needs fixed


@pytest.mark.slow
def test_ctrl_break_interrupts_a_real_job_and_resume_finishes_it_with_its_steps_kept(
    tmp_path: Path, cli: Run, envelope: Read, speech: Path
) -> None:
    home = tmp_path / "home"
    proc = start(["analyze", speech, "--mode", "solo", "--device", "cpu", "--json"], home)
    job_id, record = at_step(home, "asr", proc)
    proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)  # type: ignore[attr-defined]
    stdout, stderr = proc.communicate(timeout=60)
    found = envelope(subprocess.CompletedProcess([], proc.returncode, stdout, stderr))
    assert proc.returncode == 130 and found["error"]["code"] == "E_INTERRUPTED", stderr[-2000:]
    assert found["data"]["job_id"] == job_id and found["data"]["state"] == "interrupted"
    kept = {s["id"]: s for s in record["stages"] if s["status"] == "ok"}
    assert {"media", "vad"} <= set(kept)
    done = cli("resume", job_id, "--json", home=home, timeout=900)
    found = envelope(done)
    assert done.returncode == 0 and found["data"]["state"] in ("completed", "completed_with_gaps"), (
        done.stderr[-2000:]
    )
    after = {s["id"]: s for s in read_json(home / "jobs" / job_id / "job.json")["stages"]}
    assert all(after[step] == kept[step] for step in kept)  # the finished steps were not run again
    assert after["asr"]["status"] == "ok" and found["data"]["outputs"]["for_ai"]


@pytest.mark.slow
def test_a_real_job_the_web_app_started_is_interrupted_when_the_app_stops_and_resume_finishes_it(
    tmp_path: Path, cli: Run, envelope: Read, speech: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(home))
    tasks = Tasks(home)
    started = facade.analyze(speech, "solo", device=Device.CPU, detach=True, start=tasks.start_job)
    task = tasks.running()
    assert task is not None and task.kind == "job"
    job_id, record = at_step(home, "asr", task.proc)
    assert job_id == started.record.job_id
    asked = time.monotonic()
    tasks.stop()  # what Quit, Ctrl+C or a closed window does to the server's children: their stdin closes
    assert time.monotonic() - asked < 60 and task.proc.returncode == 130, task.last_line()
    assert read_json(home / "jobs" / job_id / "job.json")["state"] == "interrupted"
    kept = {s["id"]: s for s in record["stages"] if s["status"] == "ok"}
    assert {"media", "vad"} <= set(kept)
    done = cli("resume", job_id, "--json", home=home, timeout=900)
    found = envelope(done)
    assert done.returncode == 0 and found["data"]["state"] in ("completed", "completed_with_gaps"), (
        done.stderr[-2000:]
    )
    after = {s["id"]: s for s in read_json(home / "jobs" / job_id / "job.json")["stages"]}
    assert all(after[step] == kept[step] for step in kept)  # "Run again" keeps the finished steps


@pytest.mark.slow
def test_a_cancel_from_another_process_stops_a_real_job_within_10_s(
    tmp_path: Path, cli: Run, envelope: Read, speech: Path
) -> None:
    home = tmp_path / "home"
    proc = start(["analyze", speech, "--mode", "solo", "--device", "cpu", "--json"], home)
    job_id, _ = at_step(home, "asr", proc)
    asked = time.monotonic()
    assert envelope(cli("cancel", job_id, "--json", home=home))["data"]["requested"] is True
    stdout, stderr = proc.communicate(timeout=60)
    assert time.monotonic() - asked < 10, stderr[-2000:]
    found = envelope(subprocess.CompletedProcess([], proc.returncode, stdout, stderr))
    assert proc.returncode == 7 and found["error"]["code"] == "E_JOB_CANCELLED"
    assert found["data"]["state"] == "cancelled"
    assert not (home / "jobs" / job_id / "cancel.request").exists()


@pytest.mark.slow
def test_rerun_of_a_real_solo_job_as_an_interview_shows_its_estimate_and_keeps_the_transcription(
    tmp_path: Path, cli: Run, envelope: Read, speech: Path
) -> None:
    home = tmp_path / "home"
    first = cli("analyze", speech, "--mode", "solo", "--device", "cpu", "--json", home=home, timeout=900)
    old = home / "jobs" / envelope(first)["data"]["job_id"]
    before = {s["id"]: s for s in read_json(old / "job.json")["stages"]}
    done = cli(
        "rerun", old.name, "--mode", "interview", "--partner-consent", "--json", home=home, timeout=900
    )
    found = envelope(done)
    assert done.returncode == 0 and found["data"]["mode"] == "interview", done.stderr[-2000:]
    assert b"Runs again from the speakers step: about " in done.stderr and found["data"]["device"] == "cpu"
    new = home / "jobs" / found["data"]["job_id"]
    asr = (old / "stages" / "asr.json").read_bytes()
    assert new != old and (new / "stages" / "asr.json").read_bytes() == asr  # not transcribed again
    after = {s["id"]: s for s in read_json(new / "job.json")["stages"]}
    assert all(after[step] == before[step] for step in ("media", "vad", "language", "asr"))
    # The speaker step ran again, after the solo job had ended, as an interview's speaker separation.
    assert after["speakers"]["started_utc"] >= before["export"]["ended_utc"]
    speakers = read_json(new / "stages" / "speakers.json")
    assert speakers["mode"] == "interview" and "turns" in speakers and "second_voice" not in speakers
