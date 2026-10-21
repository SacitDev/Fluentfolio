"""A command the web app starts keeps running while the app lives and stops, as at Ctrl+C, when it ends."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jsonschema
import pytest

from fluentfolio.core.files import read_json, write_json
from fluentfolio.jobs import runner
from fluentfolio.jobs.locks import work_lock

Job = tuple[Path, str, Path]
STILL_S = 3.0  # long enough for the command to start and to have stopped by itself if it would
STOP_S = 10.0


def waiting(home: Path, job_id: str, out: Path, err: Path) -> subprocess.Popen[bytes]:
    """`status --wait` on a job that reads as running: it waits until the job stops or the timeout."""
    argv = [sys.executable, "-I", "-X", "utf8", "-m", "fluentfolio", "status", job_id, "--wait", "--timeout",
            "60", "--json"]  # fmt: skip
    with out.open("wb") as reply, err.open("wb") as lines:
        return runner.attached(argv, {**os.environ, "FLUENTFOLIO_HOME": str(home)}, reply, lines)


def test_a_command_started_by_the_app_stops_as_at_ctrl_c_once_its_stdin_closes(
    tmp_path: Path, finished: Job, validator: jsonschema.Draft202012Validator
) -> None:
    home, job_id, job = finished
    write_json(job / "job.json", {**read_json(job / "job.json"), "state": "running"})
    out, err = tmp_path / "reply.json", tmp_path / "lines.txt"
    with work_lock(home):  # the job runs
        proc = waiting(home, job_id, out, err)
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                proc.wait(STILL_S)  # the open pipe never stops it
            asked = time.monotonic()
            code = runner.stop_attached(proc, STOP_S)
            took = time.monotonic() - asked
        finally:
            if proc.poll() is None:
                proc.kill()
    reply = json.loads(out.read_text(encoding="utf-8"))
    validator.validate(reply)
    assert code == 130 and reply["error"]["code"] == "E_INTERRUPTED", err.read_text(encoding="utf-8")
    assert took < STOP_S  # stopped by the closed pipe, not ended by the kill after the wait


@pytest.mark.parametrize("command", [["--version"], ["jobs", "list", "--json"]])
def test_a_command_started_by_the_app_that_ends_by_itself_exits_cleanly(
    tmp_path: Path, finished: Job, command: list[str]
) -> None:
    home, _, _ = finished
    argv = [sys.executable, "-I", "-X", "utf8", "-m", "fluentfolio", *command]
    out, err = tmp_path / "reply.txt", tmp_path / "lines.txt"
    with out.open("wb") as reply, err.open("wb") as lines:
        proc = runner.attached(argv, {**os.environ, "FLUENTFOLIO_HOME": str(home)}, reply, lines)
        try:
            code = proc.wait(
                60
            )  # its stdin stays open: it ends on its own, with the stdin reader still waiting
        finally:
            runner.stop_attached(proc, STOP_S)
    said = err.read_text(encoding="utf-8", errors="replace")
    assert code == 0 and "Fatal Python error" not in said, said


def test_a_command_started_otherwise_ignores_a_closed_stdin(tmp_path: Path, finished: Job) -> None:
    home, job_id, _ = finished
    argv = [sys.executable, "-I", "-X", "utf8", "-m", "fluentfolio", "status", job_id, "--json"]
    done = subprocess.run(
        argv, env={**os.environ, "FLUENTFOLIO_HOME": str(home)}, stdin=subprocess.DEVNULL,
        capture_output=True, timeout=60, check=False,
    )  # fmt: skip
    assert done.returncode == 0 and json.loads(done.stdout)["data"]["state"] == "completed", done.stderr
