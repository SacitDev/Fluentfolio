"""Run fluentfolio in a real process, read its one JSON object, and make finished jobs to point it at."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import jsonschema
import pytest

from fluentfolio.core.enums import JobState
from fluentfolio.ingest import secondary
from tests.support import cli_units

SCHEMA = Path(__file__).resolve().parents[2] / "docs" / "schemas" / "cli-v1.schema.json"
CANARY = "zorblat"  # a made-up word: it is in the job's transcript and file name, and nowhere else
SEGMENTS = [(1000, 4000), (5200, 9000), (9400, 14000)]
SPOKEN = [
    ("Well,", 1000, 1300, 0), ("I", 1350, 1450, 0), ("think", 1500, 1800, 0), ("um", 1850, 2150, 0),
    ("reading", 2200, 2700, 0), ("is", 2750, 2900, 0), ("really", 2950, 3400, 0), ("useful.", 3450, 3950, 0),
    ("Because", 5200, 5600, 1), ("it", 5650, 5800, 1), ("helps", 6050, 6500, 1), ("you", 6550, 6700, 1),
    ("learn", 6750, 7100, 1), ("new", 7150, 7400, 1), ("words,", 7450, 8900, 1), ("and", 9400, 9600, 1),
    ("also", 9650, 9900, 1), ("ideas", 9950, 10500, 1), ("about", 10550, 10800, 1), ("the", 10850, 10950, 1),
    (f"{CANARY}", 11000, 11600, 1), ("world.", 11700, 12300, 1), ("So", 12400, 12600, 1),
    ("yeah.", 12650, 13900, 1),
]  # fmt: skip
OWN = (
    "Well, I think reading is really useful.\n"
    f"Because it helps you learn new words and ideas about {CANARY}.\n"
)


@pytest.fixture(scope="session")
def validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))


@pytest.fixture
def cli(tmp_path: Path) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    """fluentfolio in its own process, with stdin closed unless a test passes its own."""

    def run(
        *args: object, home: Path | None = None, env: dict[str, str] | None = None, timeout: float = 60,
        stdin: Any = subprocess.DEVNULL,
    ) -> subprocess.CompletedProcess[bytes]:  # fmt: skip
        environ = {**os.environ, **(env or {})}
        if home is not None:
            environ["FLUENTFOLIO_HOME"] = str(home)
        argv = [sys.executable, "-m", "fluentfolio", *map(str, args)]
        return subprocess.run(
            argv, cwd=tmp_path, env=environ, stdin=stdin, capture_output=True, timeout=timeout, check=False
        )

    return run


@pytest.fixture
def envelope(
    validator: jsonschema.Draft202012Validator,
) -> Callable[[subprocess.CompletedProcess[bytes]], Any]:
    """The one ASCII line of JSON on stdout, checked against the committed schema and the exit code."""

    def read(done: subprocess.CompletedProcess[bytes]) -> Any:
        assert done.stdout.isascii() and done.stdout.count(b"\n") == 1, (done.stdout, done.stderr)
        found = json.loads(done.stdout)
        validator.validate(found)
        cli_units.check(found)
        assert found["exit_code"] == done.returncode, done.stderr
        return found

    return read


@pytest.fixture
def finished(tmp_path: Path, synthetic_job: ModuleType) -> tuple[Path, str, Path]:
    """A finished solo job with the learner's own transcript: its data folder, ID and folder."""
    job = synthetic_job.make(
        tmp_path / "base", SPOKEN, SEGMENTS, 20_000, display_name=f"{CANARY}_practice.wav"
    )
    own = tmp_path / "own.txt"
    own.write_text(OWN, encoding="utf-8")
    secondary.store_transcript(job, own)
    assert synthetic_job.finish(tmp_path / "base").state is JobState.COMPLETED
    return tmp_path / "base" / "data", job.name, job
