"""`fluentfolio check-answer`: one line per rule, the audio check's verdict, and its exit codes."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from fluentfolio.core.errors import catalogue
from fluentfolio.core.files import write_json

SEGMENTS = [(1000, 4000), (5200, 9000)]
SPOKEN = [
    ("I", 1000, 1200, 0), ("like", 1250, 1500, 0), ("my", 1550, 1700, 0), ("home", 1750, 2100, 0),
    ("town", 2150, 2500, 0), ("because", 2550, 3000, 0), ("it", 3050, 3200, 0), ("is", 3250, 3400, 0),
    ("quiet.", 3450, 3950, 0), ("The", 5200, 5400, 1), ("people", 5450, 5900, 1), ("are", 5950, 6100, 1),
    ("kind", 6150, 6500, 1), ("and", 6550, 6700, 1), ("the", 6750, 6850, 1), ("parks", 6900, 7400, 1),
    ("are", 7450, 7600, 1), ("green.", 7650, 8900, 1),
]  # fmt: skip
# One piece: the audio file is the recording from 1.0 s on. The span is 1.0-9.0 s with all 18 words, of which
# 7 are content words: home, town, quiet, people, kind, parks and green ("like" is on the function-word list).
PIECES = [{"orig_ms": 1000, "audio_ms": 0, "ms": 8000}]
HIDDEN = "I like my home town because it is quiet. The people are kind and the parks are green."


def cli(*args: str, home: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "FLUENTFOLIO_HOME": str(home)}
    return subprocess.run(
        [sys.executable, "-m", "fluentfolio", "check-answer", *args], capture_output=True, text=True,
        encoding="utf-8", env=env, timeout=120, check=False,
    )  # fmt: skip


def finished_job(tmp_path: Path, synthetic_job: ModuleType) -> str:
    job = synthetic_job.make(tmp_path, SPOKEN, SEGMENTS, 12_000)
    made = {"candidate": {"file": "for_ai_audio.m4a", "duration_ms": 8000, "pieces": PIECES}}
    write_json(job / "stages" / "ai_audio.json", {"status": "ok", "candidates": made})
    synthetic_job.finish(tmp_path)
    return job.name


def test_a_wrong_audio_check_says_to_ignore_pronunciation_and_then_shows_the_words(
    tmp_path: Path, synthetic_job: ModuleType
) -> None:
    job_id = finished_job(tmp_path, synthetic_job)
    answer = tmp_path / "answer.txt"
    step0 = "audio_access: native\naudio_check: the weather was cold\ninputs: for_ai.txt\n"
    answer.write_text(step0, encoding="utf-8")
    done = cli(job_id, str(answer), home=tmp_path / "data")
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert "AC-11 audio check: FAIL (0 of 7 hidden content words heard)" in lines
    assert (
        "Ignore Pronunciation and the overall estimate from this answer: the chat did not hear the audio. "
        "Ask again without the audio file, so it estimates Pronunciation from the clues in the text."
    ) in lines
    assert lines[-1] == f"The hidden words of the audio check: {HIDDEN}"
    assert len([line for line in lines if line.startswith("AC-")]) == 12


@pytest.mark.parametrize("audio_made", [False, True])
def test_the_audio_advice_comes_only_when_a_file_was_made_and_the_answer_has_its_check_line(
    tmp_path: Path, synthetic_job: ModuleType, audio_made: bool
) -> None:
    # Without an audio file, or without an audio_check line, nothing says the chat missed the audio.
    if audio_made:
        job_id, step0 = finished_job(tmp_path, synthetic_job), "audio_access: native\ninputs: for_ai.txt\n"
    else:
        job_id = synthetic_job.make(tmp_path, SPOKEN, SEGMENTS, 12_000).name
        synthetic_job.finish(tmp_path)
        step0 = "audio_access: native\naudio_check: the weather was cold\ninputs: for_ai.txt\n"
    answer = tmp_path / "answer.txt"
    answer.write_text(step0, encoding="utf-8")
    done = cli(job_id, str(answer), home=tmp_path / "data")
    assert done.returncode == 0, done.stderr
    assert next(line for line in done.stdout.splitlines() if line.startswith("AC-11 ")).startswith(
        "AC-11 audio check: FAIL"
    )
    assert "did not hear the audio" not in done.stdout


def test_the_json_envelope_has_every_rule_and_no_hidden_word(
    tmp_path: Path, synthetic_job: ModuleType
) -> None:
    job_id = finished_job(tmp_path, synthetic_job)
    answer = tmp_path / "answer.txt"
    heard = "I like my home town because it is quiet the people are kind"  # 5 of the 7: 71%, over 60%
    answer.write_text(f"audio_access: native\naudio_check: {heard}\ninputs: for_ai.txt\n", encoding="utf-8")
    done = cli(job_id, str(answer), "--json", home=tmp_path / "data")
    assert done.returncode == 0 and done.stdout.isascii() and done.stdout.count("\n") == 1, done.stderr
    envelope = json.loads(done.stdout)
    assert (envelope["command"], envelope["ok"], envelope["error"]) == ("check-answer", True, None)
    results = {r["rule"]: r for r in envelope["data"]["results"]}
    assert len(results) == 12 and results["AC-11"]["verdict"] == "PASS"
    assert results["AC-11"]["detail"] == "5 of 7 hidden content words heard"
    assert "parks" not in done.stdout and "green" not in done.stdout  # heard by no one, shown nowhere


def test_an_unknown_job_or_a_word_document_is_an_input_error(
    tmp_path: Path, synthetic_job: ModuleType
) -> None:
    answer = tmp_path / "answer.docx"
    answer.write_bytes(b"PK\x03\x04")
    done = cli("20261013-120000-ffff", str(answer), "--json", home=tmp_path / "data")
    envelope = json.loads(done.stdout)
    assert done.returncode == catalogue()["E_JOB_NOT_FOUND"].exit_code == 4
    assert envelope["error"]["code"] == "E_JOB_NOT_FOUND"
    job_id = finished_job(tmp_path, synthetic_job)
    done = cli(job_id, str(answer), home=tmp_path / "data")
    assert done.returncode == 4 and done.stderr.startswith("E_INPUT_TEXT:")
