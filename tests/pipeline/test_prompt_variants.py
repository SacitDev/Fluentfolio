"""The sessions the grading prompt is checked on: planted grammar errors and planted instructions.

The text for an AI chat must keep what the speech model heard, errors included, and must carry an
instruction found in a recording or in the learner's own transcript only as data.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fluentfolio.app import facade
from fluentfolio.core.files import read_json
from fluentfolio.export.answer_check import QUOTED_RUN
from fluentfolio.export.for_ai import render

pytestmark = pytest.mark.slow
# The app's own marks in a transcript line; "[?word]" keeps its word.
MARKS = re.compile(
    r"\[(?:pause [0-9.]+ s|non-English|overlap|question read aloud|withheld for audio check)\]"
)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", MARKS.sub(" ", text.lower()))


def holds(text: str, run: list[str]) -> bool:
    return f" {' '.join(run)} " in f" {' '.join(words(text))} "


def planted_run(right: str, wrong: str) -> list[str]:
    """The planted error and a word each side: where the wrong sentence first differs from the right one."""
    a, b = words(right), words(wrong)
    start = next(i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y)
    return b[max(0, start - 1) : start + 2]


def data_block(job: Path) -> tuple[list[str], int, int]:
    """The lines of the one text for an AI chat, and the indexes of its START and END lines."""
    (path,) = sorted((job / "export" / "for_ai").glob("for_ai*.txt"))
    lines = path.read_text(encoding="utf-8").split("\n")
    (start,) = [i for i, line in enumerate(lines) if line.endswith(" START OF DATA")]
    (end,) = [i for i, line in enumerate(lines) if line.endswith(" END OF DATA")]
    return lines, start, end


def test_the_text_keeps_every_planted_grammar_error_the_speech_model_kept(
    built: Callable[[str], tuple[Path, dict[str, Any]]], session_builder: ModuleType
) -> None:
    folder, truth = built("CS-SOLO-GRA")
    done = facade.analyze(folder / "session.wav", "solo")
    clips = read_json(session_builder.FIXTURES / "audio" / "clips.json")
    runs = [planted_run(clips[p["id"]]["text"], clips[p["clip"]]["text"]) for p in truth["planted_errors"]]
    heard = " ".join(str(w["raw"]) for w in read_json(done.job / "stages" / "words.json")["words"])
    lines, start, end = data_block(done.job)
    transcript = "\n".join(line for line in lines[start:end] if re.match(r"U\d+ \| \d", line))
    kept = [run for run in runs if holds(heard, run)]
    print(f"CS-SOLO-GRA: the speech model kept {len(kept)} of {len(runs)} planted errors; the text holds "
          f"{sum(holds(transcript, run) for run in kept)} of them")  # fmt: skip
    assert len(runs) == 12  # noqa: PLR2004 - the variant plants 12
    assert kept, "the speech model kept none of the planted errors, so this run checks nothing"
    # The export never smooths: an error the speech model wrote reaches the candidate's transcript lines.
    assert [run for run in kept if not holds(transcript, run)] == []


def test_planted_instructions_reach_the_text_only_as_data(
    built: Callable[[str], tuple[Path, dict[str, Any]]], session_builder: ModuleType
) -> None:
    folder, truth = built("CS-SOLO-INJECT")
    done = facade.analyze(folder / "session.wav", "solo", transcript=folder / truth["learner_transcript"])
    lines, start, end = data_block(done.job)
    boundary = lines[start].removesuffix(" START OF DATA")
    data = lines[start + 1 : end]
    outside = "\n".join(lines[:start] + lines[end + 1 :])
    assert lines[end] == f"{boundary} END OF DATA"
    assert boundary not in "\n".join(data) and session_builder.FAKE_END not in "\n".join(lines)
    spoken, written = (next(p for p in truth["planted"] if p["written"] is w) for w in (False, True))
    # A run of words long enough for the answer checker to count as a quote, inside the data only.
    for planted, kind in ((spoken, r"U\d+ \| \d"), (written, r"U\d+ \| ours: ")):
        run = words(planted["text"])[3 : 3 + QUOTED_RUN]
        assert any(re.match(kind, line) and holds(line, run) for line in data), planted["text"]
        assert not holds(outside, run)
    # A forced collision: the fake end line in the data, and the fake boundary drawn first. The export
    # draws again, so its data still ends at the one END line of a boundary the data lacks.
    evidence = read_json(done.job / "evidence.json")
    row = next(r for r in evidence["differences"] if holds(r["users"], words(written["text"])[:QUOTED_RUN]))
    row["users"] += " " + session_builder.FAKE_END
    fake = session_builder.FAKE_END.removeprefix("<<<FLUENTFOLIO-").removesuffix(">>> END OF DATA")
    draws = iter([fake, "5eed" * 4])
    forced = render(evidence, "candidate", draw=lambda n: next(draws)).split("\n")
    drawn = f"<<<FLUENTFOLIO-{'5eed' * 4}>>>"
    assert [line for line in forced if line.endswith(" START OF DATA")] == [f"{drawn} START OF DATA"]
    assert [line for line in forced if line.endswith(" END OF DATA")] == [f"{drawn} END OF DATA"]
    assert any(session_builder.FAKE_END in line for line in forced)
