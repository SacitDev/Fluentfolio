"""The private role-swap recordings, each cut where the product finds the swap, run again as two interviews.

A half holds one asker and one answerer, so its interview run must give the roles that the role-swap run
gave the same stretch of the whole recording. Only numbers are recorded and compared, so a failure report
never shows a word or a path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import soundfile as sf

from fluentfolio.app import facade
from fluentfolio.core.enums import JobState
from fluentfolio.core.files import read_json

pytestmark = [pytest.mark.slow, pytest.mark.private_sample]
Row = dict[str, Any]
FINISHED = (JobState.COMPLETED, JobState.COMPLETED_WITH_GAPS)
LEVELS = ("low", "medium", "high")


@dataclass
class Run:
    job: Path
    finished: bool
    words: list[Row]
    answered: set[int]  # indexes of the words in an answer
    structure: Row


def run(path: Path, mode: str, **options: Any) -> Run:
    """Analyse a recording and read what the role check needs from its job."""
    job = facade.analyze(path, mode, partner_consent="cli-flag", **options).job
    structure = read_json(job / "stages" / "structure.json")
    return Run(
        job,
        read_json(job / "job.json")["state"] in FINISHED,
        read_json(job / "stages" / "words.json")["words"],
        {i for u in structure["utterances"] for i in range(u["first_word"], u["last_word"] + 1)},
        structure,
    )


def compare(scoring: ModuleType, whole: Run, half: Run, span: tuple[int, int]) -> tuple[float, float]:
    """Over the whole run's word time in the half's span: the share also found in the half's run, and
    among that time the share on which both runs agree that the word is, or is not, part of an answer.

    A word matches the word of the other run that shares most time with it, as in the scores against a
    composed session's truth.
    """
    start_ms, end_ms = span
    shifted = [
        {"start_ms": w["start_ms"] + start_ms, "end_ms": w["end_ms"] + start_ms, "index": j}
        for j, w in enumerate(half.words)
    ]  # fmt: skip
    total = found = agreed = 0
    for i, w in enumerate(whole.words):
        if not start_ms <= w["start_ms"] < end_ms:
            continue
        ms = w["end_ms"] - w["start_ms"]
        total += ms
        match = scoring.segment_of(w, shifted)
        if match is not None:
            found += ms
            agreed += ms if (i in whole.answered) == (match["index"] in half.answered) else 0
    return (found / total if total else 0.0), (agreed / found if found else 0.0)


@pytest.mark.parametrize("index", [0, 1], ids=["sample1", "sample2"])
def test_each_half_of_a_role_swap_recording_gets_the_same_roles_as_an_interview(
    index: int,
    tmp_path: Path,
    private_recordings: list[Path],
    private_summary: Callable[[str, dict[str, object]], None],
    scoring: ModuleType,
    tolerance: Callable[[str, str], float],
) -> None:
    sample = f"sample{index + 1}"
    whole = run(private_recordings[index], "role-swap", swap_style="once")
    people = whole.structure["candidates"]
    found: Row = {
        "finished": whole.finished, "candidates": len(people),
        "swaps": [len(c["roles"]["swap_times_ms"]) for c in people],
        "confidence": [LEVELS.index(c["roles"]["confidence"]) for c in people],
    }  # fmt: skip
    private_summary("role_swap_halves", {sample: found})
    assert found["finished"]
    assert found["swaps"] == [1, 1]  # two people and one swap, the same for both
    (swap_ms,) = people[0]["roles"]["swap_times_ms"]
    assert people[1]["roles"]["swap_times_ms"] == [swap_ms]

    audio, rate = sf.read(str(whole.job / "input" / "audio16k.wav"), dtype="int16")
    cut = swap_ms * rate // 1000
    duration_ms = len(audio) * 1000 // rate
    # If each run is right on at least the share of speech time the requirement asks of its mode, the two
    # can disagree on at most the sum of what each may get wrong.
    floor = (
        1
        - (1 - tolerance("pipeline", "role_accuracy_role_swap_min"))
        - (1 - tolerance("pipeline", "role_accuracy_interview_min"))
    )
    found.update(swap_share=round(cut / len(audio), 3), floor=round(floor, 4), halves=[])
    halves = [(audio[:cut], (0, swap_ms)), (audio[cut:], (swap_ms, duration_ms))]
    for k, (piece, span) in enumerate(halves):
        path = tmp_path / f"half{k + 1}.wav"
        sf.write(str(path), piece, rate, subtype="PCM_16")
        half = run(path, "interview")
        matched, agreement = compare(scoring, whole, half, span)
        alone = half.structure["candidates"]
        found["halves"].append({
            "duration_s": round(len(piece) / rate, 1), "finished": half.finished, "candidates": len(alone),
            "confidence": [LEVELS.index(c["roles"]["confidence"]) for c in alone],
            "role_change_warning": "W_ROLE_CHANGE" in half.structure["warnings"],
            "matched_share": round(matched, 4), "role_agreement": round(agreement, 4),
        })  # fmt: skip
    private_summary("role_swap_halves", {sample: found})
    for h in found["halves"]:
        assert h["finished"]
        assert h["candidates"] == 1
        assert not h["role_change_warning"]  # one asker and one answerer: the roles never change inside
        assert h["matched_share"] > 0.5  # the agreement speaks for most of the speech only if most matched
        assert h["role_agreement"] >= floor
