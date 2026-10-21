"""The test stand-in for the model steps: off unless FF_TEST_STUB=1, and realistic for the real steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fluentfolio import __version__
from fluentfolio.app import facade
from fluentfolio.core.enums import Layout, Mode
from fluentfolio.core.files import read_json, write_json
from fluentfolio.jobs import store, stub
from fluentfolio.speakers import ownership
from fluentfolio.structure import build as structure

STAMP = "2026-10-15T12:00:00Z"
MODEL_STEPS = ("vad", "language", "asr", "speakers")


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes", " 1", "1 "])
def test_without_ff_test_stub_set_to_1_every_plan_is_the_real_one(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(stub.SWITCH, raising=False)
    else:
        monkeypatch.setenv(stub.SWITCH, value)
    stand_ins = set(stub.STEPS.values())
    for mode in Mode:
        for layout in Layout:
            steps = facade.plan(mode, layout)
            assert not [s.id for s in steps if s.inline in stand_ins], (mode, layout)
            by_id = {s.id: s for s in steps}
            assert by_id["asr"].inline is None and by_id["vad"].inline is None, (mode, layout)


def test_with_ff_test_stub_1_only_the_model_steps_are_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(stub.SWITCH, raising=False)
    real = {s.id: s for s in facade.plan(Mode.INTERVIEW)}
    monkeypatch.setenv(stub.SWITCH, "1")
    steps = facade.plan(Mode.INTERVIEW)
    assert [s.id for s in steps] == list(real)
    for s in steps:
        if s.id in MODEL_STEPS:
            assert s.inline is stub.STEPS[s.id] and not s.gpu and not s.child and not s.fallback
        else:
            assert s == real[s.id]
    tracks = {s.id: s for s in facade.plan(Mode.INTERVIEW, Layout.PER_TRACK)}
    assert tracks["speakers"].inline is not stub.speakers  # one track per person keeps its own step


def stub_job(base: Path, mode: str, duration_ms: int, layout: str = Layout.MIXED) -> Path:
    """A job whose media step found a recording this long, with the stand-in steps and the real words and
    structure steps run on it."""
    record = store.JobRecord(
        job_id="20261015-120000-abcd", created_utc=STAMP, updated_utc=STAMP, app_version=__version__,
        mode=mode, layout=layout, asr="parakeet-tdt-0.6b-v2",
    )  # fmt: skip
    job = store.create(base, record)
    write_json(job / "stages" / "media.json", {"status": "ok", "duration_ms": duration_ms, "music_spans": []})
    for name, step in stub.STEPS.items():
        write_json(job / "stages" / f"{name}.json", step(job))
    write_json(job / "stages" / "words.json", ownership.run(job))
    write_json(job / "stages" / "structure.json", structure.run(job))
    return job


@pytest.fixture(autouse=True)
def no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stub, "WAIT_S", 0.0)


def found(job: Path, name: str) -> dict[str, Any]:
    result: dict[str, Any] = read_json(job / "stages" / f"{name}.json")
    return result


def test_every_stand_in_result_says_it_is_a_stub(tmp_path: Path) -> None:
    job = stub_job(tmp_path, Mode.INTERVIEW, 400_000)
    assert all(found(job, name)["stub"] is True for name in MODEL_STEPS)
    assert (
        found(job, "speakers")["version"] == "stub" and found(job, "asr")["provenance"]["revision"] == "stub"
    )


def test_the_first_question_and_answer_have_the_hand_worked_times(tmp_path: Path) -> None:
    job = stub_job(tmp_path, Mode.INTERVIEW, 400_000)
    words = found(job, "asr")["words"]
    # 700 ms of silence, then 400 ms a word (320 spoken), 200 ms after "town." instead of 80:
    # "Let's" 700-1020, ..., "town." 2700-3020, "Where" 3220-3540, ..., "from?" 4820-5140.
    assert [(w["raw"], w["start_ms"], w["end_ms"]) for w in words[:7]] == [
        ("Let's", 700, 1020), ("talk", 1100, 1420), ("about", 1500, 1820), ("your", 1900, 2220),
        ("home", 2300, 2620), ("town.", 2700, 3020), ("Where", 3220, 3540),
    ]  # fmt: skip
    assert (words[10]["raw"], words[10]["end_ms"]) == ("from?", 5140)
    # The answer starts 700 ms later; its 9th word, "coast.", ends at 5840 + 8 * 400 + 320 = 9360, and the
    # marked pause of 1.2 s puts the next word at 10560.
    turns = found(job, "speakers")["turns"]
    assert [(t["start_ms"], t["end_ms"], t["speaker"]) for t in turns[:3]] == [
        (700, 5140, "S1"),
        (5840, 9360, "S2"),
        (10560, turns[2]["end_ms"], "S2"),
    ]
    assert found(job, "vad")["segments"][:2] == [[700, 5140], [5840, 9360]]


def parts_of(result: dict[str, Any], who: str) -> list[int]:
    return [p["part"] for p in result["parts"] if p["candidate"] == who]


@pytest.mark.parametrize("mode", [Mode.SOLO, Mode.INTERVIEW])
def test_the_real_structure_step_finds_the_candidate_and_the_three_parts(tmp_path: Path, mode: str) -> None:
    job = stub_job(tmp_path, mode, 600_000)
    result = found(job, "structure")
    (candidate,) = result["candidates"]
    # In the script the examiner (A) speaks first; in solo the one voice reads the questions aloud.
    assert result["status"] == "ok" and candidate["speaker"] == ("S1" if mode == Mode.SOLO else "S2")
    assert parts_of(result, "candidate") == [1, 2, 3]
    # Every question found is text of an examiner line, and every examiner line gives one; in solo the
    # topic card read aloud may count as two, as its "one minute to prepare" sentence splits it.
    words = found(job, "asr")["words"]
    asked = [
        " ".join(w["raw"] for w in words if q["start_ms"] <= w["start_ms"] and w["end_ms"] <= q["end_ms"])
        for q in result["questions"]
    ]
    examiner = [" ".join(t for t in text.split() if t != "|") for who, text in stub.SCRIPT if who == "A"]
    assert all(any(q in line for line in examiner) for q in asked), asked
    assert all(any(q in line for q in asked) for line in examiner), asked


def test_the_real_structure_step_finds_the_role_swap_where_the_script_swaps(tmp_path: Path) -> None:
    job = stub_job(tmp_path, Mode.ROLE_SWAP, 600_000)
    result = found(job, "structure")
    words = found(job, "asr")["words"]
    second_half = words[sum(len(text.split()) - text.split().count("|") for _, text in stub.SCRIPT)]
    assert second_half["raw"] == "Let's"  # the first word of the script's second run
    by_id = {c["id"]: c for c in result["candidates"]}
    assert (by_id["person1"]["speaker"], by_id["person2"]["speaker"]) == ("S2", "S1")
    assert by_id["person1"]["roles"]["swap_times_ms"] == [second_half["start_ms"]]
    assert parts_of(result, "person1") == [1, 2, 3] and parts_of(result, "person2") == [1, 2, 3]


def test_a_recording_shorter_than_the_script_keeps_only_what_fits(tmp_path: Path) -> None:
    job = stub_job(tmp_path, Mode.SOLO, 20_000)
    assert max(w["end_ms"] for w in found(job, "asr")["words"]) <= 20_000
    assert found(job, "vad")["segments"][-1][1] <= 20_000
    assert found(job, "speakers")["second_voice"] == {"checked_by": stub.DIARIZER, "voices": 1, "share": 0.0}
