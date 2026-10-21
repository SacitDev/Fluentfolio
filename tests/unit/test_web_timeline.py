"""The timeline strip: known answers from small hand-made evidence, and the drawn strip of two saved runs.

The golden files only catch drift. After a deliberate change, read the new strip, then rewrite it with
`pytest --update-goldens -k strip_golden`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from fluentfolio.analyzers.measures.fluency import timing
from fluentfolio.core.files import read_json
from fluentfolio.web import timeline
from fluentfolio.web.render import TEMPLATES

TESTS = Path(__file__).resolve().parents[1]
FIXTURES = TESTS / "fixtures" / "evidence"
GOLDEN = TESTS / "golden"


def turn(start: int, end: int, kind: str = "question", to: str = "candidate") -> dict[str, Any]:
    return {"start_ms": start, "end_ms": end, "kind": kind, "addressed_to": to}


def person(
    pid: str, label: str, utterances: list[tuple[int, int]], parts: list[tuple[Any, int, int]]
) -> dict[str, Any]:
    return {
        "id": pid, "label": label,
        "utterances": [{"start_ms": a, "end_ms": b} for a, b in utterances],
        "answers": [{"start_ms": a, "end_ms": b} for a, b in utterances],
        "parts": [{"part": p, "start_ms": a, "end_ms": b} for p, a, b in parts],
    }  # fmt: skip


def evidence_of(people: list[dict[str, Any]], turns: list[dict[str, Any]], duration: int) -> dict[str, Any]:
    return {"candidates": people, "examiner_turns": turns, "manifest": {"audio": {"duration_ms": duration}}}


def words_at(*spans: tuple[int, int]) -> list[dict[str, Any]]:
    return [{"start_ms": a, "end_ms": b} for a, b in spans]


def turn_words(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """The saved evidence holds no words: each examiner turn stands in as one word."""
    return words_at(*((t["start_ms"], t["end_ms"]) for t in evidence["examiner_turns"]))


def drawn(strip: dict[str, Any]) -> str:
    return TEMPLATES.get_template("partials/timeline.html").render(strip=strip)


def test_the_pauses_parts_and_speech_are_where_the_evidence_puts_them() -> None:
    # A 20 s interview: Part 1 to 8 s, then Part 2. The answers are 1-7 s and 9-19 s.
    candidate = person(
        "candidate", "the candidate", [(1000, 7000), (9000, 19000)], [(1, 0, 8000), (2, 8000, 20000)]
    )
    evidence = evidence_of([candidate], [turn(0, 900), turn(7500, 8800)], 20000)
    # Silences inside the answers: 2.0-3.5 s (1.5 s), 12.0-12.1 s (0.1 s, too short to be a pause),
    # 15.0-16.2 s (1.2 s). So two pauses of 1 s or more, their middles at 2.75 s and 15.6 s.
    speech = [[1000, 2000], [3500, 7000], [9000, 12000], [12100, 15000], [16200, 19000]]
    strip = timeline.build(evidence, {"candidate": speech}, words_at((0, 900), (7500, 8800)))
    lanes = {lane.name: lane for lane in strip["lanes"]}
    assert list(lanes) == ["Parts", "The candidate", "The examiner", "Pauses of 1 s or more"]
    assert strip["width"] == 200  # viewBox units of 0.1 s
    assert lanes["Parts"].bars == [(0, 80), (80, 120)]
    assert lanes["The candidate"].bars == [(10, 60), (90, 100)]
    assert lanes["The examiner"].bars == [(0, 9), (75, 13)]
    assert lanes["Pauses of 1 s or more"].ticks == [27, 156]
    assert strip["caption"] == (
        "Whole recording, 20.0 s. Part 1 starts at 0:00, Part 2 at 0:08. 2 pauses of 1 s or more."
    )
    assert strip["table"] == [("Part 1", "0:00", "0:08", 1), ("Part 2", "0:08", "0:20", 1)]
    # The same count as the pause measure of the same answers and speech.
    measured = timing([(1000, 7000), (9000, 19000)], [tuple(s) for s in speech], [])  # type: ignore[misc]
    assert measured["pauses_1s"].value == 2


def test_in_role_swap_each_lane_holds_what_that_person_said_both_asking_and_answering() -> None:
    # Person 1 answers in the first half (Part 1 from 1 s, Part 2 from 5 s), Person 2 in the second half.
    one = person("person1", "Person 1", [(2000, 9000)], [(1, 1000, 5000), (2, 5000, 10000)])
    two = person("person2", "Person 2", [(12000, 19000)], [(1, 11000, 15000), (2, 15000, 20000)])
    turns = [turn(1000, 1900, to="person1"), turn(11000, 11900, to="person2")]  # each asks the other
    evidence = evidence_of([one, two], turns, 20000)
    strip = timeline.build(evidence, {"person1": [], "person2": []}, turn_words(evidence))
    lanes = {lane.name: (lane.style, lane.bars) for lane in strip["lanes"]}
    assert lanes["Parts"] == ("part", [(10, 40), (50, 50), (110, 40), (150, 50)])  # both people's Parts
    assert lanes["Person 1"] == ("spk-1", [(20, 70), (110, 9)])  # answers at 2 s, asks at 11 s
    assert lanes["Person 2"] == ("spk-2", [(10, 9), (120, 70)])
    assert "The examiner" not in lanes
    assert strip["caption"] == (
        "Whole recording, 20.0 s. Person 1: Part 1 starts at 0:01, Part 2 at 0:05. "
        "Person 2: Part 1 starts at 0:11, Part 2 at 0:15. 0 pauses of 1 s or more."
    )
    assert [row[:3] for row in strip["table"]] == [
        ("Person 1, Part 1", "0:01", "0:05"), ("Person 1, Part 2", "0:05", "0:10"),
        ("Person 2, Part 1", "0:11", "0:15"), ("Person 2, Part 2", "0:15", "0:20"),
    ]  # fmt: skip


def test_a_turn_that_holds_the_minute_to_prepare_part_2_is_drawn_only_where_someone_speaks() -> None:
    # Interview, 130 s. The cue-card turn runs 10-75.5 s: the card is read at 10-14 s, then a minute of
    # silence to prepare, then "Please start speaking now" at 74-75.5 s. The candidate's own words lie
    # outside every turn and stay off the examiner's lane.
    candidate = person(
        "candidate", "the candidate", [(2000, 9000), (76000, 120000)], [(1, 0, 10000), (2, 10000, 130000)]
    )
    turns = [turn(0, 1800), turn(10000, 75500, "cue_card")]
    spoken = words_at((0, 800), (900, 1800), (2000, 2500), (10000, 10400), (10500, 14000), (74000, 74600),
                      (74700, 75500), (76000, 76500))  # fmt: skip
    strip = timeline.build(evidence_of([candidate], turns, 130000), {"candidate": []}, spoken)
    lanes = {lane.name: lane.bars for lane in strip["lanes"]}
    assert lanes["The examiner"] == [(0, 18), (100, 40), (740, 15)]  # not one bar from 10 s to 75.5 s
    # Role-swap, 150 s: Person 2 gives Person 1 the card at 3-70 s (read at 3-6 s, the last words at 69-70 s),
    # and Person 1 asks Person 2 at 115-119 s.
    one = person("person1", "Person 1", [(70500, 110000)], [(2, 3000, 110000)])
    two = person("person2", "Person 2", [(120000, 150000)], [(1, 115000, 150000)])
    turns = [turn(3000, 70000, "cue_card", to="person1"), turn(115000, 119000, to="person2")]
    spoken = words_at((3000, 6000), (69000, 70000), (115000, 119000))
    strip = timeline.build(evidence_of([one, two], turns, 150000), {"person1": [], "person2": []}, spoken)
    lanes = {lane.name: lane.bars for lane in strip["lanes"]}
    assert lanes["Person 1"] == [(705, 395), (1150, 40)]
    assert lanes["Person 2"] == [(30, 30), (690, 10), (1200, 300)]


def test_a_long_session_is_drawn_with_fewer_bars_that_still_cover_all_the_speech() -> None:
    # Two hours of 1 s answers with 0.5 s gaps: 4,800 answers would be 4,800 bars at the usual merging.
    spans = [(k * 1500, k * 1500 + 1000) for k in range(4800)]
    candidate = person("candidate", "the candidate", spans, [])
    strip = timeline.build(evidence_of([candidate], [], 7200000), {"candidate": []}, [])
    bars = strip["lanes"][1].bars
    assert len(bars) <= timeline.MOST_BARS
    assert bars[0][0] == 0 and bars[-1][0] + bars[-1][1] == (4799 * 1500 + 1000) // 100


def test_the_strip_is_classes_only_with_no_script_and_its_words_beside_it() -> None:
    evidence = read_json(FIXTURES / "role_swap.json")
    speech = {
        c["id"]: [[u["start_ms"], u["end_ms"]] for u in c["utterances"]] for c in evidence["candidates"]
    }
    html = drawn(timeline.build(evidence, speech, turn_words(evidence)))
    assert not re.search(r"<script|\sstyle=|\son[a-z]+=", html)  # allowed by the page's strict CSP
    assert 'aria-hidden="true"' in html and '<figcaption id="strip-caption">' in html
    assert "Show as table" in html and "Person 1" in html and "Person 2" in html


@pytest.mark.parametrize("name", ["solo", "role_swap"])
def test_strip_golden(update_goldens: bool, name: str) -> None:
    evidence = read_json(FIXTURES / f"{name}.json")
    # Each person's speech is taken as their utterances, as the saved evidence holds no voice detector output.
    speech = {
        c["id"]: [[u["start_ms"], u["end_ms"]] for u in c["utterances"]] for c in evidence["candidates"]
    }
    html = drawn(timeline.build(evidence, speech, turn_words(evidence))).encode("utf-8")
    golden = GOLDEN / f"timeline_{name}.html"
    if update_goldens:
        golden.write_bytes(html)
    assert golden.read_bytes() == html
