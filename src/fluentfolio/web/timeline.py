"""The Results page's timeline strip: Parts, each person's speech and the long pauses on one time axis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

from fluentfolio.analyzers.measures.fluency import pauses
from fluentfolio.web.render import clock, length_text, upfirst

UNIT_MS = 100  # viewBox units of 0.1 s: Python works out the shapes, the partial draws them with classes only
MERGE_MS = 300  # one person's speech closer than this is drawn as one bar
LONG_PAUSE_MS = 1000
MOST_BARS = 400  # per lane: the strip stays near 1,000 shapes


class Lane(NamedTuple):
    name: str
    style: str  # the CSS class of its shapes: spk-1, spk-2, read, part or pause
    bars: Sequence[tuple[int, int]] = ()  # x and width
    ticks: Sequence[int] = ()


def _bars(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Speech closer than the gap is one bar; a very long session doubles the gap until the bars are few."""
    gap, merged = MERGE_MS, sorted(spans)
    while True:
        out: list[tuple[int, int]] = []
        for a, b in merged:
            if out and a - out[-1][1] < gap:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        if len(out) <= MOST_BARS:  # the bars still cover all the speech
            return [(a // UNIT_MS, max(1, (b - a) // UNIT_MS)) for a, b in out]
        gap, merged = gap * 2, out


def _voiced(turns: list[dict[str, Any]], words: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """The words said in these turns, not their whole spans: a turn can hold the minute to prepare Part 2."""
    spans = [(t["start_ms"], t["end_ms"]) for t in turns]  # a word belongs to the turn that holds its middle
    middles = [(w["start_ms"], w["end_ms"], (w["start_ms"] + w["end_ms"]) // 2) for w in words]
    return [(a, b) for a, b, m in middles if any(start <= m <= end for start, end in spans)]


def build(
    evidence: dict[str, Any], speech: dict[str, list[list[int]]], words: list[dict[str, Any]]
) -> dict[str, Any]:
    """The lanes, caption and per-Part table of a finished job, from its detected speech and timed words."""
    people, turns = evidence["candidates"], evidence["examiner_turns"]
    duration = evidence["manifest"]["audio"]["duration_ms"]
    parts = sorted(  # in role-swap each person has their own Parts; a table row then says whose Part it is
        ((f"{upfirst(c['label'])}, " if len(people) > 1 else "", p) for c in people for p in c["parts"]),
        key=lambda found: found[1]["start_ms"],
    )
    spans = [(p["start_ms"], p["end_ms"]) for _, p in parts]
    lanes = [Lane("Parts", "part", [(a // UNIT_MS, max(1, (b - a) // UNIT_MS)) for a, b in spans])]
    for k, person in enumerate(people):
        spans = [(u["start_ms"], u["end_ms"]) for u in person["utterances"]]
        if len(people) > 1:  # in role-swap each person also asks the other
            spans += _voiced([t for t in turns if t["addressed_to"] != person["id"]], words)
        lanes.append(Lane(upfirst(person["label"]), f"spk-{k + 1}", _bars(spans)))
    asked = _voiced([t for t in turns if t["kind"] != "read_aloud"], words)
    read = _voiced([t for t in turns if t["kind"] == "read_aloud"], words)
    if len(people) == 1 and asked:
        lanes.append(Lane("The examiner", "spk-2", _bars(asked)))
    if read:
        lanes.append(Lane("Questions read aloud", "read", _bars(read)))
    long: list[tuple[int, int]] = []
    for p in people:  # from the voice detector's speech, so these are the pause measures' own pauses
        mine, heard = [(a["start_ms"], a["end_ms"]) for a in p["answers"]], list(map(tuple, speech[p["id"]]))
        long += [(a, b) for a, b in pauses(mine, heard) if b - a >= LONG_PAUSE_MS]
    long.sort()
    lanes.append(Lane("Pauses of 1 s or more", "pause", ticks=[(a + b) // 2 // UNIT_MS for a, b in long]))
    caption = f"Whole recording, {length_text(duration)}."
    for whose in dict.fromkeys(w for w, _ in parts):  # the strip has no Part labels: the caption names them
        found = enumerate(p for w, p in parts if w == whose and p["part"] != "unknown")
        named = [f"Part {p['part']} {'at' if k else 'starts at'} {clock(p['start_ms'])}" for k, p in found]
        caption += f" {whose.replace(', ', ': ')}{', '.join(named)}." if named else ""
    caption += f" {len(long)} pause{'' if len(long) == 1 else 's'} of 1 s or more."
    table = [
        (f"{whose}Part {p['part']}", clock(p["start_ms"]), clock(p["end_ms"]),
         sum(p["start_ms"] <= (a + b) // 2 < p["end_ms"] for a, b in long))
        for whose, p in parts if p["part"] != "unknown"
    ]  # fmt: skip
    return {"width": max(1, duration // UNIT_MS), "lanes": lanes, "caption": caption, "table": table,
            "end": clock(duration)}  # fmt: skip
