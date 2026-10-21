"""Word ownership, the words step: each word goes to the speaker whose turns cover most of it.

Turns may overlap. Ownership uses exclusive turns, where an overlap belongs to the speaker whose turn around
it is longer (the one holding the floor). Words inside an overlap are flagged; they stay in the transcript.
Words heard over background music are flagged too.
"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Any

from fluentfolio.core.enums import Mode, Status
from fluentfolio.core.files import read_json

Turn = tuple[int, int, str]  # start_ms, end_ms, speaker
EDGE_MS = 500  # a word outside every turn goes to the nearest turn this close, flagged edge_assigned
EDGE = "edge_assigned"
OVERLAP = "overlap"
MUSIC = "music"  # heard over background music, so the word is unsure
NO_SPEAKERS = "Speakers could not be told apart, so no word has a speaker and roles are unavailable."


def talk_ms(spans: list[tuple[int, int]]) -> int:
    """Time covered by at least one of the spans."""
    total = reach = 0
    for a, b in sorted(spans):
        if b > reach:
            total += b - max(a, reach)
            reach = b
    return total


def pieces(turns: list[Turn]) -> list[tuple[int, int, list[Turn]]]:
    """The timeline cut at every turn edge: each stretch that some turn covers, with the turns covering it."""
    edges = sorted({t for a, b, _ in turns for t in (a, b)})
    ordered = sorted(turns)
    out: list[tuple[int, int, list[Turn]]] = []
    active: list[Turn] = []
    k = 0
    for a, b in zip(edges, edges[1:], strict=False):
        while k < len(ordered) and ordered[k][0] <= a:
            active.append(ordered[k])
            k += 1
        active = [t for t in active if t[1] > a]
        if active:
            out.append((a, b, list(active)))
    return out


def overlaps(turns: list[Turn]) -> list[tuple[int, int]]:
    """Stretches where two or more speakers talk at once, merged, in time order."""
    found: list[tuple[int, int]] = []
    for a, b, covering in pieces(turns):
        if len({s for _, _, s in covering}) > 1:
            if found and found[-1][1] == a:
                found[-1] = (found[-1][0], b)
            else:
                found.append((a, b))
    return found


def exclusive(turns: list[Turn]) -> list[Turn]:
    """Turns that never overlap: in an overlap, the speaker of the longest covering turn keeps the time."""
    out: list[Turn] = []
    for a, b, covering in pieces(turns):
        speaker = max(covering, key=lambda t: (t[1] - t[0], -t[0]))[2]
        if out and out[-1][2] == speaker and out[-1][1] == a:
            out[-1] = (out[-1][0], b, speaker)
        else:
            out.append((a, b, speaker))
    return out


def _inside(spans: list[tuple[int, int]], starts: list[int], t: float) -> bool:
    i = bisect_right(starts, t) - 1
    return i >= 0 and spans[i][0] <= t < spans[i][1]


def owner(word: dict[str, Any], turns: list[Turn], starts: list[int]) -> tuple[str | None, bool]:
    """The speaker of one word among exclusive turns, and whether it was taken from a nearby turn."""
    s, e = word["start_ms"], word["end_ms"]
    i = max(0, bisect_right(starts, s) - 1)
    best, most = None, 0
    j = i
    while j < len(turns) and turns[j][0] < max(e, s + 1):
        shared = min(e, turns[j][1]) - max(s, turns[j][0])
        if shared > most or (e == s and turns[j][0] <= s < turns[j][1]):
            best, most = turns[j][2], max(shared, 1)
        j += 1
    if best is not None:
        return best, False
    gaps = [(s - turns[k][1], k) for k in (i, i - 1) if 0 <= k < len(turns) and turns[k][1] <= s]
    gaps += [(turns[k][0] - e, k) for k in (j, i + 1) if 0 <= k < len(turns) and turns[k][0] >= e]
    near = [(g, k) for g, k in gaps if g <= EDGE_MS]
    return (turns[min(near)[1]][2], True) if near else (None, False)


def assign(words: list[dict[str, Any]], turns: list[Turn], *, by_track: bool = False) -> None:
    """Set each word's speaker and its edge_assigned and overlap flags, in place."""
    solo = exclusive(turns)
    starts = [a for a, _, _ in solo]
    shared = overlaps(turns)
    shared_starts = [a for a, _ in shared]
    for w in words:
        if by_track:  # one track per person: a word belongs to its own track
            speaker, edge = f"S{w['track']}", False
        else:
            speaker, edge = owner(w, solo, starts)
        w["speaker"] = speaker
        if edge:
            w["flags"].append(EDGE)
        if _inside(shared, shared_starts, (w["start_ms"] + w["end_ms"]) / 2):
            w["flags"].append(OVERLAP)


def run(job: Path) -> dict[str, Any]:
    """The words step: the speech model's words with their speaker, written to stages/words.json."""
    words = [{**w, "flags": list(w["flags"])} for w in read_json(job / "stages" / "asr.json")["words"]]
    path = job / "stages" / "speakers.json"
    found = read_json(path) if path.is_file() else {"status": Status.FAILED}
    status, reason = Status.OK, None
    if read_json(job / "job.json")["mode"] == Mode.SOLO:  # one person, whatever the second-voice check found
        for w in words:
            w["speaker"] = "S1"
    elif found["status"] not in (Status.OK, Status.PARTIAL):
        status, reason = Status.PARTIAL, NO_SPEAKERS
        for w in words:
            w["speaker"] = None
    else:
        turns = [(t["start_ms"], t["end_ms"], t["speaker"]) for t in found["turns"]]
        assign(words, turns, by_track=found["backend"] == "tracks")
    music = [(a, b) for a, b in read_json(job / "stages" / "media.json")["music_spans"]]
    starts = [a for a, _ in music]
    for w in words:
        if _inside(music, starts, (w["start_ms"] + w["end_ms"]) / 2):
            w["flags"].append(MUSIC)
    return {
        "status": status,
        "reason": reason,
        "words": words,
        "counts": {
            "words": len(words),
            "unknown_speaker": sum(w["speaker"] is None for w in words),
            EDGE: sum(EDGE in w["flags"] for w in words),
            OVERLAP: sum(OVERLAP in w["flags"] for w in words),
            MUSIC: sum(MUSIC in w["flags"] for w in words),
        },
    }
