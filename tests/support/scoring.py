"""Scores against a composed session's truth: speaker separation error, and roles and Parts by word time.

The truth comes from the session builder alone: a word's truth is the segment it shares most time with.
The scores use no product code, so they can catch the product's mistakes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

Row = dict[str, Any]
COLLAR_S = 0.5  # pyannote.metrics takes the whole collar: 0.25 s on each side of a boundary
NO_CONSENT = "E_OPENSMILE_CONSENT_MISSING"


def der(truth: Row, turns: list[Row]) -> float:
    """Diarization error rate against the builder's truth, overlap scored, with a 0.25 s collar."""
    reference, hypothesis = Annotation(), Annotation()
    for i, s in enumerate(truth["segments"]):
        reference[Segment(s["start_ms"] / 1000, s["end_ms"] / 1000), i] = s["speaker"]
    for i, t in enumerate(turns):
        hypothesis[Segment(t["start_ms"] / 1000, t["end_ms"] / 1000), i] = t["speaker"]
    return float(DiarizationErrorRate(collar=COLLAR_S)(reference, hypothesis))


def segment_of(word: Row, segments: list[Row]) -> Row | None:
    """The builder segment that shares the most time with the word; None when it lies in no segment."""
    best, most = None, 0
    for s in segments:
        shared = min(word["end_ms"], s["end_ms"]) - max(word["start_ms"], s["start_ms"])
        if shared > most:
            best, most = s, shared
    return best


def nearest_segment(word: Row, segments: list[Row]) -> Row:
    """The segment that holds most of the word; for a word in a gap, the segment closest to it in time."""
    held = segment_of(word, segments)
    if held is not None:
        return held
    return min(segments, key=lambda s: max(s["start_ms"] - word["end_ms"], word["start_ms"] - s["end_ms"]))


def gaps(job: Path) -> list[str]:
    """What keeps a run on a clean session from counting as finished; empty when nothing does.

    Every step but alignment ends ok with no fallback. Alignment may be partial, as its guard may hand an
    utterance to the second aligner, but no utterance may keep the speech model's own word times.
    openSMILE may be off for the one reason a default install has: the user has not agreed to its licence.
    The audio for an AI chat may be off for the same kind of reason: the user has not asked for it.
    """
    record = json.loads((job / "job.json").read_text(encoding="utf-8"))
    found = [f"job {record['state']}"] if record["state"] not in ("completed", "completed_with_gaps") else []
    no_audio = not record["settings"]["export"]["audio"]
    for s in record["stages"]:
        allowed = ("ok", "partial") if s["id"] == "alignment" else ("ok",)
        no_consent = (s["id"], s["status"], s.get("error_code")) == ("opensmile", "disabled", NO_CONSENT)
        not_asked = (s["id"], s["status"]) == ("ai_audio", "disabled") and no_audio
        if (s["status"] not in allowed and not no_consent and not not_asked) or s.get("fallback"):
            found.append(f"{s['id']}: {s['status']} {s.get('error_code')} {s.get('fallback')}")
    alignment = json.loads((job / "stages" / "alignment.json").read_text(encoding="utf-8"))
    found += [
        f"alignment {u['id']}: {u['method']} ({u['reason']})"
        for u in alignment["utterances"]
        if u["method"] not in ("mfa", "ctc")
    ]
    return found


def answers_of(structure: Row) -> dict[int, Row]:
    """Word index -> the answer that holds it."""
    by_id = {a["id"]: a for a in structure["answers"]}
    return {
        i: by_id[u["answer_id"]]
        for u in structure["utterances"]
        for i in range(u["first_word"], u["last_word"] + 1)
    }


def persons(truth: Row) -> dict[str, str]:
    """Builder speaker -> person1 or person2: whoever answers first is Person 1."""
    first = next(s["speaker"] for s in truth["segments"] if s["kind"] == "answer")
    other = next(k for k in truth["speakers"] if k != first)
    return {first: "person1", other: "person2"}


def role_accuracy(truth: Row, words: list[Row], structure: Row) -> float:
    """Share of word time whose role is right; backchannels carry no role and are left out.

    Two-person modes: answer words must be in an answer of the right person, other words in none.
    Solo: answer words in an answer, questions read aloud in none.
    """
    owner = answers_of(structure)
    who = persons(truth) if truth["mode"] == "role-swap" else {}
    right = total = 0
    for i, w in enumerate(words):
        s = segment_of(w, truth["segments"])
        if s is None or s["kind"] == "backchannel":
            continue
        ms = w["end_ms"] - w["start_ms"]
        total += ms
        if s["kind"] == "answer":
            a = owner.get(i)
            right += ms if a is not None and a["candidate"] == who.get(s["speaker"], "candidate") else 0
        else:
            right += ms if i not in owner else 0
    return right / total


def part_accuracy(truth: Row, words: list[Row], structure: Row) -> float:
    """Share of answer word time in an answer of the right Part; an unknown Part counts half."""
    owner = answers_of(structure)
    score = total = 0.0
    for i, w in enumerate(words):
        s = segment_of(w, truth["segments"])
        if s is None or s["kind"] != "answer":
            continue
        ms = w["end_ms"] - w["start_ms"]
        total += ms
        part = owner[i]["part"] if i in owner else None
        score += ms if part == s["part"] else ms / 2 if part == "unknown" else 0
    return score / total
