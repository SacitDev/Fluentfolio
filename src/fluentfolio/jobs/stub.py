"""A stand-in for the model steps, for the browser tests only: it takes over only when FF_TEST_STUB=1.

It writes hand-made voice detector, language, recogniser and speaker results for a short mock test laid over
the decoded recording, so the steps after them run for real in seconds and with no model installed. Every
result it writes says "stub": true, so the evidence of such a run says so too and never counts as real.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from fluentfolio.asr.normalize import word
from fluentfolio.core.enums import Layout, Mode
from fluentfolio.core.files import read_json
from fluentfolio.jobs.loop import Stage
from fluentfolio.speakers import checks

SWITCH = "FF_TEST_STUB"
WAIT_S = 1.0  # each stand-in step takes this long, so a browser test can watch the progress and press Cancel
WORD_MS, GAP_MS = 320, 80  # 400 ms a word: a calm 150 words a minute
SENTENCE_MS = 200  # under the 250 ms pause threshold, so only the marked pauses and turn changes count
PAUSE, PAUSE_MS = "|", 1200  # a mark in a line: a pause of 1 s or more
TURN_MS = 700
PREP_MS = 5000  # after the topic card; a real test gives a minute
MERGE_MS = 250  # the voice detector joins speech closer than this
DIARIZER = "sherpa-onnx-diarizer"  # the step it stands in for when no account is set up

# The examiner (A) and the candidate (B). Written for these tests; the long turn runs for more than 45 s.
SCRIPT = [
    ("A", "Let's talk about your home town. Where do you come from?"),
    (
        "B",
        "I come from a small town near the coast. | It is quiet most of the year, but in summer it gets "
        "really busy because of the beach.",
    ),
    ("A", "What do you like most about living there?"),
    (
        "B",
        "Um, I like the sea air and the people. Everyone knows each other, so it feels safe, and the "
        "food is, uh, really good.",
    ),
    ("A", "Do you think you will live there in the future?"),
    (
        "B",
        "Maybe not. I would like to work in a bigger city for some years, | and then perhaps I will go "
        "back when I am older.",
    ),
    (
        "A",
        "Now I'm going to give you a topic and I'd like you to talk about it for one to two minutes. "
        "You have one minute to prepare. Describe a book that you enjoyed reading. You should say what "
        "the book was, when you read it, and explain why you enjoyed it.",
    ),
    (
        "B",
        "I'd like to talk about a novel I read last winter. It was a story about a family who moves to "
        "a farm in the mountains after the father loses his job. | I read it during the holidays, when I "
        "had a lot of free time, and I finished it in about four days. Um, the reason I enjoyed it so "
        "much is that the characters felt real. The mother, the father and the two children all have "
        "their own problems, and the writer shows how they slowly learn to help each other. | There "
        "were also long descriptions of the weather and the animals, which I usually find boring, but "
        "here they made me feel that I was living on the farm myself. I think the book also taught me "
        "something about patience, because nothing in the story happens quickly. | So, uh, it is still "
        "one of my favourite books, and I have given it to two of my friends.",
    ),
    (
        "A",
        "Thank you. We've been talking about a book you enjoyed. Now I'd like to discuss some more "
        "general questions. Why do you think some people read less than before?",
    ),
    (
        "B",
        "I think the main reason is the phone. People have so many short videos and messages that "
        "they, they lose the habit of reading for a long time. | Also, many people work long hours, and "
        "when they get home they are too tired to follow a long story, so they choose something easier.",
    ),
    ("A", "How might reading change in the future?"),
    (
        "B",
        "Maybe more people will listen to books instead of reading them, | because you can do it while "
        "you travel or cook. I also think schools will use more digital books, but I hope paper books do "
        "not disappear, because many readers still like to hold a real book.",
    ),
]
Timed = tuple[str, int, int]  # speaker, start, end


def on() -> bool:
    return os.environ.get(SWITCH) == "1"


def _lines(mode: str) -> list[tuple[str, str]]:
    """The script for the mode: one voice reads everything in solo; role-swap runs it twice, roles swapped."""
    if mode == Mode.SOLO:
        return [("B", text) for _, text in SCRIPT]
    if mode == Mode.ROLE_SWAP:
        return SCRIPT + [("B" if who == "A" else "A", text) for who, text in SCRIPT]
    return SCRIPT


def _timed(job: Path) -> tuple[list[tuple[str, int, int]], list[Timed]]:
    """Each word with its times, and each stretch of speech with its speaker, cut at the recording's end."""
    mode = str(read_json(job / "job.json")["mode"])
    duration_ms = int(read_json(job / "stages" / "media.json")["duration_ms"])
    words: list[tuple[str, int, int]] = []
    speech: list[Timed] = []
    t = TURN_MS
    for who, text in _lines(mode):
        start = end = t
        for piece in text.split():
            if piece == PAUSE:
                speech.append((who, start, end))
                start = t = end + PAUSE_MS
                continue
            words.append((piece, t, t + WORD_MS))
            end = t + WORD_MS
            t = end + (SENTENCE_MS if piece[-1] in ".?!" else GAP_MS)
        speech.append((who, start, end))
        t = end + TURN_MS + (PREP_MS if "to prepare" in text else 0)
    kept = [w for w in words if w[2] <= duration_ms]
    return kept, [(who, a, min(b, duration_ms)) for who, a, b in speech if a < duration_ms]


def _merged(spans: list[tuple[int, int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] < MERGE_MS:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _wait() -> None:
    time.sleep(WAIT_S)


def vad(job: Path) -> dict[str, Any]:
    _wait()
    _, speech = _timed(job)
    found = _merged([(a, b) for _, a, b in speech])
    result: dict[str, Any] = {"status": "ok", "component": "silero-vad", "frame_ms": 32, "stub": True,
                              "speech_ms": sum(b - a for a, b in found), "segments": found}  # fmt: skip
    if read_json(job / "job.json").get("layout", Layout.MIXED) != Layout.MIXED:
        tracks = [_merged([(a, b) for w, a, b in speech if w == who]) for who in ("A", "B")]
        result["tracks"] = [
            {"track": k + 1, "speech_ms": sum(b - a for a, b in s), "segments": s}
            for k, s in enumerate(tracks)
        ]
    return result


def language(job: Path) -> dict[str, Any]:
    _wait()
    found = read_json(job / "stages" / "vad.json")["segments"]
    checked = [{"start_ms": a, "end_ms": b, "language": "en", "probability": 0.99, "english": 0.99,
                "marked": False} for a, b in found if b - a >= 3000]  # fmt: skip
    provenance = {"model": "whisper-tiny", "revision": "stub", "device": "cpu"}
    return {"status": "ok", "provenance": provenance, "segments": checked, "non_english_ms": 0, "stub": True}


def asr(job: Path) -> dict[str, Any]:
    _wait()
    model = str(read_json(job / "job.json")["asr"])
    timed, _ = _timed(job)
    found = [w for raw, a, b in timed if (w := word(raw, a, b, None, model)) is not None]
    provenance = {
        "model": model, "revision": "stub", "transcript_style": "verbatim-best-effort",
        "punctuation": "machine", "filler_capture": "best_effort",
    }  # fmt: skip
    return {
        "status": "ok",
        "provenance": provenance,
        "words": [{**w, "chunk": 0} for w in found],
        "stub": True,
    }


def speakers(job: Path) -> dict[str, Any]:
    _wait()
    mode = str(read_json(job / "job.json")["mode"])
    _, speech = _timed(job)
    turns = sorted((a, b, who) for who, a, b in speech)
    if mode == Mode.SOLO:
        result = checks.second_voice(DIARIZER, turns)
    else:
        result = checks.summarize(DIARIZER, checks.relabel(turns)[0], mode=mode)
    return {**result, "version": "stub", "device": "cpu", "stub": True}


STEPS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "vad": vad, "language": language, "asr": asr, "speakers": speakers,
}  # fmt: skip


def plan(stages: list[Stage]) -> list[Stage]:
    """The steps with each model step replaced by its stand-in; a track layout keeps its own speaker step."""
    return [
        replace(s, inline=STEPS[s.id], gpu=False, child="", fallback="")
        if s.id in STEPS and (s.id != "speakers" or s.inline is None) else s
        for s in stages
    ]  # fmt: skip
