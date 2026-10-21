"""Speech recognition: chunks cut at VAD pauses, one engine per model, words exactly as the model wrote them.

The asr stage writes every word with its times, confidence, filler and cut-off flags and the model that
made it, plus a record of how the transcript was made. No word is cleaned up, corrected or removed.
"""

from __future__ import annotations

import importlib
import time
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from fluentfolio.asr import normalize
from fluentfolio.core.errors import SetupError
from fluentfolio.core.files import read_json
from fluentfolio.core.paths import models_dir
from fluentfolio.models import catalogue
from fluentfolio.models.catalogue import Component
from fluentfolio.models.download import require

RATE = 16000
MAX_CHUNK_MS = 30000  # every model here takes 30 s at once
# A pause this long starts a new chunk: the speaker's next turn needs no context from before it, and
# the model then never sees a long stretch of noise without speech.
BREAK_GAP_MS = 2000
PAD_MS = 200  # audio kept on each side of a chunk, so its first and last sounds are whole
# Each engine lives in its own module, so a child loads only the one library family it runs.
ENGINES = {
    "onnx-asr": "fluentfolio.asr.onnx_engine:OnnxEngine",
    "faster-whisper": "fluentfolio.asr.whisper_engine:WhisperEngine",
}
HALLUCINATION = "possible_hallucination"
# A model stuck in a loop writes the same few words again and again; a 4-word run that a chunk holds
# 3 times or more is flagged on every word of it. Nothing is removed.
REPEATED = "repeated_ngram"
NGRAM = 4
NGRAM_REPEATS = 3
# A word less than one pause (250 ms) from VAD speech belongs to that speech: the model's word times move
# in 80 ms steps, and a run's first word often starts a few steps before the VAD says speech began.
NEAR_SPEECH_MS = 250
# On some inputs a model returns nothing, or stops early, for a whole chunk, while the same audio cut a
# little shorter reads fine; two voices at once made it happen in a test. A chunk whose words cover less
# than this share of its speech is read again in two pieces, split at its longest pause, down to three
# levels (pieces of about 4 s). Words reach 200 ms around their own times.
MIN_COVERED = 0.5
WORD_REACH_MS = 200
REREAD_MIN_MS = 4000  # less speech than this is too little to judge: a cough or a "mm" may rightly be no word
REREAD_DEPTH = 3
# A model can also leave out one stretch of a chunk it otherwise hears: Whisper on the GPU skipped a 2 s
# question before a long answer that each half, read alone, gave in full. So a chunk is read again in
# pieces too when this much of its speech, in one stretch that pauses under NEAR_SPEECH_MS do not end, is
# out of every word's reach. When that finds no more, it keeps its words with no warning: the stretch may
# be a laugh or a long "mm".
MISSED_RUN_MS = 1000


@dataclass(frozen=True)
class Heard:
    """A word as the model wrote it, with times in seconds from the start of its chunk."""

    raw: str
    start_s: float
    end_s: float
    confidence: float | None


class Engine(Protocol):
    device: str
    precision_ms: int
    confidence: str | None  # how a word's confidence is computed; None when the model gives none
    settings: dict[str, str]
    flags_repeats: bool  # models known to loop on unclear audio get the repeated-run flag
    timed: bool  # False when the model gives no word times, so each word spans its chunk until timed
    dropped: int  # segments dropped by the engine's own loop guard

    def versions(self) -> dict[str, str]: ...

    def heard(self, audio: np.ndarray) -> list[Heard]: ...


def chunks(segments: Sequence[Sequence[int]], duration_ms: int) -> list[tuple[int, int]]:
    """Group VAD segments into chunks of at most 30 s, pads included; (start_ms, end_ms) of the audio to read.

    Neighbouring chunks share no audio: each pad is at most half the pause between them.
    """
    groups: list[list[int]] = []
    for a, b in segments:
        if groups and b - groups[-1][0] <= MAX_CHUNK_MS and a - groups[-1][1] < BREAK_GAP_MS:
            groups[-1][1] = b
        else:
            groups.append([a, b])
    padded = []
    for k, (a, b) in enumerate(groups):
        room = MAX_CHUNK_MS - (b - a)  # never negative: the VAD cuts every run at 30 s
        before = a if k == 0 else (a - groups[k - 1][1]) // 2
        after = duration_ms - b if k + 1 == len(groups) else (groups[k + 1][0] - b) // 2
        left = min(PAD_MS, room // 2, before)
        right = min(PAD_MS, room - left, after)
        padded.append((a - left, b + right))
    return padded


def flag_outside_speech(words: list[dict[str, Any]], segments: Sequence[Sequence[int]]) -> None:
    """Mark every word that lies a pause or more away from all VAD speech; segments are in time order."""
    starts = [a for a, _ in segments]
    for w in words:
        # The last segment that starts before the widened word ends; earlier ones end even earlier.
        i = bisect_right(starts, w["end_ms"] + NEAR_SPEECH_MS - 1) - 1
        if i < 0 or segments[i][1] <= w["start_ms"] - NEAR_SPEECH_MS:
            w["flags"].append(HALLUCINATION)


def repeated_runs(keys: list[str]) -> set[int]:
    """The places of every word in a 4-word run that these words hold 3 or more times."""
    starts: dict[tuple[str, ...], list[int]] = {}
    for i in range(len(keys) - NGRAM + 1):
        starts.setdefault(tuple(keys[i : i + NGRAM]), []).append(i)
    return {
        i + k for found in starts.values() if len(found) >= NGRAM_REPEATS for i in found for k in range(NGRAM)
    }


def flag_repeats(words: list[dict[str, Any]]) -> None:
    """Flag every word of each 4-word run that these words (one chunk) hold 3 or more times."""
    for i in sorted(repeated_runs([w["text"].lower() for w in words])):
        words[i]["flags"].append(REPEATED)


def loadable(component: Component) -> bool:
    """A speech model whose engine is here."""
    return component.kind == "asr" and component.engine in ENGINES


def runnable(component: Component) -> bool:
    """A speech model a learner can choose: its engine is here, and it gives its own word times."""
    # Canary is not offered: the speech-model comparison did not add it. Only that comparison runs it,
    # with the word-time step it needs (alignment.phones.asr_times) after it.
    return loadable(component) and "asr_times" not in component.stages


def choices() -> list[str]:
    """The speech models a learner can choose, in catalogue order."""
    return [c.id for c in catalogue.components().values() if runnable(c)]


def engine_for(component: Component, folder: Path) -> Engine:
    if not loadable(component):
        raise SetupError("E_COMPONENT_UNAVAILABLE", component.id)
    module, name = ENGINES[component.engine].split(":")
    engine: Engine = getattr(importlib.import_module(module), name)(component, folder)
    return engine


def _speech(segments: Sequence[Sequence[int]], a: int, b: int) -> list[tuple[int, int]]:
    return [(max(a, s), min(b, e)) for s, e in segments if s < b and e > a]


def _speech_ms(segments: Sequence[Sequence[int]], a: int, b: int) -> int:
    return sum(e - s for s, e in _speech(segments, a, b))


def _reach(words: list[dict[str, Any]]) -> list[list[int]]:
    """The times within WORD_REACH_MS of a word, merged, in order."""
    reach: list[list[int]] = []
    for w in sorted(words, key=lambda w: w["start_ms"]):
        lo, hi = w["start_ms"] - WORD_REACH_MS, w["end_ms"] + WORD_REACH_MS
        if reach and lo <= reach[-1][1]:
            reach[-1][1] = max(reach[-1][1], hi)
        else:
            reach.append([lo, hi])
    return reach


def covered(words: list[dict[str, Any]], segments: Sequence[Sequence[int]], a: int, b: int) -> float | None:
    """The share of VAD speech in [a, b) within reach of a word; None when there is too little speech."""
    speech = _speech(segments, a, b)
    total = _speech_ms(segments, a, b)
    if total < REREAD_MIN_MS:
        return None
    near = sum(max(0, min(e, hi) - max(s, lo)) for s, e in speech for lo, hi in _reach(words))
    return near / total


def missed_run(words: list[dict[str, Any]], segments: Sequence[Sequence[int]], a: int, b: int) -> int:
    """The most VAD speech (ms) in [a, b) in one stretch out of every word's reach; short pauses go on."""
    speech = _speech(segments, a, b)
    stretches: list[list[int]] = []
    for s, e in speech:
        if stretches and s - stretches[-1][1] < NEAR_SPEECH_MS:
            stretches[-1][1] = e
        else:
            stretches.append([s, e])
    reach = _reach(words)
    most = 0
    for lo, hi in stretches:
        start = lo
        for near_lo, near_hi in reach:
            if near_hi <= start or near_lo >= hi:
                continue
            if near_lo > start:
                most = max(most, _speech_ms(speech, start, near_lo))
            start = near_hi
        if start < hi:
            most = max(most, _speech_ms(speech, start, hi))
    return most


def _cut(segments: Sequence[Sequence[int]], a: int, b: int) -> int:
    """The middle of the longest pause between VAD segments inside [a, b), else the middle of the span."""
    inside = _speech(segments, a, b)
    gaps = [(n[0] - p[1], (p[1] + n[0]) // 2) for p, n in zip(inside, inside[1:], strict=False)]
    return max(gaps)[1] if gaps else (a + b) // 2


def _read(engine: Engine, model: str, wav: Path, a: int, b: int) -> list[dict[str, Any]]:
    audio, _ = sf.read(str(wav), start=a * RATE // 1000, stop=b * RATE // 1000, dtype="float32")
    words = []
    for h in engine.heard(audio):
        start = min(b, a + round(h.start_s * 1000))
        end = min(b, max(start, a + round(h.end_s * 1000)))
        if (w := normalize.word(h.raw, start, end, h.confidence, model)) is not None:
            words.append(w)
    return words


def _heard(
    engine: Engine, model: str, wav: Path, segments: Sequence[Sequence[int]], a: int, b: int, depth: int
) -> tuple[list[dict[str, Any]], int, int]:
    """A span's words, read again in two pieces while they cover too little of its speech or miss a stretch.

    Returns the words, the pieces they came from, and the speech (ms) whose words are still too few.
    """
    words = _read(engine, model, wav, a, b)
    share = covered(words, segments, a, b)
    if share is None or (share >= MIN_COVERED and missed_run(words, segments, a, b) < MISSED_RUN_MS):
        return words, 1, 0
    if depth > 0:
        cut = _cut(segments, a, b)
        left, n_left, low_left = _heard(engine, model, wav, segments, a, cut, depth - 1)
        right, n_right, low_right = _heard(engine, model, wav, segments, cut, b, depth - 1)
        again = covered(left + right, segments, a, b) or 0
        if again > share:
            low = low_left + low_right if again >= MIN_COVERED else _speech_ms(segments, a, b)
            return left + right, n_left + n_right, low
    return words, 1, _speech_ms(segments, a, b) if share < MIN_COVERED else 0


def transcribe(
    engine: Engine, component: Component, wav: Path, segments: Sequence[Sequence[int]],
    progress: Callable[[int], None] | None = None,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]], list[dict[str, Any]]]:  # fmt: skip
    """Run the engine on each chunk of a 16 kHz WAV; returns the chunks, the words in order, the re-reads."""
    duration_ms = sf.info(str(wav)).frames * 1000 // RATE
    spans = chunks(segments, duration_ms)
    words: list[dict[str, Any]] = []
    rereads = []
    for k, (a, b) in enumerate(spans):
        heard, pieces, low = _heard(engine, component.id, wav, segments, a, b, REREAD_DEPTH)
        if engine.flags_repeats:
            flag_repeats(heard)
        for w in heard:
            w["chunk"] = k
        words += heard
        if pieces > 1 or low:
            share = covered(heard, segments, a, b) or 0
            rereads.append(
                {"chunk": k, "pieces": pieces, "covered": round(share, 3), "missing_speech_ms": low}
            )
        if progress:
            progress(min(99, 100 * (k + 1) // len(spans)))
    flag_outside_speech(words, segments)
    return spans, words, rereads


def run(job: Path, progress: Callable[[int], None], offered_only: bool = True) -> dict[str, Any]:
    """The asr stage: the chosen model on input/audio16k.wav, in the chunks that stages/vad.json gives.

    `offered_only=False` also runs a model a learner cannot choose yet; only the model comparison does.
    """
    chosen = str(read_json(job / "job.json")["settings"]["asr"]["model"])
    component = catalogue.components().get(chosen)
    if component is None or not (runnable(component) if offered_only else loadable(component)):
        raise SetupError("E_COMPONENT_UNAVAILABLE", chosen)
    started = time.monotonic()
    engine = engine_for(component, require(component, models_dir()))
    vad = read_json(job / "stages" / "vad.json")
    extra: dict[str, Any] = {}
    if "tracks" not in vad:
        spans, words, rereads = transcribe(
            engine, component, job / "input" / "audio16k.wav", vad["segments"], progress
        )
    else:  # one track per person: each track is transcribed on its own speech only, so bleed is not heard
        spans, words, owners, rereads = [], [], [], []
        for k, track in enumerate(vad["tracks"]):
            wav = job / "input" / f"track_{track['track']}.wav"
            share = lambda p, k=k: progress((100 * k + p) // len(vad["tracks"]))  # noqa: E731
            found, heard, again = transcribe(engine, component, wav, track["segments"], share)
            for w in heard:
                w["chunk"] += len(spans)
                w["track"] = track["track"]
            rereads += [{**r, "chunk": r["chunk"] + len(spans)} for r in again]
            spans += found
            owners += [track["track"]] * len(found)
            words += heard
        words.sort(key=lambda w: (w["start_ms"], w["track"]))
        extra["chunk_tracks"] = owners
    style = component.transcript_style
    progress(100)
    return {
        "status": "ok",
        "provenance": {
            "model": component.id,
            "revision": component.revision,
            "engine": engine.versions(),
            "device": engine.device,
            "transcript_style": style,
            "filler_capture": "best_effort" if style == "verbatim-best-effort" else "lower_bound",
            "word_times": engine.timed,
            "timestamp_precision_ms": engine.precision_ms,
            "confidence": engine.confidence,
            "punctuation": "machine",
            "settings": {
                "max_chunk_ms": MAX_CHUNK_MS,
                "break_gap_ms": BREAK_GAP_MS,
                "pad_ms": PAD_MS,
                "near_speech_ms": NEAR_SPEECH_MS,
                "min_covered": MIN_COVERED,
                **engine.settings,
            },
            "fallback": None,
            "dropped_segments": engine.dropped,
            "wall_ms": int((time.monotonic() - started) * 1000),
        },
        "chunks": [list(s) for s in spans],
        **extra,
        "rereads": rereads,  # chunks read again in pieces, and the speech whose words are still too few
        "warnings": ["W_WORDS_MISSING"] if any(r["missing_speech_ms"] for r in rereads) else [],
        "words": words,
    }
