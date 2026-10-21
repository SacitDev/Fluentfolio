"""Run the app's own voice-detection and speech steps over many short items at once, as the app runs them.

The items are joined into one 16 kHz recording with 3 s of silence between them, so the speech step's
chunks (which break at pauses of 2 s or more) never join two items; each word then goes back to the item
whose time holds its middle. A model that gives no word times of its own gets the app's CTC word-time pass
after it. Only fluentfolio.asr and fluentfolio.alignment are imported: the system under test, nothing else.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from fluentfolio.alignment import phones
from fluentfolio.asr import base, vad

RATE = 16000
GAP_S = 3.0
REACH_MS = 500  # a word this close to an item's edge still belongs to it


@dataclass
class Run:
    """One run's words and voice-detector speech per item, each item's span, and the speech step's record."""

    words: dict[str, list[dict[str, Any]]]
    speech: dict[str, list[list[int]]]
    spans: dict[str, tuple[int, int]]
    record: dict[str, Any] = field(default_factory=dict)
    chunks: list[list[int]] = field(default_factory=list)


def mono16k(data: bytes) -> np.ndarray:
    """Audio file bytes as 16 kHz mono float32."""
    x, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if rate != RATE:
        g = gcd(RATE, rate)
        x = resample_poly(x, RATE // g, rate // g).astype(np.float32)
    return x


def quiet(pct: int) -> None:
    """Progress is not shown."""


def steps(job: Path) -> dict[str, Any]:
    """The speech step on the job's audio and voice-detector result, then CTC word times if it gave none."""
    result = base.run(job, quiet, offered_only=False)
    (job / "stages" / "asr.json").write_text(json.dumps(result))
    if not result["provenance"]["word_times"]:
        result = phones.asr_times(job, quiet)
        (job / "stages" / "asr.json").write_text(json.dumps(result))
    return result


def run_items(items: list[tuple[str, np.ndarray]], model: str, device: str, job: Path) -> Run:
    """Each item's words (the speech step's word dicts) and speech segments, from one run of both steps.

    Times stay those of the joined recording; `spans` gives each item's place in it. A recording in which
    the voice detector finds too little speech gives no words, as the app writes none for it.
    """
    (job / "input").mkdir(parents=True, exist_ok=True)
    (job / "stages").mkdir(exist_ok=True)
    gap = np.zeros(int(GAP_S * RATE), dtype=np.float32)
    spans, pieces, at = {}, [gap], len(gap)
    for item, audio in items:
        spans[item] = (at * 1000 // RATE, (at + len(audio)) * 1000 // RATE)
        pieces += [audio, gap]
        at += len(audio) + len(gap)
    sf.write(job / "input" / "audio16k.wav", np.concatenate(pieces), RATE, subtype="PCM_16")
    (job / "job.json").write_text(json.dumps({"layout": "mixed", "settings": {"asr": {"model": model}}}))
    os.environ["FLUENTFOLIO_DEVICE"] = device  # read by the speech engine, as the app's step runner sets it
    out = Run({item: [] for item, _ in items}, {item: [] for item, _ in items}, spans)
    try:
        found = vad.run(job, quiet)
    except vad.InputError as error:
        if error.code != "E_INPUT_SILENT":
            raise
        return out
    (job / "stages" / "vad.json").write_text(json.dumps(found))
    result = steps(job)
    out.record, out.chunks = result["provenance"], result["chunks"]
    for w in result["words"]:
        middle = (w["start_ms"] + w["end_ms"]) / 2
        item = next((i for i, (a, b) in spans.items() if a - REACH_MS <= middle <= b + REACH_MS), None)
        if item is not None:
            out.words[item].append(w)
    for s, e in found["segments"]:
        for item, (a, b) in spans.items():
            if s < b + REACH_MS and e > a - REACH_MS:
                out.speech[item].append([s, e])
    return out


def transcribe(
    items: list[tuple[str, np.ndarray]], model: str, device: str, job: Path
) -> dict[str, list[Any]]:
    """Each item's words (the speech step's word dicts), from one run of the voice and speech steps."""
    return run_items(items, model, device, job).words
