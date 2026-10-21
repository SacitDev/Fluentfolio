"""The SwiftF0 step: a second pitch track from each candidate's own audio, for the pitch fusion.

SwiftF0 is an ONNX model in the swift-f0 package; it runs on the CPU in a process of its own.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import soundfile as sf

from fluentfolio.analyzers.measures import acoustic
from fluentfolio.core.files import read_json

COMPONENT = "swiftf0"
RATE = 16000
FRAME_MS = 16.0  # one frame per 256 samples at 16 kHz
VOICED = 0.5  # the model's voicing score is calibrated so that a frame at or above this is voiced


def run(job: Path, progress: Callable[[int], None]) -> dict[str, Any]:
    """SwiftF0's pitch for every candidate, from the candidate audio."""
    settings = read_json(job / "job.json")["settings"]
    if not acoustic.switched_on(settings, COMPONENT, "pitch"):
        return {"status": "disabled", "candidates": {}}
    made = acoustic.candidate_audio(job)
    if made is None:
        return {"status": "skipped", "reason": "the candidate audio step did not finish", "candidates": {}}
    import onnxruntime
    import swift_f0

    onnxruntime.disable_telemetry_events()  # the Windows trace path ignores ORT_DISABLE_TELEMETRY
    detector = swift_f0.SwiftF0()  # makes its ONNX Runtime session here
    structure = read_json(job / "stages" / "structure.json")
    out = {}
    for n, (cid, row) in enumerate(made["candidates"].items()):
        answers = [a for a in structure["answers"] if a["candidate"] == cid]
        found: list[acoustic.Frame] = []
        if row["wav"] is not None:
            result = detector.detect(sf.read(job / row["wav"], dtype="float32")[0], RATE)
            pairs = zip(result.pitch_hz, result.confidence, strict=True)
            hz = [float(f) if c >= VOICED else None for f, c in pairs]
            pieces = acoustic.pieces(read_json(job / row["map"])["pieces"])
            found = acoustic.frames([t * 1000 for t in result.timestamps], hz, pieces, answers)
        out[cid] = {"scopes": acoustic.pitch_scopes(found, FRAME_MS, answers)}
        progress(100 * (n + 1) // len(made["candidates"]))
    version = f"swift-f0 {metadata.version('swift-f0')}, onnxruntime {metadata.version('onnxruntime')}"
    return {
        "status": "ok", "provenance": {"tool": COMPONENT, "version": version}, "analyzers": ["pitch"],
        "candidates": out,
    }  # fmt: skip
