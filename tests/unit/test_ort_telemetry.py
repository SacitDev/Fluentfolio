"""Every step that loads the onnxruntime package switches its telemetry off before it makes a session.

On Windows, ONNX Runtime's trace path ignores ORT_DISABLE_TELEMETRY, so each such child process calls
onnxruntime.disable_telemetry_events() itself. sherpa-onnx is not here: it brings its own ONNX Runtime and
runs in a process without the onnxruntime package.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnxruntime
import pytest
import soundfile as sf

from fluentfolio.alignment import phones
from fluentfolio.analyzers import swiftf0
from fluentfolio.asr import vad
from fluentfolio.asr.onnx_engine import OnnxEngine
from fluentfolio.core.files import write_json
from fluentfolio.media import slices
from fluentfolio.models import catalogue

RATE = 16_000


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The telemetry switch and every session made, in order; sessions are made for real only when asked."""
    seen: list[str] = []
    real = onnxruntime.InferenceSession

    def session(*args: Any, **kwargs: Any) -> Any:
        seen.append("session")
        return real(*args, **kwargs) if Path(str(args[0])).is_file() else SimpleNamespace()

    monkeypatch.setattr(onnxruntime, "disable_telemetry_events", lambda: seen.append("telemetry off"))
    monkeypatch.setattr(onnxruntime, "InferenceSession", session)
    return seen


def tone_job(job: Path) -> Path:
    """A job whose one answer is a 2 s tone, with the candidate audio step already run."""
    (job / "input").mkdir(parents=True)
    t = np.arange(2 * RATE) / RATE
    sf.write(job / "input" / "audio16k.wav", (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), RATE)
    write_json(job / "job.json", {"settings": {}})
    media = {"status": "ok", "quality": {"class": "clean"}, "music_spans": []}
    write_json(job / "stages" / "media.json", media)
    write_json(job / "stages" / "vad.json", {"status": "ok", "segments": [[0, 2000]]})
    write_json(job / "stages" / "structure.json", {
        "candidates": [{"id": "candidate", "speaker": "S1"}], "speech": {"candidate": [[0, 2000]]},
        "answers": [{"id": "A01", "candidate": "candidate", "part": 2, "start_ms": 0, "end_ms": 2000}],
        "utterances": [{"id": "U001", "answer_id": "A01", "start_ms": 0, "end_ms": 2000}],
    })  # fmt: skip
    write_json(job / "stages" / "candidate_audio.json", slices.run(job, lambda pct: None))
    return job


def test_the_speech_finder_and_the_phone_model_switch_telemetry_off_first(
    tmp_path: Path, calls: list[str]
) -> None:
    vad.session(tmp_path / "missing.onnx")
    phones.session(tmp_path / "missing.onnx")
    assert calls == ["telemetry off", "session"] * 2


def test_the_speech_models_switch_telemetry_off_before_loading(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    import onnx_asr

    def load_model(*args: Any, **kwargs: Any) -> Any:
        onnxruntime.InferenceSession("encoder.onnx")  # onnx-asr makes its sessions while it loads
        return SimpleNamespace(with_timestamps=lambda: "timestamped")

    monkeypatch.setattr(onnx_asr, "load_model", load_model)
    OnnxEngine(catalogue.get("parakeet-tdt-0.6b-v2"), Path("unused"))
    assert calls == ["telemetry off", "session"]


def test_the_swiftf0_step_switches_telemetry_off_before_its_model_loads(
    tmp_path: Path, calls: list[str]
) -> None:
    # The swift-f0 package's own model file, loaded for real, so the step's whole path runs.
    found = swiftf0.run(tone_job(tmp_path / "job"), lambda pct: None)
    assert found["status"] == "ok"
    assert calls[0] == "telemetry off" and calls.count("session") >= 1
