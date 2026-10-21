"""The device modes on the composed interview: the card hidden from Auto, and the GPU against the CPU.

Both runs use an app folder installed with GPU support and the models already downloaded.
"""

from __future__ import annotations

import difflib
import statistics
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import jiwer
import pytest

from fluentfolio.app import facade
from fluentfolio.core.enums import Device, JobState
from fluentfolio.core.files import read_json, write_json
from fluentfolio.core.paths import models_dir
from fluentfolio.setup import device

pytestmark = pytest.mark.slow
SESSIONS = Path(__file__).resolve().parents[1] / "fixtures" / "sessions"
WHISPER = "whisper-large-v3-turbo"
GPU_STEPS = {"asr", "speakers"}
NEAR_MS = 250  # a word this close to a silence's edge may be the speech around it
Row = dict[str, Any]


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory, session_builder: ModuleType) -> tuple[Row, Path]:
    out = tmp_path_factory.mktemp("CS-INT")
    truth = session_builder.build(SESSIONS / "CS-INT.json", out)
    return truth, out / "session.wav"


@pytest.fixture
def ready_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An app folder installed with GPU support that uses the models already downloaded."""
    app = tmp_path / "app"
    write_json(app / "state.json", {"backend": "cu130", "models_dir": str(models_dir())})
    monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(app))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return app


def norm(text: str) -> str:
    return " ".join("".join(c if c.isalnum() or c == "'" else " " for c in text.lower()).split())


def words(job: Path) -> list[Row]:
    found: list[Row] = read_json(job / "stages" / "asr.json")["words"]
    return found


def shape(job: Path) -> Row:
    """What must not change with the device: roles, Parts, answers, statuses and warnings."""
    structure = read_json(job / "stages" / "structure.json")
    evidence = read_json(job / "evidence.json")
    return {
        "candidates": [(c["id"], c["speaker"], c["roles"]["confidence"]) for c in structure["candidates"]],
        "parts": [(p["candidate"], p["part"], p["source"], p["confidence"]) for p in structure["parts"]],
        "answers": [(a["candidate"], a["part"]) for a in structure["answers"]],
        "statuses": [(t["id"], t["status"]) for t in evidence["manifest"]["tools"]],
        "warnings": [w["code"] for w in evidence["manifest"]["warnings"]],
    }


def in_silence(truth: Row, job: Path) -> list[Row]:
    """Words inside the labelled silences (the Part 2 preparation minute), away from their edges."""
    spans = [(s["start_ms"] + NEAR_MS, s["end_ms"] - NEAR_MS) for s in truth["silences"]]
    return [w for w in words(job) if any(a < w["end_ms"] and w["start_ms"] < b for a, b in spans)]


def test_with_every_card_hidden_auto_runs_on_the_cpu_and_says_why(
    ready_app: Path, session: tuple[Row, Path], monkeypatch: pytest.MonkeyPatch, scoring: ModuleType
) -> None:
    _, wav = session
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    done = facade.analyze(wav, "interview", partner_consent="cli-flag")
    record = done.record
    assert scoring.gaps(done.job) == []
    why = "CUDA_VISIBLE_DEVICES hides every graphics card."
    assert record.device == {"requested": "auto", "resolved": "cpu", "reason": why}
    assert {s.device for s in record.stages} == {"cpu"} and not any(s.fallback for s in record.stages)
    assert read_json(done.job / "stages" / "speakers.json")["device"] == "cpu"
    text = (done.job / "export" / "for_ai" / "for_ai.txt").read_text(encoding="utf-8")
    assert f"device: cpu (asked: auto; {why})" in text


def test_with_the_gpu_chosen_and_every_card_hidden_the_gpu_step_fails_with_its_code_and_nothing_replaces_it(
    ready_app: Path, session: tuple[Row, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, wav = session
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    done = facade.analyze(wav, "interview", device=Device.GPU, partner_consent="cli-flag")
    record = done.record
    assert record.state is JobState.COMPLETED_WITH_GAPS  # the speaker step failed; it is optional
    speakers = next(s for s in record.stages if s.id == "speakers")
    assert (speakers.status, speakers.error_code, speakers.device) == ("failed", "E_GPU_UNAVAILABLE", "gpu")
    assert not any(s.fallback for s in record.stages)  # neither the CPU nor the other diarizer ran
    assert record.device == {"requested": "gpu", "resolved": "gpu", "reason": "You chose the GPU."}
    assert read_json(done.job / "evidence.json")["manifest"]["device"] == record.device
    text = (done.job / "export" / "for_ai" / "for_ai.txt").read_text(encoding="utf-8")
    assert "device: gpu (asked: gpu; You chose the GPU.)" in text and "E_GPU_UNAVAILABLE" in text


@pytest.mark.gpu
def test_the_driver_query_finds_this_computers_card() -> None:
    # The expected card comes from the same tool run with the whole environment, read by hand here.
    exe = device.nvidia_smi()
    assert exe is not None
    done = subprocess.run([str(exe), *device.QUERY], capture_output=True, text=True, timeout=30, check=True)
    first = tuple(f.strip() for f in done.stdout.splitlines()[0].split(","))
    card = device.probe()[0]
    assert (f"{card.compute[0]}.{card.compute[1]}", str(card.memory_mb), card.driver) == first


@pytest.mark.gpu
def test_on_the_gpu_auto_gives_the_structure_text_and_times_of_the_cpu(
    ready_app: Path, session: tuple[Row, Path], tolerance: Any, scoring: ModuleType
) -> None:
    truth, wav = session
    smoke = device.check_gpu(ready_app)
    assert (smoke["passed"], smoke["code"]) == (True, None)
    gpu = facade.analyze(wav, "interview", asr=WHISPER, partner_consent="cli-flag")
    cpu = facade.analyze(wav, "interview", asr=WHISPER, device=Device.CPU, partner_consent="cli-flag")
    assert gpu.record.device == {
        "requested": "auto", "resolved": "gpu", "reason": "The graphics card passed every check.",
    }  # fmt: skip
    assert cpu.record.device == {"requested": "cpu", "resolved": "cpu", "reason": "You chose the CPU."}
    for run, gpu_steps in ((gpu, GPU_STEPS), (cpu, set())):
        assert scoring.gaps(run.job) == []
        assert {s.id for s in run.record.stages if s.device == "gpu"} == gpu_steps
        assert not any(s.fallback for s in run.record.stages)
    assert read_json(gpu.job / "stages" / "asr.json")["provenance"]["device"] == "gpu"
    assert read_json(gpu.job / "stages" / "speakers.json")["device"] == "gpu"
    assert read_json(cpu.job / "stages" / "speakers.json")["device"] == "cpu"
    # The same people, roles, Parts, answers, statuses and warnings.
    assert shape(gpu.job) == shape(cpu.job)
    # The text: the error rate against the session's own script differs by 1 point at most.
    reference = norm(" ".join(s["text"] for s in truth["segments"]))
    wer = [jiwer.wer(reference, norm(" ".join(w["text"] for w in words(r.job)))) for r in (gpu, cpu)]
    # The times: the same words, in order, start at most 40 ms apart at the median.
    a, b = words(gpu.job), words(cpu.job)
    matcher = difflib.SequenceMatcher(None, [w["text"].lower() for w in a], [w["text"].lower() for w in b])
    pairs = [(a[m.a + k], b[m.b + k]) for m in matcher.get_matching_blocks() for k in range(m.size)]
    gaps = [abs(x["start_ms"] - y["start_ms"]) for x, y in pairs]
    print(
        f"WER gpu {wer[0]:.4f} cpu {wer[1]:.4f}; words gpu {len(a)} cpu {len(b)} matched {len(pairs)}; "
        f"median start difference {statistics.median(gaps)} ms, largest {max(gaps)} ms"
    )
    assert abs(wer[0] - wer[1]) * 100 <= tolerance("device_parity", "wer_points")
    assert len(pairs) >= 0.9 * max(len(a), len(b))  # noqa: PLR2004 - most words are the same words
    assert statistics.median(gaps) <= tolerance("device_parity", "word_time_median_ms")
    # Whisper writes nothing in the minute of preparation silence, on either device.
    assert in_silence(truth, gpu.job) == in_silence(truth, cpu.job) == []
