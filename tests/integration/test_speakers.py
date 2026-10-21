"""Both speaker separators, run as children, on a composed 60 s clip of two voices with a 2 s overlap."""

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fluentfolio import __version__
from fluentfolio.core.files import read_json, write_json
from fluentfolio.core.paths import app_root, models_dir
from fluentfolio.jobs import loop, store
from fluentfolio.jobs.runner import child_env
from fluentfolio.models import catalogue
from fluentfolio.models.download import folder
from fluentfolio.settings.model import Settings
from fluentfolio.settings.store import snapshot
from fluentfolio.speakers import checks

pytestmark = pytest.mark.slow

OFFLINE = Path(__file__).parents[1] / "support" / "offline_stage.py"
BACKENDS = {"speakers-pyannote": "pyannote-community-1", "speakers-sherpa": "sherpa-onnx-diarizer"}


def offline_argv(stage_id: str, job_id: str) -> list[str]:
    return [sys.executable, "-I", "-X", "utf8", str(OFFLINE), stage_id, job_id]


def speakers_job(
    tmp_path: Path,
    clip: tuple[Path, dict[str, Any]],
    child: str,
    fallback: str = "",
    *,
    roots: loop.Roots | None = None,
    device: str = "cpu",
    count: str = "auto",
) -> tuple[store.JobRecord, Path]:
    """Run only the speaker step of an interview job whose media step made this 16 kHz mono WAV."""
    roots = roots or loop.Roots(app=app_root(), data=tmp_path / "data", models=models_dir())
    now = store.utc_now()
    record = store.JobRecord(
        job_id=store.new_job_id(), created_utc=now, updated_utc=now, app_version=__version__,
        mode="interview", layout="mixed", device={"requested": device, "resolved": device},
        settings=snapshot(Settings.model_validate({"speakers": {"count": count}})),
    )  # fmt: skip
    job = store.create(roots.data, record)
    shutil.copyfile(clip[0], job / "input" / "audio16k.wav")  # the builder writes the media step's own format
    write_json(job / "stages" / "media.json", {"status": "ok", "duration_ms": clip[1]["duration_ms"]})
    stage = loop.Stage("speakers", "stages/speakers.json", required=False, gpu=child == "speakers-pyannote",
                       child=child, fallback=fallback)  # fmt: skip
    done = loop.run_job(roots, job.name, [stage], audio_min=2, no_progress_s=300, argv_for=offline_argv)
    return done, job


@pytest.fixture
def clip(session_builder: ModuleType, tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Part 1 and the cue card of the interview session, about 60 s, with one question 2 s into an answer."""
    base = session_builder.load_recipe(session_builder.SESSIONS / "CS-INT.json")
    edits = [
        {"change": "int_p1_q4", "set": {"gap_ms": None, "overlap_ms": 2000}},
        {"change": "int_p2_go", "set": {"gap_ms": 500, "gap_label": None}},  # no minute of preparation
    ]
    events = session_builder.apply_edits(base["events"][:20], edits)
    truth = session_builder.build_recipe(dict(base, name="CS-INT-60S", events=events), tmp_path / "clip")
    assert 55_000 <= truth["duration_ms"] <= 65_000 and {s["speaker"] for s in truth["segments"]} == {
        "E",
        "C",
    }
    assert [o["end_ms"] - o["start_ms"] for o in truth["overlaps"]] == [2000]
    return tmp_path / "clip" / "session.wav", truth


def files_in(job: Path) -> set[str]:
    return {p.relative_to(job).as_posix() for p in job.rglob("*") if p.is_file()}


@pytest.mark.parametrize("child", ["speakers-pyannote", "speakers-sherpa"])
def test_each_backend_separates_the_two_voices_within_15_percent_der(
    tmp_path: Path,
    clip: tuple[Path, dict[str, Any]],
    child: str,
    tolerance: Any,
    scoring: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = clip[1]
    token = "hf_" + "Tq8" * 12  # made here, so finding it anywhere means it leaked
    monkeypatch.setenv("HF_TOKEN", token)
    record, job = speakers_job(tmp_path, clip, child)
    step = record.stages[0]
    result = read_json(job / "stages" / "speakers.json")
    # The test needs this backend: a fallback or another backend's result fails it.
    assert (step.status, step.fallback, result["backend"]) == ("ok", None, BACKENDS[child])
    value = scoring.der(truth, result["turns"])
    voices = [(s["id"], s["speech_ms"], s["kind"]) for s in result["speakers"]]
    print(f"{child}: DER {value:.4f}, speakers {voices}, similarity {result['similarity']}")
    print(f"{child}: overlap {result['overlap_ms']} ms, warnings {result['warnings']}")
    assert value <= tolerance("integration", "diarization_der_max")
    # Two different voices: the similar-voice check stays quiet, and only sherpa caps role confidence.
    assert result["similarity"] is not None and result["similarity"] < checks.SIMILAR[result["backend"]]
    assert result["role_confidence_cap"] == ("medium" if child == "speakers-sherpa" else None)
    # Embeddings stay in the child's memory: one result file, holding no long list of numbers.
    assert files_in(job) == {"job.json", "input/audio16k.wav", "stages/media.json", "stages/speakers.json",
                             "logs/job.log"}  # fmt: skip
    assert all(len(t) == 3 for t in result["turns"]) and "embedding" not in json.dumps(result)
    # The child ran offline and without the token the parent holds: not in its command, environment or log.
    env = child_env(app_root(), tmp_path / "data", models_dir(), record.job_id, cpu=True)
    argv = loop.child_argv(child, record.job_id) + offline_argv(child, record.job_id)
    assert (env["HF_HUB_OFFLINE"], env["HF_HUB_DISABLE_IMPLICIT_TOKEN"]) == ("1", "1")
    log = (job / "logs" / "job.log").read_text(encoding="utf-8")
    assert all(token not in text for text in (json.dumps(env), " ".join(argv), log, json.dumps(result)))


@pytest.mark.parametrize("child", ["speakers-pyannote", "speakers-sherpa"])
def test_exactly_two_in_settings_gives_each_backend_two_voices(
    tmp_path: Path, clip: tuple[Path, dict[str, Any]], child: str, tolerance: Any, scoring: ModuleType
) -> None:
    # pyannote gets the count; sherpa keeps its own clustering (forced to two, it joined the two people).
    record, job = speakers_job(tmp_path, clip, child, count="exactly-2")
    result = read_json(job / "stages" / "speakers.json")
    assert (record.stages[0].status, result["backend"]) == ("ok", BACKENDS[child])
    assert [s["id"] for s in result["speakers"] if s["kind"] == "main"] == ["S1", "S2"]
    value = scoring.der(clip[1], result["turns"])
    print(f"{child} with exactly 2: DER {value:.4f}")
    assert value <= tolerance("integration", "diarization_der_max")


@pytest.mark.gpu
def test_pyannote_on_the_gpu_matches_the_cpu_bar(
    tmp_path: Path, clip: tuple[Path, dict[str, Any]], tolerance: Any, scoring: ModuleType
) -> None:
    truth = clip[1]
    record, job = speakers_job(tmp_path, clip, "speakers-pyannote", device="gpu")
    result = read_json(job / "stages" / "speakers.json")
    assert (record.stages[0].device, record.stages[0].fallback, result["device"]) == ("gpu", None, "gpu")
    assert result["peak_vram_mb"] > 0 and record.stages[0].peak_vram_mb == result["peak_vram_mb"]
    value = scoring.der(truth, result["turns"])
    print(f"pyannote on the GPU: DER {value:.4f}, peak video memory {result['peak_vram_mb']} MB")
    assert value <= tolerance("integration", "diarization_der_max")


def test_pyannote_sent_to_a_hidden_card_fails_with_its_code_and_sherpa_does_not_take_over(
    tmp_path: Path, clip: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    # GPU chosen while the user hides every card: the step fails with its code and nothing runs in its place.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    record, job = speakers_job(tmp_path, clip, "speakers-pyannote", "speakers-sherpa", device="gpu")
    step = record.stages[0]
    assert (step.status, step.error_code, step.fallback) == ("failed", "E_GPU_UNAVAILABLE", None)
    result = read_json(job / "stages" / "speakers.json")
    assert (result["status"], result.get("turns")) == ("failed", None)


def test_without_pyannote_sherpa_takes_over_and_says_so(
    tmp_path: Path, clip: tuple[Path, dict[str, Any]], tolerance: Any, scoring: ModuleType
) -> None:
    # A fresh app folder that holds the no-account models only, as for a user without a token.
    app = tmp_path / "app"
    sherpa = catalogue.get("sherpa-onnx-diarizer")
    shutil.copytree(folder(sherpa, models_dir()), folder(sherpa, app / "models"))
    roots = loop.Roots(app=app, data=tmp_path / "data", models=app / "models")
    truth = clip[1]
    record, job = speakers_job(tmp_path, clip, "speakers-pyannote", "speakers-sherpa", roots=roots)
    step = record.stages[0]
    result = read_json(job / "stages" / "speakers.json")
    assert (step.status, step.fallback) == ("ok", {"to": "speakers-sherpa", "after": "E_MODEL_MISSING"})
    assert (result["backend"], result["role_confidence_cap"]) == ("sherpa-onnx-diarizer", "medium")
    assert record.state == "completed_with_gaps"  # never a silent fallback
    log = (job / "logs" / "job.log").read_text(encoding="utf-8")
    assert "[speakers] speakers-pyannote ended with E_MODEL_MISSING; running speakers-sherpa instead" in log
    assert scoring.der(truth, result["turns"]) <= tolerance("integration", "diarization_der_max")
