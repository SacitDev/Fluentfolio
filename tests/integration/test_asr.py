"""Voice activity and speech recognition with the real models, against construction and LibriSpeech text."""

import json
import sys
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from types import ModuleType
from typing import Any

import jiwer
import numpy as np
import pytest
import soundfile as sf

from fluentfolio import __version__
from fluentfolio.app.facade import DEFAULT_ASR
from fluentfolio.asr import base, vad
from fluentfolio.core.enums import JobState
from fluentfolio.core.errors import AppError
from fluentfolio.core.files import read_json, write_json
from fluentfolio.core.paths import app_root, models_dir
from fluentfolio.ingest import upload
from fluentfolio.jobs import loop, store
from fluentfolio.models import catalogue
from fluentfolio.models.download import require
from fluentfolio.settings.model import Settings
from fluentfolio.settings.store import snapshot

pytestmark = pytest.mark.slow

RATE = 16000
GAP_S = 0.7  # the pause put between LibriSpeech utterances
OFFLINE = Path(__file__).parents[1] / "support" / "offline_stage.py"
STAGES = [loop.Stage(id=s, result=f"stages/{s}.json") for s in ("media", "vad", "asr")]
WORD_FIELDS = {
    "text", "raw", "start_ms", "end_ms", "confidence", "confidence_reason", "is_filler", "cut_off", "model",
    "flags", "chunk",
}  # fmt: skip


@pytest.fixture(scope="module")
def silero() -> Any:
    c = catalogue.get("silero-vad")
    return vad.session(require(c, Path("unused")) / c.file[0].path)


def write(path: Path, x: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, RATE, subtype="PCM_16")
    return path


def test_vad_edges_on_the_voices_clip(test_data: Path, silero: Any, silence: ModuleType) -> None:
    clip = test_data / "voices" / "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
    assert sf.info(str(clip)).samplerate == RATE
    first, last = silence.speech_span(clip)
    found = vad.segments(silero, clip)
    print(f"VAD {found}, silencedetect at {silence.NOISE_DB} dB {first}-{last} ms")
    assert len(found) == 1
    ((start, end),) = found
    # The detector pads speech, so each edge may lie up to 100 ms outside silencedetect's and 50 ms inside.
    assert first - 100 <= start <= first + 50 and last - 50 <= end <= last + 100  # noqa: PLR2004


def test_reading_in_blocks_gives_the_same_probabilities(test_data: Path, silero: Any) -> None:
    x, _ = sf.read(test_data / "voices" / "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav", dtype="float32")
    whole = vad.probabilities(silero, [x])
    step = vad.FRAME * 7  # a long recording is read in blocks; here 16 of them
    blocks = vad.probabilities(silero, [x[i : i + step] for i in range(0, len(x), step)])
    assert len(blocks) == len(whole) == -(-len(x) // vad.FRAME)
    assert np.allclose(blocks, whole, atol=1e-4)


def test_an_inserted_700_ms_pause_is_measured_within_tolerance(
    tmp_path: Path, silero: Any, session_builder: ModuleType, tolerance: Any
) -> None:
    recipe = {
        "name": "two-runs", "mode": "solo", "seed": 7, "speakers": {"L": "one reader"},
        "turns": {"a": {"speaker": "L", "kind": "answer", "part": 1}},
        "events": [
            {"clip": "ls_4446_2271_0000", "turn": "a", "gap_ms": 1000},
            {"clip": "ls_4446_2271_0003", "turn": "a", "gap_ms": 700},
        ],
    }  # fmt: skip
    truth = session_builder.build_recipe(recipe, tmp_path / "s")
    first, second = truth["segments"]
    assert second["start_ms"] - first["end_ms"] == 700
    found = vad.segments(silero, tmp_path / "s" / "session.wav")
    middle = (first["end_ms"] + second["start_ms"]) / 2
    before = max(b for a, b in found if b <= middle)
    after = min(a for a, b in found if a >= middle)
    assert abs((after - before) - 700) <= tolerance("integration", "vad_inserted_pause_ms")


def pink(n: int, seed: int, rms: float) -> np.ndarray:
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    spectrum[1:] /= np.sqrt(np.fft.rfftfreq(n, 1 / RATE)[1:])  # power falls as 1/f
    spectrum[0] = 0
    x = np.fft.irfft(spectrum, n=n)
    return x / np.sqrt(np.mean(x**2)) * rms


def chord(seconds: float, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return sum(amp * np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.0))


def test_silence_noise_or_a_chord_hold_no_speech(tmp_path: Path) -> None:
    n = 20 * RATE
    signals = {
        "silence": np.zeros(n),
        "pink": pink(n, 1, 0.03),
        "white": np.random.default_rng(2).standard_normal(n) * 0.03,
        "chord": chord(20.0),
    }
    codes = {}
    for name, x in signals.items():
        job = tmp_path / name
        write(job / "input" / "audio16k.wav", x)
        write_json(job / "job.json", {"layout": "mixed"})  # every job folder has its record
        try:
            vad.run(job, lambda pct: None)
            codes[name] = "ok"
        except AppError as error:
            codes[name] = error.code
    assert codes == dict.fromkeys(signals, "E_INPUT_SILENT")


def chapters(test_data: Path) -> list[Path]:
    root = test_data / "librispeech" / "LibriSpeech" / "test-clean"
    return sorted(p for p in root.glob("*/*") if p.is_dir())


def utterances(chapter: Path) -> Iterator[tuple[np.ndarray, str]]:
    """A chapter's recordings and their reference text, in file order."""
    for line in next(chapter.glob("*.trans.txt")).read_text(encoding="utf-8").splitlines():
        utterance, text = line.split(" ", 1)
        audio, rate = sf.read(chapter / f"{utterance}.flac", dtype="float32")
        assert rate == RATE
        yield audio, text


def clip30(test_data: Path) -> tuple[np.ndarray, str]:
    """The first chapter whose whole utterances, 0.7 s apart, fill 27 to 30 s (at least three of them)."""
    for chapter in chapters(test_data):
        parts: list[np.ndarray] = []
        texts: list[str] = []
        seconds = 0.0
        for audio, text in utterances(chapter):
            add = len(audio) / RATE + (GAP_S if parts else 0.0)
            if seconds + add > 30.0:  # noqa: PLR2004
                break
            if parts:
                parts.append(np.zeros(int(GAP_S * RATE), dtype=np.float32))
            parts.append(audio)
            texts.append(text)
            seconds += add
        if len(texts) >= 3 and seconds >= 27.0:  # noqa: PLR2004
            return np.concatenate(parts), " ".join(texts)
    raise AssertionError("no test-clean chapter fills 27 to 30 s")


def run_chain(
    tmp_path: Path,
    source: Path,
    model: str = DEFAULT_ASR,
    device: str = "cpu",
    stages: list[loop.Stage] = STAGES,
) -> tuple[store.JobRecord, Path]:
    """Ingest a file and run media, vad and asr as a job would, in children that cannot reach the network."""
    roots = loop.Roots(app=app_root(), data=tmp_path / "data", models=models_dir())
    now = store.utc_now()
    record = store.JobRecord(
        job_id=store.new_job_id(), created_utc=now, updated_utc=now, app_version=__version__,
        device={"requested": device, "resolved": device}, settings=snapshot(Settings(), asr=model),
    )  # fmt: skip
    job = store.create(roots.data, record)
    record.input = upload.store_input(job, source, Settings().limits)
    store.write(roots.data, record)
    done = loop.run_job(roots, job.name, stages, audio_min=1.0, no_progress_s=300, argv_for=offline_argv)
    return done, job


def offline_argv(stage_id: str, job_id: str) -> list[str]:
    return [sys.executable, "-I", "-X", "utf8", str(OFFLINE), stage_id, job_id]


def norm(text: str) -> str:
    return " ".join("".join(c if c.isalnum() or c == "'" else " " for c in text.lower()).split())


LIBRARIES = {"onnx-asr": {"onnx-asr", "onnxruntime"}, "faster-whisper": {"faster-whisper", "ctranslate2"}}


@pytest.mark.parametrize(
    ("model", "device"),
    [
        ("parakeet-tdt-0.6b-v2", "cpu"),
        ("parakeet-tdt-0.6b-v3", "cpu"),
        ("whisper-large-v3-turbo", "cpu"),
        pytest.param("whisper-large-v3-turbo", "gpu", marks=pytest.mark.gpu),
    ],
)
def test_each_speech_model_on_30_s_of_librispeech_through_the_offline_children(
    tmp_path: Path, test_data: Path, tolerance: Any, model: str, device: str
) -> None:
    audio, reference = clip30(test_data)
    stages = [*STAGES[:2], loop.Stage("asr", "stages/asr.json", gpu=device == "gpu")]
    record, job = run_chain(tmp_path, write(tmp_path / "in" / "clip30.wav", audio), model, device, stages)
    assert record.state is JobState.COMPLETED, [(s.id, s.error_code) for s in record.stages]
    assert (record.stages[-1].device, record.stages[-1].fallback) == (device, None)
    result = read_json(job / "stages" / "asr.json")
    words = result["words"]
    wer = jiwer.wer(norm(reference), norm(" ".join(w["text"] for w in words)))
    print(f"{model} on the {device}: WER {wer:.4f} over {len(words)} words")
    assert wer <= tolerance("integration", "asr_wer_clip_max"), wer
    duration_ms = len(audio) * 1000 // RATE
    starts = [w["start_ms"] for w in words]
    assert starts == sorted(starts)
    assert all(0 <= w["start_ms"] <= w["end_ms"] <= duration_ms for w in words)
    assert all(set(w) == WORD_FIELDS and w["model"] == model for w in words)
    assert all(0 < w["confidence"] <= 1 and w["confidence_reason"] is None for w in words)
    assert all(b - a <= base.MAX_CHUNK_MS for a, b in result["chunks"])
    component = catalogue.get(model)
    record_ = result["provenance"]
    assert (record_["model"], record_["revision"], record_["device"], record_["fallback"]) == (
        model, component.revision, device, None,
    )  # fmt: skip
    assert (record_["transcript_style"], record_["settings"].get("initial_prompt")) == (
        component.transcript_style, "None" if component.engine == "faster-whisper" else None,
    )  # fmt: skip
    assert set(record_["engine"]) == LIBRARIES[str(component.engine)] and record_["wall_ms"] > 0
    assert result["peak_ram_mb"] > 0


def test_the_language_check_marks_spanish_and_not_english(
    tmp_path: Path, test_data: Path, session_builder: ModuleType
) -> None:
    english, _ = next(utterances(chapters(test_data)[0]))
    spanish = session_builder.load_clip(Path(__file__).parents[1] / "fixtures" / "audio" / "es_park.opus")
    gap = np.zeros(2 * RATE, dtype=np.float32)
    audio = np.concatenate([english, gap, spanish, gap, english])
    first_end = len(english) * 1000 // RATE
    spanish_at = (len(english) + len(gap)) * 1000 // RATE
    spanish_end = spanish_at + len(spanish) * 1000 // RATE
    assert first_end >= 3000 and spanish_end - spanish_at >= 3000  # each part is long enough to be judged
    stages = [*STAGES[:2], loop.Stage("language", "stages/language.json")]
    record, job = run_chain(tmp_path, write(tmp_path / "in" / "mixed.wav", audio), stages=stages)
    assert record.state is JobState.COMPLETED, [(s.id, s.error_code) for s in record.stages]
    result = read_json(job / "stages" / "language.json")
    rows = result["segments"]
    print([(s["start_ms"], s["end_ms"], s["language"], s["probability"], s["english"]) for s in rows])
    inside = [s for s in rows if s["start_ms"] < spanish_end and s["end_ms"] > spanish_at]
    outside = [s for s in rows if s not in inside]
    assert inside and all(s["marked"] and s["language"] == "es" for s in inside)
    assert outside and not any(s["marked"] for s in outside)
    assert result["non_english_ms"] == sum(s["end_ms"] - s["start_ms"] for s in inside)
    assert result["provenance"]["model"] == "whisper-tiny"


def test_no_words_in_digital_silence_and_flags_on_words_in_noise_or_music(
    tmp_path: Path, test_data: Path
) -> None:
    (a, _), (b, _), (c, _), (d, _) = list(islice(utterances(chapters(test_data)[1]), 4))
    # The music and noise are short enough to fall inside a chunk, so the model hears them.
    pieces = [
        ("speech", a), ("silence", np.zeros(4 * RATE)), ("speech", b), ("music", chord(0.8)),
        ("speech", c), ("noise", pink(int(0.8 * RATE), 3, 0.03)), ("speech", d),
    ]  # fmt: skip
    spans, at = {}, 0
    for kind, x in pieces:
        spans.setdefault(kind, []).append((at * 1000 // RATE, (at + len(x)) * 1000 // RATE))
        at += len(x)
    record, job = run_chain(
        tmp_path, write(tmp_path / "in" / "mixed.wav", np.concatenate([x for _, x in pieces]))
    )
    assert record.state is JobState.COMPLETED, [(s.id, s.error_code) for s in record.stages]
    result = read_json(job / "stages" / "asr.json")
    words = result["words"]
    # Words may run a little past the speech; none may reach the inner part of the silence.
    ((s0, s1),) = spans["silence"]
    assert not [w for w in words if w["end_ms"] > s0 + 250 and w["start_ms"] < s1 - 250]
    # Any text heard inside the noise or music is flagged; 100 ms covers the VAD's edge error.
    for kind in ("music", "noise"):
        ((n0, n1),) = spans[kind]
        assert any(a <= n0 and n1 <= b for a, b in result["chunks"]), kind
        inside = [w for w in words if w["start_ms"] >= n0 + 100 and w["end_ms"] <= n1 - 100]
        assert all(base.HALLUCINATION in w["flags"] for w in inside), json.dumps(inside)
