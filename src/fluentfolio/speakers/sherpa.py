"""Speaker turns from sherpa-onnx, the speaker separation that needs no account, in its own child process.

In solo mode the same child only checks for a second voice.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from fluentfolio.core.enums import Mode
from fluentfolio.core.errors import SetupError
from fluentfolio.core.files import read_json
from fluentfolio.core.paths import models_dir
from fluentfolio.models import catalogue
from fluentfolio.models.download import require
from fluentfolio.speakers import checks
from fluentfolio.speakers.ownership import Turn, exclusive

BACKEND = "sherpa-onnx-diarizer"
SAMPLE_RATE = 16000
SEGMENTATION = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMBEDDER = "wespeaker_en_voxceleb_resnet34_LM.onnx"
# Cosine-distance threshold for joining clusters, and the shortest speech and pause kept.
CLUSTER_THRESHOLD = 0.5
MIN_SPEECH_S = 0.3
MIN_PAUSE_S = 0.5
THREADS = 4
EMBED_MS = 60000  # at most this much of each voice goes into its embedding for the similar-voice check
SKIPPED = "The second-voice check needs the sherpa-onnx speaker separation, which is not installed."


def diarize(samples: np.ndarray, folder: Path, threads: int = THREADS) -> list[tuple[int, int, int]]:
    """Speaker turns of 16 kHz mono samples in -1..1, as (start_ms, end_ms, cluster) in time order."""
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("expected a non-empty mono signal")
    # sherpa-onnx brings its own ONNX Runtime, so it runs in a process without the onnxruntime package.
    import sherpa_onnx

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(folder / SEGMENTATION)
            ),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(folder / EMBEDDER), num_threads=threads
        ),
        # Known limit: "exactly 2" in Settings is not passed on here: forced to two clusters, this clustering
        # joined both people of a test clip and kept a 3 s stray cluster apart; revisit with a new release.
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=CLUSTER_THRESHOLD),
        min_duration_on=MIN_SPEECH_S,
        min_duration_off=MIN_PAUSE_S,
    )
    if not config.validate():
        raise SetupError("E_MODEL_MISSING", BACKEND)
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    result = diarizer.process(np.ascontiguousarray(samples, dtype=np.float32)).sort_by_start_time()
    return [(round(s.start * 1000), round(s.end * 1000), int(s.speaker)) for s in result]


def embedding(samples: np.ndarray, spans: list[tuple[int, int]], folder: Path) -> np.ndarray | None:
    """One voice's embedding from up to a minute of its own speech; it stays in memory and is never saved."""
    import sherpa_onnx

    pieces, used = [], 0
    for a, b in spans:
        take = min(b - a, EMBED_MS - used)
        if take <= 0:
            break
        pieces.append(samples[a * SAMPLE_RATE // 1000 : (a + take) * SAMPLE_RATE // 1000])
        used += take
    if not pieces:
        return None
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(folder / EMBEDDER), num_threads=THREADS)
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    stream = extractor.create_stream()
    stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32))
    stream.input_finished()
    return np.asarray(extractor.compute(stream), dtype=np.float64) if extractor.is_ready(stream) else None


def run(job: Path, progress: Callable[[int], None]) -> dict[str, Any]:
    """The speakers step with sherpa-onnx: turns and their checks, or in solo mode the second-voice check."""
    mode = str(read_json(job / "job.json")["mode"])
    try:
        folder = require(catalogue.get(BACKEND), models_dir())
    except SetupError:
        if mode == Mode.SOLO:
            return {"status": "skipped", "backend": "single", "mode": mode, "note": SKIPPED, "warnings": []}
        raise
    samples, _ = sf.read(str(job / "input" / "audio16k.wav"), dtype="float32")
    progress(5)
    raw: list[Turn] = [(a, b, str(k)) for a, b, k in diarize(samples, folder)]
    progress(80)
    if mode == Mode.SOLO:
        result = checks.second_voice(BACKEND, raw)
    else:
        turns, _ = checks.relabel(raw)
        result = checks.summarize(BACKEND, turns, mode=mode)
        pair = checks.main_pair(result)
        if pair:
            own = exclusive(turns)
            a, b = (embedding(samples, [(s, e) for s, e, k in own if k == p], folder) for p in pair)
            checks.add_similarity(result, None if a is None or b is None else checks.cosine(a, b))
        # The no-account backend separates voices less reliably, so roles from it are at most "medium".
        result["role_confidence_cap"] = "medium"
    progress(100)
    return {**result, "version": f"sherpa-onnx {metadata.version('sherpa-onnx')}", "device": "cpu"}
