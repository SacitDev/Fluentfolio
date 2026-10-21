"""Speaker turns from pyannote community-1, in a child process with torch.

The model loads from its local folder (analysis children are offline and hold no token), and it gets the
recording as samples in memory, so it never decodes a file itself.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import soundfile as sf

from fluentfolio.core.enums import DEVICE_VAR, Device
from fluentfolio.core.errors import SetupError
from fluentfolio.core.files import read_json
from fluentfolio.core.paths import models_dir
from fluentfolio.models import catalogue
from fluentfolio.models.download import require
from fluentfolio.speakers import checks

BACKEND = "pyannote-community-1"
# Two-person modes allow 1 to 3 speakers; "exactly 2" in Settings passes the known count instead.
AUTO_COUNT = {"min_speakers": 1, "max_speakers": 3}
EXACTLY_TWO = {"num_speakers": 2}
# The progress share of each long step the pipeline reports.
STEPS = {"segmentation": (0, 30), "embeddings": (30, 95)}


def _hook(progress: Callable[[int], None]) -> Callable[..., None]:
    def hook(
        step: str, artefact: Any, file: Any = None, completed: int | None = None, total: int | None = None
    ) -> None:
        if step in STEPS and completed is not None and total:
            low, high = STEPS[step]
            progress(low + (high - low) * completed // total)

    return hook


def gpu_check() -> tuple[bool | None, float]:
    """The GPU test's work: does a product on the card match the CPU's (None: no card), and its seconds."""
    import torch

    started = time.monotonic()
    if not torch.cuda.is_available():
        return None, 0.0
    a = torch.randn(512, 512, generator=torch.Generator().manual_seed(0))
    matches = torch.allclose((a.cuda() @ a.cuda()).cpu(), a @ a, rtol=1e-3, atol=1e-2)
    return bool(matches), time.monotonic() - started


def run(job: Path, progress: Callable[[int], None]) -> dict[str, Any]:
    """The speakers step with pyannote: turns, the speaker checks and the similar-voice check."""
    record = read_json(job / "job.json")
    component = catalogue.get(BACKEND)
    folder = require(component, models_dir())  # a missing model ends the step here, before torch loads
    import torch
    from pyannote.audio import Pipeline

    samples, rate = sf.read(str(job / "input" / "audio16k.wav"), dtype="float32")
    on_gpu = os.environ.get(DEVICE_VAR) == Device.GPU
    if on_gpu and not torch.cuda.is_available():
        raise SetupError("E_GPU_UNAVAILABLE", BACKEND)
    device = torch.device("cuda" if on_gpu else "cpu")
    pipeline = Pipeline.from_pretrained(folder)
    if pipeline is None:  # the library returns None for a folder it cannot load
        raise SetupError("E_MODEL_MISSING", BACKEND)
    pipeline.to(device)
    count = EXACTLY_TWO if record["settings"]["speakers"]["count"] == "exactly-2" else AUTO_COUNT
    waveform = {"waveform": torch.from_numpy(samples)[None, :], "sample_rate": rate}
    output = pipeline(waveform, hook=_hook(progress), **count)
    annotation = output.speaker_diarization
    labels = [str(label) for label in annotation.labels()]
    raw = [(round(seg.start * 1000), round(seg.end * 1000), str(label))
           for seg, _, label in annotation.itertracks(yield_label=True)]  # fmt: skip
    turns, ids = checks.relabel(raw)
    result = checks.summarize(BACKEND, turns, mode=str(record["mode"]))
    rows = output.speaker_embeddings  # one centroid per label, in memory only
    pair = checks.main_pair(result)
    if rows is not None and pair:
        label_of = {new: old for old, new in ids.items()}
        a, b = (rows[labels.index(label_of[p])] for p in pair)
        checks.add_similarity(result, checks.cosine(a, b))
    progress(100)
    return {
        **result,
        "version": f"pyannote.audio {metadata.version('pyannote.audio')}",
        "revision": component.revision,
        "device": "gpu" if device.type == "cuda" else "cpu",
        "peak_vram_mb": int(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
    }
