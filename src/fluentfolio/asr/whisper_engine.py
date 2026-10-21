"""Whisper large-v3-turbo through faster-whisper: int8 on the CPU, float16 on the GPU, and never a prompt.

Whisper writes smoothed text: it leaves out many hesitations and repairs, so its transcript is not grammar
evidence and its filler count is a lower bound. No initial prompt is given, because a prompt full of fillers
makes the model invent them.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from fluentfolio.asr.base import Heard
from fluentfolio.core.enums import DEVICE_VAR, Device
from fluentfolio.core.errors import SetupError
from fluentfolio.models.catalogue import Component

SEED = 0  # the fallback temperatures sample, so a fixed seed makes a rerun give the same words
OPTIONS: dict[str, Any] = {
    "language": "en",
    "task": "transcribe",
    "beam_size": 5,
    "word_timestamps": True,
    "vad_filter": False,  # every chunk is already cut at the pauses the app's own voice detector found
    "condition_on_previous_text": False,  # earlier text would carry a mistake or a loop forward
    "initial_prompt": None,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 2.4,  # text this repetitive is decoded again at a higher temperature
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,  # with a low log-probability too, the window is taken as silence
}


def _load_cublas() -> None:
    """Make the cuBLAS library from the nvidia-cublas-cu12 wheel findable before CTranslate2 loads."""
    try:
        spec = importlib.util.find_spec("nvidia.cublas")
    except ModuleNotFoundError:  # the CPU build has no nvidia package, and find_spec imports the parent
        spec = None
    if spec is None or not spec.submodule_search_locations:
        raise SetupError("E_GPU_UNAVAILABLE", "the GPU build of Fluentfolio is not installed")
    root = Path(next(iter(spec.submodule_search_locations)))
    if sys.platform == "win32":
        os.add_dll_directory(str(root / "bin"))
        os.environ["PATH"] = str(root / "bin") + os.pathsep + os.environ.get("PATH", "")
    else:  # the loader reads LD_LIBRARY_PATH only at start, so the libraries are loaded by path instead
        for name in ("libcublasLt.so.12", "libcublas.so.12"):
            ctypes.CDLL(str(root / "lib" / name), mode=ctypes.RTLD_GLOBAL)


class WhisperEngine:
    """A CTranslate2 Whisper model. Its words carry times from cross-attention and a probability each."""

    precision_ms = 20  # one step of the audio encoder
    confidence = "the model's probability for the word"
    flags_repeats = True
    timed = True
    dropped = 0  # its loops are flagged, never dropped

    def __init__(self, component: Component, folder: Path) -> None:
        self.device = Device.GPU if os.environ.get(DEVICE_VAR) == Device.GPU else Device.CPU
        if self.device is Device.GPU:
            _load_cublas()
        import ctranslate2
        from faster_whisper import WhisperModel

        if self.device is Device.GPU and ctranslate2.get_cuda_device_count() == 0:
            raise SetupError("E_GPU_UNAVAILABLE", component.id)
        quantization = "float16" if self.device is Device.GPU else "int8"
        ctranslate2.set_random_seed(SEED)
        self.model = WhisperModel(
            str(folder), device="cuda" if self.device is Device.GPU else "cpu", compute_type=quantization,
            local_files_only=True,
        )  # fmt: skip
        self.settings = {"quantization": quantization, "seed": str(SEED)}
        self.settings.update({name: str(value) for name, value in OPTIONS.items()})

    def versions(self) -> dict[str, str]:
        return {name: metadata.version(name) for name in ("faster-whisper", "ctranslate2")}

    def heard(self, audio: np.ndarray) -> list[Heard]:
        """The words of one chunk as the model wrote them, with times from the chunk's start."""
        segments, _ = self.model.transcribe(audio, **OPTIONS)
        return [
            Heard(w.word, w.start, w.end, w.probability) for segment in segments for w in segment.words or []
        ]
