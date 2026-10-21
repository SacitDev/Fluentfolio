"""Parakeet and Canary through onnx-asr on the CPU, loaded only from the pinned files on this computer."""

from __future__ import annotations

import math
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from fluentfolio.asr import normalize
from fluentfolio.asr.base import RATE, Heard, repeated_runs
from fluentfolio.models.catalogue import Component

FRAME_S = 0.08  # one encoder step: onnx-asr's 10 ms feature window times the model's 8x subsampling
# Canary's decoder can loop on unclear audio: a segment with more tokens a second than this, or a 4-word
# run said 3 times or more, is dropped whole (Whisper's loops are flagged instead, as its guards differ).
LOOP_TOKENS_PER_S = 8.0


def _spoken(token: str) -> bool:
    return any(c.isalnum() for c in token)


class OnnxEngine:
    """A NeMo model exported to ONNX. Its tokens carry log-probabilities, and Parakeet's carry start times."""

    device = "cpu"
    precision_ms = 80
    confidence = "geometric mean of the word's token probabilities"
    settings = {"quantization": "int8"}
    flags_repeats = False  # the guard against looping text is for the attention decoders
    timed = True  # Parakeet's tokens carry start times; Canary's carry none
    dropped = 0  # segments the loop guard dropped

    def __init__(self, component: Component, folder: Path) -> None:
        import onnx_asr
        import onnxruntime

        onnxruntime.disable_telemetry_events()  # the Windows trace path ignores ORT_DISABLE_TELEMETRY
        model: Any = onnx_asr.load_model(
            str(component.engine_model), path=folder, quantization="int8", providers=["CPUExecutionProvider"]
        )
        self.model = model.with_timestamps()
        # A model that needs the word-time step gives no times of its own: Canary, an attention decoder.
        self.timed = "asr_times" not in component.stages
        self.flags_repeats = not self.timed

    def versions(self) -> dict[str, str]:
        return {name: metadata.version(name) for name in ("onnx-asr", "onnxruntime")}

    def heard(self, audio: np.ndarray) -> list[Heard]:
        """The words of one chunk as the model wrote them, with times from the chunk's start.

        Without times from the model, each word spans the whole chunk until the CTC aligner times it.
        """
        result = self.model.recognize(audio, sample_rate=RATE)
        tokens: list[str] = list(result.tokens or [])
        times: list[float] = list(result.timestamps or [])
        logprobs: list[float] = list(result.logprobs or [])
        groups: list[list[int]] = []
        for i, token in enumerate(tokens):
            if not groups or token.startswith(" "):  # a leading space starts a new word
                groups.append([])
            groups[-1].append(i)
        end = len(audio) / RATE
        words = []
        for k, group in enumerate(groups):
            spoken = [i for i in group if _spoken(tokens[i])] or group
            confidence = math.exp(sum(logprobs[i] for i in spoken) / len(spoken)) if logprobs else None
            raw = "".join(tokens[i] for i in group)
            if not self.timed:
                words.append(Heard(raw, 0.0, end, confidence))
                continue
            # The model emits a word's punctuation late, often inside the pause that follows, so the
            # word ends one step after its last spoken token, and never after the next word starts.
            following = times[groups[k + 1][0]] if k + 1 < len(groups) else end
            start = times[group[0]]
            stop = max(start, min(times[spoken[-1]] + FRAME_S, following, end))
            words.append(Heard(raw, start, stop, confidence))
        if not self.timed and self._loops(tokens, words, end):
            self.dropped += 1
            return []
        return words

    @staticmethod
    def _loops(tokens: list[str], words: list[Heard], seconds: float) -> bool:
        if len(tokens) > LOOP_TOKENS_PER_S * seconds:
            return True
        return bool(repeated_runs([normalize.split(w.raw)[0].lower() for w in words]))
