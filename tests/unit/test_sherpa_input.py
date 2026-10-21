"""The sherpa-onnx speaker separation refuses input it cannot use, before any model loads."""

import sys
from pathlib import Path

import numpy as np
import pytest

from fluentfolio.speakers.sherpa import diarize


@pytest.mark.parametrize(
    "samples",
    [np.zeros(0, dtype=np.float32), np.zeros((2, 16000), dtype=np.float32)],
    ids=["empty", "stereo"],
)
def test_unusable_signal_is_refused(samples: np.ndarray, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mono"):
        diarize(samples, tmp_path)
    assert "sherpa_onnx" not in sys.modules
