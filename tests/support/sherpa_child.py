"""Run sherpa-onnx speaker separation on one recording in its own process and save numbers only.

Usage: python -I sherpa_child.py <audio> <models folder> <result json>
On an error it prints only the error's class name, so no path or file name reaches a log.
"""

import json
import os
import sys
import time
from pathlib import Path


def decode(path: Path):
    """Decode any audio or video file to 16 kHz mono float32 samples."""
    import av
    import numpy as np

    chunks = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def main(audio: Path, models: Path, result: Path) -> None:
    from fluentfolio.speakers.sherpa import SAMPLE_RATE, diarize

    samples = decode(audio)
    started = time.perf_counter()
    turns = diarize(samples, models)
    took = time.perf_counter() - started
    out = {
        "duration_s": samples.size / SAMPLE_RATE,
        "process_s": took,
        "turns": [[start / 1000, end / 1000, speaker] for start, end, speaker in turns],
        "token_variables": sum("TOKEN" in k.upper() for k in os.environ),
        "onnxruntime_loaded": "onnxruntime" in sys.modules,
    }
    result.write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    except BaseException as exc:  # noqa: BLE001 - the class name only, never a message that may hold a path
        print(type(exc).__name__, file=sys.stderr)
        sys.exit(3)
