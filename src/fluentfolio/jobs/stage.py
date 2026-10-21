"""The entry of every analysis child: `python -m fluentfolio.jobs.stage <stage_id> <job_id>`.

A child reads its inputs from the job folder, prints JSON progress lines on stdout and writes one result file.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from fluentfolio.core.errors import AppError
from fluentfolio.core.files import write_json
from fluentfolio.core.logs import crash_lines
from fluentfolio.core.paths import data_root, secure_umask
from fluentfolio.jobs import store

# Stage ID -> the module and function that run it, and its result file in the job folder.
STAGES = {
    "media": ("fluentfolio.media.decode", "run", "stages/media.json"),
    "vad": ("fluentfolio.asr.vad", "run", "stages/vad.json"),
    "language": ("fluentfolio.asr.language", "run", "stages/language.json"),
    "asr": ("fluentfolio.asr.base", "run", "stages/asr.json"),
    "speakers-pyannote": ("fluentfolio.speakers.pyannote", "run", "stages/speakers.json"),
    "speakers-sherpa": ("fluentfolio.speakers.sherpa", "run", "stages/speakers.json"),
    "phones": ("fluentfolio.alignment.phones", "run", "stages/phones.json"),
    "candidate_audio": ("fluentfolio.media.slices", "run", "stages/candidate_audio.json"),
    "ai_audio": ("fluentfolio.media.slices", "for_ai_audio", "stages/ai_audio.json"),
    "praat": ("fluentfolio.analyzers.praat", "run", "analyzers/praat.json"),
    "swiftf0": ("fluentfolio.analyzers.swiftf0", "run", "analyzers/swiftf0.json"),
    "opensmile": ("fluentfolio.analyzers.opensmile", "run", "analyzers/opensmile.json"),
    "nlp": ("fluentfolio.analyzers.nlp", "run", "analyzers/nlp.json"),
}


def progress(pct: int) -> None:
    # Library hooks may count in numpy integers, which json cannot write.
    print(json.dumps({"pct": int(pct)}), flush=True)


def peak_ram_mb() -> int:
    """This process's peak memory: the peak working set on Windows, the peak resident size on Linux."""
    if sys.platform == "win32":
        import psutil

        return int(psutil.Process().memory_info().peak_wset / 2**20)
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)  # Linux reports KiB


def gpu_code(error: Exception) -> str | None:
    """E_GPU_OOM or E_CUDA_RUNTIME for an error from the graphics card, so the step runs again on the CPU."""
    text = f"{type(error).__name__}: {error}".lower()
    # torch raises OutOfMemoryError; CTranslate2 and cuBLAS raise RuntimeError with the CUDA message.
    if "outofmemoryerror" in text or "alloc_failed" in text or ("out of memory" in text and "cuda" in text):
        return "E_GPU_OOM"
    if any(name in text for name in ("cuda", "cublas", "cudnn")):
        return "E_CUDA_RUNTIME"
    return None


def _end_with_parent() -> None:
    """On Linux, end this child when the app dies, even in a crash; on Windows the Job Object does it."""
    if sys.platform == "linux":
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # 1 = PR_SET_PDEATHSIG


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in STAGES:  # noqa: PLR2004
        print("usage: python -m fluentfolio.jobs.stage <stage_id> <job_id>", file=sys.stderr)
        return 2
    _end_with_parent()
    secure_umask()
    stage_id, job_id = argv
    module, name, result = STAGES[stage_id]
    job: Path = store.job_dir(data_root(), job_id)
    step = getattr(importlib.import_module(module), name)
    exit_code = 0
    try:
        outcome: dict[str, Any] = step(job, progress)
    except AppError as exc:
        outcome, exit_code = {"status": "failed", "code": exc.code, "detail": exc.detail}, 1
    except MemoryError:
        raise  # no result file, so the job loop reads the exit as out of memory
    except Exception as error:  # noqa: BLE001 - the stage entry is one of the few places that catch everything
        print(crash_lines(error), file=sys.stderr, flush=True)
        outcome, exit_code = {"status": "failed", "code": gpu_code(error) or "E_INTERNAL"}, 1
    outcome["peak_ram_mb"] = peak_ram_mb()
    write_json(job / result, outcome)
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
