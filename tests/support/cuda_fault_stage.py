"""The real stage entry around a step that fails with CUDA's out-of-memory error whenever it runs on the GPU.

Usage: python -I cuda_fault_stage.py <stage_id> <job_id>. The message is the one CTranslate2 raises.
"""

import os
import sys
from collections.abc import Callable
from pathlib import Path

from fluentfolio.jobs import stage


def step(job: Path, progress: Callable[[int], None]) -> dict[str, object]:
    if os.environ.get("FLUENTFOLIO_DEVICE") == "gpu":
        raise RuntimeError("CUDA failed with error out of memory")
    progress(100)
    return {"status": "ok"}


if __name__ == "__main__":
    stage.STAGES[sys.argv[1]] = ("__main__", "step", f"stages/{sys.argv[1]}.json")
    sys.exit(stage.main(sys.argv[1:]))
