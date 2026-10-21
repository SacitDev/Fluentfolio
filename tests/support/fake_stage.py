"""A stand-in for the stage child; the part of its ID before the first "-" or "@" picks its behaviour.

It starts like a real child (absolute interpreter, stage ID and job ID only, the built environment),
so the job loop and the runner are tested at the process boundary.
"""

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path


def progress(pct: int) -> None:
    print(json.dumps({"pct": pct}), flush=True)


def result(job: Path, stage_id: str, status: str, code: str | None = None, **extra: object) -> None:
    path = job / "stages" / f"{stage_id.split('@')[-1]}.json"  # "kind@step" writes the step's result
    path.write_text(json.dumps({"status": status, "code": code, **extra}), encoding="utf-8")


def gpu_out_of_memory(job: Path, stage_id: str) -> int:
    if os.environ.get("FLUENTFOLIO_DEVICE") == "cpu":  # the runner sent this run to the CPU
        result(job, stage_id, "ok")
        return 0
    result(job, stage_id, "failed", "E_GPU_OOM")
    return 1


def spawn(job: Path, stage_id: str) -> int:
    """A grandchild, then silence: the watchdog or a cancel must end both."""
    grandchild = subprocess.Popen([sys.executable, "-I", __file__, "sleep", job.name])
    (job / "grandchild.pid").write_text(str(grandchild.pid), encoding="utf-8")
    progress(10)
    time.sleep(300)
    return 0


def slow(job: Path, stage_id: str) -> int:
    """Keeps reporting progress, so only the backstop can end it."""
    while True:
        progress(50)
        time.sleep(0.1)


def chatty(job: Path, stage_id: str) -> int:
    for n in range(300):
        print(f"line {n}", file=sys.stderr)
    progress(50)
    progress(100)
    return 0


def raise_memory_error(job: Path, stage_id: str) -> int:
    raise MemoryError


def leak(job: Path, stage_id: str) -> int:
    """Prints the canary the test planted in the job folder; the environment carries nothing extra."""
    print((job / "canary.txt").read_text(encoding="utf-8"), file=sys.stderr)
    result(job, stage_id, "ok")
    return 0


def failed(job: Path, stage_id: str) -> int:
    result(job, stage_id, "failed", "E_CUDA_RUNTIME")
    return 1


BEHAVIOURS: dict[str, Callable[[Path, str], int]] = {
    "ok": lambda job, s: result(job, s, "ok", peak_ram_mb=12) or 0,
    "partial": lambda job, s: result(job, s, "partial") or 0,
    "unavailable": lambda job, s: result(job, s, "unavailable", "E_COMPONENT_UNAVAILABLE") or 0,
    "failed": failed,
    "silent": lambda job, s: result(job, s, "failed", "E_INPUT_SILENT") or 4,  # a refused recording
    "gpufail": failed,
    "nogpu": lambda job, s: result(job, s, "failed", "E_GPU_UNAVAILABLE") or 1,  # a forced GPU with no card
    "crash": lambda job, s: 3,
    "memory": raise_memory_error,
    "gpuoom": gpu_out_of_memory,
    "spawn": spawn,
    "slow": slow,
    "chatty": chatty,
    "leak": leak,
    "env": lambda job, s: result(job, s, "ok", env=dict(os.environ)) or 0,
    "job": lambda job, s: result(job, s, "ok", record=json.loads((job / "job.json").read_text("utf-8"))) or 0,
    "stdin": lambda job, s: len(sys.stdin.read()) * 0,  # lives until its stdin closes
    "sleep": lambda job, s: time.sleep(300) or 0,
}


def main(stage_id: str, job_id: str) -> int:
    job = Path(os.environ["FLUENTFOLIO_HOME"]) / "jobs" / job_id
    kind = stage_id.split("@")[0].split("-")[0]
    if kind != "sleep":
        with (job / "calls.txt").open("a", encoding="utf-8") as calls:
            calls.write(stage_id + "\n")
        progress(0)
    return BEHAVIOURS[kind](job, stage_id)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
