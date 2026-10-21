"""Where an analysis runs: the GPU only when the build, the card, its driver and a short test all allow it.

Run as `python -I -m fluentfolio.setup.device` it is the GPU test itself, in a process of its own.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fluentfolio.core import paths
from fluentfolio.core.enums import Device
from fluentfolio.core.files import write_json
from fluentfolio.jobs import runner
from fluentfolio.jobs.stage import gpu_code
from fluentfolio.jobs.store import utc_now

MIN_COMPUTE = (7, 5)
MIN_MEMORY_MB = 4096
MIN_DRIVER = 580
PROBE_S = 10.0
SMOKE_S = 10.0  # the GPU work itself; starting Python and torch may take longer on a cold disk
SMOKE_START_S = 120.0
QUERY = ("--query-gpu=compute_cap,memory.total,driver_version", "--format=csv,noheader,nounits")
NO_GPU_STEP = "No step of this analysis uses the GPU."
CHOSEN = {Device.CPU: "You chose the CPU.", Device.GPU: "You chose the GPU."}
PASSED = "The graphics card passed every check."


@dataclass(frozen=True)
class Card:
    compute: tuple[int, int]
    memory_mb: int
    driver: str


def parse(lines: list[str]) -> list[Card]:
    """The cards in nvidia-smi's answer ("8.6, 24576, 590.48" per card); other lines are ignored."""
    cards = []
    for line in lines:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != len(QUERY) + 1:
            continue
        try:
            major, minor = fields[0].split(".")
            cards.append(Card((int(major), int(minor)), int(float(fields[1])), fields[2]))
        except ValueError:
            continue
    return cards


def nvidia_smi() -> Path | None:
    """The NVIDIA driver's own tool, at the fixed places its installers put it; never searched on PATH."""
    if sys.platform == "win32":
        root = os.environ.get("SYSTEMROOT")  # Windows always sets it; without it there is no place to look
        places = [Path(root) / "System32" / "nvidia-smi.exe"] if root else []
    else:
        places = [Path("/usr/bin/nvidia-smi"), Path("/usr/lib/wsl/lib/nvidia-smi")]
    return next((p for p in places if p.is_file()), None)


def probe() -> list[Card]:
    """The NVIDIA cards this computer has, as the driver reports them; none when it cannot tell."""
    exe = nvidia_smi()
    if exe is None:
        return []
    # On Windows the tool cannot start the driver's library (NVML) without PROGRAMFILES in its environment.
    env = {name: os.environ[name] for name in ("SYSTEMROOT", "PROGRAMFILES") if name in os.environ}
    try:
        done = runner.run([str(exe), *QUERY], env=env, timeout_s=PROBE_S, no_progress_s=PROBE_S)
    except OSError:
        return []
    # The runner keeps every line that is not a progress event, which here is the tool's whole answer.
    return parse(done.stderr_tail) if done.exit_code == 0 and done.stopped is None else []


def hidden() -> bool:
    """True when CUDA_VISIBLE_DEVICES is set to hide every card from the programs this one starts.

    An empty value counts, as it does on Linux, although CUDA on Windows may still see the card through it.
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    return value is not None and value.strip() in ("", "-1")


def card_problem(found: list[Card]) -> str | None:
    """Why the card cannot run the GPU steps, as one plain sentence; None when it can."""
    if not found:
        return "No NVIDIA graphics card found."
    card = found[0]  # Known limit: only the first card is checked; revisit if a two-card user asks.
    if card.compute < MIN_COMPUTE:
        return f"This graphics card is too old (compute {card.compute[0]}.{card.compute[1]}; 7.5 is needed)."
    if card.memory_mb < MIN_MEMORY_MB:
        return "The graphics card has less than 4 GB of video memory."
    major = card.driver.split(".")[0]
    if not major.isdigit() or int(major) < MIN_DRIVER:
        return f"Graphics driver {card.driver} is older than {MIN_DRIVER}. Update the driver."
    return None


def why_not_gpu(state: dict[str, Any], cards: Callable[[], list[Card]], hidden: bool) -> str | None:
    """Why Auto cannot use the GPU, as one plain sentence; None when every condition holds."""
    if state.get("backend") != "cu130":
        return "GPU support is not installed. Run the installer again and choose GPU."
    if hidden:
        return "CUDA_VISIBLE_DEVICES hides every graphics card."
    if problem := card_problem(cards()):
        return problem
    smoke = state.get("gpu_smoke")
    if not isinstance(smoke, dict):
        return "The GPU test has not run yet. Using the CPU."
    if smoke.get("passed") is not True:
        return f"The GPU test failed ({smoke.get('code')}). Using the CPU."
    return None


def resolve(requested: Device, gpu_steps: bool, cards: Callable[[], list[Card]] = probe) -> dict[str, str]:
    """A job's device record: what was asked for, where the GPU-capable steps run, and why."""
    if not gpu_steps:
        resolved, reason = Device.CPU, NO_GPU_STEP
    elif requested in CHOSEN:  # GPU is forced: a step that finds no usable card fails with E_GPU_UNAVAILABLE
        resolved, reason = requested, CHOSEN[requested]
    else:
        why = why_not_gpu(paths.install_state(paths.app_root()), cards, hidden())
        resolved, reason = (Device.CPU, why) if why else (Device.GPU, PASSED)
    return {"requested": requested.value, "resolved": resolved.value, "reason": reason}


def check_gpu(app: Path) -> dict[str, Any]:
    """Run the GPU test in a child and keep its result in state.json, where Auto reads it."""
    codes: list[str] = []
    env = runner.child_env(app, paths.data_root(), paths.models_dir(), None, cpu=False)
    argv = [sys.executable, "-I", "-X", "utf8", "-m", "fluentfolio.setup.device"]
    done = runner.run(
        argv, env=env, timeout_s=SMOKE_START_S, no_progress_s=SMOKE_START_S,
        on_progress=lambda event: codes.append(str(event.get("code"))),
    )  # fmt: skip
    passed = done.exit_code == 0 and done.stopped is None and not codes
    code = (
        None if passed else (codes[-1] if codes else "E_STAGE_TIMEOUT" if done.stopped else "E_STAGE_CRASHED")
    )
    result = {"passed": passed, "code": code, "utc": utc_now()}
    state = paths.install_state(app)
    state["gpu_smoke"] = result
    write_json(app / "state.json", state)
    return result


def _smoke() -> int:
    """The GPU test, in the library family the speaker step uses on the card: torch."""
    from fluentfolio.speakers.pyannote import gpu_check

    try:
        matches, seconds = gpu_check()
        if matches is None:
            code: str | None = "E_GPU_UNAVAILABLE"
        else:
            code = None if matches else "E_CUDA_RUNTIME"
            if code is None and seconds > SMOKE_S:
                code = "E_STAGE_TIMEOUT"
    except Exception as error:  # noqa: BLE001 - every failure of the test is a result, never a crash
        code = gpu_code(error) or "E_INTERNAL"
    if code:
        print(json.dumps({"code": code}), flush=True)
    return 1 if code else 0


if __name__ == "__main__":
    sys.exit(_smoke())
