"""The installer's steps after its bootstrap script: `python -I -m fluentfolio.setup device|finish`.

The bootstrap installs the CPU build. `device` switches to GPU support when the card allows it and the GPU
test passes; `finish` adds openSMILE once the learner has accepted its licence, and records the install.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from fluentfolio import __version__
from fluentfolio.analyzers.opensmile import consented
from fluentfolio.core import paths
from fluentfolio.core.enums import Device
from fluentfolio.core.files import write_json
from fluentfolio.jobs import runner
from fluentfolio.jobs.store import utc_now
from fluentfolio.setup import device as gpu
from fluentfolio.setup.sync import sync_command, sync_env, uv_path

# GPU support's download, as each installer's screen says it: Linux's CUDA packages are larger.
GPU_GB = {"linux": "3.7", "win32": "2.7"}
HERE = "linux" if sys.platform == "linux" else "win32"
SWITCH = {
    "cu130": f"Installing GPU support (about {GPU_GB[HERE]} GB to download).",
    "cpu": "Installing CPU support.",
}
NOT_FINISHED = "Problem: the install did not finish. Check the internet connection, then run it again."


def _sync(app: Path, backend: str, extras: list[str], doing: str) -> bool:
    print(doing, flush=True)
    for argv in sync_command(uv_path(app), paths.SOURCE_ROOT, backend, extras):
        if runner.foreground(argv, env=sync_env(app)) != 0:
            return False
    return True


def _save(app: Path, **changes: Any) -> dict[str, Any]:
    state = paths.install_state(app)
    state.update(changes)
    write_json(app / "state.json", state)
    return state


def device(choice: Device) -> int:
    """Install the CPU or the GPU build, as the choice and the card allow; the GPU test decides Auto."""
    app = paths.app_root()
    state = paths.install_state(app)
    backend, extras = state.get("backend", "cpu"), list(state.get("extras", []))
    problem = None if choice == Device.CPU else gpu.card_problem(gpu.probe())
    target = "cpu" if choice == Device.CPU or (problem and choice == Device.AUTO) else "cu130"
    reason = gpu.CHOSEN[Device.CPU] if choice == Device.CPU else problem or gpu.PASSED
    if problem and target == "cu130":
        print(f"Warning: {problem} GPU support is installed because you chose the GPU.")
    if target != backend and not _sync(app, target, extras, SWITCH[target]):
        if target == "cpu" or not _sync(app, "cpu", extras, SWITCH["cpu"]):
            print(NOT_FINISHED)
            return 1
        target, reason = "cpu", "GPU support could not be installed. Using the CPU."
    _save(app, backend=target)
    if target == "cu130":
        smoke = gpu.check_gpu(app)
        if not smoke["passed"] and choice == Device.AUTO:
            reason = f"The GPU test failed ({smoke['code']}). Using the CPU."
            if not _sync(app, "cpu", extras, SWITCH["cpu"]):
                print(NOT_FINISHED)
                return 1
            target = "cpu"
            _save(app, backend=target)
        elif not smoke["passed"]:
            print(f"Warning: the GPU test failed ({smoke['code']}). Analyses will fail on the GPU.")
    print(f"Device: {'GPU' if target == 'cu130' else 'CPU'}. {reason}")
    return 0


def finish() -> int:
    """Add or remove openSMILE as its licence record says, then record this version as installed."""
    app = paths.app_root()
    state = paths.install_state(app)
    backend = state.get("backend", "cpu")
    extras = ["nc"] if consented(paths.data_root()) else []
    doing = "Adding openSMILE." if extras else "Removing openSMILE."
    if sorted(extras) != sorted(state.get("extras", [])) and not _sync(app, backend, extras, doing):
        print("Problem: openSMILE could not be added or removed. Run the installer again.")
        return 1
    _save(app, version=__version__, app_dir=f"app/{__version__}", backend=backend, extras=extras,
          installed_utc=utc_now())  # fmt: skip
    print(f"Fluentfolio {__version__} is installed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -I -m fluentfolio.setup")
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("device").add_argument("--choice", choices=[d.value for d in Device], default="auto")
    steps.add_parser("finish")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[union-attr]
    return device(Device(args.choice)) if args.step == "device" else finish()


if __name__ == "__main__":
    sys.exit(main())
