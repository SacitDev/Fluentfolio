"""Run an installer script's shell in a child (PowerShell on Windows, bash on Linux), with every path passed
through the environment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the checkout, which holds the installer scripts
TIMEOUT_S = 120


def powershell(home: Path, command: str, **env: str) -> subprocess.CompletedProcess[str]:
    """One PowerShell run; its profile folders point into `home`, so nothing reaches the real profile."""
    assert sys.platform == "win32", "the Windows installer runs on Windows only"
    exe = Path(os.environ["SYSTEMROOT"]) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    own = {"APPDATA": str(home / "roaming"), "LOCALAPPDATA": str(home / "local"), "USERPROFILE": str(home)}
    argv = [str(exe), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command]
    # Without PSModulePath, Windows PowerShell builds its own; the one a PowerShell 7 shell passes on would
    # make it load 7's modules, which fail there.
    inherited = {k: v for k, v in os.environ.items() if k.upper() != "PSMODULEPATH"}
    return subprocess.run(
        argv, capture_output=True, text=True, errors="replace", timeout=TIMEOUT_S, check=False,
        env={**inherited, **own, "FF_SCRIPT": str(ROOT / "install.ps1"), **env},
    )  # fmt: skip


def bash(home: Path, command: str, **env: str) -> subprocess.CompletedProcess[str]:
    """One bash run with HOME and the XDG data folder in `home`, so nothing reaches the real profile."""
    assert sys.platform == "linux", "the Linux installer runs on Linux only"
    own = {"HOME": str(home), "XDG_DATA_HOME": str(home / "share"), "FF_SCRIPT": str(ROOT / "install.sh")}
    return subprocess.run(
        ["/bin/bash", "-c", command], capture_output=True, text=True, errors="replace", timeout=TIMEOUT_S,
        check=False, stdin=subprocess.DEVNULL, env={**os.environ, **own, **env},
    )  # fmt: skip
