"""The one place that builds the uv runs which change the app's venv, so no extra is ever dropped by accident.

uv sync is exact: a run that forgets an extra removes it. Only the installer's steps call this, with the app
closed, because Windows cannot replace a library file that a running process has loaded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKENDS = ("cpu", "cu130")
EXTRAS = ("nc",)


def uv_path(app: Path) -> Path:
    return app / "bin" / ("uv.exe" if sys.platform == "win32" else "uv")


def sync_command(uv: Path, project: Path, backend: str, extras: list[str]) -> list[list[str]]:
    """The two uv runs that make the venv match the lock, the backend and the extras.

    The first installs every dependency from wheels only, so no package's build script runs on the
    learner's computer. The second installs Fluentfolio itself, which uv builds with its own backend.
    """
    if backend not in BACKENDS or any(e not in EXTRAS for e in extras):
        raise ValueError("unknown backend or extra")
    chosen = ["--extra", backend]
    for extra in sorted(set(extras)):
        chosen += ["--extra", extra]
    base = [str(uv), "sync", "--locked", "--no-dev", "--project", str(project), *chosen]
    return [[*base, "--no-build", "--no-install-project"], base]


def sync_env(app: Path) -> dict[str, str]:
    """uv's settings: Python and every cache inside the app folder, nothing registered or put on PATH."""
    keep = ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA")
    proxies = ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "SSL_CERT_FILE")
    names = (*keep, *proxies, *(p.lower() for p in proxies[:3]))  # the proxy settings in both spellings
    env = {name: os.environ[name] for name in names if name in os.environ}
    env.update(
        UV_CACHE_DIR=str(app / "cache" / "uv"),
        UV_PYTHON_INSTALL_DIR=str(app / "python"),
        UV_PYTHON_PREFERENCE="only-managed",
        UV_PYTHON_INSTALL_REGISTRY="0",
        UV_PYTHON_INSTALL_BIN="0",
    )  # no UV_NO_CONFIG: it drops the project's own uv settings, and the lock file was made with them
    return env
