"""`fluentfolio serve` started as a launcher starts it, for the tests that talk to the real server."""

from __future__ import annotations

import http.client
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from fluentfolio.core.files import read_json

START_S = 60.0
STOP_S = 30.0


def start(home: Path, app: Path | None = None, *, installed: bool = False) -> subprocess.Popen[bytes]:
    """The command in its own process group, so a test can send it Ctrl+Break as a closed window would.

    `app` is an app folder of the test's own; by default the one the environment names. `installed` leaves
    FLUENTFOLIO_HOME out, so that app folder's state.json names the data folder, as after an install.
    """
    windows = sys.platform == "win32"
    env = {k: v for k, v in os.environ.items() if k != "FLUENTFOLIO_HOME"}
    if not installed:
        env["FLUENTFOLIO_HOME"] = str(home)
    if app is not None:
        env["FLUENTFOLIO_APP_ROOT"] = str(app)
    # Its window's text goes to a file: a pipe that nobody reads fills up, and the server stops answering.
    with output(home).open("ab") as window:
        return subprocess.Popen(
            [sys.executable, "-m", "fluentfolio", "serve", "--no-browser"], env=env, stdin=subprocess.DEVNULL,
            stdout=window, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if windows else 0,
        )  # fmt: skip


def output(home: Path) -> Path:
    """What the server printed in its window, beside its data folder."""
    return home.parent / f"{home.name}-window.txt"


def running(home: Path, proc: subprocess.Popen[bytes]) -> dict[str, Any]:
    """server.json once the server answers on its port."""
    deadline = time.monotonic() + START_S
    while time.monotonic() < deadline:
        assert proc.poll() is None, output(home).read_text(encoding="utf-8", errors="replace")
        if (home / "server.json").is_file():
            info: dict[str, Any] = read_json(home / "server.json")
            try:
                http.client.HTTPConnection("127.0.0.1", info["port"], timeout=1).connect()
                return info
            except OSError:
                pass
        time.sleep(0.1)
    raise AssertionError("the server did not start")


def interrupt(proc: subprocess.Popen[bytes]) -> None:
    """Ctrl+Break to the whole group on Windows; on Linux the command alone gets Ctrl+C and passes it on."""
    proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)  # type: ignore[attr-defined]


def stop(proc: subprocess.Popen[bytes]) -> None:
    """End a server a test left running: Ctrl+C first, then kill this test's command and its server."""
    if proc.poll() is not None:
        return
    children = psutil.Process(proc.pid).children(recursive=True)
    interrupt(proc)
    try:
        proc.wait(STOP_S)
    except subprocess.TimeoutExpired:
        for child in children:
            child.kill()
        proc.kill()
        proc.wait(STOP_S)
