"""The long actions the web app starts: each is a command of the command line, run as a child of the server.

A child stops when the server stops, by Quit, Ctrl+C, a closed window or a crash: its stdin pipe closes,
and it stops the way Ctrl+C stops it. A job then becomes interrupted and runs again from its first
unfinished step.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fluentfolio.core.errors import ResourceError
from fluentfolio.jobs import runner

if TYPE_CHECKING:
    import subprocess

STOP_WAIT_S = 30.0  # a stopping job kills its step and writes its state; a child still there is ended
SHARE = re.compile(r"(\d{1,3})% of ")  # the start of `components install`'s download line


@dataclass
class Task:
    kind: str
    proc: subprocess.Popen[bytes]
    out: Path  # the command's JSON reply, when it has one
    err: Path  # its progress lines
    detail: str  # what it works on: the parts to install, or the job for an audio file

    def running(self) -> bool:
        return self.proc.poll() is None

    def reply(self) -> dict[str, Any] | None:
        """The command's JSON reply once it has ended; None while it runs or if it wrote none."""
        if self.running():
            return None
        try:
            found = json.loads(self.out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return found if isinstance(found, dict) else None

    def lines(self) -> list[str]:
        try:
            return self.err.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []

    def last_line(self) -> str:
        return next((line.strip() for line in reversed(self.lines()) if line.strip()), "")

    def share(self) -> int | None:
        """The share of a download done, from the newest line that starts with one, like "34% of 1.2 GB"."""
        found = next((m for line in reversed(self.lines()) if (m := SHARE.match(line))), None)
        return min(int(found[1]), 100) if found else None


class Tasks:
    """The children of one server; at most one runs at a time, as work.lock allows only one anyway."""

    def __init__(self, data: Path) -> None:
        self.folder = data / "tmp"
        self.started: list[Task] = []
        self.lock = threading.Lock()

    def start(self, kind: str, argv: list[str], env: dict[str, str], detail: str = "") -> Task:
        self.folder.mkdir(parents=True, exist_ok=True)
        out, err = self.folder / f"task-{kind}.json", self.folder / f"task-{kind}.txt"
        with self.lock:
            if self.running() is not None:
                raise ResourceError("E_BUSY")
            with out.open("wb") as reply, err.open("wb") as lines:
                proc = runner.attached(argv, env, reply, lines)
            task = Task(kind, proc, out, err, detail)
            self.started.append(task)
        return task

    def start_job(self, argv: list[str], env: dict[str, str]) -> Task:
        """The job runner of an analysis or a run again: the facade's `start`."""
        return self.start("job", argv, env)

    def command(self, kind: str, *args: str, detail: str = "") -> Task:
        """A command of the command line with a JSON reply, such as `self-test --json`."""
        argv = [sys.executable, "-I", "-X", "utf8", "-m", "fluentfolio", *args, "--json"]
        return self.start(kind, argv, runner.own_env(), detail)

    def running(self) -> Task | None:
        return next((t for t in self.started if t.running()), None)

    def last(self, kind: str) -> Task | None:
        return next((t for t in reversed(self.started) if t.kind == kind), None)

    def stop(self) -> None:
        """Stop every child that still runs; called when the server stops."""
        for task in self.started:
            if task.running():
                runner.stop_attached(task.proc, STOP_WAIT_S)
