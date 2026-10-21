"""Keeps the computer from sleeping while a job runs: SetThreadExecutionState, or systemd-inhibit on Linux."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import TracebackType

from fluentfolio.jobs.runner import Background

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
INHIBIT = Path("/usr/bin/systemd-inhibit")
log = logging.getLogger(__name__)


def _execution_state(flags: int) -> int:
    import ctypes

    kernel = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    kernel.SetThreadExecutionState.restype = ctypes.c_uint32
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    return int(kernel.SetThreadExecutionState(flags))


class KeepAwake:
    """Enter and leave it on the job's own thread: Windows ties the request to the calling thread."""

    def __init__(self) -> None:
        self._inhibitor: Background | None = None

    def __enter__(self) -> KeepAwake:
        if sys.platform == "win32":
            _execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        elif INHIBIT.is_file():
            # cat holds the lock until its stdin closes, which also happens when this process dies.
            argv = [str(INHIBIT), "--what=sleep:idle", "--who=Fluentfolio", "--why=Analysing a recording",
                    "--mode=block", "/usr/bin/cat"]  # fmt: skip
            try:
                self._inhibitor = Background(argv, {"PATH": "/usr/bin:/bin"})
            except OSError as exc:
                log.info("sleep inhibitor not started: %s", exc.strerror)
        return self

    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None
    ) -> None:
        if sys.platform == "win32":
            _execution_state(ES_CONTINUOUS)
        elif self._inhibitor is not None:
            self._inhibitor.stop()
            self._inhibitor = None
