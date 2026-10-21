"""app.lock and work.lock: OS byte locks that the system releases when the holding process dies."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import TracebackType

from fluentfolio.core.errors import ResourceError

# Windows frees a dead holder's lock a moment after the process ends, so a lock is tried for about 0.5 s.
LOCK_TRIES = 10
LOCK_WAIT_S = 0.05


class FileLock:
    """A non-blocking lock on byte 0 of a file; a second holder gets E_BUSY within half a second."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def _try(self, fd: int) -> None:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def acquire(self, tries: int = LOCK_TRIES) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        for attempt in range(tries):
            try:
                self._try(fd)
                break
            except OSError as exc:
                if attempt == tries - 1:
                    os.close(fd)
                    raise ResourceError("E_BUSY", self.path.name) from exc
                time.sleep(LOCK_WAIT_S)
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        if sys.platform == "win32":
            import msvcrt

            os.lseek(self._fd, 0, os.SEEK_SET)
            msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
        os.close(self._fd)  # on Linux, closing releases the flock
        self._fd = None

    def is_free(self) -> bool:
        """Whether no process holds the lock now (one try, no wait); this process's own hold counts too.

        A holder creates the file, so a missing file is free, and the check itself creates nothing.
        """
        if self._fd is not None:
            return False
        if not self.path.exists():
            return True
        try:
            self.acquire(tries=1)
        except ResourceError:
            return False
        self.release()
        return True

    def __enter__(self) -> FileLock:
        return self.acquire()

    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.release()


def work_lock(data: Path) -> FileLock:
    """Held by whoever runs a job, a component install or removal, a job delete or a folder move."""
    return FileLock(data / "work.lock")


def app_lock(data: Path) -> FileLock:
    """Held by the web server only."""
    return FileLock(data / "app.lock")
