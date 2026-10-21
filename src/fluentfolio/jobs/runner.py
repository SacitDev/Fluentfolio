"""The only product module that starts processes: a Job Object or process group per child, with a watchdog."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import IO, Any, cast

from fluentfolio.core.enums import DEVICE_VAR, Device
from fluentfolio.core.paths import job_tmp

IS_WINDOWS = sys.platform == "win32"
STDERR_KEPT = 200
POLL_S = 0.1
KILL_WAIT_S = 10.0
STOP_ON_EOF = "FLUENTFOLIO_STOP_ON_EOF"  # the command line stops, as at Ctrl+C, when its stdin closes
SWITCHES = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_UPDATE_CHECK": "1",
    "PYANNOTE_METRICS_ENABLED": "0",
    "ORT_DISABLE_TELEMETRY": "1",
}
WINDOWS_KEPT = ("SYSTEMROOT", "WINDIR", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "USERPROFILE")
KEPT = (*WINDOWS_KEPT, "LOCALAPPDATA", "APPDATA") if IS_WINDOWS else ("HOME", "LANG", "LC_ALL")
# Python's own path and start-up switches, and Hugging Face's debug output, which prints its requests.
NOT_FOR_OWN = ("PYTHON", "HF_DEBUG")


def own_env() -> dict[str, str]:
    """The environment of the app's own processes: the job runner, the web server and the web app's commands.

    They are this app, so they keep its settings (a proxy, the chosen card, the folders); each stage child
    they start gets child_env() instead.
    """
    return {name: value for name, value in os.environ.items() if not name.upper().startswith(NOT_FOR_OWN)}


def child_env(
    app: Path, data: Path, models: Path, job_id: str | None, *, cpu: bool, extra_path: Sequence[Path] = ()
) -> dict[str, str]:
    """A child's whole environment, built from an allow-list: no token, PYTHONPATH or HF_HOME ever passes."""
    env = {name: os.environ[name] for name in KEPT if name in os.environ}
    if IS_WINDOWS:
        system = [Path(os.environ["SYSTEMROOT"]) / "System32"]
    else:
        system = [Path("/usr/bin"), Path("/bin")]
    env["PATH"] = os.pathsep.join(str(p) for p in (*extra_path, Path(sys.executable).parent, *system))
    tmp = job_tmp(app, job_id) if job_id else app / "tmp" / "misc"
    tmp.mkdir(parents=True, exist_ok=True)
    env.update(SWITCHES)
    env.update(
        HF_HUB_CACHE=str(models / "hf"), TORCH_HOME=str(app / "cache" / "torch"),
        TMP=str(tmp), TEMP=str(tmp), TMPDIR=str(tmp),
        FLUENTFOLIO_HOME=str(data), FLUENTFOLIO_APP_ROOT=str(app),
    )  # fmt: skip
    env[DEVICE_VAR] = Device.CPU if cpu else Device.GPU
    if cpu:
        # On Windows an empty value leaves the card visible to CUDA in most libraries; -1 hides it everywhere.
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    elif "CUDA_VISIBLE_DEVICES" in os.environ:  # the user's own choice of card, or of none
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    return env


@dataclass
class Outcome:
    exit_code: int
    stopped: str | None = None  # "cancel", "watchdog" or "backstop" when the runner ended the child
    wall_ms: int = 0
    stderr_tail: list[str] = field(default_factory=list)


def _checked(argv: Sequence[str]) -> list[str]:
    exe = Path(argv[0])
    if not exe.is_absolute() or exe.suffix.lower() in (".bat", ".cmd"):
        raise ValueError("a child needs an absolute executable path, never a batch file")
    return list(argv)


class _Group:
    """The child and everything it starts: a kill-on-close Job Object on Windows, a process group on Linux."""

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        self.proc = proc
        self.job = _win_job(proc) if IS_WINDOWS else None

    def kill(self) -> None:
        """End every process of the group and wait until none is left, so its files can be removed."""
        if self.job is not None:
            _kernel().TerminateJobObject(self.job, 1)
            deadline = time.monotonic() + KILL_WAIT_S
            while _active_processes(self.job) and time.monotonic() < deadline:
                time.sleep(POLL_S)
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)  # type: ignore[attr-defined]
            except ProcessLookupError:
                pass
        self.proc.wait(KILL_WAIT_S)

    def close(self) -> None:
        """End anything the child left behind, then free the handle."""
        if self.job is not None:
            _kernel().CloseHandle(self.job)  # kill on close ends any process still in the job
            self.job = None
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)  # type: ignore[attr-defined]
            except (ProcessLookupError, PermissionError):
                pass


def _read(stream: IO[bytes], sink: Callable[[str], None]) -> None:
    for raw in iter(stream.readline, b""):
        sink(raw.decode("utf-8", "replace").rstrip("\r\n"))


def run(
    argv: Sequence[str],
    *,
    env: dict[str, str],
    timeout_s: float,
    no_progress_s: float,
    cwd: Path | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Outcome:
    """Run one child to its end. stdout carries JSON progress lines; the last stderr lines are returned."""
    proc = subprocess.Popen(
        _checked(argv), env=env, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
        start_new_session=not IS_WINDOWS,
    )  # fmt: skip
    group = _Group(proc)
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    tail: deque[str] = deque(maxlen=STDERR_KEPT)
    last = [time.monotonic()]

    def progress(line: str) -> None:
        try:
            event = json.loads(line)
        except ValueError:
            tail.append(line)
            return
        last[0] = time.monotonic()
        if isinstance(event, dict):
            events.put(event)

    def deliver() -> None:
        while not events.empty():
            event = events.get()
            if on_progress:
                on_progress(event)

    readers = [
        threading.Thread(target=_read, args=(cast(IO[bytes], proc.stdout), progress), daemon=True),
        threading.Thread(target=_read, args=(cast(IO[bytes], proc.stderr), tail.append), daemon=True),
    ]
    for reader in readers:
        reader.start()
    start, stopped = time.monotonic(), None
    try:
        while proc.poll() is None:
            deliver()
            now = time.monotonic()
            if should_stop and should_stop():
                stopped = "cancel"
            elif now - last[0] > no_progress_s:
                stopped = "watchdog"
            elif now - start > timeout_s:
                stopped = "backstop"
            if stopped:
                group.kill()
                break
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(POLL_S)  # returns as soon as the child ends
    finally:
        if proc.poll() is None:  # an exception or Ctrl+C in this process: the child must not outlive it
            group.kill()
        group.close()
        for reader in readers:
            reader.join(KILL_WAIT_S)
    deliver()
    return Outcome(proc.returncode, stopped, int((time.monotonic() - start) * 1000), list(tail))


def detach(argv: Sequence[str], env: dict[str, str]) -> subprocess.Popen[bytes]:
    """Start a process that outlives this one: no window, no shared console, no stdio."""
    quiet: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                             "stderr": subprocess.DEVNULL}  # fmt: skip
    if not IS_WINDOWS:
        return subprocess.Popen(_checked(argv), env=env, start_new_session=True, **quiet)
    # CREATE_NO_WINDOW, not DETACHED_PROCESS: the steps' own children then share a hidden console.
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        # Out of the caller's Job Object, so the run goes on when the program that started it ends.
        breakaway = flags | subprocess.CREATE_BREAKAWAY_FROM_JOB
        return subprocess.Popen(_checked(argv), env=env, creationflags=breakaway, **quiet)
    except OSError:  # the caller's Job Object does not allow breaking away
        return subprocess.Popen(_checked(argv), env=env, creationflags=flags, **quiet)


def foreground(argv: Sequence[str], env: dict[str, str]) -> int:
    """Run a child in this window until it ends, and return its exit code.

    Ctrl+C and closing the window reach the child too; this process then gives it time to stop.
    """
    proc = subprocess.Popen(_checked(argv), env=env, stdin=subprocess.DEVNULL)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        if not IS_WINDOWS:
            proc.send_signal(signal.SIGTERM)  # a SIGTERM sent to this process alone reaches the child too
        try:
            return proc.wait(KILL_WAIT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()


class Background:
    """A helper process that lives until stop(), or until this process ends and its stdin closes."""

    def __init__(self, argv: Sequence[str], env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            _checked(argv), env=env, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=not IS_WINDOWS,
        )  # fmt: skip

    def stop(self) -> None:
        cast(IO[bytes], self.proc.stdin).close()
        try:
            self.proc.wait(KILL_WAIT_S)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(KILL_WAIT_S)


def attached(
    argv: Sequence[str], env: dict[str, str], out: IO[bytes], err: IO[bytes]
) -> subprocess.Popen[bytes]:
    """Start a command of the command line that stops, as at Ctrl+C, once this process closes its stdin.

    The pipe also closes when this process ends in any way. Only the pipe stops it: it has no window and a
    process group of its own, so a Ctrl+C meant for this process never reaches it as a second stop.
    """
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
    return subprocess.Popen(
        _checked(argv), env={**env, STOP_ON_EOF: "1"}, stdin=subprocess.PIPE, stdout=out, stderr=err,
        creationflags=flags, start_new_session=not IS_WINDOWS,
    )  # fmt: skip


def stop_attached(proc: subprocess.Popen[bytes], wait_s: float) -> int:
    """Close the command's stdin, give it `wait_s` to stop the way Ctrl+C stops it, then end it."""
    with contextlib.suppress(OSError):
        cast(IO[bytes], proc.stdin).close()
    try:
        return proc.wait(wait_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait(KILL_WAIT_S)


@cache
def _kernel() -> Any:
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    handle, dword, pointer = wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p
    kernel.CreateJobObjectW.restype = handle
    kernel.CreateJobObjectW.argtypes = [pointer, wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [handle, ctypes.c_int, pointer, dword]
    kernel.AssignProcessToJobObject.argtypes = [handle, handle]
    kernel.TerminateJobObject.argtypes = [handle, ctypes.c_uint]
    kernel.QueryInformationJobObject.argtypes = [handle, ctypes.c_int, pointer, dword, pointer]
    kernel.CloseHandle.argtypes = [handle]
    return kernel


def _win_job(proc: subprocess.Popen[bytes]) -> int:
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [
            ("user_time", ctypes.c_int64), ("job_time", ctypes.c_int64), ("flags", wintypes.DWORD),
            ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t), ("procs", wintypes.DWORD),
            ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD),
        ]  # fmt: skip

    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", ctypes.c_uint64 * 6), ("limits", ctypes.c_size_t * 4)]

    kernel = _kernel()
    job = kernel.CreateJobObjectW(None, None)
    info = Extended()
    info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    if not ok or not kernel.AssignProcessToJobObject(job, int(proc._handle)):  # type: ignore[attr-defined]
        error = ctypes.get_last_error()  # type: ignore[attr-defined]
        kernel.CloseHandle(job)
        proc.kill()
        raise OSError(error, "could not put the child in a Job Object")
    return int(job)


def _active_processes(job: int) -> int:
    import ctypes
    from ctypes import wintypes

    class Accounting(ctypes.Structure):
        _fields_ = [("times", ctypes.c_int64 * 4), ("faults", wintypes.DWORD), ("total", wintypes.DWORD),
                    ("active", wintypes.DWORD), ("ended", wintypes.DWORD)]  # fmt: skip

    info = Accounting()
    _kernel().QueryInformationJobObject(job, 1, ctypes.byref(info), ctypes.sizeof(info), None)
    return int(info.active)
