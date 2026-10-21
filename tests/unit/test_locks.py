"""app.lock and work.lock: one holder at a time, released by the system when the holder dies."""

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from fluentfolio.core.errors import AppError
from fluentfolio.jobs.locks import app_lock, work_lock

HOLD = (
    "import sys; from pathlib import Path; from fluentfolio.jobs.locks import {0}; "
    "{0}(Path(sys.argv[1])).acquire(); print('held', flush=True); sys.stdin.read()"
)


def hold_in_another_process(kind: str, data: Path) -> subprocess.Popen:
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD.format(kind), str(data)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
    return holder


def test_a_second_holder_is_told_busy_until_the_first_process_ends(tmp_path: Path) -> None:
    holder = hold_in_another_process("work_lock", tmp_path)
    try:
        with pytest.raises(AppError) as caught:
            work_lock(tmp_path).acquire()
        assert (caught.value.code, caught.value.entry.exit_code) == ("E_BUSY", 6)
        assert not work_lock(tmp_path).is_free()
    finally:
        holder.kill()  # no clean release: the system must free the lock of a dead process
        holder.wait(10)
    with work_lock(tmp_path):
        pass
    assert work_lock(tmp_path).is_free()


def test_checking_a_lock_that_was_never_taken_creates_no_file(tmp_path: Path) -> None:
    assert work_lock(tmp_path).is_free()
    assert list(tmp_path.iterdir()) == []  # reading a page only checks the lock (no change on a read)


def test_a_lock_freed_a_moment_late_is_still_taken(tmp_path: Path) -> None:
    # Windows frees a dead holder's lock a little after the process ends (up to 16 ms seen under load).
    holder = hold_in_another_process("work_lock", tmp_path)
    timer = threading.Timer(0.2, holder.kill)
    start = time.monotonic()
    timer.start()
    try:
        with work_lock(tmp_path):
            assert time.monotonic() - start >= 0.2
    finally:
        timer.join()
        holder.wait(10)


def test_an_idle_web_server_does_not_block_a_job(tmp_path: Path) -> None:
    holder = hold_in_another_process("app_lock", tmp_path)
    try:
        assert not app_lock(tmp_path).is_free()
        with work_lock(tmp_path):
            pass
    finally:
        holder.kill()
        holder.wait(10)


def test_a_second_lock_in_the_same_process_is_refused_too(tmp_path: Path) -> None:
    with work_lock(tmp_path):
        with pytest.raises(AppError):
            work_lock(tmp_path).acquire()
