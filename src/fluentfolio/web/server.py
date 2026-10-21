"""The web app's process: one per data folder, on 127.0.0.1 only, opened through a file that holds its link.

A second start finds app.lock taken, opens the running app instead and ends. The link with the
launch token goes only into files that this user alone can read, never onto a command line, where other
users of a Linux computer could see it.
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import secrets
import shutil
import signal
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from fluentfolio.core.errors import ResourceError
from fluentfolio.core.files import atomic_write, read_json, write_json
from fluentfolio.core.paths import app_root, data_root, hub_env, models_dir, secure_umask
from fluentfolio.jobs.locks import app_lock
from fluentfolio.jobs.store import utc_now
from fluentfolio.web import upload

HOST = "127.0.0.1"
READY_WAIT_S = 30.0
POLL_S = 0.1
STOP_WAIT_S = 3  # how long a stop waits for open progress streams before it closes them
BUSY = 6  # the command line's exit code for "another instance holds the lock"
TITLE = "Fluentfolio: keep this window open"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return int(probe.getsockname()[1])


def port_taken(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((HOST, port))
        except OSError:
            return True
    return False


def open_page(data: Path, port: int, token: str) -> Path:
    """Write open.html, which sends the browser to the sign-in link; the umask keeps it user-only on Linux."""
    link = html.escape(f"http://{HOST}:{port}/auth?t={token}")
    page = data / "open.html"
    atomic_write(
        page,
        '<!doctype html>\n<html lang="en"><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url={link}"><title>Opening Fluentfolio</title>'
        f'<p><a href="{link}">Open Fluentfolio</a></p>\n',
    )
    return page


def open_in_browser(page: Path) -> None:
    """Open the file, never the link inside it: a browser's command line is visible to other users."""
    webbrowser.open(page.as_uri())


def open_when_ready(port: int, page: Path) -> None:
    deadline = time.monotonic() + READY_WAIT_S
    while time.monotonic() < deadline:
        try:
            socket.create_connection((HOST, port), timeout=POLL_S).close()
        except OSError:
            time.sleep(POLL_S)
            continue
        open_in_browser(page)
        return


def hand_over(data: Path, browser: bool) -> int:
    """Another process holds app.lock: open the app it serves and end."""
    try:
        running = read_json(data / "server.json")
        page = open_page(data, int(running["port"]), str(running["token"]))
    except (OSError, ValueError, KeyError, TypeError):
        print("Fluentfolio is starting in another window. Wait a moment, then start it again.")
        return BUSY
    if browser:
        open_in_browser(page)
        print("Fluentfolio is already running. It is open in your browser now.")
    else:
        print(f"Fluentfolio is already running. Open this file in your browser: {page}")
    return 0


def run(data: Path, port: int, browser: bool, moved: list[Path]) -> int:
    """Serve until Ctrl+C, the window closes, Quit or a data folder move; data's app.lock is already held.

    A move adds the old data folder to `moved`, for the caller to delete once it has let the lock go.
    """
    # Imported only now: the app loads huggingface_hub, which reads its cache folders once, at import.
    from fluentfolio.web.app import build
    from fluentfolio.web.tasks import Tasks

    port = port or free_port()
    if port_taken(port):
        print(
            f"Port {port} is in use by another program. Start Fluentfolio without --port to use a free one."
        )
        return BUSY
    upload.clear(data)  # uploads that a stopped app left behind; app.lock says none is arriving now
    token, session = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    server: uvicorn.Server | None = None
    tasks = Tasks(data)

    def stop(old_data: Path | None = None) -> None:
        if old_data is not None:
            moved.append(old_data)
        if server is not None:
            server.should_exit = True

    # What uvicorn.run(host="127.0.0.1", proxy_headers=False, server_header=False) builds, kept by hand so
    # that Quit can stop it. No access log: the sign-in link holds the token.
    config = uvicorn.Config(
        build(f"http://{HOST}:{port}", token, session, stop, tasks), host=HOST, port=port,
        proxy_headers=False, server_header=False, access_log=False, log_level="warning",
        timeout_graceful_shutdown=STOP_WAIT_S,
    )  # fmt: skip
    server = uvicorn.Server(config)
    try:
        write_json(
            data / "server.json", {"port": port, "token": token, "pid": os.getpid(), "started": utc_now()}
        )
        page = open_page(data, port, token)
        if browser:
            threading.Thread(target=open_when_ready, args=(port, page), daemon=True).start()
        else:
            print(f"Open this file in your browser: {page}")
        print("Fluentfolio is running. Keep this window open. Close it, press Ctrl+C or use Quit to stop.")
        server.run()
    except KeyboardInterrupt:
        pass  # uvicorn raises the stop signal again once it has stopped; stopping is what it asked for
    finally:
        # Known limit: closing the window ends the process before this runs, so both files stay until the next
        # start writes new ones (their token is dead by then); revisit if a stale file ever stops a start.
        for name in ("server.json", "open.html"):  # the token dies with this process
            (data / name).unlink(missing_ok=True)
        if tasks.running() is not None:
            print("Stopping what was running. To finish an analysis, run it again from History.")
        tasks.stop()
    print("Fluentfolio has stopped.")
    return 0


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def name_window() -> None:
    """Title the window with the one thing a learner must know about it: closing it stops Fluentfolio."""
    if not sys.stdout.isatty():
        return
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetConsoleTitleW(TITLE)
    else:
        sys.stdout.write(f"\033]0;{TITLE}\a")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fluentfolio serve")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", dest="browser", action="store_false")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        # Line by line, so a window or a program reading through a pipe sees each line when it is written.
        stream.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[union-attr]
    name_window()
    # uvicorn raises its stop signal again after stopping; by default SIGTERM and Ctrl+Break would then end
    # the process before it removes server.json, so they stop it the way Ctrl+C does.
    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _interrupt)
    secure_umask()
    # huggingface_hub reads its cache folders once, at import, so they are set before anything imports it.
    os.environ.update(hub_env(app_root(), models_dir()))
    data = data_root()
    lock = app_lock(data)
    try:
        lock.acquire()
    except ResourceError:
        return hand_over(data, args.browser)
    moved: list[Path] = []
    try:
        return run(data, args.port, args.browser, moved)
    finally:
        lock.release()
        if moved:  # the old data folder, once this process holds nothing in it
            logging.shutdown()
        for old in moved:
            shutil.rmtree(old, ignore_errors=True)
