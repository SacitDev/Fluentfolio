"""`fluentfolio serve` in real processes: loopback only, the launch file, a second start, Quit and Ctrl+C."""

from __future__ import annotations

import http.client
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from urllib.parse import urlencode

import psutil

from fluentfolio.core.files import read_json, write_json
from fluentfolio.jobs.locks import app_lock

Run = Callable[..., subprocess.CompletedProcess[bytes]]
FORM = "application/x-www-form-urlencoded"
JOB = "20261017-120000-ab12"


def ask(
    port: int, method: str, path: str, headers: dict[str, str] | None = None, body: str | None = None
) -> http.client.HTTPResponse:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    connection.request(method, path, body=body, headers=headers or {})
    reply = connection.getresponse()
    reply.read()
    return reply


def stopped(home: Path, proc: subprocess.Popen[bytes], server_pid: int) -> None:
    assert proc.wait(30) == 0, proc.communicate()
    assert not psutil.pid_exists(server_pid) or psutil.Process(server_pid).status() == psutil.STATUS_ZOMBIE
    assert not (home / "server.json").exists() and not (home / "open.html").exists()
    assert app_lock(home).is_free()


def test_serve_stays_on_loopback_signs_in_through_its_file_hands_over_and_stops_on_quit(
    tmp_path: Path, cli: Run, serving: ModuleType
) -> None:
    home = tmp_path / "home"
    proc = serving.start(home)
    try:
        info = serving.running(home, proc)
        port, token = info["port"], info["token"]
        sockets = psutil.Process(info["pid"]).net_connections("inet")
        listening = {c.laddr.ip for c in sockets if c.status == psutil.CONN_LISTEN}
        assert listening == {"127.0.0.1"}
        for pid in (proc.pid, info["pid"]):  # the token is on no command line
            assert token not in " ".join(psutil.Process(pid).cmdline())
        if sys.platform != "win32":
            assert {(home / n).stat().st_mode & 0o777 for n in ("server.json", "open.html")} == {0o600}
        assert f"url=http://127.0.0.1:{port}/auth?t={token}" in (home / "open.html").read_text(
            encoding="utf-8"
        )

        assert ask(port, "GET", "/static/css/app.css").status == 403
        signed = ask(port, "GET", f"/auth?t={token}")
        assert signed.status == 200 and signed.getheader("server") is None
        cookie = signed.getheader("set-cookie").split(";")[0]
        assert ask(port, "GET", "/static/css/app.css", {"Cookie": cookie}).status == 200

        again = cli("serve", "--no-browser", home=home)
        assert again.returncode == 0 and b"already running" in again.stdout, again.stderr
        assert token in (home / "open.html").read_text(encoding="utf-8")

        own = {"Cookie": cookie, "Sec-Fetch-Site": "same-origin", "HX-Request": "true"}
        assert ask(port, "POST", "/quit", {**own, "Sec-Fetch-Site": "same-site"}).status == 403
        assert ask(port, "POST", "/quit", own).status == 200
        stopped(home, proc, info["pid"])
    finally:
        serving.stop(proc)


def test_a_data_folder_move_stops_the_server_and_only_then_deletes_the_old_folder(
    tmp_path: Path, serving: ModuleType
) -> None:
    old, new, app = tmp_path / "old", tmp_path / "new", tmp_path / "app"
    write_json(app / "state.json", {"selftest": {"passed": True}, "data_root": str(old)})
    write_json(old / "jobs" / JOB / "job.json", {})
    proc = serving.start(old, app, installed=True)
    try:
        info = serving.running(old, proc)
        port = info["port"]
        cookie = ask(port, "GET", f"/auth?t={info['token']}").getheader("set-cookie").split(";")[0]
        own = {"Cookie": cookie, "Sec-Fetch-Site": "same-origin", "HX-Request": "true", "Content-Type": FORM}
        reply = ask(port, "POST", "/settings/move", own, urlencode({"kind": "data", "target": str(new)}))
        assert reply.status == 200 and reply.getheader("hx-retarget") == "#main"
        assert proc.wait(30) == 0, serving.output(old).read_text(encoding="utf-8", errors="replace")
        assert read_json(app / "state.json")["data_root"] == str(new) and not old.exists()
        assert sorted(p.name for p in new.iterdir()) == ["jobs"]  # no lock, link or token file came along
        assert (new / "jobs" / JOB / "job.json").is_file()
    finally:
        serving.stop(proc)


def test_ctrl_c_stops_the_command_and_its_server_and_removes_the_link(
    tmp_path: Path, serving: ModuleType
) -> None:
    home = tmp_path / "home"
    proc = serving.start(home)
    try:
        info = serving.running(home, proc)
        serving.interrupt(proc)
        stopped(home, proc, info["pid"])
    finally:
        serving.stop(proc)
