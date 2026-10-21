"""The web app's process in this process: the open file, a second start, a busy port, the server settings."""

from __future__ import annotations

import socket
import webbrowser
from pathlib import Path

import pytest
import uvicorn

from fluentfolio.core.files import read_json, write_json
from fluentfolio.web import server


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """What the browser was asked to open."""
    seen: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *args, **kwargs: seen.append(url) or True)
    return seen


def test_open_html_holds_the_link_and_the_browser_gets_only_the_file(
    tmp_path: Path, opened: list[str]
) -> None:
    page = server.open_page(tmp_path, 8765, "Ab_c-123")
    text = page.read_text(encoding="utf-8")
    assert '<meta http-equiv="refresh" content="0; url=http://127.0.0.1:8765/auth?t=Ab_c-123">' in text
    assert '<a href="http://127.0.0.1:8765/auth?t=Ab_c-123">Open Fluentfolio</a>' in text
    server.open_in_browser(page)
    assert opened == [page.as_uri()] and "Ab_c-123" not in opened[0]


def test_a_second_start_opens_the_running_app_and_ends(
    tmp_path: Path, opened: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    write_json(tmp_path / "server.json", {"port": 8765, "token": "tok", "pid": 1, "started": "x"})
    assert server.hand_over(tmp_path, browser=True) == 0
    assert "url=http://127.0.0.1:8765/auth?t=tok" in (tmp_path / "open.html").read_text(encoding="utf-8")
    assert opened == [(tmp_path / "open.html").as_uri()]
    assert "already running" in capsys.readouterr().out
    assert server.hand_over(tmp_path, browser=False) == 0
    assert str(tmp_path / "open.html") in capsys.readouterr().out and len(opened) == 1


@pytest.mark.parametrize("found", [None, "{", {"port": 8765}, {"port": "x", "token": "t"}])
def test_a_second_start_while_the_first_has_no_server_file_yet_says_so_and_exits_busy(
    tmp_path: Path, opened: list[str], capsys: pytest.CaptureFixture[str], found: object
) -> None:
    if isinstance(found, str):
        (tmp_path / "server.json").write_text(found, encoding="utf-8")
    elif found is not None:
        write_json(tmp_path / "server.json", found)
    assert server.hand_over(tmp_path, browser=True) == 6  # noqa: PLR2004 - the command line's busy code
    assert "starting in another window" in capsys.readouterr().out and opened == []


def test_a_port_in_use_is_refused_in_plain_words_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        port = other.getsockname()[1]
        assert server.run(tmp_path, port, browser=False, moved=[]) == 6  # noqa: PLR2004
    assert f"Port {port} is in use by another program" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_the_server_listens_on_127_0_0_1_only_and_cleans_up_after_it_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, object] = {}

    class Recorder:
        """Stands in for uvicorn's server: records its settings and the files present while it runs."""

        def __init__(self, config: uvicorn.Config) -> None:
            seen["config"] = config
            self.should_exit = False

        def run(self) -> None:
            seen["files"] = sorted(p.name for p in tmp_path.iterdir())
            seen["server"] = read_json(tmp_path / "server.json")

    monkeypatch.setattr(uvicorn, "Server", Recorder)
    assert server.run(tmp_path, 0, browser=False, moved=[]) == 0
    config = seen["config"]
    assert isinstance(config, uvicorn.Config)
    assert (config.host, config.proxy_headers, config.server_header, config.access_log) == (
        "127.0.0.1", False, False, False,
    )  # fmt: skip
    running = seen["server"]
    assert isinstance(running, dict) and running["port"] == config.port > 0
    assert len(running["token"]) == 43  # noqa: PLR2004 - 32 random bytes in URL-safe base64, no padding
    assert seen["files"] == ["open.html", "server.json"]
    assert list(tmp_path.iterdir()) == []  # the token went with the server
    out = capsys.readouterr().out
    assert str(tmp_path / "open.html") in out and running["token"] not in out
