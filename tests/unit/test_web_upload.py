"""Uploads through the real app: the raw body goes to the data folder, counted against the limit.

The limit is set to 1 MB (1,000,000 bytes) in settings.toml, so each case runs at the limit and one
byte on either side of it.
"""

from __future__ import annotations

import asyncio
import math
import os
import wave
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

import pytest
from starlette.requests import ClientDisconnect, Request
from starlette.testclient import TestClient

from fluentfolio.settings.model import Limits
from fluentfolio.web import app as web_app
from fluentfolio.web import server, upload
from fluentfolio.web.security import SESSION_COOKIE
from fluentfolio.web.tasks import Tasks

ORIGIN = "http://127.0.0.1:8765"
SESSION = "session-for-tests"
LIMIT = 1_000_000  # 1 MB, written out rather than taken from the code under test
OWN_PAGE = {"sec-fetch-site": "same-origin", "hx-request": "true", "content-type": "application/octet-stream"}
WAV_HEADER = 44  # bytes before the samples in a plain PCM WAV file


@pytest.fixture
def data() -> Path:
    folder = Path(os.environ["FLUENTFOLIO_HOME"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.toml").write_text("[limits]\nmax_upload_mb = 1\n", encoding="utf-8")
    return folder


@pytest.fixture
def client(data: Path) -> TestClient:
    app = web_app.build(ORIGIN, "token", SESSION, lambda: None, Tasks(data))
    signed = TestClient(app, base_url=ORIGIN, raise_server_exceptions=False)
    signed.cookies.set(SESSION_COOKIE, SESSION)
    return signed


def wav_of_size(path: Path, total: int) -> Path:
    """An 8-bit mono 16 kHz tone whose file is exactly `total` bytes: one byte per sample."""
    frames = total - WAV_HEADER
    samples = bytes(128 + int(60 * math.sin(2 * math.pi * 220 * i / 16000)) for i in range(frames))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(16000)
        w.writeframes(samples)
    assert path.stat().st_size == total
    return path


def left_in(data: Path) -> list[str]:
    tmp = data / "tmp"
    return sorted(p.name for p in tmp.iterdir()) if tmp.is_dir() else []


class Tags(HTMLParser):
    """Every start tag with its attributes, and the text, of a page fragment."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def parsed(fragment: str) -> Tags:
    tags = Tags()
    tags.feed(fragment)
    return tags


@pytest.mark.parametrize("size", [LIMIT - 1, LIMIT])
def test_a_recording_up_to_the_limit_is_saved_under_an_id_and_described(
    client: TestClient, data: Path, tmp_path: Path, size: int
) -> None:
    source = wav_of_size(tmp_path / "practice.wav", size)
    headers = {**OWN_PAGE, "x-file-name": quote("Practice test 3.wav")}
    reply = client.post("/upload", content=source.read_bytes(), headers=headers)
    assert reply.status_code == 200, reply.text
    tags = parsed(reply.text)
    hidden = {a["name"]: a["value"] for t, a in tags.tags if t == "input"}
    upload_id = str(hidden["upload"])
    assert left_in(data) == [f"{upload_id}.upload"]  # stored under the ID, never under its own name
    assert (data / "tmp" / f"{upload_id}.upload").read_bytes() == source.read_bytes()
    assert hidden["upload_name"] == "Practice test 3.wav"
    text = " ".join(" ".join(tags.text).split())
    # 999,956 or 999,955 one-byte samples at 16,000 per second: 62.5 s.
    for fact in ("Length 1 min 2 s", "Format WAV", "Sample rate 16,000 Hz", "Channels 1"):
        assert fact in text, fact


@pytest.mark.parametrize(("free_mb", "shown"), [(1999, True), (2000, False)])
def test_the_upload_warns_before_start_when_a_step_needs_more_memory_than_is_free(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free_mb: int, shown: bool
) -> None:
    # The steps' minimums are planted: the phone step needs 2000 MB in every mode.
    monkeypatch.setattr(web_app.facade.estimate, "ram_needs_mb", lambda *a, **k: {"vad": 401, "phones": 2000})
    monkeypatch.setattr(web_app.facade, "_free_ram_mb", lambda: free_mb)
    source = wav_of_size(tmp_path / "practice.wav", 64_044)
    reply = client.post("/upload", content=source.read_bytes(), headers=OWN_PAGE)
    assert reply.status_code == 200, reply.text
    text = " ".join(" ".join(parsed(reply.text).text).split())
    note = (
        'The step "Listening to the sounds" may need about 2.0 GB of memory for a recording this long, and '
        "1.9 GB is free now. Close other programs before you start, or the analysis may stop."
    )
    assert (note in text) == shown
    assert ("of memory" in text) == shown


def chunks(total: int) -> Iterator[bytes]:
    """A body sent in pieces, with no Content-Length."""
    while total > 0:
        piece = min(total, 65536)
        total -= piece
        yield b"\0" * piece


@pytest.mark.parametrize(
    "send",
    [
        pytest.param(
            lambda c: c.post("/upload", content=b"\0" * (LIMIT + 1), headers=OWN_PAGE), id="declared"
        ),
        pytest.param(
            lambda c: c.post("/upload", content=chunks(LIMIT + 1), headers=OWN_PAGE), id="undeclared"
        ),
        pytest.param(
            lambda c: c.post(
                "/upload", content=chunks(LIMIT + 1), headers={**OWN_PAGE, "content-length": "10"}
            ),
            id="understated",
        ),
    ],
)
def test_one_byte_over_the_limit_gets_413_and_leaves_no_file(
    client: TestClient, data: Path, send: object
) -> None:
    reply = send(client)  # type: ignore[operator]
    assert reply.status_code == 413
    assert "The file is too large" in reply.text and "Error code E_INPUT_TOO_LARGE" in reply.text
    assert "<header" not in reply.text  # the card alone, for the page to put in place
    assert left_in(data) == []


# Every other refusal answers 200 with the card, as any answer put in place does: an error status would
# be written into the browser's console for a normal choice, such as picking a file that is not audio.


@pytest.mark.parametrize(
    ("size", "refused"), [(5 * 2**20, False), (5 * 2**20 + 1, True)]
)  # 5 MiB, written out
def test_a_pasted_text_over_the_form_limit_gets_the_text_card(
    client: TestClient, size: int, refused: bool
) -> None:
    field = b"transcript="
    form = {**OWN_PAGE, "content-type": "application/x-www-form-urlencoded"}
    reply = client.post("/jobs", content=field + b"a" * (size - len(field)), headers=form)
    assert reply.status_code == 200 and 'class="error-card' in reply.text
    # At the limit the form is read, and the next check finds no recording chosen.
    assert ("Error code E_INPUT_TEXT" in reply.text) is refused
    assert ("Error code E_NO_RECORDING" in reply.text) is not refused
    # The pasted text is the problem, so the card sends the learner back to the form.
    assert ('href="#form">Go back to the form</a>' in reply.text) is refused
    assert "Choose another recording" not in reply.text


def test_a_file_that_is_not_a_recording_is_refused_and_deleted(client: TestClient, data: Path) -> None:
    reply = client.post("/upload", content=b"this is a text file, not audio\n" * 40, headers=OWN_PAGE)
    assert reply.status_code == 200 and "Error code E_INPUT_UNSUPPORTED" in reply.text
    assert "<header" not in reply.text and 'class="error-card' in reply.text
    assert left_in(data) == []


def test_too_little_free_space_is_refused_before_anything_is_written(
    client: TestClient, data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(upload.shutil, "disk_usage", lambda _: Usage(5000))
    reply = client.post("/upload", content=b"\0" * 6000, headers=OWN_PAGE)
    assert reply.status_code == 200 and "Error code E_DISK_FULL" in reply.text
    assert "<header" not in reply.text and 'class="error-card' in reply.text
    assert left_in(data) == []


class Usage:
    """What shutil.disk_usage returns, with only `free` read."""

    def __init__(self, free: int) -> None:
        self.free = free


PAYLOADS = [
    "<script>alert(1)</script>.wav",
    '<img src=x onerror="alert(1)">.wav',
    '" hx-get="/x" hx-trigger="load" x=".wav',
    "evil‮wav.exe",
]


@pytest.mark.parametrize("name", PAYLOADS)
def test_a_hostile_file_name_is_shown_as_inert_text(client: TestClient, tmp_path: Path, name: str) -> None:
    source = wav_of_size(tmp_path / "a.wav", 40000)
    reply = client.post(
        "/upload", content=source.read_bytes(), headers={**OWN_PAGE, "x-file-name": quote(name)}
    )
    assert reply.status_code == 200, reply.text
    tags = parsed(reply.text)
    names = {t for t, _ in tags.tags}
    assert names <= {"div", "p", "span", "svg", "path", "dl", "dt", "dd", "input"}, names
    attributes = {k for _, a in tags.tags for k in a}
    assert not [k for k in attributes if k.startswith("on") or k in ("hx-get", "hx-trigger", "src")]
    assert tags.tags[0] == ("div", {"class": "upload", "hx-disable": None})  # htmx ignores what is inside
    assert "‮" not in reply.text
    shown = [a["value"] for t, a in tags.tags if t == "input" and a.get("name") == "upload_name"]
    assert shown == [name.replace("‮", "")]  # the same characters, as text


@pytest.mark.parametrize(
    ("length", "shown"), [(119, "n" * 119), (120, "n" * 120), (121, "n" * 117 + "...")]
)  # fmt: skip
def test_a_long_file_name_is_cut_to_120_characters(
    client: TestClient, tmp_path: Path, length: int, shown: str
) -> None:
    source = wav_of_size(tmp_path / "a.wav", 40000)
    name = "n" * length
    reply = client.post(
        "/upload", content=source.read_bytes(), headers={**OWN_PAGE, "x-file-name": quote(name)}
    )
    tags = parsed(reply.text)
    assert [a["value"] for t, a in tags.tags if a.get("name") == "upload_name"] == [shown]
    assert f"<span>{shown}</span>" in reply.text  # the name the learner sees ends in "..." when shortened


def test_a_browser_that_goes_away_mid_upload_leaves_no_file(tmp_path: Path) -> None:
    messages = [
        {"type": "http.request", "body": b"\0" * 1000, "more_body": True},
        {"type": "http.disconnect"},
    ]

    async def receive() -> dict[str, object]:
        return messages.pop(0)

    scope = {"type": "http", "method": "POST", "path": "/upload", "headers": [], "query_string": b""}
    request = Request(scope, receive)  # type: ignore[arg-type]
    with pytest.raises(ClientDisconnect):
        asyncio.run(upload.receive(request, tmp_path, Limits(max_upload_mb=1)))
    assert left_in(tmp_path) == []


def test_only_well_formed_ids_of_finished_uploads_are_found(tmp_path: Path) -> None:
    tmp = upload.folder(tmp_path)
    tmp.mkdir()
    good = "0123456789abcdef0123456789abcdef"
    (tmp / f"{good}.upload").write_bytes(b"x")
    (tmp / f"{good[:-1]}e.part").write_bytes(b"x")
    # A file outside tmp/ that an ID with a path in it would reach: tmp/<good>/../../outside.upload.
    (tmp / good).mkdir()
    (tmp_path / "outside.upload").write_bytes(b"x")
    assert upload.find(tmp_path, good) == tmp / f"{good}.upload"
    for wrong in (good[:-1] + "e", good.upper(), "../" + good, good + "0", "", good + "/../../outside"):
        assert upload.find(tmp_path, wrong) is None, wrong


def test_uploads_left_by_a_stopped_app_are_deleted_when_it_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp = upload.folder(tmp_path)
    tmp.mkdir()
    for name in ("a.part", "b.upload", "keep.txt"):
        (tmp / name).write_bytes(b"x")
    (tmp / "job-folder").mkdir()

    class Recorder:
        def __init__(self, config: object) -> None:
            self.should_exit = False

        def run(self) -> None:
            pass

    monkeypatch.setattr(server.uvicorn, "Server", Recorder)
    assert server.run(tmp_path, 0, browser=False, moved=[]) == 0
    assert left_in(tmp_path) == ["job-folder", "keep.txt"]
