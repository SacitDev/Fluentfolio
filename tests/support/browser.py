"""The app in a real browser: the installed Edge on Windows, Playwright's Chromium headless shell on Linux.

No browser is downloaded on Windows. The helpers start the real server, open it as the shortcut does, and
take a recording through the New analysis page the way a learner does.
"""

from __future__ import annotations

import re
import sys
import time
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from playwright.sync_api import Browser, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from fluentfolio.core.files import write_json


@dataclass(frozen=True)
class Served:
    home: Path
    port: int
    pid: int  # the web server's own process
    open_html: Path

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@contextmanager
def edge() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        if sys.platform == "win32":
            launched = playwright.chromium.launch(channel="msedge")
        else:
            launched = playwright.chromium.launch()
        yield launched
        launched.close()


@contextmanager
def running_app(serving: ModuleType, home: Path, app: Path) -> Iterator[Served]:
    proc = serving.start(home, app)
    try:
        info = serving.running(home, proc)
        yield Served(home, int(info["port"]), int(info["pid"]), home / "open.html")
    finally:
        serving.stop(proc)


def checked_app(app: Path) -> Path:
    """An app folder whose last self-test passed, so the app opens on New analysis, not on its first run."""
    write_json(app / "state.json", {"selftest": {"passed": True}})
    return app


CLOSED = "http://127.0.0.1:9"  # a closed local port: a request sent there fails at once


@contextmanager
def stubbed(folder: Path) -> Iterator[None]:
    """The model steps stood in for (FF_TEST_STUB=1), and no network for the server and its children:
    Hugging Face offline, every other request sent to a closed port, and a token file of the test's own."""
    with pytest.MonkeyPatch.context() as env:
        env.setenv("FF_TEST_STUB", "1")
        env.setenv("HF_HUB_OFFLINE", "1")
        env.setenv("HF_TOKEN_PATH", str(folder / "hf" / "token"))
        env.setenv("HF_TOKEN", "")
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env.setenv(name, CLOSED)
        env.setenv("NO_PROXY", "127.0.0.1,localhost")
        yield


def speech_like(path: Path, seconds: float) -> Path:
    """Noise in bursts of 1.6 s with 0.4 s of silence between them: loud enough to be speech to the decoder,
    and with no steady pitch, so no music is found in it. The stand-in steps supply the words."""
    rate = 16000
    noise = np.random.default_rng(7).normal(0, 0.15, int(rate * seconds))
    noise[(np.arange(noise.size) % (2 * rate)) >= int(1.6 * rate)] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(noise, -1, 1) * 32767).astype("<i2").tobytes())
    return path


JOB_URL = re.compile(r"/jobs/(\d{8}-\d{6}-[0-9a-f]{4})$")


def signed_in(browser: Browser, served: Served, **context: Any) -> Page:
    """A new browser window that opened the launch file, as the shortcut does."""
    page = browser.new_context(**context).new_page()
    page.goto(served.open_html.as_uri())
    # The sign-in page moves on to the first page by itself; another visit before that would cut it short.
    page.wait_for_url(lambda url: url.startswith(served.origin) and "/auth" not in url)
    page.wait_for_load_state("load")
    return page


def upload(page: Page, served: Served, recording: Path, mode: str) -> str:
    """Choose the mode and the recording on the New analysis page; returns what the page says about it."""
    page.goto(served.origin + "/")
    page.check(f'input[name="mode"][value="{mode}"]')
    page.set_input_files("#recording", str(recording))
    found = page.locator("#recording-result .upload")
    found.wait_for(timeout=60_000)
    return found.inner_text()


def start(page: Page, consent: bool) -> str:
    """Press Start analysis; returns the new job's ID from the page it opens."""
    if consent:
        page.check('input[name="consent"]')
    page.get_by_role("button", name="Start analysis").click()
    page.wait_for_url(JOB_URL)
    found = JOB_URL.search(page.url)
    assert found
    return found.group(1)


def follow(page: Page, limit_s: float = 120) -> list[float]:
    """The times the progress list changed, until the results or the stopped view replace it."""
    changes: list[float] = []
    last = None
    until = time.monotonic() + limit_s
    while time.monotonic() < until:
        if page.locator("nav.tabs, #stopped-title").count():
            return changes
        progress = page.locator("[data-progress]")
        try:
            shown = progress.inner_html(timeout=1000) if progress.count() else None
        except PlaywrightTimeoutError:  # the results replaced the list between the two reads
            shown = None
        if shown is not None and shown != last:
            changes.append(time.monotonic())
            last = shown
        page.wait_for_timeout(200)
    raise AssertionError(f"no results within {limit_s} s")
