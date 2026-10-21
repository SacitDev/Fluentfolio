"""Every page in both themes at two widths, then every control on every page, in a real browser.

Each page gets a full-page screenshot, an axe-core accessibility check and a check for cut-off text. The model
steps are stood in for (FF_TEST_STUB=1) and the server has no network, so the analyses take seconds.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import psutil
import pytest
from axe_playwright_python.sync_playwright import Axe
from playwright.sync_api import Browser, BrowserContext, ConsoleMessage, Error, Page

from fluentfolio.core.files import read_json, write_json
from fluentfolio.web.views import HELP_TABS
from tests.support.browser import (
    Served,
    checked_app,
    follow,
    running_app,
    signed_in,
    speech_like,
    start,
    stubbed,
    upload,
)

pytestmark = pytest.mark.slow  # a real browser and a real server; the full check runs them

THEMES = ("light", "dark")
WIDTHS = (1280, 640)
HEIGHT = 720
RECORDING_S = 210
SERIOUS = ("serious", "critical")
MISSING = "/no-such-page"  # the not-found page; loading it logs its 404, as every browser does
SETTLED = """() => !document.querySelector(".htmx-added, .htmx-settling, .htmx-swapping")
  && document.getAnimations().every((a) => a.playState !== "running")"""
BUSY = ("queued", "running")
SETTLE_S = 300.0  # the longest a clicked control's own work (a job, an install, the check) may run

# Visible text that is wider than its box, outside the boxes that scroll on purpose, and sideways page scroll.
CUT_OFF = """() => {
  const scrolls = ".table-wrap, textarea, pre, select, option";
  const cut = [];
  for (const el of document.body.querySelectorAll("*")) {
    if (el.closest(scrolls) || el.closest("svg")) continue;
    const style = getComputedStyle(el);
    if (style.display === "inline" || style.display === "none" || style.visibility === "hidden") continue;
    const own = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
    if (own && el.scrollWidth > el.clientWidth + 1) {
      cut.push(el.tagName + ": " + el.textContent.trim().slice(0, 50));
    }
  }
  return {cut, sideways: document.documentElement.scrollWidth > window.innerWidth};
}"""


@dataclass
class Findings:
    """What the tour found, by page; written next to the screenshots for the review."""

    pages: list[str] = field(default_factory=list)
    console: list[str] = field(default_factory=list)
    axe: list[str] = field(default_factory=list)
    cut_off: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=1), encoding="utf-8")


def listen(page: Page, found: Findings) -> None:
    """Every console error and page error, with the address it came from. Expected: only the not-found page's
    own load, which answers 404 on purpose."""

    def console(message: ConsoleMessage) -> None:
        if message.type != "error":
            return
        if page.url.endswith(MISSING) and "status of 404" in message.text:
            return
        found.console.append(f"{page.url}: {message.text} ({message.location.get('url', '')})")

    def page_error(error: Error) -> None:
        found.console.append(f"{page.url}: page error: {error.message}")

    page.on("console", console)
    page.on("pageerror", page_error)


def window(browser: Browser, served: Served, theme: str, width: int, found: Findings) -> Page:
    """A signed-in window of one size, with the theme chosen in its cookie as Settings would."""
    page = signed_in(browser, served, viewport={"width": width, "height": HEIGHT}, color_scheme=theme)
    themed(page.context, served, theme)
    listen(page, found)
    return page


def themed(context: BrowserContext, served: Served, theme: str) -> None:
    context.add_cookies([{"name": "theme", "value": theme, "url": served.origin}])


def look(page: Page, shots: Path, name: str, theme: str, width: int, found: Findings) -> None:
    """Screenshot, axe-core and the cut-off check of the page as it is now."""
    label = f"{name}-{theme}-{width}"
    page.wait_for_function(SETTLED)  # an answer that fades in is measured at full strength
    # caret="initial": hiding the caret writes an inline style into each field, which the page would carry on
    page.screenshot(path=str(shots / f"{label}.png"), full_page=True, caret="initial")
    found.pages.append(label)
    for v in Axe().run(page).response["violations"]:
        if v["impact"] in SERIOUS:
            node = v["nodes"][0]
            found.axe.append(
                f"{label}: {v['id']} ({v['impact']}) at {node['target']}: {node['failureSummary']}"
            )
    measured = page.evaluate(CUT_OFF)
    found.cut_off += [f"{label}: {c}" for c in measured["cut"]]
    if measured["sideways"]:
        found.cut_off.append(f"{label}: the page scrolls sideways")


@pytest.fixture(scope="module")
def tour_app(
    browser: Browser, tmp_path_factory: pytest.TempPathFactory, serving: ModuleType
) -> Iterator[tuple[Served, str]]:
    """A server with one finished interview, analysed through the pages."""
    folder = tmp_path_factory.mktemp("tour")
    with stubbed(folder), running_app(serving, folder / "home", checked_app(folder / "app")) as served:
        page = signed_in(browser, served)
        upload(page, served, speech_like(folder / "mock test.wav", RECORDING_S), "interview")
        job = start(page, consent=True)
        follow(page)
        page.context.close()
        yield served, job


@pytest.fixture(scope="module")
def new_app(tmp_path_factory: pytest.TempPathFactory, serving: ModuleType) -> Iterator[Served]:
    """A server on a computer where Fluentfolio has never run: no parts, no check, no analyses."""
    folder = tmp_path_factory.mktemp("tour-new")
    with stubbed(folder), running_app(serving, folder / "home", folder / "app") as served:
        yield served


@pytest.fixture(scope="module")
def empty_app(tmp_path_factory: pytest.TempPathFactory, serving: ModuleType) -> Iterator[Served]:
    """A server whose check passed and that has no analysis yet, so the New page chooses no mode."""
    folder = tmp_path_factory.mktemp("tour-empty")
    with stubbed(folder), running_app(serving, folder / "home", checked_app(folder / "app")) as served:
        yield served


@contextmanager
def as_per_track(job: Path) -> Iterator[None]:
    """The tour's interview as if each voice had its own track, put back afterwards for the other tours."""
    files = [job / "evidence.json", job / "stages" / "words.json"]
    kept = [f.read_bytes() for f in files]
    try:
        evidence, words = (read_json(f) for f in files)
        evidence["manifest"]["layout"] = "per-track"
        for w in words["words"]:
            w["track"] = int(w["speaker"][1:]) if w["speaker"] else None
        for f, data in zip(files, (evidence, words), strict=True):
            write_json(f, data)
        yield
    finally:
        for f, data in zip(files, kept, strict=True):
            f.write_bytes(data)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("theme", THEMES)
def test_every_page_has_no_serious_accessibility_finding_no_cut_off_text_and_no_console_error(
    browser: Browser,
    tour_app: tuple[Served, str],
    new_app: Served,
    empty_app: Served,
    shots: Path,
    tmp_path: Path,
    theme: str,
    width: int,
) -> None:
    served, job = tour_app
    found = Findings()
    page = window(browser, served, theme, width, found)
    origin = served.origin

    def visit(name: str, path: str) -> None:
        page.goto(origin + path)
        look(page, shots, name, theme, width, found)

    visit("new-empty", "/")
    page.set_input_files("#recording", str(write_broken(tmp_path / "broken.wav")))
    page.locator("#recording-result .error-card").wait_for(timeout=30_000)
    look(page, shots, "new-error", theme, width, found)
    upload(page, served, speech_like(tmp_path / "mock test.wav", RECORDING_S), "solo")
    look(page, shots, "new-chosen", theme, width, found)
    start(page, consent=False)
    assert page.locator("#progress").count() == 1
    look(page, shots, "job-progress", theme, width, found)
    page.get_by_role("button", name="Cancel analysis").click()
    page.locator("#stopped-title").wait_for(timeout=30_000)
    look(page, shots, "job-stopped", theme, width, found)
    visit("history-list", "/jobs")
    for tab in ("overview", "measures", "transcript", "words"):
        visit(f"results-{tab}", f"/jobs/{job}?tab={tab}")
    visit("results-export", f"/jobs/{job}/export")
    with as_per_track(served.home / "jobs" / job):  # the review form then lists the tracks
        visit("results-tracks", f"/jobs/{job}")
        assert page.get_by_role("button", name="Play 5 s from").count() == 2
    visit("components-list", "/components")
    visit("settings-basic", "/settings")
    visit("settings-advanced", "/settings?tab=advanced")
    visit("token-empty", "/token")
    page.locator("#token-value").fill("hf_" + "q" * 34)
    page.get_by_role("button", name="Save token").click()
    page.locator("#token .notice").wait_for(timeout=30_000)  # offline: it could not be checked
    look(page, shots, "token-error", theme, width, found)
    visit("doctor-report", "/doctor")
    for tab in HELP_TABS:
        visit(f"help-{tab}", f"/about?tab={tab}")
    visit("notfound-error", MISSING)
    page.context.close()
    look_before_any_analysis(browser, (new_app, empty_app), shots, (theme, width), found)

    found.save(shots / f"tour-{theme}-{width}.json")
    assert len(found.pages) == 22 + len(HELP_TABS)  # each Help tab is its own page
    assert found.axe == []
    assert found.cut_off == []
    assert found.console == []


def look_before_any_analysis(
    browser: Browser, apps: tuple[Served, Served], shots: Path, size: tuple[str, int], found: Findings
) -> None:
    """The pages of a computer where Fluentfolio never ran, and the New page where it ran no analysis yet."""
    (new_app, empty_app), (theme, width) = apps, size
    first = window(browser, new_app, theme, width, found)
    first.goto(new_app.origin + "/")  # sends a new computer to the first-run page
    first.wait_for_url(new_app.origin + "/components?first=1")
    look(first, shots, "components-firstrun", theme, width, found)
    first.goto(new_app.origin + "/jobs")
    look(first, shots, "history-empty", theme, width, found)
    first.context.close()
    empty = window(browser, empty_app, theme, width, found)
    empty.goto(empty_app.origin + "/")
    assert empty.locator('input[name="mode"]:checked').count() == 0
    look(empty, shots, "new-firstuse", theme, width, found)
    empty.context.close()


def write_broken(path: Path) -> Path:
    path.write_bytes(b"this is not a recording " * 400)
    return path


# Controls a learner can activate. A file input opens the system's file dialog; the upload journeys cover it.
CONTROLS = "a[href], button, input[type=checkbox], input[type=radio], select, summary"
OPEN_ALL = "() => document.querySelectorAll('details').forEach((d) => { d.open = true; })"
RESULT_TABS = ("overview", "measures", "transcript", "words")


@dataclass(frozen=True)
class Place:
    """A page in one state: its address and what a learner does there first."""

    name: str
    path: str
    setup: Callable[[Page], None] = lambda page: None


def two_people(page: Page) -> None:
    page.check('input[name="mode"][value="role-swap"]')
    page.check('input[name="layout"][value="two-files"]')


@dataclass
class Clicking:
    """The server for the control tour, and copies of its jobs, settings and app state to put back."""

    served: Served
    app: Path
    saved: Path
    done: str
    stopped: str

    def places(self) -> list[Place]:
        return [
            Place("new", "/"), Place("new two people", "/", two_people), Place("history", "/jobs"),
            *[Place(f"results {t}", f"/jobs/{self.done}?tab={t}") for t in RESULT_TABS],
            Place("copy for AI", f"/jobs/{self.done}/export"), Place("stopped", f"/jobs/{self.stopped}"),
            Place("components", "/components"), Place("first run", "/components?first=1"),
            Place("settings", "/settings"), Place("settings advanced", "/settings?tab=advanced"),
            Place("token", "/token"), Place("doctor", "/doctor"),
            *[Place(f"help {t}", f"/about?tab={t}") for t in HELP_TABS],
            Place("not found", MISSING),
        ]  # fmt: skip

    def keep(self) -> None:
        shutil.copytree(self.served.home / "jobs", self.saved / "jobs")
        shutil.copy2(self.served.open_html, self.saved / "open.html")
        settings = self.served.home / "settings.toml"
        if settings.is_file():
            shutil.copy2(settings, self.saved / "settings.toml")

    def settle(self, page: Page) -> None:
        """Wait until what the control started has ended, then put the jobs, settings and app state back."""
        jobs = self.served.home / "jobs"
        until = time.monotonic() + SETTLE_S
        while self.busy(page, jobs):
            assert time.monotonic() < until, "a job or task still runs"
            time.sleep(0.25)
        for kept in (self.saved / "jobs").iterdir():
            here = jobs / kept.name
            if not here.is_dir() or (here / "job.json").read_bytes() != (kept / "job.json").read_bytes():
                shutil.rmtree(here, ignore_errors=True)
                shutil.copytree(kept, here)
        for here in jobs.iterdir():
            if not (self.saved / "jobs" / here.name).exists():
                shutil.rmtree(here)
        if not self.served.open_html.is_file():  # Delete all my data removes it; a new start writes it again
            shutil.copy2(self.saved / "open.html", self.served.open_html)
        settings = self.served.home / "settings.toml"
        if (self.saved / "settings.toml").is_file():
            shutil.copy2(self.saved / "settings.toml", settings)
        else:
            settings.unlink(missing_ok=True)
        checked_app(self.app)

    def busy(self, page: Page, jobs: Path) -> bool:
        states = [
            read_json(j / "job.json").get("state") for j in jobs.iterdir() if (j / "job.json").is_file()
        ]
        if any(s in BUSY for s in states):
            return True
        # A long action's status address answers with its progress line while it runs, else asks for a reload.
        for kind in ("install", "selftest", "export"):
            if "hx-refresh" not in page.request.get(f"{self.served.origin}/tasks/{kind}").headers:
                return True
        return False


@pytest.fixture
def clicking(browser: Browser, tmp_path: Path, serving: ModuleType) -> Iterator[Clicking]:
    """A server with one finished interview and one cancelled analysis, made through the pages."""
    app = checked_app(tmp_path / "app")
    with stubbed(tmp_path), running_app(serving, tmp_path / "home", app) as served:
        page = signed_in(browser, served)
        upload(page, served, speech_like(tmp_path / "mock test.wav", RECORDING_S), "interview")
        done = start(page, consent=True)
        follow(page)
        upload(page, served, speech_like(tmp_path / "mock test.wav", RECORDING_S), "solo")
        stopped = start(page, consent=False)
        page.get_by_role("button", name="Cancel analysis").click()
        page.locator("#stopped-title").wait_for(timeout=30_000)
        page.context.close()
        state = Clicking(served, app, tmp_path / "saved", done, stopped)
        state.keep()
        yield state


def ready(page: Page, place: Place, origin: str) -> list[Any]:
    """The page loaded afresh, its details opened so nothing is out of reach, and its visible controls."""
    page.goto(origin + place.path)
    place.setup(page)
    page.evaluate(OPEN_ALL)
    every = page.locator(CONTROLS)
    return [every.nth(i) for i in range(every.count()) if every.nth(i).is_visible()]


def activate(page: Page, control: Any, origin: str) -> str:
    """Use one control as a learner would; returns what was done, for the record."""
    tag = control.evaluate("(el) => el.tagName.toLowerCase()")
    text = " ".join(control.inner_text().split())[:60] if tag in ("a", "button", "summary") else ""
    name = text or str(control.get_attribute("name") or "") + "=" + str(control.get_attribute("value") or "")
    if not control.is_enabled():
        return f"{tag} {name}: disabled here"
    if tag == "a":
        href = str(control.get_attribute("href"))
        if href.startswith("http") and not href.startswith(origin):
            assert control.get_attribute("rel") == "noopener noreferrer", href
            return f"link {name}: leaves the app ({href.split('/')[2]}), not followed"
        reply = page.request.get(href if href.startswith("http") else origin + href)
        assert reply.status < 400, (href, reply.status)  # noqa: PLR2004
        if control.get_attribute("download") is not None:
            return f"link {name}: download answered {reply.status}"
    # By keyboard, as every control must work so; the skip link is on screen only while it has the focus.
    if tag == "select":
        control.select_option(index=control.locator("option").count() - 1)
    else:
        control.focus()
        page.keyboard.press("Space" if tag == "input" else "Enter")
    page.wait_for_load_state("load")
    page.wait_for_timeout(300)  # an answer put in place arrives within this on a local server
    return f"{tag} {name}: activated"


def test_every_control_on_every_page_works_without_a_console_error_in_both_themes(
    browser: Browser, clicking: Clicking, shots: Path
) -> None:
    found = Findings()
    origin = clicking.served.origin
    for theme in THEMES:
        page = window(browser, clicking.served, theme, WIDTHS[0], found)
        for place in clicking.places():
            controls = ready(page, place, origin)
            count = len(controls)
            for i in range(count):
                controls = ready(page, place, origin)
                assert len(controls) == count, f"{place.name}: the page changed between loads"
                if " ".join(controls[i].inner_text().split()) == "Quit":
                    continue  # last of all: it ends the server
                began = time.monotonic()
                done = activate(page, controls[i], origin)
                clicking.settle(page)
                found.controls.append(f"{theme} {place.name}: {done} ({time.monotonic() - began:.1f} s)")
                found.save(shots / "controls.json")  # kept as it grows, so a stop shows the last control
                themed(page.context, clicking.served, theme)
        page.context.close()
    page = window(browser, clicking.served, THEMES[0], WIDTHS[0], found)
    page.goto(origin + "/about")
    page.get_by_role("button", name="Quit").click()
    gone = time.monotonic() + 30
    while psutil.pid_exists(clicking.served.pid) and time.monotonic() < gone:
        time.sleep(0.25)
    found.controls.append(f"Quit: the server ended: {not psutil.pid_exists(clicking.served.pid)}")
    page.context.close()
    found.save(shots / "controls.json")
    assert not psutil.pid_exists(clicking.served.pid)
    assert found.console == []
    assert len(found.controls) > 2 * len(clicking.places()) * 8  # at least the header's controls on each page
