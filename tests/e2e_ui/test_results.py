"""The first run, the mode choice, the results page and Copy for AI in a real browser, on a finished job of
chosen words."""

from __future__ import annotations

import pytest
from playwright.sync_api import Browser

from fluentfolio.core.files import read_json, write_json
from tests.support.browser import Served, signed_in

pytestmark = pytest.mark.slow  # a real browser; the full check runs it

JOB = "20261013-120000-abcd"  # the conftest's finished job
AUDIO_S = 6.0


def test_the_first_run_opens_on_the_components_page(browser: Browser, first_run: Served) -> None:
    page = signed_in(browser, first_run)
    page.wait_for_url(f"{first_run.origin}/components?first=1")
    assert page.get_by_role("heading", level=1).inner_text() == "Set up Fluentfolio"


def test_no_mode_is_chosen_on_first_use_and_each_shows_its_picture(browser: Browser, served: Served) -> None:
    page = signed_in(browser, served)
    page.goto(served.origin + "/")
    modes = page.locator('input[name="mode"]')
    assert modes.count() == 3 and all(not modes.nth(k).is_checked() for k in range(3))
    for k in range(3):
        box = page.locator(".option").nth(k).locator("svg.diagram").bounding_box()
        assert box and box["width"] >= 100 and box["height"] >= 20  # drawn, at the size the CSS gives it
    # Questions about the other person wait until a mode with two people is chosen.
    agreement = page.get_by_role("group", name="Agreement")
    assert not agreement.is_visible()
    page.check('input[name="mode"][value="interview"]')
    assert agreement.is_visible()
    page.check('input[name="mode"][value="solo"]')
    assert not agreement.is_visible()


def test_the_last_used_mode_is_chosen_and_a_track_preview_stops_after_5_s(
    browser: Browser, served_job: Served
) -> None:
    page = signed_in(browser, served_job)
    page.goto(served_job.origin + "/")
    assert page.locator('input[name="mode"]:checked').get_attribute("value") == "solo"  # the job's mode
    # The job as if its one voice came on track 1 of two, from 0.4 s, so a 5 s preview ends before the
    # recording's end at 6 s.
    job = served_job.home / "jobs" / JOB
    evidence = read_json(job / "evidence.json")
    evidence["manifest"]["layout"] = "per-track"
    write_json(job / "evidence.json", evidence)
    words = read_json(job / "stages" / "words.json")
    for w in words["words"]:
        w["track"] = 1
    words["words"][0]["start_ms"] = 400
    write_json(job / "stages" / "words.json", words)
    page.goto(f"{served_job.origin}/jobs/{JOB}")
    page.wait_for_function("() => document.getElementById('player').readyState >= 1", timeout=15_000)
    page.get_by_role("button", name="Play 5 s from 0:00").click()
    player = "document.getElementById('player')"
    page.wait_for_function(f"() => !{player}.paused", timeout=5_000)
    page.wait_for_function(f"() => {player}.paused", timeout=10_000)
    at, ended = page.evaluate(f"() => [{player}.currentTime, {player}.ended]")
    assert not ended and 5.4 <= at < 5.9  # paused by the preview, within a time update of its end


def test_copy_for_ai_puts_exactly_the_downloaded_file_on_the_clipboard(
    browser: Browser, served_job: Served
) -> None:
    page = signed_in(browser, served_job, permissions=["clipboard-read", "clipboard-write"])
    page.goto(f"{served_job.origin}/jobs/{JOB}/export")
    page.get_by_role("button", name="Copy for AI").click()
    status = page.locator('[data-copy-result="text-candidate"]')
    status.filter(has_text="Copied").wait_for()
    download = page.request.get(f"{served_job.origin}/jobs/{JOB}/files/for_ai.txt")
    assert download.ok and len(download.text()) > 1000
    assert page.locator("#text-candidate").input_value() == download.text()  # what the button copies
    # The clipboard holds the same text; Windows keeps its line ends as CR LF there.
    copied = page.evaluate("() => navigator.clipboard.readText()").replace("\r\n", "\n")
    assert copied == download.text()
    assert status.inner_text() == f"Copied {len(copied):,} characters."


def test_when_the_browser_refuses_the_clipboard_the_text_is_selected_for_ctrl_c(
    browser: Browser, served_job: Served
) -> None:
    page = signed_in(browser, served_job)
    # A browser that refuses the clipboard, as some do for a page on this computer.
    page.add_init_script(
        "navigator.clipboard.writeText = () => Promise.reject(new DOMException('refused', 'NotAllowedError'))"
    )
    page.goto(f"{served_job.origin}/jobs/{JOB}/export")
    page.get_by_role("button", name="Copy for AI").click()
    status = page.locator('[data-copy-result="text-candidate"]')
    status.filter(has_text="selected").wait_for()
    assert status.inner_text() == "The text is selected. Press Ctrl+C to copy it."
    box = page.evaluate(
        "() => { const e = document.activeElement;"
        " return [e.id, e.selectionStart, e.selectionEnd, e.value.length]; }"
    )
    assert box[0] == "text-candidate" and box[1] == 0 and box[2] == box[3] > 1000


def test_the_player_seeks_from_the_time_buttons_and_answers_its_keys(
    browser: Browser, served_job: Served
) -> None:
    page = signed_in(browser, served_job)
    page.goto(f"{served_job.origin}/jobs/{JOB}?tab=transcript")  # the player sits above the transcript
    page.wait_for_function("() => document.getElementById('player').readyState >= 1", timeout=15_000)
    # Seekable to the end: the server answers range requests, so the browser can jump anywhere.
    seekable = page.evaluate(
        "() => { const s = document.getElementById('player').seekable; return s.end(s.length - 1); }"
    )
    assert seekable == pytest.approx(AUDIO_S, abs=0.05)
    page.locator('.transcript button[data-seek="1000"]').click()  # the utterance that starts at 1.0 s
    at = page.evaluate("() => document.getElementById('player').currentTime")
    assert at == pytest.approx(1.0, abs=0.3)  # it may play on a little before this is read
    page.evaluate("() => { const p = document.getElementById('player'); p.pause(); p.currentTime = 0; }")
    page.locator("h1").click()  # the focus on the page itself, not on a control
    player = "document.getElementById('player')"
    page.keyboard.press("ArrowRight")
    assert page.evaluate(f"() => {player}.currentTime") == pytest.approx(5.0, abs=0.05)
    page.keyboard.press("ArrowRight")  # never past the end
    assert page.evaluate(f"() => {player}.currentTime") == pytest.approx(AUDIO_S, abs=0.05)
    page.keyboard.press("ArrowLeft")
    assert page.evaluate(f"() => {player}.currentTime") == pytest.approx(AUDIO_S - 5, abs=0.05)
    page.keyboard.press(" ")
    page.wait_for_function(f"() => !{player}.paused", timeout=5_000)
    page.keyboard.press(" ")
    page.wait_for_function(f"() => {player}.paused", timeout=5_000)
