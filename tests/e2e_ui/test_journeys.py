"""A learner's journeys through the web app in a real browser, with the model steps stood in for.

FF_TEST_STUB=1 makes every analysis the server starts use the scripted stand-ins of fluentfolio.jobs.stub,
so it takes seconds and needs no model; every step after them is the real code. The server has no network:
Hugging Face runs offline and every other request goes to a closed port.
"""

from __future__ import annotations

import json
import re
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import ModuleType

import jsonschema
import psutil
import pytest
from playwright.sync_api import Browser, Locator, Page, Response, Route

from fluentfolio.core.files import read_json
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

SCHEMA = Path(__file__).resolve().parents[2] / "docs" / "schemas" / "evidence-v1.schema.json"
RECORDING_S = 210  # longer than the stand-in's script, so the whole mock test fits
EVENT_GAP_S = 5.0  # live progress: an update at least this often
CANCEL_S = 10.0  # Cancel ends the analysis within this time


@dataclass(frozen=True)
class Analysis:
    served: Served
    job_id: str
    upload_text: str
    changes: list[float]

    @property
    def job(self) -> Path:
        return self.served.home / "jobs" / self.job_id


@pytest.fixture(scope="module")
def stub_app(tmp_path_factory: pytest.TempPathFactory, serving: ModuleType) -> Iterator[Served]:
    folder = tmp_path_factory.mktemp("journeys")
    with stubbed(folder), running_app(serving, folder / "home", checked_app(folder / "app")) as found:
        yield found


@pytest.fixture(scope="module")
def analysed(browser: Browser, stub_app: Served, tmp_path_factory: pytest.TempPathFactory) -> Analysis:
    """One interview analysed through the pages, as a learner does it: upload, agreement, Start."""
    recording = speech_like(tmp_path_factory.mktemp("recording") / "mock test.wav", RECORDING_S)
    page = signed_in(browser, stub_app)
    shown = upload(page, stub_app, recording, "interview")
    job_id = start(page, consent=True)
    changes = follow(page)
    page.context.close()
    return Analysis(stub_app, job_id, shown, changes)


def test_u1_a_first_run_lists_the_missing_parts_and_the_licence_box_starts_unticked(
    browser: Browser, tmp_path: Path, serving: ModuleType
) -> None:
    with stubbed(tmp_path), running_app(serving, tmp_path / "home", tmp_path / "app") as served:
        page = signed_in(browser, served)
        page.wait_for_url(f"{served.origin}/components?first=1")
        assert page.get_by_role("heading", level=1).inner_text() == "Set up Fluentfolio"
        assert page.locator("ol.steps li").count() == 3
        # A computer with no parts: the speech models need downloading, and the total says how much.
        row = page.locator("tr", has=page.get_by_text("Parakeet TDT 0.6B v2", exact=True))
        assert "Needs download" in row.inner_text()
        assert re.match(r"To download: [1-9][0-9.,]* (MB|GB)", page.locator("#totals").inner_text())
        assert not page.locator('input[name="licence.opensmile"]').is_checked()
        page.goto(served.origin + "/")  # New analysis waits until the check has passed
        page.wait_for_url(f"{served.origin}/components?first=1")
        page.context.close()


def test_u2_an_analysis_shows_its_estimate_live_progress_and_results_with_roles_parts_and_statuses(
    browser: Browser, analysed: Analysis
) -> None:
    assert "Time to analyse it on this computer's" in analysed.upload_text  # shown before Start
    gaps = [b - a for a, b in zip(analysed.changes, analysed.changes[1:], strict=False)]
    assert len(analysed.changes) >= 3 and max(gaps) <= EVENT_GAP_S, gaps
    evidence = read_json(analysed.job / "evidence.json")
    assert evidence["manifest"]["stub"] is True  # the stand-ins ran, and the evidence says so
    origin, job = analysed.served.origin, analysed.job_id
    page = signed_in(browser, analysed.served)
    page.goto(f"{origin}/jobs/{job}")
    assert page.locator(".page-head .tag").inner_text() in ("Finished", "Finished, with gaps")
    page.goto(f"{origin}/jobs/{job}?tab=transcript")
    text = page.locator(".transcript").inner_text()
    assert "The examiner, asking" in text and "The candidate, answering" in text  # roles in words
    for part in ("Part 1", "Part 2", "Part 3"):
        assert part in text, part
    page.goto(f"{origin}/jobs/{job}?tab=measures")
    page.get_by_text("More details: what each step did").click()
    steps = page.locator("details[open] li").all_inner_texts()
    words = "|".join(("done, with gaps", "done", "skipped", "not available", "failed", "switched off"))
    assert len(steps) == len(evidence["manifest"]["tools"])
    assert all(re.match(rf"[A-Z][^:]+: ({words})", s) for s in steps), steps
    page.context.close()


def test_u4_the_zip_holds_evidence_that_follows_the_published_schema(
    browser: Browser, analysed: Analysis
) -> None:
    page = signed_in(browser, analysed.served)
    page.goto(f"{analysed.served.origin}/jobs/{analysed.job_id}/export")
    link = page.get_by_role("link", name="Download all texts as a ZIP file")
    reply = page.request.get(analysed.served.origin + str(link.get_attribute("href")))
    assert reply.ok
    with zipfile.ZipFile(BytesIO(reply.body())) as archive:
        assert archive.testzip() is None
        evidence = json.loads(archive.read("details/evidence.json"))
        assert any(n.startswith("for_ai/") and n.endswith(".txt") for n in archive.namelist())
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(evidence)
    assert evidence == read_json(analysed.job / "evidence.json")
    page.context.close()


def test_leaving_the_transcript_while_its_audio_loads_prints_no_error_in_the_server_window(
    browser: Browser, analysed: Analysis, serving: ModuleType
) -> None:
    page = signed_in(browser, analysed.served)
    for _ in range(20):  # each new page drops the audio connection the one before had open
        page.goto(f"{analysed.served.origin}/jobs/{analysed.job_id}?tab=transcript")
        page.wait_for_timeout(300)
    page.context.close()
    window = serving.output(analysed.served.home).read_text(encoding="utf-8", errors="replace")
    assert "Traceback" not in window, window[-1500:]


def before(first: Locator, then: Locator) -> bool:
    """Whether the first element comes earlier in the page than the second."""
    return bool(first.evaluate("(a, b) => !!(a.compareDocumentPosition(b) & 4)", then.element_handle()))


def test_priv05_a_two_person_job_warns_of_the_partner_before_each_copy_and_upload(
    browser: Browser, analysed: Analysis
) -> None:
    page = signed_in(browser, analysed.served)
    page.goto(f"{analysed.served.origin}/jobs/{analysed.job_id}/export")
    words = page.get_by_text("This text contains your partner's words; it goes to that AI company.")
    recording = page.get_by_text("Your own recording holds your partner's voice all the way through.")
    audio = page.get_by_text("Your partner's voice can still be heard where the speakers were mixed up")
    copy = page.locator('button[data-copy^="text-"]').first
    make = page.get_by_role("button", name="Make the audio file")
    assert all(w.is_visible() for w in (words, recording, audio, copy, make))
    assert before(words, copy) and before(recording, copy) and before(audio, make)
    page.context.close()


def test_priv06_history_and_the_job_say_what_deleting_does_and_a_synced_data_folder_is_named(
    browser: Browser, analysed: Analysis, tmp_path: Path, serving: ModuleType
) -> None:
    line = "Deleting skips the Recycle Bin. On SSDs and in synced folders it is not secure erasure."
    synced = "Your data folder is inside OneDrive or Dropbox, so copies of it may also be kept online."
    page = signed_in(browser, analysed.served)
    page.goto(f"{analysed.served.origin}/jobs")
    assert page.get_by_text(line).is_visible() and not page.get_by_text(synced).count()
    page.goto(f"{analysed.served.origin}/jobs/{analysed.job_id}")
    page.get_by_text("Delete this analysis", exact=True).first.click()  # opens the details
    assert page.get_by_text(line).is_visible()
    page.context.close()
    synced_home = tmp_path / "OneDrive" / "home"
    synced_home.mkdir(parents=True)  # the server's window log goes beside it
    with stubbed(tmp_path), running_app(serving, synced_home, checked_app(tmp_path / "app")) as served:
        page = signed_in(browser, served)
        page.goto(f"{served.origin}/jobs")
        assert page.get_by_text(synced).is_visible() and page.get_by_text(line).is_visible()
        page.context.close()


SETTINGS_PAGES = ("/settings", "/settings?tab=advanced")
# Each setting as the form shows it: a box ticked or not, the chosen option, a number.
SHOWN = """() => Object.fromEntries([...document.querySelectorAll('#settings input')]
    .filter(e => e.type !== 'hidden' && (e.type !== 'radio' || e.checked))
    .map(e => [e.name, e.type === 'checkbox' ? String(e.checked) : e.value]))"""


def settings_shown(page: Page, origin: str) -> dict[str, str]:
    shown: dict[str, str] = {}
    for address in SETTINGS_PAGES:
        page.goto(origin + address)
        shown.update(page.evaluate(SHOWN))
    return shown


def saving(response: Response) -> bool:
    return response.url.endswith("/settings") and response.request.method == "POST"


def change_every_setting(page: Page, origin: str) -> dict[str, str]:
    """Move each setting on both pages away from its value, one at a time as a learner would."""
    chosen: dict[str, str] = {}
    for address in SETTINGS_PAGES:
        page.goto(origin + address)
        for name, value in page.evaluate(SHOWN).items():
            if name == "ui.theme":  # a theme change draws the page again; it is chosen last
                continue
            control = page.locator(f'#settings input[name="{name}"]')
            kind = control.first.get_attribute("type")
            with page.expect_response(saving) as answer:
                if kind == "checkbox":
                    new = "false" if value == "true" else "true"
                    control.set_checked(new == "true")
                elif kind == "radio":
                    new = [v for v in control.evaluate_all("es => es.map(e => e.value)") if v != value][-1]
                    page.check(f'#settings input[name="{name}"][value="{new}"]')
                else:
                    new = str(int(value) + 1)
                    control.fill(new)
                    control.press("Tab")  # the change counts once the field is left
            assert answer.value.text().startswith("Saved: "), (name, answer.value.text())
            chosen[name] = new
    page.goto(origin + "/settings")
    with page.expect_navigation():
        page.get_by_role("radio", name="Dark").check()
    chosen["ui.theme"] = "dark"
    return chosen


def test_u5_every_setting_is_kept_after_a_restart_and_the_theme_comes_without_a_cookie(
    browser: Browser, tmp_path: Path, serving: ModuleType
) -> None:
    home, app = tmp_path / "home", checked_app(tmp_path / "app")
    with stubbed(tmp_path):
        with running_app(serving, home, app) as served:
            page = signed_in(browser, served)
            first = settings_shown(page, served.origin)
            chosen = change_every_setting(page, served.origin)
            assert set(chosen) == set(first) and all(chosen[k] != first[k] for k in first)
            assert {"speakers.backend", "limits.max_min", "analyzers.pronunciation"} <= set(chosen)
            page.context.close()
        with running_app(serving, home, app) as served:
            page = signed_in(browser, served, color_scheme="light")  # a new window: no theme cookie
            assert settings_shown(page, served.origin) == chosen
            assert page.locator("html").get_attribute("data-theme") == "dark"
            page.goto(served.origin + "/components")
            assert not page.locator('input[name="licence.opensmile"]').is_checked()
            page.context.close()


def test_the_check_answers_right_under_its_button_which_is_busy_while_its_request_runs(
    browser: Browser, tmp_path: Path, serving: ModuleType
) -> None:
    with stubbed(tmp_path), running_app(serving, tmp_path / "home", checked_app(tmp_path / "app")) as served:
        page = signed_in(browser, served)
        page.goto(served.origin + "/components")
        button = page.get_by_role("button", name="Run the check")

        held: list[Route] = []

        def hold_first(route: Route) -> None:
            if held:  # the page's polls of /tasks/selftest go on
                route.continue_()
            else:
                held.append(route)  # the check's own request waits until its busy state has been seen

        page.route("**/selftest", hold_first)
        button.click()
        assert button.get_attribute("aria-busy") == "true" and button.is_disabled()
        assert page.locator("#selftest-busy").is_visible()
        assert len(held) == 1
        held[0].continue_()
        running = page.locator("#selftest-status .task")
        running.wait_for()
        assert button.get_attribute("aria-busy") is None and button.is_enabled()
        gap = running.bounding_box()["y"] - button.bounding_box()["y"]  # type: ignore[index]
        assert 0 < gap < 200, gap  # right under the button, not at the top of a long page
        outcome = page.locator("#selftest-status .outcome")
        outcome.wait_for(timeout=120_000)  # the page loads again with the result, in the same place
        assert 0 < outcome.bounding_box()["y"] - button.bounding_box()["y"] < 200  # type: ignore[index]
        page.context.close()


def test_u6_a_file_that_is_not_audio_gets_a_plain_error_card(
    browser: Browser, stub_app: Served, tmp_path: Path
) -> None:
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"this is not a recording " * 400)
    page = signed_in(browser, stub_app)
    page.set_input_files("#recording", str(broken))
    card = page.locator("#recording-result .error-card")
    card.wait_for(timeout=30_000)
    text = card.inner_text()
    assert "Traceback" not in text and re.search(r"Error code E_[A-Z_]+", text)
    assert card.get_by_role("link").count() == 1  # one next step
    card.get_by_text("Diagnostic report", exact=True).click()
    assert card.get_by_role("button", name="Copy diagnostic report").is_visible()
    page.context.close()


def test_u6_cancel_ends_the_analysis_within_10_s_and_leaves_no_process(
    browser: Browser, stub_app: Served, tmp_path: Path
) -> None:
    recording = speech_like(tmp_path / "cancel me.wav", RECORDING_S)
    page = signed_in(browser, stub_app)
    upload(page, stub_app, recording, "solo")
    job_id = start(page, consent=False)
    page.locator("[data-progress] li.running").first.wait_for(timeout=30_000)
    pressed = time.monotonic()
    page.get_by_role("button", name="Cancel analysis").click()
    page.locator("#stopped-title").wait_for(timeout=int(CANCEL_S * 1000))
    assert time.monotonic() - pressed <= CANCEL_S
    assert page.locator("#stopped-title").inner_text() == "Cancelled"
    assert read_json(stub_app.home / "jobs" / job_id / "job.json")["state"] == "cancelled"
    # On Linux a child that has ended stays a zombie until the server next polls it; a zombie runs nothing.
    children = psutil.Process(stub_app.pid).children(recursive=True)
    left = [p.cmdline() for p in children if p.status() != psutil.STATUS_ZOMBIE]
    assert left == [], left  # the analysis child and anything it started are gone
    page.context.close()


def test_u7_the_token_field_is_masked_and_its_value_is_never_shown_or_written(
    browser: Browser, stub_app: Served, shots: Path
) -> None:
    secret = "hf_" + "z" * 34  # a made-up token of the right shape; the server has no network
    page = signed_in(browser, stub_app, viewport={"width": 1280, "height": 720})
    page.goto(stub_app.origin + "/token")
    field = page.locator("#token-value")
    assert field.get_attribute("type") == "password"
    field.fill(secret)
    page.get_by_label("Show the token").check()
    assert field.get_attribute("type") == "text"
    page.get_by_label("Show the token").uncheck()
    assert field.get_attribute("type") == "password"
    page.screenshot(
        path=str(shots / "token-filled-light-1280.png"), full_page=True, caret="initial"
    )  # masked
    assert secret not in page.content()  # typed text is the field's value, never written into the page
    page.get_by_role("button", name="Save token").click()
    # Offline: the token cannot be checked, so it is not saved, and the form stays for another try.
    page.locator("#token .notice").filter(has_text="could not reach Hugging Face").wait_for(timeout=30_000)
    assert page.locator("#token-value").get_attribute("type") == "password"
    assert secret not in page.content()
    folder = stub_app.home.parent  # the server's data and app folders and its token file
    assert not (folder / "hf" / "token").exists()
    written, locked = [], []
    for path in (p for p in folder.rglob("*") if p.is_file()):
        try:
            if secret.encode() in path.read_bytes():
                written.append(path)
        except PermissionError:  # a lock the running server holds; it is checked by name
            locked.append(path.name)
    assert written == []  # nowhere in the data, the logs or the app state
    assert all(name.endswith(".lock") for name in locked), locked
    page.context.close()
