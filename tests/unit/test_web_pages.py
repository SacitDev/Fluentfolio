"""The web pages and actions through the real app, on hand-made jobs and on evidence saved from real runs.

Each expected value comes from the test: words and times it chose, the saved evidence it reads itself, or
the requirement's own wording. Long actions are never started here; their start function is replaced.
"""

from __future__ import annotations

import html
import inspect
import os
import re
import sys
import threading
import time
import wave
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from typing import Any, get_args
from urllib.parse import urlencode

import pytest
from fastapi.routing import APIRoute
from starlette.testclient import TestClient

from fluentfolio import __version__
from fluentfolio.analyzers import registry
from fluentfolio.app import facade
from fluentfolio.core import paths
from fluentfolio.core.enums import JobState
from fluentfolio.core.errors import AppError
from fluentfolio.core.errors import catalogue as catalogue_of_errors
from fluentfolio.core.files import read_json, write_json
from fluentfolio.doctor import checks
from fluentfolio.jobs import store
from fluentfolio.jobs.locks import work_lock
from fluentfolio.models import catalogue
from fluentfolio.models.catalogue import SYSTEM
from fluentfolio.settings.model import Section, Settings
from fluentfolio.web import app as web_app
from fluentfolio.web import pages, render, views
from fluentfolio.web.security import SESSION_COOKIE
from fluentfolio.web.tasks import Tasks
from tests.support.fake_https import local_server

ORIGIN = "http://127.0.0.1:8765"
SESSION = "session-for-tests"
OWN_PAGE = {"sec-fetch-site": "same-origin", "hx-request": "true"}
FORM = "application/x-www-form-urlencoded"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "evidence"
JOB = "20261013-120000-abcd"  # the hand-made job's ID
SAVED = "20261014-090000-beef"  # a job made here from saved evidence
CANARY = "quorbistan"  # a made-up word: after "delete all" it must be nowhere
SPOKEN = [
    ("Reading", 1000, 1500, 0), ("is", 1550, 1700, 0), ("useful", 1750, 2300, 0),
    ("because", 3600, 4100, 0), ("it", 4150, 4250, 0), ("helps.", 4300, 4800, 0),
]  # fmt: skip
SEGMENTS = [(1000, 2300), (3600, 4800)]  # one pause of 1.3 s inside the answer
UI_03_LINES = (
    "Unofficial practice feedback. Not an IELTS score.",
    "Fluentfolio is not affiliated with or endorsed by the British Council, IDP IELTS or Cambridge "
    "University Press & Assessment.",
    "Run the same prompt twice. If the estimates differ by more than one band, trust neither.",
)
COMMAND = re.compile(r"`|fluentfolio (components|token|self-test|doctor|export|cancel)")
# The tabs of a finished analysis, as a learner reads them, and the link of each.
TABS = {
    "Overview": f"/jobs/{JOB}", "Measures": f"/jobs/{JOB}?tab=measures",
    "Transcript": f"/jobs/{JOB}?tab=transcript", "Words to check": f"/jobs/{JOB}?tab=words",
    "Copy for AI": f"/jobs/{JOB}/export",
}  # fmt: skip
HELP = [f"/about?tab={key}" for key in views.HELP_TABS]  # each Help tab is its own link
# Each facade function a learner reaches from the pages, and the route that reaches it.
FACADE_ROUTES = {
    "settings_show": "GET /", "jobs": "GET /jobs", "analyze": "POST /jobs",
    "probe": "POST /upload", "estimates": "POST /upload",
    "results": "GET /jobs/{job_id}", "corrections": "GET /jobs/{job_id}",
    "status": "GET /jobs/{job_id}/events", "cancel": "POST /jobs/{job_id}/cancel",
    "resume_detached": "POST /jobs/{job_id}/resume", "review": "POST /jobs/{job_id}/review",
    "job_evidence": "POST /jobs/{job_id}/review", "rerun": "POST /jobs/{job_id}/rerun",
    "delete_job": "POST /jobs/{job_id}/delete", "delete_all": "POST /data/delete-all",
    "export_files": "GET /jobs/{job_id}/files/{name}", "for_ai_files": "GET /jobs/{job_id}/export",
    "hidden_words": "GET /jobs/{job_id}/export",
    "export": "POST /jobs/{job_id}/export",
    "components": "GET /components", "consent": "POST /components/install",
    "install": "POST /components/install", "remove": "POST /components/remove",
    "self_test": "POST /selftest", "token_set": "POST /token", "token_status": "POST /token/check",
    "token_clear": "POST /token/clear", "settings_set": "POST /settings",
    "settings_restore": "POST /settings/restore", "doctor": "GET /doctor",
    "diagnostic_report": "GET /doctor", "move_folder": "POST /settings/move",
    "check_updates": "POST /updates/check",
}  # fmt: skip
# Facade functions with no page of their own, and why.
NO_ROUTE = {
    "serve": "it starts the pages", "agent_guide": "for AI agents", "write_skill": "for AI agents",
    "wait": "the progress page follows events", "dry_run": "the upload shows the estimate",
    "start_detached": "the pages start a job with analyze", "resume": "the pages resume detached",
    "export_text": "the export page shows the files", "token_file_place": "the command's own message",
    "in_use": "a helper of install", "install_plan": "a helper of install",
    "check_removable": "a helper of remove", "plan": "a helper", "steps": "a helper",
    "check_layout": "a helper", "estimate_ms": "a helper", "job_estimate_ms": "a helper",
    "job_steps": "a helper", "rerun_offer": "a helper of the results page",
    "check_answer": "the answer check is on the command line only",
    "prompt": "the export page shows each file's own prompt, with its export ID filled in",
}  # fmt: skip


class Page(HTMLParser):
    """The start tags with their attributes, and the visible text, of a page or a fragment."""

    def __init__(self, source: str) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.words: list[str] = []
        self.feed(source)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data: str) -> None:
        self.words.append(data)

    @property
    def text(self) -> str:
        """The words as a reader sees them: one space between text nodes, none before a comma."""
        return " ".join(" ".join(self.words).split()).replace(" ,", ",")

    def find(self, tag: str, **wanted: str | None) -> list[dict[str, str | None]]:
        """The tags with these attributes; None asks for an attribute that has no value, such as checked."""
        wanted = {k.replace("_", "-"): v for k, v in wanted.items()}
        return [a for t, a in self.tags if t == tag and all(k in a and a[k] == v for k, v in wanted.items())]

    def form(self) -> list[tuple[str, str]]:
        """What the browser sends for the page's forms as they stand: ticked boxes, chosen radios, fields."""
        sent = []
        for tag, a in self.tags:
            name, kind = a.get("name"), a.get("type")
            if tag == "input" and name and (kind not in ("checkbox", "radio") or "checked" in a):
                sent.append((str(name), str(a.get("value") or "")))
        return sent


VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
REQUESTS = ("hx-get", "hx-post", "hx-put", "hx-patch", "hx-delete")
BUSY = ("hx-disabled-elt", "hx-indicator")  # what a request shows or disables while it runs


class Inherited(HTMLParser):
    """Each element that makes its own request, with the busy settings it takes from a requesting ancestor."""

    def __init__(self, source: str) -> None:
        super().__init__()
        self.open: list[tuple[str, dict[str, str | None]]] = []
        self.found: list[str] = []
        self.feed(source)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        own = dict(attrs)
        if any(r in own for r in REQUESTS):
            for name in BUSY:
                if name in own:
                    continue
                for _, above in reversed(self.open):  # the nearest ancestor that sets it decides, as in htmx
                    if name in above:
                        kept = (above.get("hx-disinherit") or "").split()
                        if "*" not in kept and name not in kept and any(r in above for r in REQUESTS):
                            self.found.append(f"<{tag} {own}> takes {name} from <{above}>")
                        break
        if tag not in VOID:
            self.open.append((tag, own))

    def handle_endtag(self, tag: str) -> None:
        while self.open and self.open.pop()[0] != tag:
            pass


class Starts:
    """The web app's long-action children, recorded instead of started; a started job is marked as taken."""

    def __init__(self, data: Path) -> None:
        self.tasks = Tasks(data)
        self.jobs: list[list[str]] = []
        self.data = data
        self.tasks.start_job = self.start_job  # type: ignore[method-assign]

    def start_job(self, argv: list[str], env: dict[str, str]) -> None:
        self.jobs.append(argv)
        path = self.data / "jobs" / argv[-1] / "job.json"
        write_json(path, {**read_json(path), "state": "interrupted"})  # a runner took it, then stopped


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The data folder, and an app folder of this test's own whose last self-test passed."""
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(tmp_path / "app"))
    write_json(tmp_path / "app" / "state.json", {"selftest": {"passed": True}})
    (tmp_path / "data").mkdir()
    return tmp_path / "data"


@pytest.fixture
def starts(home: Path) -> Starts:
    return Starts(home)


@pytest.fixture
def client(starts: Starts) -> TestClient:
    app = web_app.build(ORIGIN, "token", SESSION, lambda: None, starts.tasks)
    signed = TestClient(app, base_url=ORIGIN, raise_server_exceptions=False)
    signed.cookies.set(SESSION_COOKIE, SESSION)
    return signed


@pytest.fixture
def finished(home: Path, synthetic_job: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A finished solo job; the facade's plan for it leaves out the steps that need audio or a model."""
    job = synthetic_job.make(home.parent, SPOKEN, SEGMENTS, 6000, display_name=f"{CANARY} talk.wav")
    assert synthetic_job.finish(home.parent).state is JobState.COMPLETED
    stages = synthetic_job.plan("solo")
    monkeypatch.setattr(facade, "plan", lambda *choices: stages)
    return job


def saved_job(home: Path, name: str, **record: Any) -> Path:
    """A finished job holding the evidence saved from a real run; each person's speech is their utterances."""
    evidence = read_json(FIXTURES / f"{name}.json")
    job = store.create(home, store.JobRecord(
        job_id=SAVED, created_utc="2026-10-14T09:00:00Z", updated_utc="2026-10-14T09:00:00Z",
        app_version=__version__, mode=evidence["manifest"]["mode"], layout="mixed",
        asr="parakeet-tdt-0.6b-v2", state=JobState.COMPLETED,
        input={"display_name": "practice.m4a", "duration_ms": 400000}, **record,
    ))  # fmt: skip
    write_json(job / "evidence.json", evidence)
    speech = {
        c["id"]: [[u["start_ms"], u["end_ms"]] for u in c["utterances"]] for c in evidence["candidates"]
    }
    write_json(job / "stages" / "structure.json", {"speech": speech, "transcript": []})
    # The saved evidence holds no words: each examiner turn stands in as one word.
    turns = [{"start_ms": t["start_ms"], "end_ms": t["end_ms"]} for t in evidence["examiner_turns"]]
    write_json(job / "stages" / "words.json", {"words": turns})
    return job


def post(client: TestClient, path: str, data: dict[str, Any] | None = None) -> Any:
    return client.post(path, data=data or {}, headers=OWN_PAGE)


def card_code(reply: Any) -> str:
    found = re.search(r"Error code (E_[A-Z_]+)", reply.text)
    return found.group(1) if found else ""


def files_under(*folders: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for folder in folders if folder.exists() for p in folder.rglob("*") if p.is_file()
    }  # fmt: skip


def test_a_new_install_lands_on_the_first_run_page_until_a_self_test_passed(
    client: TestClient, home: Path
) -> None:
    state = home.parent / "app" / "state.json"
    for written in ({}, {"selftest": {"passed": False}}):
        write_json(state, written)
        reply = client.get("/", follow_redirects=False)
        assert (reply.status_code, reply.headers["location"]) == (303, "/components?first=1")
    write_json(state, {"selftest": {"passed": True}})
    page = Page(client.get("/").text)
    assert page.find("form", id="form", hx_post="/jobs")  # the form an error card's next step returns to
    assert page.find("input", type="file", data_slot="upload") and page.find(
        "input", type="file", data_slot="second"
    )
    assert [a["value"] for a in page.find("input", type="radio", name="mode")] == [
        "solo",
        "interview",
        "role-swap",
    ]
    assert "The other person agreed to be recorded and analysed on this computer." in page.text


def test_every_page_answers_in_plain_words_without_a_command_line(client: TestClient, finished: Path) -> None:
    for path in (
        "/",
        "/jobs",
        *TABS.values(),
        "/components",
        "/components?first=1",
        "/token",
        "/settings",
        "/settings?tab=advanced",
        "/doctor",
        *HELP,
    ):
        reply = client.get(path)
        assert reply.status_code == 200, (path, reply.text[:400])
        # The diagnostic report box holds the same text as the command line's report file, for a helper.
        own = re.sub(r'<textarea id="report".*?</textarea>', "", reply.text, flags=re.DOTALL)
        assert own.count("<textarea") == reply.text.count("<textarea") - (path == "/doctor"), path
        text = Page(own).text
        assert "New analysis History Components Settings Help" in text, path
        assert not COMMAND.search(text) and "Traceback" not in text, path


def test_no_request_takes_another_requests_busy_line_or_disabled_button(
    client: TestClient, finished: Path
) -> None:
    # htmx hands hx-indicator and hx-disabled-elt down to child elements: a child's own request would show
    # its parent's busy line, and a "find" selector that matches nothing logs a console error.
    for path in ("/", "/jobs", *TABS.values(), "/components", "/components?first=1", "/token", "/settings",
                 "/settings?tab=advanced", "/doctor", *HELP):  # fmt: skip
        assert Inherited(client.get(path).text).found == [], path


def test_the_busy_check_finds_a_child_request_under_a_form_with_busy_settings() -> None:
    form = '<form hx-post="/a" hx-indicator="#busy"{}><div hx-get="/b" hx-trigger="change"></div></form>'
    assert len(Inherited(form.format("")).found) == 1
    assert Inherited(form.format(' hx-disinherit="*"')).found == []
    assert Inherited(form.format(' hx-disinherit="hx-indicator"')).found == []


def test_reading_any_page_changes_no_file(client: TestClient, finished: Path, home: Path) -> None:
    app = home.parent / "app"
    with wave.open(str(finished / "input" / "audio16k.wav"), "wb") as w:  # the player's copy, read in ranges
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(3200))
    before = files_under(home, app)
    assert client.get(f"/jobs/{JOB}/audio", headers={"range": "bytes=0-99"}).status_code == 206
    for path in (
        "/jobs",
        *TABS.values(),
        f"/jobs/{JOB}/events",
        "/components",
        "/components/totals?part=whisper-tiny",
        "/settings",
        "/doctor",
        "/token",
        *HELP,
    ):
        assert client.get(path).status_code == 200, path
    assert files_under(home, app) == before


def test_the_results_show_the_audio_transcript_measures_and_the_required_lines(
    client: TestClient, finished: Path
) -> None:
    pages = {name: Page(client.get(link).text) for name, link in TABS.items()}
    for name, page in pages.items():
        for line in UI_03_LINES:  # the disclaimer and the non-affiliation line on every tab
            assert line in page.text, (name, line)
    page = pages["Transcript"]
    assert page.find("audio", id="player", src=f"/jobs/{JOB}/audio")
    # The one utterance starts at 1.0 s with the first word; its time button plays from there.
    assert page.find("button", data_seek="1000")
    assert "0:01 The candidate, answering" in page.text
    page = pages["Measures"]
    assert "Speaking time 2.5 s" in page.text  # the two speech segments: 1.3 s and 1.2 s
    assert re.search(r"Pauses of 1 s or more 1 (High|Medium|Low)", page.text)  # a count, without a unit
    assert "Mean silent pause" in page.text and "Not available" in page.text
    # Why a value is missing is a sentence of its own after the definition, so it starts with a capital,
    # unless it starts with a name spelled with a capital inside, such as openSMILE.
    assert not re.search(r"[a-z]\. [a-z]+\b", page.text)
    assert pages["Words to check"].find("audio", id="player")  # its time buttons play the recording too
    assert pages["Overview"].find("a", href=f"/jobs/{JOB}/export", **{"class": "button primary"})


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("no answer has 10 s of voiced speech", "No answer has 10 s of voiced speech"),
        ("openSMILE not enabled", "openSMILE not enabled"),
        ("eGeMAPS", "eGeMAPS"),
        ("MATTR", "MATTR"),
        ("x", "X"),
        ("", ""),
    ],
)
def test_a_sentence_gets_a_capital_first_letter_but_a_name_keeps_its_spelling(text: str, shown: str) -> None:
    assert render.upfirst(text) == shown


def profile(monkeypatch: pytest.MonkeyPatch, base: Path) -> None:
    """A made-up home folder: on Windows its two folder variables, elsewhere HOME."""
    if os.name == "nt":
        monkeypatch.setenv("USERPROFILE", str(base))
        monkeypatch.setenv("LOCALAPPDATA", str(base / "local"))
    else:
        monkeypatch.setenv("HOME", str(base))


def in_profile(*parts: str) -> str:
    """How a page should write a path under the made-up home folder, spelled out by hand."""
    return "\\".join(["%USERPROFILE%", *parts]) if os.name == "nt" else "/".join(["~", *parts])


def test_a_page_writes_the_home_part_of_a_path_as_a_variable_so_no_user_name_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile(monkeypatch, tmp_path)
    local = tmp_path / "local"
    if os.name == "nt":
        cases = [
            (local / "Fluentfolio-data", "%LOCALAPPDATA%\\Fluentfolio-data"),
            # With forward slashes, as the command path comes, and in another case: still the same folder.
            ((local / "Fluentfolio" / "f.exe").as_posix(), "%LOCALAPPDATA%\\Fluentfolio\\f.exe"),
            (str(local).upper() + "\\X", "%LOCALAPPDATA%\\X"),
            # "localx" only starts with the letters of "local"; it is not inside it.
            (tmp_path / "localx" / "f", "%USERPROFILE%\\localx\\f"),
            (tmp_path, "%USERPROFILE%"),
        ]
    else:
        cases = [(tmp_path / "fluentfolio", "~/fluentfolio"), (tmp_path, "~")]
    # The same letters and more, but a folder beside the home folder: written in full.
    outside = str(tmp_path.parent / (tmp_path.name + "x") / "f")
    for given, shown in [*cases, (outside, outside)]:
        assert render.path_text(str(given)) == shown, given


def test_each_tab_of_the_results_is_its_own_page_and_says_which_it_is(
    client: TestClient, finished: Path
) -> None:
    for name, link in TABS.items():
        page = Page(client.get(link).text)
        bar = [a for t, a in page.tags if t == "a" and a.get("data-label")]
        assert [(a["data-label"], a["href"]) for a in bar] == list(TABS.items()), name
        assert [a["data-label"] for a in bar if a.get("aria-current") == "page"] == [name]
        title = re.search(r"<title>(.*?)</title>", client.get(link).text)
        shown = f"{CANARY} talk.wav" if name == "Overview" else f"{CANARY} talk.wav - {name}"
        assert title and title.group(1) == f"{shown} - Fluentfolio", name
    # What belongs to one tab is not on the others: the measures table, the transcript.
    assert "Speaking time" not in Page(client.get(TABS["Overview"]).text).text
    assert not Page(client.get(TABS["Measures"]).text).find("button", data_seek="1000")
    for odd in ("export", "nothing", ""):  # an old or mistyped link opens the Overview
        page = Page(client.get(f"/jobs/{JOB}?tab={odd}").text)
        tabs = [a for t, a in page.tags if t == "a" and a.get("data-label")]
        assert [a["data-label"] for a in tabs if a.get("aria-current") == "page"] == ["Overview"], odd


def test_the_transcript_marks_unsure_and_other_language_words_and_shows_pauses_quietly(
    client: TestClient, finished: Path
) -> None:
    evidence = read_json(finished / "evidence.json")
    said = "[pause 1.3 s] I [?red] a book [non-English] hola amigos [withheld for audio check] then"
    evidence["candidates"][0]["utterances"][0]["text"] = said
    write_json(finished / "evidence.json", evidence)
    source = client.get(TABS["Transcript"]).text
    marked = re.findall(r"<mark>(.*?)</mark>", source)
    quiet = re.findall(r'<span class="aside">(.*?)</span>', source)
    assert marked == ["[?red]", "[non-English]"]  # whole, as the help line above the transcript names them
    assert quiet == ["[pause 1.3 s]", "[withheld for audio check]"]
    assert said in Page(source).text  # every word kept, in its order, with its spaces


def test_a_combined_value_names_its_tools_weights_and_whether_they_agree(
    client: TestClient, home: Path
) -> None:
    saved_job(home, "solo")
    text = Page(client.get(f"/jobs/{SAVED}?tab=measures").text).text
    assert "Praat 0.5, SwiftF0 0.5; tools agree" in text  # the saved run's weights, with openSMILE off
    assert "Too little speech" in text  # a value too thin to trust shows its label, not a number
    for heading in ("Whole test", "Part 1", "Part 2"):
        assert heading in text


def measure(scope: str, value: float | None, agreement: str | None = None) -> dict[str, Any]:
    missing = None if value is not None else {"code": "E_STEP_FAILED", "text": "the step failed"}
    return {
        "id": "M30", "name": "loudness variation", "unit": "dB", "scope": scope, "value": value,
        "reliability": "medium", "missing": missing, "sources": [{"tool": "praat", "weight": 1.0}],
        "agreement": agreement,
    }  # fmt: skip


def test_a_measure_with_no_number_in_any_column_has_no_reliability_word() -> None:
    blank = {"measures": [measure("all", None), measure("part1", None), measure("part2", None, "unreliable")]}
    (row,), _ = views.measures(blank, {}, {})
    assert row.cells == ["Not available", "Not available", "Unreliable"] and row.reliability == ""
    one = {"measures": [measure("all", None), measure("part1", 2.5)]}  # one number is enough for its word
    (row,), _ = views.measures(one, {}, {})
    assert row.cells == ["Not available", "2.5 dB"] and row.reliability == "medium"


def test_each_measures_table_wrapper_has_a_name_of_its_own(client: TestClient, home: Path) -> None:
    # A scrolling region named like its section would be two landmarks of one name to a screen reader.
    saved_job(home, "role_swap")
    source = client.get(f"/jobs/{SAVED}?tab=measures").text
    regions = Page(source).find("div", role="region")
    headings = re.findall(r'<h2 id="m-[a-z0-9]+">(.*?)</h2>', source)
    names = [r.get("aria-label") for r in regions]
    assert len(regions) == len(headings) == 2 and not any(r.get("aria-labelledby") for r in regions)
    assert len(set(names)) == 2 and all(n and n not in headings for n in names)


def test_the_measures_tab_writes_not_available_where_a_reliability_word_would_be(
    client: TestClient, home: Path
) -> None:
    job = saved_job(home, "solo")
    evidence = read_json(job / "evidence.json")
    first = evidence["candidates"][0]["measures"][0]
    for m in evidence["candidates"][0]["measures"]:
        if m["id"] == first["id"]:
            m.update(value=None, missing={"code": "E_STEP_FAILED", "text": "the step failed"})
    write_json(job / "evidence.json", evidence)
    source = client.get(f"/jobs/{SAVED}?tab=measures").text
    row = re.search(rf'<th scope="row">{re.escape(first["name"].capitalize())}</th>(.*?)</tr>', source, re.S)
    assert row and "Not available</td>" in row.group(1) and 'class="rel ' not in row.group(1)


def test_learner_text_names_no_list_id_and_calls_a_job_an_analysis(client: TestClient) -> None:
    definition = "different linking words / all their uses (list X03)"  # the AI chat's text names its lists
    (row,), _ = views.measures({"measures": [measure("all", 2.5)]}, {"M30": definition}, {})
    assert row.how == "different linking words / all their uses"
    stopped = catalogue_of_errors()["E_JOB_CANCELLED"]
    assert "job" not in f"{stopped.title} {stopped.message}".lower()
    missing = client.get("/jobs/20990101-000000-dead").text
    assert "an analysis may have been deleted" in missing and " job " not in Page(missing).text


def pointer(pid: str, name: str, total: int, note: str | None) -> dict[str, Any]:
    items = [{"utterance_id": "u1", "first": 0, "last": 0, "text": "well"}] * total
    return {"id": pid, "name": name, "items": items, "shown": total, "total": total, "note": note}


def test_a_pointer_list_says_not_checked_or_what_was_looked_for_and_never_a_bare_zero() -> None:
    off = "Not checked: the pronunciation check is off in Settings."
    rep, words, cover, rare = views.pointers({"pointers": [
        pointer("X01", "repetition candidates", 0, None),
        pointer("X06", "words to check (pronunciation)", 0, "Not checked."),
        pointer("X07", "what the pronunciation check covered", 0, off),
        pointer("X04", "rare words used", 2, "Content words outside the list."),
    ]})  # fmt: skip
    none = "None found in these answers. This tool can miss problems."
    assert (rep.summary, rep.note, rep.empty) == ("Repetition candidates: 0", "", none)
    assert words.summary == "Words to check (pronunciation): not checked" and (words.note, words.empty) == (
        off,
        "",
    )
    assert (cover.summary, cover.note, cover.empty) == ("What the pronunciation check covered", off, "")
    assert (rare.summary, len(rare.items), rare.empty) == ("Rare words used: 2", 2, "")
    # A check that ran and listed none says so once, not twice.
    ran = "None flagged. Patterns: no repeated pattern."
    (zero,) = views.pointers({"pointers": [pointer("X06", "words to check (pronunciation)", 0, ran)]})
    assert (zero.summary, zero.note, zero.empty) == (
        "Words to check (pronunciation): 0",
        "Patterns: no repeated pattern.",
        none,
    )


def test_the_words_tab_shows_a_check_that_did_not_run_as_not_checked(client: TestClient, home: Path) -> None:
    saved_job(home, "solo")  # its pronunciation check was off, and it found no repetition
    text = Page(client.get(f"/jobs/{SAVED}?tab=words").text).text
    assert "Words to check (pronunciation): not checked" in text
    assert "Words to check (pronunciation): 0" not in text and "None found." not in text
    assert "Repetition candidates: 0" in text and "This tool can miss problems." in text


def test_role_swap_results_say_who_asks_and_who_answers_in_words(client: TestClient, home: Path) -> None:
    job = saved_job(home, "role_swap", settings={"review": {"policy": "always"}})  # chosen for this run
    evidence = read_json(job / "evidence.json")
    page = Page(client.get(f"/jobs/{SAVED}?tab=transcript").text)
    question = next(t for t in evidence["examiner_turns"] if t["kind"] == "question")
    asker = "Person 2" if question["addressed_to"] == "person1" else "Person 1"
    assert f"{asker}, asking" in page.text
    assert "Person 1, answering" in page.text and "Person 2, answering" in page.text
    assert {a["class"] for a in page.find("span") if "who" in str(a.get("class"))} == {
        "who spk-1",
        "who spk-2",
    }
    text = Page(client.get(f"/jobs/{SAVED}?tab=measures").text).text
    assert "Measures: Person 1" in text and "Measures: Person 2" in text
    page = Page(client.get(f"/jobs/{SAVED}").text)  # the review form is on the Overview
    swap = evidence["candidates"][0]["roles"]["swap_times_ms"][0]
    assert page.find("input", name="swap_times", value=f"{swap // 60000}:{swap // 1000 % 60:02d}")
    # Voice S1 is Person 2's, 115,723 ms of speech; S2 is Person 1's, 96,593 ms.
    assert "Voice S1 (Person 2, 1 min 55 s of speech) is someone else" in page.text
    assert "Voice S2 (Person 1, 1 min 36 s of speech) is someone else" in page.text
    assert not page.find("input", name="one_voice")  # two voices were found


def test_the_review_form_sends_only_what_the_learner_changed(
    client: TestClient, home: Path, starts: Starts
) -> None:
    job = saved_job(home, "role_swap", settings={"review": {"policy": "always"}})
    shown = Page(client.get(f"/jobs/{SAVED}").text).form()
    assert ("swap_times", "3:07") in shown and ("person1.part3", "2:38") in shown
    for changes, saved in (
        ({}, {}),  # unchanged: the times found are kept to the millisecond
        ({"swap_times": "3:10"}, {"swap_times_ms": [190000]}),
        ({"swap_times": ""}, {"swap_times_ms": []}),  # the roles never change
        ({"other": "S1"}, {"other_speakers": ["S1"]}),
        ({"person2.part3": "5:50"}, {"parts": {"person2": {"part3_start_ms": 350000}}}),
    ):
        path = job / "job.json"
        write_json(path, {**read_json(path), "state": "completed"})
        # Each change on its own, to the page as shown: an analysis with nothing saved before it.
        (job / "overrides.json").unlink(missing_ok=True)
        sent = [(k, v) for k, v in shown if k not in changes] + list(changes.items())
        reply = client.post(f"/jobs/{SAVED}/review", content=urlencode(sent),
                            headers={**OWN_PAGE, "content-type": FORM})  # fmt: skip
        assert reply.headers["hx-redirect"] == f"/jobs/{SAVED}", changes
        assert read_json(job / "overrides.json") == saved, changes


def test_the_one_voice_question_is_asked_only_when_one_voice_was_found(
    client: TestClient, home: Path, starts: Starts
) -> None:
    job = saved_job(home, "role_swap", settings={"review": {"policy": "when-unsure"}})
    with_warning(job, "W_ONE_VOICE")
    page = Page(client.get(f"/jobs/{SAVED}").text)
    assert page.find("details", id="review", open=None)  # unsure: the form is shown, open
    assert page.find("input", name="one_voice", value="candidate")
    sent = [*page.form(), ("one_voice", "candidate")]  # the whole form, as the browser sends it
    client.post(f"/jobs/{SAVED}/review", content=urlencode(sent), headers={**OWN_PAGE, "content-type": FORM})
    assert read_json(job / "overrides.json") == {"one_voice": "candidate"}


def finished_again(job: Path, structure: dict[str, Any] | None = None) -> None:
    """The job as the analysis started by its last review save leaves it: finished, with the structure step's
    result, which the save deleted, written again."""
    path = job / "job.json"
    write_json(path, {**read_json(path), "state": "completed"})
    if structure is not None:
        write_json(job / "stages" / "structure.json", structure)


def send_review(client: TestClient, job: Path, sent: list[tuple[str, str]]) -> Any:
    """Post the review form of a finished job; returns the corrections it saved."""
    finished_again(job)
    reply = client.post(f"/jobs/{SAVED}/review", content=urlencode(sent),
                        headers={**OWN_PAGE, "content-type": FORM})  # fmt: skip
    assert reply.headers["hx-redirect"] == f"/jobs/{SAVED}", reply.text
    return read_json(job / "overrides.json")


def test_saving_the_review_form_again_keeps_the_corrections_saved_before(
    client: TestClient, home: Path, starts: Starts
) -> None:
    job = saved_job(home, "role_swap", settings={"review": {"policy": "always"}})
    saved = {
        "swap_people": True, "swap_times_ms": [190000], "other_speakers": ["S1"],
        "parts": {"person2": {"part3_start_ms": 350000}},
    }  # fmt: skip
    # The page shows the analysis these corrections made: the swap time and Person 2's Part 3 as saved.
    evidence = read_json(job / "evidence.json")
    for c in evidence["candidates"]:
        c["roles"]["swap_times_ms"] = [190000]
        if c["id"] == "person2":
            next(p for p in c["parts"] if p["part"] == 3)["start_ms"] = 350000
    write_json(job / "evidence.json", evidence)
    without = {k: v for k, v in saved.items() if k not in ("swap_people", "other_speakers")}
    structure = read_json(job / "stages" / "structure.json")
    for changes, dropped, expected in (
        ({}, (), saved),  # saved again as it stands: nothing is lost
        (
            {"person1.part3": "2:40"},
            (),
            {**saved, "parts": {**saved["parts"], "person1": {"part3_start_ms": 160000}}},
        ),
        ({"swap_people": "yes"}, ("other",), without),  # the people turned back, the voice put back
    ):
        write_json(job / "overrides.json", saved)
        finished_again(job, structure)
        shown = Page(client.get(f"/jobs/{SAVED}").text).form()
        assert {("swap_times", "3:10"), ("person2.part3", "5:50"), ("other", "S1")} <= set(shown)
        sent = [(k, v) for k, v in shown if k not in changes and k not in dropped] + list(changes.items())
        assert send_review(client, job, sent) == expected, changes


TRACK_WORDS = "my hometown is a small place near the coast with a busy harbour and narrow old streets".split()


def per_track(job: Path, on_track: dict[int, tuple[int, list[str]]]) -> None:
    """The saved job as if recorded with one track per person, with these words from these times on each."""
    write_json(job / "job.json", {**read_json(job / "job.json"), "layout": "per-track"})
    evidence = read_json(job / "evidence.json")
    evidence["manifest"]["layout"] = "per-track"
    write_json(job / "evidence.json", evidence)
    words = [
        {"raw": word, "track": n, "speaker": f"S{n}", "start_ms": at + 400 * k, "end_ms": at + 400 * k + 300}
        for n, (at, said) in on_track.items() for k, word in enumerate(said)
    ]  # fmt: skip
    write_json(job / "stages" / "words.json", {"words": sorted(words, key=lambda w: w["start_ms"])})


def test_with_a_track_per_person_the_learner_says_whose_each_track_is_from_a_preview_and_its_first_words(
    client: TestClient, home: Path, starts: Starts
) -> None:
    job = saved_job(home, "role_swap")  # the review form only if unsure, and this run was sure
    assert not Page(client.get(f"/jobs/{SAVED}").text).find("details", id="review")
    per_track(job, {1: (2000, TRACK_WORDS), 2: (9000, ["yes", "it", "is"])})
    page = Page(client.get(f"/jobs/{SAVED}").text)
    assert page.find("details", id="review", open=None) and page.find("audio", id="player")
    # Track 1 is voice S1, which the saved run gave to Person 2; the form shows its first 12 words and a mark
    # that more follow.
    assert f"Track 1: Person 2 Play 5 s from 0:02 {' '.join(TRACK_WORDS[:12])}... Track 2:" in page.text
    assert "Track 2: Person 1 Play 5 s from 0:09 yes it is" in page.text
    assert page.find("button", data_seek="2000", data_length="5000")
    assert page.find("button", data_seek="9000", data_length="5000")
    assert page.find("input", name="tracks", value="as-shown", checked=None)
    assert "Track 1 is Person 2 and track 2 is Person 1" in page.text
    assert "Track 1 is Person 1 and track 2 is Person 2" in page.text
    assert "Person 1 is the one who answered first." in page.text
    # The voices are the tracks here, and named so: S1 spoke 115,723 ms in the saved run.
    assert "Track 1 (Person 2, 1 min 55 s of speech) is someone else" in page.text
    swapped = [(k, v) for k, v in page.form() if k != "tracks"] + [("tracks", "swap")]
    assert send_review(client, job, swapped) == {"swap_people": True}
    # The page then shows the people swapped; saving it as it stands keeps that, and swapping again undoes it.
    assert send_review(client, job, page.form()) == {"swap_people": True}
    assert send_review(client, job, swapped) == {}


def test_whose_track_is_whose_follows_the_people_found_and_the_voices_set_aside() -> None:
    words = [{"raw": word, "track": 2, "start_ms": 900 + 400 * k} for k, word in enumerate(TRACK_WORDS)]
    evidence: dict[str, Any] = {
        "manifest": {"layout": "two-files", "mode": "interview"},
        "candidates": [{"speaker": "S2", "label": "the candidate"}],
    }
    one, two = views.tracks(evidence, words, {})
    assert (one.number, one.who, one.person, one.start_ms, one.words) == (1, "the examiner", True, None, "")
    assert (two.number, two.who, two.person, two.start_ms) == (2, "the candidate", True, 900)
    assert two.words == " ".join(TRACK_WORDS[:12]) + "..."  # 17 words: the first 12 and a mark for the rest
    shorter = views.tracks(evidence, words[:12], {})[1]
    assert shorter.words == " ".join(TRACK_WORDS[:12])  # all of them shown: nothing left out, no mark
    one, _ = views.tracks(evidence, words, {"other_speakers": ["S1"]})
    assert (one.who, one.person) == ("someone else", False)  # set aside in the review form: no one to swap
    evidence["manifest"]["mode"] = "role-swap"
    assert views.tracks(evidence, words, {})[0].who == "neither person"
    assert views.tracks({**evidence, "candidates": []}, words, {}) == []  # no people found: none to map
    evidence["manifest"]["layout"] = "mixed"
    assert views.tracks(evidence, words, {}) == []


def test_the_speech_models_own_text_reaches_the_page_without_hidden_characters() -> None:
    # Direction, zero-width and tag characters: the evidence text is cleaned of them, so the raw text is too.
    hidden = "".join(map(chr, (0x202E, 0x200B, 0xE0041)))
    evidence: dict[str, Any] = {
        "manifest": {"layout": "two-files", "mode": "interview"},
        "candidates": [{"speaker": "S1", "label": "the candidate"}],
    }
    track = views.tracks(evidence, [{"raw": f"yes{hidden}", "track": 1, "start_ms": 0}], {})[0]
    assert track.words == "yes"
    run = {"start_ms": 0, "speaker": "S1", "text": f'I{hidden} said <b hx-get="/x">'}
    line = views.transcript({"candidates": [], "examiner_turns": []}, [run])[0]
    assert line.text == 'I said <b hx-get="/x">'  # the template escapes the markup itself


def test_a_voice_that_is_not_the_candidate_is_the_examiner_only_where_that_is_known() -> None:
    evidence: dict[str, Any] = {
        "manifest": {"layout": "mixed", "mode": "interview"},
        "candidates": [{"speaker": "S1", "label": "the candidate"}],
        "speakers": [{"id": "S1", "speech_ms": 90_000}, {"id": "S2", "speech_ms": 30_000}],
    }
    # Mixed audio: the second voice may be the examiner or a third voice; the evidence does not say which.
    assert [text for _, text in views.voices(evidence, {})] == [
        "Voice S1 (the candidate, 1 min 30 s of speech)", "Voice S2 (not the candidate, 30.0 s of speech)",
    ]  # fmt: skip
    evidence["manifest"]["layout"] = "per-track"  # both tracks are the two people
    assert views.voices(evidence, {})[1][1] == "Track 2 (the examiner, 30.0 s of speech)"
    set_aside = views.voices(evidence, {"other_speakers": ["S2"]})[1][1]
    assert set_aside == "Track 2 (not the candidate, 30.0 s of speech)"
    evidence["manifest"].update(layout="mixed", mode="role-swap")
    assert views.voices(evidence, {})[1][1] == "Voice S2 (neither person, 30.0 s of speech)"


def test_no_mode_is_chosen_before_the_first_analysis_then_the_last_one_is(
    client: TestClient, home: Path, starts: Starts
) -> None:
    page = Page(client.get("/").text)
    assert not page.find("input", name="mode", checked=None)
    assert [a["value"] for a in page.find("input", name="layout", checked=None)] == ["mixed"]
    upload_id = "0" * 32
    recording(home / "tmp" / f"{upload_id}.upload", 12.0)
    reply = post(client, "/jobs", {"upload": upload_id, "upload_name": "talk.wav", "consent": "yes"})
    assert card_code(reply) == "E_NO_MODE" and "Choose who speaks in the recording" in reply.text
    assert store.list_ids(home) == [] and starts.jobs == []
    job = saved_job(home, "role_swap")  # an analysis of two files
    write_json(job / "job.json", {**read_json(job / "job.json"), "layout": "two-files"})
    page = Page(client.get("/").text)
    assert [a["value"] for a in page.find("input", name="mode", checked=None)] == ["role-swap"]
    assert [a["value"] for a in page.find("input", name="layout", checked=None)] == ["two-files"]
    for later, mode, selftest in (
        ("20261014-100000-cafe", "interview", False),
        ("20261014-110000-f00d", "solo", True),
    ):
        store.create(home, store.JobRecord(
            job_id=later, created_utc="2026-10-14T10:00:00Z", updated_utc="2026-10-14T10:00:00Z",
            app_version=__version__, mode=mode, layout="mixed", asr="parakeet-tdt-0.6b-v2",
            state=JobState.COMPLETED, input={"display_name": "later.m4a", "duration_ms": 1000},
            selftest=selftest,
        ))  # fmt: skip
    # The newest analysis is the learner's interview; a self-test is not one of theirs.
    page = Page(client.get("/").text)
    assert [a["value"] for a in page.find("input", name="mode", checked=None)] == ["interview"]
    assert [a["value"] for a in page.find("input", name="layout", checked=None)] == ["mixed"]


def test_each_mode_shows_a_sentence_and_a_small_picture_of_who_speaks_when(
    client: TestClient, home: Path
) -> None:
    source = client.get("/").text
    bars: dict[str, list[tuple[float, str, float, float]]] = {}
    for mode in views.MODES:
        option = Page(source.split(f'name="mode" value="{mode}"', 1)[1].split("</label>", 1)[0])
        assert option.find("span", id=f"mode-{mode}") and option.text  # its sentence
        assert option.find("svg", aria_hidden="true")  # the sentence says in words what the picture shows
        found = sorted((float(str(r["x"])), str(r["class"]), float(str(r["y"])), float(str(r["width"])))
                       for r in option.find("rect"))  # fmt: skip
        bars[mode] = found

    def lanes(mode: str) -> dict[str, set[float]]:
        return {kind: {y for _, k, y, _ in bars[mode] if k == kind} for _, kind, _, _ in bars[mode]}

    def widths(mode: str, kind: str) -> list[float]:
        return [w for _, k, _, w in bars[mode] if k == kind]

    # Solo: one person, who reads each question aloud and then answers.
    assert set(lanes("solo")) == {"read", "spk-1"} and len(set().union(*lanes("solo").values())) == 1
    assert bars["solo"][0][1] == "read"
    # Interview: the examiner asks short questions on their own lane; the learner answers at length.
    asks, answers = lanes("interview")["spk-2"], lanes("interview")["spk-1"]
    assert set(lanes("interview")) == {"spk-1", "spk-2"} and not asks & answers
    assert bars["interview"][0][1] == "spk-2"
    assert max(widths("interview", "spk-2")) < min(widths("interview", "spk-1"))
    # Role swap: each person both asks and answers; the first to answer asks later.
    for kind in ("spk-1", "spk-2"):
        short, long = min(widths("role-swap", kind)), max(widths("role-swap", kind))
        assert short < long / 2, kind
    order = [(k, w) for _, k, _, w in bars["role-swap"]]
    assert order[0][0] == "spk-2" and order[1][0] == "spk-1" and order[1][1] > order[0][1]


def with_warning(job: Path, code: str) -> None:
    """The evidence of a run that gave this one warning, worded as the evidence step words it."""
    evidence = read_json(job / "evidence.json")
    warning = {"code": code, "text": catalogue_of_errors()[code].message}
    write_json(
        job / "evidence.json", {**evidence, "manifest": {**evidence["manifest"], "warnings": [warning]}}
    )


def test_a_second_voice_in_solo_offers_the_interview_run_with_its_estimate_and_the_agreement(
    client: TestClient, home: Path, starts: Starts
) -> None:
    job = saved_job(home, "solo")
    (job / "input").mkdir(exist_ok=True)
    (job / "input" / "audio16k.wav").write_bytes(b"RIFF")  # the new job gets its own copy
    with_warning(job, "W_SECOND_VOICE_SOLO")
    page = Page(client.get(f"/jobs/{SAVED}").text)
    assert "Another voice speaks for more than 5 percent of this solo recording." in page.text
    assert "If someone asked you the questions, analyse the recording in interview mode." in page.text
    offer = r'Runs again from "Telling the voices apart", about \d+ min on the (CPU|GPU)\. The words are not'
    assert re.search(offer, page.text) and "Analyse again in interview mode" in page.text
    assert page.find("input", type="hidden", name="mode", value="interview")
    assert page.find("input", name="consent", value="yes")  # a solo job has no agreement on record
    reply = post(client, f"/jobs/{SAVED}/rerun", {"mode": "interview"})
    assert card_code(reply) == "E_CONSENT_REQUIRED" and 'href="#form"' in reply.text
    assert store.list_ids(home) == [SAVED] and starts.jobs == []
    reply = post(client, f"/jobs/{SAVED}/rerun", {"mode": "interview", "consent": "yes"})
    (new,) = [i for i in store.list_ids(home) if i != SAVED]
    assert reply.headers["hx-redirect"] == f"/jobs/{new}" and starts.jobs[0][-1] == new
    record = read_json(home / "jobs" / new / "job.json")
    assert (record["mode"], record["rerun_of"], record["consent"]["source"]) == ("interview", SAVED, "ui")
    assert read_json(job / "job.json")["mode"] == "solo"  # the old job stays as it was


def test_only_a_mode_mismatch_offers_another_mode_and_solo_needs_no_agreement(
    client: TestClient, home: Path
) -> None:
    job = saved_job(home, "role_swap", consent={"utc": "2026-10-14T09:00:00Z", "source": "ui"})
    with_warning(job, "W_THIRD_VOICE")  # corrected in the review form, not by another mode
    page = Page(client.get(f"/jobs/{SAVED}").text)
    assert "Three voices each speak for a noticeable time." in page.text and not page.find("form", id="form")
    with_warning(job, "W_ONE_VOICE")
    page = Page(client.get(f"/jobs/{SAVED}").text)
    assert page.find("input", name="mode", value="solo") and not page.find("input", name="consent")
    assert "Analyse again in solo mode" in page.text
    evidence = read_json(job / "evidence.json")
    write_json(job / "evidence.json", {**evidence, "manifest": {**evidence["manifest"], "warnings": []}})
    assert "Check these results" not in Page(client.get(f"/jobs/{SAVED}").text).text  # no warning, no section


def test_text_from_a_recording_is_shown_as_inert_text(client: TestClient, finished: Path) -> None:
    payload = '<script>alert(1)</script><img src=x onerror="alert(2)"><b hx-get="/x">'
    evidence = read_json(finished / "evidence.json")
    evidence["candidates"][0]["utterances"][0]["text"] = f"I said {payload}"
    write_json(finished / "evidence.json", evidence)
    for link in TABS.values():  # the transcript shows the words; every other tab must stay inert too
        reply = client.get(link)
        assert payload not in reply.text and "<script>alert" not in reply.text, link
        assert not [a for _, a in Page(reply.text).tags if any(k.startswith("on") for k in a)], link
        assert not [a for _, a in Page(reply.text).tags if a.get("hx-get") == "/x"], link
    assert html.escape("<script>alert(1)</script>", quote=False) in client.get(TABS["Transcript"]).text


def test_the_player_gets_the_normalised_copy_with_range_requests(client: TestClient, finished: Path) -> None:
    wav = finished / "input" / "audio16k.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(range(256)) * 125)
    whole = wav.read_bytes()
    reply = client.get(f"/jobs/{JOB}/audio")
    assert (reply.status_code, reply.content, reply.headers["content-type"]) == (200, whole, "audio/wav")
    part = client.get(f"/jobs/{JOB}/audio", headers={"range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == whole[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{len(whole)}"
    wav.unlink()
    assert client.get(f"/jobs/{JOB}/audio").status_code == 404


def test_only_the_export_files_can_be_downloaded(client: TestClient, finished: Path) -> None:
    text = (finished / "export" / "for_ai" / "for_ai.txt").read_bytes()
    reply = client.get(f"/jobs/{JOB}/files/for_ai.txt")
    assert reply.status_code == 200 and reply.content == text
    assert reply.headers["content-disposition"].startswith("attachment")
    for name in ("job.json", "evidence.json", "..%2Fjob.json", "for_ai_person1.txt"):
        assert client.get(f"/jobs/{JOB}/files/{name}").status_code == 404, name


def test_each_view_has_one_primary_button_and_each_label_says_what_it_does(
    client: TestClient, home: Path
) -> None:
    job = saved_job(home, "role_swap")
    (job / "export" / "for_ai").mkdir(parents=True, exist_ok=True)
    for person in ("person1", "person2"):  # the texts made from the same saved evidence
        made = (FIXTURES.parent.parent / "golden" / f"for_ai_role_swap_{person}.txt").read_bytes()
        (job / "export" / "for_ai" / f"for_ai_{person}.txt").write_bytes(made)
    page = Page(client.get(f"/jobs/{SAVED}/export").text)
    copies = [a for a in page.find("button") if str(a.get("data-copy", "")).startswith("text-")]
    assert [a["data-copy"] for a in copies] == ["text-person1", "text-person2"]
    assert [a["class"] for a in copies] == ["primary", ""]  # the second person's copy button is secondary
    # The Overview link opens the Copy for AI tab; it copies nothing, so it does not say "Copy".
    source = client.get(f"/jobs/{SAVED}").text
    assert f'<a class="button primary" href="/jobs/{SAVED}/export">Open Copy for AI</a>' in source


def test_copy_for_ai_holds_the_file_the_prompt_and_the_warnings(client: TestClient, finished: Path) -> None:
    text = (finished / "export" / "for_ai" / "for_ai.txt").read_text(encoding="utf-8")
    reply = client.get(f"/jobs/{JOB}/export")
    boxes = dict(re.findall(r'<textarea id="([a-z0-9-]+)"[^>]*>(.*?)</textarea>', reply.text, re.S))
    assert html.unescape(boxes["text-candidate"]) == text  # exactly what Copy for AI copies
    # Copy prompt only: the prompt as this file holds it, its export ID and boundary filled in.
    prompt = html.unescape(boxes["prompt-candidate"])
    assert prompt.startswith("=== FLUENTFOLIO GRADING PROMPT") and text.startswith(prompt)
    assert text[len(prompt) :].split("\n", 1)[0].endswith(" START OF DATA")
    assert "{EXPORT_ID}" not in reply.text and "{BOUNDARY}" not in reply.text
    first, second = (html.unescape(boxes[f"part-{n}-candidate"]) for n in (1, 2))
    assert first.split("\n")[:3] == text.split("\n")[:3] and second.startswith("PART 2 OF 2")
    page = Page(reply.text)
    assert page.find("button", data_copy="text-candidate")
    for line in (
        *UI_03_LINES,
        "If the chat refuses or cuts the paste, upload for_ai.txt as a file, or use Copy in 2 parts.",
        "Paste into a normal chat. If you use an AI agent that can run commands, treat the transcript as "
        "untrusted text.",
    ):
        assert line in page.text, line
    assert "partner's words" not in page.text  # a solo recording has no partner
    assert "partner's voice" in page.text  # the audio step's warning comes before the button
    assert "Your own recording holds" not in page.text
    # The answer check is on the command line only: the page has no field or form for an answer.
    assert not page.find("textarea", name="answer") and "check-answer" not in reply.text


def test_a_two_person_copy_for_ai_warns_of_the_partner_in_the_text_and_in_the_learners_own_recording(
    client: TestClient, home: Path
) -> None:
    saved_job(home, "role_swap")
    page = Page(client.get(f"/jobs/{SAVED}/export").text)
    for line in (
        "This text contains your partner's words; it goes to that AI company.",
        "Your own recording holds your partner's voice all the way through. Upload it to an AI chat only if "
        "your partner agrees.",
    ):
        assert line in page.text, line


def test_each_made_audio_file_of_a_two_person_job_has_the_partner_warning_above_its_download(
    client: TestClient, home: Path
) -> None:
    # Once both files exist, the "Make the audio file" section and its warning are gone: each link needs it.
    job = saved_job(home, "role_swap")
    made = {p: {"file": f"for_ai_audio_{p}.m4a"} for p in ("person1", "person2")}
    write_json(job / "stages" / "ai_audio.json", {"status": "ok", "candidates": made})
    text = Page(client.get(f"/jobs/{SAVED}/export").text).text
    warning = (
        "Your partner's voice can still be heard where the speakers were mixed up or you both spoke at once."
    )
    assert "Make the audio file" not in text and text.count(warning) == 2
    first, second = (text.index(f"Download for_ai_audio_{p}.m4a") for p in ("person1", "person2"))
    assert text.index(warning) < first < text.index(warning, first) < second


def test_the_job_page_says_that_deleting_skips_the_recycle_bin_and_is_not_secure_erasure(
    client: TestClient, finished: Path
) -> None:
    line = "Deleting skips the Recycle Bin. On SSDs and in synced folders it is not secure erasure."
    assert line in Page(client.get(f"/jobs/{JOB}").text).text


def test_copy_in_2_parts_stays_closed_and_the_page_states_no_size_limit(
    client: TestClient, finished: Path
) -> None:
    # The text has no size limit, so the page never opens the two parts for a long text or calls it too long.
    html = client.get(f"/jobs/{JOB}/export").text
    assert "<details>\n    <summary>Copy in 2 parts" in html and "<details open>" not in html
    assert "longer than many chats" not in html


def test_a_two_person_start_without_the_agreement_is_refused_and_leaves_no_job(
    client: TestClient, home: Path, starts: Starts, tmp_path: Path
) -> None:
    upload_id = "0" * 32
    recording(home / "tmp" / f"{upload_id}.upload", 12.0)
    reply = post(client, "/jobs", {"mode": "interview", "upload": upload_id, "upload_name": "talk.wav"})
    assert card_code(reply) == "E_CONSENT_REQUIRED" and reply.status_code == 200  # put in place, by the form
    assert 'href="#form"' in reply.text and store.list_ids(home) == [] and starts.jobs == []
    (home / "settings.toml").write_text('[review]\npolicy = "never"\n', encoding="utf-8")
    assert Page(client.get("/").text).find(
        "input", name="review", value="never", checked=None
    )  # as in Settings
    reply = post(client, "/jobs", {
        "mode": "interview", "upload": upload_id, "upload_name": "talk.wav", "consent": "yes",
        "review": "always",
    })  # fmt: skip
    (job_id,) = store.list_ids(home)
    assert reply.headers["hx-redirect"] == f"/jobs/{job_id}" and starts.jobs[0][-2:] == ["_run", job_id]
    record = read_json(home / "jobs" / job_id / "job.json")
    assert set(record["consent"]) == {"utc", "source"} and record["consent"]["source"] == "ui"
    assert record["settings"]["review"]["policy"] == "always"  # this run's choice, in its settings copy
    assert record["input"]["display_name"] == "talk.wav"
    assert not (home / "tmp" / f"{upload_id}.upload").exists()  # the job has its own copy now


def recording(path: Path, seconds: float) -> Path:
    """A 220 Hz tone, loud enough not to count as silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = int(16000 * seconds)
        w.writeframes(b"".join(int(8000 if (i // 36) % 2 else -8000).to_bytes(2, "little", signed=True)
                               for i in range(frames)))  # fmt: skip
    return path


def test_a_start_without_its_upload_asks_for_the_recording_again(client: TestClient, home: Path) -> None:
    for sent in ({"mode": "solo", "upload": "f" * 32}, {"mode": "solo", "upload": "../settings"}):
        reply = post(client, "/jobs", sent)
        assert card_code(reply) == "E_UPLOAD_GONE" and "Choose the recording again" in reply.text
    assert card_code(post(client, "/jobs", {"mode": "duet", "upload": "f" * 32})) == "E_INPUT_OPTION"


def test_a_start_with_no_recording_chosen_asks_for_one_first(client: TestClient, home: Path) -> None:
    two = {"mode": "interview", "layout": "two-files", "upload": "f" * 32}
    for sent in ({"mode": "solo"}, {"mode": "solo", "upload": ""}, two):
        reply = post(client, "/jobs", sent)
        assert card_code(reply) == "E_NO_RECORDING" and "Choose a recording first" in reply.text, sent
        assert "no longer waiting" not in reply.text
        assert ">Choose a recording<" in reply.text and "Choose another recording" not in reply.text


def test_review_corrections_that_do_not_fit_are_refused(
    client: TestClient, finished: Path, starts: Starts
) -> None:
    path = finished / "job.json"
    record = read_json(path)
    write_json(
        path, {**record, "input": {**record["input"], "duration_ms": 6000}}
    )  # as an analysis records it
    for times in (("0:02", "0:0x", ""), ("0:02", "0:04", "0:03"), ("0:02", "9:00", "")):
        start, end, third = times
        reply = post(client, f"/jobs/{JOB}/review", {
            "candidate.part2_start": start, "candidate.part2_end": end, "candidate.part3": third,
        })  # fmt: skip
        assert card_code(reply) == "E_REVIEW_INVALID", times
    assert not (finished / "overrides.json").exists() and starts.jobs == []
    reply = post(
        client, f"/jobs/{JOB}/review", {"candidate.part2_start": "0:02", "candidate.part2_end": "0:05"}
    )
    assert reply.headers["hx-redirect"] == f"/jobs/{JOB}"
    assert read_json(finished / "overrides.json") == {"parts": {"candidate": {"part2": [2000, 5000]}}}
    assert not (finished / "stages" / "structure.json").exists() and starts.jobs[0][-1] == JOB


def test_cancel_asks_only_a_running_job_to_stop(client: TestClient, finished: Path, home: Path) -> None:
    assert post(client, f"/jobs/{JOB}/cancel").text == "This analysis is not running."
    path = finished / "job.json"
    write_json(path, {**read_json(path), "state": "running"})
    with work_lock(home):  # the job's runner holds it while it runs
        assert post(client, f"/jobs/{JOB}/cancel").text == "Stopping. This takes a few seconds."
    assert (finished / store.CANCEL_FILE).is_file()


def test_progress_streams_the_steps_until_the_job_ends(client: TestClient, finished: Path) -> None:
    path = finished / "job.json"
    write_json(path, {**read_json(path), "state": "queued"})
    page = Page(client.get(f"/jobs/{JOB}").text)
    assert page.find("section", id="progress", data_events=f"/jobs/{JOB}/events")
    assert "Your computer will stay awake until the analysis ends. Keep the lid open." in page.text
    assert "Run again then goes on from the first step that did not finish." in page.text
    threading.Timer(1.5, lambda: write_json(path, {**read_json(path), "state": "completed"})).start()
    started = time.monotonic()
    with client.stream("GET", f"/jobs/{JOB}/events") as stream:
        body = "".join(stream.iter_text())
    events = re.findall(r"^event: (\w+)$", body, re.M)
    assert events.count("progress") >= 2 and events[-1] == "done" and time.monotonic() - started < 10
    nows = re.findall(r"event: now\ndata: (.*)\n", body)
    assert nows == ["Waiting to start.", "The analysis has ended."]
    assert "Writing the evidence" in body and '<span class="word">Done</span>' in body


def test_delete_one_job_removes_its_folder(client: TestClient, finished: Path, home: Path) -> None:
    reply = post(client, f"/jobs/{JOB}/delete")
    assert reply.headers["hx-redirect"] == "/jobs" and not finished.exists()
    assert client.get(f"/jobs/{JOB}").status_code == 404


def test_delete_all_my_data_leaves_no_job_or_its_words(
    client: TestClient, finished: Path, home: Path
) -> None:
    app = home.parent / "app"
    for leftover in (home / "tmp" / "x.upload", app / "tmp" / JOB / "chunk.wav", home / "logs" / "app.log"):
        leftover.parent.mkdir(parents=True, exist_ok=True)
        leftover.write_text(f"{CANARY} {JOB}", encoding="utf-8")
    (home / "open.html").write_text("link", encoding="utf-8")
    write_json(home / "consent.json", {"opensmile": {"utc": "2026-10-14T09:00:00Z", "source": "ui"}})
    (home / "settings.toml").write_text('[ui]\ntheme = "dark"\n', encoding="utf-8")
    page = Page(client.get("/jobs").text)
    assert "Delete all 1 analysis and the logs" in page.text and "not secure erasure" in page.text
    assert post(client, "/data/delete-all").headers["hx-redirect"] == "/jobs"
    left = [p for p in [*home.rglob("*"), *(app / "tmp").rglob("*")] if p.is_file()]
    assert {p.name for p in left} == {"settings.toml", "work.lock"}  # the settings stay
    for p in left:
        assert CANARY not in p.read_text(encoding="utf-8") and JOB not in p.read_text(encoding="utf-8")


def test_the_self_test_result_reads_as_plain_sentences_with_titles_not_ids() -> None:
    missing = catalogue_of_errors()["E_MODEL_MISSING"].message  # a reason that ends in a full stop
    titles = {"swiftf0": "SwiftF0 (pitch)", "espeak-ng": "eSpeak NG"}
    parts = [{"id": "swiftf0", "status": "failed", "reason": "found no pitch in the test clip"},
             {"id": "espeak-ng", "status": "unavailable", "reason": missing}]  # fmt: skip
    failed = catalogue_of_errors()["E_SELFTEST_FAILED"].message
    reply = {"ok": False, "data": {"passed": False, "components": parts},
             "error": {"code": "E_SELFTEST_FAILED", "message": f"{failed} (swiftf0, espeak-ng)"}}  # fmt: skip
    ok, lines = views.task_result("selftest", reply, titles)
    assert not ok and lines == [
        "SwiftF0 (pitch) did not pass: found no pitch in the test clip.",
        f"eSpeak NG did not pass: {missing}",
        failed,
    ]
    ready = {"ok": True, "data": {"passed": True, "components": parts[1:]}, "error": None}
    assert views.task_result("selftest", ready, titles)[1] == [
        f"Ready, with limits: eSpeak NG is off because {missing}"
    ]
    assert all(".." not in line for line in lines)


@pytest.mark.parametrize("route", ["/data/delete-all", f"/jobs/{JOB}/delete"])
def test_a_file_another_program_holds_stays_the_rest_goes_and_the_card_says_so(
    client: TestClient, finished: Path, home: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    held = home.parent / "app" / "tmp" / JOB / "held.wav"
    held.parent.mkdir(parents=True, exist_ok=True)
    held.write_bytes(b"RIFF")
    (home / "logs").mkdir(exist_ok=True)
    (home / "logs" / "app.log").write_text("line", encoding="utf-8")
    real = os.unlink

    def unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        if Path(path).name == held.name:  # Windows' answer while a virus scan or a player reads the file
            raise PermissionError(32, "The process cannot access the file: another process is using it")
        real(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)

    def heading(text: str) -> str:
        return str(re.findall(r"<h1[^>]*>(.*?)</h1>", text, re.S)[0])

    title = heading(client.get(f"/jobs/{JOB}").text)
    reply = post(client, route)
    assert reply.headers["HX-Redirect"] == "/jobs?incomplete=true"  # History as it is now, not the old list
    # Only the record stays beside the held file: the same job, now saying that its delete stopped half way.
    assert held.is_file() and [p.name for p in finished.iterdir()] == ["job.json"]
    assert (home / "logs").exists() is (route != "/data/delete-all")  # delete-all went on past the held file
    history = client.get(reply.headers["HX-Redirect"]).text
    assert "Some files were not deleted" in history and "so you can delete it again." in history
    assert "results are kept" not in history and f"/jobs/{JOB}" in history
    assert "Partly deleted" in history and "Failed" not in history and "Stopped" not in history
    page = client.get(f"/jobs/{JOB}").text
    assert heading(page) == title and "Another program was using some files" in page
    assert "such as a media player, then delete again." in page and "<details open>" in page
    assert "Run again" not in page  # the recording and the results are gone
    assert "Some files were not deleted" not in client.get("/jobs").text
    monkeypatch.setattr(os, "unlink", real)  # the other program lets go: a second delete removes the rest
    post(client, f"/jobs/{JOB}/delete")
    assert not finished.exists() and not held.exists() and "No analyses yet." in client.get("/jobs").text


def test_a_job_with_a_damaged_record_starts_its_line_with_the_start_time(
    client: TestClient, finished: Path
) -> None:
    (finished / "job.json").write_text('{"state": "runn', encoding="utf-8")  # no mode and no length to show
    meta = re.findall(r'<p class="meta">(.*?)</p>', client.get(f"/jobs/{JOB}").text, re.S)[0]
    assert re.sub(r"<[^>]+>", "", meta).startswith("Started ")


def test_history_warns_when_the_data_folder_is_synced(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "inside OneDrive or Dropbox" not in client.get("/jobs").text
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(tmp_path / "OneDrive - Home" / "Fluentfolio"))
    assert "inside OneDrive or Dropbox" in client.get("/jobs").text
    assert "No analyses yet." in client.get("/jobs").text


def sections(tab: str) -> list[type[Section]]:
    found = [info.annotation for info in Settings.model_fields.values()]
    return [s for s in found if isinstance(s, type) and issubclass(s, Section) and s.tab == tab]


@pytest.mark.parametrize("tab", ["basic", "advanced"])
def test_the_settings_page_draws_every_field_of_the_model_and_each_switchable_analyzer(
    client: TestClient, tab: str
) -> None:
    page = Page(client.get(f"/settings?tab={tab}").text)
    names = {str(a["name"]) for t, a in page.tags if t in ("input", "select") and a.get("name")}
    for section in sections(tab):
        assert str(section.model_config["title"]) in page.text
        for field, spec in section.model_fields.items():
            assert spec.title and spec.title in page.text, field
            key = next(k for k, i in Settings.model_fields.items() if i.annotation is section) + "." + field
            assert key in names, key
    switches = {f"analyzers.{a.id}" for a in registry.ANALYZERS if a.switchable}
    assert (switches <= names) == (tab == "basic") and "analyzers.fluency" not in names


def test_each_label_names_an_allowed_value() -> None:
    for tab in ("basic", "advanced"):
        for section in sections(tab):
            for field, spec in section.model_fields.items():
                labels = (spec.json_schema_extra or {}).get("labels", {})  # type: ignore[union-attr]
                if labels:
                    allowed = {str(v) for v in get_args(spec.annotation) or list(spec.annotation)}  # type: ignore[call-overload]
                    assert set(labels) <= allowed, field


def changed(client: TestClient, tab: str, **values: str) -> list[tuple[str, str]]:
    """The tab's form as the browser sends it after the learner changed these controls."""
    keys = {k.replace("__", "."): v for k, v in values.items()}
    sent = [(k, v) for k, v in Page(client.get(f"/settings?tab={tab}").text).form() if k not in keys]
    return [*sent, *keys.items()]


def test_a_settings_change_is_saved_at_once_and_a_theme_change_sets_the_cookie(
    client: TestClient, home: Path
) -> None:
    reply = client.post("/settings", content=urlencode(changed(client, "basic", speakers__backend="sherpa")),
                        headers={**OWN_PAGE, "content-type": FORM})  # fmt: skip
    assert reply.text == "Saved: Tool." and "hx-refresh" not in reply.headers
    assert 'backend = "sherpa"' in (home / "settings.toml").read_text(encoding="utf-8")
    reply = client.post("/settings", content=urlencode(changed(client, "basic", ui__theme="dark")),
                        headers={**OWN_PAGE, "content-type": FORM})  # fmt: skip
    assert reply.text == "Saved: Theme." and reply.headers["hx-refresh"] == "true"
    assert "theme=dark" in reply.headers["set-cookie"]
    reply = client.post("/settings", content=urlencode(changed(client, "advanced", limits__max_min="abc")),
                        headers={**OWN_PAGE, "content-type": FORM})  # fmt: skip
    assert card_code(reply) == "E_SETTINGS_INVALID"
    reply = post(client, "/settings/restore")
    assert reply.headers["hx-refresh"] == "true" and 'theme=""' in reply.headers["set-cookie"]
    assert "sherpa" not in (home / "settings.toml").read_text(encoding="utf-8")


def test_the_advanced_settings_show_both_folders_and_a_data_folder_set_by_the_environment_stays(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile(monkeypatch, home.parent)
    assert not Page(client.get("/settings").text).find("section", id="folders")
    page = Page(client.get("/settings?tab=advanced").text)
    assert page.find("section", id="folders") and in_profile("data") in page.text
    assert in_profile("app", "models") in page.text and str(home.parent) not in page.text
    # This test's FLUENTFOLIO_HOME chooses the data folder, so only the models folder has a form.
    assert "The FLUENTFOLIO_HOME variable chooses this folder. To move it, change that variable." in page.text
    assert [a["value"] for a in page.find("input", name="kind")] == ["models"]
    assert page.find("input", type="text", name="target", required=None, aria_describedby="move-help-models")
    # The help names the rule the move checks, which allows digits and punctuation in the path.
    assert (
        "with no spaces and no letters outside A to Z." in page.text and "only the letters" not in page.text
    )


def test_the_models_folder_moves_and_the_page_then_shows_the_new_one(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "path_problems", lambda _: [])  # tested on its own; tmp paths are long
    old, new = home.parent / "app" / "models", home.parent / "models2"
    (old / "hf").mkdir(parents=True)
    (old / "hf" / "model.onnx").write_bytes(b"\0" * 10)
    reply = post(client, "/settings/move", {"kind": "models", "target": str(new)})
    assert reply.headers["hx-refresh"] == "true"
    assert paths.models_dir() == new and (new / "hf" / "model.onnx").stat().st_size == 10 and not old.exists()
    profile(monkeypatch, home.parent)
    assert in_profile("models2") in client.get("/settings?tab=advanced").text


def test_a_data_folder_move_switches_then_stops_the_server_which_deletes_the_old_folder(
    home: Path, starts: Starts, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FLUENTFOLIO_HOME")  # as after an install: state.json names the data folder
    write_json(home.parent / "app" / "state.json", {"selftest": {"passed": True}, "data_root": str(home)})
    (home / "settings.toml").write_text("", encoding="utf-8")
    stops: list[Path] = []
    signed = TestClient(
        web_app.build(ORIGIN, "token", SESSION, stops.append, starts.tasks), base_url=ORIGIN,
        raise_server_exceptions=False,
    )  # fmt: skip
    signed.cookies.set(SESSION_COOKIE, SESSION)
    new = home.parent / "new data"
    profile(monkeypatch, home.parent)
    reply = post(signed, "/settings/move", {"kind": "data", "target": str(new)})
    assert reply.headers["hx-retarget"] == "#main" and "Fluentfolio has stopped" in reply.text
    assert in_profile("new data") in html.unescape(reply.text) and str(home.parent) not in reply.text
    assert stops == [home]  # the server stops after the reply, and deletes this folder once it lets it go
    assert paths.data_root() == new and (new / "settings.toml").is_file() and home.is_dir()


def test_a_move_target_may_start_with_the_home_variable_the_page_shows(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "path_problems", lambda _: [])
    profile(monkeypatch, home.parent)
    (home.parent / "app" / "models").mkdir(parents=True)
    reply = post(client, "/settings/move", {"kind": "models", "target": in_profile("models3")})
    assert reply.headers["hx-refresh"] == "true" and paths.models_dir() == home.parent / "models3"


def test_the_components_page_writes_the_command_path_with_the_home_part_as_a_variable(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile(monkeypatch, home.parent)
    command = (home.parent / "venv" / "bin" / "fluentfolio").as_posix()
    monkeypatch.setattr(pages, "executable", lambda: command)
    text = html.unescape(client.get("/components").text)
    assert in_profile("venv", "bin", "fluentfolio") in text and home.parent.as_posix() not in text


@pytest.mark.parametrize("given", ["", "relative\\folder", "a file"])
def test_a_move_to_anything_but_the_full_path_of_a_folder_is_refused_in_place(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, given: str
) -> None:
    monkeypatch.setattr(paths, "path_problems", lambda _: [])
    (home.parent / "notes.txt").write_text("x", encoding="utf-8")
    target = str(home.parent / "notes.txt") if given == "a file" else given
    reply = post(client, "/settings/move", {"kind": "models", "target": target})
    assert card_code(reply) == "E_MOVE_FAILED" and 'href="/settings?tab=advanced#folders"' in reply.text
    assert "In this case: not the full path of a folder." in Page(reply.text).text  # which rule failed
    assert paths.models_dir() == home.parent / "app" / "models"


def test_a_move_into_the_old_folder_says_so_and_a_card_of_another_code_shows_no_detail(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "path_problems", lambda _: [])
    inside = home.parent / "app" / "models" / "deeper"
    reply = post(client, "/settings/move", {"kind": "models", "target": str(inside)})
    assert card_code(reply) == "E_MOVE_FAILED"
    assert "In this case: one folder is inside the other." in Page(reply.text).text

    def busy(*_: Any, **__: Any) -> None:
        raise AppError("E_BUSY", str(home))  # a detail that holds a path stays off the card

    monkeypatch.setattr(facade, "move_folder", busy)
    other = post(client, "/settings/move", {"kind": "models", "target": str(home.parent / "m")})
    assert card_code(other) == "E_BUSY" and "In this case" not in other.text and str(home) not in other.text


def test_a_move_waits_while_a_job_runs_and_a_folder_kind_it_does_not_know_is_refused(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "path_problems", lambda _: [])
    with work_lock(home):
        reply = post(client, "/settings/move", {"kind": "models", "target": str(home.parent / "m")})
    assert card_code(reply) == "E_BUSY" and not (home.parent / "m").exists()
    unknown = post(client, "/settings/move", {"kind": "logs", "target": str(home.parent / "m")})
    assert unknown.status_code == 400


def test_each_component_row_says_its_status_size_and_licence(client: TestClient, home: Path) -> None:
    rows = {r.id: r for r in facade.components()}
    page = Page(client.get("/components").text)
    whisper = catalogue.get("whisper-large-v3-turbo")
    assert rows[whisper.id].status == "needs-download"  # this test's app folder has no models
    size = sum(f.size for f in whisper.files_for(SYSTEM))
    row = re.search(r"Whisper large-v3-turbo(.*?)</tr>", client.get("/components").text, re.S)
    assert row and "Needs download" in row.group(1) and "Not for grammar evidence" in row.group(1)
    assert f"{size / 1e9:.1f} GB" in row.group(1) or f"{size / 1e6:.0f} MB" in row.group(1)
    assert page.find("input", name="part", value="opensmile", checked=None)  # its row starts ticked
    (licence,) = page.find("input", name="licence.opensmile", value="yes")
    assert "checked" not in licence  # and its separate licence box is not
    for r in rows.values():
        if r.status == "not-available":
            assert page.find("input", value=r.id, disabled=None), r.id
    # The licence in plain words: openSMILE's research licence allows non-commercial use only.
    row = re.search(r"openSMILE \(loudness and pitch\)(.*?)</tr>", client.get("/components").text, re.S)
    assert row and "Non-commercial use only" in row.group(1) and ">nc<" not in row.group(1)


def test_a_part_that_failed_its_accuracy_check_here_starts_unticked_and_out_of_the_total(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # MFA did not pass its word-timing check on Windows or Linux (models.toml), so it is opt-in on both.
    source = client.get("/components").text
    page = Page(source)
    assert page.find("input", name="part", value="mfa") and not page.find("input", value="mfa", checked=None)
    ticked = [a["value"] for a in page.find("input", name="part", checked=None)]
    assert "sherpa-onnx-diarizer" in ticked and "phone-model" in ticked  # the other downloads stay ticked
    # The total under the table is the total of the ticked boxes, and adding MFA would change it.
    shown = re.search(r'<p id="totals"[^>]*>(.*?)</p>', source, re.S)
    assert shown is not None

    def size_line(ids: list[str]) -> str:
        return client.get("/components/totals", params=[("part", i) for i in ids]).text.split(" Free")[0]

    assert shown.group(1).split(" Free")[0] == size_line(ticked) != size_line([*ticked, "mfa"])
    # Where it passed, its box would start ticked like the others.
    aligner = catalogue.get("mfa").model_copy(update={"experimental_on": []})
    passed = {**catalogue.components(), "mfa": aligner}
    monkeypatch.setattr(catalogue, "components", lambda: passed)
    assert Page(client.get("/components").text).find("input", name="part", value="mfa", checked=None)


def test_each_long_action_on_components_answers_right_under_its_own_button(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = facade.components()
    ready = [replace(r, status="ready") if r.id == "sherpa-onnx-diarizer" else r for r in rows]
    monkeypatch.setattr(facade, "components", lambda: ready)  # a downloaded part, so "Remove a part" shows
    source = client.get("/components").text
    page = Page(source)
    assert not page.find("div", id="task")  # no shared status box at the top of a long page
    actions = [("Install selected", "install"), ("Run the check", "selftest"), ("Delete its files", "remove")]
    order = [source.index(label) for label, _ in actions]
    for (label, kind), at in zip(actions, order, strict=True):
        status, busy = source.index(f'id="{kind}-status"'), source.index(f'id="{kind}-busy"')
        assert at < busy < status, label  # the busy line, then the result, follow the button
        later = [o for o in order if o > at]
        assert not later or status < min(later), label  # and come before the next action's button
        assert page.find("p", id=f"{kind}-busy", role="status", **{"class": "htmx-indicator"})
    assert page.find("form", id="components", hx_target="#install-status", hx_indicator="#install-busy")
    assert page.find("button", hx_post="/selftest", hx_target="#selftest-status", hx_disabled_elt="this")


def test_the_components_are_grouped_by_what_they_do(client: TestClient) -> None:
    source = client.get("/components").text
    groups = re.findall(r'<th colspan="5" scope="rowgroup">(.*?)</th>', source)
    assert groups[0] == "Writing down the words" and len(groups) == len(set(groups))
    # Each part sits under the group of its kind: every speech model under the first heading.
    first = source.split(groups[0], 1)[1].split(groups[1], 1)[0]
    speech_models = [c.title for c in catalogue.offered().values() if c.kind == "asr"]
    assert speech_models and all(html.escape(t) in first for t in speech_models)
    assert "Silero" not in first and "Praat" not in first


def test_a_part_kept_only_for_the_speech_model_comparison_is_never_offered(
    client: TestClient, home: Path
) -> None:
    canary = catalogue.get("canary-1b-v2")  # still in the catalogue: the comparison in eval/ runs it
    assert not canary.offered and canary.id not in catalogue.offered()
    assert len(catalogue.offered()) == len(catalogue.components()) - 1  # the one part held back
    for url in ("/components", "/components?first=true", *HELP):
        assert "Canary" not in client.get(url).text, url
    assert canary.id not in {r.id for r in facade.components()}
    # Naming it anyway changes nothing: the install form skips it and removing it is refused.
    reply = post(client, "/components/install", {"part": [canary.id]})
    assert reply.status_code == 200 and reply.headers.get("hx-refresh") == "true"
    assert post(client, "/components/remove", {"part": canary.id}).status_code == 400


def test_the_licence_box_records_the_consent_and_the_row_tick_alone_does_not(
    client: TestClient, home: Path
) -> None:
    reply = post(client, "/components/install", {"part": ["opensmile"]})
    assert reply.headers["hx-refresh"] == "true" and not (home / "consent.json").exists()
    post(client, "/components/install", {"part": ["opensmile"], "licence.opensmile": "yes"})
    record = read_json(home / "consent.json")["opensmile"]
    assert record["source"] == "ui" and set(record) == {"utc", "source"}


def test_after_the_licence_box_the_row_says_the_installer_adds_the_package(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # openSMILE is a package inside the app, and only the installer may add one: the row names that step.
    real = checks.util.find_spec
    missing = lambda name, *a: None if name == "opensmile" else real(name, *a)  # noqa: E731
    monkeypatch.setattr(checks.util, "find_spec", missing)
    post(client, "/components/install", {"part": ["opensmile"], "licence.opensmile": "yes"})
    row = re.search(r"openSMILE \(loudness and pitch\)(.*?)</tr>", client.get("/components").text, re.S)
    step = "Only the installer adds its package, so close Fluentfolio and run the installer again (Repair)."
    assert row and step in row.group(1) and "Not available" in row.group(1)


def test_the_download_total_follows_the_ticked_parts(client: TestClient) -> None:
    ids = ["sherpa-onnx-diarizer", "spacy-en-core-web-sm"]  # downloads that need no other part
    total = sum(f.size for i in ids for f in catalogue.get(i).files_for(SYSTEM))
    reply = client.get("/components/totals", params=[("part", i) for i in ids])
    shown = f"{total / 1e9:.1f} GB" if total >= 999_500_000 else f"{total / 1e6:.0f} MB"
    assert reply.text.startswith(f"To download: {shown},") and "at 50 Mbit/s" in reply.text
    assert client.get("/components/totals", params={"part": "no-such-part"}).text.startswith(
        "To download: 0 bytes."
    )
    assert "stays in the total" not in reply.text


def test_an_unticked_part_that_a_ticked_part_needs_is_named_beside_the_total(client: TestClient) -> None:
    # The CTC aligner is built in but needs the phone recogniser (models.toml), which this test's empty
    # models folder lacks: ticking the aligner alone still counts the recogniser's files.
    phones = catalogue.get("phone-model")
    total = sum(f.size for f in phones.files_for(SYSTEM))
    reply = client.get("/components/totals", params={"part": "ctc-aligner"}).text
    assert reply.startswith(f"To download: {total / 1e9:.1f} GB,")
    assert f"{phones.title} stays in the total: CTC aligner needs it." in reply
    both = client.get("/components/totals", params=[("part", "ctc-aligner"), ("part", phones.id)]).text
    assert both.startswith(f"To download: {total / 1e9:.1f} GB,") and "stays in the total" not in both


def test_doctor_rows_link_to_the_page_that_fixes_them(client: TestClient) -> None:
    page = Page(client.get("/doctor").text)
    assert page.find("a", href="/components")  # this test's app folder has no models
    assert {"OK", "Warning", "Problem"} & set(page.text.split())
    assert "Copy diagnostic report" in page.text


def test_the_token_steps_say_that_accepting_the_terms_shares_contact_details(client: TestClient) -> None:
    page = Page(client.get("/token").text)
    assert "accept its terms. Accepting them shares your contact details with pyannote's makers." in page.text


def test_the_token_page_says_which_problem_it_found_and_never_shows_the_value(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "hf_" + "q" * 34  # a made-up token of the right shape
    refused = facade.AppError("E_TOKEN_INVALID")

    def invalid(value: str) -> Any:
        raise refused

    monkeypatch.setattr(facade, "token_set", invalid)
    reply = post(client, "/token", {"token": secret})
    assert "does not accept this token" in reply.text and "huggingface.co/settings/tokens" in reply.text
    assert secret not in reply.text
    refused = facade.AppError("E_NETWORK")  # no connection: the form stays, with its own message
    reply = post(client, "/token", {"token": secret})
    assert "could not reach Hugging Face to check this token" in reply.text and "error-card" not in reply.text
    assert 'id="token-value"' in reply.text and secret not in reply.text
    status = type("Status", (), {"present": True, "valid": True, "pyannote_terms_ok": False})()
    monkeypatch.setattr(facade, "token_set", lambda value: (status, True))
    reply = post(client, "/token", {"token": secret})
    repo = catalogue.get("pyannote-community-1").repo
    assert "terms are not accepted yet" in reply.text and f"https://huggingface.co/{repo}" in reply.text
    assert secret not in reply.text and 'type="password"' in reply.text


def test_a_running_install_polls_from_outside_its_form_so_it_takes_none_of_the_forms_settings(
    client: TestClient, starts: Starts
) -> None:
    task = starts.tasks.start(
        "install", [sys.executable, "-c", "import time; time.sleep(30)"], dict(os.environ)
    )
    try:
        source = client.get("/components").text
    finally:
        task.proc.kill()
        task.proc.wait()
    start = source.index('<form id="components"')
    end, poll = source.index("</form>", start), source.index('hx-get="/tasks/install"')
    # Inside, the progress line would take the form's target and replace the whole status box on its first
    # update, so the next update would find no box.
    assert not start < poll < end


def test_a_running_long_action_is_polled_and_its_end_reloads_the_page(
    client: TestClient, starts: Starts, tmp_path: Path
) -> None:
    assert client.get("/tasks/install").headers["hx-refresh"] == "true"  # nothing ran yet
    assert client.get("/tasks/anything").status_code == 404
    task = starts.tasks.start(
        "install", [sys.executable, "-c", "import time; time.sleep(30)"], dict(os.environ)
    )
    try:
        page = Page(client.get("/tasks/install").text)
        assert page.find("div", hx_get="/tasks/install", hx_trigger="every 2s")
        assert card_code(post(client, "/selftest")) == "E_BUSY"  # one long action at a time
    finally:
        task.proc.kill()
        task.proc.wait()
    assert client.get("/tasks/install").headers["hx-refresh"] == "true"


def test_an_install_shows_a_bar_from_its_newest_download_line(client: TestClient, starts: Starts) -> None:
    lines = [
        "[1/2] whisper-large-v3-turbo", "  model.bin (1.6 GB)", "12% of 1.6 GB downloaded. Now: Whisper.",
        "34% of 1.6 GB downloaded, 3.1 MB/s, about 7 min left. Now: Whisper.", "[2/2] silero-vad",
    ]  # fmt: skip
    for count, shares in ((2, []), (5, ["34"])):  # before any download line, then after two
        shown = "; ".join(f"print({line!r}, file=sys.stderr, flush=True)" for line in lines[:count])
        task = starts.tasks.start(
            "install", [sys.executable, "-c", f"import sys, time; {shown}; time.sleep(30)"], dict(os.environ)
        )
        newest = lines[count - 1].strip()
        try:
            for _ in range(100):  # until the child has written its lines
                if task.last_line() == newest:
                    break
                time.sleep(0.1)
            page = Page(client.get("/tasks/install").text)
            assert [bar["value"] for bar in page.find("progress", max="100")] == shares
            assert newest in page.text
        finally:
            task.proc.kill()
            task.proc.wait()


def test_check_for_updates_answers_in_place_and_always_keeps_the_plain_steps(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    about = Page(client.get("/about?tab=about").text)
    assert about.find("button", hx_post="/updates/check", hx_target="#updates", hx_swap="outerHTML")
    assert "Nothing about you or your recordings is sent." in about.text
    with local_server(monkeypatch) as served:
        # GitHub's answers in turn: a newer release, no answer, no release published yet.
        served.bodies["/repos/SacitDev/Fluentfolio/releases/latest"] = [b'{"tag_name": "v99.0.0"}', 503, 404]
        newer, failed, none = (Page(post(client, "/updates/check").text) for _ in range(3))
    assert f"Version 99.0.0 is available. You have version {__version__}." in newer.text
    assert "could not reach GitHub to look for a newer version" in failed.text
    assert f"No version has been published yet. You have version {__version__}." in none.text
    for answer in (newer, failed, none):
        assert answer.find("section", id="updates") and answer.find("p", autofocus=None)
        assert answer.find("a", href="https://github.com/SacitDev/Fluentfolio/releases", target="_blank",
                           rel="noopener noreferrer")  # fmt: skip
        assert "double-click Install.cmd" in answer.text and not COMMAND.search(answer.text)


def test_every_offered_part_has_its_plain_line_on_the_help_page(client: TestClient) -> None:
    assert set(views.PART_ABOUT) == set(catalogue.offered())
    shown = Page(client.get("/about?tab=parts").text).text
    for cid, part in catalogue.offered().items():
        assert f"{part.title} {views.PART_ABOUT[cid]}" in shown, cid


def test_each_help_tab_shows_its_own_page_and_an_odd_link_shows_the_first(client: TestClient) -> None:
    own = {
        "start": "Your first practice, step by step",
        "how": "Why not send the recording straight to an AI chat?",
        "record": "With a video call", "parts": "Fluentfolio uses several free parts.",
        "about": "Parts and their licences",
    }  # fmt: skip
    assert set(own) == set(views.HELP_TABS)
    for key, link in zip(views.HELP_TABS, HELP, strict=True):
        page = Page(client.get(link).text)
        assert [k for k, words in own.items() if words in page.text] == [key], link
        current = [a["href"] for a in page.find("a", aria_current="page")]  # the header marks Help too
        assert [href for href in current if href in HELP] == [link], link
    assert client.get("/about?tab=nonsense").text == client.get("/about?tab=start").text
    assert client.get("/about").text == client.get("/about?tab=start").text


def test_the_about_tab_credits_both_makers_and_keeps_the_version_and_the_licences(client: TestClient) -> None:
    about = Page(client.get("/about?tab=about").text)
    for maker in ("SacitDev", "sina-builds"):
        link = f"https://github.com/{maker}"
        assert about.find("a", href=link, target="_blank", rel="noopener noreferrer"), maker
    assert f"Fluentfolio {__version__} is free software under the MIT licence." in about.text
    assert UI_03_LINES[1] in about.text
    parts = catalogue.offered().values()
    assert all(part.title in about.text and part.licence in about.text for part in parts)


def test_every_facade_function_has_its_web_route_or_a_reason() -> None:
    from tests.unit.test_cli import FACADE_COMMANDS  # the command table; one source for both halves

    public = {
        name for name, value in vars(facade).items()
        if inspect.isfunction(value) and value.__module__ == facade.__name__ and not name.startswith("_")
    }  # fmt: skip
    assert public == set(FACADE_ROUTES) | set(NO_ROUTE) and not set(FACADE_ROUTES) & set(NO_ROUTE)
    app = web_app.build("http://127.0.0.1:1", "t", "s", lambda: None, Tasks(paths.data_root()))
    # FastAPI keeps an included router as its own object, so the pages' routes come from the router itself.
    every = [*app.app.routes, *pages.router.routes]  # type: ignore[attr-defined]
    routes = {f"{m} {r.path}": r.endpoint for r in every if isinstance(r, APIRoute) for m in r.methods}
    assert not [r for r in routes if "check-answer" in r]  # the answer check is on the command line only
    for name, route in FACADE_ROUTES.items():
        assert route in routes, (name, route)
        source = inspect.getsource(routes[route])
        # The route calls the function, or runs the command that calls it (install, self-test, export).
        words = FACADE_COMMANDS[name].split() if name in FACADE_COMMANDS else []
        runs = ".command(" in source and all(f'"{w}"' in source for w in words) and words
        # Called, or handed to a thread.
        assert re.search(rf"\bfacade\.{name}\b", source) or runs, (name, route)
