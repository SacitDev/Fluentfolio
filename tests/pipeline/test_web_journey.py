"""A learner's first analysis in a real browser with the real models: the check, the upload, the results.

The browser journeys with stand-in model steps cover each page; this one runs every real step once, on the
composed interview. The app folder reuses the downloaded models through links, as in the self-test tests.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import jsonschema
import pytest
from playwright.sync_api import Browser

from fluentfolio.core import paths
from fluentfolio.core.files import read_json
from fluentfolio.setup import device
from tests.support import linked_app
from tests.support.browser import edge, follow, running_app, signed_in, start, upload

pytestmark = pytest.mark.slow
TESTS = Path(__file__).resolve().parents[1]
SCHEMA = TESTS.parent / "docs" / "schemas" / "evidence-v1.schema.json"
EVENT_GAP_S = 5.0  # live progress: an update at least this often
CHECK_S = 900  # the check's own budget is 3 min on the GPU and 6 on the CPU
ANALYSIS_S = 1800
STEP = re.compile(r"[A-Z][^:]+: (done, with gaps|done|skipped|not available|failed|switched off)")


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with edge() as launched:
        yield launched


def test_u2_with_the_real_models_the_check_then_an_interview_through_the_pages(
    browser: Browser, tmp_path: Path, app_root: Path, serving: ModuleType, session_builder: ModuleType
) -> None:
    backend = "cu130" if device.probe() else "cpu"  # what the installer chooses on this computer
    app = linked_app.make(tmp_path / "app", app_root, paths.models_dir(), backend)
    session_builder.build(TESTS / "fixtures" / "sessions" / "CS-INT.json", tmp_path / "session")
    with running_app(serving, tmp_path / "home", app) as served:
        page = signed_in(browser, served)
        page.wait_for_url(f"{served.origin}/components?first=1")  # no check has passed on this computer yet
        page.get_by_role("button", name="Run the check").click()
        running = page.locator(".task")  # the check's progress line, there until the check ends
        running.wait_for()
        running.wait_for(state="detached", timeout=CHECK_S * 1000)
        check = read_json(app / "state.json")["selftest"]
        assert check["passed"] is True and check["elapsed_ms"] <= check["budget_ms"], check
        page.get_by_role("link", name="Start analysis").click()
        page.wait_for_url(f"{served.origin}/")
        shown = upload(page, served, tmp_path / "session" / "session.wav", "interview")
        assert "Time to analyse it on this computer's" in shown
        began = time.monotonic()
        job_id = start(page, consent=True)
        changes = follow(page, ANALYSIS_S)
        took = time.monotonic() - began
        gaps = [b - a for a, b in zip(changes, changes[1:], strict=False)]
        evidence = read_json(served.home / "jobs" / job_id / "evidence.json")
        m = evidence["manifest"]
        ran = m["device"]["resolved"]
        print(f"check: {check['device']}, {check['elapsed_ms'] / 1000:.1f} s; analysis: {ran}, {took:.1f} s, "
              f"{len(changes)} progress changes, longest gap {max(gaps):.1f} s")  # fmt: skip
        assert m["stub"] is False  # every step was the real one
        assert ran == check["device"]  # the analysis ran where the check did
        jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(evidence)
        assert len(changes) >= 3 and max(gaps) <= EVENT_GAP_S, gaps
        page.goto(f"{served.origin}/jobs/{job_id}")
        assert page.locator(".page-head .tag").inner_text() in ("Finished", "Finished, with gaps")
        page.goto(f"{served.origin}/jobs/{job_id}?tab=transcript")
        text = page.locator(".transcript").inner_text()
        assert "The examiner, asking" in text and "The candidate, answering" in text
        for part in ("Part 1", "Part 2", "Part 3"):
            assert part in text, part
        page.goto(f"{served.origin}/jobs/{job_id}?tab=measures")
        page.get_by_text("More details: what each step did").click()
        steps = page.locator("details[open] li").all_inner_texts()
        assert len(steps) == len(m["tools"]) and all(STEP.match(s) for s in steps), steps
        page.context.close()
