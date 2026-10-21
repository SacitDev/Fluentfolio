"""Words to check through a whole analysis with the real phone model, on by default and switched off.

The check did not pass its accuracy check (eval/pronunciation_gate.py), so it is labelled experimental;
how often it flags native speech and planted sound swaps is measured there.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from fluentfolio.analyzers.measures import pronunciation
from fluentfolio.app import facade
from fluentfolio.core.enums import JobState
from fluentfolio.core.files import read_json
from fluentfolio.settings import store as settings_store
from fluentfolio.settings.model import Limits, Settings

pytestmark = pytest.mark.slow
RATE = 16000
CLIPS = Path(__file__).parents[1] / "fixtures" / "audio"


def analyse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, audio: np.ndarray, *, on: bool) -> Path:
    """A solo analysis of the audio, with the pronunciation check left at its default or switched off."""
    home = tmp_path / "home"
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(home))
    home.mkdir(parents=True)
    analyzers = Settings().analyzers if on else {pronunciation.ANALYZER: False}
    settings_store.save(home / "settings.toml", Settings(limits=Limits(min_s=1), analyzers=analyzers))
    source = tmp_path / "answer.wav"
    sf.write(source, audio, RATE, subtype="PCM_16")
    done = facade.analyze(source, "solo")
    assert done.record.state is JobState.COMPLETED, done.record.stages
    return done.job


def pointers(job: Path) -> dict[str, dict[str, Any]]:
    (c,) = read_json(job / "evidence.json")["candidates"]
    return {p["id"]: p for p in [*c["pointers"], *c["details"]["pointers"]]}


@pytest.fixture
def planted(session_builder: ModuleType) -> np.ndarray:
    """One answer with a planted "sink" for "think" between two clips said as written."""
    gap = np.zeros(RATE // 2, dtype=np.float32)
    first, swap, last = (
        session_builder.load_clip(CLIPS / f"{name}.opus") for name in ("sub02_ok", "sub01_bad", "sub04_ok")
    )
    return np.concatenate([first, gap, swap, gap, last])


def test_by_default_the_check_scores_each_word_and_says_it_is_experimental(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planted: np.ndarray
) -> None:
    job = analyse(tmp_path, monkeypatch, planted, on=True)
    phones = read_json(job / "stages" / "phones.json")
    assert phones["status"] == "ok" and phones["provenance"]["forms"] == ["en-us", "en-gb"]
    assert phones["provenance"]["dictionary"] is not None  # guessed pronunciations can be left out
    fitted = [f for u in phones["utterances"] for f in u["fits"] if f is not None]
    assert fitted and all(f["score"] <= 0 and f["expected"] for f in fitted)
    found = pointers(job)
    print(found["X06"], found["X07"]["note"], sep="\n")
    note = found["X07"]["note"]
    assert note.startswith("Checked ") and not note.startswith("Checked 0 ")
    assert pronunciation.ACCENT in note and "Experimental: did not pass its accuracy check." in note
    assert found["X06"]["shown"] == found["X06"]["total"]  # every listed word reaches the AI


def test_switched_off_the_check_is_off_and_the_phone_step_only_times_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planted: np.ndarray
) -> None:
    job = analyse(tmp_path, monkeypatch, planted, on=False)
    phones = read_json(job / "stages" / "phones.json")
    assert phones["status"] == "ok" and phones["provenance"]["forms"] is None
    assert all("fits" not in u and u["status"] == "ok" for u in phones["utterances"])
    found = pointers(job)
    assert (found["X06"]["total"], found["X06"]["note"]) == (0, "Not checked.")
    assert found["X07"]["note"] == "Not checked: the pronunciation check is off in Settings."
