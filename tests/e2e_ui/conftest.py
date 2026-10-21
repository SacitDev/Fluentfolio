"""The app in a real browser. Each test gets its own running app in a throwaway data folder."""

from __future__ import annotations

import os
import wave
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from playwright.sync_api import Browser

from tests.support.browser import Served, checked_app, edge, running_app


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with edge() as launched:
        yield launched


@pytest.fixture
def served(tmp_path: Path, serving: ModuleType) -> Iterator[Served]:
    with running_app(serving, tmp_path / "home", checked_app(tmp_path / "app")) as found:
        yield found


@pytest.fixture
def first_run(tmp_path: Path, serving: ModuleType) -> Iterator[Served]:
    """The app on a computer where it has never passed its self-test."""
    with running_app(serving, tmp_path / "home", tmp_path / "app") as found:
        yield found


@pytest.fixture(scope="session")
def shots(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where screenshots go: the folder FF_UI_SHOTS names, for a reviewer to look at, or a temporary one."""
    named = os.environ.get("FF_UI_SHOTS")
    folder = Path(named) if named else tmp_path_factory.mktemp("shots")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# A finished solo job made from hand-chosen words: one answer from 1.0 s with a pause of 1.3 s inside.
JOB = "20261013-120000-abcd"
SPOKEN = [
    ("Reading", 1000, 1500, 0), ("is", 1550, 1700, 0), ("useful", 1750, 2300, 0),
    ("because", 3600, 4100, 0), ("it", 4150, 4250, 0), ("helps.", 4300, 4800, 0),
]  # fmt: skip
SEGMENTS = [(1000, 2300), (3600, 4800)]
AUDIO_S = 6


@pytest.fixture
def served_job(tmp_path: Path, serving: ModuleType, synthetic_job: ModuleType) -> Iterator[Served]:
    """The app with that finished job, and its 6 s playback copy: a 220 Hz tone, as no words are needed."""
    job = synthetic_job.make(tmp_path, SPOKEN, SEGMENTS, AUDIO_S * 1000)
    assert synthetic_job.finish(tmp_path).state.value == "completed"
    (job / "input").mkdir(exist_ok=True)
    with wave.open(str(job / "input" / "audio16k.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(int(8000 if (i // 36) % 2 else -8000).to_bytes(2, "little", signed=True)
                               for i in range(16000 * AUDIO_S)))  # fmt: skip
    with running_app(serving, tmp_path / "data", checked_app(tmp_path / "app")) as found:
        yield found
