"""The self-test with the real models on the bundled clip: green on the GPU and on the CPU, and two faults.

Each test has its own app folder that reuses the downloaded models and MFA's environment through links.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fluentfolio.app import facade
from fluentfolio.core import paths
from fluentfolio.core.enums import Device
from fluentfolio.core.files import read_json
from tests.support import linked_app

pytestmark = pytest.mark.slow
ALL_ON = "[components]\nmfa = true\n\n[analyzers]\npronunciation = true\n"
Row = dict[str, Any]


@pytest.fixture
def setup(tmp_path: Path, app_root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Make the app folder and the settings, and give openSMILE's consent unless told not to."""
    models = paths.models_dir()
    home = Path(os.environ["FLUENTFOLIO_HOME"])

    def make(backend: str, mfa_env: bool = True, consent: bool = True) -> Path:
        app = linked_app.make(tmp_path / "app", app_root, models, backend, mfa_env)
        monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(app))
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        home.mkdir(parents=True, exist_ok=True)
        (home / "settings.toml").write_text(ALL_ON, encoding="utf-8")
        if consent:
            facade.consent("opensmile", "ui")
        return app

    return make


def statuses(outcome: Row) -> dict[str, str]:
    return {k: v["status"] for k, v in outcome["components"].items()}


EVERY_PART_OK = {
    "parakeet-tdt-0.6b-v2": "ok", "whisper-tiny": "ok", "silero-vad": "ok", "pyannote-community-1": "ok",
    "sherpa-onnx-diarizer": "not-tested", "mfa": "ok", "ctc-aligner": "ok", "phone-model": "ok",
    "espeak-ng": "ok", "pronunciation-dictionary": "ok", "praat": "ok", "swiftf0": "ok", "opensmile": "ok",
    "spacy-en-core-web-sm": "ok",
}  # fmt: skip


def finished_clean(app: Path, outcome: Row) -> None:
    """The outcome is the one kept in state.json, and no job of the clip is left behind."""
    assert read_json(app / "state.json")["selftest"] == outcome
    assert not list((Path(os.environ["FLUENTFOLIO_HOME"]) / "jobs").iterdir())
    assert facade.jobs(10) == []


@pytest.mark.gpu
def test_the_self_test_passes_on_the_gpu_within_its_budget(setup: Any) -> None:
    app = setup("cu130")
    outcome = facade.self_test()
    assert read_json(app / "state.json")["gpu_smoke"]["passed"] is True  # Auto chose the GPU after its test
    assert outcome["device"] == "gpu" and outcome["passed"] is True, outcome
    assert statuses(outcome) == EVERY_PART_OK
    assert outcome["elapsed_ms"] <= outcome["budget_ms"] == 180_000  # noqa: PLR2004
    finished_clean(app, outcome)


def test_the_self_test_passes_on_a_forced_cpu_within_its_budget(setup: Any) -> None:
    app = setup("cu130")
    outcome = facade.self_test(Device.CPU)
    assert "gpu_smoke" not in read_json(app / "state.json")  # a forced CPU needs no GPU test
    assert outcome["device"] == "cpu" and outcome["passed"] is True, outcome
    assert statuses(outcome) == EVERY_PART_OK
    assert outcome["elapsed_ms"] <= outcome["budget_ms"] == 360_000  # noqa: PLR2004
    finished_clean(app, outcome)


def test_an_empty_mfa_env_makes_mfa_unavailable_and_the_rest_ok(setup: Any) -> None:
    app = setup("cpu", mfa_env=False)
    outcome = facade.self_test()
    assert outcome["passed"] is True and outcome["device"] == "cpu"
    assert statuses(outcome) == EVERY_PART_OK | {"mfa": "unavailable"}
    assert outcome["components"]["mfa"]["reason"] == "its environment is not installed"
    finished_clean(app, outcome)


def test_without_opensmile_consent_it_is_disabled_and_the_test_passes(setup: Any) -> None:
    app = setup("cpu", consent=False)
    outcome = facade.self_test()
    assert outcome["passed"] is True
    assert statuses(outcome) == EVERY_PART_OK | {"opensmile": "disabled"}
    assert "non-commercial" in outcome["components"]["opensmile"]["reason"]
    finished_clean(app, outcome)
