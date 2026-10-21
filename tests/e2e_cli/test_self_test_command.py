"""`self-test` in its own process with the real models: a model file cut short fails its part, exit 3."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fluentfolio.core import paths
from fluentfolio.models import catalogue, download
from tests.support import linked_app

pytestmark = pytest.mark.slow
ALL_ON = "[components]\nmfa = true\n\n[analyzers]\npronunciation = true\n"
DAMAGED = "spacy-en-core-web-sm"


def test_a_model_file_cut_short_fails_its_part_and_asks_for_a_new_download(
    tmp_path: Path, app_root: Path,
    cli: Callable[..., subprocess.CompletedProcess[bytes]], envelope: Callable[..., Any],
) -> None:  # fmt: skip
    real = paths.models_dir()
    models = linked_app.damaged_models(real, tmp_path / "models", DAMAGED)
    app = linked_app.make(tmp_path / "app", app_root, models, "cpu")
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.toml").write_text(ALL_ON, encoding="utf-8")
    done = cli("self-test", "--json", home=home, env={"FLUENTFOLIO_APP_ROOT": str(app)}, timeout=900)
    found = envelope(done)
    assert done.returncode == 3, done.stderr[-2000:]
    assert found["error"]["code"] == "E_SELFTEST_FAILED" and found["error"]["message"].endswith(
        f"({DAMAGED})"
    )
    parts = {p["id"]: p for p in found["data"]["components"]}
    assert parts[DAMAGED]["status"] == "failed"
    assert parts[DAMAGED]["reason"] == "a file of this part is missing or damaged: re-download this model"
    others = {k: p["status"] for k, p in parts.items() if k != DAMAGED}
    assert set(others.values()) <= {"ok", "not-tested", "disabled"}  # no consent here, so openSMILE is off
    assert others["opensmile"] == "disabled" and others["parakeet-tdt-0.6b-v2"] == "ok"
    assert found["data"]["passed"] is False and not list((home / "jobs").iterdir())
    download.require(catalogue.get(DAMAGED), real)  # the real copy is whole: only the test's copy was cut
