"""Importing the package switches off library telemetry and keeps heavy libraries out of the app."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import fluentfolio

TELEMETRY = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_UPDATE_CHECK": "1",
    "PYANNOTE_METRICS_ENABLED": "0",
    "ORT_DISABLE_TELEMETRY": "1",
}
HEAVY = [
    "torch",
    "onnxruntime",
    "ctranslate2",
    "faster_whisper",
    "pyannote",
    "sherpa_onnx",
    "parselmouth",
    "opensmile",
    "av",
    "spacy",
]
LIGHT_APP_MODULES = ["fluentfolio.cli"]


def run_python(code: str, env: dict[str, str]) -> str:
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60, check=True
    )
    return done.stdout


def test_import_sets_every_telemetry_switch() -> None:
    env = {k: v for k, v in os.environ.items() if k not in TELEMETRY}
    env["PYANNOTE_METRICS_ENABLED"] = "1"  # a user value must not turn telemetry back on
    names = list(TELEMETRY)
    code = f"import json, os, fluentfolio; print(json.dumps({{k: os.environ.get(k) for k in {names!r}}}))"
    assert json.loads(run_python(code, env)) == TELEMETRY


def test_light_app_modules_import_no_heavy_library() -> None:
    imports = "".join(f"import {m}\n" for m in LIGHT_APP_MODULES)
    code = f"import json, sys\n{imports}print(json.dumps(sorted(m for m in {HEAVY!r} if m in sys.modules)))"
    assert json.loads(run_python(code, dict(os.environ))) == []


def calls_to(folder: Path, name: str) -> list[str]:
    """Each Python file under a folder whose syntax tree calls `name`, as a function or a method."""
    found = []
    for path in sorted(folder.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                called = getattr(node.func, "attr", getattr(node.func, "id", None))
                if called == name:
                    found.append(path.name)
    return found


def test_pyannotes_telemetry_switch_is_never_called(tmp_path: Path) -> None:
    assert calls_to(Path(fluentfolio.__file__).parent, "set_telemetry_metrics") == []
    plant = tmp_path / "plant.py"
    plant.write_text("from pyannote.audio import telemetry\ntelemetry.set_telemetry_metrics(True)\n", "utf-8")
    assert calls_to(tmp_path, "set_telemetry_metrics") == ["plant.py"]  # the search finds a real call
