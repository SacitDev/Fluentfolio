"""Test markers, the only skip rules, and the environment every test runs in."""

import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hypothesis import settings

from fluentfolio.core.logs import close_app_log
from fluentfolio.setup import device

MARKERS = {
    "slow": "needs real models, tools, audio or a browser; runs in the full check",
    "gpu": "needs an NVIDIA GPU; skipped without one unless FF_REQUIRE_GPU=1",
    "network": "may use the internet; runs only when asked for",
    "private_sample": "runs on private recordings from FF_PRIVATE_SAMPLES; numbers only",
}
# Tests of a property only one system has: the test, the system it runs on, and why it runs nowhere else.
ONE_SYSTEM = {
    "test_a_reader_waits_while_windows_refuses_the_file_for_a_moment": (
        "win32",
        "only Windows refuses to read a locked file",
    ),
    "test_the_saved_token_file_is_readable_by_its_owner_only": ("linux", "file modes are a Linux property"),
    # Whole files: each installer script runs on its own system only.
    "test_installer.py": ("win32", "the Windows installer is a PowerShell script"),
    "test_install_linux.py": ("linux", "the Linux installer is a bash script"),
}
# Tests that read what openSMILE measured. It comes only with the nc extra, since its licence allows
# non-commercial use only, so the GitHub checks and a contributor without that extra do not have it.
NEEDS_OPENSMILE = {
    "test_opensmile_hears_a_120_hz_tone_the_same_way_twice",
    "test_the_three_pitch_tools_agree_on_a_120_hz_tone",
    "test_opensmile_hears_6_db_more_as_louder",
    "test_loudness_is_not_measured_on_noisy_audio",
}
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
DEAD_PROXY = "http://127.0.0.1:9"
SUMMARY_KEY = re.compile(r"[a-z0-9_]{1,40}")
AUDIO = {".wav", ".mp3", ".mp4", ".m4a", ".mov", ".webm", ".mkv", ".ogg", ".opus", ".flac"}

settings.register_profile("fluentfolio", derandomize=True, deadline=None, database=None)
settings.load_profile("fluentfolio")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-goldens", action="store_true", help="rewrite the golden files the selected tests compare"
    )


def pytest_configure(config: pytest.Config) -> None:
    for name, text in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {text}")


@pytest.fixture
def update_goldens(request: pytest.FixtureRequest) -> bool:
    """True only with --update-goldens; a golden never changes as a side effect of a normal run."""
    return bool(request.config.getoption("--update-goldens"))


def _cuda_present() -> bool:
    # nvidia-smi lists a card that CUDA_VISIBLE_DEVICES hides; the app treats it as absent, and so do tests.
    exe = shutil.which("nvidia-smi")
    if exe is None or device.hidden():
        return False
    try:
        done = subprocess.run([exe, "-L"], capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and b"GPU" in done.stdout


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    need_gpu = any(item.get_closest_marker("gpu") for item in items)
    skip_gpu = need_gpu and os.environ.get("FF_REQUIRE_GPU") != "1" and not _cuda_present()
    skip_private = not os.environ.get("FF_PRIVATE_SAMPLES") and os.environ.get("FF_REQUIRE_PRIVATE") != "1"
    no_opensmile = importlib.util.find_spec("opensmile") is None
    for item in items:
        if no_opensmile and getattr(item, "originalname", item.name) in NEEDS_OPENSMILE:
            item.add_marker(pytest.mark.skip(reason="openSMILE is not installed (the nc extra)"))
        if skip_gpu and item.get_closest_marker("gpu"):
            item.add_marker(pytest.mark.skip(reason="no NVIDIA GPU found"))
        if skip_private and item.get_closest_marker("private_sample"):
            item.add_marker(pytest.mark.skip(reason="FF_PRIVATE_SAMPLES is not set"))
        name = item.path.name if item.path.name in ONE_SYSTEM else getattr(item, "originalname", item.name)
        system, why = ONE_SYSTEM.get(name, (sys.platform, ""))
        if system != sys.platform:
            item.add_marker(pytest.mark.skip(reason=why))


@pytest.fixture(scope="session")
def app_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The shared app folder, so real models download once and are read by every test."""
    if os.environ.get("FLUENTFOLIO_APP_ROOT"):
        return Path(os.environ["FLUENTFOLIO_APP_ROOT"])
    if os.environ.get("FF_TEST_DATA"):
        return Path(os.environ["FF_TEST_DATA"]) / "approot"
    return tmp_path_factory.mktemp("approot")


@pytest.fixture(scope="session")
def test_data() -> Path:
    """The public test data that tools/fetch_test_data.py downloads; a test that needs it fails without it."""
    folder = os.environ.get("FF_TEST_DATA")
    assert folder, "FF_TEST_DATA is not set; run tools/fetch_test_data.py first"
    return Path(folder)


@pytest.fixture(scope="session")
def tolerance() -> Callable[[str, str], float]:
    """A limit from tests/tolerances.toml, by section and name."""
    table = tomllib.loads((Path(__file__).parent / "tolerances.toml").read_text(encoding="utf-8"))
    return lambda section, name: float(table[section][name]["floor"])


def _support(name: str) -> ModuleType:
    """A helper module from tests/support, which is test code rather than a package."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / "support" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def session_builder() -> ModuleType:
    """The composed-session builder."""
    return _support("session_builder")


@pytest.fixture(scope="session")
def built(
    session_builder: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> Callable[[str], tuple[Path, dict[str, Any]]]:
    """A composed session's folder and truth, built once per run for the tests that only read it."""
    folders: dict[str, Path] = {}

    def get(name: str) -> tuple[Path, dict[str, Any]]:
        if name not in folders:
            out = tmp_path_factory.mktemp(name)
            if name in session_builder.DERIVED:
                session_builder.build_derived(name, out)
            else:
                session_builder.build(session_builder.SESSIONS / f"{name}.json", out)
            folders[name] = out
        # Read afresh each time, so no test sees another test's changes to the truth.
        return folders[name], json.loads((folders[name] / "truth.json").read_text(encoding="utf-8"))

    return get


@pytest.fixture(scope="session")
def serving() -> ModuleType:
    """Start `fluentfolio serve` as a launcher does, wait until it answers, and stop it."""
    return _support("serving")


@pytest.fixture(scope="session")
def scoring() -> ModuleType:
    """Scores against a composed session's truth: speaker separation error, roles and Parts."""
    return _support("scoring")


@pytest.fixture(scope="session")
def silence() -> ModuleType:
    """Reference speech edges from FFmpeg's silencedetect filter."""
    return _support("silence")


@pytest.fixture(scope="session")
def synthetic_job() -> ModuleType:
    """Job folders with hand-made results of the steps up to speech recognition."""
    return _support("synthetic_job")


def _numbers_only(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, list):
        return all(_numbers_only(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and SUMMARY_KEY.fullmatch(k) for k in value) and all(
            _numbers_only(v) for v in value.values()
        )
    return False


@pytest.fixture
def private_recordings() -> list[Path]:
    """The two recordings in FF_PRIVATE_SAMPLES, in name order; a failure shows a count, never a name."""
    folder = os.environ.get("FF_PRIVATE_SAMPLES")
    assert folder, "FF_PRIVATE_SAMPLES is not set"
    found = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in AUDIO)
    count = len(found)
    assert count == 2, f"expected 2 recordings, found {count}"
    return found


@pytest.fixture
def private_summary() -> Callable[[str, dict[str, object]], None]:
    """Merge numbers-only results into FF_PRIVATE_SUMMARY_DIR/summary.json; text never gets in."""

    def record(section: str, values: dict[str, object]) -> None:
        folder = os.environ.get("FF_PRIVATE_SUMMARY_DIR")
        assert folder, "FF_PRIVATE_SUMMARY_DIR is not set"
        path = Path(folder) / "summary.json"
        assert _numbers_only({section: values}), "only numbers and short lower-case keys may be recorded"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data.setdefault(section, {}).update(values)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        part.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(part, path)

    return record


@pytest.fixture(autouse=True)
def test_environment(
    request: pytest.FixtureRequest, tmp_path: Path, app_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Give each test its own data folder, and keep it offline unless it is marked `network`.

    A private-sample test's folder holds jobs made from a private recording, so it goes when the test ends.
    """
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(tmp_path / "ffhome"))
    monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(app_root))
    if request.node.get_closest_marker("private_sample"):
        request.addfinalizer(lambda: shutil.rmtree(tmp_path))
    # An in-process command opens the app log in this test's data folder; it closes before the rmtree above.
    request.addfinalizer(close_app_log)
    if request.node.get_closest_marker("network"):
        return
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, DEAD_PROXY)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    real_connect = socket.socket.connect

    def loopback_only(sock: socket.socket, address: object) -> None:
        host = address[0] if isinstance(address, tuple) else address
        if sock.family in (socket.AF_INET, socket.AF_INET6) and host not in LOOPBACK:
            raise OSError(f"network access in a test that is not marked network: {host}")
        real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
