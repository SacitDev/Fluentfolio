"""The installer's steps after its bootstrap script: the uv runs they build, the device step, the record of
the install, and the update check. uv and the GPU test are child processes, recorded or answered here."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fluentfolio import __version__
from fluentfolio.core.enums import Device
from fluentfolio.core.errors import SetupError
from fluentfolio.core.files import read_json, write_json
from fluentfolio.jobs import runner
from fluentfolio.setup import __main__ as steps
from fluentfolio.setup import device, sync, updates
from fluentfolio.setup.device import Card
from tests.support.fake_https import local_server

ROOT = Path(__file__).resolve().parents[2]  # the checkout, with its pyproject.toml and uv.lock

GOOD = Card((8, 6), 24576, "581.57")
OLD = Card((6, 1), 24576, "581.57")
LATEST = "/repos/SacitDev/Fluentfolio/releases/latest"  # the path of updates.LATEST


def test_each_sync_names_the_backend_and_every_extra_and_builds_no_package_but_fluentfolio() -> None:
    uv, project = Path("app/bin/uv"), Path("release")
    first, second = sync.sync_command(uv, project, "cu130", ["nc", "nc"])
    assert second == [
        str(uv), "sync", "--locked", "--no-dev", "--project", str(project),
        "--extra", "cu130", "--extra", "nc",
    ]  # fmt: skip
    assert first == [*second, "--no-build", "--no-install-project"]
    assert sync.sync_command(uv, project, "cpu", [])[1][-2:] == ["--extra", "cpu"]
    for backend, extras in (("cu128", []), ("cpu", ["dev"])):
        with pytest.raises(ValueError):
            sync.sync_command(uv, project, backend, extras)


def test_uv_keeps_python_and_its_caches_in_the_app_folder_and_registers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("HF_TOKEN", "hf_not_for_uv")
    monkeypatch.setenv("UV_INDEX_URL", "https://elsewhere.example/simple")
    env = sync.sync_env(tmp_path)
    assert env["UV_CACHE_DIR"] == str(tmp_path / "cache" / "uv")
    assert env["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / "python")
    settings = ("UV_PYTHON_PREFERENCE", "UV_PYTHON_INSTALL_REGISTRY", "UV_PYTHON_INSTALL_BIN")
    assert [env[name] for name in settings] == ["only-managed", "0", "0"]
    assert env["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert "HF_TOKEN" not in env and "UV_INDEX_URL" not in env


def test_the_lock_file_still_matches_under_the_installers_uv_settings(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv, "uv is needed to build and test Fluentfolio"
    # A matching lock needs no package index; the check needs a Python, here this test's own.
    env = {**sync.sync_env(tmp_path), "UV_OFFLINE": "1", "UV_PYTHON": sys.executable}
    done = subprocess.run(
        [uv, "lock", "--check", "--project", str(ROOT)], env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr


class Children:
    """uv runs recorded instead of started, each answering with the next exit code; the GPU test's child."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, cards: list[Card], uv_codes: tuple[int, ...] = (),
                 gpu_code: str | None = None) -> None:  # fmt: skip
        self.runs: list[list[str]] = []
        self.codes = list(uv_codes)
        self.gpu_code = gpu_code
        self.gpu_tests = 0
        monkeypatch.setattr(runner, "foreground", self.uv)
        monkeypatch.setattr(runner, "run", self.gpu_test)
        monkeypatch.setattr(device, "probe", lambda: cards)  # the driver's answer

    def uv(self, argv: list[str], env: dict[str, str]) -> int:
        assert env["UV_PYTHON_INSTALL_DIR"].endswith("python")
        self.runs.append(argv)
        return self.codes.pop(0) if self.codes else 0

    def gpu_test(self, argv: list[str], env: dict[str, str], on_progress: Callable[[dict[str, Any]], None],
                 **limits: float) -> runner.Outcome:  # fmt: skip
        assert argv[-2:] == ["-m", "fluentfolio.setup.device"]
        self.gpu_tests += 1
        if self.gpu_code:
            on_progress({"code": self.gpu_code})
        return runner.Outcome(1 if self.gpu_code else 0)

    @property
    def backends(self) -> list[str]:
        """The backend of each sync that started (its first, wheels-only run always starts)."""
        return [argv[argv.index("--extra") + 1] for argv in self.runs if "--no-build" in argv]


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A first install's app folder: the bootstrap installed the CPU build."""
    monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(tmp_path / "app"))
    write_json(tmp_path / "app" / "state.json", {"backend": "cpu", "extras": [], "data_root": "kept"})
    return tmp_path / "app"


def last_line(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out.strip().splitlines()[-1]


def test_auto_with_a_good_card_switches_to_gpu_support_and_keeps_it_when_the_gpu_test_passes(
    app: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    children = Children(monkeypatch, [GOOD])
    assert steps.device(Device.AUTO) == 0
    assert children.backends == ["cu130"] and children.gpu_tests == 1
    state = read_json(app / "state.json")
    assert (state["backend"], state["gpu_smoke"]["passed"], state["data_root"]) == ("cu130", True, "kept")
    assert last_line(capsys) == "Device: GPU. The graphics card passed every check."


TOO_OLD = "This graphics card is too old (compute 6.1; 7.5 is needed)."


@pytest.mark.parametrize(("cards", "reason"), [([], "No NVIDIA graphics card found."), ([OLD], TOO_OLD)])
def test_auto_without_a_usable_card_stays_on_the_cpu_and_says_why(
    app: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cards: list[Card],
    reason: str,
) -> None:
    children = Children(monkeypatch, cards)
    assert steps.device(Device.AUTO) == 0
    assert children.runs == [] and children.gpu_tests == 0  # the CPU build is already there
    assert read_json(app / "state.json")["backend"] == "cpu"
    assert last_line(capsys) == f"Device: CPU. {reason}"


def test_a_failed_gpu_test_under_auto_goes_back_to_the_cpu(
    app: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    children = Children(monkeypatch, [GOOD], gpu_code="E_CUDA_RUNTIME")
    assert steps.device(Device.AUTO) == 0
    assert children.backends == ["cu130", "cpu"]
    state = read_json(app / "state.json")
    assert (state["backend"], state["gpu_smoke"]["code"]) == ("cpu", "E_CUDA_RUNTIME")
    assert last_line(capsys) == "Device: CPU. The GPU test failed (E_CUDA_RUNTIME). Using the CPU."


def test_a_chosen_gpu_is_installed_even_when_the_card_or_its_test_fails_and_says_so(
    app: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    children = Children(monkeypatch, [OLD], gpu_code="E_CUDA_RUNTIME")
    assert steps.device(Device.GPU) == 0
    assert children.backends == ["cu130"] and read_json(app / "state.json")["backend"] == "cu130"
    out = capsys.readouterr().out
    assert f"Warning: {TOO_OLD} GPU support is installed" in out
    assert "Warning: the GPU test failed (E_CUDA_RUNTIME). Analyses will fail on the GPU." in out


def test_the_cpu_choice_never_asks_the_driver_and_takes_gpu_support_out(
    app: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_json(app / "state.json", {"backend": "cu130", "extras": ["nc"]})
    children = Children(monkeypatch, [GOOD])
    monkeypatch.setattr(device, "probe", lambda: pytest.fail("the driver was asked"))
    assert steps.device(Device.CPU) == 0
    assert children.backends == ["cpu"] and children.runs[0][-4:-2] == ["--extra", "nc"]  # the extra stays
    assert read_json(app / "state.json")["backend"] == "cpu"
    assert last_line(capsys) == "Device: CPU. You chose the CPU."


def test_gpu_support_that_cannot_be_downloaded_leaves_a_working_cpu_install(
    app: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    children = Children(monkeypatch, [GOOD], uv_codes=(1,))
    assert steps.device(Device.AUTO) == 0
    assert children.backends == ["cu130", "cpu"] and children.gpu_tests == 0
    assert read_json(app / "state.json")["backend"] == "cpu"
    assert last_line(capsys) == "Device: CPU. GPU support could not be installed. Using the CPU."
    children = Children(monkeypatch, [GOOD], uv_codes=(1, 1))
    assert steps.device(Device.AUTO) == 1
    assert "Problem: the install did not finish." in capsys.readouterr().out


def test_finish_adds_opensmile_only_after_its_licence_was_accepted_and_records_the_version(
    app: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = Path(steps.paths.data_root())
    children = Children(monkeypatch, [])
    assert steps.finish() == 0 and children.runs == []
    state = read_json(app / "state.json")
    assert (state["version"], state["app_dir"], state["extras"]) == (__version__, f"app/{__version__}", [])
    assert state["installed_utc"] and state["data_root"] == "kept"
    assert last_line(capsys) == f"Fluentfolio {__version__} is installed."
    write_json(data / "consent.json", {"opensmile": {"utc": "2026-10-17T00:00:00Z", "source": "yes"}})
    assert steps.finish() == 0 and children.runs == []  # not a record the user made
    capsys.readouterr()
    write_json(data / "consent.json", {"opensmile": {"utc": "2026-10-17T00:00:00Z", "source": "ui"}})
    assert steps.finish() == 0
    assert children.runs[-1][-4:] == ["--extra", "cpu", "--extra", "nc"]
    assert read_json(app / "state.json")["extras"] == ["nc"]
    said = capsys.readouterr().out  # the step names what it changes, not a backend switch
    assert "Adding openSMILE." in said and "support" not in said
    (data / "consent.json").unlink()
    children = Children(monkeypatch, [], uv_codes=(1,))
    assert steps.finish() == 1  # the extra could not be taken out: nothing new is recorded
    assert read_json(app / "state.json")["extras"] == ["nc"]
    assert "Removing openSMILE." in capsys.readouterr().out


def test_the_gpu_step_names_the_download_each_installers_screen_names() -> None:
    script = {
        "linux": ("install.sh", r"GPU_MB=(\d+)"),
        "win32": ("install.ps1", r"\$GpuMB, \$GpuDiskGB = \d+, \d+, (\d+)"),
    }  # fmt: skip
    for system, (name, size) in script.items():
        megabytes = re.search(size, (ROOT / name).read_text(encoding="utf-8"))
        assert megabytes and f"{int(megabytes[1]) / 1000:.1f}" == steps.GPU_GB[system], name
    assert f"about {steps.GPU_GB[steps.HERE]} GB" in steps.SWITCH["cu130"]


def test_the_install_steps_never_load_torch_which_a_backend_switch_replaces() -> None:
    # Windows cannot replace a library file that a running process has loaded.
    check = (
        "import sys, fluentfolio.setup.__main__; "
        "print(sorted(m for m in ('torch', 'torchaudio') if m in sys.modules))"
    )
    done = subprocess.run([sys.executable, "-I", "-c", check], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


@pytest.mark.parametrize(
    ("answer", "latest", "newer"),
    [
        (404, None, False),  # no release published yet
        (b'{"tag_name": "v0.0.1"}', "0.0.1", False),
        (f'{{"tag_name": "v{__version__}"}}'.encode(), __version__, False),
        (b'{"tag_name": "v99.0.0"}', "99.0.0", True),
    ],
)
def test_check_for_updates_compares_the_newest_release_with_this_version(
    monkeypatch: pytest.MonkeyPatch, answer: bytes | int, latest: str | None, newer: bool
) -> None:
    with local_server(monkeypatch) as served:
        served.bodies[LATEST] = [answer]
        found = updates.check()
    assert (found.current, found.latest, found.newer) == (__version__, latest, newer)
    assert served.seen == [(LATEST, None)]  # one request, nothing else


def test_check_for_updates_that_gets_no_answer_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    with local_server(monkeypatch) as served:
        served.bodies[LATEST] = [503]
        with pytest.raises(SetupError) as caught:
            updates.check()
    assert caught.value.code == "E_NETWORK"
