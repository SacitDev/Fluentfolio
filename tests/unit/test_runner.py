"""The process runner: the built child environment, the argv rules, the output contract and keeping awake."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fluentfolio.app import facade
from fluentfolio.jobs import awake, runner
from fluentfolio.web.tasks import Tasks

FAKE = Path(__file__).parents[1] / "support" / "fake_stage.py"
FORBIDDEN = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PYTHONPATH", "PYTHONSTARTUP", "HF_HOME", "HF_DEBUG")


def test_child_env_is_built_from_the_allow_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*FORBIDDEN, "SOME_OTHER_VARIABLE"):
        monkeypatch.setenv(name, "planted")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    app, data, models = tmp_path / "app", tmp_path / "data", tmp_path / "models"
    env = runner.child_env(
        app, data, models, "20261013-170509-abcd", cpu=False, extra_path=[tmp_path / "bin"]
    )
    assert not set(FORBIDDEN) & set(env) and "SOME_OTHER_VARIABLE" not in env
    assert set(env) - set(runner.KEPT) == {
        *runner.SWITCHES, "PATH", "HF_HUB_CACHE", "TORCH_HOME", "TMP", "TEMP", "TMPDIR",
        "FLUENTFOLIO_HOME", "FLUENTFOLIO_APP_ROOT", "FLUENTFOLIO_DEVICE",
    }  # fmt: skip
    assert (
        env["HF_HUB_OFFLINE"] == env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == env["ORT_DISABLE_TELEMETRY"] == "1"
    )
    assert env["PYANNOTE_METRICS_ENABLED"] == "0"
    job_tmp = app / "tmp" / "20261013-170509-abcd"
    assert env["TMP"] == env["TEMP"] == env["TMPDIR"] == str(job_tmp) and job_tmp.is_dir()
    assert (env["HF_HUB_CACHE"], env["FLUENTFOLIO_HOME"], env["FLUENTFOLIO_APP_ROOT"]) == (
        str(models / "hf"), str(data), str(app),
    )  # fmt: skip
    path = env["PATH"].split(os.pathsep)
    assert path[0] == str(tmp_path / "bin") and str(Path(sys.executable).parent) in path
    assert "CUDA_VISIBLE_DEVICES" not in env and env["FLUENTFOLIO_DEVICE"] == "gpu"
    outside = runner.child_env(app, data, models, None, cpu=True)
    assert outside["CUDA_VISIBLE_DEVICES"] == "-1" and outside["TMP"] == str(app / "tmp" / "misc")
    assert outside["FLUENTFOLIO_DEVICE"] == "cpu"


def test_the_apps_own_processes_keep_its_settings_but_not_pythons_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("PYTHONPATH", "PYTHONSTARTUP", "PythonHome", "HF_DEBUG"):
        monkeypatch.setenv(name, "planted")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
    monkeypatch.setenv("FLUENTFOLIO_HOME", "kept")
    env = runner.own_env()
    assert not {"PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME", "HF_DEBUG"} & {name.upper() for name in env}
    assert (env["HTTPS_PROXY"], env["FLUENTFOLIO_HOME"]) == ("http://proxy.example.com:8080", "kept")


def test_the_web_app_and_its_commands_start_with_the_apps_own_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "planted")
    seen: list[dict[str, str]] = []
    monkeypatch.setattr(runner, "foreground", lambda argv, env: seen.append(env) or 0)
    monkeypatch.setattr(runner, "attached", lambda argv, env, out, err: seen.append(env) or SimpleNamespace())
    assert facade.serve(8765, browser=False) == 0
    Tasks(tmp_path).command("selftest", "self-test")
    assert len(seen) == 2 and all("PYTHONPATH" not in {name.upper() for name in env} for env in seen)


def test_a_cpu_child_sees_no_graphics_card_in_either_gpu_library(tmp_path: Path) -> None:
    # An empty CUDA_VISIBLE_DEVICES left the card visible to torch and CTranslate2 on Windows.
    probe = "import torch, ctranslate2; print(torch.cuda.is_available(), ctranslate2.get_cuda_device_count())"
    env = runner.child_env(tmp_path / "app", tmp_path / "data", tmp_path / "models", None, cpu=True)
    done = runner.run([sys.executable, "-I", "-c", probe], env=env, timeout_s=120, no_progress_s=120)
    assert (done.exit_code, done.stderr_tail[-1]) == (0, "False 0")


@pytest.mark.parametrize(("users", "cpu", "seen"), [("1", False, "1"), ("", False, ""), ("1", True, "-1")])
def test_the_users_choice_of_card_reaches_a_gpu_child_and_never_a_cpu_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, users: str, cpu: bool, seen: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", users)
    env = runner.child_env(tmp_path / "app", tmp_path / "data", tmp_path / "models", None, cpu=cpu)
    assert env["CUDA_VISIBLE_DEVICES"] == seen


@pytest.mark.parametrize(
    "program", ["python", "tools/run.bat", str(Path(sys.executable).with_name("run.CMD")), "/opt/run.cmd"]
)
def test_a_child_needs_an_absolute_program_that_is_not_a_batch_file(program: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        runner.run([program], env={}, timeout_s=5, no_progress_s=5)


def test_progress_lines_are_delivered_and_only_the_last_200_error_lines_kept(tmp_path: Path) -> None:
    (tmp_path / "data" / "jobs" / "j" / "stages").mkdir(parents=True)
    env = runner.child_env(tmp_path / "app", tmp_path / "data", tmp_path / "models", None, cpu=True)
    events: list[dict] = []
    outcome = runner.run(
        [sys.executable, "-I", str(FAKE), "chatty-a", "j"], env=env, timeout_s=60, no_progress_s=60,
        on_progress=events.append,
    )  # fmt: skip
    assert (outcome.exit_code, outcome.stopped) == (0, None)
    assert events == [{"pct": 0}, {"pct": 50}, {"pct": 100}]
    assert outcome.stderr_tail == [f"line {n}" for n in range(100, 300)]


def test_a_background_helper_ends_when_stopped(tmp_path: Path) -> None:
    (tmp_path / "data" / "jobs" / "j" / "stages").mkdir(parents=True)
    env = runner.child_env(tmp_path / "app", tmp_path / "data", tmp_path / "models", None, cpu=True)
    helper = runner.Background([sys.executable, "-I", str(FAKE), "stdin-a", "j"], env)
    assert helper.proc.poll() is None
    helper.stop()
    assert helper.proc.returncode == 0


def test_the_computer_is_kept_awake_while_a_job_runs_and_released_after() -> None:
    if sys.platform == "win32":
        import ctypes

        kernel = ctypes.WinDLL("kernel32")
        kernel.SetThreadExecutionState.restype = ctypes.c_uint32
        kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
        continuous, system = 0x80000000, 0x00000001
        with awake.KeepAwake():
            # Setting the same state again returns the state the job set.
            assert kernel.SetThreadExecutionState(continuous | system) == continuous | system
        assert kernel.SetThreadExecutionState(continuous) == continuous
    else:
        # Best effort: the inhibitor starts when systemd-inhibit exists, and is always ended afterwards.
        with awake.KeepAwake() as keeper:
            inhibitor = keeper._inhibitor
            assert (inhibitor is not None) == awake.INHIBIT.is_file()
        assert keeper._inhibitor is None
        assert inhibitor is None or inhibitor.proc.poll() is not None
