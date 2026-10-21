"""Every program a full interview run starts: its command line, its environment and no network."""

import os
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fluentfolio.app import facade
from fluentfolio.core.enums import Status
from fluentfolio.core.paths import app_root, data_root, secure_umask
from fluentfolio.jobs import runner
from fluentfolio.jobs.store import JobState
from fluentfolio.setup import device

pytestmark = pytest.mark.slow
PLANTED_TOKEN = "hf_" + "PlantedNotReal" * 3  # made up; it must reach no child
SHELLS = {"cmd", "powershell", "pwsh", "sh", "bash", "wsl"}
SCRIPTS = {".bat", ".cmd", ".ps1", ".sh", ".vbs"}
LONGEST_COMMAND = 1024  # characters, as Windows would see the whole command line
# Inside the data folder; with a data folder path of up to 50 characters, under Windows' old 200-260 limit.
DEEPEST_INSIDE = 150
STAGE = ["-I", "-X", "utf8", "-m", "fluentfolio.jobs.stage"]
OFFLINE = Path(__file__).parents[1] / "support" / "offline_stage.py"


@pytest.fixture
def app_umask() -> Iterator[None]:
    """The umask the app's entry points set, starting from a login shell's usual one; restored afterwards."""
    old = os.umask(0o022)
    secure_umask()
    yield
    os.umask(old)


def deepest(folder: Path) -> tuple[int, str]:
    """The longest path under a folder, counted from inside it."""
    found = [
        (len(p.relative_to(folder).as_posix()), p.relative_to(folder).as_posix()) for p in folder.rglob("*")
    ]
    return max(found, default=(0, ""))


def test_a_full_interview_run_starts_plain_programs_offline_with_no_token_or_pythonpath(
    built: Callable[[str], tuple[Path, dict[str, Any]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    app_umask: None,
) -> None:
    seen: list[tuple[list[str], dict[str, str]]] = []

    def recording(args: list[str], *rest: Any, **options: Any) -> "subprocess.Popen[bytes]":
        argv = [str(a) for a in args]
        seen.append((argv, dict(options.get("env") or {})))
        if argv[1 : len(STAGE) + 1] == STAGE:  # the same child, with every connection refused inside it
            argv = [argv[0], "-I", "-X", "utf8", str(OFFLINE), *argv[len(STAGE) + 1 :]]
        return subprocess.Popen(argv, *rest, **options)  # noqa: S603  # the runner's own checked call

    # Only the runner's starts are recorded: libraries that call subprocess for themselves are not ours.
    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(**{**vars(subprocess), "Popen": recording}))
    monkeypatch.setenv("HF_TOKEN", PLANTED_TOKEN)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "planted"))
    folder, _ = built("CS-INT")
    done = facade.analyze(folder / "session.wav", "interview", partner_consent="cli-flag")

    assert done.record.state in (JobState.COMPLETED, JobState.COMPLETED_WITH_GAPS), done.record.stages
    failed = [(s.id, s.error_code) for s in done.record.stages if s.status is Status.FAILED or s.fallback]
    assert failed == []  # offline, every step still ran as itself
    stages = {argv[len(STAGE) + 1] for argv, _ in seen if argv[1 : len(STAGE) + 1] == STAGE}
    # The steps that always run as a child in an interview; words, structure and measures run inline.
    assert {"media", "vad", "asr", "praat", "nlp"} <= stages, sorted(stages)
    assert any(s.startswith("speakers-") for s in stages), sorted(stages)
    for argv, env in seen:
        program = Path(argv[0])
        assert program.is_absolute(), argv[0]
        assert program.suffix.lower() not in SCRIPTS, argv[0]
        assert program.stem.lower() not in SHELLS, argv[0]
        assert "-c" not in argv, argv[:3]
        assert len(subprocess.list2cmdline(argv)) <= LONGEST_COMMAND, argv[:3]
        assert "HF_TOKEN" not in env and "PYTHONPATH" not in env, argv[:3]
        assert not any(PLANTED_TOKEN in value for value in env.values()), argv[:3]
        if program == device.nvidia_smi():  # the driver's own tool reads no model: it gets two names only
            assert set(env) <= {"SYSTEMROOT", "PROGRAMFILES"}, sorted(env)
            continue
        assert env.get("HF_HUB_OFFLINE") == "1", argv[:3]
    for root in (data_root(), app_root() / "tmp"):
        length, where = deepest(root)
        assert length <= DEEPEST_INSIDE, where
    if sys.platform != "win32":  # Windows keeps others out with the folder's access list (test_paths)
        shared = [p for p in done.job.rglob("*") if stat.S_IMODE(p.stat().st_mode) & 0o077]
        assert shared == [], shared[:5]  # the whole job tree, written by the app and by every child
