"""`fluentfolio analyze --json` prints one ASCII envelope, also when it refuses the job."""

import json
import os
import subprocess
import sys
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fluentfolio.core.errors import catalogue
from fluentfolio.models.catalogue import components

KEYS = {"schema", "command", "ok", "exit_code", "version", "data", "warnings", "error"}
Run = Callable[..., subprocess.CompletedProcess[bytes]]
Read = Callable[[subprocess.CompletedProcess[bytes]], Any]


def catalogue_ids() -> list[str]:
    return list(components())


def silence(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))
    return path


def test_a_refused_option_gives_an_error_envelope_and_no_job(tmp_path: Path) -> None:
    clip = silence(tmp_path / "\u00fcbung.wav", 1)
    done = subprocess.run(
        [sys.executable, "-m", "fluentfolio", "analyze", str(clip), "--mode", "solo", "--swap-style", "once",
         "--json"],
        cwd=tmp_path, capture_output=True, timeout=120, check=False,
    )  # fmt: skip
    expected = catalogue()["E_INPUT_OPTION"].exit_code
    assert done.returncode == expected != 0, done.stderr
    assert done.stdout.isascii() and done.stdout.count(b"\n") == 1
    envelope = json.loads(done.stdout)
    assert set(envelope) == KEYS
    assert envelope["schema"] == "fluentfolio.cli/1" and envelope["command"] == "analyze"
    assert envelope["ok"] is False and envelope["exit_code"] == expected and envelope["data"] is None
    assert envelope["error"]["code"] == "E_INPUT_OPTION" and "solo" in envelope["error"]["message"]
    home = Path(os.environ["FLUENTFOLIO_HOME"])
    assert not home.exists() or not any(home.rglob("job.json"))


@pytest.mark.parametrize(("seconds", "code"), [(0.5, "E_INPUT_TOO_SHORT"), (12, "E_INPUT_SILENT")])
def test_a_recording_the_analysis_refuses_gives_its_own_code_and_leaves_no_job(
    tmp_path: Path, seconds: float, code: str
) -> None:
    clip = silence(tmp_path / "quiet.wav", seconds)
    done = subprocess.run(
        [sys.executable, "-m", "fluentfolio", "analyze", str(clip), "--mode", "solo", "--json"],
        cwd=tmp_path, capture_output=True, timeout=120, check=False,
    )  # fmt: skip
    assert done.returncode == catalogue()[code].exit_code, done.stderr
    assert b"Traceback" not in done.stderr and done.stdout.count(b"\n") == 1
    envelope = json.loads(done.stdout)
    assert envelope["error"]["code"] == code and envelope["error"]["retryable"] is False
    assert envelope["data"] is None
    # Neither the job nor its copy and decode of the recording stay behind.
    home = Path(os.environ["FLUENTFOLIO_HOME"])
    assert not any(home.rglob("job.json")) and not any(home.rglob("*.wav"))


# The solo steps, in run order, without the two that are off by default: openSMILE waits for consent and
# the audio file for an AI chat is switched off in Settings.
SOLO_STEPS = [
    "media", "vad", "language", "asr", "speakers", "words", "structure", "candidate_audio", "phones",
    "alignment", "praat", "swiftf0", "nlp", "measures", "evidence", "export",
]  # fmt: skip


def settings(text: str) -> None:
    home = Path(os.environ["FLUENTFOLIO_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "settings.toml").write_text(text, encoding="utf-8")


def no_job() -> bool:
    return not (Path(os.environ["FLUENTFOLIO_HOME"]) / "jobs").exists()


def test_a_dry_run_gives_the_length_estimate_and_steps_and_starts_nothing(
    tmp_path: Path, cli: Run, envelope: Read
) -> None:
    clip = silence(tmp_path / "quiet.wav", 30)
    done = cli("analyze", clip, "--mode", "solo", "--dry-run", "--json")
    found = envelope(done)
    assert done.returncode == 0, done.stderr
    data = found["data"]
    assert data["probe"] == [{"container": "wav", "codec": "pcm_s16le", "sample_rate": 16000, "channels": 1,
                              "tracks": 1, "lossy": False, "duration_ms": 30_000}]  # fmt: skip
    # By hand from the step times: 12.75 s fixed plus 12.71 s per minute, solo on the CPU by default.
    assert (data["audio_duration_ms"], data["device"], data["estimate_ms"]) == (30_000, "cpu", 19_105)
    assert data["will_run"] == SOLO_STEPS and data["needs"] == []
    assert set(data["missing"]) <= set(catalogue_ids())
    codes = [w["code"] for w in found["warnings"]]
    # W_FREE_RAM comes too while this PC has less memory free than a step needs; its own test sets the free
    # memory (test_job_commands), as this run of the command line cannot.
    assert codes[0] == "W_INPUT_SHORT" and set(codes[1:]) <= {"W_FREE_RAM"}
    assert no_job()
    plain = cli("analyze", clip, "--mode", "solo", "--dry-run")
    assert plain.returncode == 0 and b"Estimate: about 1 min on the CPU" in plain.stdout
    refused = cli("analyze", clip, "--mode", "solo", "--dry-run", "--detach", "--json")
    assert refused.returncode == 2 and envelope(refused)["error"]["code"] == "E_USAGE"


def test_a_dry_run_names_the_decisions_a_person_must_make(tmp_path: Path, cli: Run, envelope: Read) -> None:
    settings("[limits]\nconfirm_min = 1\n")
    clip = silence(tmp_path / "quiet.wav", 75)
    asked = envelope(cli("analyze", clip, "--mode", "interview", "--dry-run", "--json"))
    assert asked["data"]["needs"] == ["partner_consent", "confirm_long"]
    given = cli(
        "analyze", clip, "--mode", "interview", "--partner-consent", "--confirm-long", "--dry-run", "--json"
    )
    assert envelope(given)["data"]["needs"] == [] and no_job()


def test_a_long_recording_waits_for_confirmation_and_shows_its_estimate(
    tmp_path: Path, cli: Run, envelope: Read
) -> None:
    settings("[limits]\nconfirm_min = 1\n")
    clip = silence(tmp_path / "quiet.wav", 75)
    refused = cli("analyze", clip, "--mode", "solo", "--json")
    found = envelope(refused)
    assert refused.returncode == 5 and found["error"]["code"] == "E_CONFIRM_LONG"
    # 75 s is 1.25 min; by hand 12.75 + 12.71 * 1.25 = 28.64 s, under half a minute, still "about 1 min".
    assert found["error"]["message"].endswith("(1.2 min, about 1 min to analyse)") and no_job()
    # Confirmed, the analysis starts; the silent recording then fails its own check, which shows it ran.
    confirmed = cli("analyze", clip, "--mode", "solo", "--confirm-long", "--json", timeout=180)
    assert confirmed.returncode == 4 and envelope(confirmed)["error"]["code"] == "E_INPUT_SILENT"


def test_a_light_run_is_only_for_recordings_over_the_light_run_length(
    tmp_path: Path, cli: Run, envelope: Read
) -> None:
    settings("[limits]\nconfirm_min = 1\nlong_min = 1\n")
    long, short = silence(tmp_path / "long.wav", 75), silence(tmp_path / "short.wav", 45)
    full = envelope(cli("analyze", long, "--mode", "solo", "--confirm-long", "--dry-run", "--json"))["data"]
    light = envelope(cli("analyze", long, "--mode", "solo", "--light-run", "--dry-run", "--json"))["data"]
    # The light run counts as the confirmation, and leaves out the phone model (MFA and openSMILE are off
    # already with the default settings).
    assert light["needs"] == [] and [s for s in full["will_run"] if s not in light["will_run"]] == ["phones"]
    assert light["estimate_ms"] < full["estimate_ms"]
    refused = cli("analyze", short, "--mode", "solo", "--light-run", "--json")
    found = envelope(refused)
    assert refused.returncode == 4 and found["error"]["code"] == "E_INPUT_OPTION"
    assert "over 1 min" in found["error"]["message"] and no_job()
    ran = cli("analyze", long, "--mode", "solo", "--light-run", "--json", timeout=180)
    assert ran.returncode == 4 and envelope(ran)["error"]["code"] == "E_INPUT_SILENT"


def test_detach_returns_once_a_runner_has_the_job_and_status_follows_it(
    tmp_path: Path, cli: Run, envelope: Read
) -> None:
    clip = silence(tmp_path / "quiet.wav", 12)
    started = cli("analyze", clip, "--mode", "solo", "--detach", "--json")
    found = envelope(started)
    # Usually the runner is still working; on a fast computer it may already have finished.
    assert (started.returncode, found["data"]["state"]) in ((0, "running"), (7, "failed")), started.stderr
    assert found["data"]["audio_duration_ms"] == 12_000
    done = cli("status", found["data"]["job_id"], "--wait", "--timeout", "120", "--json", timeout=180)
    result = envelope(done)
    # The silent recording fails its voice step in the runner; the job stays, so its state can be read.
    assert done.returncode == 7 and result["data"]["state"] == "failed"
    assert result["error"]["code"] == "E_INPUT_SILENT"
