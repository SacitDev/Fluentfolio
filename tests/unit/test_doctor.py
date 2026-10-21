"""Doctor's checks and the parts list: statuses, limits at their edges, no network and no heavy library."""

import json
import os
import socket
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from types import ModuleType

import pytest
from huggingface_hub import constants

from fluentfolio import __version__
from fluentfolio.core.files import write_json
from fluentfolio.doctor import checks
from fluentfolio.doctor import report as diagnostic
from fluentfolio.jobs.loop import Roots
from fluentfolio.models import catalogue, download
from fluentfolio.models.catalogue import SYSTEM
from fluentfolio.settings.model import Settings

GB = 2**30
CPU_ONLY = {"requested": "auto", "resolved": "cpu", "reason": "no NVIDIA graphics card was found."}
CONSENT = {"opensmile": {"utc": "2026-10-15T00:00:00Z", "source": "cli-tty"}}
CANARY = "quandrix"  # a made-up word, put where the report must not look
USER = "Brindlewick"  # a made-up user name
Usage = namedtuple("Usage", "total used free")


@pytest.fixture(autouse=True)
def private_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A token file of the test's own, so the checks never see a token saved on this computer."""
    folder = tmp_path / "hf"
    folder.mkdir()
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(folder / "token"))
    monkeypatch.setattr(constants, "HF_STORED_TOKENS_PATH", str(folder / "stored_tokens"))
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        monkeypatch.delenv(name, raising=False)
    return folder / "token"


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Roots:
    found = Roots(tmp_path / "app", tmp_path / "data", tmp_path / "models")
    for folder in (found.app, found.data, found.models):
        folder.mkdir()
    monkeypatch.setattr(checks.gpu, "resolve", lambda *_, **__: CPU_ONLY)
    return found


def fake_token() -> str:
    return "hf_" + "Q7" * 17  # made here, so no real token is ever written down


def by_id(report: checks.Report) -> dict[str, checks.Check]:
    return {c.id: c for c in report.checks}


def test_an_empty_models_folder_blocks_on_the_chosen_speech_model(roots: Roots) -> None:
    report = checks.run(Settings(), roots, ram_bytes=16 * GB)
    rows = by_id(report)
    assert report.blocking
    asr = rows["parakeet-tdt-0.6b-v2"]
    assert (asr.level, asr.code, asr.blocking) == ("problem", "E_MODEL_MISSING", True)
    assert asr.fix == "Run `fluentfolio components install parakeet-tdt-0.6b-v2`."
    # A part that is on but only adds a measure warns without blocking.
    language = rows["whisper-tiny"]
    assert (language.level, language.code, language.blocking) == ("warning", "E_MODEL_MISSING", False)
    # The other speech models are choices, not problems. MFA did not pass its word-timing check on either
    # system (models.toml), so it is off until the learner installs it, and Doctor says that, not Settings.
    assert {"parakeet-tdt-0.6b-v3", "whisper-large-v3-turbo", "canary-1b-v2"}.isdisjoint(rows)
    assert rows["mfa"].level == "ok" and rows["mfa"].message == (
        "Montreal Forced Aligner: off. It did not pass its accuracy check on this system, so it runs only "
        "once you install it."
    )
    offline = rows["offline"]
    assert offline.level == "warning" and offline.code == "W_NOT_OFFLINE"
    # Parts by their titles, never their IDs (models.toml).
    assert "Parakeet TDT 0.6B v2" in offline.message and "Parakeet TDT 0.6B v3" not in offline.message
    assert "pyannote community-1" in offline.message  # it waits for a token, so it is not here either
    assert "Phone recogniser" in offline.message
    assert all(i not in offline.message for i in catalogue.components() if "-" in i)
    assert rows["network"].message.startswith("This check uses no network.")
    assert rows["installer"].code == "W_NOT_INSTALLED"
    assert rows["token"].code == "E_TOKEN_MISSING" and not rows["token"].blocking


def test_a_part_the_learner_switched_off_is_named_as_switched_off_in_settings(roots: Roots) -> None:
    chosen = Settings(components={"mfa": False, "swiftf0": False})
    rows = by_id(checks.run(chosen, roots, ram_bytes=16 * GB))
    assert rows["mfa"].message == "Montreal Forced Aligner: switched off in Settings."
    assert rows["swiftf0"].message == "SwiftF0 (pitch): switched off in Settings."
    assert rows["praat"].message == "Praat (pitch and voice): ready."  # on, and built into the app


def test_with_every_part_in_place_nothing_blocks_and_built_in_parts_see_their_needs(
    roots: Roots, private_token: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every file present at its recorded size, a token saved and the installer's record of this version.
    monkeypatch.setattr(download, "require", lambda c, models: models)
    private_token.write_text(fake_token(), encoding="utf-8")
    write_json(roots.app / "state.json", {"version": __version__})
    report = checks.run(Settings(), roots, ram_bytes=16 * GB)
    rows = by_id(report)
    assert not report.blocking, [c for c in report.checks if c.blocking]
    assert rows["parakeet-tdt-0.6b-v2"].message.endswith("ready.")
    # The aligner is built in and listed before the phone model and eSpeak NG it needs.
    assert rows["ctc-aligner"].level == "ok" and rows["ctc-aligner"].message.endswith("ready.")
    assert rows["offline"].level == "ok" and rows["offline"].code is None
    assert rows["installer"].level == "ok" and rows["token"].level == "ok"
    listed = {r.id: r for r in checks.components(Settings(), roots)}
    assert listed["ctc-aligner"].status == "ready"
    # Its needs decide: with the phone model off this computer, the aligner cannot run.
    real = checks.component_status
    statuses = {"phone-model": "needs-download", "espeak-ng": "ready"}
    found = real(catalogue.get("ctc-aligner"), roots, {}, True, statuses)
    assert found == ("not-available", "it needs Phone recogniser")  # its title, not its ID


def test_each_status_has_its_own_cause(roots: Roots, private_token: Path) -> None:
    status = checks.component_status
    parakeet = catalogue.get("parakeet-tdt-0.6b-v2")
    assert status(parakeet, roots, {}, False, {})[0] == "needs-download"
    # A file cut short reads as damaged, from its size alone.
    first = parakeet.files_for(SYSTEM)[0]
    target = download.local_path(parakeet, first, roots.models)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * (first.size - 1))
    assert status(parakeet, roots, {}, False, {}) == (
        "needs-download", "a file is damaged or cut short: download this model again"
    )  # fmt: skip
    pyannote = catalogue.get("pyannote-community-1")
    assert status(pyannote, roots, {}, False, {})[0] == "needs-token"
    assert status(pyannote, roots, {}, True, {})[0] == "needs-download"
    opensmile = catalogue.get("opensmile")
    assert status(opensmile, roots, {}, False, {})[0] == "needs-consent"
    assert checks.consent_record(roots.data, "opensmile") is None
    write_json(roots.data / "consent.json", CONSENT)
    assert status(opensmile, roots, {}, False, {})[0] != "needs-consent"
    assert checks.consent_record(roots.data, "opensmile") == CONSENT["opensmile"]
    other = "linux" if SYSTEM == "windows" else "windows"
    elsewhere = catalogue.get("silero-vad").model_copy(update={"os": [other]})
    assert status(elsewhere, roots, {}, False, {}) == (
        "not-available",
        f"not available on {SYSTEM.capitalize()}",
    )
    failed = {"selftest": {"components": {"silero-vad": {"status": "failed", "reason": "no speech found"}}}}
    vad = catalogue.get("silero-vad")
    assert status(vad, roots, {}, False, {}) == ("ready", "")
    assert status(vad, roots, failed, False, {}) == ("self-test-failed", "no speech found")


def test_a_failed_self_test_of_a_needed_part_blocks(roots: Roots) -> None:
    failed = {"selftest": {"components": {"silero-vad": {"status": "failed", "reason": "no speech found"}}}}
    write_json(roots.app / "state.json", {"version": __version__, **failed})
    vad = by_id(checks.run(Settings(), roots, ram_bytes=16 * GB))["silero-vad"]
    assert (vad.level, vad.code, vad.blocking) == ("problem", "E_SELFTEST_FAILED", True)
    assert vad.message == "Silero VAD v6: no speech found."


@pytest.mark.parametrize(
    ("state", "level", "says"),
    [
        ({}, "warning", "not set up by the installer"),
        ({"selftest": {"passed": True}}, "warning", "not set up by the installer"),  # a check, no install
        ({"version": "0.0.9"}, "warning", "version 0.0.9, but this is"),
        ({"version": __version__}, "ok", f"version {__version__}."),
    ],
)  # fmt: skip
def test_the_install_row_names_the_installed_version_only_when_the_installer_wrote_one(
    state: dict[str, object], level: str, says: str
) -> None:
    (row,) = checks._install(state)
    assert row.level == level and says in row.message and "None" not in row.message


def test_a_speech_model_that_cannot_be_chosen_blocks(roots: Roots) -> None:
    settings = Settings.model_validate({"asr": {"model": "no-such-model"}})
    asr = by_id(checks.run(settings, roots, ram_bytes=16 * GB))["asr"]
    assert (asr.level, asr.code, asr.blocking) == ("problem", "E_COMPONENT_UNAVAILABLE", True)


@pytest.mark.parametrize(
    ("ram_bytes", "level"),
    [
        (12 * GB - 1, "warning"), (12 * GB, "ok"), (12 * GB + 1, "ok"),
        (round(7.8 * GB), "warning"),  # what an 8 GB computer reports
        (round(15.8 * GB), "ok"),  # and a 16 GB one
    ],
)  # fmt: skip
def test_memory_under_12_gb_warns_that_analyses_are_slow(ram_bytes: int, level: str) -> None:
    ram = {c.id: c for c in checks._compute(ram_bytes, CPU_ONLY)}["ram"]
    assert ram.level == level and ram.code == ("W_LOW_RAM" if level == "warning" else None)
    assert not ram.blocking


def test_the_gpu_row_says_why_the_cpu_runs() -> None:
    def gpu(where: dict[str, str]) -> checks.Check:
        return {c.id: c for c in checks._compute(16 * GB, where)}["gpu"]

    assert gpu(CPU_ONLY).code == "W_GPU_NOT_USED"
    assert gpu({**CPU_ONLY, "requested": "cpu"}).level == "ok"
    assert gpu({"requested": "auto", "resolved": "gpu", "reason": "the GPU test passed."}).level == "ok"


@pytest.mark.parametrize(
    ("free", "level", "code"),
    [
        (10**9 - 1, "problem", "E_DISK_FULL"), (10**9, "warning", "W_DISK_LOW"),
        (5 * 10**9 - 1, "warning", "W_DISK_LOW"), (5 * 10**9, "ok", None),
    ],
)  # fmt: skip
def test_free_space_limits_at_their_edges(
    free: int, level: str, code: str | None, roots: Roots, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In the drive makers' GB of 10**9 bytes, the unit the Components page uses for the same drive.
    monkeypatch.setattr(checks.shutil, "disk_usage", lambda _: Usage(100 * GB, 100 * GB - free, free))
    disks = [c for c in checks._disks(roots)]
    assert [(c.id, c.level, c.code, c.blocking) for c in disks] == [
        (name, level, code, level == "problem") for name in ("data", "models")
    ]


def test_free_space_reads_as_the_components_page_writes_it(
    roots: Roots, monkeypatch: pytest.MonkeyPatch
) -> None:
    free = 1_100_980_000_000  # 1100.98 GB, which is 1025.4 GiB
    monkeypatch.setattr(checks.shutil, "disk_usage", lambda _: Usage(2 * free, free, free))
    assert checks._disks(roots)[0].message == "1101.0 GB free on the drive of the data folder."


def test_doctor_opens_no_connection_and_looks_up_no_host(
    roots: Roots, monkeypatch: pytest.MonkeyPatch
) -> None:
    tried: list[object] = []

    def refuse(*args: object, **kwargs: object) -> None:
        tried.append(args)
        raise OSError("no network in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    report = checks.run(Settings(), roots)
    checks.components(Settings(), roots)
    assert tried == [] and by_id(report)["network"].level == "ok"


def test_a_token_file_inside_the_app_data_or_a_git_tree_warns(
    roots: Roots, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def place() -> checks.Check | None:
        return by_id(checks.run(Settings(), roots, ram_bytes=16 * GB)).get("token-file")

    assert place() is None
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(roots.data / "token"))
    found = place()
    assert found is not None and found.code == "W_TOKEN_FILE_PLACE" and "the data folder" in found.message
    tree = tmp_path / "project"
    (tree / ".git").mkdir(parents=True)
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(tree / "secrets" / "token"))
    found = place()
    assert found is not None and "a git working tree" in found.message


HEAVY = ["torch", "onnxruntime", "ctranslate2", "faster_whisper", "pyannote", "sherpa_onnx", "transformers",
         "parselmouth", "opensmile", "spacy", "av", "swift_f0", "espeakng_loader"]  # fmt: skip
CHILD = """
import json, sys
from pathlib import Path
from fluentfolio.doctor import checks
from fluentfolio.jobs.loop import Roots
from fluentfolio.settings.model import Settings
base = Path(sys.argv[1])
roots = Roots(base / "app", base / "data", base / "models")
report = checks.run(Settings(), roots)
rows = checks.components(Settings(), roots)
print(json.dumps({"checks": len(report.checks), "rows": len(rows), "loaded": sorted(sys.modules)}))
"""


def test_the_checks_import_no_model_library(tmp_path: Path) -> None:
    for name in ("app", "data", "models"):
        (tmp_path / name).mkdir()
    env = {**os.environ, "HF_TOKEN_PATH": str(tmp_path / "hf" / "token")}
    env.pop("HF_TOKEN", None)
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(tmp_path)], env=env, capture_output=True, text=True, timeout=120,
        check=True,
    )  # fmt: skip
    found = json.loads(done.stdout)
    assert found["checks"] > 10 and found["rows"] == len(catalogue.offered())
    loaded = {m.split(".")[0] for m in found["loaded"]}
    assert loaded.isdisjoint(HEAVY), sorted(loaded & set(HEAVY))


def test_the_report_holds_no_private_text_paths_or_token(
    roots: Roots, synthetic_job: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    spoken = [("I", 1000, 1200, 0), ("like", 1250, 1600, 0), (CANARY, 1650, 2300, 0), ("a", 2350, 2450, 0),
              ("lot.", 2500, 3000, 0)]  # fmt: skip
    job = synthetic_job.make(base, spoken, [(1000, 3000)], 6000, display_name=f"{CANARY}_talk.wav")
    found = synthetic_job.roots(base)
    found.app.mkdir(exist_ok=True)
    found.models.mkdir(exist_ok=True)
    (job / "mfa" / "logs").mkdir(parents=True)
    (job / "mfa" / "logs" / "align.log").write_text(f"aligned {CANARY}\n", encoding="utf-8")
    logs = found.data / "logs"
    logs.mkdir()
    lines = [
        f"job {synthetic_job.JOB_ID} started in {found.data}",
        f'{{"models": {json.dumps(str(found.models))}, "app": "{found.app.as_posix()}"}}',
        f"Authorization: Bearer {fake_token()}",
        f"token {fake_token()} in {Path.home() / 'notes'}",
        f"signed in as {USER}",
    ]
    monkeypatch.setenv("USERNAME", USER)
    monkeypatch.setenv("USER", USER)
    (logs / "app.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = checks.run(Settings(), found, ram_bytes=16 * GB)
    text = diagnostic.build(report, checks.components(Settings(), found), found)
    assert CANARY not in text.lower() and fake_token() not in text
    for path in (found.data, found.models, found.app, Path.home()):
        for form in (str(path), path.as_posix(), json.dumps(str(path))[1:-1]):
            assert form.lower() not in text.lower(), form
    assert f"job {synthetic_job.JOB_ID} started in <DATA>" in text
    assert '{"models": "<MODELS>", "app": "<APP>"}' in text
    assert "token hf_<REDACTED> in ~" in text
    assert "signed in as <user>" in text and USER.lower() not in text.lower()
    # The job's record is there without its file name.
    assert synthetic_job.JOB_ID in text and "display_name" not in text
