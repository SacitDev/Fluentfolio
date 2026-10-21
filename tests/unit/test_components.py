"""Installing and removing parts, and the learner's own consent: what each one needs, fetches and leaves."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from huggingface_hub import constants

from fluentfolio.alignment import mfa
from fluentfolio.analyzers.opensmile import consented
from fluentfolio.app import facade
from fluentfolio.core.errors import AppError
from fluentfolio.jobs.loop import Roots
from fluentfolio.models import catalogue, download
from fluentfolio.models.catalogue import SYSTEM, Component, ModelFile
from fluentfolio.setup import components as installer
from fluentfolio.setup import mfa_env
from tests.support.fake_https import Server, local_server
from tests.support.in_process import Command

A, B, C = b"model weights " * 700, b"dictionary " * 90, b"\x7fELF tool " * 50


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def url_file(name: str, data: bytes, **changes: Any) -> ModelFile:
    url = f"https://github.com/owner/repo/releases/download/v1/{name}"
    return ModelFile.model_validate(
        {"path": name, "sha256": sha(data), "size": len(data), "url": url, **changes}
    )


def part(files: list[ModelFile], **changes: Any) -> Component:
    base: dict[str, Any] = {
        "id": "test-part", "title": "Test part", "note": "", "kind": "acoustic", "licence": "MIT",
        "licence_class": "permissive", "attribution": "test", "host": "github.com", "device": "cpu",
        "install": "download", "default_on": True, "gated": False, "requires_consent": False, "needs": [],
        "stages": ["praat"], "os": ["windows", "linux"], "file": files,
    }  # fmt: skip
    return Component.model_validate({**base, **changes})


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Server]:
    with local_server(monkeypatch) as served:
        yield served


def serve(server: Server, *pairs: tuple[ModelFile, bytes]) -> None:
    for f, data in pairs:
        server.bodies[urlparse(str(f.url)).path] = [data]


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Roots:
    """A fresh app, data and models folder, and a token file of the test's own."""
    monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(tmp_path / "app"))
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(tmp_path / "hf" / "token"))
    monkeypatch.setattr(constants, "HF_STORED_TOKENS_PATH", str(tmp_path / "hf" / "stored_tokens"))
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        monkeypatch.delenv(name, raising=False)
    found = Roots(tmp_path / "app", tmp_path / "data", tmp_path / "app" / "models")
    return found


def test_a_part_comes_after_the_parts_it_needs_and_each_part_once() -> None:
    # Parakeet needs the voice detector; the CTC aligner needs the phone model and espeak-ng, and the phone
    # model needs espeak-ng too (models.toml), worked out by hand.
    found = [c.id for c in installer.with_needs(["parakeet-tdt-0.6b-v2", "ctc-aligner", "phone-model"])]
    assert found == ["silero-vad", "parakeet-tdt-0.6b-v2", "espeak-ng", "phone-model", "ctc-aligner"]
    # Canary is in the catalogue for the speech-model comparison only, so a learner cannot name it.
    for named in (["praat", "no-such-part"], ["canary-1b-v2"]):
        with pytest.raises(AppError) as unknown:
            installer.with_needs(named)
        assert unknown.value.code == "E_INPUT_OPTION" and named[-1] not in str(unknown.value)  # never echoed


def test_a_download_is_fetched_whole_and_heard_of_file_by_file(server: Server, roots: Roots) -> None:
    files = [url_file("a.bin", A), url_file("b.bin", B)]
    serve(server, (files[0], A), (files[1], B))
    c = part(files)
    assert installer.missing_bytes(c, roots) == len(A) + len(B)
    heard: list[str] = []
    installer.install(c, roots, lambda f: heard.append(f.path))
    assert heard == ["a.bin", "b.bin"] and installer.missing_bytes(c, roots) == 0
    assert download.verify(c, roots.models) == []
    installer.remove(c, roots)
    assert installer.missing_bytes(c, roots) == len(A) + len(B)


@pytest.fixture
def env_part(server: Server, roots: Roots, monkeypatch: pytest.MonkeyPatch) -> tuple[Component, list[Path]]:
    """MFA's kind of part: micromamba for this system and two model files; making the env is recorded."""
    tool = "micromamba.exe" if sys.platform == "win32" else "micromamba"
    files = [url_file(tool, C, os=SYSTEM), url_file("english_us_arpa.dict", B), url_file("g2p.zip", A)]
    serve(server, (files[0], C), (files[1], B), (files[2], A))
    made: list[Path] = []

    def make(app: Path) -> str:
        made.append(app)
        mfa.python(mfa.env_dir(app)).parent.mkdir(parents=True, exist_ok=True)
        mfa.python(mfa.env_dir(app)).write_bytes(b"")
        return "3.4.3"

    monkeypatch.setattr(mfa_env, "create", make)
    monkeypatch.setattr(mfa_env, "installed_version", lambda app: "3.4.3" if made else None)
    return part(files, id="mfa-like", host="conda-forge", install="env", kind="aligner"), made


def test_the_env_files_go_where_the_mfa_step_reads_them_and_the_env_is_made_once(
    env_part: tuple[Component, list[Path]], roots: Roots
) -> None:
    c, made = env_part
    installer.install(c, roots)
    assert mfa_env.micromamba(roots.app).read_bytes() == C
    assert (roots.models / "mfa" / "english_us_arpa.dict").read_bytes() == B
    assert (roots.models / "mfa" / "g2p.zip").read_bytes() == A
    if sys.platform != "win32":
        assert mfa_env.micromamba(roots.app).stat().st_mode & 0o111  # the tool can run
    assert made == [roots.app] and installer.missing_bytes(c, roots) == 0
    installer.install(c, roots)  # whole files and a working env: nothing is fetched or made again
    assert made == [roots.app]
    installer.remove(c, roots)
    assert not mfa.env_dir(roots.app).exists() and not mfa_env.micromamba(roots.app).exists()
    assert installer.missing_bytes(c, roots) == len(A) + len(B) + len(C)


def test_a_damaged_env_file_is_fetched_twice_then_refused(
    env_part: tuple[Component, list[Path]], roots: Roots, server: Server
) -> None:
    c, made = env_part
    server.bodies[urlparse(str(c.file[1].url)).path] = [B[:-1] + b"?"]
    with pytest.raises(AppError) as refused:
        installer.install(c, roots)
    assert refused.value.code == "E_HASH_MISMATCH" and made == []
    assert not (roots.models / "mfa" / "english_us_arpa.dict").exists()


def test_opensmile_needs_consent_then_the_installer(roots: Roots, monkeypatch: pytest.MonkeyPatch) -> None:
    smile = catalogue.get("opensmile")
    with pytest.raises(AppError) as waiting:
        installer.install(smile, roots)
    assert waiting.value.code == "E_OPENSMILE_CONSENT_MISSING"
    record = installer.consent(roots.data, "opensmile", "cli-tty")
    assert record["source"] == "cli-tty" and record["utc"].endswith("Z") and consented(roots.data)
    monkeypatch.setattr(installer.util, "find_spec", lambda name: None)  # the nc extra is not installed
    with pytest.raises(AppError) as needs:
        installer.install(smile, roots)
    assert needs.value.code == "E_INSTALLER_NEEDED" and needs.value.entry.exit_code == 3


@pytest.mark.parametrize(("component_id", "source"), [("opensmile", "cli-flag"), ("opensmile", "yes"),
                                                      ("praat", "cli-tty")])  # fmt: skip
def test_consent_comes_only_from_the_learners_own_act_and_only_for_a_part_that_asks(
    roots: Roots, component_id: str, source: str
) -> None:
    with pytest.raises(ValueError):
        installer.consent(roots.data, component_id, source)
    assert not (roots.data / installer.CONSENT_FILE).exists() and not consented(roots.data)


def test_a_part_for_another_system_is_not_available(roots: Roots) -> None:
    other = "linux" if SYSTEM == "windows" else "windows"
    with pytest.raises(AppError) as refused:
        installer.install(part([], os=[other]), roots)
    assert refused.value.code == "E_COMPONENT_UNAVAILABLE"


def test_install_all_plans_each_part_in_use_that_is_missing(roots: Roots) -> None:
    # A fresh install with the default settings, by hand from models.toml: the default speech model, the
    # language check, the no-account speaker separation, the phone model, the pronunciation dictionary and
    # the grammar model are downloads; pyannote waits for a token, openSMILE for consent, MFA is off, the
    # rest are in the app.
    planned = facade.install_plan([], every=True)
    expected = {"parakeet-tdt-0.6b-v2", "whisper-tiny", "sherpa-onnx-diarizer", "phone-model",
                "pronunciation-dictionary", "spacy-en-core-web-sm"}  # fmt: skip
    assert {c.id for c in planned.parts} == expected
    assert planned.download_bytes == sum(
        f.size for i in expected for f in catalogue.get(i).files_for(SYSTEM)
    )  # fmt: skip
    named = facade.install_plan(["mfa"])
    assert [c.id for c in named.parts] == ["mfa"] and named.download_bytes == sum(
        f.size for f in catalogue.get("mfa").files_for(SYSTEM)
    )  # fmt: skip


def test_remove_deletes_nothing_when_one_part_belongs_to_the_app(server: Server, roots: Roots) -> None:
    phones = catalogue.get("phone-model")
    folder = download.folder(phones, roots.models)
    folder.mkdir(parents=True)
    (folder / "kept.bin").write_bytes(A)
    with pytest.raises(AppError) as refused:
        facade.remove(["phone-model", "praat"])
    assert refused.value.code == "E_INPUT_OPTION" and "praat" in str(refused.value)
    assert (folder / "kept.bin").read_bytes() == A
    facade.remove(["phone-model"])
    assert not folder.exists()


def test_installing_an_opt_in_part_switches_it_on_and_deleting_it_switches_it_off(
    roots: Roots, monkeypatch: pytest.MonkeyPatch
) -> None:
    # MFA did not pass its word-timing check on this system (models.toml), so it stays off until chosen.
    monkeypatch.setattr(installer, "install", lambda c, r, *heard: None)
    monkeypatch.setattr(installer, "remove", lambda c, r: None)
    aligner, separation = catalogue.get("mfa"), catalogue.get("sherpa-onnx-diarizer")

    def chosen() -> dict[str, bool]:
        return facade.settings_show()[0].settings.components

    facade.install(facade.InstallPlan([separation], 0))
    assert not (roots.data / "settings.toml").exists()  # a part that is on anyway records no choice
    assert not catalogue.switched_on(chosen(), "mfa")
    facade.install(facade.InstallPlan([aligner], 0))
    assert chosen() == {"mfa": True} and catalogue.switched_on(chosen(), "mfa")
    facade.remove(["mfa"])
    assert chosen() == {"mfa": False} and not catalogue.switched_on(chosen(), "mfa")

    def broken(c: Component, r: Roots, *heard: Any) -> None:
        raise AppError("E_HASH_MISMATCH", c.id)

    monkeypatch.setattr(installer, "install", broken)
    done = facade.install(facade.InstallPlan([aligner], 0))
    assert done["mfa"] is not None and chosen() == {"mfa": False}  # a failed install chooses nothing


def test_install_and_remove_wait_for_a_running_job(roots: Roots) -> None:
    from fluentfolio.jobs.locks import work_lock

    roots.data.mkdir(parents=True)
    with work_lock(roots.data):
        for call in (
            lambda: facade.install(facade.InstallPlan([], 0)),
            lambda: facade.remove(["phone-model"]),
        ):
            with pytest.raises(AppError) as busy:
                call()
            assert busy.value.code == "E_BUSY"


def test_the_consent_record_names_its_source_and_time_only(roots: Roots) -> None:
    installer.consent(roots.data, "opensmile", "ui")
    record = json.loads((roots.data / installer.CONSENT_FILE).read_text(encoding="utf-8"))
    assert set(record) == {"opensmile"} and set(record["opensmile"]) == {"utc", "source"}


@pytest.fixture
def command(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> Command:
    return Command(capsys, monkeypatch)


@pytest.mark.parametrize("typed", ["i accept", "I ACCEPTED", "yes", "", "I  ACCEPT"])
def test_consent_needs_the_exact_words_typed_at_a_terminal(
    roots: Roots, command: Command, typed: str
) -> None:
    exit_code, found, printed = command(
        "components", "consent", "opensmile", stdin=typed + "\n", terminal=True
    )
    assert (exit_code, found["error"]["code"]) == (5, "E_OPENSMILE_CONSENT_MISSING")
    assert "non-commercial use only" in printed and not consented(roots.data)


def test_consent_typed_at_a_terminal_is_recorded_once(roots: Roots, command: Command) -> None:
    exit_code, found, printed = command(
        "components", "consent", "opensmile", stdin="I ACCEPT\n", terminal=True
    )
    assert exit_code == 0 and found["data"]["recorded"] is True and consented(roots.data)
    assert found["data"]["consent"]["source"] == "cli-tty" and "audEERING" in printed
    again = command("components", "consent", "opensmile", terminal=True)[1]["data"]
    assert again["recorded"] is False and again["consent"] == found["data"]["consent"]


def test_a_no_at_the_size_or_delete_question_changes_nothing(
    server: Server, roots: Roots, command: Command
) -> None:
    exit_code, found, _ = command("components", "install", "phone-model", stdin="n\n", terminal=True)
    assert exit_code == 0 and found["data"]["parts"] == [] and server.seen == []
    assert found["data"]["download_bytes"] == sum(f.size for f in catalogue.get("phone-model").file)
    folder = download.folder(catalogue.get("phone-model"), roots.models)
    folder.mkdir(parents=True)
    exit_code, found, _ = command("components", "remove", "phone-model", stdin="n\n", terminal=True)
    assert exit_code == 0 and found["data"] == {"parts": []} and folder.exists()
    exit_code, found, _ = command("components", "remove", "phone-model", stdin="y\n", terminal=True)
    assert found["data"] == {"parts": [{"id": "phone-model", "status": "needs-download", "code": None}]}
    assert not folder.exists()


def test_a_yes_downloads_and_a_failed_part_gives_its_code_and_exit_code(
    server: Server, roots: Roots, command: Command, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download.time, "sleep", lambda s: None)  # the waits between network tries
    exit_code, found, printed = command("components", "install", "sherpa-onnx-diarizer", stdin="y\n",
                                        terminal=True)  # fmt: skip
    files = catalogue.get("sherpa-onnx-diarizer").files_for(SYSTEM)
    assert (exit_code, found["error"]["code"]) == (3, "E_NETWORK")  # the local server has no such file
    assert found["data"]["parts"] == [{"id": "sherpa-onnx-diarizer", "status": "needs-download",
                                       "code": "E_NETWORK"}]  # fmt: skip
    assert len(server.seen) == 3 and files[0].path in printed  # three tries of its first file, then stop
