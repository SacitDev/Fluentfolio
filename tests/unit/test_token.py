"""The learner's own token: saved only when it works, never over one that works, removed only by its name.

The Hugging Face site is replaced at the HTTP layer, so huggingface_hub's own requests and error handling
run unchanged; the token files live in the test's own folder.
"""

from __future__ import annotations

import configparser
import ssl
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import huggingface_hub
import pytest
from huggingface_hub import _login as hf_login
from huggingface_hub import constants, get_token
from huggingface_hub.utils import get_stored_tokens
from huggingface_hub.utils._http import default_client_factory

from fluentfolio.core import paths
from fluentfolio.core.errors import AppError
from fluentfolio.core.paths import install_state
from fluentfolio.models import catalogue
from fluentfolio.setup import token as hf_token
from tests.support.in_process import Command

GATED_PATH = f"/api/models/{catalogue.get(hf_token.GATED).repo}/auth-check"


def made(tag: str) -> str:
    return "hf_" + tag * (34 // len(tag))  # made here, so no real token is ever written down


GOOD, NO_TERMS, OTHER, REFUSED = made("Gd"), made("Nt"), made("Ot"), made("Rf")
ORGANISATION = "api_org_" + "Og" * 15


@dataclass
class Site:
    """A fake huggingface.co: the accounts it knows, the tokens whose account accepted the terms."""

    names: dict[str, str] = field(default_factory=dict)  # token -> the token's name on the site
    accepted: set[str] = field(default_factory=set)
    seen: list[tuple[str, str]] = field(default_factory=list)  # (path, token) of every request
    fault: Exception | None = None

    def answer(self, request: httpx.Request) -> httpx.Response:
        if self.fault:
            raise self.fault
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        self.seen.append((request.url.path, token))
        if token not in self.names:
            return httpx.Response(401, json={"error": "Invalid credentials in Authorization header"})
        if request.url.path == "/api/whoami-v2":
            access = {"displayName": self.names[token], "role": "read"}
            return httpx.Response(
                200, json={"name": "learner", "type": "user", "auth": {"accessToken": access}}
            )
        if request.url.path == GATED_PATH:
            if token in self.accepted:
                return httpx.Response(200, json={})
            return httpx.Response(403, headers={"X-Error-Code": "GatedRepo"}, json={"error": "gated"})
        return httpx.Response(404)


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    folder = tmp_path / "hf"
    folder.mkdir()
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(folder / "token"))
    monkeypatch.setattr(constants, "HF_STORED_TOKENS_PATH", str(folder / "stored_tokens"))
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        monkeypatch.delenv(name, raising=False)
    # logout() also asks git to forget a huggingface.co login; this computer's git store stays untouched.
    forgets: list[object] = []
    monkeypatch.setattr(hf_login, "unset_git_credential", lambda *a, **k: forgets.append(a))
    fake = Site(names={GOOD: "fluentfolio-read", NO_TERMS: "no-terms", OTHER: "other", ORGANISATION: "org"},
                accepted={GOOD, OTHER})  # fmt: skip
    huggingface_hub.set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(fake.answer)))
    yield fake
    huggingface_hub.set_client_factory(default_client_factory)


def app(tmp_path: Path) -> Path:
    return tmp_path / "app"


def store_other(token: str) -> None:
    """A token saved by another program under the name "other", and active."""
    stored = configparser.ConfigParser()
    stored["other"] = {"hf_token": token}
    with Path(constants.HF_STORED_TOKENS_PATH).open("w", encoding="utf-8") as out:
        stored.write(out)
    Path(constants.HF_TOKEN_PATH).write_text(token, encoding="utf-8")


def test_without_a_token_nothing_is_asked_of_the_site(site: Site) -> None:
    assert hf_token.status() == hf_token.TokenStatus(present=False, valid=None, pyannote_terms_ok=None)
    assert site.seen == []


def test_a_working_token_is_saved_by_its_name_and_its_value_stays_in_the_library_files(
    site: Site, tmp_path: Path
) -> None:
    found, wrote = hf_token.save(f"  {GOOD}\n", app(tmp_path))
    assert wrote and found == hf_token.TokenStatus(present=True, valid=True, pyannote_terms_ok=True)
    assert get_token() == GOOD and get_stored_tokens() == {"fluentfolio-read": GOOD}
    assert install_state(app(tmp_path))[hf_token.SAVED_NAME] == "fluentfolio-read"
    assert GOOD not in (app(tmp_path) / "state.json").read_text(encoding="utf-8")
    assert hf_token.status() == found


def test_saving_a_token_never_hands_it_to_git(
    site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handed: list[object] = []  # git is made to look as if it had a credential store
    monkeypatch.setattr(hf_login, "_is_git_credential_helper_configured", lambda: True)
    monkeypatch.setattr(hf_login, "set_git_credential", lambda *a, **k: handed.append(a))
    assert hf_token.save(GOOD, app(tmp_path))[1]
    assert handed == []


def test_a_token_that_works_is_reused_and_the_new_one_is_never_sent(site: Site, tmp_path: Path) -> None:
    store_other(OTHER)
    found, wrote = hf_token.save(GOOD, app(tmp_path))
    assert not wrote and found.valid and found.pyannote_terms_ok
    assert get_token() == OTHER and get_stored_tokens() == {"other": OTHER}
    assert all(token == OTHER for _, token in site.seen)
    assert hf_token.SAVED_NAME not in install_state(app(tmp_path))


def test_a_token_without_the_terms_is_replaced_by_one_that_has_them(site: Site, tmp_path: Path) -> None:
    store_other(NO_TERMS)
    found, wrote = hf_token.save(GOOD, app(tmp_path))
    assert wrote and found.pyannote_terms_ok and get_token() == GOOD
    assert get_stored_tokens() == {"other": NO_TERMS, "fluentfolio-read": GOOD}


def test_a_refused_or_empty_token_is_never_written(site: Site, tmp_path: Path) -> None:
    for value in (REFUSED, "", "\n"):
        with pytest.raises(AppError) as refused:
            hf_token.save(value, app(tmp_path))
        assert refused.value.code == "E_TOKEN_INVALID"
    assert get_token() is None and get_stored_tokens() == {}
    assert not (app(tmp_path) / "state.json").exists()


def test_an_organisation_token_is_refused(site: Site, tmp_path: Path) -> None:
    with pytest.raises(AppError) as refused:
        hf_token.save(ORGANISATION, app(tmp_path))
    assert refused.value.code == "E_TOKEN_INVALID" and get_token() is None


def test_a_token_whose_account_has_not_accepted_the_terms_is_saved_and_says_so(
    site: Site, tmp_path: Path
) -> None:
    found, wrote = hf_token.save(NO_TERMS, app(tmp_path))
    assert wrote and found == hf_token.TokenStatus(present=True, valid=True, pyannote_terms_ok=False)


def test_clear_removes_only_the_token_fluentfolio_saved(site: Site, tmp_path: Path) -> None:
    assert hf_token.clear(app(tmp_path)) is False  # nothing saved by Fluentfolio: nothing to remove
    store_other(NO_TERMS)
    hf_token.save(GOOD, app(tmp_path))
    assert hf_token.clear(app(tmp_path)) is True
    assert get_stored_tokens() == {"other": NO_TERMS}  # the other program's token stays
    assert get_token() is None  # the active token was ours, so the active file went with it
    assert hf_token.SAVED_NAME not in install_state(app(tmp_path))
    assert hf_token.clear(app(tmp_path)) is False


def test_clear_leaves_the_record_when_the_token_is_already_gone(site: Site, tmp_path: Path) -> None:
    hf_token.save(GOOD, app(tmp_path))
    Path(constants.HF_STORED_TOKENS_PATH).unlink()
    assert hf_token.clear(app(tmp_path)) is False
    assert hf_token.SAVED_NAME not in install_state(app(tmp_path))


def test_clear_finishes_when_logout_fails_after_the_token_is_gone(
    site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # After deleting the token, logout() asks git to forget it, which fails on a computer without git.
    def no_git(*_: object, **__: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(hf_login, "unset_git_credential", no_git)
    hf_token.save(GOOD, app(tmp_path))
    assert hf_token.clear(app(tmp_path)) is True
    assert get_stored_tokens() == {} and get_token() is None
    assert hf_token.SAVED_NAME not in install_state(app(tmp_path))


def test_clear_keeps_the_record_when_the_active_token_file_stays(
    site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The saved entry goes but the active file stays, as when another program holds it open.
    whole = hf_login._logout_from_token

    def half(name: str) -> None:
        active = Path(constants.HF_TOKEN_PATH).read_text(encoding="utf-8")
        whole(name)
        Path(constants.HF_TOKEN_PATH).write_text(active, encoding="utf-8")
        raise PermissionError(constants.HF_TOKEN_PATH)

    monkeypatch.setattr(hf_login, "_logout_from_token", half)
    hf_token.save(GOOD, app(tmp_path))
    with pytest.raises(PermissionError):
        hf_token.clear(app(tmp_path))
    assert get_stored_tokens() == {} and get_token() == GOOD
    assert install_state(app(tmp_path))[hf_token.SAVED_NAME] == "fluentfolio-read"


def tls_failure() -> httpx.ConnectError:
    error = httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]")
    error.__cause__ = ssl.SSLCertVerificationError(1, "certificate verify failed")
    return error


@pytest.mark.parametrize(
    ("fault", "code"), [(httpx.ConnectError("refused"), "E_NETWORK"), (tls_failure(), "E_TLS_INTERCEPTED")]
)
def test_a_site_that_cannot_be_reached_gives_a_network_code(
    site: Site, tmp_path: Path, fault: Exception, code: str
) -> None:
    Path(constants.HF_TOKEN_PATH).write_text(GOOD, encoding="utf-8")
    site.fault = fault
    for call in (hf_token.status, lambda: hf_token.save(OTHER, app(tmp_path))):
        with pytest.raises(AppError) as failed:
            call()
        assert failed.value.code == code


@pytest.mark.parametrize("inside", ["app", "data", "git", "elsewhere"])
def test_the_token_file_place_is_named_when_it_may_be_shared(
    site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inside: str
) -> None:
    roots = {
        "app": tmp_path / "a",
        "data": tmp_path / "d",
        "git": tmp_path / "repo",
        "elsewhere": tmp_path / "e",
    }
    (roots["git"] / ".git").mkdir(parents=True)
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(roots[inside] / "hf" / "token"))
    expected = {
        "app": "the app folder",
        "data": "the data folder",
        "git": "a git working tree",
        "elsewhere": None,
    }
    assert hf_token.file_place(roots["app"], roots["data"]) == expected[inside]


@pytest.fixture
def command(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Command:
    """The command line in this process, where the fake site answers, with a fresh app folder of its own.

    The token commands write the saved token's name into the app folder's state.json; the shared test app
    folder must not get it.
    """
    monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(tmp_path / "app"))
    return Command(capsys, monkeypatch)


def test_the_command_reads_the_token_from_stdin_and_never_prints_it(site: Site, command: Any) -> None:
    exit_code, envelope, printed = command("token", "set", "--stdin", stdin=GOOD + "\n")
    assert exit_code == 0 and envelope["data"] == {
        "token": {"present": True, "valid": True, "pyannote_terms_ok": True}, "saved": True,
    }  # fmt: skip
    _, status, printed_status = command("token", "status")
    assert status["data"] == {"present": True, "valid": True, "pyannote_terms_ok": True}
    _, kept, printed_kept = command("token", "set", "--stdin", stdin=NO_TERMS + "\n")
    assert kept["data"]["saved"] is False  # the working token stays
    _, cleared, _ = command("token", "clear", "--yes")
    assert cleared["data"] == {"removed": True} and get_token() is None
    for output in (printed, printed_status, printed_kept):
        assert GOOD not in output and GOOD[3:20] not in output and NO_TERMS[3:20] not in output


def test_a_token_set_without_a_terminal_or_stdin_asks_for_a_person(site: Site, command: Any) -> None:
    exit_code, envelope, _ = command("token", "set")
    assert (exit_code, envelope["error"]["code"]) == (5, "E_TOKEN_MISSING")
    assert get_token() is None and site.seen == []


def test_a_token_without_the_terms_is_saved_with_the_next_step_and_a_shared_file_warns(
    site: Site, command: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(paths.app_root() / "hf" / "token"))
    exit_code, envelope, _ = command("token", "set", "--stdin", stdin=NO_TERMS)
    assert exit_code == 0 and envelope["data"]["token"]["pyannote_terms_ok"] is False
    assert [w["code"] for w in envelope["warnings"]] == ["W_TOKEN_FILE_PLACE", "E_HF_TERMS_NOT_ACCEPTED"]
    assert "the app folder" in envelope["warnings"][0]["message"]


def test_the_saved_token_file_is_readable_by_its_owner_only(site: Site, tmp_path: Path) -> None:
    hf_token.save(GOOD, app(tmp_path))
    assert Path(constants.HF_TOKEN_PATH).stat().st_mode & 0o777 == 0o600
