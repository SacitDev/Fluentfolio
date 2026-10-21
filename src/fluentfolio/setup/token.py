"""The learner's own Hugging Face token: saved, tested and removed only through huggingface_hub's own files.

Fluentfolio never keeps the token's value. It reuses a token that works, never overwrites it, and removes
only a token it saved itself, by the name it recorded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fluentfolio.core.errors import AppError, SetupError
from fluentfolio.core.files import write_json
from fluentfolio.core.paths import install_state
from fluentfolio.models import catalogue, download

GATED = "pyannote-community-1"  # the one part whose terms the learner accepts on its model page
SAVED_NAME = "hf_token_name"  # in the app folder's state.json: the name of the token Fluentfolio saved
HTTP_UNAUTHORIZED, HTTP_FORBIDDEN = 401, 403


@dataclass(frozen=True)
class TokenStatus:
    present: bool
    valid: bool | None  # None: nothing to test
    pyannote_terms_ok: bool | None


def file_place(app: Path, data: Path) -> str | None:
    """The shared place the token file lies in (the app or data folder, or a git working tree), if any."""
    from huggingface_hub import constants

    token_file = Path(constants.HF_TOKEN_PATH).resolve()
    for name, root in (("app", app), ("data", data)):
        if token_file.is_relative_to(root.resolve()):
            return f"the {name} folder"
    return "a git working tree" if any((p / ".git").exists() for p in token_file.parents) else None


def _online[T](call: Callable[[], T]) -> T:
    """One call to the site; a failure to reach it, or an answer that is not about the token, is E_NETWORK."""
    from huggingface_hub.errors import HfHubHTTPError

    try:
        return call()
    except HfHubHTTPError:
        raise  # the site answered; the caller reads the status
    except Exception as exc:  # noqa: BLE001 - every socket, TLS and client error means the site was not reached
        code = "E_TLS_INTERCEPTED" if download.is_tls_failure(exc) else "E_NETWORK"
        raise SetupError(code, "huggingface.co") from exc


def _status_of(exc: Exception) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def _account(token: str) -> dict[str, Any] | None:
    """The account behind a token, or None when the site refuses the token."""
    from huggingface_hub import hf_api
    from huggingface_hub.errors import HfHubHTTPError

    try:
        return _online(lambda: hf_api.whoami(token=token))
    except HfHubHTTPError as exc:
        if _status_of(exc) == HTTP_UNAUTHORIZED:
            return None
        raise SetupError("E_NETWORK", f"huggingface.co, HTTP {_status_of(exc)}") from exc


def _terms_ok(token: str) -> bool:
    """Whether the account accepted the gated model's terms on its page."""
    from huggingface_hub import hf_api
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    try:
        _online(lambda: hf_api.auth_check(str(catalogue.get(GATED).repo), token=token))
    except RepositoryNotFoundError:  # also the gated answer: the terms are not accepted yet
        return False
    except HfHubHTTPError as exc:
        if _status_of(exc) in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
            return False
        raise SetupError("E_NETWORK", f"huggingface.co, HTTP {_status_of(exc)}") from exc
    return True


def _test(token: str) -> TokenStatus:
    if _account(token) is None:
        return TokenStatus(present=True, valid=False, pyannote_terms_ok=None)
    return TokenStatus(present=True, valid=True, pyannote_terms_ok=_terms_ok(token))


def status() -> TokenStatus:
    """Present, valid and pyannote_terms_ok; the token is tested online only when there is one."""
    from huggingface_hub import get_token

    token = get_token()
    return TokenStatus(present=False, valid=None, pyannote_terms_ok=None) if token is None else _test(token)


def save(token: str, app: Path) -> tuple[TokenStatus, bool]:
    """Save a new token, unless one that works is there already; returns its status and whether it wrote."""
    from huggingface_hub import get_token, login
    from huggingface_hub.utils import get_stored_tokens

    token = token.strip()
    if found := get_token():
        current = _test(found)
        if current.valid and current.pyannote_terms_ok:
            return current, False
    if not token or _account(token) is None:
        raise AppError("E_TOKEN_INVALID")
    try:
        login(token=token, add_to_git_credential=False)
    except ValueError as exc:  # an organisation token, which only a personal account may replace
        raise AppError("E_TOKEN_INVALID") from exc
    name = next(n for n, value in get_stored_tokens().items() if value == token)
    state = install_state(app)
    state[SAVED_NAME] = name
    app.mkdir(parents=True, exist_ok=True)
    write_json(app / "state.json", state)
    return TokenStatus(present=True, valid=True, pyannote_terms_ok=_terms_ok(token)), True


def clear(app: Path) -> bool:
    """Remove the token only if Fluentfolio saved it; returns whether it removed one."""
    from huggingface_hub import constants, logout
    from huggingface_hub.utils import get_stored_tokens

    state = install_state(app)
    name = state.pop(SAVED_NAME, None)
    if not name:
        return False
    token = get_stored_tokens().get(name)
    if token is not None:
        try:
            logout(token_name=name)  # never logout() without a name: that removes every saved token
        except OSError:  # also raised once the token is gone: without git, or while HF_TOKEN holds a token
            active = Path(constants.HF_TOKEN_PATH)
            if name in get_stored_tokens() or (active.is_file() and active.read_text().strip() == token):
                raise
    write_json(app / "state.json", state)
    return token is not None
