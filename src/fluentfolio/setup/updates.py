"""Whether a newer release exists. GitHub is asked only when the learner clicks Check for updates."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from fluentfolio import __version__
from fluentfolio.core.errors import SetupError
from fluentfolio.models import download

LATEST = "https://api.github.com/repos/SacitDev/Fluentfolio/releases/latest"
RELEASES = "https://github.com/SacitDev/Fluentfolio/releases"
TIMEOUT_S = 10
VERSION = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")
NOT_FOUND = 404  # GitHub's answer while the project has published no release


@dataclass(frozen=True)
class Release:
    current: str
    latest: str | None  # None until a release is published
    newer: bool
    page: str = RELEASES


def _number(text: str) -> tuple[int, int, int] | None:
    found = VERSION.fullmatch(text.strip())
    return (int(found[1]), int(found[2]), int(found[3])) if found else None


def latest_tag() -> str | None:
    """The newest release's tag on GitHub; None when the project has none yet. Sends only the request."""
    request = urllib.request.Request(  # noqa: S310 - a fixed https address
        LATEST, headers={"Accept": "application/vnd.github+json", "User-Agent": f"fluentfolio/{__version__}"}
    )
    try:
        with download.https_opener().open(request, timeout=TIMEOUT_S) as answer:
            tag = json.loads(answer.read(100_000)).get("tag_name")
    except urllib.error.HTTPError as exc:
        if exc.code == NOT_FOUND:
            return None
        raise SetupError("E_NETWORK", f"api.github.com, HTTP {exc.code}") from exc
    except (OSError, ValueError, AttributeError) as exc:
        code = "E_TLS_INTERCEPTED" if download.is_tls_failure(exc) else "E_NETWORK"
        raise SetupError(code, "api.github.com") from exc
    return tag if isinstance(tag, str) else None


def check() -> Release:
    tag = latest_tag()
    new, old = _number(tag or ""), _number(__version__)
    return Release(__version__, tag.removeprefix("v") if tag else None, bool(new and old and new > old))
