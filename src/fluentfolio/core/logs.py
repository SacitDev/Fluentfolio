"""Standard logging with one redaction filter on every handler: no tokens, home paths or user names."""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import traceback
from pathlib import Path

TOKEN = re.compile(r"hf_[A-Za-z0-9]{20,}")
# A request header or a URL parameter can carry a token in any form, so its whole value goes.
AUTHORIZATION = re.compile(r"(authorization:\s*).+", re.IGNORECASE)
TOKEN_PARAMETER = re.compile(r"(token=)[^\s&\"']+", re.IGNORECASE)
APP_LOG_BYTES = 5_000_000
APP_LOG_FILES = 5


def _secrets() -> list[tuple[re.Pattern[str], str]]:
    home = str(Path.home())
    rules = [(TOKEN, "hf_<REDACTED>"), (AUTHORIZATION, r"\1<REDACTED>"), (TOKEN_PARAMETER, r"\1<REDACTED>")]
    for form in {home, home.replace("\\", "/")}:
        rules.append((re.compile(re.escape(form), re.IGNORECASE), "~"))
    user = os.environ.get("USERNAME") or os.environ.get("USER")
    if user and len(user) > 1:
        name = re.compile(rf"(?<![A-Za-z0-9]){re.escape(user)}(?![A-Za-z0-9])", re.IGNORECASE)
        rules.append((name, "<user>"))
    return rules


def redact(text: str) -> str:
    """Replace a token, an Authorization value, the home folder and the user name in one log line."""
    for pattern, replacement in _secrets():
        text = pattern.sub(replacement, text)
    return text


def crash_lines(error: BaseException) -> str:
    """An unexpected error as its type and code lines only: its message may hold a learner's words."""
    return type(error).__name__ + "\n" + "".join(traceback.format_tb(error.__traceback__))


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        if record.exc_info:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return True


def handler(target: logging.Handler, level: int = logging.INFO) -> logging.Handler:
    """A handler with the redaction filter and the one line format."""
    target.addFilter(RedactionFilter())
    target.setLevel(level)
    target.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    return target


def _app_logs() -> list[logging.handlers.RotatingFileHandler]:
    root = logging.getLogger("fluentfolio")
    return [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]


def setup_app_log(data: Path, level: str = "info") -> None:
    """logs/app.log in the data folder, kept as 5 files of 5 MB; once per process and file."""
    folder = data / "logs"
    if any(h.baseFilename == os.path.abspath(folder / "app.log") for h in _app_logs()):
        return  # a second handler on the same file would write every line twice
    folder.mkdir(parents=True, exist_ok=True)
    rotating = logging.handlers.RotatingFileHandler(
        folder / "app.log", maxBytes=APP_LOG_BYTES, backupCount=APP_LOG_FILES - 1, encoding="utf-8"
    )
    root = logging.getLogger("fluentfolio")
    root.setLevel(level.upper())
    root.addHandler(handler(rotating, getattr(logging, level.upper())))


def close_app_log() -> None:
    """Close the app log, so its files can be deleted; a later setup_app_log opens it again."""
    root = logging.getLogger("fluentfolio")
    for found in _app_logs():
        root.removeHandler(found)
        found.close()
