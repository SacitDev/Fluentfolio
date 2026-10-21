"""Logs never hold a token, the home folder or the user name, in a message or in a traceback."""

import getpass
import logging
import logging.handlers
from pathlib import Path

from fluentfolio.core import logs


def test_planted_canaries_never_reach_the_log_file(tmp_path: Path) -> None:
    token = "hf_" + "aB3" * 11
    home = Path.home()
    user = getpass.getuser()
    log = logging.getLogger("fluentfolio.test_canary")
    log.propagate = False
    target = logs.handler(logging.FileHandler(tmp_path / "app.log", encoding="utf-8"))
    log.addHandler(target)
    try:
        log.warning("token %s in %s and %s", token, home / "a.wav", home.as_posix() + "/b.wav")
        log.warning("user %s finished", user)
        try:
            raise ValueError(f"cannot read {home / 'c.wav'} with {token}")
        except ValueError:
            log.exception("step failed")
    finally:
        log.removeHandler(target)
        target.close()
    text = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert token not in text and str(home) not in text and home.as_posix() not in text
    assert f" {user} " not in text and "user <user> finished" in text
    assert text.count("hf_<REDACTED>") == 2 and "ValueError: cannot read ~" in text


def test_header_and_url_parameter_values_are_masked_whatever_their_form() -> None:
    secret = "Zq9" * 8  # not in the hf_ form, so only the header and parameter rules can catch it
    assert logs.redact(f"GET /x?token={secret}&rev=main") == "GET /x?token=<REDACTED>&rev=main"
    assert logs.redact(f"authorization: Bearer {secret}") == "authorization: <REDACTED>"
    assert logs.redact("token: none given") == "token: none given"


def test_the_app_log_rotates_at_5_mb_and_keeps_5_files(tmp_path: Path) -> None:
    logs.setup_app_log(tmp_path, "warning")
    root = logging.getLogger("fluentfolio")
    try:
        rotating = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(rotating) == 1
        assert (rotating[0].maxBytes, rotating[0].backupCount + 1) == (5_000_000, 5)
        assert Path(rotating[0].baseFilename) == tmp_path / "logs" / "app.log"
        assert root.level == logging.WARNING
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        root.setLevel(logging.NOTSET)
