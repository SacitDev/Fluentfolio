"""The rules every test runs under: its own data folder and no network unless marked."""

import json
import os
import socket
from pathlib import Path

import pytest


def test_data_folder_is_inside_this_test(tmp_path: Path) -> None:
    assert Path(os.environ["FLUENTFOLIO_HOME"]).parent == tmp_path


def test_outside_connection_is_refused() -> None:
    with pytest.raises(OSError, match="not marked network"):
        socket.create_connection(("192.0.2.1", 9), timeout=1)


def test_loopback_connection_is_allowed() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=5):
            pass


def test_subprocesses_are_sent_offline() -> None:
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HTTPS_PROXY"].startswith("http://127.0.0.1:")


@pytest.fixture
def summary_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FF_PRIVATE_SUMMARY_DIR", str(tmp_path / "summary"))
    return tmp_path / "summary"


def test_private_summary_merges_numbers(summary_dir: Path, private_summary) -> None:
    private_summary("speakers", {"sample1": {"n_speakers": 2, "shares": [0.6, 0.4], "ok": True}})
    private_summary("speakers", {"sample2": {"n_speakers": 1, "rtf": None}})
    data = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
    assert data == {"speakers": {"sample1": {"n_speakers": 2, "shares": [0.6, 0.4], "ok": True},
                                 "sample2": {"n_speakers": 1, "rtf": None}}}  # fmt: skip


@pytest.mark.parametrize(
    "values",
    [{"word": "hello"}, {"words": [1, "two"]}, {"Upper Key": 1}, {"nested": {"a": b"x"}}, {"t": (1, 2)}],
    ids=["string", "string-in-list", "text-key", "bytes", "tuple"],
)
def test_private_summary_refuses_text(summary_dir: Path, private_summary, values: dict) -> None:
    with pytest.raises(AssertionError, match="only numbers"):
        private_summary("speakers", values)
    assert not (summary_dir / "summary.json").exists()
