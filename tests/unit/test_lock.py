"""The lock file keeps the file decoder out and takes torch only from the PyTorch index."""

import tomllib
from pathlib import Path

LOCK = Path(__file__).resolve().parents[2] / "uv.lock"
PYTORCH_INDEX = "https://download.pytorch.org/whl/"


def locked_packages(text: str) -> list[dict]:
    return tomllib.loads(text).get("package", [])


def names(packages: list[dict]) -> set[str]:
    return {p["name"] for p in packages}


def test_lock_has_no_torchcodec() -> None:
    assert "torchcodec" not in names(locked_packages(LOCK.read_text(encoding="utf-8")))


def test_planted_torchcodec_is_found() -> None:
    planted = '[[package]]\nname = "torchcodec"\nversion = "0.1"\n'
    assert "torchcodec" in names(locked_packages(planted))


def test_torch_comes_only_from_the_pytorch_index() -> None:
    packages = locked_packages(LOCK.read_text(encoding="utf-8"))
    torch_like = [p for p in packages if p["name"] in ("torch", "torchaudio")]
    assert torch_like, "torch is missing from the lock"
    for p in torch_like:
        assert p["source"].get("registry", "").startswith(PYTORCH_INDEX), (p["name"], p["version"])
