"""No unsafe deserialisation in our own code.

No pickle-based loader, no weights_only=False, no allow_pickle=True, and a locked torch of at least 2.6, where
torch.load defaults to weights_only=True. The pickled weights inside pinned, hashed model files (the pyannote
speaker model) are an accepted risk, named in SECURITY.md.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fluentfolio"
# Modules that unpickle, and the call forms that unpickle or switch the safe default off.
BANNED_MODULES = {"pickle", "cPickle", "dill", "cloudpickle", "joblib", "shelve", "marshal"}
BANNED_KEYWORDS = {("weights_only", False), ("allow_pickle", True)}
BANNED_CALLS = {"torch.load", "np.load", "numpy.load"}  # each needs its safe keyword checked by eye


def findings(source: str, name: str) -> list[str]:
    """Each banned import, call or keyword in one module's source, as "name:line what"."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import | ast.ImportFrom):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            banned = [m for m in modules if m.split(".")[0] in BANNED_MODULES]
            found += [f"{name}:{node.lineno} import {m}" for m in banned]
        elif isinstance(node, ast.Call):
            called = ast.unparse(node.func)
            if called in BANNED_CALLS:
                found.append(f"{name}:{node.lineno} {called}")
            for kw in node.keywords:
                if isinstance(kw.value, ast.Constant) and (kw.arg, kw.value.value) in BANNED_KEYWORDS:
                    found.append(f"{name}:{node.lineno} {kw.arg}={kw.value.value}")
    return found


def test_our_code_unpickles_nothing_and_never_turns_the_safe_default_off() -> None:
    found = [f for p in sorted(SRC.rglob("*.py")) for f in findings(p.read_text(encoding="utf-8"), p.name)]
    assert found == []


@pytest.mark.parametrize(
    "planted",
    [
        "import pickle\n",
        "from joblib import load\n",
        "import torch\ntorch.load('m.pt', weights_only=False)\n",
        "import numpy as np\nnp.load('a.npy', allow_pickle=True)\n",
    ],
)
def test_the_check_finds_each_planted_loader(planted: str) -> None:
    assert findings(planted, "planted.py") != []


def test_the_lock_has_no_torch_before_2_6() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = [p["version"] for p in lock["package"] if p["name"] == "torch"]
    assert versions  # the lock holds torch for the cpu and cu130 extras
    for v in versions:
        major, minor = (int(x) for x in v.split("+")[0].split(".")[:2])
        assert (major, minor) >= (2, 6), v
