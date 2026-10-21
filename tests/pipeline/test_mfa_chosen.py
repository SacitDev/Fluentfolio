"""MFA switched on by the learner, on a composed session: its phones reach the rhythm measures.

MFA is off by default where it did not pass its word-timing check, so this path runs only when chosen.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fluentfolio.app import facade
from fluentfolio.core.files import read_json
from fluentfolio.evidence import build
from fluentfolio.settings import store as settings_store
from fluentfolio.settings.model import Settings

pytestmark = pytest.mark.slow
VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
# Wide bounds for English speech, not targets: %V near 40 (Ramus, Nespor and Mehler 1999) and nPVI-V near 57
# (Grabe and Low 2002) for English; phone lengths in ms, where MFA's shortest phone is 30 ms.
PLAUSIBLE = {"DM03": (20, 100), "DM04": (20, 100), "DM05": (30, 60), "DM06": (40, 200), "DM07": (30, 200)}


def test_mfa_chosen_on_cs_solo_gives_rhythm_from_its_phones(
    built: Callable[[str], tuple[Path, dict[str, Any]]], scoring: ModuleType, tolerance: Any
) -> None:
    folder, _ = built("CS-SOLO")
    home = Path(os.environ["FLUENTFOLIO_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    settings_store.save(home / "settings.toml", Settings(components={"mfa": True}))
    done = facade.analyze(folder / "session.wav", "solo")
    assert scoring.gaps(done.job) == []
    alignment = read_json(done.job / "stages" / "alignment.json")
    ran = alignment["mfa"]
    print(f"MFA {ran['status']} {ran['version']}: {ran['aligned']} of {ran['utterances']} utterances")
    assert (ran["status"], ran["version"], ran["experimental"]) == ("ok", "3.4.3", True)
    assert ran["aligned"] >= 1
    evidence = read_json(done.job / "evidence.json")
    (row,) = [t for t in evidence["manifest"]["tools"] if t["id"] == "alignment"]
    assert row["reason"] == f"MFA is {build.MFA_EXPERIMENTAL}"
    # Counted here from MFA's own phones: rhythm may use only the answers' share of them.
    aligned = [u for u in alignment["utterances"] if u["method"] == "mfa"]
    vowels = sum(str(p[2]).rstrip("012") in VOWELS for u in aligned for p in u["phones"])
    (candidate,) = evidence["candidates"]
    found = {
        m["id"]: m for m in candidate["details"]["measures"] if m["id"] in PLAUSIBLE and m["scope"] == "all"
    }
    print({k: (m["value"], m["n"]) for k, m in sorted(found.items())})
    assert sorted(found) == sorted(PLAUSIBLE)
    assert all(m["sources"][0]["tool"] == "mfa" and m["value"] is not None for m in found.values())
    assert tolerance("rhythm", "min_vowels") <= found["DM06"]["n"] <= vowels  # n counts the vowel sounds
    for key, (low, high) in PLAUSIBLE.items():
        assert low <= found[key]["value"] <= high, key
