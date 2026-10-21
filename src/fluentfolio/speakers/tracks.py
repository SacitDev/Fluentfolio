"""One track per person: each person's speech is what the speech detector finds on their own track.

A voice also leaks into the other person's microphone, more quietly. Where both tracks hold speech, the track
louder by 6 dB or more keeps the frame alone; closer than that, both keep it and it counts as overlap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from fluentfolio.core.files import read_json
from fluentfolio.speakers import checks

LOUDER_DB = 6.0
# Levels are compared over about half a second, so the short quiet gaps inside words decide nothing.
WINDOW_FRAMES = 15
# Known limit: a starting value for "the tracks hear each other too much"; true overlap in conversation
# rarely passes it. Revisit with real two-track recordings.
BLEED_PCT = 30
TINY = 1e-12


def frame_energy(wav: Path, frame: int) -> np.ndarray:
    """The mean square of each frame of a WAV, the last frame padded with silence."""
    out = []
    for block in sf.blocks(str(wav), blocksize=frame * 4096, dtype="float32"):
        x = np.pad(block.astype(np.float64), (0, -len(block) % frame)).reshape(-1, frame)
        out.append(np.mean(x**2, axis=1))
    return np.concatenate(out) if out else np.zeros(0)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return np.pad(x, (0, n - len(x)))


def resolve(voiced: list[np.ndarray], energy: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """Per frame, the speech each of two tracks keeps after bleed, and the frames both keep (overlap)."""
    n = max(len(v) for v in (*voiced, *energy))
    v = [_fit(x, n) for x in voiced]
    window = np.ones(WINDOW_FRAMES) / WINDOW_FRAMES
    level = [10 * np.log10(np.convolve(_fit(e, n), window, mode="same") + TINY) for e in energy]
    diff = np.round(level[0] - level[1], 2)  # to 0.01 dB, so a ratio of exactly 6 dB is exactly 6
    both = v[0] & v[1]
    owned = [v[0] & ~(both & (diff <= -LOUDER_DB)), v[1] & ~(both & (diff >= LOUDER_DB))]
    return owned, owned[0] & owned[1]


def mask_runs(mask: np.ndarray, frame_ms: int, min_ms: int) -> list[tuple[int, int]]:
    """Runs of true frames as (first frame, end frame), keeping those longer than min_ms."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2) if (b - a) * frame_ms > min_ms]


def bleed_report(
    voiced: list[np.ndarray], both: np.ndarray, frame_ms: int
) -> tuple[dict[str, Any], list[str]]:
    """How much voiced time neither track wins, and the warning when that suggests the mixed layout."""
    n = len(both)
    any_voice = int(np.count_nonzero(_fit(voiced[0], n) | _fit(voiced[1], n)))
    share = int(np.count_nonzero(both)) / any_voice if any_voice else 0.0
    report = {"louder_db": LOUDER_DB, "voiced_ms": any_voice * frame_ms,
              "both_ms": int(np.count_nonzero(both)) * frame_ms, "both_share": round(share, 3)}  # fmt: skip
    return report, ["W_TRACK_BLEED"] if np.count_nonzero(both) * 100 > BLEED_PCT * any_voice else []


def run(job: Path) -> dict[str, Any]:
    """The speakers step for one track per person: track n is speaker Sn, with the bleed already removed."""
    vad = read_json(job / "stages" / "vad.json")
    turns = [(a, b, f"S{t['track']}") for t in vad["tracks"] for a, b in t["segments"]]
    return checks.summarize("tracks", sorted(turns), mode=str(read_json(job / "job.json")["mode"]))
