"""sherpa-onnx speaker separation on the private recordings; it records numbers only."""

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from fluentfolio.core.paths import models_dir
from fluentfolio.models import catalogue
from fluentfolio.models.download import folder
from fluentfolio.speakers.sherpa import BACKEND, EMBEDDER, SEGMENTATION

CHILD = Path(__file__).parents[1] / "support" / "sherpa_child.py"
# Any variable with these words in its name stays out of the child.
SECRET_WORDS = ("TOKEN", "SECRET", "PASSWORD", "KEY")
# A cluster with less talk time than this becomes "other", not a speaker.
OTHER_SHARE = 0.05


def speech_s(turns: list[list[float]]) -> float:
    """Seconds covered by at least one turn."""
    total, reach = 0.0, 0.0
    for start, end, _ in sorted(turns):
        if end > reach:
            total += end - max(start, reach)
            reach = end
    return total


def overlap_s(turns: list[list[float]]) -> float:
    """Seconds where turns of two different speakers run at the same time."""
    total = 0.0
    for i, (s1, e1, k1) in enumerate(turns):
        for s2, e2, k2 in turns[i + 1 :]:
            if k1 != k2:
                total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


@pytest.mark.private_sample
@pytest.mark.slow
@pytest.mark.parametrize("index", [0, 1], ids=["sample1", "sample2"])
def test_sherpa_on_private_sample(
    index: int,
    tmp_path: Path,
    private_recordings: list[Path],
    private_summary: Callable[[str, dict[str, object]], None],
) -> None:
    # Every assertion compares plain numbers, so a failure report never shows a path or file name.
    # Where an install keeps the part, as the speaker step reads it.
    models = folder(catalogue.get(BACKEND), models_dir())
    assert (models / SEGMENTATION).is_file() and (models / EMBEDDER).is_file(), "sherpa models not found"
    env = {k: v for k, v in os.environ.items() if not any(w in k.upper() for w in SECRET_WORDS)}
    env.update(TMP=str(tmp_path), TEMP=str(tmp_path), TMPDIR=str(tmp_path))
    result = tmp_path / "result.json"
    done = subprocess.run(
        [sys.executable, "-I", str(CHILD), str(private_recordings[index]), str(models), str(result)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    # The child prints only an error class name; anything else stays out of the message.
    code, last = done.returncode, (done.stderr.strip().splitlines() or [""])[-1]
    error = last if last.isidentifier() else ""
    assert code == 0, f"child exit {code} {error}"
    out = json.loads(result.read_text(encoding="utf-8"))
    duration, turns = out["duration_s"], out["turns"]
    speakers = sorted({k for _, _, k in turns})
    talk = {k: sum(e - s for s, e, kk in turns if kk == k) for k in speakers}
    total = sum(talk.values())
    speech = speech_s(turns)
    main = sum(v / total >= OTHER_SHARE for v in talk.values())
    private_summary(
        "sherpa_diarization",
        {
            f"sample{index + 1}": {
                "duration_s": round(duration, 2),
                "n_clusters": len(speakers),
                "n_speakers_over_5pct": main,
                "other_share": round(sum(v for v in talk.values() if v / total < OTHER_SHARE) / total, 4),
                "n_turns": len(turns),
                "talk_shares": sorted((round(v / total, 3) for v in talk.values()), reverse=True),
                "speech_s": round(speech, 2),
                "overlap_s": round(overlap_s(turns), 2),
                "rtf_cpu": round(out["process_s"] / duration, 4),
                "token_variables": out["token_variables"],
                "onnxruntime_loaded": out["onnxruntime_loaded"],
            }
        },
    )
    assert out["token_variables"] == 0
    assert out["onnxruntime_loaded"] is False
    # Small clusters are expected on long recordings; the 5 % rule turns them into "other".
    assert 1 <= main <= 3
    assert all(0.0 <= s < e <= duration + 0.01 for s, e, _ in turns)
    assert 0.2 * duration <= speech <= duration
