"""Run one pitch tool's step on a job folder in a process of its own, as the app runs each tool.

Usage: python eval/pitch_tool.py praat|swiftf0|opensmile JOB
Writes JOB/analyzers/<tool>.json. openSMILE runs only with a consent record in FLUENTFOLIO_HOME, and its
process must hold no GPL code, so this script imports nothing but the one tool's step.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

TOOLS = ("praat", "swiftf0", "opensmile")


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in TOOLS:
        print(__doc__)
        return 2
    tool, job = argv[0], Path(argv[1])
    step = importlib.import_module(f"fluentfolio.analyzers.{tool}")
    found = step.run(job, lambda pct: None)
    out = job / "analyzers" / f"{tool}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(found, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
