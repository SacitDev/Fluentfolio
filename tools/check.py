"""The one check command: `fast` before every commit, `full` before a release.

Usage, from the repository root:  uv run --locked python tools/check.py fast|full [--list]

Every step runs, even after a failure, and a missing tool counts as a failure. The last line is
exactly "check <mode>: PASS" or "check <mode>: FAIL (<n> failed)". --list prints the commands and
runs nothing. The environment passes through, so FF_* variables reach pytest; the default tests also
get OPENBLAS_NUM_THREADS=1.
Exit codes: 0 pass, 1 fail, 2 usage.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Each test worker imports numpy, whose OpenBLAS sets memory aside for every core: on 32 cores a worker
# committed 2.6 GB, and 32 workers used up the system's commit on a busy PC, so a new thread crashed.
# One BLAS thread each cuts a worker to 1.1 GB, and 16 workers run the tests as fast as 32.
FAST_TESTS = ["pytest", "-q", "-n", str(min(16, os.cpu_count() or 1)), "--dist", "worksteal"]
FAST_TESTS_ENV = {"OPENBLAS_NUM_THREADS": "1"}

LINT = [
    ["ruff", "format", "--check", "."],
    ["ruff", "check", "."],
    ["lint-imports"],
    ["mypy"],
    ["bandit", "-q", "-r", "src", "-t", "B614,B615"],
]
STEPS = {
    # The default tests spread over up to 16 workers, and an idle worker takes tests from a busy one, so
    # a few long tests do not hold up the end; the slow, GPU and private runs below stay in one process,
    # so only one model at a time uses the graphics card.
    "fast": [*LINT, FAST_TESTS],
    "full": [
        *LINT,
        ["coverage", "run", "--branch", "-m", "pytest", "-q"],
        ["coverage", "report"],
        ["pytest", "-q", "-m", "slow or gpu or private_sample"],
        ["uv", "lock", "--check"],
        ["osv-scanner", "scan", "source", "--lockfile", "uv.lock"],
        ["deptry", "src"],
        ["python", "tools/notices.py", "--check"],
        ["gitleaks", "git", ".", "--redact", "--no-banner", "--ignore-gitleaks-allow"],
        ["shellcheck", "install.sh"],
    ],
}


def run_step(cmd: list[str]) -> bool:
    """Run one step in the repository root and say whether it passed."""
    if cmd[0] == "shellcheck" and not (ROOT / "install.sh").exists():
        print("shellcheck: install.sh not present yet", flush=True)
        return True
    exe = sys.executable if cmd[0] == "python" else shutil.which(cmd[0])
    if exe is None:
        print(f"{cmd[0]}: not found", flush=True)
        return False
    env = {**os.environ, **FAST_TESTS_ENV} if cmd is FAST_TESTS else None
    try:
        return subprocess.run([exe, *cmd[1:]], cwd=ROOT, check=False, env=env).returncode == 0
    except OSError as exc:
        print(f"{cmd[0]}: could not start ({exc})", flush=True)
        return False


def shown(cmd: list[str]) -> str:
    """The step as a command line, with the variables it gets in front."""
    extra = FAST_TESTS_ENV if cmd is FAST_TESTS else {}
    return " ".join([*(f"{name}={value}" for name, value in extra.items()), subprocess.list2cmdline(cmd)])


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--list"]
    if len(args) != 1 or args[0] not in STEPS:
        print("usage: python tools/check.py fast|full [--list]", file=sys.stderr)
        return 2
    mode = args[0]
    if "--list" in argv:
        for cmd in STEPS[mode]:
            print(shown(cmd))
        return 0
    failed = []
    for cmd in STEPS[mode]:
        line = shown(cmd)
        print(f"== {line}", flush=True)
        ok = run_step(cmd)
        print(f"{'PASS' if ok else 'FAIL'}: {line}", flush=True)
        if not ok:
            failed.append(line)
    for line in failed:
        print(f"failed: {line}")
    print(f"check {mode}: PASS" if not failed else f"check {mode}: FAIL ({len(failed)} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
