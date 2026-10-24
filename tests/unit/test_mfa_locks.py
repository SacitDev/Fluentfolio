"""The MFA environment locks come only from conda-forge, pin every package with a hash and hold MFA 3.4.3."""

import re
from importlib import resources

import pytest

PLATFORMS = ("win-64", "linux-64")
LINE = re.compile(
    r"^https://conda\.anaconda\.org/conda-forge/(?P<subdir>[a-z0-9-]+)/(?P<file>[^/#\s]+)#[0-9a-f]{32}$"
)


def lock_problems(text: str, platform: str) -> list[str]:
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    problems = [] if lines[:1] == ["@EXPLICIT"] else ["the first entry is not @EXPLICIT"]
    for line in lines[1:]:
        match = LINE.match(line)
        if match is None:
            problems.append(f"not a hashed conda-forge package: {line}")
        elif match["subdir"] not in (platform, "noarch"):
            problems.append(f"wrong platform: {line}")
    files = " ".join(match["file"] for line in lines[1:] if (match := LINE.match(line)))
    for needed in (r"montreal-forced-aligner-3\.4\.3-", r"kaldi-[\d.]+-cpu_", r"python-3\.12\.\d+-"):
        if not re.search(needed, files):
            problems.append(f"missing {needed}")
    return problems


@pytest.mark.parametrize("platform", PLATFORMS)
def test_committed_lock_is_clean(platform: str) -> None:
    text = resources.files("fluentfolio.data").joinpath(f"mfa-env-{platform}.txt").read_text(encoding="utf-8")
    assert lock_problems(text, platform) == []


@pytest.mark.parametrize("platform", PLATFORMS)
def test_security_packages_are_unique_and_keep_the_verified_conda_hashes(platform: str) -> None:
    text = resources.files("fluentfolio.data").joinpath(f"mfa-env-{platform}.txt").read_text(encoding="utf-8")
    expected = {
        "filelock": "filelock-4.2.1-pyhd8ed1ab_0.conda#f3f085703753cf802b2719d28a570465",
        "setuptools": "setuptools-84.1.0-pyhe730653_0.conda#1541d7acfe174f08d7696ea882765533",
    }
    prefix = "https://conda.anaconda.org/conda-forge/noarch/"
    for name, package in expected.items():
        lines = [line for line in text.splitlines() if f"/{name}-" in line]
        assert lines == [prefix + package]


def test_planted_defaults_channel_and_missing_hash_are_found() -> None:
    text = resources.files("fluentfolio.data").joinpath("mfa-env-win-64.txt").read_text(encoding="utf-8")
    planted = (
        text
        + "https://repo.anaconda.com/pkgs/main/win-64/zlib-1.3.1-h0_0.conda#0123456789abcdef0123456789abcdef\n"
    )
    planted += "https://conda.anaconda.org/conda-forge/win-64/xz-5.8.1-h0_0.conda\n"
    problems = lock_problems(planted, "win-64")
    assert len(problems) == 2
    assert all(p.startswith("not a hashed conda-forge package") for p in problems)
