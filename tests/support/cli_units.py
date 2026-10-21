"""The units every JSON reply of the command line keeps: UTC times with Z, whole-ms durations, plain paths."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]+)?Z$")
PATHS = {"path", "data_dir", "executable", "job_dir", "report", "zip"}
OTHER_UNIT = re.compile(r"(_s|_sec|seconds|_min|minutes|_hours?)$")  # a duration named in another unit


def check(value: Any, key: str = "") -> None:
    """Fail on any value of the reply that breaks the units the JSON contract promises."""
    if isinstance(value, dict):
        for k, v in value.items():
            check(v, k)
    elif isinstance(value, list):
        for v in value:
            check(v, key)
    elif value is not None:
        assert not OTHER_UNIT.search(key), f"{key}: durations are whole milliseconds under a _ms name"
        if key.endswith("_ms"):
            assert type(value) is int, (key, value)
        if key == "utc" or key.endswith("_utc"):
            assert UTC.match(value), (key, value)
        if key in PATHS:
            assert "\\" not in value and Path(value).is_absolute(), (key, value)
