"""Fluentfolio: measured evidence from IELTS Speaking practice recordings, for any AI chat."""

import os
from importlib import metadata

# Set before any library is imported, because some read these switches at import time.
for _name, _value in (
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
    ("DO_NOT_TRACK", "1"),
    ("HF_HUB_DISABLE_UPDATE_CHECK", "1"),
    ("PYANNOTE_METRICS_ENABLED", "0"),
    ("ORT_DISABLE_TELEMETRY", "1"),
):
    os.environ[_name] = _value

__version__ = metadata.version("fluentfolio")
