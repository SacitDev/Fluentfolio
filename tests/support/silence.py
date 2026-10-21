"""Reference speech edges from FFmpeg's silencedetect filter, run through PyAV, with no Fluentfolio code.

Tests compare the voice activity detector and the aligners with these edges, so the expected values come
from a tool outside the code under test.
"""

from __future__ import annotations

from pathlib import Path

import av

NOISE_DB = -30  # the level at which silencedetect finds silence at both ends of the VOiCES clip
MIN_SILENCE_S = 0.2


def _ready(graph: av.filter.Graph) -> list[tuple[str, float]]:
    """The silence starts and ends the filter has marked on the frames ready so far."""
    out = []
    while True:
        try:
            frame = graph.pull()
        except (av.error.BlockingIOError, av.error.EOFError):
            return out
        for key in ("lavfi.silence_start", "lavfi.silence_end"):
            if key in frame.metadata:
                out.append((key.rsplit("_", 1)[-1], float(frame.metadata[key])))


def events(path: Path, noise_db: float, min_s: float = MIN_SILENCE_S) -> list[tuple[str, float]]:
    """Every ("start", s) and ("end", s) of a silence of at least `min_s` below `noise_db`, in time order."""
    found = []
    with av.open(str(path)) as box:
        stream = box.streams.audio[0]
        graph = av.filter.Graph()
        source = graph.add_abuffer(template=stream)
        detect = graph.add("silencedetect", f"noise={noise_db}dB:d={min_s}")
        sink = graph.add("abuffersink")
        source.link_to(detect)
        detect.link_to(sink)
        graph.configure()
        for frame in box.decode(stream):
            graph.push(frame)
            found += _ready(graph)
        graph.push(None)
        found += _ready(graph)
    return found


def onset_ms(path: Path, noise_db: float) -> int:
    """Where speech starts: the end of the first silence."""
    ends = [t for kind, t in events(path, noise_db) if kind == "end"]
    assert ends, f"no silence before the speech at {noise_db} dB"
    return round(ends[0] * 1000)


def offset_ms(path: Path, noise_db: float) -> int:
    """Where speech ends: the start of a silence that lasts to the end of the file."""
    found = events(path, noise_db)
    assert found and found[-1][0] == "start", f"no silence after the speech at {noise_db} dB: {found}"
    return round(found[-1][1] * 1000)


def speech_span(path: Path, noise_db: float = NOISE_DB) -> tuple[int, int]:
    """The end of the leading silence and the start of the trailing one, both found at one level, in ms."""
    found = events(path, noise_db)
    assert found[:1] == [("start", 0.0)], f"no silence at the start at {noise_db} dB: {found}"
    return onset_ms(path, noise_db), offset_ms(path, noise_db)
