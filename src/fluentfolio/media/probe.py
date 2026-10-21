"""Opening a recording safely: a forced container reader, local files only, and its first audio track."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import av
from av.container import InputContainer

from fluentfolio.core.errors import InputError
from fluentfolio.ingest.upload import Kind, sniff_file

# Only the local file itself may be read: no URLs, and no other file a container points to (the mov reader's
# external references stay at their default, off).
OPTIONS = {"protocol_whitelist": "file"}
LOSSLESS = ("flac", "alac", "wavpack", "tta")


@dataclass(frozen=True)
class Source:
    container: str
    codec: str
    sample_rate: int
    channels: int
    tracks: int  # audio tracks in the file
    lossy: bool


def open_input(path: Path, kind: Kind) -> InputContainer:
    try:
        return av.open(str(path), format=kind.demuxer, options=OPTIONS)
    except av.FFmpegError as exc:
        raise InputError("E_INPUT_DAMAGED") from exc


def describe(container: InputContainer, kind: Kind) -> Source:
    """The first audio track's codec, rate and channels; a file without one is refused."""
    tracks = container.streams.audio
    if not tracks:
        raise InputError("E_INPUT_NO_AUDIO")
    ctx = tracks[0].codec_context
    codec = ctx.name
    return Source(
        container=kind.name, codec=codec, sample_rate=ctx.sample_rate, channels=ctx.layout.nb_channels,
        tracks=len(tracks), lossy=not (codec.startswith("pcm_") or codec in LOSSLESS),
    )  # fmt: skip


def length_ms(container: InputContainer) -> int:
    """The length the file's header gives; without one, the sum of its first audio track's packets."""
    if container.duration:
        return int(container.duration // 1000)  # in microseconds
    stream = container.streams.audio[0]
    if stream.duration and stream.time_base:
        return int(stream.duration * stream.time_base * 1000)
    total = sum(p.duration * p.time_base for p in container.demux(stream) if p.duration and p.time_base)
    return int(total * 1000)


def main(argv: list[str]) -> int:
    """`python -m fluentfolio.media.probe FILE [FILE2]`: one JSON line, each file's format and length.

    It runs as a child, so the decoder library never loads in the app or the web server.
    """
    try:
        found = []
        for name in argv:
            path = Path(name)
            kind = sniff_file(path)
            with open_input(path, kind) as container:
                found.append({**asdict(describe(container, kind)), "duration_ms": length_ms(container)})
    except InputError as exc:
        print(json.dumps({"code": exc.code, "detail": exc.detail}), flush=True)
        return 1
    print(json.dumps({"probe": found}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
