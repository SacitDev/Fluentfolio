"""The media stage: decode the recording once into a 16 kHz mono WAV; measure its quality; look for music.

With one track per person (`per-track`, `two-files`) it also writes input/track_1.wav and input/track_2.wav,
and the main WAV is their average.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import av
import numpy as np
import soundfile as sf
from av.container import InputContainer

from fluentfolio.core.enums import Layout
from fluentfolio.core.errors import InputError
from fluentfolio.core.files import read_json
from fluentfolio.ingest.upload import KINDS, Kind, check_duration
from fluentfolio.media import probe, quality
from fluentfolio.settings.model import Limits

RATE = 16000
CLIP_LEVEL = 0.999  # a source sample this close to full scale counts as clipped
PROGRESS_EVERY_S = 1.0
BLOCK = 1 << 20  # samples per block when silence is written or tracks are mixed
LENGTHS_DIFFER_MS = 2000  # two files that differ by more than this after the offset get a warning


def _samples(frame: av.AudioFrame) -> np.ndarray:
    """One decoded frame as float samples in [-1, 1], one row per channel."""
    x = frame.to_ndarray()
    if not frame.format.is_planar:
        x = x.reshape(-1, frame.layout.nb_channels).T
    if x.dtype.kind == "f":
        return x.astype(np.float32, copy=False)
    if x.dtype == np.uint8:
        return (x.astype(np.float32) - 128) / 128
    return (x / float(np.iinfo(x.dtype).max + 1)).astype(np.float32)


class _Writer:
    """One 16 kHz mono 16-bit WAV, written through its own resampler and kept as .part until it is whole."""

    def __init__(self, path: Path) -> None:
        self.path, self.part = path, path.with_name(path.name + ".part")
        self.resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
        self.wav = sf.SoundFile(self.part, "w", samplerate=RATE, channels=1, subtype="PCM_16", format="WAV")
        self.written = 0

    def silence(self, samples: int) -> None:
        for start in range(0, samples, BLOCK):
            self.wav.write(np.zeros(min(BLOCK, samples - start), dtype=np.float32))
        self.written += samples

    def put(self, mono: np.ndarray | None, rate: int) -> None:
        """Resample one block of mono samples and write it; None flushes the resampler."""
        frame = None
        if mono is not None:
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(mono[None, :]), format="flt", layout="mono"
            )
            frame.sample_rate = rate
        for f in self.resampler.resample(frame):
            y = f.to_ndarray()[0]
            self.wav.write(np.clip(y, -1.0, 1.0))
            self.written += len(y)

    def close(self) -> None:
        self.put(None, RATE)
        self.wav.close()
        os.replace(self.part, self.path)

    def discard(self) -> None:
        self.wav.close()
        self.part.unlink(missing_ok=True)


def _routes(container: InputContainer, outputs: int) -> tuple[list[Any], bool]:
    """The audio tracks to read, and whether a stereo track's two channels are the two people."""
    audio = list(container.streams.audio)
    if outputs == 1:
        return audio[:1], False
    if len(audio) >= outputs:
        return audio[:outputs], False  # Known limit: tracks after the second are not read.
    if audio[0].codec_context.layout.nb_channels == outputs:
        return audio[:1], True
    channels = audio[0].codec_context.layout.nb_channels
    raise InputError("E_INPUT_TRACKS", f"{len(audio)} audio track(s) with {channels} channel(s)")


def decode(
    container: InputContainer, outs: Path | list[Path], progress: Callable[[int], None], *, lead: int = 0
) -> tuple[list[int], int, int]:
    """Write the audio as 16 kHz mono 16-bit PCM with no tags: one WAV of all channels averaged, or, given
    two outputs, one WAV per person (the two channels of a stereo track, or the first two audio tracks).

    `lead` samples of silence come first. Returns the samples written per WAV, and the clipped and total
    samples of the source (every channel).
    """
    paths = [outs] if isinstance(outs, Path) else outs
    streams, split = _routes(container, len(paths))
    target = {s.index: k for k, s in enumerate(streams)}
    seconds = float(streams[0].duration * streams[0].time_base) if streams[0].duration else None
    writers: list[_Writer] = []
    clipped = total = 0
    last = time.monotonic()
    try:
        writers = [_Writer(p) for p in paths]
        for w in writers:
            w.silence(lead)
        for packet in container.demux(streams):
            for frame in packet.decode():
                x = _samples(frame)
                clipped += int(np.count_nonzero(np.abs(x) >= CLIP_LEVEL))
                total += x.size
                # Channels are averaged here, not by the resampler, whose stereo-to-mono mix adds 3 dB.
                if split:
                    for k, w in enumerate(writers):
                        w.put(x[k], frame.sample_rate)
                else:
                    writers[target[packet.stream.index]].put(x.mean(axis=0), frame.sample_rate)
                if seconds and time.monotonic() - last >= PROGRESS_EVERY_S:
                    progress(min(99, int(100 * (writers[0].written - lead) / RATE / seconds)))
                    last = time.monotonic()
        for w in writers:
            w.close()
    except (av.FFmpegError, ValueError) as exc:  # ValueError: the rate or channels change inside the track
        for w in writers:
            w.discard()
        raise InputError("E_INPUT_DAMAGED") from exc
    return [w.written for w in writers], clipped, total


def mix(tracks: list[Path], out: Path) -> int:
    """Average the track WAVs into one; a shorter track is silent after its end. Returns the samples."""
    total = max(sf.info(str(p)).frames for p in tracks)
    part = out.with_name(out.name + ".part")
    readers = [sf.SoundFile(str(p)) for p in tracks]
    try:
        with sf.SoundFile(part, "w", samplerate=RATE, channels=1, subtype="PCM_16", format="WAV") as wav:
            for start in range(0, total, BLOCK):
                block = np.zeros(min(BLOCK, total - start), dtype=np.float32)
                for r in readers:
                    x = r.read(len(block), dtype="float32")
                    block[: len(x)] += x
                wav.write(block / len(readers))
    finally:
        for r in readers:
            r.close()
    os.replace(part, out)
    return total


def _one(path: Path, kind: Kind, outs: list[Path], progress: Callable[[int], None], lead: int = 0) -> tuple:
    with probe.open_input(path, kind) as container:
        source = probe.describe(container, kind)
        return (source, *decode(container, outs, progress, lead=lead))


def run(job: Path, progress: Callable[[int], None]) -> dict[str, Any]:
    """The media stage: decode input/source.<ext> (and source_2.<ext>) and describe the recording."""
    record = read_json(job / "job.json")
    limits = Limits.model_validate(record["settings"]["limits"])
    layout = record.get("layout") or Layout.MIXED
    folder = job / "input"
    wav = folder / "audio16k.wav"
    kind = KINDS[record["input"]["container"]]
    source_path = folder / f"source.{kind.ext}"
    extra: dict[str, Any] = {}
    found: list[str] = []
    if layout == Layout.MIXED:
        source, (written,), clipped, total = _one(source_path, kind, [wav], progress)
        lossy = source.lossy
    else:
        paths = [folder / "track_1.wav", folder / "track_2.wav"]
        if layout == Layout.PER_TRACK:
            source, lengths, clipped, total = _one(source_path, kind, paths, progress)
            origins = ["channel 1", "channel 2"] if source.tracks == 1 else ["audio track 1", "audio track 2"]
            lossy = source.lossy
        else:  # two-files: the second file starts offset_ms after the first
            second = KINDS[record["input"]["second"]["container"]]
            lead = int(record.get("offset_ms") or 0) * RATE // 1000
            source, (n1,), c1, t1 = _one(source_path, kind, paths[:1], lambda p: progress(p // 2))
            source_2, (n2,), c2, t2 = _one(
                folder / f"source_2.{second.ext}", second, paths[1:], lambda p: progress(50 + p // 2), lead
            )  # fmt: skip
            lengths, clipped, total = [n1, n2], c1 + c2, t1 + t2
            origins = ["file 1", "file 2"]
            lossy = source.lossy or source_2.lossy
            extra["source_2"] = asdict(source_2)
            if abs(n1 - n2) * 1000 > LENGTHS_DIFFER_MS * RATE:
                found.append("W_TRACK_LENGTHS")
        written = mix(paths, wav)
        extra["tracks"] = [
            {"track": k + 1, "from": origins[k], "duration_ms": n * 1000 // RATE}
            for k, n in enumerate(lengths)
        ]
    # Run here only after the user confirmed a long recording, so only the hard limits can still refuse it.
    warnings = check_duration(written / RATE, limits, confirmed=True) + found
    measures = quality.measure(wav)
    if quality.silent(measures):
        raise InputError(
            "E_INPUT_SILENT", f"level {measures['level_dbfs']} dBFS, SNR {measures['snr_db']} dB"
        )
    measures["clipping_pct"] = round(100 * clipped / total, 3)
    measures["class"] = quality.classify(measures, lossy=lossy)
    music = quality.music_spans(wav)
    if music:
        warnings.append("W_BACKGROUND_MUSIC")
    progress(100)
    return {
        "status": "ok",
        "duration_ms": written * 1000 // RATE,
        "layout": layout,
        "source": asdict(source),
        **extra,
        "quality": measures,
        "music_spans": music,
        "warnings": warnings,
    }
