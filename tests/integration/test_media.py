"""The media stage on real files: the decode matrix, bad inputs, known-answer quality measures, no speech."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
import soundfile as sf

from fluentfolio import __version__
from fluentfolio.app import facade
from fluentfolio.core.enums import JobState, Status
from fluentfolio.core.errors import AppError
from fluentfolio.core.files import read_json
from fluentfolio.ingest import upload
from fluentfolio.jobs import loop, store
from fluentfolio.media import decode, probe, quality
from fluentfolio.settings.model import Settings
from fluentfolio.settings.store import snapshot

SECONDS = 12.0
ON, PERIOD = 0.25, 0.4  # bursts: 250 ms on, 150 ms off
DUTY = ON / PERIOD
AMP = 0.4
MEDIA = loop.Stage(id="media", result="stages/media.json", base_s=120.0, per_min_s=10.0)
# Frequencies 100-4000 Hz on the 512-point grid of 31.25 Hz: white noise puts this share of its power there.
BAND_SHARE = len([k for k in range(257) if 100 <= k * 31.25 <= 4000]) / 256

# (container, encoder, rate, channels, extension): every type and rate the decoder must handle.
MATRIX = [
    ("mp4", "aac", 44100, 2, "mp4"),
    ("ipod", "aac", 48000, 1, "m4a"),
    ("mov", "aac", 48000, 2, "mov"),
    ("ogg", "libopus", 48000, 2, "opus"),
    ("webm", "libopus", 48000, 1, "webm"),
    ("mp3", "libmp3lame", 44100, 2, "mp3"),
    ("flac", "flac", 48000, 1, "flac"),
    ("wav", "pcm_s16le", 8000, 1, "wav"),
    ("wav", "pcm_s16le", 44100, 2, "wav"),
    ("wav", "pcm_f32le", 96000, 2, "wav"),
]
LOSSY = ("aac", "libopus", "libmp3lame")
# FFmpeg's default decoder for each encoder's output; for MP3 it is the floating-point one.
DECODER = {"aac": "aac", "libopus": "opus", "libmp3lame": "mp3float", "flac": "flac",
           "pcm_s16le": "pcm_s16le", "pcm_f32le": "pcm_f32le"}  # fmt: skip


def gate(rate: int, seconds: float = SECONDS) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (t % PERIOD) < ON


def tone(rate: int, freq: float, amp: float, seconds: float = SECONDS) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return amp * np.sin(2 * np.pi * freq * t)


def white(n: int, seed: int, rms: float) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n) * rms


def write_wav(path: Path, x: np.ndarray, rate: int = 16000, subtype: str = "PCM_16") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x.T if x.ndim == 2 else x, rate, subtype=subtype)
    return path


def encode(
    path: Path, fmt: str, codec: str, rate: int, x: np.ndarray, tags: dict[str, str] | None = None
) -> Path:
    """Encode channels x samples with PyAV's own encoders, as a recording app would."""
    path.parent.mkdir(parents=True, exist_ok=True)
    layout = "mono" if x.shape[0] == 1 else "stereo"
    with av.open(str(path), "w", format=fmt) as out:
        out.metadata.update(tags or {})
        stream = out.add_stream(codec, rate=rate, layout=layout)
        for i in range(0, x.shape[1], 1024):
            block = np.ascontiguousarray(x[:, i : i + 1024], dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=layout)
            frame.sample_rate, frame.pts = rate, i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return path


def tone_db(y: np.ndarray, freq: float) -> float:
    """The amplitude of one frequency in a 16 kHz signal, in dB, from a single-frequency Fourier sum."""
    n = np.arange(len(y))
    return float(20 * np.log10(2 * abs(np.sum(y * np.exp(-2j * np.pi * freq * n / 16000))) / len(y)))


def lag_ms(y: np.ndarray) -> float:
    """How far the bursts in y sit from where they were made, by their on-off pattern, in 1/16 ms steps."""
    ref = gate(16000)[: len(y)].astype(float)
    env = np.abs(y[: len(ref)])
    # The circular cross-correlation at every shift at once; the first best shift in -320..320 wins.
    score = np.fft.irfft(np.fft.rfft(env) * np.conj(np.fft.rfft(ref)), n=len(ref))
    shifts = np.arange(-320, 321)
    return float(shifts[np.argmax(score[shifts])]) / 16


@pytest.fixture
def roots(tmp_path: Path) -> loop.Roots:
    return loop.Roots(app=tmp_path / "app", data=tmp_path / "data", models=tmp_path / "app" / "models")


def new_job(roots: loop.Roots, source: Path) -> Path:
    now = store.utc_now()
    record = store.JobRecord(
        job_id=store.new_job_id(), created_utc=now, updated_utc=now, app_version=__version__,
        device={"requested": "cpu", "resolved": "cpu"}, settings=snapshot(Settings()),
    )  # fmt: skip
    job = store.create(roots.data, record)
    record.input = upload.store_input(job, source, Settings().limits)
    store.write(roots.data, record)
    return job


def media(roots: loop.Roots, source: Path) -> dict:
    """Ingest a file and run the media stage in this process."""
    return decode.run(new_job(roots, source), lambda pct: None)


def code_of(roots: loop.Roots, source: Path) -> str:
    try:
        media(roots, source)
    except AppError as error:
        return error.code
    return "ok"


@pytest.mark.parametrize(("fmt", "codec", "rate", "channels", "ext"), MATRIX)
def test_every_type_and_rate_decodes_to_16_khz_mono_at_the_same_times(
    roots: loop.Roots, tmp_path: Path, fmt: str, codec: str, rate: int, channels: int, ext: str
) -> None:
    # Left 440 Hz, right 1000 Hz (or both in one channel), in 250 ms bursts.
    low, high = tone(rate, 440, AMP) * gate(rate), tone(rate, 1000, AMP) * gate(rate)
    x = np.stack([low, high]) if channels == 2 else (low + high)[None, :]
    source = encode(tmp_path / "in" / f"recording.{ext}", fmt, codec, rate, x)
    result = media(roots, source)
    wav = next(roots.data.glob("jobs/*/input/audio16k.wav"))
    info = sf.info(str(wav))
    assert (info.samplerate, info.channels, info.subtype, info.format) == (16000, 1, "PCM_16", "WAV")
    assert result["source"]["sample_rate"] == rate and result["source"]["channels"] == channels
    y, _ = sf.read(wav)
    # AAC leaves up to one 1024-sample block of padding at the end (10.7 ms at 48 kHz, measured).
    assert (
        abs(result["duration_ms"] - SECONDS * 1000) <= 15 and result["duration_ms"] == len(y) * 1000 // 16000
    )
    assert abs(lag_ms(y)) <= 1.0  # encoder delays are removed, so word times match the original
    # Averaging two channels halves each tone. Lossless types keep the level exactly. MP3 loses 0.445 dB at
    # the burst edges in its own encoder (the same loss shows when it is decoded alone at its own rate).
    expected = 20 * np.log10(AMP * DUTY / channels)
    tolerance = 0.6 if result["source"]["lossy"] else 0.01
    for freq in (440, 1000):
        assert abs(tone_db(y, freq) - expected) <= tolerance, (freq, tone_db(y, freq) - expected)


def test_tags_in_the_container_are_not_copied(roots: loop.Roots, tmp_path: Path) -> None:
    x = (tone(44100, 440, AMP) * gate(44100))[None, :]
    source = encode(tmp_path / "in" / "tagged.m4a", "ipod", "aac", 44100, x, {"title": "TAG-4417-HERE"})
    assert b"TAG-4417-HERE" in source.read_bytes()
    media(roots, source)
    assert b"TAG-4417-HERE" not in next(roots.data.glob("jobs/*/input/audio16k.wav")).read_bytes()


def test_a_recording_is_opened_with_its_own_reader_and_local_files_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[dict[str, Any]] = []
    monkeypatch.setattr(probe.av, "open", lambda path, **options: opened.append(options))
    probe.open_input(tmp_path / "talk.m4a", upload.MP4)
    assert opened == [{"format": "mov", "options": {"protocol_whitelist": "file"}}]  # the reader is forced
    # The mov reader's external references stay off: nothing may switch them on.
    assert "enable_drefs" not in opened[0]["options"]


def test_bad_inputs_get_their_own_error_codes(roots: loop.Roots, tmp_path: Path) -> None:
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "random.wav").write_bytes(np.random.default_rng(7).bytes(4096))
    (folder / "list.mp4").write_bytes(b"#EXTM3U\n#EXTINF:-1,x\nhttp://example.com/a.wav\n")
    (folder / "concat.mp4").write_bytes(b"ffconcat version 1.0\nfile 'a.wav'\n")
    with av.open(str(folder / "video.mp4"), "w", format="mp4") as out:  # a picture track and no sound
        stream = out.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(20):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 10, np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    write_wav(folder / "empty.wav", np.zeros(0))
    write_wav(folder / "short.wav", tone(16000, 440, 0.3, 9.99) * gate(16000, 9.99))
    write_wav(folder / "ten.wav", tone(16000, 440, 0.3, 10.0) * gate(16000, 10.0))
    codes = {path.name: code_of(roots, path) for path in sorted(folder.iterdir())}
    assert codes == {
        "concat.mp4": "E_INPUT_UNSUPPORTED", "list.mp4": "E_INPUT_UNSUPPORTED",
        "random.wav": "E_INPUT_UNSUPPORTED", "video.mp4": "E_INPUT_NO_AUDIO",
        "empty.wav": "E_INPUT_TOO_SHORT", "short.wav": "E_INPUT_TOO_SHORT", "ten.wav": "ok",
    }  # fmt: skip


def test_the_probe_child_reads_each_type_and_its_length_before_any_job(tmp_path: Path) -> None:
    files = []
    for fmt, codec, rate, channels, ext in MATRIX:
        x = (tone(rate, 440, AMP) * gate(rate))[None, :].repeat(channels, axis=0)
        files.append(encode(tmp_path / "in" / f"{fmt}_{rate}_{channels}.{ext}", fmt, codec, rate, x))
    for path, (_, codec, rate, channels, _) in zip(files, MATRIX, strict=True):
        (found,) = facade.probe([path])
        assert (found["sample_rate"], found["channels"], found["tracks"]) == (rate, channels, 1), path.name
        assert found["codec"] == DECODER[codec] and found["lossy"] == (codec in LOSSY)
        # The same bound as for the decoded length: AAC may pad its last 1024-sample block.
        assert abs(found["duration_ms"] - SECONDS * 1000) <= 15, (path.name, found["duration_ms"])
    assert not (Path(os.environ["FLUENTFOLIO_HOME"]) / "jobs").exists()
    # Two files are read in one child, in the order given.
    pair = facade.probe([files[-1], files[0]])
    assert [p["sample_rate"] for p in pair] == [96000, 44100]


def test_the_probe_child_gives_the_code_of_a_bad_file(tmp_path: Path) -> None:
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "random.wav").write_bytes(np.random.default_rng(7).bytes(4096))
    whole = encode(folder / "whole.m4a", "ipod", "aac", 48000, (tone(48000, 440, AMP) * gate(48000))[None, :])
    (folder / "cut.m4a").write_bytes(whole.read_bytes()[: whole.stat().st_size // 2])
    for name, code in (("random.wav", "E_INPUT_UNSUPPORTED"), ("cut.m4a", "E_INPUT_DAMAGED")):
        with pytest.raises(AppError) as caught:
            facade.probe([folder / name])
        assert caught.value.code == code, name
    # One bad file of two refuses both, before any job exists.
    with pytest.raises(AppError) as caught:
        facade.probe([whole, folder / "random.wav"])
    assert caught.value.code == "E_INPUT_UNSUPPORTED"


def test_a_truncated_mp4_fails_its_step_and_the_app_goes_on(roots: loop.Roots, tmp_path: Path) -> None:
    x = (tone(48000, 440, AMP) * gate(48000))[None, :]
    whole = encode(tmp_path / "in" / "whole.m4a", "ipod", "aac", 48000, x)
    cut = tmp_path / "in" / "cut.m4a"
    cut.write_bytes(whole.read_bytes()[: whole.stat().st_size // 2])
    jobs = {name: new_job(roots, path) for name, path in (("cut", cut), ("whole", whole))}
    run = {
        name: loop.run_job(roots, job.name, [MEDIA], audio_min=0.2, no_progress_s=60)
        for name, job in jobs.items()
    }
    assert (run["cut"].state, run["cut"].stages[0].status) == (JobState.FAILED, Status.FAILED)
    assert run["cut"].stages[0].error_code == "E_INPUT_DAMAGED"
    assert read_json(jobs["cut"] / "stages" / "media.json")["code"] == "E_INPUT_DAMAGED"
    assert not (jobs["cut"] / "input" / "audio16k.wav").exists()
    # The same process runs the next job: the failure stayed inside the child.
    assert run["whole"].state is JobState.COMPLETED
    done = read_json(jobs["whole"] / "stages" / "media.json")
    assert done["status"] == "ok" and abs(done["duration_ms"] - SECONDS * 1000) <= 15
    assert done["peak_ram_mb"] > 0 and run["whole"].stages[0].peak_ram_mb == done["peak_ram_mb"]


def pink(n: int, seed: int, rms: float) -> np.ndarray:
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    spectrum[1:] /= np.sqrt(np.fft.rfftfreq(n, 1 / 16000)[1:])  # power falls as 1/f
    spectrum[0] = 0
    x = np.fft.irfft(spectrum, n=n)
    return x / np.sqrt(np.mean(x**2)) * rms


def test_silence_noise_or_a_steady_chord_means_no_speech(roots: loop.Roots, tmp_path: Path) -> None:
    n = 20 * 16000
    chord = sum(tone(16000, f, 0.1, 20.0) for f in (261.63, 329.63, 392.0))
    signals = {
        "silence": np.zeros(n),
        "pink": pink(n, 1, 0.03),
        "white": white(n, 2, 0.03),
        "chord": chord,
        # Speech-like bursts as loud as the noise under them (0 dB) still count as speech.
        "bursts at 0 dB": white(n, 3, 0.03) * gate(16000, 20.0) / np.sqrt(DUTY) + white(n, 4, 0.03),
    }
    codes = {
        name: code_of(roots, write_wav(tmp_path / "in" / f"{name}.wav", x)) for name, x in signals.items()
    }
    assert codes == {
        "silence": "E_INPUT_SILENT", "pink": "E_INPUT_SILENT", "white": "E_INPUT_SILENT",
        "chord": "E_INPUT_SILENT", "bursts at 0 dB": "ok",
    }  # fmt: skip
    job = new_job(roots, tmp_path / "in" / "pink.wav")  # and through the real child, as a job would run it
    record = loop.run_job(roots, job.name, [MEDIA], audio_min=0.4, no_progress_s=60)
    assert (record.state, record.stages[0].error_code) == (JobState.FAILED, "E_INPUT_SILENT")


def test_level_and_snr_estimates_match_signals_of_known_power(tmp_path: Path) -> None:
    sine = tone(16000, 1000, 0.1 * np.sqrt(2))  # RMS 0.1 = -20 dBFS, all inside the speech band
    assert quality.measure(write_wav(tmp_path / "sine.wav", sine, subtype="FLOAT"))[
        "level_dbfs"
    ] == pytest.approx(-20.0, abs=0.05)
    for snr in (10.0, 25.0, 40.0):
        burst = tone(16000, 1000, 0.1 * np.sqrt(2)) * gate(16000)
        floor_rms = 0.1 / np.sqrt(BAND_SHARE) * 10 ** (-snr / 20)  # its in-band power is snr below the tone's
        x = burst + white(len(burst), int(snr), floor_rms)
        q = quality.measure(write_wav(tmp_path / f"snr{snr}.wav", x, subtype="FLOAT"))
        expected = 10 * np.log10(1 + 10 ** (snr / 10))  # the louder frames hold tone plus noise
        # The floor is the 10th percentile of noise-only frames, which reads up to about 1 dB low.
        assert expected - 0.2 <= q["snr_db"] <= expected + 1.2, (snr, q["snr_db"])
        assert q["noise_dbfs"] == pytest.approx(-20 - snr, abs=1.2)


def test_the_band_limit_is_where_the_spectrum_stops(tmp_path: Path) -> None:
    x = white(int(16000 * SECONDS), 5, 0.1)
    spectrum = np.fft.rfft(x)
    spectrum[np.fft.rfftfreq(len(x), 1 / 16000) > 3000] = 0
    low = np.fft.irfft(spectrum, n=len(x))
    assert (
        3000
        <= quality.measure(write_wav(tmp_path / "low.wav", low, subtype="FLOAT"))["band_limit_hz"]
        <= 3300
    )
    assert quality.measure(write_wav(tmp_path / "full.wav", x, subtype="FLOAT"))["band_limit_hz"] >= 7900
    narrow = write_wav(tmp_path / "phone.wav", white(int(8000 * SECONDS), 6, 0.1), rate=8000)
    kind = upload.sniff_file(narrow)
    with probe.open_input(narrow, kind) as container:
        decode.decode(container, tmp_path / "phone16k.wav", lambda pct: None)
    # Nothing above 4 kHz; the resampler's roll-off carries the edge a little higher (4625 Hz when measured),
    # still inside the phone rule.
    assert 3800 <= quality.measure(tmp_path / "phone16k.wav")["band_limit_hz"] <= quality.NARROW_UP_TO_HZ


def test_clipped_samples_are_counted_in_the_source(roots: loop.Roots, tmp_path: Path) -> None:
    x = tone(16000, 440, 0.3) * gate(16000) + white(int(16000 * SECONDS), 7, 0.001)
    fine = write_wav(tmp_path / "in" / "fine.wav", x)
    hit = gate(16000) & (np.arange(len(x)) % 4 == 0)  # every fourth sample inside the bursts, at full scale
    x[hit] = 1.0
    expected = round(100 * np.count_nonzero(hit) / len(x), 3)
    assert media(roots, write_wav(tmp_path / "in" / "clipped.wav", x))["quality"]["clipping_pct"] == expected
    assert media(roots, fine)["quality"]["clipping_pct"] == 0.0


def speechlike(rate: int, snr_db: float | None, seed: int) -> np.ndarray:
    """Broadband bursts like syllables, over a white floor snr_db below them (no floor: digital silence)."""
    n = int(rate * SECONDS)
    x = white(n, seed, 0.1) * gate(rate)
    return x if snr_db is None else x + white(n, seed + 100, 0.1 * 10 ** (-snr_db / 20))


def gated(path: Path, fmt: str, codec: str) -> Path:
    """Speech-like bursts with digital silence between them, as noise suppression leaves a recording."""
    return encode(path, fmt, codec, 48000, speechlike(48000, None, 5)[None, :])


@pytest.mark.parametrize(
    ("name", "make", "expected"),
    [
        ("clean", lambda d: write_wav(d / "a.wav", speechlike(16000, 45, 1)), "clean"),
        ("laptop", lambda d: write_wav(d / "a.wav", speechlike(16000, 28, 2)), "laptop"),
        ("noisy", lambda d: write_wav(d / "a.wav", speechlike(16000, 12, 3)), "noisy"),
        ("phone", lambda d: write_wav(d / "a.wav", speechlike(8000, 45, 4), rate=8000), "phone"),
        ("gated, lossy", lambda d: gated(d / "a.webm", "webm", "libopus"), "meeting_codec"),
        ("gated, lossless", lambda d: gated(d / "a.flac", "flac", "flac"), "clean"),
    ],
)
def test_each_quality_class_from_a_recording_made_for_it(
    roots: loop.Roots, tmp_path: Path, name: str, make: Callable[[Path], Path], expected: str
) -> None:
    assert media(roots, make(tmp_path / "in"))["quality"]["class"] == expected, name


def test_the_class_rules_apply_in_order() -> None:
    base = {"snr_db": 40.0, "noise_dbfs": -60.0, "band_limit_hz": 8000}
    cases = [
        ({}, False, "clean"),
        ({"snr_db": 35.0}, False, "clean"),
        ({"snr_db": 34.9}, False, "laptop"),
        ({"snr_db": 20.0}, False, "laptop"),
        ({"snr_db": 19.9}, False, "noisy"),
        ({"band_limit_hz": 5000}, False, "phone"),
        ({"band_limit_hz": 5031}, False, "clean"),
        ({"noise_dbfs": -75.0}, True, "meeting_codec"),
        ({"noise_dbfs": -74.9}, True, "clean"),
        ({"noise_dbfs": -90.0}, False, "clean"),  # a gated floor in a lossless file is just quiet
        ({"snr_db": 12.0, "band_limit_hz": 4000}, False, "noisy"),  # noise lowers more than a narrow band
        ({"noise_dbfs": -90.0, "band_limit_hz": 4000}, True, "meeting_codec"),
    ]
    for change, lossy, expected in cases:
        assert quality.classify(base | change, lossy=lossy) == expected, change


def test_the_composed_noisy_session_is_classed_noisy_and_the_clean_one_clean(
    roots: loop.Roots, built: Callable[[str], tuple[Path, Any]]
) -> None:
    classes = {}
    for name in ("CS-INT", "CS-NOISY"):
        classes[name] = media(roots, built(name)[0] / "session.wav")["quality"]["class"]
    assert classes == {"CS-INT": "clean", "CS-NOISY": "noisy"}


def test_music_under_speech_warns_and_the_same_speech_without_it_does_not(
    roots: loop.Roots, built: Callable[[str], tuple[Path, Any]]
) -> None:
    found = {name: media(roots, built(name)[0] / "session.wav") for name in ("CS-MUSIC", "CS-SOLO", "CS-INT")}
    music = found["CS-MUSIC"]
    assert "W_BACKGROUND_MUSIC" in music["warnings"] and music["music_spans"]
    end = built("CS-MUSIC")[1]["music"]["end_ms"]  # the music plays from the start to the end
    assert all(0 <= a < b <= end for a, b in music["music_spans"])
    for name in ("CS-SOLO", "CS-INT"):  # the same speech without music, and the composed interview
        assert ("W_BACKGROUND_MUSIC" in found[name]["warnings"], found[name]["music_spans"]) == (False, []), (
            name
        )


def test_a_data_folder_with_spaces_and_letters_outside_a_to_z_works(
    roots: loop.Roots, tmp_path: Path
) -> None:
    x = (tone(48000, 440, AMP) * gate(48000))[None, :]
    source = encode(tmp_path / "Zoë Ü folder" / "answer.m4a", "ipod", "aac", 48000, x)
    data = tmp_path / "Dàta Ünï"
    job = new_job(loop.Roots(roots.app, data, roots.models), source)
    record = loop.run_job(
        loop.Roots(roots.app, data, roots.models), job.name, [MEDIA], audio_min=0.2, no_progress_s=60
    )
    assert record.state is JobState.COMPLETED and (job / "input" / "audio16k.wav").is_file()


def layout_job(
    roots: loop.Roots, layout: str, source: Path, second: Path | None = None, offset_ms: int = 0
) -> Path:
    """A job with one track per person: a file with two channels or tracks, or two files and an offset."""
    now = store.utc_now()
    record = store.JobRecord(
        job_id=store.new_job_id(), created_utc=now, updated_utc=now, app_version=__version__,
        mode="interview", layout=layout, offset_ms=offset_ms if second else None,
        device={"requested": "cpu", "resolved": "cpu"}, settings=snapshot(Settings()),
    )  # fmt: skip
    job = store.create(roots.data, record)
    record.input = upload.store_input(job, source, Settings().limits)
    if second is not None:
        record.input["second"] = upload.store_input(job, second, Settings().limits, stem="source_2")
    store.write(roots.data, record)
    return job


def encode_tracks(path: Path, rate: int, tracks: list[np.ndarray]) -> Path:
    """A Matroska file with one lossless mono audio track per signal, as a recorder with two tracks writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", format="matroska") as out:
        streams = [out.add_stream("flac", rate=rate, layout="mono") for _ in tracks]
        for i in range(0, len(tracks[0]), 1024):
            for stream, x in zip(streams, tracks, strict=True):
                block = np.ascontiguousarray(x[None, i : i + 1024], dtype=np.float32)
                frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="mono")
                frame.sample_rate, frame.pts = rate, i
                for packet in stream.encode(frame):
                    out.mux(packet)
        for stream in streams:
            for packet in stream.encode(None):
                out.mux(packet)
    return path


def read(job: Path, name: str) -> np.ndarray:
    y, rate = sf.read(job / "input" / name)
    assert rate == 16000
    return np.asarray(y)


@pytest.mark.parametrize("kind", ["stereo", "two tracks"])
def test_one_track_per_person_gives_one_wav_each_and_their_average(
    roots: loop.Roots, tmp_path: Path, kind: str
) -> None:
    low, high = tone(16000, 440, AMP) * gate(16000), tone(16000, 1000, AMP) * gate(16000)
    if kind == "stereo":
        source = write_wav(tmp_path / "in" / "two.wav", np.stack([low, high]))
    else:
        source = encode_tracks(tmp_path / "in" / "two.mkv", 16000, [low, high])
    job = layout_job(roots, "per-track", source)
    result = decode.run(job, lambda pct: None)
    origin = ["channel 1", "channel 2"] if kind == "stereo" else ["audio track 1", "audio track 2"]
    assert result["layout"] == "per-track"
    assert result["tracks"] == [
        {"track": 1, "from": origin[0], "duration_ms": 12000},
        {"track": 2, "from": origin[1], "duration_ms": 12000},
    ]
    full = 20 * np.log10(AMP * DUTY)  # each person's track keeps its tone at its own level
    one, two, both = read(job, "track_1.wav"), read(job, "track_2.wav"), read(job, "audio16k.wav")
    assert abs(tone_db(one, 440) - full) <= 0.01 and tone_db(one, 1000) < full - 60
    assert abs(tone_db(two, 1000) - full) <= 0.01 and tone_db(two, 440) < full - 60
    # The main WAV averages the two tracks, so each tone is half as loud there: 6.02 dB lower.
    for freq in (440, 1000):
        assert abs(tone_db(both, freq) - (full - 20 * np.log10(2))) <= 0.01
    assert abs(lag_ms(one)) <= 1.0 and abs(lag_ms(two)) <= 1.0


def test_one_track_per_person_needs_two_channels_or_two_tracks(roots: loop.Roots, tmp_path: Path) -> None:
    signal = tone(16000, 440, AMP, 3.0)
    mono = write_wav(tmp_path / "in" / "mono.wav", signal)
    three = write_wav(tmp_path / "in" / "three.wav", np.stack([signal] * 3))
    for source in (mono, three):
        with pytest.raises(AppError) as caught:
            decode.run(layout_job(roots, "per-track", source), lambda pct: None)
        assert caught.value.code == "E_INPUT_TRACKS"
    assert not list(roots.data.glob("jobs/*/input/track_*"))  # nothing half-written is left


@pytest.mark.parametrize(("shorter_ms", "warned"), [(1990, False), (2000, False), (2010, True)])
def test_two_files_line_up_by_the_offset_and_lengths_warn(
    roots: loop.Roots, tmp_path: Path, shorter_ms: int, warned: bool
) -> None:
    # The second file starts 1.5 s into the session, so it holds the session from 1.5 s on.
    signal = tone(16000, 1000, AMP) * gate(16000)
    first = write_wav(tmp_path / "in" / "first.wav", tone(16000, 440, AMP) * gate(16000))
    end = len(signal) - shorter_ms * 16
    second = write_wav(tmp_path / "in" / "second.flac", signal[24000:end])
    job = layout_job(roots, "two-files", first, second, offset_ms=1500)
    result = decode.run(job, lambda pct: None)
    assert ("W_TRACK_LENGTHS" in result["warnings"]) == warned
    assert [t["from"] for t in result["tracks"]] == ["file 1", "file 2"]
    assert result["tracks"][1]["duration_ms"] == 12000 - shorter_ms  # 1.5 s of silence plus the file
    two = read(job, "track_2.wav")
    assert not two[:24000].any()  # the offset is silence
    assert abs(lag_ms(two)) <= 1.0  # its bursts sit where the session made them
