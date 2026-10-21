"""The model catalogue's rules, and model downloads: hashes, resume, redirects, gated models and TLS."""

import hashlib
import io
import os
import re
import ssl
import tarfile
import warnings
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import httpx
import huggingface_hub
import pytest
from huggingface_hub import constants, file_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError
from huggingface_hub.utils import logging as hub_logging

from fluentfolio.core.errors import AppError
from fluentfolio.models import catalogue, download
from fluentfolio.models.catalogue import Component, ModelFile
from tests.support.fake_https import Server, local_server

# The IDs the design names: 14 components plus the language check and the grammar model; and the
# pronunciation dictionary, its own download so the pronunciation check runs without MFA.
EXPECTED_IDS = {
    "parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v3", "whisper-large-v3-turbo", "canary-1b-v2", "silero-vad",
    "pyannote-community-1", "sherpa-onnx-diarizer", "mfa", "ctc-aligner", "phone-model", "espeak-ng", "praat",
    "swiftf0", "opensmile", "whisper-tiny", "spacy-en-core-web-sm", "pronunciation-dictionary",
}  # fmt: skip
A = b"first model file " * 1000
B = b"second " * 300


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_the_catalogue_has_every_component_once() -> None:
    assert set(catalogue.components()) == EXPECTED_IDS
    assert [c.id for c in catalogue.components().values() if c.kind == "asr"] == [
        "parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v3", "canary-1b-v2", "whisper-large-v3-turbo",
    ]  # fmt: skip


def test_every_source_is_pinned_and_every_url_is_https_on_an_allowed_first_host() -> None:
    for c in catalogue.components().values():
        if c.host == "huggingface.co":
            assert c.repo and re.fullmatch(r"[0-9a-f]{40}", str(c.revision)), c.id
            assert all(f.url is None for f in c.file), c.id
        for f in c.file:
            assert re.fullmatch(r"[0-9a-f]{64}", f.sha256) and f.size > 0, (c.id, f.path)
            if f.url is not None:
                parts = urlparse(f.url)
                assert (parts.scheme, parts.hostname) in {("https", h) for h in catalogue.URL_HOSTS}, f.url
        if c.install in ("download", "env"):
            assert c.file, c.id
        if c.host not in ("huggingface.co", "pypi", "built-in"):
            assert all(f.url for f in c.file), c.id


def test_licences_needs_and_banned_embedders() -> None:
    every = catalogue.components()
    for c in every.values():
        assert c.requires_consent == (c.licence_class == "nc"), c.id
        assert set(c.needs) <= set(every), c.id
        text = " ".join([c.id, *(f.path for f in c.file), *(str(f.url) for f in c.file)]).lower()
        assert "cnceleb" not in text and "reverb" not in text, c.id
        assert (c.kind == "asr") == (c.transcript_style is not None) == (c.engine is not None), c.id
    assert every["whisper-large-v3-turbo"].transcript_style == "smoothed"
    assert every["parakeet-tdt-0.6b-v2"].default_on and every["silero-vad"].default_on


def test_mfa_is_on_by_default_only_where_it_passed_its_word_timing_check() -> None:
    # MFA's word-timing check failed on both systems, so on both it waits for the learner to switch it on.
    mfa = catalogue.get("mfa")
    assert mfa.experimental_on == ["windows", "linux"]
    assert not mfa.on_by_default("windows") and not mfa.on_by_default("linux")
    assert "experimental" in mfa.note and "Windows or Linux" in mfa.note
    assert [c.id for c in catalogue.components().values() if c.experimental_on] == ["mfa"]


@pytest.mark.parametrize(
    ("default_on", "experimental_on", "on_windows", "on_linux"),
    [(True, [], True, True), (True, ["windows"], False, True), (True, ["windows", "linux"], False, False),
     (False, [], False, False)],
)  # fmt: skip
def test_a_component_is_on_by_default_only_where_it_is_not_experimental(
    default_on: bool, experimental_on: list[str], on_windows: bool, on_linux: bool
) -> None:
    c = component([], default_on=default_on, experimental_on=experimental_on)
    assert (c.on_by_default("windows"), c.on_by_default("linux")) == (on_windows, on_linux)
    with pytest.raises(ValueError):
        component([], experimental_on=["mac"])


def test_the_catalogue_model_refuses_bad_entries() -> None:
    good = {"path": "a.bin", "sha256": sha(A), "size": len(A)}
    for bad in ({"sha256": "abc"}, {"size": 0}, {"extra": 1}, {"os": "mac"}):
        with pytest.raises(ValueError):
            ModelFile.model_validate({**good, **bad})


def component(files: list[ModelFile], **changes: Any) -> Component:
    base: dict[str, Any] = {
        "id": "test-model", "title": "Test model", "note": "", "kind": "asr", "licence": "MIT",
        "licence_class": "permissive", "attribution": "test", "host": "github.com", "device": "cpu",
        "install": "download", "default_on": False, "gated": False, "requires_consent": False, "needs": [],
        "stages": ["asr"], "os": ["windows", "linux"], "file": files,
    }  # fmt: skip
    return Component.model_validate({**base, **changes})


def url_file(name: str, data: bytes, **changes: Any) -> ModelFile:
    url = f"https://github.com/owner/repo/releases/download/v1/{name}"
    return ModelFile.model_validate(
        {"path": name, "sha256": sha(data), "size": len(data), "url": url, **changes}
    )


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Server]:
    with local_server(monkeypatch) as served:
        yield served


@pytest.fixture(autouse=True)
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The waits between network tries, recorded instead of slept."""
    seen: list[float] = []
    monkeypatch.setattr(download, "time", SimpleNamespace(sleep=seen.append))
    return seen


def path_of(f: ModelFile) -> str:
    return urlparse(str(f.url)).path


def test_a_url_file_is_kept_only_whole_and_no_part_file_stays(server: Server, tmp_path: Path) -> None:
    files = [url_file("a.bin", A), url_file("b.bin", B)]
    for f, data in zip(files, (A, B), strict=True):
        server.bodies[path_of(f)] = [data]
    c = component(files)
    download.download(c, tmp_path)
    assert (tmp_path / "other" / "test-model" / "a.bin").read_bytes() == A
    assert download.verify(c, tmp_path) == [] and not list(tmp_path.rglob("*.part"))
    assert download.disk_use(c, tmp_path) == len(A) + len(B)
    assert download.require(c, tmp_path) == tmp_path / "other" / "test-model"
    server.seen.clear()
    download.download(c, tmp_path)  # whole files are not fetched again
    assert server.seen == []
    download.remove(c, tmp_path)
    assert not (tmp_path / "other" / "test-model").exists() and download.disk_use(c, tmp_path) == 0


def test_a_partial_download_continues_where_it_stopped(server: Server, tmp_path: Path) -> None:
    f = url_file("a.bin", A)
    server.bodies[path_of(f)] = [A]
    part = tmp_path / "other" / "test-model" / "a.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(A[:5000])
    download.download(component([f]), tmp_path)
    assert server.seen == [(path_of(f), "bytes=5000-")]
    assert (part.parent / "a.bin").read_bytes() == A and not part.exists()


def test_the_bytes_on_disk_are_heard_as_they_arrive_with_a_continued_part_file_counted(
    server: Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download, "CHUNK", 4096)
    f = url_file("a.bin", A)  # 17,000 bytes
    server.bodies[path_of(f)] = [A]
    part = tmp_path / "other" / "test-model" / "a.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(A[:5000])
    heard: list[int] = []
    download.download(component([f]), tmp_path, on_bytes=heard.append)
    assert heard == [9096, 13192, 17000]  # 5,000 on disk already, two blocks of 4,096, then the last 3,808
    download.remove(component([f]), tmp_path)
    heard.clear()
    download.download(component([f]), tmp_path, on_bytes=heard.append)
    assert heard == [4096, 8192, 12288, 16384, 17000]


def test_hub_progress_hears_the_bytes_so_far_on_the_http_and_the_xet_paths() -> None:
    from huggingface_hub.utils._xet_progress_reporting import XetDownloadProgressReporter

    heard: list[int] = []
    counter = download._counter(heard.append)
    bar = counter(total=1000, initial=400, desc="a.bin", unit="B")  # an http download continued at 400 bytes
    for block in (300, 300):  # the http path counts each block as written and as received
        bar.update(block)
        bar.update_transfer(block)
    assert heard == [700, 700, 1000, 1000]
    heard.clear()
    # Xet reports the bytes written and the bytes received apart, and writes in large pieces: one bar takes
    # the more of the two, so the count never goes back.
    reporter = XetDownloadProgressReporter(
        reconstruction_desc="a.bin", total=1000, log_level=30, tqdm_class=counter
    )
    for written, received in ((0, 300), (200, 600), (1000, 1000)):
        reporter.update_progress(SimpleNamespace(
            total_bytes_completed=written, total_transfer_bytes_completed=received, total_bytes=1000,
            total_bytes_completion_rate=None, total_transfer_bytes_completion_rate=None,
        ))  # fmt: skip
    reporter.close()
    assert heard == [300, 300, 600, 1000, 1000]


def test_a_server_without_ranges_sends_the_whole_file_again(server: Server, tmp_path: Path) -> None:
    f = url_file("a.bin", A)
    server.bodies[path_of(f)] = [A]
    server.ranges = False
    part = tmp_path / "other" / "test-model" / "a.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(A[:5000])
    download.download(component([f]), tmp_path)
    assert (part.parent / "a.bin").read_bytes() == A


def flipped(data: bytes) -> bytes:
    return data[:100] + bytes([data[100] ^ 1]) + data[101:]


def test_a_flipped_byte_is_fetched_twice_then_fails_and_leaves_nothing(
    server: Server, tmp_path: Path
) -> None:
    f = url_file("a.bin", A)
    server.bodies[path_of(f)] = [flipped(A)]
    with pytest.raises(AppError) as caught:
        download.download(component([f]), tmp_path)
    assert caught.value.code == "E_HASH_MISMATCH"
    assert len(server.seen) == 2
    assert not [p for p in (tmp_path / "other").rglob("*") if p.is_file()]


def test_one_bad_copy_then_a_good_one_is_kept(server: Server, tmp_path: Path) -> None:
    f = url_file("a.bin", A)
    server.bodies[path_of(f)] = [flipped(A), A]
    download.download(component([f]), tmp_path)
    assert (tmp_path / "other" / "test-model" / "a.bin").read_bytes() == A and len(server.seen) == 2


def test_redirects_are_followed_only_over_https(server: Server, tmp_path: Path) -> None:
    f = url_file("a.bin", A)
    server.redirects[path_of(f)] = "https://objects.example.org/blob/a"
    server.bodies["/blob/a"] = [A]
    download.download(component([f]), tmp_path)
    assert (tmp_path / "other" / "test-model" / "a.bin").read_bytes() == A
    g = url_file("b.bin", B)
    server.redirects[path_of(g)] = "http://objects.example.org/blob/b"
    server.bodies["/blob/b"] = [B]
    with pytest.raises(AppError) as caught:
        download.download(component([g]), tmp_path)
    assert caught.value.code == "E_DOWNLOAD_UNSAFE"
    assert "/blob/b" not in [p for p, _ in server.seen]


def test_an_address_without_https_is_refused_before_any_request(server: Server, tmp_path: Path) -> None:
    f = url_file("a.bin", A, url="http://github.com/owner/repo/a.bin")
    with pytest.raises(AppError) as caught:
        download.download(component([f]), tmp_path)
    assert caught.value.code == "E_DOWNLOAD_UNSAFE" and server.seen == []


def test_a_missing_file_on_the_server_is_a_network_error(server: Server, tmp_path: Path) -> None:
    with pytest.raises(AppError) as caught:
        download.download(component([url_file("a.bin", A)]), tmp_path)
    assert (caught.value.code, caught.value.detail) == ("E_NETWORK", "github.com, HTTP 404")
    assert len(server.seen) == 3


def test_a_network_error_is_tried_three_times_and_the_partial_file_continues(
    server: Server, tmp_path: Path, waits: list[float]
) -> None:
    f = url_file("a.bin", A)
    server.bodies[path_of(f)] = [503, 503, A]
    part = tmp_path / "other" / "test-model" / "a.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(A[:5000])
    download.download(component([f]), tmp_path)
    assert (part.parent / "a.bin").read_bytes() == A and waits == [5.0, 10.0]
    assert [r for _, r in server.seen] == ["bytes=5000-"] * 3
    g = url_file("b.bin", B)
    server.bodies[path_of(g)] = [503, 503, 503, B]
    with pytest.raises(AppError) as caught:
        download.download(component([g]), tmp_path)
    assert caught.value.code == "E_NETWORK" and len(server.seen) == 6


def test_too_little_disk_space_stops_before_any_request(
    server: Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Usage:
        free = len(A) + len(B) - 1  # one byte short

    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: Usage)
    files = [url_file("a.bin", A), url_file("b.bin", B)]
    with pytest.raises(AppError) as caught:
        download.download(component(files), tmp_path)
    assert caught.value.code == "E_DISK_FULL" and server.seen == []
    Usage.free += 1  # exactly enough
    for f, data in zip(files, (A, B), strict=True):
        server.bodies[path_of(f)] = [data]
    download.download(component(files), tmp_path)
    assert download.verify(component(files), tmp_path) == []


def test_an_archive_is_unpacked_within_its_size_budget(server: Server, tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as archive:
        member = tarfile.TarInfo("model/weights.onnx")
        member.size = len(B)
        archive.addfile(member, io.BytesIO(B))
    data = buffer.getvalue()
    f = url_file("model.tar.bz2", data, unpacked_size=len(B))
    server.bodies[path_of(f)] = [data]
    download.download(component([f]), tmp_path)
    assert (tmp_path / "other" / "test-model" / "model" / "weights.onnx").read_bytes() == B


def test_analysis_finds_missing_or_damaged_files_without_downloading(tmp_path: Path) -> None:
    f = url_file("a.bin", A)
    c = component([f])
    with pytest.raises(AppError) as caught:
        download.require(c, tmp_path)
    assert caught.value.code == "E_MODEL_MISSING"
    target = tmp_path / "other" / "test-model" / "a.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(A[:-1])
    with pytest.raises(AppError) as caught:
        download.require(c, tmp_path)
    assert caught.value.code == "E_HASH_MISMATCH"
    target.write_bytes(flipped(A))  # right size, wrong content: only the full check sees it
    assert download.require(c, tmp_path) == target.parent and download.verify(c, tmp_path) == ["a.bin"]


def test_a_component_in_the_venv_is_found_in_its_package() -> None:
    vad = catalogue.get("silero-vad")
    folder = download.require(vad, Path("unused"))
    assert (folder / vad.file[0].path).is_file() and download.verify(vad, Path("unused")) == []
    with pytest.raises(ValueError):
        download.download(vad, Path("unused"))


REV = "a" * 40


def hub_component(gated: bool = False) -> Component:
    f = ModelFile(path="onnx/model.onnx", sha256=sha(A), size=len(A))
    return component([f], host="huggingface.co", repo="org/name", revision=REV, gated=gated)


def http_error(status: int, cls: type[HfHubHTTPError] = HfHubHTTPError) -> HfHubHTTPError:
    response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/org/name"))
    return cls(f"HTTP {status}", response=response)


def tls_error() -> OSError:
    error = OSError("connection failed")
    error.__cause__ = ssl.SSLError(1, "CERTIFICATE_VERIFY_FAILED")
    return error


class FakeHub:
    """Stands in for hf_hub_download: writes the snapshot file or raises the next planned error."""

    def __init__(self, models: Path, outcomes: list[bytes | Exception]) -> None:
        self.models = models
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def __call__(self, repo: str, filename: str, **kw: Any) -> str:
        self.calls.append({**kw, "repo": repo, "filename": filename, "xet_off": constants.HF_HUB_DISABLE_XET})
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        target = download.hf_folder(self.models, repo) / "snapshots" / kw["revision"] / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(outcome)
        return str(target)


def hub(monkeypatch: pytest.MonkeyPatch, models: Path, *outcomes: bytes | Exception) -> FakeHub:
    fake = FakeHub(models, list(outcomes))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    return fake


def hub_code(c: Component, models: Path) -> str:
    with pytest.raises(AppError) as caught:
        download.download(c, models)
    return caught.value.code


def test_a_hub_file_is_pinned_and_only_a_gated_one_gets_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = hub(monkeypatch, tmp_path, A)
    download.download(hub_component(), tmp_path)
    download.download(hub_component(gated=True), tmp_path / "gated", on_bytes=print)
    # The hub's own bars are replaced only when the bytes are heard; that bar draws nothing.
    assert fake.calls[0]["tqdm_class"] is None and fake.calls[1]["tqdm_class"](total=1).disable
    assert [(c["revision"], c["token"], c["cache_dir"]) for c in fake.calls] == [
        (REV, False, tmp_path / "hf"), (REV, True, tmp_path / "gated" / "hf"),
    ]  # fmt: skip
    path = tmp_path / "hf" / "models--org--name" / "snapshots" / REV / "onnx" / "model.onnx"
    assert path.read_bytes() == A and download.require(hub_component(), tmp_path) == path.parent.parent


def test_a_hub_archive_is_unpacked_outside_the_cache_and_a_repair_puts_it_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as wheel:
        wheel.writestr("pkg/pkg-1.0/config.cfg", B)
    data = buffer.getvalue()
    f = ModelFile(path="pkg-any.whl", sha256=sha(data), size=len(data), unpacked_size=len(B))
    c = component([f], host="huggingface.co", repo="org/name", revision=REV)
    fake = hub(monkeypatch, tmp_path, data)
    download.download(c, tmp_path)
    config = tmp_path / "other" / "test-model" / "pkg" / "pkg-1.0" / "config.cfg"
    assert config.read_bytes() == B and download.disk_use(c, tmp_path) == len(data) + len(B)
    (tmp_path / "other" / "test-model" / "pkg" / "pkg-1.0" / "config.cfg").unlink()
    download.download(c, tmp_path)  # the archive is whole, so it is not fetched again, only unpacked
    assert config.read_bytes() == B and len(fake.calls) == 1
    download.remove(c, tmp_path)
    assert not config.exists() and download.disk_use(c, tmp_path) == 0
    assert not (tmp_path / "hf" / "models--org--name").exists()


def test_a_bad_hub_copy_is_forced_once_more_then_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = hub(monkeypatch, tmp_path, flipped(A), A)
    download.download(hub_component(), tmp_path)
    assert [c["force_download"] for c in fake.calls] == [False, True]
    fake = hub(monkeypatch, tmp_path / "again", flipped(A))
    assert hub_code(hub_component(), tmp_path / "again") == "E_HASH_MISMATCH"
    assert len(fake.calls) == 2 and not [p for p in (tmp_path / "again").rglob("*") if p.is_file()]


def test_gated_model_errors_get_their_own_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = hub(monkeypatch, tmp_path, http_error(403, GatedRepoError))
    with pytest.raises(AppError) as caught:
        download.download(hub_component(gated=True), tmp_path)
    assert len(fake.calls) == 1
    assert (caught.value.code, caught.value.detail) == (
        "E_HF_TERMS_NOT_ACCEPTED",
        "https://huggingface.co/org/name",
    )
    hub(monkeypatch, tmp_path, http_error(401))
    assert hub_code(hub_component(gated=True), tmp_path) == "E_TOKEN_INVALID"
    hub(monkeypatch, tmp_path, LocalTokenNotFoundError("no token"))
    assert hub_code(hub_component(gated=True), tmp_path) == "E_TOKEN_MISSING"
    hub(monkeypatch, tmp_path, http_error(401))  # an open model never asks for a token
    assert hub_code(hub_component(), tmp_path) == "E_NETWORK"
    fake = hub(monkeypatch, tmp_path, http_error(503))  # only a network error is tried again
    assert hub_code(hub_component(gated=True), tmp_path) == "E_NETWORK" and len(fake.calls) == 3


def test_a_tls_error_retries_once_without_xet_and_restores_the_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)
    fake = hub(monkeypatch, tmp_path, tls_error(), A)
    download.download(hub_component(), tmp_path)
    assert [c["xet_off"] for c in fake.calls] == [False, True] and constants.HF_HUB_DISABLE_XET is False
    fake = hub(monkeypatch, tmp_path / "again", tls_error())
    assert hub_code(hub_component(), tmp_path / "again") == "E_TLS_INTERCEPTED"
    assert len(fake.calls) == 2 and constants.HF_HUB_DISABLE_XET is False


def test_the_hubs_own_warnings_never_reach_the_learners_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # Without Developer Mode, Windows makes no links, and the cache warns with the full models path in its
    # text; the Hub sends a notice to requests without a token. Both reached the install page's progress line.
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_SYMLINKS_WARNING", False)
    monkeypatch.setattr(os, "symlink", refuse_link)
    fake = hub(monkeypatch, tmp_path, A)

    def noisy(repo: str, filename: str, **kw: Any) -> str:
        file_download.are_symlinks_supported(kw["cache_dir"])
        notices = hub_logging.get_logger("huggingface_hub.utils._http")
        notices.warning("You are sending unauthenticated requests")
        return fake(repo, filename, **kw)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", noisy)
    level = hub_logging.get_verbosity()
    hub_logging.set_verbosity_warning()
    try:
        with warnings.catch_warnings(record=True) as shown:
            warnings.simplefilter("always")
            download.download(hub_component(), tmp_path)
    finally:
        hub_logging.set_verbosity(level)
    assert len(fake.calls) == 1 and not [str(w.message)[:80] for w in shown]
    assert "unauthenticated" not in capfd.readouterr().err


def refuse_link(*args: object, **kwargs: object) -> None:
    raise OSError("links are not allowed here")


@pytest.mark.network  # one real download from the internet: the pronunciation dictionary, one 7.2 MB file
def test_one_real_download_keeps_the_listed_file_whole_and_leaves_no_part_file(tmp_path: Path) -> None:
    c = catalogue.components()["pronunciation-dictionary"]
    download.download(c, tmp_path)
    (f,) = c.file
    path = download.local_path(c, f, tmp_path)
    assert path.stat().st_size == f.size and hashlib.sha256(path.read_bytes()).hexdigest() == f.sha256
    assert download.verify(c, tmp_path) == [] and not list(tmp_path.rglob("*.part"))
