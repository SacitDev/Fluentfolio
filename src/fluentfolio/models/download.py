"""Model files on this computer: where they live, whether they are whole, and their download and removal.

Downloads run in the app process only, which holds work.lock, so they never run during a job. Analysis
children only check that a component's files are present (HF_HUB_OFFLINE=1 there).
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from importlib import util
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fluentfolio.core.errors import SetupError
from fluentfolio.core.files import extract, size_text
from fluentfolio.models.catalogue import SYSTEM, Component, ModelFile

CHUNK = 1 << 20
TIMEOUT_S = 60
NETWORK_TRIES = 3
RETRY_WAIT_S = 5.0  # times the try number: 5 s, then 10 s
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
Progress = Callable[[int], None]  # hears how many bytes of the file being fetched are on disk so far


def hf_folder(models: Path, repo: str) -> Path:
    """A repository's folder in the Hugging Face cache, in huggingface_hub's documented layout."""
    return models / "hf" / ("models--" + repo.replace("/", "--"))


def unpacked(c: Component, models: Path) -> Path:
    """Where a component's archives are unpacked, and where its files from other hosts are kept."""
    return models / "other" / c.id


def folder(c: Component, models: Path) -> Path:
    """The folder that holds a component's files on this computer."""
    if c.install == "venv":
        spec = util.find_spec(str(c.package))  # finds the package without importing it
        if spec is None or spec.origin is None:
            raise SetupError("E_MODEL_MISSING", c.id)
        return Path(spec.origin).parent
    if c.host == "huggingface.co":
        return hf_folder(models, str(c.repo)) / "snapshots" / str(c.revision)
    return unpacked(c, models)


def _roots(c: Component, models: Path) -> list[Path]:
    """Every folder a downloaded component writes: its download folder and its unpacked archives."""
    root = hf_folder(models, str(c.repo)) if c.host == "huggingface.co" else unpacked(c, models)
    return list(dict.fromkeys([root, unpacked(c, models)]))


def local_path(c: Component, f: ModelFile, models: Path) -> Path:
    return folder(c, models) / f.path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _whole(path: Path, f: ModelFile) -> bool:
    return path.is_file() and path.stat().st_size == f.size and sha256_of(path) == f.sha256


def require(c: Component, models: Path) -> Path:
    """Check before an analysis that every file is present at its recorded size; returns the files' folder.

    The size check is cheap; the full SHA-256 runs after each download and in verify().
    """
    for f in c.files_for(SYSTEM):
        path = local_path(c, f, models)
        if not path.is_file():
            raise SetupError("E_MODEL_MISSING", c.id)
        if path.stat().st_size != f.size:
            raise SetupError("E_HASH_MISMATCH", f"{c.id}: {f.path}")
    return folder(c, models)


def verify(c: Component, models: Path) -> list[str]:
    """The files that are missing or whose SHA-256 differs from the catalogue."""
    return [f.path for f in c.files_for(SYSTEM) if not _whole(local_path(c, f, models), f)]


def disk_use(c: Component, models: Path) -> int:
    """Bytes the component's downloaded files take on disk."""
    if c.install != "download":
        return 0
    return sum(p.stat().st_size for root in _roots(c, models) for p in root.rglob("*") if p.is_file())


def remove(c: Component, models: Path) -> None:
    """Delete a downloaded component's files; the caller holds work.lock."""
    if c.install == "download":
        for root in _roots(c, models):
            shutil.rmtree(root, ignore_errors=True)


def _check_space(folder: Path, files: list[ModelFile]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    need, free = sum(f.size for f in files), shutil.disk_usage(folder).free
    if need > free:
        raise SetupError("E_DISK_FULL", f"need {size_text(need)}, have {size_text(free)}")


def download(
    c: Component, models: Path, on_file: Callable[[ModelFile], None] | None = None,
    on_bytes: Progress | None = None,
) -> None:  # fmt: skip
    """Fetch every missing or damaged file of a component, check its SHA-256 and keep it.

    The caller holds work.lock. A file that fails its hash twice is deleted and stops the install.
    `on_file` hears of each file before it is fetched, `on_bytes` of its bytes on disk as they arrive.
    """
    if c.install != "download":
        raise ValueError(f"{c.id} is not installed by download")
    todo = [f for f in c.files_for(SYSTEM) if not _whole(local_path(c, f, models), f)]
    _check_space(models, todo)
    for f in todo:
        if on_file:
            on_file(f)
        for again in (False, True):
            path = _fetch(c, f, models, again, on_bytes)
            if _whole(path, f):
                break
            _discard(path)
        else:
            raise SetupError("E_HASH_MISMATCH", f"{c.id}: {f.path}")
    # Unpacked again on every install, so a repair puts back a deleted folder next to a whole archive.
    for f in c.files_for(SYSTEM):
        if f.unpacked_size is not None:
            extract(local_path(c, f, models), unpacked(c, models), f.unpacked_size)


def fetch_file(c: Component, f: ModelFile, target: Path, on_bytes: Progress | None = None) -> None:
    """Fetch one URL file to a place of its own (the MFA files), with the same checks as any download.

    The caller holds work.lock. A whole file is kept as it is; one that fails its hash twice is deleted.
    """
    if _whole(target, f):
        return
    _check_space(target.parent, [f])
    for _ in range(2):
        path = _retried(lambda: _from_url(f, target, on_bytes))
        if _whole(path, f):
            return
        _discard(path)
    raise SetupError("E_HASH_MISMATCH", f"{c.id}: {f.path}")


def _retried(get: Callable[[], Path]) -> Path:
    """One file, tried up to three times while the network fails; a partial URL file continues each time."""
    for attempt in range(1, NETWORK_TRIES):
        try:
            return get()
        except SetupError as error:
            if error.code != "E_NETWORK":
                raise
        time.sleep(RETRY_WAIT_S * attempt)
    return get()  # the last try's error stands


def _fetch(c: Component, f: ModelFile, models: Path, force: bool, on_bytes: Progress | None) -> Path:
    if c.host == "huggingface.co":
        return _retried(lambda: _from_hub(c, f, models, force, on_bytes))
    return _retried(lambda: _from_url(f, local_path(c, f, models), on_bytes))


def _discard(path: Path) -> None:
    target = path.resolve()  # the cache's blob, when the snapshot entry is a link to it
    path.unlink(missing_ok=True)
    target.unlink(missing_ok=True)


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: BaseException | None = exc
    while seen is not None:
        yield seen
        reason = getattr(seen, "reason", None)  # urllib keeps the socket's error here
        if isinstance(reason, BaseException):
            yield reason
        seen = seen.__cause__ or seen.__context__


def is_tls_failure(exc: BaseException) -> bool:
    """A certificate or TLS failure, which a proxy or antivirus that inspects https traffic causes."""
    return any(isinstance(e, ssl.SSLError) for e in _chain(exc))


def _hub_error(c: Component, exc: Exception) -> SetupError:
    from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

    if isinstance(exc, LocalTokenNotFoundError):
        return SetupError("E_TOKEN_MISSING", c.id)
    status = exc.response.status_code if isinstance(exc, HfHubHTTPError) else None
    if c.gated and status == HTTP_UNAUTHORIZED:
        return SetupError("E_TOKEN_INVALID", c.id)
    if c.gated and status == HTTP_FORBIDDEN:
        return SetupError("E_HF_TERMS_NOT_ACCEPTED", f"https://huggingface.co/{c.repo}")
    if is_tls_failure(exc):
        return SetupError("E_TLS_INTERCEPTED", "huggingface.co and its download hosts")
    return SetupError("E_NETWORK", f"huggingface.co, HTTP {status}" if status else "huggingface.co")


def _counter(on_bytes: Progress) -> type:
    """A progress bar class for huggingface_hub that draws nothing and hands on the bytes so far: the more of
    those written and those received, since Xet writes in large pieces while its bytes arrive steadily."""
    from huggingface_hub.utils import tqdm

    class Counter(tqdm):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **{**kwargs, "disable": True})
            self.written = self.received = int(kwargs.get("initial") or 0)  # a partial file that continues

        def update(self, n: float | None = 1) -> bool | None:
            self.written += int(n or 0)
            on_bytes(max(self.written, self.received))
            return super().update(n)

        # With these two, Xet gives this one bar both counts instead of drawing a second bar.
        def update_transfer(self, n: float) -> None:
            self.received += int(n)
            on_bytes(max(self.written, self.received))

        def set_transfer_postfix_str(self, s: str = "", refresh: bool = True) -> None:
            pass

    return Counter


def _hub_call(c: Component, f: ModelFile, models: Path, force: bool, on_bytes: Progress | None) -> Path:
    from huggingface_hub import hf_hub_download

    # Only a gated repository gets the user's token, which huggingface_hub reads from its own file.
    path = hf_hub_download(
        str(c.repo), f.path, revision=c.revision, cache_dir=models / "hf", token=True if c.gated else False,
        force_download=force, tqdm_class=_counter(on_bytes) if on_bytes else None,
    )  # fmt: skip
    return Path(path)


def _from_hub(c: Component, f: ModelFile, models: Path, force: bool, on_bytes: Progress | None) -> Path:
    from huggingface_hub import constants
    from huggingface_hub.utils import logging as hub_logging

    # The cache's warning that Windows makes no links (it names the full models path) and the Hub's notices,
    # such as one about requests without a token, would reach a learner's progress line. Every file is
    # checked by its hash, and a failure becomes one of the app's codes below.
    constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True
    hub_logging.set_verbosity_error()
    try:
        return _hub_call(c, f, models, force, on_bytes)
    except Exception as exc:  # noqa: BLE001 - every library error becomes one of the app's codes below
        error = _hub_error(c, exc)
        if error.code != "E_TLS_INTERCEPTED":
            raise error from exc
    # One retry without the Xet transfer service, whose hosts an https-inspecting proxy may not allow.
    saved, constants.HF_HUB_DISABLE_XET = constants.HF_HUB_DISABLE_XET, True
    try:
        return _hub_call(c, f, models, force, on_bytes)
    except Exception as exc:  # noqa: BLE001 - as above
        raise _hub_error(c, exc) from exc
    finally:
        constants.HF_HUB_DISABLE_XET = saved


class _HttpsOnly(urllib.request.HTTPRedirectHandler):
    """Follow a redirect to any host, but only over https; the SHA-256 decides what is kept."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        if urlparse(newurl).scheme != "https":
            raise SetupError("E_DOWNLOAD_UNSAFE", "a redirect to an address without https")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def https_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_HttpsOnly)


def _from_url(f: ModelFile, target: Path, on_bytes: Progress | None = None) -> Path:
    """Download one file through a .part file, continuing an earlier partial download where it stopped."""
    url = str(f.url)
    if urlparse(url).scheme != "https":
        raise SetupError("E_DOWNLOAD_UNSAFE", "an address without https")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have >= f.size:  # a whole-sized leftover failed its check before: start again
        part.unlink()
        have = 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})  # noqa: S310
    try:
        with https_opener().open(request, timeout=TIMEOUT_S) as response:
            # 206: the server continues the partial file; 200: it sends the whole file again.
            with part.open("ab" if response.status == 206 else "wb") as out:  # noqa: PLR2004
                while block := response.read(CHUNK):
                    out.write(block)
                    if on_bytes:
                        on_bytes(out.tell())
    except urllib.error.HTTPError as exc:
        raise SetupError("E_NETWORK", f"{urlparse(url).hostname}, HTTP {exc.code}") from exc
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            part.unlink(missing_ok=True)
            free = shutil.disk_usage(target.parent).free
            raise SetupError("E_DISK_FULL", f"need {size_text(f.size)}, have {size_text(free)}") from exc
        code = "E_TLS_INTERCEPTED" if is_tls_failure(exc) else "E_NETWORK"
        raise SetupError(code, str(urlparse(url).hostname)) from exc
    os.replace(part, target)
    return target
