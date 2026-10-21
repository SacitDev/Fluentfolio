"""The test-data fetcher against a local web server: hash checks, mirrors, the cache and safe unpacking."""

import hashlib
import http.server
import io
import subprocess
import sys
import tarfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "fetch_test_data.py"


class Files(http.server.BaseHTTPRequestHandler):
    served: dict[str, bytes] = {}
    hits: list[str] = []

    def do_GET(self) -> None:  # noqa: N802
        self.hits.append(self.path)
        body = self.served.get(self.path)
        self.send_response(200 if body is not None else 404)
        self.end_headers()
        self.wfile.write(body or b"")

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    Files.served, Files.hits = {}, []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Files)
    # A short poll lets the server stop at once; the default makes each test wait up to 0.5 s at the end.
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def manifest(tmp_path: Path, urls: list[str], body: bytes, extra: str = "", sha: str | None = None) -> Path:
    path = tmp_path / "manifest.toml"
    digest = sha or hashlib.sha256(body).hexdigest()
    path.write_text(
        f'[[file]]\nname = "clip"\nurls = {urls!r}\nsize = {len(body)}\nsha256 = "{digest}"\n'
        f'licence = "CC-BY-4.0"\ncredit = "test"\nsave_as = "set/clip.bin"\n{extra}',
        encoding="utf-8",
    )
    return path


def run(path: Path, dest: Path) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(TOOL), "--manifest", str(path), "--dest", str(dest)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)


def test_downloads_checks_and_then_uses_the_copy(server: str, tmp_path: Path) -> None:
    Files.served["/a.bin"] = b"x" * 5000
    path = manifest(tmp_path, [f"{server}/a.bin"], b"x" * 5000)
    first = run(path, tmp_path / "data")
    assert (first.returncode, first.stdout.strip()) == (0, "clip: downloaded")
    assert (tmp_path / "data" / "set" / "clip.bin").read_bytes() == b"x" * 5000
    second = run(path, tmp_path / "data")
    assert (second.returncode, second.stdout.strip()) == (0, "clip: present")
    assert Files.hits == ["/a.bin"]


def test_a_wrong_hash_keeps_nothing(server: str, tmp_path: Path) -> None:
    Files.served["/a.bin"] = b"tampered"
    path = manifest(tmp_path, [f"{server}/a.bin"], b"tampered", sha="0" * 64)
    done = run(path, tmp_path / "data")
    assert done.returncode == 1
    assert "SHA-256" in done.stderr
    assert list((tmp_path / "data" / "set").iterdir()) == []


def test_the_next_mirror_is_tried_when_one_fails(server: str, tmp_path: Path) -> None:
    Files.served["/b.bin"] = b"mirror"
    path = manifest(tmp_path, [f"{server}/missing.bin", f"{server}/b.bin"], b"mirror")
    done = run(path, tmp_path / "data")
    assert (done.returncode, Files.hits) == (0, ["/missing.bin", "/b.bin"])


def test_an_archive_cannot_write_outside_its_folder(server: str, tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"hi"))
    Files.served["/c.tgz"] = buffer.getvalue()
    path = manifest(tmp_path, [f"{server}/c.tgz"], buffer.getvalue(), extra='unpack_to = "set/out"\n')
    done = run(path, tmp_path / "data")
    assert done.returncode == 1
    assert not (tmp_path / "data" / "set" / "escaped.txt").exists()


def test_an_unpack_cut_short_is_finished_by_the_next_run(server: str, tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("a.flac", "b.flac"):
            info = tarfile.TarInfo(f"set/{name}")
            info.size, info.mode = 4, 0o444  # read-only, as in LibriSpeech
            archive.addfile(info, io.BytesIO(name[0].encode() * 4))
    Files.served["/d.tgz"] = buffer.getvalue()
    path = manifest(tmp_path, [f"{server}/d.tgz"], buffer.getvalue(), extra='unpack_to = "out"\n')
    assert run(path, tmp_path / "data").returncode == 0
    (tmp_path / "data" / "out" / ".clip.unpacked").unlink()  # the run stopped before it finished
    (tmp_path / "data" / "out" / "set" / "b.flac").chmod(0o644)
    (tmp_path / "data" / "out" / "set" / "b.flac").unlink()
    again = run(path, tmp_path / "data")
    assert (again.returncode, again.stdout.strip()) == (0, "clip: present, unpacked")
    assert (tmp_path / "data" / "out" / "set" / "b.flac").read_bytes() == b"bbbb"


def test_a_save_path_outside_the_folder_is_refused(server: str, tmp_path: Path) -> None:
    path = manifest(tmp_path, [f"{server}/a.bin"], b"x")
    path.write_text(path.read_text(encoding="utf-8").replace("set/clip.bin", "../clip.bin"), encoding="utf-8")
    done = run(path, tmp_path / "data")
    assert (done.returncode, Files.hits) == (1, [])
    assert "not a safe relative path" in done.stderr


def test_plain_http_is_refused_for_other_hosts(tmp_path: Path) -> None:
    path = manifest(tmp_path, ["http://example.org/a.bin"], b"x")
    done = run(path, tmp_path / "data")
    assert done.returncode == 1
    assert "only https" in done.stderr
