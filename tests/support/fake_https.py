"""A local web server that answers the product's https downloads, so tests never use the network."""

from __future__ import annotations

import http.client
import threading
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fluentfolio.models import download


class Server(ThreadingHTTPServer):
    """Serves bodies by path, the next one per request when there are several; a number is an HTTP error."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.bodies: dict[str, list[bytes | int]] = {}
        self.redirects: dict[str, str] = {}
        self.ranges = True
        self.seen: list[tuple[str, str | None]] = []


class Handler(BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        served = self.server
        served.seen.append((self.path, self.headers.get("Range")))
        if self.path in served.redirects:
            self.send_response(302)
            self.send_header("Location", served.redirects[self.path])
            self.end_headers()
            return
        bodies = served.bodies.get(self.path)
        if not bodies:
            self.send_error(404)
            return
        body = bodies.pop(0) if len(bodies) > 1 else bodies[0]
        if isinstance(body, int):
            self.send_error(body)
            return
        asked = self.headers.get("Range")
        start = int(asked.removeprefix("bytes=").removesuffix("-")) if asked and served.ranges else 0
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start:])

    def log_message(self, *args: object) -> None:
        pass


class LocalHttps(urllib.request.AbstractHTTPHandler):
    """Answers every https address from the local server, so the product's own opener runs unchanged."""

    handler_order = 400  # before the real https handler

    def __init__(self, port: int) -> None:
        super().__init__()
        self.port = port

    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(lambda host, **kw: http.client.HTTPConnection("127.0.0.1", self.port, **kw), req)


@contextmanager
def local_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Server]:
    """A running server that every https download of the product reaches instead of the internet."""
    served = Server()
    thread = threading.Thread(target=served.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    product_opener = download.https_opener

    def local_opener() -> urllib.request.OpenerDirector:
        opener = product_opener()
        opener.add_handler(LocalHttps(served.server_address[1]))
        return opener

    monkeypatch.setattr(download, "https_opener", local_opener)
    monkeypatch.setenv("NO_PROXY", "*")  # the test's dead proxy stays set for everything else
    try:
        yield served
    finally:
        served.shutdown()
        served.server_close()
