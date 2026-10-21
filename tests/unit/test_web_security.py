"""The web server's guards, through the real app: sign-in, writes from its own pages, hosts and headers."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from opentelemetry import trace
from starlette.routing import Mount
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from fluentfolio.core.errors import ResourceError, catalogue
from fluentfolio.core.paths import data_root
from fluentfolio.web import app as web_app
from fluentfolio.web.security import SESSION_COOKIE
from fluentfolio.web.tasks import Tasks

ORIGIN = "http://127.0.0.1:8765"
TOKEN = "launch-token-for-tests"  # noqa: S105 (a made-up launch token, not a password)
SESSION = "session-for-tests"
CANARY = "zorblat"  # a made-up word in an error's text; it must not reach the page or the log
# Written out from the requirement, not taken from the code under test.
EXPECTED_HEADERS = {
    "content-security-policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'none'"
    ),
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "cross-origin-resource-policy": "same-origin",
}
HTMX_CONFIG = (
    '<meta name="htmx-config" content=\'{"historyCacheSize":0,"allowEval":false,"allowScriptTags":false,'
    '"includeIndicatorStyles":false,"selfRequestsOnly":true}\'>'
)
NAV = ["Fluentfolio", "New analysis", "History", "Components", "Settings", "Help", "Quit"]
OWN_PAGE = {"sec-fetch-site": "same-origin", "hx-request": "true"}


@pytest.fixture
def stops() -> list[int]:
    return []


@pytest.fixture
def server(stops: list[int]) -> tuple[FastAPI, TestClient]:
    """The FastAPI app inside the guards, and a client that talks to the whole stack."""
    outer = web_app.build(ORIGIN, TOKEN, SESSION, lambda: stops.append(1), Tasks(data_root()))
    inner = outer.app  # type: ignore[attr-defined]
    assert isinstance(inner, FastAPI)
    return inner, TestClient(outer, base_url=ORIGIN, raise_server_exceptions=False)


def signed_in(client: TestClient) -> TestClient:
    client.cookies.set(SESSION_COOKIE, SESSION)
    return client


def has_headers(reply: object) -> None:
    headers = reply.headers  # type: ignore[attr-defined]
    for name, value in EXPECTED_HEADERS.items():
        assert headers[name] == value, name
    assert not [h for h in headers if h.lower().startswith("access-control-")]


def walk(app: FastAPI) -> list[tuple[str, str]]:
    """One request per route and method, with a stand-in for each path parameter."""
    found = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            path = re.sub(r"\{[^}]+\}", "x", route.path)
            found += [(method, path) for method in sorted(route.methods)]
        elif isinstance(route, Mount):
            found.append(("GET", f"{route.path}/css/app.css"))
    return found


def test_every_route_answers_403_without_the_session_cookie(server: tuple[FastAPI, TestClient]) -> None:
    app, client = server
    requests = walk(app)
    assert {("GET", "/auth"), ("POST", "/quit"), ("GET", "/static/css/app.css")} <= set(requests)
    for method, path in [*requests, ("GET", "/no-such-page")]:
        if path == "/auth":
            continue
        for cookie in (None, "a-wrong-session", SESSION + "x"):
            client.cookies.clear()
            if cookie:
                client.cookies.set(SESSION_COOKIE, cookie)
            reply = client.request(method, path, headers=OWN_PAGE)
            assert reply.status_code == 403, (method, path, cookie)
            assert "not signed in" in reply.text
            has_headers(reply)


def test_the_launch_link_signs_in_with_a_strict_session_cookie(server: tuple[FastAPI, TestClient]) -> None:
    _, client = server
    for wrong in ("", TOKEN[:-1], TOKEN + "x", TOKEN.upper(), "\u00e4" * len(TOKEN)):
        reply = client.get("/auth", params={"t": wrong})
        assert reply.status_code == 403 and "set-cookie" not in reply.headers, wrong
    reply = client.get("/auth", params={"t": TOKEN})
    assert reply.status_code == 200
    parts = [p.strip() for p in reply.headers["set-cookie"].split(";")]
    assert parts[0] == f"{SESSION_COOKIE}={SESSION}"
    assert {"HttpOnly", "SameSite=strict", "Path=/"} <= set(parts)
    assert not [p for p in parts if p.lower().startswith(("max-age", "expires"))]  # ends with the browser
    assert '<meta http-equiv="refresh" content="0; url=/">' in reply.text and TOKEN not in reply.text
    assert client.get("/static/css/app.css").status_code == 200  # the client keeps the cookie it was given


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1.nip.io", "evil.example:8765", "127.0.0.2:8765"])
def test_a_request_for_another_host_name_gets_400(server: tuple[FastAPI, TestClient], host: str) -> None:
    _, client = server
    reply = signed_in(client).get("/static/css/app.css", headers={"host": host})
    assert reply.status_code == 400
    has_headers(reply)


@pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:8765", "127.0.0.1", "localhost"])
def test_127_0_0_1_and_localhost_are_served(server: tuple[FastAPI, TestClient], host: str) -> None:
    _, client = server
    assert signed_in(client).get("/static/css/app.css", headers={"host": host}).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {"sec-fetch-site": "cross-site", "hx-request": "true"},
        {"sec-fetch-site": "same-site", "hx-request": "true"},  # another port on this computer
        {"sec-fetch-site": "none", "hx-request": "true"},
        {"sec-fetch-site": "cross-site", "origin": ORIGIN, "hx-request": "true"},
        {"origin": "http://evil.example", "hx-request": "true"},
        {"origin": "http://127.0.0.1:9999", "hx-request": "true"},
        {"origin": "null", "hx-request": "true"},
        {"hx-request": "true"},  # neither Sec-Fetch-Site nor Origin
        {"sec-fetch-site": "same-origin"},  # no HX-Request
        {"sec-fetch-site": "same-origin", "hx-request": "false"},
        {"origin": ORIGIN},
    ],
)
def test_a_write_from_outside_the_apps_pages_gets_403_and_changes_nothing(
    server: tuple[FastAPI, TestClient], stops: list[int], headers: dict[str, str]
) -> None:
    _, client = server
    reply = signed_in(client).post("/quit", headers=headers)
    assert reply.status_code == 403 and "did not come from a Fluentfolio page" in reply.text
    has_headers(reply)
    assert stops == []


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
def test_the_other_writing_methods_from_outside_get_403_before_any_route(
    server: tuple[FastAPI, TestClient], stops: list[int], method: str
) -> None:
    _, client = server  # the app has no such route; the guard answers first, so a later one is covered too
    foreign = {"origin": "http://evil.example", "hx-request": "true"}
    reply = signed_in(client).request(method, "/quit", headers=foreign)
    assert reply.status_code == 403 and "did not come from a Fluentfolio page" in reply.text
    assert stops == []


@pytest.mark.parametrize("headers", [OWN_PAGE, {"origin": ORIGIN, "hx-request": "true"}])
def test_quit_from_the_apps_own_page_stops_the_server_after_answering(
    server: tuple[FastAPI, TestClient], stops: list[int], headers: dict[str, str]
) -> None:
    _, client = server
    reply = signed_in(client).post("/quit", headers=headers)
    assert reply.status_code == 200
    # The heading takes the focus when it is put in place, so a screen reader reads it.
    assert '<h1 tabindex="-1" autofocus>Fluentfolio has stopped</h1>' in reply.text
    has_headers(reply)
    assert stops == [1]


def test_a_refusal_page_names_the_one_way_back_and_follows_the_system_theme(
    server: tuple[FastAPI, TestClient],
) -> None:
    _, client = server
    for reply in (client.get("/no-such-page"), client.get("/auth", params={"t": "wrong"})):
        assert reply.status_code == 403
        assert '<meta name="color-scheme" content="light dark">' in reply.text
    said = re.search(r"<p>(.*)</p>", client.get("/no-such-page").text).group(1)  # type: ignore[union-attr]
    assert said == "This browser is not signed in to Fluentfolio. Open Fluentfolio again from its shortcut."


def test_every_page_names_its_icon_so_the_browser_asks_for_no_favicon_file(
    server: tuple[FastAPI, TestClient],
) -> None:
    _, client = server
    # Before sign-in no file of the app is served, so those pages name an empty icon.
    for reply in (client.get("/no-such-page"), client.get("/auth", params={"t": "wrong"}),
                  client.get("/auth", params={"t": TOKEN})):  # fmt: skip
        assert '<link rel="icon" href="data:,">' in reply.text
    signed = signed_in(client)
    assert '<link rel="icon" href="/static/icon.svg" type="image/svg+xml">' in signed.get("/").text
    icon = signed.get("/static/icon.svg")
    assert icon.status_code == 200 and icon.headers["content-type"].startswith("image/svg+xml")
    assert icon.content.startswith(b"<svg ") and b"<script" not in icon.content


@pytest.mark.parametrize("in_place", [False, True])
def test_a_missing_page_leads_to_history_and_a_card_put_in_place_comes_as_200(
    server: tuple[FastAPI, TestClient], in_place: bool
) -> None:
    _, client = server
    reply = signed_in(client).get("/no-such-page", headers=OWN_PAGE if in_place else {})
    # htmx puts nothing in place for an error status, so the card for it comes with 200.
    assert reply.status_code == (200 if in_place else 404)
    # Put in place, the page around the card keeps the one primary button of the view.
    button = "button" if in_place else "button primary"
    assert f'<a class="{button}" href="/jobs">Open History</a>' in reply.text
    assert ("<html" in reply.text) is not in_place
    assert 'class="error-card in-place"' in reply.text if in_place else 'class="error-card"' in reply.text
    assert "Error code E_PAGE_NOT_FOUND" in reply.text  # like every other card
    # Put in place, the card sits under the page's own h1.
    heading = '<h2 id="error-title" tabindex="-1" autofocus>Page not found</h2>'
    assert (heading if in_place else '<h1 id="error-title">Page not found</h1>') in reply.text


@pytest.mark.parametrize(
    ("error", "title", "code"),
    [
        (ResourceError("E_BUSY"), "Another job is running", "E_BUSY"),
        (RuntimeError("not expected"), "Fluentfolio stopped on an error it did not expect", "E_INTERNAL"),
    ],
)
def test_an_error_card_put_in_place_comes_as_200_with_its_code_and_takes_the_focus(
    server: tuple[FastAPI, TestClient], error: Exception, title: str, code: str
) -> None:
    app, client = server

    def fail() -> None:
        raise error

    app.add_api_route("/act", fail, methods=["POST"])
    reply = signed_in(client).post("/act", headers=OWN_PAGE)
    assert reply.status_code == 200 and "<html" not in reply.text
    assert f'<h2 id="error-title" tabindex="-1" autofocus>{title}</h2>' in reply.text
    assert f"Error code {code}" in reply.text
    has_headers(reply)


def test_pages_have_the_exact_htmx_config_the_header_in_order_and_no_inline_code(
    server: tuple[FastAPI, TestClient],
) -> None:
    _, client = server
    reply = signed_in(client).get("/no-such-page")
    assert reply.status_code == 404 and "Page not found" in reply.text
    has_headers(reply)
    page = reply.text
    assert page.count(HTMX_CONFIG) == 1
    assert re.findall(r"<script[^>]*>", page) == [
        '<script src="/static/js/htmx.min.js" defer>', '<script src="/static/js/app.js" defer>'
    ]  # fmt: skip
    assert "<style" not in page and "style=" not in page and "hx-on" not in page
    header = page[page.index("<header") : page.index("</header>")]
    found = re.findall(r"<(a|button)[^>]*>(.*?)</\1>", header, flags=re.DOTALL)
    labels = [re.sub(r"<svg.*?</svg>", "", text).strip() for _, text in found]
    assert labels == NAV
    assert header.count('aria-hidden="true" focusable="false"') == 1  # the Quit icon, beside its word


@pytest.mark.parametrize(
    ("cookie", "tag"),
    [
        (None, '<html lang="en">'),
        ("light", '<html lang="en" data-theme="light">'),
        ("dark", '<html lang="en" data-theme="dark">'),
        ("system", '<html lang="en">'),
        ('dark" onload="alert(1)', '<html lang="en">'),
    ],
)
def test_the_server_writes_the_chosen_theme_on_the_page(
    server: tuple[FastAPI, TestClient], cookie: str | None, tag: str
) -> None:
    _, client = server
    if cookie is not None:
        client.cookies.set("theme", cookie)
    page = signed_in(client).get("/no-such-page").text
    assert re.search(r"<html[^>]*>", page).group(0) == tag  # type: ignore[union-attr]


def test_an_unexpected_error_shows_a_plain_card_and_logs_no_error_text(
    server: tuple[FastAPI, TestClient],
) -> None:
    app, client = server
    # The words live in data, as a learner's would: a traceback shows each frame's source line.
    words = f"{CANARY} in practice/recording.wav"

    def fail() -> None:
        raise RuntimeError(words)

    app.add_api_route("/fail", fail)
    reply = signed_in(client).get("/fail")
    assert reply.status_code == 500
    has_headers(reply)
    page = reply.text
    card = re.sub(r"<textarea.*?</textarea>", "", page, flags=re.DOTALL)
    assert "Fluentfolio stopped on an error it did not expect" in card and "Error code E_INTERNAL" in card
    assert '<a class="button primary" href="/doctor">Run the system check</a>' in card
    assert "RuntimeError" not in card and "File " not in card  # no error type or code lines on the card
    assert 'data-copy="report"' in page and "Fluentfolio diagnostic report" in page
    assert CANARY not in page
    log = (Path(os.environ["FLUENTFOLIO_HOME"]) / "logs" / "app.log").read_text(encoding="utf-8")
    assert "E_INTERNAL RuntimeError" in log and CANARY not in log


def test_an_app_error_shows_its_plain_words_and_one_next_step_never_a_command(
    server: tuple[FastAPI, TestClient],
) -> None:
    app, client = server

    def busy() -> None:
        raise ResourceError("E_BUSY")

    app.add_api_route("/busy", busy)
    reply = signed_in(client).get("/busy")
    assert reply.status_code == 409
    page = reply.text
    assert '<h1 id="error-title">Another job is running</h1>' in page and "Error code E_BUSY" in page
    assert '<a class="button primary" href="/jobs">Open History</a>' in page
    assert catalogue()["E_BUSY"].hint not in page and "fluentfolio cancel" not in page  # the CLI's hint


def test_no_telemetry_exporter_starts_even_when_the_environment_asks_for_one(
    stops: list[int], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # With automatic setup on, FastAPI would add an exporter for these at start: it installs a tracer
    # provider that sends to this address when the OpenTelemetry SDK is present (the speaker-separation
    # packages bring it), and fails or logs about it when the SDK is missing.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    with TestClient(
        web_app.build(ORIGIN, TOKEN, SESSION, lambda: stops.append(1), Tasks(data_root())), base_url=ORIGIN
    ) as client:
        assert signed_in(client).get("/static/css/app.css").status_code == 200
    assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)  # nothing was installed
    assert not [r for r in caplog.records if "telemetry" in r.getMessage().lower()]


def test_a_connection_the_browser_reset_is_left_out_and_any_other_loop_error_is_reported() -> None:
    reported: list[str] = []

    class Loop:
        def default_exception_handler(self, context: dict[str, object]) -> None:
            reported.append(str(context["message"]))

    for context in (
        {"message": "reset", "exception": ConnectionResetError(10054, "forcibly closed")},
        {"message": "other error", "exception": OSError("disk full")},
        {"message": "no exception"},
    ):
        web_app.skip_resets(Loop(), context)  # type: ignore[arg-type]
    assert reported == ["other error", "no exception"]


def test_a_websocket_is_closed_before_it_opens(server: tuple[FastAPI, TestClient]) -> None:
    _, client = server
    url = ORIGIN.replace("http:", "ws:") + "/anything"  # the client's own default host is refused earlier
    with pytest.raises(WebSocketDisconnect) as closed, signed_in(client).websocket_connect(url):
        pass
    assert type(closed.value) is WebSocketDisconnect and closed.value.code == 1008
