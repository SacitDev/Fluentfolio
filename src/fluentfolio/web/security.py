"""The local web server's guards: one signed-in browser, writes only from its own pages, strict headers.

Only this computer can reach the server (it listens on 127.0.0.1), but any web page open in a browser can
still send requests to it. So every request needs the session cookie that the launch link sets, and every
change also needs proof that it came from one of the app's own pages.
"""

from __future__ import annotations

import hmac

from starlette.datastructures import Headers
from starlette.requests import cookie_parser
from starlette.responses import HTMLResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

SESSION_COOKIE = "fluentfolio_session"
AUTH_PATH = "/auth"
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
WRITES = frozenset({"POST", "PUT", "PATCH", "DELETE"})
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
HEADERS = [
    (b"content-security-policy", CSP.encode()),
    (b"cache-control", b"no-store"),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-resource-policy", b"same-origin"),
]
SET_HERE = {name for name, _ in HEADERS}
# A second start from the shortcut opens the running app in this browser, signed in.
SIGNED_OUT = "This browser is not signed in to Fluentfolio. Open Fluentfolio again from its shortcut."
FOREIGN = (
    "This change was refused because it did not come from a Fluentfolio page. Reload the page and try again."
)


def same_secret(given: str, expected: str) -> bool:
    """Compare in constant time; as bytes, because compare_digest refuses text with non-ASCII letters."""
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def plain_page(status: int, text: str) -> HTMLResponse:
    """A page with no styles or scripts, for a refusal that comes before the app's own pages.

    The color-scheme line lets the browser draw it in the system's light or dark colours.
    """
    # Known limit: these pages use the browser's own font, as the app's font is served only after sign-in;
    # revisit if one of them needs more than one plain sentence.
    body = (
        '<!doctype html>\n<html lang="en"><meta charset="utf-8">'
        '<meta name="color-scheme" content="light dark">'  # an empty icon: the browser asks for no other file
        f'<title>Fluentfolio</title><link rel="icon" href="data:,"><p>{text}</p>\n'
    )
    return HTMLResponse(body, status_code=status)


class SecurityHeaders:
    """The outermost layer: every reply gets the headers, refusals and server errors included."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                kept = [(k, v) for k, v in message.get("headers", []) if k.lower() not in SET_HERE]
                message = {**message, "headers": kept + HEADERS}
            await send(message)

        await self.app(scope, receive, with_headers)


class Guard:
    """Refuse a request without the session cookie, and a write that did not come from the app's pages."""

    def __init__(self, app: ASGIApp, origin: str, session: str) -> None:
        self.app = app
        self.origin = origin  # built once at start, never read from the request
        self.session = session

    def refusal(self, scope: Scope) -> str | None:
        headers = Headers(scope=scope)
        if scope["path"] != AUTH_PATH:
            cookies = cookie_parser(headers.get("cookie", ""))
            if not same_secret(cookies.get(SESSION_COOKIE, ""), self.session):
                return SIGNED_OUT
        if scope["method"] in WRITES:
            # Another port on 127.0.0.1 counts as the same site, so only "same-origin" is enough.
            site = headers.get("sec-fetch-site")
            own = site == "same-origin" if site is not None else headers.get("origin") == self.origin
            if not own or headers.get("hx-request") != "true":
                return FOREIGN
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":  # the app has no websocket, so none is accepted
            await send({"type": "websocket.close", "code": 1008})
            return
        refused = self.refusal(scope) if scope["type"] == "http" else None
        if refused is None:
            await self.app(scope, receive, send)
        else:
            await plain_page(403, refused)(scope, receive, send)
