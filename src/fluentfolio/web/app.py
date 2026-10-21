"""The web app: its pages and actions call the same facade as the command line."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.telemetry import TelemetryConfig
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp

from fluentfolio.app import facade
from fluentfolio.core.errors import AppError, Entry, catalogue
from fluentfolio.core.files import size_text
from fluentfolio.core.logs import crash_lines, setup_app_log
from fluentfolio.core.paths import data_root
from fluentfolio.ingest.upload import display_name
from fluentfolio.web import pages, upload, views
from fluentfolio.web.render import HERE, TEMPLATES, length_text, page
from fluentfolio.web.security import (
    ALLOWED_HOSTS,
    AUTH_PATH,
    SESSION_COOKIE,
    Guard,
    SecurityHeaders,
    plain_page,
    same_secret,
)
from fluentfolio.web.tasks import Tasks

# An error card's one next step, by the error's kind: the catalogue's hints are commands, for the CLI.
# A decision only the learner makes is asked on the form the request came from, which has the id "form".
NEXT_STEP = {
    "input": ("Choose another recording", "/"),
    "needs_user": ("Go back to the form", "#form"),
    "setup": ("Open Components", "/components"),
    "busy": ("Open History", "/jobs"),
    "stage": ("Open History", "/jobs"),
    "internal": ("Run the system check", "/doctor"),
}
FOLDERS = ("Choose another folder", "/settings?tab=advanced#folders")
STEP_OF_CODE = {
    "E_NO_RECORDING": ("Choose a recording", "/"),  # none was chosen, so not "another"
    "E_INPUT_TEXT": ("Go back to the form", "#form"),  # the text is the problem, not the recording
    "E_MOVE_FAILED": FOLDERS, "E_PATH_RULE": FOLDERS,  # raised only by a folder move
}  # fmt: skip
# Codes whose detail is a fixed phrase naming the rule that failed, never a path, so the card can show it.
SHOWN_DETAIL = ("E_MOVE_FAILED", "E_PATH_RULE")
NO_TELEMETRY: TelemetryConfig = {
    "tracing": False, "metrics": False, "logs": False, "operation_spans": False, "auto_configure": False,
}  # fmt: skip
HTTP_STATUS = {"input": 400, "needs_user": 400, "setup": 409, "busy": 409, "stage": 500, "internal": 500}
STATUS_OF_CODE = {"E_INPUT_TOO_LARGE": 413}
NOT_FOUND = catalogue()["E_PAGE_NOT_FOUND"]
SLOTS = ("upload", "second")  # the recording, and the second file of the two-files layout
log = logging.getLogger("fluentfolio.web")


def skip_resets(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """Leave out a connection the browser reset; report everything else as asyncio would.

    A browser that leaves a page while its audio still loads resets that connection. The Windows event loop
    reports it with a traceback in the server's window, which alarms the learner and is not a fault.
    """
    if not isinstance(context.get("exception"), ConnectionResetError):
        loop.default_exception_handler(context)


@asynccontextmanager
async def quiet_resets(_: FastAPI) -> AsyncIterator[None]:
    asyncio.get_running_loop().set_exception_handler(skip_resets)
    yield


def diagnostic() -> str:
    try:
        return facade.diagnostic_report()
    except Exception as exc:  # noqa: BLE001 - the card must still show when the report itself fails
        return f"The diagnostic report could not be made ({type(exc).__name__}). Run the system check."


def http_status(exc: AppError) -> int:
    return STATUS_OF_CODE.get(exc.code) or HTTP_STATUS[exc.entry.category or "internal"]


def error_card(
    request: Request,
    entry: Entry,
    status: int,
    *,
    keep_status: bool = False,
    step: tuple[str, str] | None = None,
    detail: str = "",
) -> HTMLResponse:
    """The error page, or only its card for a request from a page that puts the answer in place.

    Such a card goes out as 200 with its code inside, unless keep_status: htmx puts nothing in place for an
    error status, and the browser writes every error status into its console. The card's heading takes the
    focus, so a screen reader reads what happened.
    """
    label, link = step or STEP_OF_CODE.get(entry.code) or NEXT_STEP[entry.category or "internal"]
    in_place = request.headers.get("hx-request") == "true"
    name = "partials/error_card.html" if in_place else "error.html"
    sent = 200 if in_place and not keep_status else status
    return page(
        request,
        name,
        sent,
        entry=entry,
        detail=detail,
        next_label=label,
        next_link=link,
        report=diagnostic(),
        in_place=in_place,
    )


def upload_facts(found: dict[str, Any], size: int) -> list[tuple[str, str]]:
    """What the probe found, shown before the analysis starts."""
    tracks = f", {found['tracks']} audio tracks" if found["tracks"] > 1 else ""
    return [
        ("Length", length_text(found["duration_ms"])),
        ("Size", size_text(size)),
        ("Format", str(found["container"]).upper()),
        ("Sample rate", f"{found['sample_rate']:,} Hz"),
        ("Channels", f"{found['channels']}{tracks}"),
    ]


def build(origin: str, token: str, session: str, stop: Callable[..., None], tasks: Tasks) -> ASGIApp:
    """The app for one server run: its launch token, its session cookie, how Quit or a data folder move
    stops it, and the long actions it starts, which stop with it."""
    # No API pages: they would load scripts from the internet. FastAPI's OpenTelemetry hooks are all off:
    # Fluentfolio sends nothing anywhere, and they could add exporters from environment variables.
    app = FastAPI(
        docs_url=None, redoc_url=None, openapi_url=None, telemetry=NO_TELEMETRY, lifespan=quiet_resets
    )
    app.state.tasks, app.state.stop = tasks, stop
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.include_router(pages.router)

    @app.get(AUTH_PATH)
    def auth(request: Request, t: str = "") -> HTMLResponse:
        if not same_secret(t, token):
            return plain_page(
                403, "This link is not valid any more. Start Fluentfolio again from its shortcut."
            )
        # A page that moves on to / by itself, not a redirect: after a redirect from the file the browser
        # would hold back a SameSite=Strict cookie on the first page.
        response = page(request, "auth.html")
        response.set_cookie(SESSION_COOKIE, session, httponly=True, samesite="strict", path="/")
        return response

    @app.post("/quit")
    def quit_app(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "partials/stopped.html", background=BackgroundTask(stop))

    @app.post("/upload")
    async def upload_file(request: Request, slot: str = "upload") -> Response:
        if slot not in SLOTS:
            raise HTTPException(400)
        data = data_root()
        limits = facade.settings_show()[0].settings.limits
        path = None
        try:
            upload_id, path = await upload.receive(request, data, limits)
            found = (await run_in_threadpool(facade.probe, [path]))[0]
        except ClientDisconnect:
            return Response(status_code=400)  # the browser went away; nobody reads the answer
        except AppError as exc:
            if path is not None:
                path.unlink(missing_ok=True)
            # A file over the size limit keeps its 413; the upload's own script shows the card either way.
            too_large = exc.code == "E_INPUT_TOO_LARGE"
            return error_card(request, exc.entry, http_status(exc), keep_status=too_large)
        # The browser sends the name percent-encoded; it is display text only and never names a file.
        shown = display_name(unquote(request.headers.get("x-file-name", "")))
        context: dict[str, Any] = {
            "slot": slot, "upload_id": upload_id, "shown": shown, "estimates": None, "memory": ""
        }  # fmt: skip
        if slot == "upload":  # the time each mode takes, and the choices a long recording needs
            found_ms = int(found["duration_ms"])
            times, where, short = await run_in_threadpool(facade.estimates, found_ms)
            context.update(
                estimates=[(views.MODES[m], views.about(ms)) for m, ms in times.items()],
                where=views.DEVICES[where], confirm=found_ms > limits.confirm_min * 60000,
                light=found_ms > limits.long_min * 60000, memory=views.memory_note(short),
            )  # fmt: skip
        return page(
            request, "partials/upload.html", facts=upload_facts(found, path.stat().st_size), **context
        )

    @app.exception_handler(HTTPException)
    def http_error(request: Request, exc: HTTPException) -> HTMLResponse:
        if exc.status_code == 404:  # noqa: PLR2004
            # An old link or a deleted job: History lists what is there.
            return error_card(request, NOT_FOUND, 404, step=("Open History", "/jobs"))
        return plain_page(exc.status_code, "This request was not understood. Reload the page and try again.")

    @app.exception_handler(AppError)
    def app_error(request: Request, exc: AppError) -> HTMLResponse:
        detail = exc.detail if exc.code in SHOWN_DETAIL else ""
        return error_card(request, exc.entry, http_status(exc), detail=detail)

    @app.exception_handler(Exception)
    def internal(request: Request, exc: Exception) -> HTMLResponse:
        try:
            setup_app_log(data_root())
            log.error("E_INTERNAL %s", crash_lines(exc))
        except OSError:
            pass  # the card still shows; only its log line is lost
        return error_card(request, catalogue()["E_INTERNAL"], 500)

    app.add_middleware(Guard, origin=origin, session=session)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    return SecurityHeaders(app)
