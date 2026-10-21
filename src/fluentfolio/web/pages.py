"""The pages and actions of the web app: each reads or acts through the facade and fills one template.

A long action (an analysis, an install, the self-test, the audio file) runs as a command-line child of the
server (web/tasks.py); its page follows it by polling or by server-sent events. GET never changes anything.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import secrets
import shutil
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from fluentfolio.app import facade
from fluentfolio.asr.base import choices
from fluentfolio.core.enums import JobState
from fluentfolio.core.errors import AppError, InputError, ResourceError, catalogue
from fluentfolio.core.files import size_text
from fluentfolio.core.paths import app_root, data_root, install_state, models_dir
from fluentfolio.doctor.checks import executable, token_present
from fluentfolio.export.bundle import NAMES
from fluentfolio.export.for_ai import prompt_part, split_in_two
from fluentfolio.models import catalogue as parts_catalogue
from fluentfolio.setup.components import OPENSMILE_TERMS
from fluentfolio.web import timeline, views
from fluentfolio.web import upload as uploads
from fluentfolio.web.render import TEMPLATES, THEME_COOKIE, clock, length_text, page, upfirst
from fluentfolio.web.tasks import Task, Tasks

FORM_MAX = 5 * 2**20  # pasted questions or a transcript; recordings come through /upload
HISTORY_MAX = 1000
EVENT_EVERY_S = 1.0
OPEN = (JobState.QUEUED, JobState.RUNNING)
FINISHED = (JobState.COMPLETED, JobState.COMPLETED_WITH_GAPS)
TASK_KINDS = ("install", "selftest", "export")
SYNCED = ("onedrive", "dropbox")  # folder names of the sync clients the data folder may sit in
DOWNLOAD_MBIT = 50  # the line speed the download time on the Components page is given for
THEME_DAYS = 400
# Doctor's fixes name commands; the page links to the page that does the same instead.
FIX_PAGES = {
    **dict.fromkeys(
        ("E_MODEL_MISSING", "E_HASH_MISMATCH", "E_SELFTEST_FAILED", "E_COMPONENT_UNAVAILABLE",
         "E_OPENSMILE_CONSENT_MISSING", "W_NOT_OFFLINE"),
        ("Open Components", "/components"),
    ),
    "E_TOKEN_MISSING": ("Add your token", "/token"),
}  # fmt: skip
LEVELS = {"ok": "OK", "warning": "Warning", "problem": "Problem"}
router = APIRouter()


async def form(request: Request) -> dict[str, list[str]]:
    """A posted form, read up to FORM_MAX; it has no file field, so the standard library parses it."""
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if (
            len(body) > FORM_MAX
        ):  # only a pasted text makes a form this large; the card says what a text may be
            raise InputError("E_INPUT_TEXT", f"a form over {FORM_MAX} bytes")
    found: dict[str, list[str]] = {}
    for key, value in parse_qsl(body.decode("utf-8", "replace"), keep_blank_values=True):
        found.setdefault(key, []).append(value)
    return found


def one(found: dict[str, list[str]], key: str, default: str = "") -> str:
    return found.get(key, [default])[-1]


def go(link: str) -> Response:
    """Send the browser to another page after an action that htmx started."""
    return Response(headers={"HX-Redirect": link})


def reload() -> Response:
    return Response(headers={"HX-Refresh": "true"})


def tasks_of(request: Request) -> Tasks:
    return request.app.state.tasks  # type: ignore[no-any-return]


def record_of(job_id: str) -> facade.Analysis:
    """The job, or the not-found page for an old link or a deleted job."""
    try:
        return facade.status(job_id)
    except InputError as exc:
        if exc.code == "E_JOB_NOT_FOUND":
            raise HTTPException(404) from None
        raise


def pasted(text: str) -> Path:
    """Pasted text as a .txt file in the data folder, which the facade reads like a file it was given."""
    folder = data_root() / "tmp"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{secrets.token_hex(16)}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def model_titles() -> dict[str, str]:
    return {c.id: c.title for c in parts_catalogue.components().values()}


@router.get("/")
def new(request: Request) -> Response:
    if not install_state(app_root()).get("selftest", {}).get("passed"):
        return RedirectResponse("/components?first=1", 303)
    policy = facade.settings_show()[0].settings.review.policy  # this run's choice starts as the Settings one
    last = next(iter(facade.jobs(1)), None)  # no mode is chosen for the learner before their first analysis
    return page(
        request, "new.html", nav="new", modes=views.MODES, layouts=views.LAYOUTS, reviews=views.REVIEW_RUN,
        review=policy, diagrams=views.DIAGRAMS, mode=last.mode if last else None,
        layout=last.layout if last and last.mode != "solo" and last.layout else "mixed",
    )  # fmt: skip


@router.post("/jobs")
async def start(request: Request) -> Response:
    found = await form(request)
    mode = one(found, "mode")
    layout = "mixed" if mode == "solo" else one(found, "layout", "mixed")
    if mode and mode not in views.MODES:
        raise InputError("E_INPUT_OPTION", "choose a mode")
    if tasks_of(request).running() is not None:
        raise ResourceError("E_BUSY")
    if not one(found, "upload") or (layout == "two-files" and not one(found, "second")):
        raise InputError("E_NO_RECORDING")  # nothing was chosen, which is not the same as a lost upload
    if not mode:  # the page chooses none before the first analysis, and the tool never guesses one
        raise InputError("E_NO_MODE")
    data = data_root()
    source = uploads.find(data, one(found, "upload"))
    second = uploads.find(data, one(found, "second")) if layout == "two-files" else None
    if source is None or (layout == "two-files" and second is None):
        raise InputError("E_UPLOAD_GONE")
    texts = {k: pasted(one(found, k)) for k in ("questions", "transcript") if one(found, k).strip()}

    def run() -> facade.Analysis:
        return facade.analyze(
            source, mode, layout=layout, second=second, questions=texts.get("questions"),
            transcript=texts.get("transcript"),
            swap_style=one(found, "swap_style") if mode == "role-swap" else None,
            partner_consent="ui" if one(found, "consent") == "yes" else None,
            confirm_long=one(found, "confirm_long") == "yes", light_run=one(found, "light_run") == "yes",
            review=one(found, "review") or None, detach=True, start=tasks_of(request).start_job,
            names=(one(found, "upload_name") or None, one(found, "second_name") or None),
        )  # fmt: skip

    try:
        done = await run_in_threadpool(run)
    finally:
        for path in texts.values():
            path.unlink(missing_ok=True)
    for used in (source, second):  # the job keeps its own copy; the form's upload is kept until it starts
        if used is not None:
            used.unlink(missing_ok=True)
    return go(f"/jobs/{done.record.job_id}")


@router.get("/jobs")
def history(request: Request, incomplete: bool = False) -> HTMLResponse:
    rows = [
        (r.job_id, views.title(r), views.when(r.created_utc), views.MODES.get(str(r.mode), ""),
         *views.state(r))
        for r in facade.jobs(HISTORY_MAX)
    ]  # fmt: skip
    synced = any(p.lower().startswith(SYNCED) for p in data_root().resolve().parts)
    left = catalogue()["E_DELETE_INCOMPLETE"] if incomplete else None
    return page(request, "history.html", nav="history", rows=rows, synced=synced, left=left)


def progress(found: facade.Analysis) -> dict[str, Any]:
    """The progress list's steps, times and the one sentence a screen reader hears when the step changes."""
    record = found.record
    steps = views.steps(facade.job_steps(record), record)
    running = next((name for name, word in steps if word == "Running"), None)
    if record.state is JobState.QUEUED:
        now = "Waiting to start."
    elif record.state is JobState.RUNNING:
        now = f"Now: {running.lower()}." if running else "Starting the next step."
    else:
        now = "The analysis has ended."
    estimate = facade.job_estimate_ms(record)
    return {
        "steps": steps, "elapsed": clock(views.elapsed_ms(record)), "now": now,
        "estimate": views.about(estimate) if estimate is not None else None,
    }  # fmt: skip


def heading(record: Any) -> dict[str, Any]:
    """What the head of every page of one analysis shows: its name, mode, length, start, state and tabs."""
    return {
        "nav": "history", "job_id": record.job_id, "name": views.title(record),
        "when": views.when(record.created_utc), "mode": views.MODES.get(str(record.mode), ""),
        "state": views.state(record)[0], "tone": views.state(record)[1],
        "length": length_text(record.input["duration_ms"]) if "duration_ms" in record.input else "",
        "device": views.DEVICES.get(str(record.device.get("resolved")), ""),
        "tabs": views.result_tabs(record.job_id), "tab_names": views.RESULT_TABS,
    }  # fmt: skip


@router.get("/jobs/{job_id}")
def job_page(request: Request, job_id: str, tab: str = "overview") -> HTMLResponse:
    found = record_of(job_id)
    record = found.record
    shared = heading(record)
    if record.state in OPEN:
        return page(request, "job.html", view="progress", **shared, **progress(found))
    if record.state not in FINISHED:
        problem = catalogue().get((record.error or {}).get("code", ""))
        # A delete that stopped at a held file took the recording and the results: nothing is left to run.
        half_deleted = problem is not None and problem.code == "E_DELETE_INCOMPLETE"
        return page(request, "job.html", view="stopped", problem=problem, half_deleted=half_deleted, **shared)
    evidence, speech, runs, words = facade.results(job_id)
    policy = record.settings.get("review", {}).get("policy", "when-unsure")
    unsure = bool(evidence["manifest"]["warnings"]) or any(
        c["roles"]["confidence"] == "low" for c in evidence["candidates"]
    )
    saved = facade.corrections(job_id)
    tracks = views.tracks(evidence, words, saved)  # with a track per person, the learner says whose each is
    definitions = {d["measure_id"]: d["how"] for d in evidence["definitions"]}
    utterances = {u["id"]: u["start_ms"] for c in evidence["candidates"] for u in c["utterances"]}
    names = {c.id: c.name for c in parts_catalogue.components().values()}  # "Praat", not its whole title
    people = [(c, *views.measures(c, definitions, names)) for c in evidence["candidates"]]
    warnings = evidence["manifest"]["warnings"]
    return page(
        request, "job.html", view="results", **shared, evidence=evidence, people=people,
        tab=tab if tab in views.RESULT_TABS and tab != "export" else "overview",  # an odd link: Overview
        utterances=utterances,
        strip=timeline.build(evidence, speech, words) if evidence["candidates"] else None,
        lines=views.transcript(evidence, runs), tools=views.tools(evidence),
        review=policy == "always" or (policy == "when-unsure" and (unsure or bool(tracks))),
        review_open=unsure or bool(tracks), tracks=tracks, saved=saved, voices=views.voices(evidence, saved),
        times={c["id"]: views.part_times(c) for c in evidence["candidates"]},
        pointers={c["id"]: views.pointers(c) for c in evidence["candidates"]},
        notes=warning_notes(record, warnings), modes=views.MODES,
        one_voice=any(w["code"] == "W_ONE_VOICE" for w in warnings),
    )  # fmt: skip


def warning_notes(record: Any, warnings: list[dict[str, str]]) -> list[views.Note]:
    """The run's warnings; a mode mismatch also offers the run in the other mode, with its time estimate."""
    found = []
    for w in warnings:
        entry = catalogue().get(w["code"])
        text = " ".join(t for t in (w["text"], views.plain(entry.hint) if entry else "") if t)
        mode = views.RERUN_FOR.get(w["code"])
        try:
            offer = facade.rerun_offer(record.job_id, mode) if mode and mode != record.mode else None
        except InputError:  # this job cannot run in that mode: another version, or two files for solo
            offer = None
        if offer is None:
            found.append(views.Note(text))
        else:
            what = views.rerun_offer(offer.first, offer.estimate_ms, offer.device["resolved"])
            found.append(views.Note(text, mode, what, offer.needs_consent))
    return found


def event(name: str, text: str) -> str:
    """One server-sent event; each line of the text goes in a data line of its own."""
    return f"event: {name}\n" + "".join(f"data: {line}\n" for line in text.split("\n")) + "\n"


@router.get("/jobs/{job_id}/events")
async def events(request: Request, job_id: str) -> StreamingResponse:
    record_of(job_id)
    fragment = TEMPLATES.env.get_template("partials/progress.html")

    async def stream() -> AsyncIterator[str]:
        said = ""
        while True:
            try:
                found = await run_in_threadpool(facade.status, job_id)
            except AppError:  # deleted while the page was open
                yield event("done", "")
                return
            shown = progress(found)
            yield event("progress", fragment.render(**shown))
            if shown["now"] != said:
                said = shown["now"]
                yield event("now", said)
            if found.record.state not in OPEN:
                yield event("done", "")
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(EVENT_EVERY_S)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str) -> HTMLResponse:
    record_of(job_id)
    _, asked = facade.cancel(job_id)
    return HTMLResponse("Stopping. This takes a few seconds." if asked else "This analysis is not running.")


@router.post("/jobs/{job_id}/resume")
async def resume(request: Request, job_id: str) -> Response:
    record_of(job_id)
    if tasks_of(request).running() is not None:
        raise ResourceError("E_BUSY")
    await run_in_threadpool(facade.resume_detached, job_id, tasks_of(request).start_job)
    return go(f"/jobs/{job_id}")


@router.post("/jobs/{job_id}/review")
async def review(request: Request, job_id: str) -> Response:
    record_of(job_id)
    found = await form(request)
    edits = views.review_edits(found, facade.job_evidence(job_id), facade.corrections(job_id))
    if tasks_of(request).running() is not None:
        raise ResourceError("E_BUSY")
    await run_in_threadpool(facade.review, job_id, edits, tasks_of(request).start_job)
    return go(f"/jobs/{job_id}")


@router.post("/jobs/{job_id}/rerun")
async def rerun(request: Request, job_id: str) -> Response:
    """A finished job again in another mode, as a new job; the old one stays as it is."""
    record_of(job_id)
    found = await form(request)
    if tasks_of(request).running() is not None:
        raise ResourceError("E_BUSY")
    consent: Literal["ui"] | None = "ui" if one(found, "consent") == "yes" else None

    def run() -> facade.Analysis:
        return facade.rerun(
            job_id,
            one(found, "mode"),
            partner_consent=consent,
            detach=True,
            start=tasks_of(request).start_job,
        )

    return go(f"/jobs/{(await run_in_threadpool(run)).record.job_id}")


def deleted(delete: Callable[[], object]) -> Response:
    """History as it is after a delete, which says so when another program kept some files."""
    try:
        delete()
    except AppError as exc:
        if exc.code != "E_DELETE_INCOMPLETE":
            raise
        return go("/jobs?incomplete=true")
    return go("/jobs")


@router.post("/jobs/{job_id}/delete")
def delete(job_id: str) -> Response:
    record_of(job_id)
    return deleted(lambda: facade.delete_job(job_id))


@router.post("/data/delete-all")
def delete_all() -> Response:
    return deleted(facade.delete_all)


@router.get("/jobs/{job_id}/audio")
def audio(job_id: str) -> FileResponse:
    """The 16 kHz mono copy that every analysis decodes first: every browser plays it, with seeking."""
    path = record_of(job_id).job / "input" / "audio16k.wav"
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


@router.get("/jobs/{job_id}/files/{name}")
def download(job_id: str, name: str) -> FileResponse:
    record_of(job_id)
    found = {p.name: p for p in facade.export_files(job_id)}  # only these names, never a path from the link
    if name not in found:
        raise HTTPException(404)
    return FileResponse(found[name], filename=name)


@router.get("/jobs/{job_id}/export")
def export_page(request: Request, job_id: str) -> HTMLResponse:
    found = record_of(job_id)
    export_id, people = facade.for_ai_files(job_id)
    shown = []
    for p in people:
        shown.append({
            "id": p.person, "label": upfirst(p.label), "text": p.text, "audio": p.audio, "step": p.step,
            "parts": split_in_two(p.text, export_id) if p.text else None, "file": NAMES[p.person],
            "prompt": prompt_part(p.text) if p.text else None,
            "words": facade.hidden_words(job_id, p.person) if p.step else None,
        })  # fmt: skip
    task = tasks_of(request).last("export")
    return page(
        request, "export.html", **heading(found.record), view="results", tab="export", people=shown,
        two=found.record.mode != "solo",
        zip_name=next((p.name for p in facade.export_files(job_id) if p.suffix == ".zip"), None),
        task=task if task is not None and task.detail == job_id else None,
        result=None if task is None or task.running() else views.task_result("export", task.reply(), {}),
    )  # fmt: skip


def task_view(request: Request, task: Task) -> HTMLResponse:
    return page(request, "partials/task.html", kind=task.kind, line=task.last_line(), share=task.share())


@router.post("/jobs/{job_id}/export")
async def export_again(request: Request, job_id: str) -> HTMLResponse:
    record_of(job_id)
    found = await form(request)
    audio_too = ["--audio"] if one(found, "audio") == "yes" else []
    return task_view(
        request, tasks_of(request).command("export", "export", job_id, *audio_too, detail=job_id)
    )


@router.get("/tasks/{kind}")
def task_status(request: Request, kind: str) -> Response:
    """A long action's progress line while it runs; when it ends, the page loads again with its result."""
    if kind not in TASK_KINDS:
        raise HTTPException(404)
    task = tasks_of(request).last(kind)
    return reload() if task is None or not task.running() else task_view(request, task)


def totals(ids: list[str]) -> dict[str, Any]:
    """The chosen parts' download size, the free space where models go, and the time at DOWNLOAD_MBIT."""
    known = parts_catalogue.offered()
    chosen = [i for i in ids if i in known]
    planned = facade.install_plan(chosen)
    folder = models_dir()
    free = shutil.disk_usage(next(p for p in (folder, *folder.parents) if p.exists())).free
    seconds = planned.download_bytes * 8 / (DOWNLOAD_MBIT * 1e6)

    def needs(i: str) -> set[str]:
        return {n for d in known[i].needs for n in (d, *needs(d))}

    # An unticked part that a ticked part needs is still counted, so its box would seem to change nothing.
    kept = []
    for c in planned.parts:
        if c.id not in chosen and c.install in ("download", "env"):
            by = [known[i].title for i in chosen if c.id in needs(i)]
            verb = "needs" if len(by) == 1 else "need"
            kept.append(f"{c.title} stays in the total: {' and '.join(by)} {verb} it.")
    return {
        "size": size_text(planned.download_bytes), "free": size_text(free), "kept": kept,
        "time": views.about(int(seconds * 1000)) if planned.download_bytes else "",
    }  # fmt: skip


@router.get("/components")
def components_page(request: Request, first: bool = False) -> HTMLResponse:
    rows = facade.components()
    tasks, titles = tasks_of(request), model_titles()
    results = {}
    for kind in ("install", "selftest"):
        task = tasks.last(kind)
        if task is not None and not task.running():
            results[kind] = views.task_result(kind, task.reply(), titles)
    running = tasks.running()
    known = parts_catalogue.offered()
    notes = {c.id: upfirst(c.note) + "." if c.note else "" for c in known.values()}
    # A part that did not pass its accuracy check on this system is opt-in: its box starts unticked.
    optin = {i for i, c in known.items() if parts_catalogue.SYSTEM in c.experimental_on}
    groups = [
        (title, [r for r in rows if known[r.id].kind == kind]) for kind, title in views.PART_GROUPS.items()
    ]
    return page(
        request, "components.html", nav="components", first=first, rows=rows, words=views.COMPONENT_WORDS,
        groups=[(title, found) for title, found in groups if found], tones=views.COMPONENT_TONES,
        licences=views.LICENCE_WORDS, installable=views.INSTALLABLE, notes=notes, terms=OPENSMILE_TERMS,
        results=results, optin=optin,
        command=executable(),
        totals=totals([r.id for r in rows if r.status in views.INSTALLABLE and r.id not in optin]),
        task=running if running is not None and running.kind in ("install", "selftest") else None,
        removable=[r for r in rows if r.status == "ready" and known[r.id].install in ("download", "env")],
        size_text=size_text, selftest=install_state(app_root()).get("selftest"),
    )  # fmt: skip


@router.get("/components/totals")
def components_totals(request: Request) -> HTMLResponse:
    """The totals line again as parts are ticked and unticked; it reads file sizes only."""
    return page(request, "partials/totals.html", totals=totals(request.query_params.getlist("part")))


@router.post("/components/install")
async def install(request: Request) -> Response:
    found = await form(request)
    known = parts_catalogue.offered()
    ids = [i for i in found.get("part", []) if i in known]
    for i in ids:  # the licence box is the learner's own consent, separate from the row's tick
        if known[i].requires_consent and one(found, f"licence.{i}") == "yes":
            facade.consent(i, "ui")
    status = {r.id: r.status for r in facade.components()}
    todo = [i for i in ids if status[i] in views.INSTALLABLE]
    if not todo:
        return reload()
    return task_view(request, tasks_of(request).command("install", "components", "install", *todo, "--yes"))


@router.post("/components/remove")
async def remove(request: Request) -> Response:
    part = one(await form(request), "part")
    if part not in parts_catalogue.offered():
        raise HTTPException(400)
    await run_in_threadpool(facade.remove, [part])
    return reload()


@router.post("/selftest")
def selftest(request: Request) -> HTMLResponse:
    return task_view(request, tasks_of(request).command("selftest", "self-test"))


def token_context(message: str = "", link: tuple[str, str] | None = None) -> dict[str, Any]:
    repo = parts_catalogue.get("pyannote-community-1").repo
    return {
        "saved": token_present(),
        "model": f"https://huggingface.co/{repo}",
        "message": message,
        "link": link,
    }


TOKENS_PAGE = ("Open your Hugging Face token settings", "https://huggingface.co/settings/tokens")


def token_answer(request: Request, status: Any) -> HTMLResponse:
    """The token section again, with what the check found; the value itself never comes back to the page."""
    context = token_context()
    if status.valid is False:
        context.update(
            message="Hugging Face does not accept this token. Make a new read token.", link=TOKENS_PAGE
        )
    elif status.pyannote_terms_ok is False:
        context.update(
            message="The token works, but the speaker model's terms are not accepted yet. Accept them, then "
            "check again.", link=("Open the model page", context["model"]),
        )  # fmt: skip
    elif status.present:
        context["message"] = "The token works, and the speaker model's terms are accepted."
    return page(request, "partials/token.html", **context)


@router.get("/token")
def token_page(request: Request) -> HTMLResponse:
    return page(request, "token.html", **token_context())


@router.post("/token")
async def token_save(request: Request) -> HTMLResponse:
    value = one(await form(request), "token")
    try:
        status, _ = await run_in_threadpool(facade.token_set, value)
    except AppError as exc:
        # Both answers keep the form in place, so the learner can try again on the same page.
        if exc.code == "E_TOKEN_INVALID":
            context = token_context(
                "Hugging Face does not accept this token. Make a new read token.", TOKENS_PAGE
            )
        elif exc.code == "E_NETWORK":
            context = token_context(
                "Fluentfolio could not reach Hugging Face to check this token, so it did not save it. "
                "Check the internet connection, then save it again."
            )
        else:
            raise
        return page(request, "partials/token.html", **context)
    return token_answer(request, status)


@router.post("/token/check")
async def token_check(request: Request) -> HTMLResponse:
    return token_answer(request, await run_in_threadpool(facade.token_status))


@router.post("/token/clear")
def token_clear(request: Request) -> HTMLResponse:
    removed = facade.token_clear()
    text = (
        "The token is removed." if removed else "Fluentfolio did not save this token, so it left it in place."
    )
    return page(request, "partials/token.html", **token_context(text))


def folders() -> list[dict[str, str]]:
    """The two folders a learner can move, where each is now, and why the data folder may stay put."""
    fixed = "The FLUENTFOLIO_HOME variable chooses this folder. To move it, change that variable."
    return [
        {
            "kind": "data", "title": "Data folder", "path": str(data_root()),
            "fixed": fixed if os.environ.get("FLUENTFOLIO_HOME") else "",
            "help": "Your recordings, results and settings. "
            "After the move, Fluentfolio stops: start it again. On Windows, a moved folder keeps the "
            "access rights of its new place, so other accounts may read it.",
        },
        {
            "kind": "models", "title": "Models folder", "path": str(models_dir()), "fixed": "",
            "help": "The downloaded models. On Windows the folder must be on a local disk, outside OneDrive, "
            "under 60 characters long, with no spaces and no letters outside A to Z. A moved folder keeps "
            "the access rights of its new place, so other accounts may change the model files, which run "
            "as code.",
        },
    ]  # fmt: skip


def settings_context(tab: str) -> dict[str, Any]:
    loaded, _ = facade.settings_show()
    models = {m: model_titles()[m] for m in choices()}
    return {
        "tab": tab, "groups": views.settings_form(loaded.settings, tab, models),
        "read_only": loaded.read_only,
        "problems": [catalogue()[code].title for code, _ in loaded.warnings if code in catalogue()],
        "folders": folders() if tab == "advanced" else [],
    }  # fmt: skip


@router.get("/settings")
def settings_page(request: Request, tab: str = "basic") -> HTMLResponse:
    return page(
        request, "settings.html", nav="settings", **settings_context(tab if tab == "advanced" else "basic")
    )


@router.post("/settings")
async def settings_save(request: Request) -> HTMLResponse:
    found = await form(request)
    tab = "advanced" if one(found, "tab") == "advanced" else "basic"
    saved, theme = [], None
    for _, controls in settings_context(tab)["groups"]:
        for c in controls:
            if c.kind == "switch":
                new = "true" if one(found, c.key) == "yes" else "false"
            else:
                new = one(found, c.key, c.value).strip()
            if new == c.value:
                continue
            await run_in_threadpool(
                facade.settings_set, c.key, json.dumps(new) if c.kind == "choice" else new
            )
            saved.append(c.title)
            theme = new if c.key == "ui.theme" else theme
    # Only the status line comes back, so the focus stays on the control that changed.
    response = HTMLResponse(html.escape(f"Saved: {', '.join(saved)}.") if saved else "")
    if theme is not None:  # the page is drawn again in the new theme
        if theme == "system":
            response.delete_cookie(THEME_COOKIE)
        else:
            response.set_cookie(
                THEME_COOKIE, theme, max_age=THEME_DAYS * 86400, httponly=True, samesite="strict"
            )
        response.headers["HX-Refresh"] = "true"
    return response


@router.post("/settings/restore")
def settings_restore() -> Response:
    facade.settings_restore()
    response = reload()  # every control shows its default again
    response.delete_cookie(THEME_COOKIE)
    return response


@router.post("/settings/move")
async def move_folder(request: Request) -> Response:
    found = await form(request)
    # The page writes the home part as %LOCALAPPDATA% or ~, so a target typed that way works too.
    typed = os.path.expanduser(os.path.expandvars(one(found, "target").strip()))
    kind, target = one(found, "kind"), Path(typed)
    if kind == "models":
        await run_in_threadpool(facade.move_folder, "models", target)
        return reload()
    if kind != "data":
        raise HTTPException(400)
    old = data_root()
    await run_in_threadpool(facade.move_folder, "data", target, keep_old=True)
    # This server keeps its lock, log and launch files in the old folder: it stops, then the folder goes.
    response = page(request, "partials/stopped.html", moved=str(target))
    response.headers["HX-Retarget"] = "#main"  # the whole page says so, as after Quit
    response.background = BackgroundTask(request.app.state.stop, old)
    return response


@router.get("/doctor")
def doctor(request: Request) -> HTMLResponse:
    report, _ = facade.doctor()
    rows = [
        (c.group, LEVELS[c.level], c.level, views.plain(c.message),
         "" if c.code in FIX_PAGES else views.plain(c.fix), FIX_PAGES.get(c.code or ""))
        for c in report.checks
    ]  # fmt: skip
    return page(request, "doctor.html", rows=rows, report=facade.diagnostic_report())


@router.get("/about")
def about(request: Request, tab: str = "start") -> HTMLResponse:
    offered = parts_catalogue.offered()
    links = [(key, name, f"/about?tab={key}") for key, name in views.HELP_TABS.items()]
    return page(
        request, "about.html", nav="help", help_tabs=links,
        tab=tab if tab in views.HELP_TABS else "start",  # an odd link shows the first tab, as Settings does
        parts=[(c.title, c.licence, c.attribution) for c in offered.values()],
        explained=[(c.title, views.PART_ABOUT[cid]) for cid, c in offered.items()],
        releases=facade.RELEASES_PAGE,
    )  # fmt: skip


@router.post("/updates/check")
async def updates_check(request: Request) -> HTMLResponse:
    try:
        found = await run_in_threadpool(facade.check_updates)
    except AppError as exc:
        if exc.code not in ("E_NETWORK", "E_TLS_INTERCEPTED"):
            raise
        # The steps stay on the page, so the learner can still update by hand.
        return page(request, "partials/updates.html", failed=True, releases=facade.RELEASES_PAGE)
    return page(request, "partials/updates.html", release=found, releases=facade.RELEASES_PAGE)
