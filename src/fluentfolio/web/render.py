"""Pages in the base layout, and the few ways a value is written the same on every page."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import jinja2
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from fluentfolio import __version__
from fluentfolio.app import facade

HERE = Path(__file__).parent
THEME_COOKIE = "theme"
THEMES = ("light", "dark")  # without the cookie, or with another value, the page follows the system
TEMPLATES = Jinja2Templates(
    env=jinja2.Environment(
        loader=jinja2.FileSystemLoader(HERE / "templates"), autoescape=True, undefined=jinja2.StrictUndefined
    )
)


def theme_of(request: Request) -> str | None:
    """The theme the browser's cookie names, else the one saved in Settings; None follows the system."""
    chosen = request.cookies.get(THEME_COOKIE)
    if chosen is None:
        chosen = facade.settings_show()[0].settings.ui.theme
    return chosen if chosen in THEMES else None


def page(request: Request, template: str, /, status: int = 200, **context: Any) -> HTMLResponse:
    """A page in the base layout, in the theme the user chose; the context may hold any other name."""
    shared = {"theme": theme_of(request), "version": __version__, "nav": ""}
    return TEMPLATES.TemplateResponse(request, template, {**shared, **context}, status_code=status)


def length_text(ms: int) -> str:
    """A length as a learner reads it: "32.8 s" or "14 min 2 s"."""
    if ms < 60000:  # noqa: PLR2004 - one minute
        return f"{ms / 1000:.1f} s"
    return f"{ms // 60000} min {ms % 60000 // 1000} s"


def clock(ms: int) -> str:
    """A time in the recording, as its player shows it: "4:31", or "1:02:05" past an hour."""
    s = ms // 1000
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"  # noqa: PLR2004


def upfirst(text: str) -> str:
    """The text with a capital first letter; unlike str.capitalize, "MATTR" and "Hugging Face" stay as is.

    A first word with a capital inside it, such as "openSMILE", is a name spelled that way and stays too.
    """
    if any(c.isupper() for c in text.split(" ", 1)[0][1:]):
        return text
    return text[:1].upper() + text[1:]


HOME_VARIABLES = ("LOCALAPPDATA", "USERPROFILE")  # on Windows; elsewhere the home folder is ~


def path_text(path: str) -> str:
    """A path as a page shows it, with the home part as %LOCALAPPDATA% or ~, so no user name is on screen.

    The command line keeps full paths, for scripts and AI agents.
    """
    if os.name == "nt":
        bases = [(Path(value), f"%{name}%") for name in HOME_VARIABLES if (value := os.environ.get(name))]
    else:
        bases = [(Path.home(), "~")]
    given = Path(path)
    # The deepest first: Fluentfolio's own folders sit in %LOCALAPPDATA%, which is inside %USERPROFILE%.
    for base, name in sorted(bases, key=lambda b: len(b[0].parts), reverse=True):
        if base.is_absolute() and given.is_relative_to(base):
            return str(Path(name, given.relative_to(base)))
    return path


TEMPLATES.env.filters.update(clock=clock, upfirst=upfirst, path_text=path_text)
