"""The web foundation in a real browser: sign-in by the launch file, the page policy, themes, fonts, Quit."""

from __future__ import annotations

import psutil
import pytest
from playwright.sync_api import Browser

from tests.support.browser import Served, signed_in

pytestmark = pytest.mark.slow  # a real browser; the full check runs it

# The ground colour of each theme, from the token table in the UI rules, as the browser reports it.
GROUND = {"light": "rgb(242, 244, 245)", "dark": "rgb(18, 22, 25)"}  # #F2F4F5 and #121619


def test_the_launch_file_lands_on_the_app_signed_in_with_a_strict_cookie(
    browser: Browser, served: Served
) -> None:
    page = signed_in(browser, served)
    assert "not signed in" not in page.content()
    assert page.locator("header.site nav").is_visible()
    cookies = {c["name"]: c for c in page.context.cookies()}
    session = cookies["fluentfolio_session"]
    assert (session["httpOnly"], session["sameSite"], session["path"]) == (True, "Strict", "/")
    assert session["expires"] == -1  # a session cookie: it ends with the browser
    assert page.url == f"{served.origin}/"


def test_the_page_policy_blocks_an_injected_inline_script(browser: Browser, served: Served) -> None:
    page = signed_in(browser, served)
    blocked: list[str] = []
    page.on("console", lambda message: blocked.append(message.text) if message.type == "error" else None)
    ran = page.evaluate(
        """() => {
            window.injected = false;
            const script = document.createElement("script");
            script.textContent = "window.injected = true";
            document.body.appendChild(script);
            return window.injected;
        }"""
    )
    assert ran is False
    page.wait_for_timeout(200)
    assert any("Content Security Policy" in text for text in blocked), blocked


@pytest.mark.parametrize(
    ("cookie", "system", "ground"),
    [
        (None, "light", GROUND["light"]),
        (None, "dark", GROUND["dark"]),
        ("dark", "light", GROUND["dark"]),
        ("light", "dark", GROUND["light"]),
    ],
)
def test_the_theme_follows_the_system_unless_one_was_chosen(
    browser: Browser, served: Served, cookie: str | None, system: str, ground: str
) -> None:
    page = signed_in(browser, served, color_scheme=system)
    if cookie is not None:
        page.context.add_cookies([{"name": "theme", "value": cookie, "url": served.origin}])
        page.reload()
    assert page.evaluate("() => getComputedStyle(document.documentElement).backgroundColor") == ground


def test_the_bundled_font_loads_under_the_page_policy(browser: Browser, served: Served) -> None:
    page = signed_in(browser, served)
    page.evaluate("() => document.fonts.ready")
    for weight in ("400", "600"):
        loaded = page.evaluate(f"() => document.fonts.check('{weight} 16px \"Atkinson Hyperlegible Next\"')")
        assert loaded is True, weight
    faces = page.evaluate(
        "() => [...document.fonts].filter(f => f.status === 'loaded').map(f => [f.family, f.weight])"
    )
    # Browsers differ on whether the family keeps its quotes.
    assert sorted((family.strip('"'), weight) for family, weight in faces) == [
        ("Atkinson Hyperlegible Next", "400"), ("Atkinson Hyperlegible Next", "600"),
    ]  # fmt: skip


def test_an_error_put_in_place_shows_its_card_and_moves_the_focus_to_it(
    browser: Browser, served: Served
) -> None:
    page = signed_in(browser, served)
    errors: list[str] = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    # What a form on a page does when its action fails: htmx sends it and puts the answer in #main.
    page.evaluate("() => htmx.ajax('POST', '/no-such-action', {target: '#main'})")
    page.wait_for_function("() => document.activeElement.id === 'error-title'", timeout=10_000)
    assert page.locator("#error-title").inner_text() == "Page not found"
    assert page.locator("#main").get_by_role("link", name="Open History").is_visible()
    assert errors == []


def test_quit_in_the_browser_stops_the_server(browser: Browser, served: Served) -> None:
    page = signed_in(browser, served)
    page.get_by_role("button", name="Quit").click()
    page.get_by_role("heading", name="Fluentfolio has stopped").wait_for()
    # The answer takes the focus from Quit, so a screen reader reads it.
    page.wait_for_function("() => document.activeElement.tagName === 'H1'", timeout=10_000)
    assert page.evaluate("() => document.activeElement.textContent") == "Fluentfolio has stopped"
    try:
        psutil.Process(served.pid).wait(30)  # raises if the server is still running after 30 s
    except psutil.NoSuchProcess:
        pass
    assert not (served.home / "server.json").exists()
