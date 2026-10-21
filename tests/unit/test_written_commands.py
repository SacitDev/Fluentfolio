"""Every fluentfolio command in the help, the agent guide and the docs parses; test_errors runs the hints."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import click
import pytest

from fluentfolio.cli.main import cli

PLACEHOLDER = re.compile(r"[A-Z][A-Z_]+|<[^<>\s]+>")  # JOB, COMMAND, <folder>: the reader fills them in
VALUE = re.compile(r"[./\\0-9]")  # a file name, a dotted setting or a number, as opposed to a prose word
END = ".,;:)`"
ROOT = Path(__file__).resolve().parents[2]
PAGES = ("README.md", "docs/*.md", "src/fluentfolio/web/templates/about.html")
# In the docs a command is code: an inline `span`, a line of a fenced block (no diagram), or <code> in Help.
DIAGRAM = re.compile(r"```mermaid\n.*?```", re.DOTALL)
CODE = re.compile(r"```[a-z]*\n(.*?)```|`([^`\n]+)`|<code>(.*?)</code>", re.DOTALL)
WINDOWS = '& "<command>"'  # the docs' PowerShell form of the fluentfolio command


def clean(word: str) -> tuple[str, bool]:
    """The word without quotes or brackets, and whether punctuation after it ends the command."""
    bare = word.lstrip("(`").rstrip(END)
    return bare, bare != word.lstrip("(`")


def value_fits(word: str, param: click.Parameter) -> bool:
    if PLACEHOLDER.fullmatch(word):
        return True
    if isinstance(param.type, click.Choice):
        return word in param.type.choices
    return bool(VALUE.search(word))


def written(text: str) -> list[list[str]]:
    """The words after each "fluentfolio" in a text, up to the end of its sentence."""
    words = text.split()
    found = []
    for i, word in enumerate(words):
        if word.lstrip("(`") == "fluentfolio":
            rest = []
            for after in words[i + 1 :]:
                rest.append(after)
                if clean(after)[1]:
                    break
            found.append(rest)
    return found


def parse(words: list[str]) -> str:
    """Click's complaint about one written command, or "" when it parses."""
    group: click.Command = cli
    path: list[str] = []
    i = 0
    while isinstance(group, click.Group) and i < len(words):
        word, end = clean(words[i])
        name = "doctor" if word == "COMMAND" else word
        if name not in group.commands:
            break
        path.append(name)
        group = group.commands[name]
        i += 1
        if end:
            words = words[:i]
    if isinstance(group, click.Group) and i < len(words) and not clean(words[i])[0].startswith("-"):
        return f"no command {clean(words[i])[0]!r}"
    options = {flag: p for p in group.params if isinstance(p, click.Option) for flag in p.opts}
    positional = [p for p in group.params if isinstance(p, click.Argument)]
    args: list[str] = []
    while i < len(words):
        word, end = clean(words[i])
        i += 1
        if word.startswith("-"):
            args.append(word)
            option = options.get(word)
            if option is not None and not option.is_flag and not end and i < len(words):
                value, end = clean(words[i])
                if value_fits(value, option):
                    args.append(value)
                    i += 1
        elif positional and value_fits(word, positional[0]):
            args.append(word)
            positional.pop(0)
        else:
            break
        if end:
            break
    # Click's own parser: unknown options and missing option values fail here; a missing recording does not.
    ctx = click.Context(group, info_name="fluentfolio " + " ".join(path))
    try:
        _, extra, _ = group.make_parser(ctx).parse_args(args)
    except click.UsageError as exc:
        return exc.format_message()
    return f"unexpected words {extra}" if extra else ""


def sources() -> list[tuple[str, str]]:
    """Each place a learner or an agent reads a command, with its text."""
    found: list[tuple[str, str]] = []

    def walk(command: click.Command, name: str) -> None:
        found.append((f"help of {name}", command.epilog or ""))
        for sub, child in getattr(command, "commands", {}).items():
            walk(child, f"{name} {sub}")

    walk(cli, "fluentfolio")
    for doc in ("agent-guide.md", "SKILL.md"):
        text = resources.files("fluentfolio.agent").joinpath(doc).read_text(encoding="utf-8")
        body = text.split("---", 2)[2] if text.startswith("---") else text  # not the front matter
        found.append((doc, body))
    for page in sorted(p for pattern in PAGES for p in ROOT.glob(pattern)):
        text = DIAGRAM.sub("", page.read_text(encoding="utf-8")).replace(WINDOWS, "fluentfolio")
        for block, inline, html in CODE.findall(text):
            for line in block.splitlines() if block else [inline or html]:
                found.append((page.relative_to(ROOT).as_posix(), line))
    return found


def test_every_written_command_parses() -> None:
    bad = [
        f"{where}: fluentfolio {' '.join(words)} -> {problem}"
        for where, text in sources()
        for words in written(text)
        if (problem := parse(words))
    ]
    assert not bad, "\n".join(bad)


def test_written_commands_are_found_at_all() -> None:
    assert sum(len(written(text)) for _, text in sources()) >= 40
    pages = [text for where, text in sources() if where == "README.md" or where.startswith("docs/")]
    assert sum(len(written(text)) for text in pages) >= 30  # README.md and docs/ hold dozens


@pytest.mark.parametrize(
    ("text", "complaint"),
    [
        ("Run fluentfolio doctor --report", "requires an argument"),
        (
            "If it stops again, fluentfolio doctor --report makes a report you can share.",
            "requires an argument",
        ),
        ("Run fluentfolio doctor --reprot report.txt", "No such option"),
        ("Add fluentfolio export JOB --candidate person3", "requires an argument"),
        ("Run fluentfolio jobs lsit", "no command 'lsit'"),
        ("Run fluentfolio doctr", "no command 'doctr'"),
    ],
)
def test_a_broken_command_is_caught(text: str, complaint: str) -> None:
    (words,) = written(text)
    assert complaint in parse(words)


@pytest.mark.parametrize(
    "text",
    [
        "Run fluentfolio doctor --report report.txt; tell the person.",
        "cancel it with fluentfolio cancel JOB.",
        "fluentfolio settings set export.audio true",
        "fluentfolio COMMAND --help",
        "run fluentfolio components consent opensmile in a terminal.",
        "Run `fluentfolio guide agents --write-skill <folder>`: it writes the skill.",
    ],
)
def test_a_good_command_passes(text: str) -> None:
    (words,) = written(text)
    assert parse(words) == ""
