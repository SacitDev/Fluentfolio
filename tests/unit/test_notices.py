"""The licence gate: classes, unknown licences and which packages every install has."""

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "notices.py"
spec = importlib.util.spec_from_file_location("notices", TOOL)
assert spec is not None and spec.loader is not None
notices = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notices)

TABLE = {"proprietary": {"licences": ["LicenseRef-Vendor"]}, "noncommercial": {"licences": ["LicenseRef-NC"]}}


def test_unknown_licence_is_refused() -> None:
    with pytest.raises(notices.LicenceError, match="unknown licence 'Made-Up-1.0'"):
        notices.licence_class("MIT AND Made-Up-1.0", TABLE)


@pytest.mark.parametrize(
    ("expr", "klass"),
    [
        ("MIT", "permissive"),
        ("MIT OR GPL-3.0-only", "permissive"),
        ("MIT AND GPL-3.0-only", "gpl"),
        ("(Apache-2.0 WITH LLVM-exception) AND BSD-3-Clause", "permissive"),
        ("LicenseRef-Vendor", "proprietary"),
        ("LicenseRef-NC OR GPL-2.0-or-later", "gpl"),
        ("LicenseRef-NC AND MIT", "nc"),
    ],
)
def test_licence_classes(expr: str, klass: str) -> None:
    assert notices.licence_class(expr, TABLE) == klass


def lock(*packages: dict, dev: tuple[str, ...] = ()) -> dict:
    root = {
        "name": "app",
        "version": "1",
        "source": {"editable": "."},
        "dependencies": [{"name": "a"}, {"name": "b"}],
        "dev-dependencies": {"dev": [{"name": d} for d in dev]},
    }
    return {"package": [root, *packages]}


def test_extra_asked_by_a_later_edge_is_followed() -> None:
    # "a" is reached plainly first and again with its "http" extra, which brings "c".
    graph = lock(
        {"name": "a", "version": "1", "optional-dependencies": {"http": [{"name": "c"}]}},
        {"name": "b", "version": "1", "dependencies": [{"name": "a", "extra": ["http"]}]},
        {"name": "c", "version": "1"},
    )
    _, everywhere, used_by = notices.lock_graph(graph)
    assert ("c", "1") in everywhere
    assert used_by[("c", "1")] == "app"


def test_platform_only_package_is_not_everywhere() -> None:
    graph = lock(
        {"name": "a", "version": "1", "dependencies": [{"name": "w", "marker": "sys_platform == 'win32'"}]},
        {"name": "b", "version": "1"},
        {"name": "w", "version": "1"},
    )
    _, everywhere, used_by = notices.lock_graph(graph)
    assert ("w", "1") not in everywhere
    assert used_by[("w", "1")] == "app"
    assert ("a", "1") in everywhere


def test_dev_only_and_other_system_packages() -> None:
    graph = lock(
        {"name": "a", "version": "1", "dependencies": [{"name": "m", "marker": "sys_platform == 'darwin'"}]},
        {"name": "b", "version": "1"},
        {"name": "m", "version": "1"},
        {"name": "t", "version": "1"},
        dev=("t",),
    )
    _, everywhere, used_by = notices.lock_graph(graph)
    assert used_by[("t", "1")] == "development"
    assert ("t", "1") in everywhere
    assert ("m", "1") not in used_by
    assert ("m", "1") not in everywhere


def test_check_fails_end_to_end_on_an_unknown_licence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    declared = notices.from_metadata
    monkeypatch.setattr(
        notices,
        "from_metadata",
        lambda name, version: "Made-Up-1.0" if name == "click" else declared(name, version),
    )
    assert notices.main(["--check"]) == 1
    printed = capsys.readouterr().out.splitlines()
    assert any(
        line.startswith("notices: click ") and "unknown licence 'Made-Up-1.0'" in line for line in printed
    )
    assert printed[-1].startswith("notices: FAIL")


def test_check_fails_when_the_notices_file_is_not_the_generated_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    stale = tmp_path / "THIRD_PARTY_NOTICES.md"
    stale.write_text("# Third-party notices\n", encoding="utf-8")
    monkeypatch.setattr(notices, "OUT", stale)
    assert notices.main(["--check"]) == 1
    assert "THIRD_PARTY_NOTICES.md differs from the generated text" in capsys.readouterr().out
    assert stale.read_text(encoding="utf-8") == "# Third-party notices\n"  # --check writes nothing
