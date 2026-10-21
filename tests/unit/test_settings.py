"""Settings: defaults, precedence, reliable loading, migration, the per-job snapshot and no secrets."""

from pathlib import Path

import pytest

from fluentfolio.core.enums import Device
from fluentfolio.core.errors import AppError
from fluentfolio.settings import store
from fluentfolio.settings.model import Settings


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_the_defaults_are_the_documented_ones(tmp_path: Path) -> None:
    loaded = store.load(tmp_path / "settings.toml")  # no file yet
    assert (loaded.warnings, loaded.read_only) == ([], False)
    assert loaded.settings.model_dump(mode="json") == {
        "schema_version": 1,
        "general": {"log_level": "info", "retention_days": 0},
        "device": {"choice": "auto"},
        "asr": {"model": "parakeet-tdt-0.6b-v2"},
        "speakers": {"backend": "auto", "count": "auto"},
        "review": {"policy": "when-unsure"},
        "components": {},
        "analyzers": {"pronunciation": True},  # on, though experimental: it did not pass its accuracy check
        "limits": {
            "max_upload_mb": 2000,
            "min_s": 10,
            "short_s": 60,
            "confirm_min": 60,
            "long_min": 90,
            "max_min": 180,
            "no_progress_min": 10,
        },  # fmt: skip
        "export": {"audio": False},
        "ui": {"theme": "system"},
    }
    assert loaded.settings.component_on("opensmile") and loaded.settings.analyzer_on("praat")
    assert loaded.settings.analyzer_on("pronunciation")
    switched = store.load(write(tmp_path / "off.toml", "[analyzers]\npronunciation = false\n")).settings
    assert not switched.analyzer_on("pronunciation") and switched.analyzers == {"pronunciation": False}


def test_the_file_overrides_the_defaults_and_this_runs_flags_override_the_file(tmp_path: Path) -> None:
    path = write(
        tmp_path / "settings.toml",
        '[asr]\nmodel = "whisper-turbo"\n[device]\nchoice = "cpu"\n[components]\nopensmile = false\n'
        "[general]\nretention_days = 30\n",
    )
    settings = store.load(path).settings
    assert (settings.asr.model, settings.device.choice, settings.general.retention_days) == (
        "whisper-turbo",
        "cpu",
        30,
    )
    assert not settings.component_on("opensmile")
    frozen = store.snapshot(settings, asr="canary-1b-v2", device=Device.GPU, light_run=True)
    assert (frozen["asr"]["model"], frozen["device"]["choice"], frozen["light_run"]) == (
        "canary-1b-v2",
        "gpu",
        True,
    )
    assert frozen["general"]["retention_days"] == 30  # everything else is the file's
    # A light run switches MFA, the phone model and openSMILE off for this job only.
    assert frozen["components"] == {"opensmile": False, "mfa": False, "phone-model": False}
    assert store.snapshot(settings)["asr"]["model"] == "whisper-turbo"
    assert store.snapshot(settings)["components"] == {"opensmile": False}
    assert settings.components == {"opensmile": False}


def test_unknown_keys_are_ignored_and_bad_values_fall_back(tmp_path: Path) -> None:
    path = write(
        tmp_path / "settings.toml",
        '[general]\nlog_level = "loud"\nretention_days = 7\ncolour = "red"\n[ui]\ntheme = "dark"\n'
        '[analyzers]\npraat = "yes"\nswiftf0 = false\n[plugins]\nx = 1\n[limits]\nmax_min = -5\n',
    )
    loaded = store.load(path)
    assert sorted(loaded.warnings) == [
        ("W_SETTINGS_INVALID", "analyzers.praat"),
        ("W_SETTINGS_INVALID", "general.log_level"),
        ("W_SETTINGS_INVALID", "limits.max_min"),
        ("W_SETTINGS_UNKNOWN_KEY", "general.colour"),
        ("W_SETTINGS_UNKNOWN_KEY", "plugins"),
    ]
    s = loaded.settings
    assert (s.general.log_level, s.general.retention_days, s.ui.theme, s.limits.max_min) == (
        "info",
        7,
        "dark",
        180,
    )
    assert s.analyzer_on("praat") and not s.analyzer_on("swiftf0")


@pytest.mark.parametrize(
    "raw", [b'[general\nlog_level = "info"', b"\xff\xfe\x00broken", b'schema_version = "two"\n']
)
def test_an_unreadable_file_is_backed_up_and_defaults_used(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "settings.toml"
    path.write_bytes(raw)
    loaded = store.load(path)
    assert loaded.warnings == [("W_SETTINGS_UNREADABLE", "settings.toml")]
    assert loaded.settings == Settings() and not path.exists()
    backups = list(tmp_path.glob("settings.toml.bad-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == raw


def test_a_newer_file_is_used_read_only_and_never_reset(tmp_path: Path) -> None:
    raw = 'schema_version = 2\n[ui]\ntheme = "dark"\nnew_key = 1\n[future]\nx = 1\n'
    path = write(tmp_path / "settings.toml", raw)
    loaded = store.load(path)
    assert loaded.read_only and ("W_SETTINGS_NEWER", "2") in loaded.warnings
    assert loaded.settings.ui.theme == "dark"
    assert path.read_text(encoding="utf-8") == raw


def test_an_older_schema_is_migrated_one_step_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rename(table: dict) -> dict:  # a pretend schema 2 that renamed [display] to [ui]
        table = dict(table)
        table["ui"] = table.pop("display")
        table["schema_version"] = 2
        return table

    monkeypatch.setattr(store, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(store, "MIGRATIONS", {1: rename})
    loaded = store.load(write(tmp_path / "settings.toml", 'schema_version = 1\n[display]\ntheme = "light"\n'))
    assert (loaded.settings.ui.theme, loaded.warnings, loaded.read_only) == ("light", [], False)


def test_settings_never_hold_a_secret(tmp_path: Path) -> None:
    secret_words = ("token", "secret", "password", "key")
    names = [
        name
        for section in Settings.model_fields.values()
        for name in getattr(section.annotation, "model_fields", {})
    ]
    assert not [n for n in [*Settings.model_fields, *names] if any(w in n for w in secret_words)]
    token = "hf_" + "Z9" * 15
    path = write(tmp_path / "settings.toml", f'[general]\nhf_token = "{token}"\n')
    loaded = store.load(path)
    assert ("W_SETTINGS_UNKNOWN_KEY", "general.hf_token") in loaded.warnings
    store.save(path, loaded.settings)
    assert token not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("key", "text", "section", "expected"),
    [
        ("export.audio", "true", "export", {"audio": True}),
        ("limits.confirm_min", "90", "limits", {"confirm_min": 90}),
        ("ui.theme", "dark", "ui", {"theme": "dark"}),  # a bare word is text
        ("ui.theme", '"light"', "ui", {"theme": "light"}),
        ("device.choice", "cpu", "device", {"choice": "cpu"}),
        ("general.retention_days", "0", "general", {"retention_days": 0}),
        ("components.mfa", "false", "components", {"mfa": False}),
        ("analyzers.pronunciation", "false", "analyzers", {"pronunciation": False}),
    ],
)
def test_one_setting_changes_from_its_text_and_nothing_else_does(
    key: str, text: str, section: str, expected: dict[str, object]
) -> None:
    before = Settings().model_dump(mode="json")
    after = store.changed(Settings(), key, text).model_dump(mode="json")
    assert after[section] == {**before[section], **expected}
    assert {k: v for k, v in after.items() if k != section} == {
        k: v for k, v in before.items() if k != section
    }


@pytest.mark.parametrize(
    ("key", "text"),
    [
        ("export.audio", "yes"),  # a switch takes true or false only
        ("export.audio", "1"),
        ("components.mfa", "1"),
        ("limits.max_min", "true"),  # a number is not a switch
        ("limits.confirm_min", "0"),  # at least 1
        ("limits.confirm_min", "9.5"),  # whole minutes
        ("limits.confirm_min", '"90"'),
        ("general.retention_days", "-1"),
        ("ui.theme", "blue"),
        ("device.choice", "tpu"),
        ("review.policy", "sometimes"),
    ],
)
def test_a_value_a_setting_does_not_allow_is_refused_and_names_the_setting(key: str, text: str) -> None:
    with pytest.raises(AppError) as caught:
        store.changed(Settings(), key, text)
    shown = "a components setting" if key == "components.mfa" else key  # no part has a row until it is set
    assert (caught.value.code, caught.value.detail) == ("E_SETTINGS_INVALID", shown)


@pytest.mark.parametrize(
    "key", ["ui.colour", "nosuch.key", "export", "schema_version", "schema_version.x", "ui.Zorblat Smith"]
)
def test_a_setting_that_does_not_exist_is_refused_without_its_name_typed(key: str) -> None:
    with pytest.raises(AppError) as caught:
        store.changed(Settings(), key, "1")
    assert (caught.value.code, caught.value.detail) == ("E_SETTINGS_INVALID", "no such setting")


def test_saved_settings_load_back_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    changed = Settings.model_validate({"review": {"policy": "never"}, "components": {"pyannote": False}})
    store.save(path, changed)
    assert store.load(path) == store.Loaded(changed)
