"""Load and save settings.toml robustly, and freeze the settings a job runs with."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, ValidationError

from fluentfolio.core.enums import Device
from fluentfolio.core.errors import InputError
from fluentfolio.core.files import atomic_write, read_replaced
from fluentfolio.settings.model import SCHEMA_VERSION, Settings

# One function per schema bump: MIGRATIONS[n] turns a version-n table into version n + 1.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}
SWITCH_TABLES = ("components", "analyzers")
# A light run switches off the three slowest optional parts measured in full analyses: the MFA aligner,
# the phone model and openSMILE.
LIGHT_RUN_OFF = ("mfa", "phone-model", "opensmile")


@dataclass
class Loaded:
    settings: Settings
    warnings: list[tuple[str, str]] = field(default_factory=list)  # (code, "section.key")
    read_only: bool = False  # written by a newer version: used, never saved over


def _merge(raw: dict[str, Any], warnings: list[tuple[str, str]]) -> Settings:
    """Apply each known key on top of the defaults; an unknown key or a bad value only gives a warning."""
    base = Settings().model_dump()
    for name, value in raw.items():
        if name == "schema_version":
            continue
        if name not in base:
            warnings.append(("W_SETTINGS_UNKNOWN_KEY", name))
        elif not isinstance(value, dict):
            warnings.append(("W_SETTINGS_INVALID", name))
        elif name in SWITCH_TABLES:
            for key, on in value.items():
                if isinstance(on, bool):
                    base[name][key] = on
                else:
                    warnings.append(("W_SETTINGS_INVALID", f"{name}.{key}"))
        else:
            section_model: type[BaseModel] = type(getattr(Settings(), name))
            for key, item in value.items():
                if key not in section_model.model_fields:
                    warnings.append(("W_SETTINGS_UNKNOWN_KEY", f"{name}.{key}"))
                    continue
                try:
                    section_model.model_validate({**base[name], key: item})
                except ValidationError:
                    warnings.append(("W_SETTINGS_INVALID", f"{name}.{key}"))
                    continue
                base[name][key] = item
    return Settings.model_validate(base)


def load(path: Path) -> Loaded:
    """Read settings.toml; a damaged file is kept as settings.toml.bad-<utc> and the defaults are used."""
    if not path.exists():
        return Loaded(Settings())
    try:
        raw = tomllib.loads(read_replaced(path))
        version = raw.get("schema_version", SCHEMA_VERSION)
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError("schema_version")
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValueError):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path.replace(path.with_name(f"{path.name}.bad-{stamp}"))
        return Loaded(Settings(), [("W_SETTINGS_UNREADABLE", path.name)])
    warnings: list[tuple[str, str]] = []
    if version > SCHEMA_VERSION:
        warnings.append(("W_SETTINGS_NEWER", str(version)))
    while version < SCHEMA_VERSION:
        raw = MIGRATIONS[version](raw)
        version += 1
    settings = _merge(raw, warnings)
    return Loaded(settings, warnings, read_only=version > SCHEMA_VERSION)


def save(path: Path, settings: Settings) -> None:
    atomic_write(path, tomli_w.dumps(settings.model_dump(mode="json")))


def changed(settings: Settings, key: str, text: str) -> Settings:
    """The settings with one `section.key` set from its text form, checked by the settings model.

    The text is read as a TOML value (true, 120, "dark"); a bare word such as dark is taken as text.
    """
    section, _, name = key.partition(".")
    try:
        value = tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        value = text
    data = settings.model_dump(mode="json")
    table = data.get(section)
    if not isinstance(table, dict) or not name or (section not in SWITCH_TABLES and name not in table):
        raise InputError("E_SETTINGS_INVALID", "no such setting")  # never the name typed
    shown = key if name in table else f"a {section} setting"
    # The value keeps the kind of the default: a switch takes true or false, never "yes" or 1.
    if type(value) is not type(table.get(name, True)):
        raise InputError("E_SETTINGS_INVALID", shown)
    table[name] = value
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise InputError("E_SETTINGS_INVALID", shown) from exc


def snapshot(
    settings: Settings, *, asr: str | None = None, device: Device | None = None, light_run: bool = False
) -> dict[str, Any]:
    """The settings a job keeps for good: the file's values, then this run's --asr, --device and light run."""
    changes: dict[str, Any] = {}
    if asr is not None:
        changes["asr"] = settings.asr.model_copy(update={"model": asr})
    if device is not None:
        changes["device"] = settings.device.model_copy(update={"choice": device})
    if light_run:
        changes["components"] = {**settings.components, **dict.fromkeys(LIGHT_RUN_OFF, False)}
    frozen = settings.model_copy(update=changes).model_dump(mode="json")
    frozen["light_run"] = light_run
    return frozen
