"""MFA refuses stale security-package records before starting its isolated Python environment."""

import sys
from importlib import resources
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import soundfile as sf

from fluentfolio.alignment import mfa
from fluentfolio.core.files import read_json, write_json
from fluentfolio.doctor import checks
from fluentfolio.jobs import runner
from fluentfolio.jobs.loop import Roots
from fluentfolio.models import catalogue
from fluentfolio.models.catalogue import Component
from fluentfolio.setup import components, mfa_env

PACKAGES = ("filelock", "setuptools")
FIELDS = ("name", "version", "build", "md5")
LOCK = f"mfa-env-{'win-64' if sys.platform == 'win32' else 'linux-64'}.txt"


def locked_records(lock: str = LOCK) -> dict[str, dict[str, str]]:
    """Read fixture expectations from the explicit URLs, independently of the runtime checker."""
    text = resources.files("fluentfolio.data").joinpath(lock).read_text(encoding="utf-8")
    records = {}
    for line in text.splitlines():
        if not line.startswith("https://"):
            continue
        url = urlsplit(line)
        name, version, build = Path(url.path).name.removesuffix(".conda").rsplit("-", 2)
        if name in PACKAGES:
            assert name not in records
            records[name] = {"name": name, "version": version, "build": build, "md5": url.fragment}
    assert set(records) == set(PACKAGES)
    return records


def installed_records(app: Path, lock: str = LOCK) -> dict[str, Path]:
    """Two fictional conda records and a placeholder executable; no environment is installed or run."""
    meta = mfa.env_dir(app) / "conda-meta"
    files = {}
    for name, record in locked_records(lock).items():
        path = meta / f"{name}-{record['version']}-{record['build']}.json"
        write_json(path, record)
        files[name] = path
    exe = mfa.python(mfa.env_dir(app))
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"fictional python")
    return files


def tiny_part() -> Component:
    """The real MFA component with tiny fictional downloads at the same destinations."""
    part = catalogue.get("mfa")
    return part.model_copy(update={"file": [f.model_copy(update={"size": 1}) for f in part.file]})


def present_files(part: Component, roots: Roots) -> None:
    for f in part.files_for(catalogue.SYSTEM):
        target = mfa_env.file_target(f.path, roots.app, roots.models)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")


@pytest.mark.parametrize("platform", ["win-64", "linux-64"])
def test_matching_security_records_from_each_platform_lock_are_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    lock = f"mfa-env-{platform}.txt"
    monkeypatch.setattr(mfa, "LOCK", lock, raising=False)
    installed_records(tmp_path, lock)
    # Other package records are outside this check's deliberately narrow scope.
    (mfa.env_dir(tmp_path) / "conda-meta" / "unrelated-1-0.json").write_bytes(b"not JSON")
    assert mfa.env_current(tmp_path)


@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("missing", [False, True])
def test_each_security_record_field_must_match_the_lock(
    tmp_path: Path, package: str, field: str, missing: bool
) -> None:
    path = installed_records(tmp_path)[package]
    record = read_json(path)
    if missing:
        del record[field]
    else:
        record[field] = "incorrect"
    write_json(path, record)  # Its filename still claims the current version and build.
    assert not mfa.env_current(tmp_path)


@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("problem", ["absent", "duplicate", "old-version"])
def test_absent_duplicate_and_stale_security_records_are_not_current(
    tmp_path: Path, package: str, problem: str
) -> None:
    path = installed_records(tmp_path)[package]
    if problem == "absent":
        path.unlink()
    elif problem == "duplicate":
        path.with_name(f"{package}-0.0.1-old_0.json").write_bytes(path.read_bytes())
    else:
        record = read_json(path)
        record["version"] = "0.0.1"
        write_json(path, record)
    assert not mfa.env_current(tmp_path)


@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("raw", [b"{", b"[]", b"null", b"42", b'"a record"', b"\xff"])
def test_malformed_nonobject_and_undecodable_security_records_fail_closed(
    tmp_path: Path, package: str, raw: bytes
) -> None:
    installed_records(tmp_path)[package].write_bytes(raw)
    assert not mfa.env_current(tmp_path)


@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("error", [PermissionError, FileNotFoundError])
def test_unreadable_or_disappearing_security_records_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: str, error: type[OSError]
) -> None:
    unreadable = installed_records(tmp_path)[package]
    real = Path.read_text

    def read(path: Path, *args: object, **kwargs: object) -> str:
        if path == unreadable:
            raise error("fictional unreadable conda record")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert not mfa.env_current(tmp_path)


@pytest.mark.parametrize("problem", ["absent", "duplicate"])
def test_the_bundled_lock_must_name_each_security_package_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    installed_records(tmp_path)
    text = resources.files("fluentfolio.data").joinpath(LOCK).read_text(encoding="utf-8")
    entry = next(line for line in text.splitlines() if "/filelock-" in line)
    text = text.replace(entry, "") if problem == "absent" else text + "\n" + entry + "\n"
    data = tmp_path / "bundled-data"
    data.mkdir()
    (data / LOCK).write_text(text, encoding="utf-8")
    monkeypatch.setattr(mfa.resources, "files", lambda package: data)
    assert not mfa.env_current(tmp_path)


def test_assessing_a_missing_environment_writes_nothing(tmp_path: Path) -> None:
    assert not mfa.env_current(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_installed_version_rejects_stale_records_before_any_process_or_environment_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_records(tmp_path)["filelock"].write_text("{}", encoding="utf-8")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("a stale MFA environment must not be launched or prepared for a launch")

    monkeypatch.setattr(runner, "run", forbidden)
    monkeypatch.setattr(runner, "child_env", forbidden)
    assert mfa_env.installed_version(tmp_path) is None
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_installed_version_still_probes_a_matching_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_records(tmp_path)
    called = []

    def probe(argv: list[str], **kwargs: object) -> runner.Outcome:
        called.append(argv)
        return runner.Outcome(0, stderr_tail=["3.4.3"])

    monkeypatch.setattr(runner, "child_env", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "run", probe)
    assert mfa_env.installed_version(tmp_path) == "3.4.3"
    assert called == [
        [str(mfa.python(mfa.env_dir(tmp_path))), "-I", "-m", "montreal_forced_aligner", "version"]
    ]
    mfa.python(mfa.env_dir(tmp_path)).unlink()
    assert mfa_env.installed_version(tmp_path) is None
    assert len(called) == 1  # The existing executable check still applies.


@pytest.mark.parametrize("platform", ["win-64", "linux-64"])
def test_create_and_freshness_use_the_same_selected_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    lock = f"mfa-env-{platform}.txt"
    monkeypatch.setattr(mfa, "LOCK", lock, raising=False)
    tool = mfa_env.micromamba(tmp_path)
    tool.parent.mkdir()
    tool.write_bytes(b"fictional micromamba")
    used = []

    def run(argv: list[str], **kwargs: object) -> runner.Outcome:
        if "create" in argv:
            used.append(Path(argv[argv.index("--file") + 1]).name)
            installed_records(tmp_path, lock)
        return runner.Outcome(0, stderr_tail=["3.4.3"])

    monkeypatch.setattr(mfa_env, "_env", lambda app: {})
    monkeypatch.setattr(runner, "child_env", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "run", run)
    assert mfa_env.create(tmp_path) == "3.4.3"
    assert used == [lock]


def test_doctor_requests_an_update_for_stale_records_and_accepts_matching_records(
    tmp_path: Path,
) -> None:
    roots = Roots(tmp_path / "app", tmp_path / "data", tmp_path / "models")
    part = tiny_part()
    files = installed_records(roots.app)
    present_files(part, roots)
    assert checks._files_status(part, roots) is None
    files["setuptools"].write_text("{}", encoding="utf-8")
    status, reason = checks._files_status(part, roots)
    assert status == "needs-download" and "update" in reason
    assert checks.component_status(part, roots, {}, False, {}) == (status, reason)
    installed_records(roots.app)
    (roots.models / "mfa" / mfa.ACOUSTIC).unlink()
    assert checks._files_status(part, roots)[0] == "needs-download"


def alignment_job(base: Path) -> tuple[Path, Roots]:
    roots = Roots(base / "app", base / "data", base / "models")
    job = roots.data / "job"
    write_json(
        job / "job.json",
        {
            "job_id": "20261024-000000-abcd",
            "settings": {"components": {"mfa": True}},
        },
    )
    write_json(
        job / "stages" / "words.json",
        {
            "words": [
                {"text": "hello", "start_ms": 100, "end_ms": 300, "cut_off": False},
            ]
        },
    )
    write_json(
        job / "stages" / "structure.json",
        {
            "answers": [{"id": "A1", "candidate": "candidate"}],
            "utterances": [{"id": "U1", "first_word": 0, "last_word": 0, "answer_id": "A1"}],
        },
    )
    write_json(job / "stages" / "media.json", {"duration_ms": 1000})
    (job / "input").mkdir()
    sf.write(str(job / "input" / "audio16k.wav"), [0.0] * mfa.RATE, mfa.RATE)
    present_files(tiny_part(), roots)
    (roots.models / "mfa" / mfa.DICTIONARY).write_text("hello\tHH AH0 L OW1\n", encoding="utf-8")
    installed_records(roots.app)
    return job, roots


def test_prepare_rejects_stale_records_without_writing_a_corpus_and_keeps_disabled_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, roots = alignment_job(tmp_path)
    next((mfa.env_dir(roots.app) / "conda-meta").glob("filelock-*.json")).write_bytes(b"null")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("a stale or disabled MFA environment must not prepare an alignment command")

    monkeypatch.setattr(mfa, "program", forbidden)
    assert mfa.prepare(job, roots.app, roots.models) == {
        "status": "unavailable",
        "code": "E_COMPONENT_UNAVAILABLE",
    }
    assert not (roots.app / "tmp").exists()
    record = read_json(job / "job.json")
    record["settings"]["components"]["mfa"] = False
    write_json(job / "job.json", record)
    assert mfa.prepare(job, roots.app, roots.models) == {"status": "disabled"}


def test_prepare_with_matching_records_keeps_the_corpus_and_file_guards(tmp_path: Path) -> None:
    job, roots = alignment_job(tmp_path)
    result = mfa.prepare(job, roots.app, roots.models)
    assert isinstance(result, mfa.Program)
    assert (result.scratch / "corpus" / "candidate" / "U1.lab").read_text(encoding="utf-8") == "hello\n"
    assert (result.scratch / "dictionary.dict").is_file()
    (roots.models / "mfa" / mfa.ACOUSTIC).unlink()
    assert mfa.prepare(job, roots.app, roots.models) == {
        "status": "unavailable",
        "code": "E_COMPONENT_UNAVAILABLE",
    }


@pytest.mark.parametrize("stale", [False, True])
def test_explicit_component_install_reuses_matching_env_and_rebuilds_stale_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale: bool
) -> None:
    roots = Roots(tmp_path / "app", tmp_path / "data", tmp_path / "models")
    files = installed_records(roots.app)
    if stale:
        record = read_json(files["filelock"])
        record["version"] = "0.0.1"
        write_json(files["filelock"], record)
    rebuilt = []
    probes = []

    def rebuild(app: Path) -> str:
        rebuilt.append(app)
        installed_records(app)
        return "3.4.3"

    def probe(argv: list[str], **kwargs: object) -> runner.Outcome:
        probes.append(argv)
        return runner.Outcome(0, stderr_tail=["3.4.3"])

    def fetch(part: Component, file: object, target: Path, progress: object) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")

    monkeypatch.setattr(components.download, "fetch_file", fetch)
    monkeypatch.setattr(mfa_env, "create", rebuild)
    monkeypatch.setattr(runner, "child_env", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "run", probe)
    components.install(tiny_part(), roots)
    assert rebuilt == ([roots.app] if stale else [])
    assert len(probes) == (0 if stale else 1)
    components.install(tiny_part(), roots)
    assert rebuilt == ([roots.app] if stale else [])
    assert len(probes) == (1 if stale else 2)
