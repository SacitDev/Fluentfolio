"""The installer scripts as a learner's computer runs them: each refusal comes before any download, with its
code, and no script uses a launch shape that security programs flag.

Each installer run loads install.ps1's functions and replaces only the questions put to Windows (the
administrator check, Smart App Control) or a tool's process, then runs the real steps. Every run stops before
the network: the release folder is empty unless a test fills it, and the fallback folder has a space, so the
installer never creates it.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from fluentfolio.core.files import write_json
from fluentfolio.jobs.locks import app_lock
from fluentfolio.setup.token import SAVED_NAME
from tests.support.installer import ROOT, TIMEOUT_S, powershell

# The launch shapes that security programs flag, each with a line of its own kind that the pattern must catch.
BANNED = {
    "Invoke-Expression": (r"\biex\b|invoke-expression", "iex (New-Object Net.WebClient).DownloadString($u)"),
    "irm piped": (r"\birm\b[^\r\n]*\|", "irm https://example.com/get.ps1 | powershell"),
    "an encoded command": (r"-encodedcommand\b|(?<![\w-])-enc\b|(?<![\w-])-e\s", "powershell -e SQBFAFgA"),
    "a hidden window": (r"-windowstyle\s+hidden|(?<![\w-])-w\s+hidden", "powershell -W Hidden -File a.ps1"),
    "DownloadString": (r"downloadstring", "$c.DownloadString('https://example.com')"),
    "mshta": (r"\bmshta\b", "mshta https://example.com/a.hta"),
    "rundll32": (r"\brundll32\b", "rundll32 url.dll,FileProtocolHandler https://example.com"),
    "wscript or cscript": (r"\b[wc]script(?:\.exe\b|\s|$)", "cscript //nologo run.vbs"),
    "certutil": (r"\bcertutil\b", "certutil -urlcache -f https://example.com/a.exe a.exe"),
    "bitsadmin": (r"\bbitsadmin\b", "bitsadmin /transfer job https://example.com/a.exe a.exe"),
    "python -c": (r"\bpython[\w.]*(?:\s+-\w+)*\s+-c\b", 'python3 -I -c "import os"'),
    "the temp folder": (
        r"\$env:te?mp\b|%te?mp%|\bmktemp\b|/tmp\b|gettemppath",
        'Set-Content "$env:TEMP\\a.ps1"',
    ),
    "Defender settings": (r"\b(?:add|set)-mppreference\b", "Add-MpPreference -ExclusionPath ff"),
    "registry writes": (
        r"\b(?:new|set)-itemproperty\b|\breg(?:\.exe)?\s+add\b",
        "reg add HKCU\\Software\\X /f",
    ),
    "PATH edits": (r"\bsetx\b|setenvironmentvariable", 'setx PATH "%PATH%;ff"'),
}
# The line Install.cmd and Update.cmd run.
ALLOWED_LAUNCH = 'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"'
SKIPPED_FOLDERS = {".git", ".venv", "__pycache__", "node_modules"}


def installer_files() -> list[Path]:
    found = []
    for folder, subfolders, files in os.walk(ROOT):
        subfolders[:] = [s for s in subfolders if s not in SKIPPED_FOLDERS]
        found += [Path(folder) / f for f in files if f.endswith((".ps1", ".cmd", ".sh"))]
    return found


def test_the_installer_files_use_none_of_the_launch_shapes_that_security_programs_flag() -> None:
    files = installer_files()
    assert {"install.ps1", "uninstall.ps1", "Install.cmd", "Update.cmd"} <= {f.name for f in files}
    for name, (pattern, planted) in BANNED.items():
        assert re.search(pattern, planted, re.IGNORECASE), name  # the pattern catches its own shape
        assert not re.search(pattern, ALLOWED_LAUNCH, re.IGNORECASE), name
        for file in files:
            for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                assert not re.search(pattern, line, re.IGNORECASE), f"{file.name}:{number}: {name}"
    for launcher in ("Install.cmd", "Update.cmd"):
        assert ALLOWED_LAUNCH in (ROOT / launcher).read_text(encoding="utf-8"), launcher


def test_the_test_runs_give_windows_powershell_its_own_module_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A PowerShell 7 shell passes its module folders on, and Windows PowerShell cannot load 7's modules.
    foreign = tmp_path / "other modules"
    foreign.mkdir()
    monkeypatch.setenv("PSModulePath", f"{foreign};{os.environ.get('PSModulePath', '')}")
    done = powershell(tmp_path, "$env:PSModulePath")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "WindowsPowerShell" in done.stdout and str(foreign) not in done.stdout, done.stdout


@pytest.mark.parametrize(("launcher", "arguments"), [("Install.cmd", ""), ("Update.cmd", "-Update")])
def test_the_launchers_give_windows_powershell_its_own_module_path(
    tmp_path: Path, launcher: str, arguments: str
) -> None:
    # The real launcher, started as from a PowerShell 7 window, runs a stand-in for the installer next to it.
    shutil.copy(ROOT / launcher, tmp_path / launcher)
    stand_in = "'module path: ' + $env:PSModulePath; 'arguments: ' + ($args -join ' ')"
    (tmp_path / "install.ps1").write_text(stand_in + "\n", encoding="utf-8")
    foreign = tmp_path / "other modules"
    foreign.mkdir()
    env = {k: v for k, v in os.environ.items() if k.upper() != "PSMODULEPATH"}
    env |= {"PSModulePath": f"{foreign};{os.environ.get('PSModulePath', '')}",
            "APPDATA": str(tmp_path / "roaming"), "LOCALAPPDATA": str(tmp_path / "local"),
            "USERPROFILE": str(tmp_path)}  # fmt: skip
    cmd = Path(os.environ["SYSTEMROOT"]) / "System32" / "cmd.exe"
    done = subprocess.run([str(cmd), "/d", "/c", str(tmp_path / launcher)], input="\n",
                          capture_output=True, text=True, errors="replace", timeout=TIMEOUT_S, env=env,
                          check=False)  # fmt: skip
    assert f"arguments: {arguments}\n" in done.stdout, done.stdout + done.stderr  # the same launch as before
    assert "module path: " in done.stdout and str(foreign) not in done.stdout, done.stdout


def test_the_bootstrap_scripts_stay_thin() -> None:
    for script in ("install.ps1", "install.sh"):  # 250 lines, raised by 60 for the progress display
        assert len((ROOT / script).read_text(encoding="utf-8").splitlines()) <= 310, script


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, str]:
    """An earlier install's app folder (so the folder rule passes on any temporary path), an empty release
    folder and a fallback folder that the rule refuses."""
    write_json(tmp_path / "app" / "state.json", {"version": "0.0.9", "backend": "cpu", "extras": []})
    (tmp_path / "release").mkdir()
    return {"FF_APP": str(tmp_path / "app"), "FF_SOURCE": str(tmp_path / "release"),
            "FF_FALLBACK": str(tmp_path / "fall back")}  # fmt: skip


LOAD = (
    ". $env:FF_SCRIPT -AppRoot $env:FF_APP -FallbackRoot $env:FF_FALLBACK -Source $env:FF_SOURCE"
    " -PreflightOnly"
)
ORDINARY = f"{LOAD} -AllowElevatedTest; function Get-SacState {{ 0 }}; "  # neither question stops the run


def stopped(done: subprocess.CompletedProcess[str], code: str) -> str:
    assert done.returncode == 1, done.stdout + done.stderr
    assert f"Code: {code} (docs/troubleshooting.md#{code.lower()})." in done.stdout, done.stdout + done.stderr
    return done.stdout


def app_files(setup: dict[str, str]) -> list[str]:
    app = Path(setup["FF_APP"])
    return sorted(str(p.relative_to(app)) for p in app.rglob("*"))


def test_an_unsupported_processor_or_an_administrator_window_is_refused_before_anything_is_written(
    tmp_path: Path, setup: dict[str, str]
) -> None:
    env = {**setup, "PROCESSOR_ARCHITECTURE": "ARM64"}
    out = stopped(powershell(tmp_path, f"{ORDINARY} Main", **env), "E_NOT_SUPPORTED")
    assert "64-bit Intel or AMD processor" in out
    elevated = f"{LOAD}; function Test-Elevated {{ $true }}; Main"
    out = stopped(powershell(tmp_path, elevated, **setup), "E_ELEVATED")
    assert 'without "Run as administrator"' in out
    assert app_files(setup) == ["state.json"]


def test_smart_app_control_on_stops_the_install_and_evaluation_mode_warns(
    tmp_path: Path, setup: dict[str, str]
) -> None:
    def sac(state: str) -> subprocess.CompletedProcess[str]:
        """A run where Windows reports this Smart App Control state."""
        command = f"{LOAD} -AllowElevatedTest; function Get-SacState {{ {state} }}; Main"
        return powershell(tmp_path, command, **setup)

    out = stopped(sac("1"), "E_SAC_ENFORCED")
    assert "Smart App Control is on." in out and app_files(setup) == ["state.json"]
    out = stopped(sac("2"), "E_SOURCE")
    assert "Note (W_SAC_EVALUATION): Smart App Control is in evaluation mode" in out
    for off in ("0", "$null"):
        assert "Smart App Control" not in stopped(sac(off), "E_SOURCE")


def test_a_folder_without_a_release_is_refused_and_logged(tmp_path: Path, setup: dict[str, str]) -> None:
    out = stopped(powershell(tmp_path, f"{ORDINARY} Main", **setup), "E_SOURCE")
    assert f"No Fluentfolio release was found in {setup['FF_SOURCE']}." in out
    log = Path(setup["FF_APP"]) / "logs" / "install.log"
    assert f"Details: {log}" in out and "Code: E_SOURCE" in log.read_text(encoding="utf-8")


def test_an_error_no_step_expected_ends_with_the_plain_lines_and_its_details_in_the_log(
    tmp_path: Path, setup: dict[str, str]
) -> None:
    broken = tmp_path / "release.zip"
    broken.write_bytes(b"not a zip file")
    done = powershell(tmp_path, f"{ORDINARY} Main", **{**setup, "FF_SOURCE": str(broken)})
    out = stopped(done, "E_INSTALL_FAILED")
    assert "Problem: The install stopped at an unexpected error: " in out and "What to do: " in out
    assert "At line" not in out and "CategoryInfo" not in out  # no PowerShell error record on the screen
    log = (Path(setup["FF_APP"]) / "logs" / "install.log").read_text(encoding="utf-8")
    assert "CategoryInfo" in log or "At line" in log or "Expand-Archive" in log  # the record went to the log


def test_the_folder_rule_at_its_edges(tmp_path: Path) -> None:
    root = "C:\\"
    paths = {
        "FF_P1": root + "a" * 56,  # 59 characters
        "FF_P2": root + "a" * 57,
        "FF_P3": root + "a" * 58,
        "FF_P4": root + "Zo\u00eb \u00dc",
        "FF_P5": "C:\\Users\\x\\CompanySync\\ff",  # synced by OneDrive under another name
        "FF_P6": "\\\\server\\share\\ff",
    }
    lines = "; ".join(f"'{name}=' + ((Get-PathProblems $env:{name}) -join '|')" for name in paths)
    synced = {"OneDriveCommercial": "C:\\Users\\x\\CompanySync"}
    done = powershell(tmp_path, f"{ORDINARY} {lines}", **synced, **paths)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == [
        "FF_P1=",
        "FF_P2=60 characters (at most 59)",
        "FF_P3=61 characters (at most 59)",
        "FF_P4=letters outside A to Z|a space",
        "FF_P5=inside OneDrive",
        "FF_P6=not a local NTFS disk",
    ]


def test_the_shortcut_reaches_a_start_menu_whose_path_has_letters_outside_a_to_z(tmp_path: Path) -> None:
    roaming = tmp_path / "Fluent Test Şükrü سینا" / "roaming"
    app = tmp_path / "app"
    app.mkdir()
    make = f"{ORDINARY} New-Link 'Fluentfolio' $env:ComSpec '/k' 'Start it.'"
    done = powershell(tmp_path, make, APPDATA=str(roaming), FF_APP=str(app))
    assert done.returncode == 0, done.stdout + done.stderr
    link = roaming / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Fluentfolio.lnk"
    assert "Start it.".encode("utf-16-le") in link.read_bytes()  # the saved shortcut, with its description
    assert not any(app.iterdir())  # the draft in the app folder was moved, not copied


def test_a_folder_that_breaks_the_rule_with_no_usable_fallback_stops_and_creates_nothing(
    tmp_path: Path, setup: dict[str, str]
) -> None:
    usual = tmp_path / "my app"
    done = powershell(tmp_path, f"{ORDINARY} Main", **{**setup, "FF_APP": str(usual)})
    out = stopped(done, "E_PATH_RULE")
    assert f"The usual folder {usual} cannot be used (a space" in out
    assert not usual.exists() and not Path(setup["FF_FALLBACK"]).exists()


def test_too_little_disk_space_stops_the_install_with_both_numbers(tmp_path: Path) -> None:
    gb = 10**9  # decimal, as the app's pages print sizes
    out = stopped(powershell(tmp_path, f"{ORDINARY} Assert-Disk 3 {int(2.5 * gb)}"), "E_DISK_PREFLIGHT")
    assert "Not enough free disk space: need 3.0 GB, have 2.5 GB." in out
    stopped(powershell(tmp_path, f"{ORDINARY} Assert-Disk 3 {3 * gb - 1}"), "E_DISK_PREFLIGHT")
    done = powershell(tmp_path, f"{ORDINARY} Assert-Disk 3 {3 * gb}; Assert-Disk 3 {3 * gb + 1}; 'enough'")
    assert (done.returncode, done.stdout.strip()) == (0, "enough"), done.stderr


def test_a_missing_visual_cpp_runtime_is_named_with_its_download(
    tmp_path: Path, setup: dict[str, str]
) -> None:
    project = '[project]\nname = "fluentfolio"\nversion = "9.9.9"\n'
    (Path(setup["FF_SOURCE"]) / "pyproject.toml").write_text(project, encoding="utf-8")
    # A runtime file that is not there.
    command = f"{ORDINARY} $VcRuntime = '{tmp_path / 'vcruntime140_1.dll'}'; Main"
    out = stopped(powershell(tmp_path, command, **setup), "E_VCRUNTIME_MISSING")
    assert "https://aka.ms/vs/17/release/vc_redist.x64.exe" in out


@pytest.mark.parametrize(("device", "shown"), [("auto", True), ("gpu", True), ("cpu", False)])
@pytest.mark.parametrize(("first", "about"), [(True, "about"), (False, "at most about")])
def test_the_confirmation_names_the_sizes_and_times_and_no_answer_installs_nothing(
    tmp_path: Path, setup: dict[str, str], device: str, shown: bool, first: bool, about: str
) -> None:
    project = '[project]\nname = "fluentfolio"\nversion = "9.9.9"\n'
    (Path(setup["FF_SOURCE"]) / "pyproject.toml").write_text(project, encoding="utf-8")
    quiet = "function Get-SacState { 0 }; function Test-Hosts { }; function Measure-Speed { 3000000 }"
    if first:  # no earlier install, so uv has downloaded nothing yet; the folder rule is tested on its own
        quiet += "; function Get-State { $null }; function Get-PathProblems { return , @() }"
    load = LOAD.replace(" -PreflightOnly", f" -Device {device} -AllowElevatedTest")
    done = powershell(tmp_path, f"{load}; {quiet}; Main", **setup)
    assert done.returncode == 0, done.stdout + done.stderr
    what = "installed" if first else "updated from 0.0.9"
    for line in (
        f"Fluentfolio 9.9.9 will be {what} for this Windows user. No admin rights are needed.",
        (
            f"  Download now: {about} 0.5 GB (uv, Python 3.12 and the packages). "
            "Disk space needed: about 3 GB; free: "
        ),
        (  # 500 MB at 3 MB/s; the rest of the step is not measured beforehand, so the line says so
            f"  Time: {about} 3 minutes to download at the speed measured now (3.0 MB/s). "
            "Unpacking and slower servers can double or triple that."
        ),
        "Nothing was installed.",
    ):
        assert line in done.stdout, line
    gpu = (
        f"  If an NVIDIA graphics card can be used, GPU support adds {about} 2.7 GB to download"
        f" ({about} 15 more minutes) and about 4 GB of disk space."
    )
    assert (gpu in done.stdout) is shown  # 2,700 MB at 3 MB/s is 900 s
    assert ("at most" in done.stdout) is not first
    assert app_files(setup) == ["logs", "logs\\install.log", "state.json"]


def test_a_running_app_stops_the_installer_and_the_uninstaller(tmp_path: Path, setup: dict[str, str]) -> None:
    data = Path(os.environ["FLUENTFOLIO_HOME"])
    lock = app_lock(data).acquire()
    try:
        stopped(powershell(tmp_path, f"{ORDINARY} Main", **setup), "E_APP_RUNNING")
        uninstall = f"& '{ROOT / 'uninstall.ps1'}' -AppRoot $env:FF_APP -FallbackRoot $env:FF_FALLBACK -Yes"
        stopped(powershell(tmp_path, uninstall, **setup), "E_APP_RUNNING")
        assert (Path(setup["FF_APP"]) / "state.json").is_file()
    finally:
        lock.release()
    stopped(powershell(tmp_path, f"{ORDINARY} Main", **setup), "E_SOURCE")  # a lock file nobody holds is old


def test_a_blocked_network_is_named_with_the_proxy_in_use(tmp_path: Path) -> None:
    check = f"{ORDINARY} $Hosts = @('https://127.0.0.1:9'); Test-Hosts"
    out = stopped(powershell(tmp_path, check), "E_NETWORK")  # the tests' dead proxy is set
    assert "https://127.0.0.1:9 could not be reached through the proxy http://127.0.0.1:9." in out
    direct = {name: "" for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")}
    out = stopped(powershell(tmp_path, check, **direct), "E_NETWORK")
    assert "https://127.0.0.1:9 could not be reached." in out
    # A user name and password in the proxy address stay out of the message and the log a learner may send;
    # a / inside the password counts too.
    log = tmp_path / "install.log"
    logged = f"{ORDINARY} $script:LogFile = '{log}'; $Hosts = @('https://127.0.0.1:9'); Test-Hosts"
    for proxy in ("http://learner:s3cret@127.0.0.1:9", "http://learner:s3/cret@127.0.0.1:9"):
        log.unlink(missing_ok=True)
        out = stopped(powershell(tmp_path, logged, HTTPS_PROXY=proxy), "E_NETWORK")
        assert "could not be reached through the proxy http://127.0.0.1:9." in out
        shown = out + log.read_text(encoding="utf-8-sig")
        assert "learner" not in shown and "s3cret" not in shown and "s3/cret" not in shown


def tool(tmp_path: Path, says: str, exit_code: int, before: str = "") -> subprocess.CompletedProcess[str]:
    """Invoke-Tool on a stand-in tool process that prints `says` and ends with `exit_code`."""
    run = f"Invoke-Tool cmd.exe @('/c', 'echo {says}& exit /b {exit_code}')"
    log = tmp_path / "install.log"
    return powershell(tmp_path, f"{ORDINARY} $script:LogFile = '{log}'; {before} {run}")


@pytest.mark.parametrize(
    ("says", "code"),
    [
        ("error: failed to remove file: Access is denied. (os error 4551)", "E_SAC_ENFORCED"),
        (
            "error: Operation did not complete successfully because the file contains a virus (os error 225)",
            "E_FILE_BLOCKED",
        ),
        ("error: There is not enough space on the disk. (os error 112)", "E_DISK_FULL"),
        ("error: invalid peer certificate: UnknownIssuer", "E_TLS_INTERCEPTED"),
        ("error: dns error: No such host is known.", "E_NETWORK"),
        ("error: something nobody has seen before", "E_INSTALL_FAILED"),
    ],
)
def test_a_failed_tool_gets_its_code_from_its_own_words(tmp_path: Path, says: str, code: str) -> None:
    stopped(tool(tmp_path, says, 2), code)
    assert says in (tmp_path / "install.log").read_text(encoding="utf-8")


def test_a_tool_that_works_shows_its_lines_without_the_package_list_and_logs_all(tmp_path: Path) -> None:
    done = tool(tmp_path, "+ numpy==2.5.3& echo Installed 1 package", 0)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Installed 1 package" in done.stdout and "numpy" not in done.stdout
    assert "+ numpy==2.5.3" in (tmp_path / "install.log").read_text(encoding="utf-8")


ASCII_BAR = (
    "-replace [char]0x2588, '#' -replace [char]0x2591, '-'"  # the blocks, as the test's output shows them
)


@pytest.mark.parametrize(
    ("done", "total", "seconds", "shown"),
    [  # worked by hand: 24 blocks, the share rounded down, MB of 10^6 bytes, the time left rounded up
        (50e6, 200e6, 10, "  ######------------------  25%  50.0 of 200.0 MB  5.0 MB/s, about 30 s left"),
        (10e6, 200e6, 60, "  #-----------------------   5%  10.0 of 200.0 MB  0.2 MB/s, about 19 min left"),
        (50e6, 200e6, 0.5, "  ######------------------  25%  50.0 of 200.0 MB"),  # under 1 s: no speed yet
        (16e6, 16e6, 3, "  ######################## 100%  16.0 of 16.0 MB"),
        (0, 0, 5, "  ------------------------   0%  0.0 of 0.0 MB"),
    ],
)
def test_the_bar_shows_the_share_the_sizes_the_speed_and_the_time_left(
    tmp_path: Path, done: float, total: float, seconds: float, shown: str
) -> None:
    out = powershell(tmp_path, f"{ORDINARY} [CultureInfo]::CurrentCulture = 'en-US'; "
                     f"Write-Host ((Get-Bar {done} {total} {seconds}) {ASCII_BAR})").stdout  # fmt: skip
    assert out.splitlines() == [shown], out


def test_in_a_real_console_the_downloads_of_a_tool_become_one_bar_and_its_other_lines_stay(
    tmp_path: Path,
) -> None:
    # uv's own words when its output is captured: each large download with its size, then when it is done.
    says = (
        "Downloading torch (118.3MiB)& echo Downloading numpy (12.0MiB)& echo Resolved 2 packages"
        "& echo  Downloaded numpy& echo  Downloaded torch& echo + numpy==2.5.3& echo Installed 2 packages"
    )
    record = f"function Show-Live([string]$Text) {{ Write-Host ('BAR:' + ($Text {ASCII_BAR})) }};"
    done = tool(tmp_path, says, 0, f"[CultureInfo]::CurrentCulture = 'en-US'; $Live = $true; {record}")
    assert done.returncode == 0, done.stdout + done.stderr
    bars = [
        re.sub(r" MB  .*", " MB", line[4:]) for line in done.stdout.splitlines() if line.startswith("BAR:")
    ]
    # 118.3 MiB is 124.0 MB; with 12.0 MiB (12.6 MB) 136.6 MB, of which 12.6 MB is 9 %.
    assert [b[27:] for b in bars] == [
        "  0%  0.0 of 124.0 MB", "  0%  0.0 of 136.6 MB", "  9%  12.6 of 136.6 MB", "100%  136.6 of 136.6 MB",
    ]  # fmt: skip
    assert bars[2][:26] == "  ##" + "-" * 22  # 9 % of 24 blocks
    shown = [line for line in done.stdout.splitlines() if not line.startswith("BAR:")]
    assert "  Resolved 2 packages" in shown and "  Installed 2 packages" in shown
    assert not [line for line in shown if "Download" in line or "numpy" in line]
    assert "Downloaded torch" in (tmp_path / "install.log").read_text(encoding="utf-8")
    # A captured output keeps the plain lines: no bar is drawn.
    plain = tool(tmp_path, says, 0)
    assert "BAR:" not in plain.stdout and "  Downloading numpy (12.0MiB)" in plain.stdout.splitlines()


def test_a_failed_update_removes_its_new_folder_and_keeps_the_old_version(tmp_path: Path) -> None:
    old, new = tmp_path / "app" / "app" / "0.0.9", tmp_path / "app" / "app" / "0.1.0"
    for folder in (old, new):
        (folder / ".venv").mkdir(parents=True)
    before = f"$script:Kept, $script:NewApp = '0.0.9', '{new}';"
    failed = tool(tmp_path, "error: dns error: No such host is known.", 2, before)
    out = stopped(failed, "E_UPDATE_FAILED_OLD_KEPT")
    assert "A download failed. Finished downloads are kept. Version 0.0.9 is kept and still works." in out
    assert "What to do: Check the internet connection" in out  # the cause's own action stays
    assert old.is_dir() and not new.exists()


def test_uninstall_removes_the_program_and_keeps_the_learners_data(
    tmp_path: Path, setup: dict[str, str]
) -> None:
    uninstall = f"& '{ROOT / 'uninstall.ps1'}' -AppRoot $env:FF_APP -FallbackRoot $env:FF_FALLBACK -Yes"
    missing = {**setup, "FF_APP": str(tmp_path / "none")}
    stopped(powershell(tmp_path, uninstall, **missing), "E_NOT_INSTALLED")
    app, data = Path(setup["FF_APP"]), Path(os.environ["FLUENTFOLIO_HOME"])
    (app / "app" / "0.0.9" / ".venv").mkdir(parents=True)
    (app / "cache" / "uv").mkdir(parents=True)
    with (app / "cache" / "uv" / "big.bin").open("wb") as big:
        big.truncate(160_000_000)  # 0.16 GB in decimal units; 0.149 if divided by 2**30
    # uv links its cached files into the venv: one file with two names takes its space once (0.32 if twice)
    os.link(app / "cache" / "uv" / "big.bin", app / "app" / "0.0.9" / ".venv" / "big.bin")
    (data / "jobs").mkdir(parents=True)
    (data / "settings.toml").write_text("", encoding="utf-8")
    menu = tmp_path / "roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    menu.mkdir(parents=True)
    for name in ("Fluentfolio.lnk", "Uninstall Fluentfolio.lnk", "Other program.lnk"):
        (menu / name).write_bytes(b"")
    done = powershell(tmp_path, uninstall, **setup)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "(0.2 GB: the program, its tools and the speech models)" in done.stdout
    assert not app.exists() and (data / "settings.toml").is_file() and (data / "jobs").is_dir()
    assert sorted(p.name for p in menu.iterdir()) == ["Other program.lnk"]
    assert f"Your recordings and results are still in {data}." in done.stdout


def test_each_question_names_the_folder_it_asks_about(tmp_path: Path, setup: dict[str, str]) -> None:
    # Windows PowerShell shows no question for answers typed ahead, as the Start-menu cases below type them,
    # so here Read-Host, where the script meets the keyboard, shows each question and answers yes.
    answer = 'function Read-Host($Prompt) { Write-Host "ASKED: $Prompt"; "y" }'
    uninstall = f"{answer}; & '{ROOT / 'uninstall.ps1'}' -AppRoot $env:FF_APP -FallbackRoot $env:FF_FALLBACK"
    app, models = Path(setup["FF_APP"]), tmp_path / "models"
    data = models / "recordings"  # inside the models folder: with yes to both, both go
    write_json(app / "state.json", {"version": "0.0.9", "app_dir": "app/0.0.9", "models_dir": str(models)})
    (data / "jobs").mkdir(parents=True)
    done = powershell(tmp_path, uninstall, **setup, FLUENTFOLIO_HOME=str(data))
    end = "? Type Y and press Enter (just Enter says no)"
    asked = [line.removeprefix("ASKED: ") for line in done.stdout.splitlines() if line.startswith("ASKED: ")]
    assert asked == [
        f"Remove Fluentfolio{end}", f"Also delete your recordings and results in {data}{end}",
        f"Also delete your models folder {models}{end}",
    ], done.stdout + done.stderr  # fmt: skip
    assert done.returncode == 0 and not app.exists() and not models.exists(), done.stdout + done.stderr


WAIT = "Press Enter to close this window."


def renames(folder: Path) -> bool:
    """Whether the folder can be renamed: Windows refuses while a program has it as its current folder."""
    moved = folder.with_name(f"{folder.name}-moved")
    try:
        folder.rename(moved)
    except OSError:
        return False
    moved.rename(folder)
    return True


def start_menu_uninstall(
    home: Path, app: Path, answers: str, shortcut: bool, env: dict[str, str]
) -> tuple[int, str, bool | None]:
    """Run the uninstaller as its shortcut does; type `answers`, then Enter when it asks to close."""
    # The shortcut's command line from install.ps1, started in the app folder, its working folder. Without
    # `shortcut`, the same script is given its app folder by name, and must end without asking.
    script = (ROOT / "install.ps1").read_text(encoding="utf-8")
    line = next(text for text in script.splitlines() if "New-Link 'Uninstall Fluentfolio'" in text)
    exe = Path(os.environ["SYSTEMROOT"]) / re.findall(r"\$env:SystemRoot '([^']+)'", line)[0]
    # The arguments, a PowerShell string, with nothing in it to expand but the app folder.
    template = re.findall(r'"((?:`.|[^"`])*)"', line)[0].replace('`"', '"')
    assert "$" not in template.replace("$AppRoot", "") and "`" not in template, template
    assert template.endswith(" -FromShortcut"), template
    if not shortcut:
        template = template.removesuffix(" -FromShortcut") + ' -AppRoot "$AppRoot"'
    # The fallback goes to an empty folder too, so the run can never reach a real install.
    command = f'"{exe}" {template.replace("$AppRoot", str(app))} -FallbackRoot "{home / "none"}"'
    own = {"APPDATA": str(home / "roaming"), "LOCALAPPDATA": str(home / "local"), "USERPROFILE": str(home)}
    # Hugging Face's own files lie in the test's folder and no token comes from the environment, so the tool
    # that clears the token never reaches a real one.
    own |= {"HF_HOME": str(home / "hf"), "HF_TOKEN_PATH": str(home / "hf" / "token")}
    inherited = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    # Known limit: input is a pipe, so the script skips its input flush and no case here covers it; revisit if
    # the end of the uninstaller changes.
    child = subprocess.Popen(
        command, cwd=app, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", env={**inherited, **own, **env},
    )  # fmt: skip
    stdin, stdout = child.stdin, child.stdout
    assert stdin and stdout
    out: list[str] = []
    turn = threading.Event()

    def read() -> None:
        for text in stdout:
            out.append(text)
            if text.strip() == WAIT:
                turn.set()
        turn.set()  # the window ended without asking

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    free = None  # whether the app folder, if still there, could be renamed while the window waited
    try:
        stdin.write(answers)
        stdin.flush()
        if shortcut:
            turn.wait(TIMEOUT_S)
            time.sleep(1)  # time for a window that does not wait to end
            asked = any(text.strip() == WAIT for text in out)
            assert asked and child.poll() is None, "".join(out) + "\nThe window did not wait for Enter."
            free = renames(app) if app.exists() else None
            stdin.write("\n")
            stdin.flush()
        code = child.wait(TIMEOUT_S)  # input stays open, so a window that asks for more never ends
    finally:
        if child.poll() is None:  # a failed check leaves no window behind
            child.kill()
            child.wait()
        with contextlib.suppress(OSError):  # keeps the first error in view
            stdin.close()
    reader.join(TIMEOUT_S)
    return code, "".join(out), free


REMOVED = "Fluentfolio has been removed."
FAILED = "Code: E_UNINSTALL_FAILED (docs/troubleshooting.md#e_uninstall_failed)."
UNFINISHED = "The uninstall stopped before removing the program: removing the Hugging Face token did not"
KEPT = "The Hugging Face token that Fluentfolio saved was not removed."
LINKS = ["Fluentfolio.lnk", "Other program.lnk", "Uninstall Fluentfolio.lnk"]
NESTED = {  # a recordings folder and a models folder beside the app folder, one inside the other
    "recordings in the models deleted": ("models\\recordings", "models"),
    "models in the recordings deleted": ("recordings", "recordings\\models"),
}


def start_menu_app(app: Path, case: str) -> None:
    """The app folder an install leaves, with the record and the command line tool the case needs."""
    state = {"version": "0.0.9", "backend": "cpu", "extras": [], "app_dir": "app/0.0.9"}
    if case.startswith("token"):
        state[SAVED_NAME] = f"fluentfolio-test-{uuid.uuid4().hex}"  # a name that no real saved token has
    if case == "models record broken":  # an error that no step expects, before anything is removed
        state["models_dir"] = "C:\\bad|name"
    if case == "models moved back inside":  # moved out in Settings, then back to where they were at first
        state["models_dir"] = str(app / "models")
    if case in NESTED:
        state["models_dir"] = str(app.parent / NESTED[case][1])
    write_json(app / "state.json", state)
    shutil.copy(ROOT / "uninstall.ps1", app)  # the installer copies it there
    if case == "record unreadable":
        (app / "state.json").write_text('{"version": "0.0', encoding="utf-8")
    if case.startswith("token"):
        venv = app / "app" / "0.0.9" / ".venv" / "Scripts"
        venv.mkdir(parents=True)
        if case == "token tool broken":
            (venv / "fluentfolio.exe").write_text("not a program", encoding="utf-8")
        else:  # the real command line tool, which clears the record in the folder the uninstaller names
            shutil.copy(Path(sys.executable).with_name("fluentfolio.exe"), venv)


@pytest.mark.parametrize(
    ("case", "answers", "code", "said"),
    [
        ("yes", "y\n\n", 0, [REMOVED]),  # yes to the program, no to the recordings
        ("no", "\n", 0, ["Nothing was removed."]),
        ("app running", "", 1, ["Code: E_APP_RUNNING (docs/troubleshooting.md#e_app_running)."]),
        ("file in use", "y\n\n", 1, [FAILED, "stopped while removing files", "what is left of {app} in"]),
        ("yes, folder named", "y\n\n", 0, [REMOVED]),
        ("file in use, folder named", "y\n\n", 1, [FAILED]),
        ("record unreadable", "", 1, [FAILED, "Nothing was removed: {app}\\state.json cannot be read."]),
        ("recordings inside", "y\n\n", 1, [FAILED, "Nothing was removed: {app}\\recordings is inside"]),
        ("recordings inside, deleted too", "y\ny\n", 0, [REMOVED]),
        ("models moved back inside", "y\n\n", 0, [REMOVED]),  # they go with the program: no question
        ("models record broken", "y\n", 1, [FAILED, "Nothing was removed: ", "Illegal characters in"]),
        ("recordings in the models deleted", "y\n\ny\n", 1, [FAILED]),  # yes to the models only
        ("models in the recordings deleted", "y\ny\n\n", 1, [FAILED]),  # yes to the recordings only
        # No token is saved in these cases: the tool only clears the record. test_token.py removes real ones.
        ("token, yes", "y\n\ny\n", 0, [REMOVED]),  # yes, no to the recordings, yes
        ("token, no", "y\n\n\n", 0, [REMOVED, KEPT]),
        ("token tool fails", "y\n\ny\n", 1, [FAILED, UNFINISHED, "Its program ended with exit code 1;"]),
        ("token tool broken", "y\n\ny\n", 1, [FAILED, UNFINISHED]),
    ],
)
def test_the_start_menu_uninstaller_removes_the_folder_it_starts_in_and_waits_for_enter(
    tmp_path: Path, case: str, answers: str, code: int, said: list[str]
) -> None:
    app = tmp_path / "apps" / "Fluentfolio"  # a folder the learner chose: the shortcut must find it
    start_menu_app(app, case)
    menu = tmp_path / "roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    menu.mkdir(parents=True)
    for name in LINKS:
        (menu / name).write_bytes(b"")
    env = {}
    beside = app.with_name("Fluentfolio-data")  # the usual data folder: its name starts like the app folder's
    if case == "yes":
        env["FLUENTFOLIO_HOME"] = str(beside)
        beside.mkdir()
        (beside / "settings.toml").write_text("", encoding="utf-8")
    if case.startswith("recordings inside"):  # set by the learner, as the move in Settings never allows
        env["FLUENTFOLIO_HOME"] = str(app / "recordings")
        (app / "recordings").mkdir()
        (app / "recordings" / "settings.toml").write_text("", encoding="utf-8")
    if case in NESTED:
        recordings, models = (app.parent / name for name in NESTED[case])
        env["FLUENTFOLIO_HOME"] = str(recordings)
        for folder in recordings, models:
            folder.mkdir(parents=True, exist_ok=True)
        (recordings / "settings.toml").write_text("", encoding="utf-8")
    shortcut = not case.endswith("folder named")
    with contextlib.ExitStack() as held:
        if case == "app running":
            held.enter_context(app_lock(Path(os.environ["FLUENTFOLIO_HOME"])))
        if case.startswith("file in use"):  # Python opens a file without delete sharing, so it stays
            held.enter_context((app / "open.txt").open("w"))
        if case == "token tool fails":  # so the tool cannot replace the record it changed
            held.enter_context((app / "state.json").open())
        exit_code, out, free = start_menu_uninstall(tmp_path, app, answers, shortcut, env)
    assert exit_code == code and all(text.format(app=app) in out for text in said), out
    assert (WAIT in out) is shortcut, out
    assert "CategoryInfo" not in out  # no PowerShell error record on the screen
    assert ("open.txt" in out) is case.startswith("file in use")  # the stop names what is still in use
    assert ("was not removed" in out) is (case == "token, no")  # only a token the learner kept is named
    assert "Your models are still in" not in out  # no case ends with models kept outside the app folder
    assert (beside / "settings.toml").exists() is (case == "yes")  # beside the app folder, so it stays
    gone = REMOVED in said
    assert app.exists() is not gone
    # While the window waits, the learner can delete what is left by hand: the window holds no app folder.
    in_use = case.startswith("file in use") or case == "token tool fails"  # the test holds a file in it open
    assert free is (None if gone or not shortcut else not in_use), out
    # The shortcuts go just before the folder; any other program's stays.
    links = ["Other program.lnk"] if gone or case.startswith("file in use") else LINKS
    assert sorted(p.name for p in menu.iterdir()) == links
    assert not (tmp_path / "hf").exists()  # nothing was written where Hugging Face keeps tokens
    if case in NESTED:  # nothing was removed, the folder kept included
        inner, outer = (recordings, models) if models in recordings.parents else (models, recordings)
        assert f"Nothing was removed: {inner} is inside {outer}, so it would be deleted with it." in out, out
        assert (recordings / "settings.toml").exists() and models.is_dir()
