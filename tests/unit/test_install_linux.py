"""The Linux installer scripts as a learner's computer runs them: each refusal comes before any download, with
its code, and the menu entry and the stored choices are what the app expects.

Each bash run loads install.sh's functions and replaces only the questions put to the system (the root check,
the processor, the network probes), then runs the real steps. Every run stops before the network: the release
folder is empty unless a test fills it.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from fluentfolio.core.files import write_json
from fluentfolio.jobs.locks import app_lock
from tests.support.installer import ROOT, bash

UBUNTU = 'PRETTY_NAME="Ubuntu 24.04.3 LTS"\nNAME="Ubuntu"\nVERSION_ID="24.04"\nID=ubuntu\nID_LIKE=debian\n'


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, str]:
    """An earlier install's app folder, an empty release folder and an Ubuntu 24.04 system description."""
    write_json(tmp_path / "app" / "state.json", {"version": "0.0.9", "backend": "cpu", "extras": []})
    (tmp_path / "release").mkdir()
    (tmp_path / "os-release").write_text(UBUNTU, encoding="utf-8")
    return {"FF_APP": str(tmp_path / "app"), "FF_SOURCE": str(tmp_path / "release"),
            "FF_OS": str(tmp_path / "os-release")}  # fmt: skip


LOAD = '. "$FF_SCRIPT" --app-root "$FF_APP" --source "$FF_SOURCE" --preflight-only; OS_RELEASE=$FF_OS'
ORDINARY = f"{LOAD}; is_root() {{ false; }}; "  # a normal user's shell on a supported system
OFFLINE = f"{ORDINARY} test_hosts() {{ :; }}; measure_speed() {{ echo 2000000; }}; "  # no network question


def stopped(done: subprocess.CompletedProcess[str], code: str) -> str:
    assert done.returncode == 1, done.stdout + done.stderr
    assert f"Code: {code} (docs/troubleshooting.md#{code.lower()})." in done.stdout, done.stdout + done.stderr
    return done.stdout


def app_files(setup: dict[str, str]) -> list[str]:
    app = Path(setup["FF_APP"])
    return sorted(str(p.relative_to(app)) for p in app.rglob("*"))


def release(setup: dict[str, str], version: str = "9.9.9") -> None:
    project = f'[project]\nname = "fluentfolio"\nversion = "{version}"\n'
    (Path(setup["FF_SOURCE"]) / "pyproject.toml").write_text(project, encoding="utf-8")


def test_an_unsupported_system_or_a_root_shell_is_refused_before_anything_is_written(
    tmp_path: Path, setup: dict[str, str]
) -> None:
    debian = tmp_path / "debian"
    debian.write_text('NAME="Debian GNU/Linux"\nVERSION_ID="12"\nID=debian\n', encoding="utf-8")
    out = stopped(bash(tmp_path, f"{ORDINARY} main", **{**setup, "FF_OS": str(debian)}), "E_NOT_SUPPORTED")
    assert "Ubuntu 22.04 or 24.04 on a 64-bit Intel or AMD processor" in out
    arm = f"{ORDINARY} uname() {{ echo aarch64; }}; main"
    stopped(bash(tmp_path, arm, **setup), "E_NOT_SUPPORTED")
    root = f"{LOAD}; is_root() {{ true; }}; main"
    assert "Run ./install.sh again, without sudo." in stopped(bash(tmp_path, root, **setup), "E_ELEVATED")
    assert app_files(setup) == ["state.json"]


@pytest.mark.parametrize(
    ("version_line", "supported"),
    [('VERSION_ID="22.04"', True), ('VERSION_ID="24.04"', True), ("VERSION_ID=24.04", True),
     ('VERSION_ID="23.10"', False), ('VERSION_ID="20.04"', False), ('VERSION_ID="24.10"', False)],
)  # fmt: skip
def test_only_the_two_supported_ubuntu_versions_pass(
    tmp_path: Path, version_line: str, supported: bool
) -> None:
    described = tmp_path / "os-release"
    described.write_text(f"ID=ubuntu\n{version_line}\n", encoding="utf-8")
    done = bash(tmp_path, f'{ORDINARY} uname() {{ echo x86_64; }}; os_supported && echo yes || echo no',
                FF_OS=str(described), FF_APP="", FF_SOURCE="")  # fmt: skip
    assert done.stdout.strip() == ("yes" if supported else "no"), done.stderr


def test_wsl_is_named_from_the_kernel_release(tmp_path: Path) -> None:
    for kernel, inside in (("5.15.167.4-microsoft-standard-WSL2", True), ("6.8.0-45-generic", False)):
        (tmp_path / "osrelease").write_text(kernel + "\n", encoding="utf-8")
        check = f'{ORDINARY} KERNEL_RELEASE="{tmp_path / "osrelease"}"; in_wsl && echo yes || echo no'
        done = bash(tmp_path, check, FF_APP="", FF_SOURCE="", FF_OS="")
        assert done.stdout.strip() == ("yes" if inside else "no"), done.stderr


def test_a_folder_without_a_release_is_refused_and_logged(tmp_path: Path, setup: dict[str, str]) -> None:
    out = stopped(bash(tmp_path, f"{ORDINARY} main", **setup), "E_SOURCE")
    assert f"No Fluentfolio release was found in {setup['FF_SOURCE']}." in out
    log = Path(setup["FF_APP"]) / "logs" / "install.log"
    assert f"Details: {log}" in out and "Code: E_SOURCE" in log.read_text(encoding="utf-8")


def test_an_error_no_step_expected_ends_with_the_plain_lines_and_its_details_in_the_log(
    tmp_path: Path, setup: dict[str, str]
) -> None:
    broken = tmp_path / "release.tar.gz"
    broken.write_bytes(b"not an archive")
    done = bash(tmp_path, f"{ORDINARY} main", **{**setup, "FF_SOURCE": str(broken)})
    out = stopped(done, "E_INSTALL_FAILED")
    assert "Problem: The install stopped at an unexpected error." in out and "What to do: " in out
    assert "gzip" not in out  # the tool's own words stay off the screen
    assert "gzip" in (Path(setup["FF_APP"]) / "logs" / "install.log").read_text(encoding="utf-8")


def test_the_whole_preflight_runs_and_passes_on_a_supported_system(
    tmp_path: Path, setup: dict[str, str]
) -> None:
    release(setup)
    done = bash(tmp_path, f"{OFFLINE} main", **setup)
    assert (done.returncode, done.stdout.strip().splitlines()[-1]) == (0, "Preflight passed."), done.stderr


@pytest.mark.parametrize(("first", "about"), [(True, "about"), (False, "at most about")])
def test_the_confirmation_names_the_folders_sizes_and_time_and_no_answer_installs_nothing(
    tmp_path: Path, setup: dict[str, str], first: bool, about: str
) -> None:
    release(setup)
    load = OFFLINE.replace(" --preflight-only", "")
    fresh = "state_value() { :; }; " if first else ""  # no earlier install, so uv has downloaded nothing yet
    done = bash(tmp_path, f"{load} {fresh}main", **setup)
    assert done.returncode == 0, done.stdout + done.stderr
    data = os.environ["FLUENTFOLIO_HOME"]
    what = "installed" if first else "updated from 0.0.9"
    for line in (
        f"Fluentfolio 9.9.9 will be {what} for this Linux user. No sudo is needed.",
        f"  Program, tools and models: {setup['FF_APP']}",
        f"  Your recordings and results: {data}",
        (
            f"  Download now: {about} 0.7 GB (uv, Python 3.12 and the packages). "
            "Disk space needed: about 3 GB; free: "
        ),
        (  # 660 MB at 2 MB/s is 5.5 minutes; the rest of the step is not measured beforehand, so it says so
            f"  Time: {about} 6 minutes to download at the speed measured now (2.0 MB/s). "
            "Unpacking and slower servers can double or triple that."
        ),
        "Nothing was installed.",
    ):
        assert line in done.stdout, line
    assert ("at most" in done.stdout) is not first
    assert not (Path(setup["FF_APP"]) / "bin").exists() and not (Path(setup["FF_APP"]) / "app").exists()


@pytest.mark.parametrize(("device", "shown"), [("auto", True), ("gpu", True), ("cpu", False)])
def test_the_confirmation_adds_gpu_supports_download_and_time_unless_the_cpu_was_chosen(
    tmp_path: Path, setup: dict[str, str], device: str, shown: bool
) -> None:
    release(setup)
    load = OFFLINE.replace(" --preflight-only", f" --device {device}").replace("echo 2000000", "echo 3000000")
    done = bash(tmp_path, f"{load} main", **setup)
    assert done.returncode == 0, done.stdout + done.stderr
    time = "  Time: at most about 4 minutes to download at the speed measured now (3.0 MB/s)."  # 3.7 min
    assert time in done.stdout
    gpu = (  # an update: uv reuses what it downloaded before
        "  If an NVIDIA graphics card can be used, GPU support adds at most about 3.7 GB to download"
        " (at most about 21 more minutes) and about 7 GB of disk space."
    )
    assert (gpu in done.stdout) is shown  # 3,700 MB at 3 MB/s is 1,233 s, 20.6 minutes


def test_the_speed_measure_follows_the_release_link_to_the_file(tmp_path: Path) -> None:
    # GitHub answers the uv link with a redirect and no body. The stand-in prints what curl printed in a test
    # install: 0 bytes without -L (so no time line was shown); with it, 2 MB in 2.67 s after the handshakes.
    fake = (
        'curl() { case " $* " in *" -L "*) echo "2000000 3.732087 1.065986" ;; '
        '*) echo "0 0.945733 0.529374" ;; esac; }; '
    )
    done = bash(tmp_path, f"{ORDINARY} {fake} measure_speed", FF_APP="", FF_SOURCE="", FF_OS="")
    assert (done.returncode, done.stdout.strip()) == (0, "750159"), done.stderr  # 2,000,000 B / 2.666101 s


def test_too_little_disk_space_stops_the_install_with_both_numbers(tmp_path: Path) -> None:
    gb = 10**9  # decimal, as the app's pages print sizes
    empty = {"FF_APP": "", "FF_SOURCE": "", "FF_OS": ""}
    out = stopped(bash(tmp_path, f"{ORDINARY} assert_disk 3 {int(2.5 * gb)}", **empty), "E_DISK_PREFLIGHT")
    assert "Not enough free disk space: need 3.0 GB, have 2.5 GB." in out
    stopped(bash(tmp_path, f"{ORDINARY} assert_disk 3 {3 * gb - 1}", **empty), "E_DISK_PREFLIGHT")
    both = f"{ORDINARY} assert_disk 3 {3 * gb}; assert_disk 3 {3 * gb + 1}; echo enough"
    done = bash(tmp_path, both, **empty)
    assert (done.returncode, done.stdout.strip()) == (0, "enough"), done.stderr


def test_a_running_app_stops_the_installer_and_the_uninstaller(tmp_path: Path, setup: dict[str, str]) -> None:
    lock = app_lock(Path(os.environ["FLUENTFOLIO_HOME"])).acquire()
    uninstall = f'bash "{ROOT / "uninstall.sh"}" --app-root "$FF_APP" --yes'
    try:
        stopped(bash(tmp_path, f"{ORDINARY} main", **setup), "E_APP_RUNNING")
        stopped(bash(tmp_path, uninstall, **setup), "E_APP_RUNNING")
        assert (Path(setup["FF_APP"]) / "state.json").is_file()
    finally:
        lock.release()
    stopped(bash(tmp_path, f"{ORDINARY} main", **setup), "E_SOURCE")  # a lock file nobody holds is old


def test_a_blocked_network_is_named_with_the_proxy_in_use(tmp_path: Path) -> None:
    check = f'{ORDINARY} LOG="{tmp_path / "install.log"}"; HOSTS=(https://127.0.0.1:9); test_hosts'
    empty = {"FF_APP": "", "FF_SOURCE": "", "FF_OS": ""}
    out = stopped(bash(tmp_path, check, **empty), "E_NETWORK")  # the tests' dead proxy is set
    assert "https://127.0.0.1:9 could not be reached through the proxy http://127.0.0.1:9." in out
    direct = {name: "" for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")}
    out = stopped(bash(tmp_path, check, **empty, **direct), "E_NETWORK")
    assert "https://127.0.0.1:9 could not be reached." in out
    # A user name and password in the proxy address stay out of the message and the log a learner may send;
    # an @ or a / inside the password counts too. Every proxy variable holds it and nothing bypasses it, so
    # curl itself meets it, as on a learner's computer: curl's own error quotes an address it cannot read.
    for proxy in ("http://learner:s3c@ret@127.0.0.1:9", "http://learner:s3/c@ret@127.0.0.1:9"):
        (tmp_path / "install.log").unlink(missing_ok=True)
        secret = {name: proxy for name in direct} | {"NO_PROXY": ""}
        out = stopped(bash(tmp_path, check, **empty, **secret), "E_NETWORK")
        assert "could not be reached through the proxy http://127.0.0.1:9." in out
        shown = out + (tmp_path / "install.log").read_text(encoding="utf-8")
        assert "learner" not in shown and "s3c" not in shown and "ret@" not in shown


def tool(tmp_path: Path, says: str, exit_code: int, before: str = "") -> subprocess.CompletedProcess[str]:
    """run_tool on a stand-in tool process that prints `says` and ends with `exit_code`."""
    (tmp_path / "app" / "logs").mkdir(parents=True, exist_ok=True)
    run = f'run_tool bash -c \'printf "%s\\n" "$1"; exit {exit_code}\' tool "$FF_SAYS"'
    load = f'{ORDINARY} LOG="{tmp_path / "install.log"}"; {before} {run}'
    return bash(tmp_path, load, FF_APP=str(tmp_path / "app"), FF_SOURCE="", FF_OS="", FF_SAYS=says)


@pytest.mark.parametrize(
    ("says", "code"),
    [
        ("error: failed to write to file: No space left on device (os error 28)", "E_DISK_FULL"),
        ("error: invalid peer certificate: UnknownIssuer", "E_TLS_INTERCEPTED"),
        ("error: dns error: failed to lookup address information", "E_NETWORK"),
        ("error: something nobody has seen before", "E_INSTALL_FAILED"),
    ],
)
def test_a_failed_tool_gets_its_code_from_its_own_words(tmp_path: Path, says: str, code: str) -> None:
    stopped(tool(tmp_path, says, 2), code)
    assert says in (tmp_path / "install.log").read_text(encoding="utf-8")


def test_a_tool_that_works_shows_its_lines_without_the_package_list_and_logs_all(tmp_path: Path) -> None:
    done = tool(tmp_path, " + numpy==2.5.3\nInstalled 1 package", 0)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Installed 1 package" in done.stdout and "numpy" not in done.stdout
    assert "+ numpy==2.5.3" in (tmp_path / "install.log").read_text(encoding="utf-8")
    assert not (tmp_path / "app" / "logs" / "step.log").exists()


@pytest.mark.parametrize(
    ("done", "total", "seconds", "shown"),
    [  # worked by hand: 24 blocks, the share rounded down, MB of 10^6 bytes, the time left rounded up
        (50e6, 200e6, 10, "  ######------------------  25%  50.0 of 200.0 MB  5.0 MB/s, about 30 s left"),
        (10e6, 200e6, 60, "  #-----------------------   5%  10.0 of 200.0 MB  0.2 MB/s, about 19 min left"),
        (50e6, 200e6, 0, "  ######------------------  25%  50.0 of 200.0 MB"),  # under 1 s: no speed yet
        (16e6, 16e6, 3, "  ######################## 100%  16.0 of 16.0 MB"),
        (0, 0, 5, "  ------------------------   0%  0.0 of 0.0 MB"),
    ],
)
def test_the_bar_shows_the_share_the_sizes_the_speed_and_the_time_left(
    tmp_path: Path, done: float, total: float, seconds: float, shown: str
) -> None:
    run = f"{ORDINARY} BLOCK='#' SHADE='-'; bar {done:.0f} {total:.0f} {seconds}"
    out = bash(tmp_path, run, FF_APP=str(tmp_path / "app"), FF_SOURCE="", FF_OS="")
    assert out.stdout == shown, out.stdout + out.stderr


def test_in_a_real_terminal_the_downloads_of_a_tool_become_one_bar_and_its_other_lines_stay(
    tmp_path: Path,
) -> None:
    # uv's own words when its output is captured: each large download with its size, then when it is done.
    says = (
        "Downloading torch (118.3MiB)\nDownloading cpython-3.12.13-linux (download) (20.9MiB)\n"
        "Resolved 2 packages\n Downloaded cpython-3.12.13-linux (download)\n Downloaded torch\n"
        " + numpy==2.5.3\nInstalled 2 packages"
    )
    record = "LIVE=1 BLOCK='#' SHADE='-'; live() { printf 'BAR:%s\\n' \"$1\"; };"
    done = tool(tmp_path, says, 0, record)
    assert done.returncode == 0, done.stdout + done.stderr
    bars = [
        re.sub(r" MB  .*", " MB", line[4:]) for line in done.stdout.splitlines() if line.startswith("BAR:")
    ]
    # 118.3 MiB is 124.0 MB; with 20.9 MiB (21.9 MB) 146.0 MB, of which 21.9 MB is 15 %. The third and the
    # last bar are drawn again under a line of the tool.
    assert [b[27:] for b in bars] == [
        "  0%  0.0 of 124.0 MB", "  0%  0.0 of 146.0 MB", "  0%  0.0 of 146.0 MB", " 15%  21.9 of 146.0 MB",
        "100%  146.0 of 146.0 MB", "100%  146.0 of 146.0 MB",
    ]  # fmt: skip
    assert bars[3][:26] == "  ###" + "-" * 21  # 15 % of 24 blocks
    shown = [line for line in done.stdout.splitlines() if not line.startswith("BAR:")]
    assert "Resolved 2 packages" in "".join(shown) and "  Installed 2 packages" in "".join(shown)
    assert not [line for line in shown if "Download" in line or "numpy" in line]
    assert " Downloaded torch" in (tmp_path / "install.log").read_text(encoding="utf-8")
    plain = tool(tmp_path, says, 0)  # a captured output keeps the plain lines: no bar is drawn
    assert "BAR:" not in plain.stdout and "  Downloading torch (118.3MiB)" in plain.stdout.splitlines()


def test_a_failed_update_removes_its_new_folder_and_keeps_the_old_version(tmp_path: Path) -> None:
    old, new = tmp_path / "app" / "app" / "0.0.9", tmp_path / "app" / "app" / "0.1.0"
    for folder in (old, new):
        (folder / ".venv").mkdir(parents=True)
    failed = tool(tmp_path, "error: dns error: failed to lookup address", 2, f'KEPT=0.0.9 NEW_APP="{new}";')
    out = stopped(failed, "E_UPDATE_FAILED_OLD_KEPT")
    assert "A download failed. Finished downloads are kept. Version 0.0.9 is kept and still works." in out
    assert "What to do: Check the internet connection" in out  # the cause's own action stays
    assert old.is_dir() and not new.exists()


def test_the_stored_choices_are_read_back_with_spaces_and_other_letters(tmp_path: Path) -> None:
    moved = '/srv/My Data Şükrü "x" \\y'
    state = {"version": "0.1.0", "backend": "cu130", "extras": ["nc"], "data_root": moved}
    write_json(tmp_path / "app" / "state.json", state)
    read = f"{ORDINARY} state_value data_root; stored_choice"
    done = bash(tmp_path, read, FF_APP=str(tmp_path / "app"), FF_SOURCE="", FF_OS="")
    assert done.stdout.splitlines() == [moved, "--extra", "cu130", "--extra", "nc"], done.stderr
    write_json(tmp_path / "app" / "state.json", {"version": "0.1.0", "backend": "cpu", "extras": []})
    done = bash(tmp_path, read, FF_APP=str(tmp_path / "app"), FF_SOURCE="", FF_OS="")
    assert done.stdout.splitlines() == ["--extra", "cpu"], done.stderr


def test_the_menu_entry_starts_the_app_in_its_own_window(tmp_path: Path) -> None:
    command = f'{ORDINARY} new_launcher "/opt/my apps/fluentfolio"'
    done = bash(tmp_path, command, FF_APP="", FF_SOURCE="", FF_OS="")
    assert done.returncode == 0, done.stderr
    entry = tmp_path / "share" / "applications" / "fluentfolio.desktop"
    lines = entry.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "[Desktop Entry]" and 'Exec="/opt/my apps/fluentfolio" serve' in lines
    assert "Terminal=true" in lines and "Name=Fluentfolio" in lines
    assert stat.S_IMODE(entry.stat().st_mode) == 0o600  # the script's umask: this user only


def test_uninstall_removes_the_program_and_keeps_the_learners_data(
    tmp_path: Path, setup: dict[str, str]
) -> None:
    uninstall = f'bash "{ROOT / "uninstall.sh"}" --app-root "$FF_APP" --yes'
    stopped(bash(tmp_path, uninstall, **{**setup, "FF_APP": str(tmp_path / "none")}), "E_NOT_INSTALLED")
    app, data = Path(setup["FF_APP"]), Path(os.environ["FLUENTFOLIO_HOME"])
    (app / "app" / "0.0.9" / ".venv").mkdir(parents=True)
    with (app / "app" / "0.0.9" / ".venv" / "big.bin").open("wb") as big:
        big.truncate(160_000_000)  # 0.16 GB in decimal units; 0.149 if divided by 2**30
    (data / "jobs").mkdir(parents=True)
    (data / "settings.toml").write_text("", encoding="utf-8")
    menu = tmp_path / "share" / "applications"
    menu.mkdir(parents=True)
    for name in ("fluentfolio.desktop", "other-program.desktop"):
        (menu / name).write_text("", encoding="utf-8")
    done = bash(tmp_path, uninstall, **setup)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "(0.2 GB: the program, its tools and the speech models)" in done.stdout
    assert not app.exists() and (data / "settings.toml").is_file() and (data / "jobs").is_dir()
    assert sorted(p.name for p in menu.iterdir()) == ["other-program.desktop"]
    assert f"Your recordings and results are still in {data}." in done.stdout


@pytest.mark.parametrize(
    ("recordings", "models", "answers"),
    [("models/recordings", "models", r"y\n\ny\n"), ("recordings", "recordings/models", r"y\ny\n\n")],
)  # yes to the program, then yes to deleting the outer folder only
def test_a_folder_kept_inside_one_to_delete_stops_the_uninstall_before_anything_goes(
    tmp_path: Path, setup: dict[str, str], recordings: str, models: str, answers: str
) -> None:
    app, data, moved = Path(setup["FF_APP"]), tmp_path / recordings, tmp_path / models
    write_json(app / "state.json", {"version": "0.0.9", "app_dir": "app/0.0.9", "models_dir": str(moved)})
    for folder in data, moved:
        folder.mkdir(parents=True, exist_ok=True)
    (data / "settings.toml").write_text("", encoding="utf-8")
    uninstall = f'printf \'{answers}\' | bash "{ROOT / "uninstall.sh"}" --app-root "$FF_APP"'
    out = stopped(bash(tmp_path, uninstall, **setup, FLUENTFOLIO_HOME=str(data)), "E_UNINSTALL_FAILED")
    inner, outer = (data, moved) if moved in data.parents else (moved, data)
    assert f"Nothing was removed: {inner} is inside {outer}, so it would be deleted with it." in out
    assert (app / "state.json").is_file() and (data / "settings.toml").is_file() and moved.is_dir()


@pytest.mark.parametrize(
    ("inside", "answers", "code"),
    [("recordings", r"y\n\n", 1), ("recordings", r"y\ny\n", 0), ("models", r"y\n\n", 0)],
)  # yes to the program, then no or yes to deleting the recordings
def test_a_recordings_folder_in_the_app_folder_is_kept_and_a_models_folder_there_goes_with_it(
    tmp_path: Path, setup: dict[str, str], inside: str, answers: str, code: int
) -> None:
    app = Path(setup["FF_APP"])
    data = app / "recordings" if inside == "recordings" else Path(os.environ["FLUENTFOLIO_HOME"])
    state = {"version": "0.0.9", "app_dir": "app/0.0.9"}
    if inside == "models":  # moved out in Settings, then back to where they were at first
        state["models_dir"] = str(app / "models")
    write_json(app / "state.json", state)
    (data / "settings.toml").parent.mkdir(parents=True, exist_ok=True)
    (data / "settings.toml").write_text("", encoding="utf-8")
    uninstall = f'printf \'{answers}\' | bash "{ROOT / "uninstall.sh"}" --app-root "$FF_APP"'
    done = bash(tmp_path, uninstall, **setup, FLUENTFOLIO_HOME=str(data))
    assert done.returncode == code, done.stdout + done.stderr
    stop = f"Nothing was removed: {data} is inside the app folder {app}, so it would be deleted"
    assert (stop in done.stdout) is (code == 1) and app.exists() is (code == 1), done.stdout
    assert (data / "settings.toml").exists() is (answers == r"y\n\n")  # kept unless the learner said yes
    assert "Your models are still in" not in done.stdout  # a models folder in the app folder goes with it


def test_an_app_folder_named_from_its_parent_folder_is_the_one_removed(tmp_path: Path) -> None:
    # The uninstaller goes to the home folder, tmp_path here, before it deletes,
    # and there the same name can mean another folder.
    app, other = tmp_path / "apps" / "fluentfolio", tmp_path / "fluentfolio"
    write_json(app / "state.json", {"version": "0.0.9", "app_dir": "app/0.0.9"})
    other.mkdir()
    (other / "mine.txt").write_text("", encoding="utf-8")
    done = bash(tmp_path, f'cd "{app.parent}" && bash "{ROOT / "uninstall.sh"}" --app-root fluentfolio --yes')
    assert done.returncode == 0, done.stdout + done.stderr
    assert not app.exists() and (other / "mine.txt").is_file()
    empty = bash(tmp_path, f'bash "{ROOT / "uninstall.sh"}" --app-root ""')
    assert (empty.returncode, empty.stderr.strip()) == (2, "The option --app-root needs a value.")
