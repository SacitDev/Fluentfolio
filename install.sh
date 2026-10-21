#!/usr/bin/env bash
# Installs, repairs or updates Fluentfolio for the current Linux user. No sudo is needed.
# Run ./install.sh in the extracted release folder. The flags are for repair and testing; docs/install.md lists them.
set -euo pipefail
umask 077  # files and folders for this user only, as the app itself sets

UV_VERSION=0.13.4
UV_SHA256=2b428396f12572d4429cf210ca234dd599a33b3af0958a4d5ed569d63e3f344f
UV_URL="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz"
RELEASES=https://github.com/SacitDev/Fluentfolio/releases
CORE_MB=660 CORE_DISK_GB=3 GPU_MB=3700 GPU_DISK_GB=7  # the core download (uv, Python 3.12, CPU packages) and its disk need; GPU support's download and the disk it adds
HOSTS=(https://github.com https://release-assets.githubusercontent.com https://pypi.org https://files.pythonhosted.org
    https://download.pytorch.org https://conda.anaconda.org https://huggingface.co https://cas-bridge.xethub.hf.co
    https://releases.astral.sh)
OS_RELEASE=/etc/os-release KERNEL_RELEASE=/proc/sys/kernel/osrelease
RETRY='Run ./install.sh again.'
NET='Check the internet connection (and the proxy settings, if you use one), then run ./install.sh again.'
TLS='A secure connection was not trusted. A proxy or security program may inspect secure connections.'
TLS_ACTION='Ask whoever manages this network, or try another network.'
PROXY=${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}
# The proxy as messages and the log show it: a user name and password in its address are left out, up to the
# last @, as a password may hold a / or an @.
PROXY_SHOWN=$(printf '%s' "$PROXY" | sed 's#^\([A-Za-z][A-Za-z0-9+.-]*://\)\{0,1\}.*@#\1#')
DATA_HOME=${XDG_DATA_HOME:-$HOME/.local/share}
LIVE='' BLOCK='#' SHADE='-'  # a real terminal gets colour and bars drawn in place; a captured output gets plain lines
if [ -t 1 ]; then LIVE=1; fi
if [ "$(locale charmap 2> /dev/null)" = UTF-8 ]; then BLOCK=$(printf '█') SHADE=$(printf '░'); fi

SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) DEVICE=auto APP_ROOT=$DATA_HOME/fluentfolio
PREFLIGHT_ONLY='' YES='' UPDATE='' UNINSTALL='' LOG='' KEPT='' NEW_APP='' STOPPED=''
usage() { echo "$1 The options are in docs/install.md." >&2; exit 2; }
while [ $# -gt 0 ]; do
    case $1 in
        --source | --device | --app-root)
            [ $# -ge 2 ] || usage "The option $1 needs a value."
            case $1 in --source) SOURCE=$2 ;; --device) DEVICE=$2 ;; *) APP_ROOT=$2 ;; esac
            shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --yes) YES=1; shift ;;
        --update) UPDATE=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        *) usage "There is no option $1." ;;
    esac
done
case $DEVICE in auto | gpu | cpu) ;; *) usage "The device is auto, gpu or cpu." ;; esac

say() {  # a line for the terminal, in the colour given (an ANSI code) when it is a real one, and for the log
    if [ -n "${2:-}" ] && [ -n "$LIVE" ]; then printf '\033[%sm%s\033[0m\n' "$2" "$1"; else printf '%s\n' "$1"; fi
    if [ -n "$LOG" ]; then printf '%s\n' "$1" >> "$LOG"; fi
}

step() { say "" && say "Step $1 of 5: $2" '1;36'; }

bar() {  # 24 blocks, the share, the sizes, then the speed and the time left: done bytes, total bytes, seconds
    awk -v d="$1" -v t="$2" -v s="$3" -v on="$BLOCK" -v off="$SHADE" 'function up(x) { return x == int(x) ? x : int(x) + 1 }
    BEGIN {
        f = t > 0 ? d / t : 0; if (f > 1) f = 1; n = int(f * 24); b = ""
        for (i = 0; i < 24; i++) b = b (i < n ? on : off)
        printf "  %s %3d%%  %.1f of %.1f MB", b, int(f * 100), d / 1e6, t / 1e6
        if (d <= 0 || s < 1 || f >= 1) exit
        r = d / s; l = (t - d) / r
        printf "  %.1f MB/s, about %s left", r / 1e6, (l < 60 ? up(l) " s" : up(l / 60) " min")
    }'
}

live() {  # one line drawn again in place, cut to the terminal so it never wraps
    local width
    width=$(tput cols 2> /dev/null || echo 80)
    printf '\r\033[K%s' "${1:0:$((width - 1))}"
}

stop_with() {  # the code, what failed and the one action
    local code=$1 what=$2 where=''
    if [ -n "$KEPT" ]; then  # an update that fails before the switch removes its new folder; the old version still runs
        rm -rf -- "${NEW_APP:?}"
        code=E_UPDATE_FAILED_OLD_KEPT what="$what Version $KEPT is kept and still works."
    fi
    if [ -n "$LOG" ]; then where=" Details: $LOG"; fi
    say "" && say "Problem: $what" '33' && say "What to do: $3" && say "Code: $code (docs/troubleshooting.md#${code,,}).$where"
    STOPPED=1
    exit 1
}

confirm() {
    local answer=''
    if [ -n "$YES" ]; then return 0; fi
    read -r -p "$1 Type Y and press Enter (just Enter stops) " answer || return 1
    [[ ${answer,,} =~ ^[[:space:]]*(y|yes)[[:space:]]*$ ]]
}

# The questions put to the system itself, each in one place, so the tests can answer them.
is_root() { [ "$(id -u)" -eq 0 ]; }
os_supported() {
    [ "$(uname -m)" = x86_64 ] && grep -qx 'ID=ubuntu' "$OS_RELEASE" && grep -qxE 'VERSION_ID="?(22|24)\.04"?' "$OS_RELEASE"
}
in_wsl() { grep -qi microsoft "$KERNEL_RELEASE" 2>/dev/null; }
free_bytes() { df -Pk -- "$1" | awk 'NR == 2 { printf "%.0f", $4 * 1024 }'; }

state_value() {  # one text value from state.json; the file is written with one key per line
    if ! [ -f "$APP_ROOT/state.json" ]; then return 0; fi
    sed -n "s/^ *\"$1\": \"\\(.*\\)\",\\{0,1\\}\$/\\1/p" "$APP_ROOT/state.json" 2>/dev/null | sed 's/\\\(.\)/\1/g'
}

stored_choice() {  # the backend the device step chose and the extra the licence step added, read back
    local backend=cpu
    if [ "$(state_value backend)" = cu130 ]; then backend=cu130; fi
    printf '%s\n' --extra "$backend"
    if grep -qE '^ *"nc",?$' "$APP_ROOT/state.json" 2> /dev/null; then printf '%s\n' --extra nc; fi
}

assert_disk() {  # needed GB, free bytes; decimal units, as the app's pages print sizes
    if awk -v need="$1" -v free="$2" 'BEGIN { exit !(free / 1e9 < need) }'; then
        stop_with E_DISK_PREFLIGHT "$(awk -v need="$1" -v free="$2" \
            'BEGIN { printf "Not enough free disk space: need %.1f GB, have %.1f GB.", need, free / 1e9 }')" "Free some space. $RETRY"
    fi
}

test_hosts() {
    local uri status
    for uri in "${HOSTS[@]}"; do
        status=0
        # Only curl's exit code goes to the log: its own words quote a proxy address it cannot read, password too.
        curl -sS -o /dev/null --head --max-time 20 "$uri" 2> /dev/null || status=$?
        if [ "$status" -ne 0 ]; then printf 'curl exit %s for %s\n' "$status" "$uri" >> "$LOG"; fi
        case $status in
            0) ;;  # any answer counts, an error page too: the host is reachable
            35 | 51 | 60 | 77 | 83) stop_with E_TLS_INTERCEPTED "$TLS ($uri)" "$TLS_ACTION" ;;
            *) stop_with E_NETWORK "$uri could not be reached${PROXY:+ through the proxy $PROXY_SHOWN}." "$NET" ;;
        esac
    done
}

measure_speed() {  # reads the first 2 MB of the uv download: the time estimate rests on this connection's speed
    local got  # -L: GitHub answers the release link with a redirect to the file, and the redirect has no body
    got=$(curl -sS -L --proto-redir =https -o /dev/null -r 0-1999999 --max-time 60 -w '%{size_download} %{time_total} %{time_appconnect}' "$UV_URL" 2> /dev/null) || got=''
    awk -v got="$got" 'BEGIN { split(got, v, " "); t = v[2] - v[3]; printf "%.0f", (got == "" ? 0 : v[1] / (t > 0.001 ? t : 0.001)) }'
}

run_tool() {  # shows the tool's lines (not its package list) and its downloads as one bar; the log keeps all
    local step=$APP_ROOT/logs/step.log status=0
    local starting='^[[:space:]]*Downloading (.+) \(([0-9.]+) ?([KMG])iB\)[[:space:]]*$' fetched='^[[:space:]]*Downloaded (.*[^[:space:]])[[:space:]]*$'
    "$@" 2>&1 | {
        local -A sizes=()
        local line key got=0 want=0 drawn='' start=$SECONDS
        while IFS= read -r line; do
            printf '  %s\n' "$line" >> "$LOG"
            printf '%s\n' "$line" >> "$step"
            if [ -n "$LIVE" ] && [[ $line =~ $starting ]]; then  # uv names each large download with its size, then says when it is done
                key=${BASH_REMATCH[1]}
                sizes[$key]=$(awk -v v="${BASH_REMATCH[2]}" -v u="${BASH_REMATCH[3]}" 'BEGIN { printf "%.0f", v * (u == "G" ? 2^30 : u == "M" ? 2^20 : 2^10) }')
                want=$((want + ${sizes[$key]}))
                drawn=$(bar "$got" "$want" $((SECONDS - start))) && live "$drawn"
            elif [ -n "$LIVE" ] && [[ $line =~ $fetched ]] && [ -n "${sizes[${BASH_REMATCH[1]}]:-}" ]; then
                got=$((got + ${sizes[${BASH_REMATCH[1]}]}))
                drawn=$(bar "$got" "$want" $((SECONDS - start))) && live "$drawn"
            elif ! [[ $line =~ ^[[:space:]]*[-+~][[:space:]] ]]; then
                if [ -n "$drawn" ]; then printf '\r\033[K  %s\n' "$line" && live "$drawn"; else printf '  %s\n' "$line"; fi
            fi
        done
        if [ -n "$drawn" ]; then printf '\n'; fi
    } || status=$?
    if [ "$status" -eq 0 ]; then rm -f -- "$step"; return 0; fi
    if grep -qiE 'os error 28|no space left' "$step"; then stop_with E_DISK_FULL 'The disk became full during the install.' "Free some space. $RETRY"; fi
    if grep -qiE 'certificate|UnknownIssuer' "$step"; then stop_with E_TLS_INTERCEPTED "$TLS" "$TLS_ACTION"; fi
    if grep -qiE 'proxy|dns|connect|timed out|network' "$step"; then stop_with E_NETWORK 'A download failed. Finished downloads are kept.' "$NET"; fi
    stop_with E_INSTALL_FAILED "The step '$(basename -- "$1") $2' failed." "$RETRY If it fails again, send the details file with your question."
}

new_launcher() {  # the menu entry; Terminal=true gives the app its own window, and closing it stops the app
    local menu=$DATA_HOME/applications
    mkdir -p -- "$menu"
    printf '%s\n' '[Desktop Entry]' 'Type=Application' 'Name=Fluentfolio' 'Categories=Education;' \
        'Comment=Start Fluentfolio. Keep its window open while you use it.' "Exec=\"$1\" serve" 'Terminal=true' \
        > "$menu/fluentfolio.desktop"
}

on_exit() {  # an error no step expected still ends with the plain three lines; the log keeps its error output
    if [ "$1" -ne 0 ] && [ -z "$STOPPED" ]; then
        stop_with E_INSTALL_FAILED 'The install stopped at an unexpected error.' "$RETRY If it fails again, send the details file with your question."
    fi
}

main() {
    trap 'on_exit $?' EXIT
    if ! os_supported; then
        stop_with E_NOT_SUPPORTED 'This computer is not supported in version 1. Fluentfolio needs Ubuntu 22.04 or 24.04 on a 64-bit Intel or AMD processor.' 'Use a supported computer.'
    fi
    if is_root; then
        stop_with E_ELEVATED 'The installer was started as root. Fluentfolio installs for your user only.' 'Run ./install.sh again, without sudo.'
    fi
    if [ -n "$UNINSTALL" ]; then exec bash "$(dirname -- "${BASH_SOURCE[0]}")/uninstall.sh" --app-root "$APP_ROOT" ${YES:+--yes}; fi
    if [ -n "$UPDATE" ]; then
        say 'To update Fluentfolio:' && say "  1. Download the newest release from $RELEASES (the Source code tar.gz)."
        say '  2. Extract it to a folder of your choice.' && say '  3. Run ./install.sh in that folder. Your recordings, results and settings are kept.'
        if [ -z "$YES" ] && command -v xdg-open > /dev/null; then xdg-open "$RELEASES" > /dev/null 2>&1 || true; fi
        exit 0
    fi
    if in_wsl; then say 'Note: this Linux runs inside WSL. Its GPU driver comes from Windows; this installer never installs one.'; fi
    for tool in curl tar sha256sum flock; do
        command -v "$tool" > /dev/null || stop_with E_INSTALL_FAILED "The program $tool is missing." "Install it (on Ubuntu: sudo apt install $tool), then run ./install.sh again."
    done
    mkdir -p -- "$APP_ROOT/logs"
    LOG=$APP_ROOT/logs/install.log
    exec 3>&2 2> >(tee -a "$LOG" >&2)  # every error line also goes to the details file, until the app starts
    say "Fluentfolio installer, $(date -u '+%Y-%m-%d %H:%M:%SZ')"
    step 1 'Checking this computer and the connection'
    local state data version what about speed free
    state=$(state_value version)
    data=${FLUENTFOLIO_HOME:-$(state_value data_root)}
    data=${data:-$DATA_HOME/fluentfolio-data}
    for lock in "$data/app.lock" "$data/work.lock"; do  # the running app holds its lock files
        if [ -f "$lock" ] && ! flock -n "$lock" true; then
            stop_with E_APP_RUNNING 'Fluentfolio is running.' 'Close the Fluentfolio window (the one that says keep this window open), then run this again.'
        fi
    done
    if [[ $SOURCE =~ \.(tar\.gz|tgz|zip)$ ]]; then
        local unpacked=$APP_ROOT/cache/source
        rm -rf -- "$unpacked" && mkdir -p -- "$unpacked"
        if [[ $SOURCE == *.zip ]]; then
            command -v unzip > /dev/null || stop_with E_SOURCE "$SOURCE is a zip file, and this computer has no unzip program." 'Use the release file that ends in .tar.gz, or extract the zip first.'
            unzip -q -- "$SOURCE" -d "$unpacked" >> "$LOG" 2>&1
        else
            tar -xzf "$SOURCE" -C "$unpacked" >> "$LOG" 2>&1
        fi
        SOURCE=$(find "$unpacked" -maxdepth 2 -name pyproject.toml -printf '%h\n' | head -n 1)
    fi
    version=$(sed -n 's/^version = "\([0-9A-Za-z.+-]*\)"$/\1/p' "$SOURCE/pyproject.toml" 2> /dev/null | head -n 1 || true)
    if [ -z "$version" ]; then stop_with E_SOURCE "No Fluentfolio release was found in $SOURCE." 'Extract the whole release archive, then run ./install.sh inside it.'; fi
    free=$(free_bytes "$APP_ROOT")
    assert_disk "$CORE_DISK_GB" "$free"
    test_hosts
    speed=$(measure_speed)
    if [ -n "$PREFLIGHT_ONLY" ]; then say 'Preflight passed.'; exit 0; fi
    about="at most about"  # an update or a repair reuses what uv downloaded before
    if [ -z "$state" ]; then what=installed; about=about; elif [ "$state" != "$version" ]; then what="updated from $state"; else what=repaired; fi
    say "" && say "Fluentfolio $version will be $what for this Linux user. No sudo is needed."
    say "  Program, tools and models: $APP_ROOT" && say "  Your recordings and results: $data"
    say "$(awk -v mb="$CORE_MB" -v gb="$CORE_DISK_GB" -v free="$free" -v a="$about" 'BEGIN { printf "  Download now: %s %.1f GB (uv, Python 3.12 and the packages). Disk space needed: about %d GB; free: %.1f GB.", a, mb / 1000, gb, free / 1e9 }')"
    if [ "$speed" -gt 0 ]; then
        say "$(awk -v mb="$CORE_MB" -v s="$speed" -v a="$about" 'BEGIN { m = mb * 1e6 / s / 60; printf "  Time: %s %.0f minutes to download at the speed measured now (%.1f MB/s). Unpacking and slower servers can double or triple that.", a, (m > 1 ? m : 1), s / 1e6 }')"
    fi
    if [ "$DEVICE" != cpu ]; then
        say "$(awk -v mb="$GPU_MB" -v gb="$GPU_DISK_GB" -v s="$speed" -v a="$about" 'BEGIN { printf "  If an NVIDIA graphics card can be used, GPU support adds %s %.1f GB to download", a, mb / 1000; if (s > 0) { m = mb * 1e6 / s / 60; printf " (%s %.0f more minutes)", a, (m > 1 ? m : 1) }; printf " and about %d GB of disk space.", gb }')"
    fi
    say '  Then a page in your browser lists the speech models (several GB more). Untick what you do not want.'
    if ! confirm 'Continue?'; then say 'Nothing was installed.'; exit 0; fi
    step 2 "Getting uv $UV_VERSION, the tool that installs Python and the packages"
    local bin=$APP_ROOT/bin uv=$APP_ROOT/bin/uv
    mkdir -p -- "$bin"
    if [ -x "$uv" ] && [[ $("$uv" --version 2> /dev/null) == "uv $UV_VERSION "* ]]; then
        say '  It is here already.'
    else
        local archive=$bin/uv.download.tar.gz good=''
        if [ -n "$LIVE" ]; then  # curl draws its own bar in a real terminal
            curl -fL --progress-bar --max-time 900 -o "$archive" "$UV_URL" 2>&3 || stop_with E_NETWORK 'The download of uv failed.' "$NET"
        else
            curl -fsSL --max-time 900 -o "$archive" "$UV_URL" 2>> "$LOG" || stop_with E_NETWORK 'The download of uv failed.' "$NET"
        fi
        if [ "$(sha256sum -- "$archive" | cut -d ' ' -f 1)" = "$UV_SHA256" ]; then good=1; fi
        if [ -n "$good" ]; then tar -xzf "$archive" -C "$bin" --strip-components=1; fi
        rm -f -- "$archive"
        if [ -z "$good" ]; then stop_with E_HASH_MISMATCH 'The downloaded uv file is damaged or was changed on the way.' "$RETRY"; fi
    fi
    local app=$APP_ROOT/app/$version
    if [[ $what == updated* ]]; then KEPT=$state NEW_APP=$app; fi
    step 3 "Copying the program to $app"
    mkdir -p -- "$app"
    if [ "$(cd -- "$SOURCE" && pwd -P)" != "$(cd -- "$app" && pwd -P)" ]; then  # a run from the installed copy keeps it
        find "$app" -mindepth 1 -maxdepth 1 ! -name .venv -exec rm -rf -- {} +
        tar -C "$SOURCE" --exclude=./.git --exclude=./.venv --exclude=__pycache__ -cf - . | tar -C "$app" -xf -
    fi
    export UV_CACHE_DIR=$APP_ROOT/cache/uv UV_PYTHON_INSTALL_DIR=$APP_ROOT/python UV_PYTHON_PREFERENCE=only-managed
    export UV_PYTHON_INSTALL_REGISTRY=0 UV_PYTHON_INSTALL_BIN=0 FLUENTFOLIO_APP_ROOT=$APP_ROOT
    unset UV_PROJECT_ENVIRONMENT  # no UV_NO_CONFIG: it drops the project's own uv settings, so the lock never matches
    local chosen
    mapfile -t chosen < <(stored_choice)
    step 4 "Installing Python 3.12 and the packages (${chosen[1]})"
    run_tool "$uv" sync --locked --no-dev --project "$app" "${chosen[@]}" --no-build --no-install-project  # wheels only: no build script runs
    run_tool "$uv" sync --locked --no-dev --project "$app" "${chosen[@]}"  # then Fluentfolio itself, which uv builds
    local python=$app/.venv/bin/python exe=$app/.venv/bin/fluentfolio
    if ! [ -x "$python" ] || ! [ -x "$exe" ]; then
        stop_with E_FILE_BLOCKED 'A file of the install is missing right after it was made. A security program may have removed it.' "Check your security program's list of blocked files, then run ./install.sh again."
    fi
    if [ -n "$KEPT" ]; then step 5 'Starting the new version once, and finishing'; else
        step 5 'Choosing between the graphics card (GPU) and the processor (CPU), and finishing'; fi
    if [ -n "$KEPT" ]; then  # keeps the backend chosen before; the self-test checks the GPU again
        [[ $("$python" -I -m fluentfolio --version 2>&1) == *"$version"* ]] || stop_with E_INSTALL_FAILED "Version $version did not start." "$RETRY"
    else
        "$python" -I -m fluentfolio.setup device --choice "$DEVICE" || stop_with E_INSTALL_FAILED 'The device step failed.' "$RETRY"
    fi
    "$python" -I -m fluentfolio.setup finish || stop_with E_INSTALL_FAILED 'The last install step failed.' "$RETRY"  # the switch
    KEPT=''
    cp -- "$app/install.sh" "$app/uninstall.sh" "$APP_ROOT/"
    new_launcher "$exe"
    for old in "$APP_ROOT"/app/*/; do
        if [ "$(basename -- "$old")" != "$version" ]; then rm -rf -- "$old"; fi
    done
    say "" && say "Ready: Fluentfolio $version is $what. Start it from the applications menu: Fluentfolio." '32'
    say "Or run: \"$exe\" serve"
    if [ -z "$YES" ]; then say 'Starting it now. Keep this window open while you use Fluentfolio.'; exec "$exe" serve 2>&3 3>&-; fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main; fi
