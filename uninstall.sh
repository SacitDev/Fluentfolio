#!/usr/bin/env bash
# Removes Fluentfolio for the current Linux user: the program folder and its menu entry.
# Your recordings, results, models folder and Hugging Face token stay unless you say yes to each.
set -euo pipefail
DATA_HOME=${XDG_DATA_HOME:-$HOME/.local/share}
APP_ROOT=$DATA_HOME/fluentfolio YES=''
while [ $# -gt 0 ]; do
    case $1 in
        --app-root)
            if [ $# -lt 2 ] || [ -z "$2" ]; then echo 'The option --app-root needs a value.' >&2; exit 2; fi
            APP_ROOT=$2; shift 2 ;;
        --yes) YES=1; shift ;;
        *) echo "There is no option $1. The options are in docs/install.md." >&2; exit 2 ;;
    esac
done
APP_ROOT=$(realpath -ms -- "$APP_ROOT")  # a folder given from here stays the same after the move to the home folder

ask() {  # with --yes nobody answers: the program goes, everything that belongs to the learner stays
    local answer=''
    if [ -n "$YES" ]; then [ "$2" = yes ]; return; fi
    read -r -p "$1 Type Y and press Enter (just Enter says no) " answer || return 1
    [[ ${answer,,} =~ ^[[:space:]]*(y|yes)[[:space:]]*$ ]]
}

stop_with() {
    printf '\nProblem: %s\nWhat to do: %s\nCode: %s (docs/troubleshooting.md#%s).\n' "$2" "$3" "$1" "${1,,}"
    exit 1
}

state_value() {  # one text value from state.json; the file is written with one key per line
    sed -n "s/^ *\"$1\": \"\\(.*\\)\",\\{0,1\\}\$/\\1/p" "$APP_ROOT/state.json" | sed 's/\\\(.\)/\1/g'
}

inside() {  # whether folder $1 is folder $2 or lies in it; a folder whose name only starts like it does not
    [[ $(realpath -m -- "$1")/ == "$(realpath -m -- "$2")"/* ]]
}

if ! [ -f "$APP_ROOT/state.json" ]; then stop_with E_NOT_INSTALLED 'Fluentfolio is not installed for this user.' 'Nothing to remove.'; fi
data=${FLUENTFOLIO_HOME:-$(state_value data_root)}
data=$(realpath -ms -- "${data:-$DATA_HOME/fluentfolio-data}")
for lock in "$data/app.lock" "$data/work.lock"; do  # the running app holds its lock files
    if [ -f "$lock" ] && ! flock -n "$lock" true; then
        stop_with E_APP_RUNNING 'Fluentfolio is running.' 'Close the Fluentfolio window (the one that says keep this window open), then run this again.'
    fi
done
size=$(du -sb -- "$APP_ROOT" | awk '{ printf "%.1f", $1 / 1e9 }')  # decimal GB, as the app prints sizes
echo "This removes Fluentfolio from $APP_ROOT ($size GB: the program, its tools and the speech models) and its menu entry."
echo "Your recordings and results in $data are kept unless you say yes below."
if ! ask 'Remove Fluentfolio?' yes; then echo 'Nothing was removed.'; exit 0; fi
models=$(state_value models_dir) drop_data='' drop_models=''
if [ -n "$models" ] && inside "$models" "$APP_ROOT"; then models=''; fi  # there, as at first, they are part of the program
if ask "Also delete your recordings and results in $data?" no; then drop_data=1; fi
if [ -n "$models" ] && ask "Also delete your models folder $models?" no; then drop_models=1; fi
# Removing the app folder removes all it holds, so a data folder that stays must lie outside it.
if [ -z "$drop_data" ] && inside "$data" "$APP_ROOT"; then
    stop_with E_UNINSTALL_FAILED "Nothing was removed: $data is inside the app folder $APP_ROOT, so it would be deleted with the program." 'Move it out of the app folder first, in Fluentfolio under Settings, Advanced tab. If you set FLUENTFOLIO_HOME to it, move it by hand and change the variable instead. Then run this again.'
fi
# Likewise for the folder you keep when you say yes to deleting the other one.
kept='' gone=''
if [ -n "$drop_models" ] && [ -z "$drop_data" ]; then kept=$data gone=$models
elif [ -n "$drop_data" ] && [ -n "$models" ] && [ -z "$drop_models" ]; then kept=$models gone=$data; fi
if [ -n "$gone" ] && inside "$kept" "$gone"; then
    stop_with E_UNINSTALL_FAILED "Nothing was removed: $kept is inside $gone, so it would be deleted with it." 'Run this again and answer no to deleting that folder, or move the one inside it out first, in Fluentfolio under Settings, Advanced tab.'
fi
exe=$APP_ROOT/$(state_value app_dir)/.venv/bin/fluentfolio
if [ -n "$(state_value hf_token_name)" ] && [ -x "$exe" ] && ask 'Also remove the Hugging Face token that Fluentfolio saved?' no; then
    "$exe" token clear --yes > /dev/null
fi
rm -f -- "$DATA_HOME/applications/fluentfolio.desktop"
cd -- "$HOME"
rm -rf -- "$APP_ROOT"
if [ -n "$drop_data" ] && [ -d "$data" ]; then rm -rf -- "$data"; fi
if [ -n "$drop_models" ] && [ -d "$models" ]; then rm -rf -- "$models"; fi
echo 'Fluentfolio has been removed.'
if [ -z "$drop_data" ]; then echo "Your recordings and results are still in $data."; fi
if [ -n "$models" ] && [ -z "$drop_models" ]; then echo "Your models are still in $models."; fi
