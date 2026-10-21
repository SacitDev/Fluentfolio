# Removes Fluentfolio for the current Windows user: the program folder and its Start-menu shortcuts.
# Your recordings, results, models folder and Hugging Face token stay unless you say yes to each.
param(
    [string]$AppRoot = (Join-Path $env:LOCALAPPDATA 'Fluentfolio'),
    [string]$FallbackRoot = 'C:\Fluentfolio',
    [switch]$Yes,
    [switch]$FromShortcut  # the Start-menu shortcut, which runs the copy in the app folder in a window of its own
)
$ErrorActionPreference = 'Stop'

function Ask([string]$Question, [bool]$Scripted) {
    # With -Yes nobody answers: the program goes, everything that belongs to the learner stays.
    if ($Yes) { return $Scripted }
    try { $answer = Read-Host "$Question Type Y and press Enter (just Enter says no)" } catch { return $false }
    return $answer -match '^\s*(y|yes)\s*$'
}

function Close-Window {
    # Windows closes the shortcut's window as soon as this script ends, so it waits for Enter: the learner can
    # read how the uninstall ended.
    if (-not $FromShortcut) { return }
    # An Enter pressed while the files were going would otherwise answer at once. A window that cannot clear the
    # keys typed ahead still asks.
    try { if (-not [Console]::IsInputRedirected) { $Host.UI.RawUI.FlushInputBuffer() } } catch { Write-Verbose 'Keys typed ahead were kept.' }
    Write-Host "`nPress Enter to close this window."
    try { [void](Read-Host) } catch { return }
}

function Stop-With([string]$Code, [string]$What, [string]$Action) {
    Write-Host "`nProblem: $What`nWhat to do: $Action`nCode: $Code (docs/troubleshooting.md#$($Code.ToLower()))."
    Close-Window
    exit 1
}

function Test-Inside([string]$Path, [string]$Folder) {
    # A folder whose name only starts like Folder's is not inside it.
    $full, $root = foreach ($name in $Path, $Folder) { [IO.Path]::GetFullPath($name).TrimEnd('\') + '\' }
    return $full.StartsWith($root, 'OrdinalIgnoreCase')
}

$removing = $null  # the folder being removed, once removing has begun
trap {
    # An error no step expected, such as a file that another program has open.
    if (-not $removing) { Stop-With 'E_UNINSTALL_FAILED' "Nothing was removed: $($_.Exception.Message)" 'Close Fluentfolio and any program that uses the file or folder named above, then run this again.' }
    Stop-With 'E_UNINSTALL_FAILED' "The uninstall stopped while removing files: $($_.Exception.Message)" "Close any program that uses the file or folder named above. Then delete what is left of $removing in File Explorer, and of any other folder you said yes to deleting."
}
# The shortcut starts this window inside the app folder. Windows does not delete a folder that is still a
# program's current folder, also while this window waits for Enter, and Set-Location alone does not change that.
Set-Location -LiteralPath $env:USERPROFILE
[Environment]::CurrentDirectory = $env:USERPROFILE

if ($FromShortcut) { $AppRoot = $PSScriptRoot }  # the app folder, which may be one the learner chose
elseif (-not (Test-Path -LiteralPath (Join-Path $AppRoot 'state.json'))) { $AppRoot = $FallbackRoot }
$file = Join-Path $AppRoot 'state.json'
if (-not (Test-Path -LiteralPath $file)) { Stop-With 'E_NOT_INSTALLED' 'Fluentfolio is not installed for this user.' 'Nothing to remove.' }
try { $state = Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json }
catch { Stop-With 'E_UNINSTALL_FAILED' "Nothing was removed: $file cannot be read. $($_.Exception.Message)" "Close Fluentfolio and run this again. If it stops again, delete $AppRoot in File Explorer." }
$data = if ($env:FLUENTFOLIO_HOME) { $env:FLUENTFOLIO_HOME } elseif ($state.data_root) { $state.data_root } else { Join-Path $env:LOCALAPPDATA 'Fluentfolio-data' }
foreach ($lock in 'app.lock', 'work.lock') {
    # The running app keeps its lock files open, so nobody else can open them alone.
    $path = Join-Path $data $lock
    try { if (Test-Path -LiteralPath $path) { [IO.File]::Open($path, 'Open', 'ReadWrite', 'None').Close() } }
    catch { Stop-With 'E_APP_RUNNING' 'Fluentfolio is running.' 'Close the Fluentfolio window (the one that says keep this window open), then run this again.' }
}
# uv links each package file from its cache into the venv: a file with two names takes its space once, so
# only its cache name is counted.
$cache = Join-Path $AppRoot 'cache\'
$files = Get-ChildItem -LiteralPath $AppRoot -Recurse -Force -File -ErrorAction SilentlyContinue
$once = $files | Where-Object { $_.LinkType -ne 'HardLink' -or $_.FullName.StartsWith($cache, 'OrdinalIgnoreCase') }
$gb = ($once | Measure-Object Length -Sum).Sum / 1e9  # decimal GB, as the app prints sizes
Write-Host ('This removes Fluentfolio from {0} ({1:N1} GB: the program, its tools and the speech models) and its Start-menu shortcuts.' -f $AppRoot, $gb)
Write-Host "Your recordings and results in $data are kept unless you say yes below."
if (-not (Ask 'Remove Fluentfolio?' $true)) { Write-Host 'Nothing was removed.'; Close-Window; exit 0 }
$models = $state.models_dir
if ($models -and (Test-Inside $models $AppRoot)) { $models = $null }  # there, as at first, they are part of the program
$dropData = Ask "Also delete your recordings and results in ${data}?" $false
$dropModels = $models -and (Ask "Also delete your models folder ${models}?" $false)
# Removing the app folder removes all it holds, so a data folder that stays must lie outside it.
if (-not $dropData -and (Test-Inside $data $AppRoot)) {
    Stop-With 'E_UNINSTALL_FAILED' "Nothing was removed: $data is inside the app folder $AppRoot, so it would be deleted with the program." 'Move it out of the app folder first, in Fluentfolio under Settings, Advanced tab. If you set FLUENTFOLIO_HOME to it, move it by hand and change the variable instead. Then run this again.'
}
# Likewise for the folder you keep when you say yes to deleting the other one.
$kept, $gone = if ($dropModels -and -not $dropData) { $data, $models } elseif ($dropData -and $models -and -not $dropModels) { $models, $data }
if ($gone -and (Test-Inside $kept $gone)) {
    Stop-With 'E_UNINSTALL_FAILED' "Nothing was removed: $kept is inside $gone, so it would be deleted with it." 'Run this again and answer no to deleting that folder, or move the one inside it out first, in Fluentfolio under Settings, Advanced tab.'
}
$exe = Join-Path $AppRoot "$($state.app_dir)\.venv\Scripts\fluentfolio.exe"
$tokenKept = [bool]$state.hf_token_name
if ($tokenKept -and (Test-Path -LiteralPath $exe) -and (Ask 'Also remove the Hugging Face token that Fluentfolio saved?' $false)) {
    # The program stays when the token cannot go, so that the learner can try again. Its errors reach the window.
    $env:FLUENTFOLIO_APP_ROOT = $AppRoot  # the record of the folder being removed
    $why = try {
        & $exe token clear --yes | Out-Null
        if ($LASTEXITCODE) { "Its program ended with exit code $LASTEXITCODE; any lines above say why." }
    } catch { $_.Exception.Message }
    if ($why) { Stop-With 'E_UNINSTALL_FAILED' "The uninstall stopped before removing the program: removing the Hugging Face token did not finish. $why" 'Run this again. To remove the program and keep the token, answer no to the token question.' }
    $tokenKept = $false
}
$menu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
$removing = $AppRoot
Remove-Item -LiteralPath (Join-Path $menu 'Fluentfolio.lnk'), (Join-Path $menu 'Uninstall Fluentfolio.lnk') -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $AppRoot -Recurse -Force
if ($dropData -and (Test-Path -LiteralPath $data)) { $removing = $data; Remove-Item -LiteralPath $data -Recurse -Force }
if ($dropModels -and (Test-Path -LiteralPath $models)) { $removing = $models; Remove-Item -LiteralPath $models -Recurse -Force }
Write-Host 'Fluentfolio has been removed.'
if (-not $dropData) { Write-Host "Your recordings and results are still in $data." }
if ($models -and -not $dropModels) { Write-Host "Your models are still in $models." }
if ($tokenKept) { Write-Host 'The Hugging Face token that Fluentfolio saved was not removed.' }
Close-Window
exit 0  # not the exit code of the last program this script ran
