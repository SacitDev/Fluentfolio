# Installs, repairs or updates Fluentfolio for the current Windows user. No admin rights are needed.
# Double-click Install.cmd to run it. The flags are for repair and testing; docs/install.md lists them.
param(
    [string]$Source = $PSScriptRoot,
    [ValidateSet('auto', 'gpu', 'cpu')][string]$Device = 'auto',
    [string]$AppRoot = (Join-Path $env:LOCALAPPDATA 'Fluentfolio'),
    [string]$FallbackRoot = 'C:\Fluentfolio',
    [switch]$PreflightOnly,
    [switch]$Yes,
    [switch]$Update,
    [switch]$Uninstall,
    [switch]$AllowElevatedTest
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$UvVersion = '0.13.4'
$UvSha256 = 'ccec7508a07d9933d7cb1261880633efcf1f30d6e6eddab21823f1dae6f23225'
$UvUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
$Releases = 'https://github.com/SacitDev/Fluentfolio/releases'
$GB, $MB = 1e9, 1e6  # decimal units, as the app's pages print sizes
$CoreMB, $CoreDiskGB, $GpuMB, $GpuDiskGB = 500, 3, 2700, 4  # the core download (uv, Python 3.12, CPU packages) and its disk need; GPU support's download and the disk it adds
$VcRuntime = Join-Path $env:SystemRoot 'System32\vcruntime140_1.dll'  # Python and the speech tools need it
$Hosts = 'https://github.com', 'https://release-assets.githubusercontent.com', 'https://pypi.org', 'https://files.pythonhosted.org',
    'https://download.pytorch.org', 'https://conda.anaconda.org', 'https://huggingface.co', 'https://cas-bridge.xethub.hf.co', 'https://releases.astral.sh'
$Retry = 'Run Install.cmd again.'
$Net = 'Check the internet connection (and the proxy settings, if you use one), then run Install.cmd again.'
$Shield = "Check your security program's list of blocked files, then run Install.cmd again."
$Tls, $TlsAction = 'A secure connection was not trusted. A proxy or security program may inspect secure connections.', 'Ask whoever manages this network, or try another network.'
$Known = @(  # known failure texts in a tool's output, with the code, the plain cause and the one action
    @('os error 4551', 'E_SAC_ENFORCED', 'Smart App Control blocked a program that Fluentfolio needs.', 'Read docs/troubleshooting.md#e_sac_enforced.'),
    @('os error 22[56]|virus', 'E_FILE_BLOCKED', 'A security program blocked or removed a file of the install.', $Shield),
    @('os error 112|not enough space', 'E_DISK_FULL', 'The disk became full during the install.', "Free some space. $Retry"),
    @('(?i)certificate|UnknownIssuer', 'E_TLS_INTERCEPTED', $Tls, $TlsAction),
    @('(?i)proxy|dns|connect|timed out|network', 'E_NETWORK', 'A download failed. Finished downloads are kept.', $Net)
)
$script:Proxy = @($env:HTTPS_PROXY, $env:https_proxy, $env:HTTP_PROXY, $env:http_proxy) | Where-Object { $_ } | Select-Object -First 1
$script:ProxyShown = $script:Proxy -replace '^([a-z][a-z0-9+.-]*://)?.*@', '$1'  # as messages and the log show it: no user name or password (up to the last @)
$Live = -not [Console]::IsOutputRedirected  # a real console gets bars drawn in place; a captured output gets plain lines

function Say([string]$Text, [string]$Colour) {
    $look = if ($Colour) { @{ ForegroundColor = $Colour } } else { @{} }
    Write-Host $Text @look
    if ($script:LogFile) { Add-Content -LiteralPath $script:LogFile -Value $Text -Encoding UTF8 }
}

function Step([int]$Number, [string]$Text) { Say "`nStep $Number of 5: $Text" 'Cyan' }

function Get-Bar([double]$Done, [double]$Total, [double]$Seconds) {  # 24 blocks, the share, the sizes, then the speed and the time left
    $share = if ($Total -gt 0) { [Math]::Min(1.0, $Done / $Total) } else { 0 }
    $full = [int][Math]::Floor($share * 24)
    $text = '  {0}{1} {2,3}%  {3:N1} of {4:N1} MB' -f ([string][char]0x2588 * $full), ([string][char]0x2591 * (24 - $full)), [int][Math]::Floor($share * 100), ($Done / $MB), ($Total / $MB)
    if ($Done -le 0 -or $Seconds -lt 1 -or $share -ge 1) { return $text }
    $rate = $Done / $Seconds; $left = ($Total - $Done) / $rate
    $when = if ($left -lt 60) { '{0:N0} s' -f [Math]::Ceiling($left) } else { '{0:N0} min' -f [Math]::Ceiling($left / 60) }
    return $text + ('  {0:N1} MB/s, about {1} left' -f ($rate / $MB), $when)
}

function Show-Live([string]$Text) {  # one line drawn again in place, cut to the window so it never wraps
    if (-not $Live) { return }
    $width = [Math]::Max(20, [Console]::WindowWidth - 1)
    Write-Host -NoNewline ("`r" + $Text.PadRight($width).Substring(0, $width))
    $script:Bar = $Text
}

function Write-Above([string]$Text) {  # a line above the bar, which is drawn again under it
    if (-not $script:Bar) { Write-Host $Text; return }
    $bar = $script:Bar; Show-Live ''; Write-Host "`r$Text"; Show-Live $bar
}

function Stop-Live { if ($script:Bar) { Write-Host ''; $script:Bar = '' } }  # the last bar stays on screen

function Stop-With([string]$Code, [string]$What, [string]$Action) {
    if ($script:Kept) {  # an update that fails before the switch removes its new folder; the old version still runs
        Remove-Item -LiteralPath $script:NewApp -Recurse -Force -ErrorAction SilentlyContinue
        $Code, $What = 'E_UPDATE_FAILED_OLD_KEPT', "$What Version $script:Kept is kept and still works."
    }
    $where = if ($script:LogFile) { " Details: $script:LogFile" } else { '' }
    Stop-Live
    Say "`nProblem: $What`nWhat to do: $Action`nCode: $Code (docs/troubleshooting.md#$($Code.ToLower())).$where" 'Yellow'
    exit 1
}

function Confirm-Step([string]$Question) { if ($Yes) { return $true }; try { return (Read-Host "$Question Type Y and press Enter (just Enter stops)") -match '^\s*(y|yes)\s*$' } catch { return $false } }

function Get-PathProblems([string]$Path) {
    # The app's own folder rule: MFA, Kaldi and espeak fail in such folders.
    $found = @()
    if ($Path -match '[^\x00-\x7F]') { $found += 'letters outside A to Z' }
    if ($Path.Contains(' ')) { $found += 'a space' }
    if ($Path.Length -ge 60) { $found += "$($Path.Length) characters (at most 59)" }
    $synced = @($env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial) | Where-Object { $_ -and $Path.StartsWith($_, 'OrdinalIgnoreCase') }
    if ($Path -match '(?i)onedrive' -or $synced) { $found += 'inside OneDrive' }
    try {
        $drive = New-Object IO.DriveInfo([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path)))
        if ($drive.DriveType -ne 'Fixed' -or $drive.DriveFormat -ne 'NTFS') { $found += 'not a local NTFS disk' }
    } catch { $found += 'not a local NTFS disk' }
    return , $found
}

function Assert-Disk([double]$NeedGB, [double]$FreeBytes) {
    if ($FreeBytes / $GB -lt $NeedGB) { Stop-With 'E_DISK_PREFLIGHT' ('Not enough free disk space: need {0:N1} GB, have {1:N1} GB.' -f $NeedGB, ($FreeBytes / $GB)) "Free some space. $Retry" }
}

# The two questions put to Windows itself, each in one place.
function Test-Elevated { return ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator) }
function Get-SacState { return (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' -ErrorAction SilentlyContinue).VerifiedAndReputablePolicyState }

function Set-OwnerOnly([string]$Folder) {
    # A folder made at C:\ inherits Modify for every account; then only this user, SYSTEM and Administrators (SIDs: names are translated).
    $allowed = @([Security.Principal.WindowsIdentity]::GetCurrent().User.Value, 'S-1-5-18', 'S-1-5-32-544')
    & (Join-Path $env:SystemRoot 'System32\icacls.exe') $Folder /inheritance:r /grant:r "*$($allowed[0]):(OI)(CI)F" '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' | Out-Null
    if ($LASTEXITCODE -ne 0) { return $false }
    $acl = Get-Acl -LiteralPath $Folder  # read back: a folder another account made earlier keeps that owner and its own entries
    $others = @($acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]) | Where-Object { $allowed -notcontains $_.IdentityReference.Value })
    return ($allowed -contains $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value) -and $others.Count -eq 0
}

function Test-Hosts {
    foreach ($uri in $Hosts) {
        $request = @{ Uri = $uri; UseBasicParsing = $true; TimeoutSec = 20; Method = 'Head' }
        if ($script:Proxy) { $request.Proxy = $script:Proxy }
        $how = if ($script:Proxy) { " through the proxy $script:ProxyShown" } else { '' }
        try { Invoke-WebRequest @request | Out-Null } catch [Net.WebException] {
            if ($_.Exception.Status -eq 'TrustFailure') { Stop-With 'E_TLS_INTERCEPTED' "$Tls ($uri)" $TlsAction }
            if ($null -eq $_.Exception.Response) { Stop-With 'E_NETWORK' "$uri could not be reached$how." $Net }
        } catch [Management.Automation.ParameterBindingException] {  # a proxy address that is no URL: its own error quotes it, password too
            Stop-With 'E_NETWORK' "$uri could not be reached$how." $Net
        }
    }
}

function Measure-Speed {  # reads the first 2 MB of the uv download: the time estimate rests on this connection's speed
    try {
        $request = [Net.HttpWebRequest]::Create($UvUrl)
        $request.AddRange(0, 1999999)
        if ($script:Proxy) { $request.Proxy = New-Object Net.WebProxy($script:Proxy) }
        $response = $request.GetResponse()  # the clock starts after the connection is made
        $watch, $stream, $buffer, $total = [Diagnostics.Stopwatch]::StartNew(), $response.GetResponseStream(), (New-Object byte[] 65536), 0
        while (($n = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) { $total += $n }
        $response.Close()
        return $total / [Math]::Max($watch.Elapsed.TotalSeconds, 0.001)
    } catch { return 0 }
}

function Save-File([string]$Uri, [string]$OutFile) {  # a download drawn as a bar
    $request = [Net.HttpWebRequest]::Create($Uri)
    $request.Timeout = 20000; $request.ReadWriteTimeout = 20000
    if ($script:Proxy) { $request.Proxy = New-Object Net.WebProxy($script:Proxy) }
    $response = $request.GetResponse()
    $out, $stream, $buffer, $done = [IO.File]::Create($OutFile), $response.GetResponseStream(), (New-Object byte[] 65536), 0
    $watch, $drawn = [Diagnostics.Stopwatch]::StartNew(), -1000
    try {
        while (($n = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $out.Write($buffer, 0, $n); $done += $n
            if ($watch.ElapsedMilliseconds - $drawn -ge 200) { Show-Live (Get-Bar $done $response.ContentLength $watch.Elapsed.TotalSeconds); $drawn = $watch.ElapsedMilliseconds }
        }
        Show-Live (Get-Bar $done $done 0)
    } finally { $out.Close(); $response.Close(); Stop-Live }
}

function Invoke-Tool([string]$Exe, [string[]]$Arguments) {  # shows the tool's lines (not its package list) and its downloads as one bar; the log keeps all
    $ErrorActionPreference = 'Continue'  # the tool writes its progress to stderr; only its exit code counts
    $sizes, $want, $got, $watch = @{}, 0, 0, [Diagnostics.Stopwatch]::StartNew()
    $lines = & $Exe @Arguments 2>&1 | ForEach-Object {
        $text = "$_"
        Add-Content -LiteralPath $script:LogFile -Value "  $text" -Encoding UTF8
        if ($Live -and $text -match '^\s*Downloading (.+) \(([\d.]+) ?([KMG])iB\)\s*$') {  # uv names each large download with its size, then says when it is done
            $sizes[$Matches[1]] = [double]$Matches[2] * @{ K = 1KB; M = 1MB; G = 1GB }[$Matches[3]]; $want += $sizes[$Matches[1]]
            Show-Live (Get-Bar $got $want $watch.Elapsed.TotalSeconds)
        } elseif ($Live -and $text -match '^\s*Downloaded (.+?)\s*$' -and $sizes.ContainsKey($Matches[1])) {
            $got += $sizes[$Matches[1]]; Show-Live (Get-Bar $got $want $watch.Elapsed.TotalSeconds)
        } elseif ($text -notmatch '^\s*[-+~] ') { Write-Above "  $text" }
        $text
    }
    Stop-Live
    if ($LASTEXITCODE -eq 0) { return }
    foreach ($k in $Known) { if (($lines -join "`n") -match $k[0]) { Stop-With $k[1] $k[2] $k[3] } }
    Stop-With 'E_INSTALL_FAILED' "The step '$(Split-Path -Leaf $Exe) $($Arguments[0])' failed." "$Retry If it fails again, send the details file with your question."
}

function Get-State([string]$Root) { try { return Get-Content -LiteralPath (Join-Path $Root 'state.json') -Raw -Encoding UTF8 | ConvertFrom-Json } catch { return $null } }

function New-Link([string]$Name, [string]$Target, [string]$Arguments, [string]$Text) {  # saved in AppRoot (ASCII by rule), then moved: the shortcut object writes its path in the old code page
    $menu = (New-Item -ItemType Directory -Force -Path (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs')).FullName
    $link = (New-Object -ComObject WScript.Shell).CreateShortcut(($draft = Join-Path $AppRoot "$Name.lnk"))
    $link.TargetPath, $link.Arguments, $link.WorkingDirectory, $link.Description = $Target, $Arguments, $AppRoot, $Text
    $link.Save()
    Move-Item -LiteralPath $draft -Destination (Join-Path $menu "$Name.lnk") -Force
}

function Main {  # an error no step expected still ends with the plain three lines; its details go to the log
    try { Install-Steps } catch {
        if ($script:LogFile) { Add-Content -LiteralPath $script:LogFile -Value ($_ | Out-String) -Encoding UTF8 }
        Stop-With 'E_INSTALL_FAILED' "The install stopped at an unexpected error: $($_.Exception.Message)" "$Retry If it fails again, send the details file with your question."
    }
}

function Install-Steps {
    $os = [Environment]::OSVersion.Version
    $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    if ($os.Major -lt 10 -or $os.Build -lt 19045 -or $arch -ne 'AMD64') {
        Stop-With 'E_NOT_SUPPORTED' 'This computer is not supported in version 1. Fluentfolio needs Windows 10 22H2 or later, or Windows 11, on a 64-bit Intel or AMD processor.' 'Use a supported computer.'
    }
    if ((Test-Elevated) -and -not $AllowElevatedTest) {
        Stop-With 'E_ELEVATED' 'The installer was started as administrator. Fluentfolio installs for your user only.' 'Double-click Install.cmd again, without "Run as administrator".'
    }
    if ($Uninstall) { & (Join-Path $PSScriptRoot 'uninstall.ps1') -AppRoot $AppRoot -FallbackRoot $FallbackRoot -Yes:$Yes; exit $LASTEXITCODE }
    if ($Update) {
        Say "To update Fluentfolio:`n  1. Download the newest release from $Releases (the Source code zip).`n  2. Extract the zip to a folder of your choice.`n  3. Double-click Install.cmd in that folder. Your recordings, results and settings are kept."
        if (-not $Yes) { Start-Process $Releases }
        exit 0
    }
    $sac = Get-SacState
    if ($sac -eq 1) { Stop-With 'E_SAC_ENFORCED' 'Smart App Control is on. It blocks programs that Fluentfolio needs.' 'Read docs/troubleshooting.md#e_sac_enforced.' }
    if ($sac -eq 2) { Say 'Note (W_SAC_EVALUATION): Smart App Control is in evaluation mode and may later block Fluentfolio.' }
    $problems = Get-PathProblems $AppRoot
    if ($problems.Count -gt 0 -and -not (Get-State $AppRoot)) {
        Say "The usual folder $AppRoot cannot be used ($($problems -join ', '))."
        if ((Get-PathProblems $FallbackRoot).Count -gt 0) { Stop-With 'E_PATH_RULE' "Neither $AppRoot nor $FallbackRoot can be used." 'Run install.ps1 with -AppRoot and a short folder path, for example on another drive.' }
        if (-not (Confirm-Step "Use $FallbackRoot instead, readable only by you?")) { Say 'Nothing was installed.'; exit 0 }
        $script:AppRoot = $FallbackRoot
        New-Item -ItemType Directory -Force -Path $AppRoot | Out-Null
        if (-not (Set-OwnerOnly $AppRoot)) { Stop-With 'E_PATH_RULE' "$AppRoot could not be made readable by you only. Another account may own it." 'Choose another folder with -AppRoot.' }
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $AppRoot 'logs') | Out-Null
    $script:LogFile = Join-Path $AppRoot 'logs\install.log'
    Say "Fluentfolio installer, $((Get-Date).ToUniversalTime().ToString('u'))"
    Step 1 'Checking this computer and the connection'
    $state = Get-State $AppRoot
    $data = if ($env:FLUENTFOLIO_HOME) { $env:FLUENTFOLIO_HOME } elseif ($state.data_root) { $state.data_root } else { Join-Path $env:LOCALAPPDATA 'Fluentfolio-data' }
    foreach ($lock in (Join-Path $data 'app.lock'), (Join-Path $data 'work.lock')) {
        # The running app keeps its lock files open, so nobody else can open them alone.
        try { if (Test-Path -LiteralPath $lock) { [IO.File]::Open($lock, 'Open', 'ReadWrite', 'None').Close() } }
        catch { Stop-With 'E_APP_RUNNING' 'Fluentfolio is running.' 'Close the Fluentfolio window (the one that says keep this window open), then run this again.' }
    }
    if ($Source -match '\.zip$') {
        $unpacked = Join-Path $AppRoot 'tmp\source'
        Remove-Item -LiteralPath $unpacked -Recurse -Force -ErrorAction SilentlyContinue
        Expand-Archive -LiteralPath $Source -DestinationPath $unpacked
        $Source = (Get-ChildItem -LiteralPath $unpacked -Recurse -Filter pyproject.toml | Select-Object -First 1).DirectoryName
    } else { Get-ChildItem -LiteralPath $Source -Recurse -File -ErrorAction SilentlyContinue | Unblock-File }  # its own folder by default
    $found = if ($Source) { Select-String -LiteralPath (Join-Path $Source 'pyproject.toml') -Pattern '^version = "([0-9A-Za-z.+-]+)"' -ErrorAction SilentlyContinue }
    if (-not $found) { Stop-With 'E_SOURCE' "No Fluentfolio release was found in $Source." 'Extract the whole release zip, then double-click Install.cmd inside it.' }
    $version = $found[0].Matches[0].Groups[1].Value
    $free = (New-Object IO.DriveInfo([IO.Path]::GetPathRoot($AppRoot))).AvailableFreeSpace
    Assert-Disk $CoreDiskGB $free
    if (-not (Test-Path -LiteralPath $VcRuntime)) { Stop-With 'E_VCRUNTIME_MISSING' 'The Microsoft Visual C++ runtime is missing.' "Install 'Visual C++ Redistributable x64' from https://aka.ms/vs/17/release/vc_redist.x64.exe. $Retry" }
    Test-Hosts
    $speed = Measure-Speed
    if ($PreflightOnly) { Say 'Preflight passed.'; exit 0 }
    $what = if (-not $state) { 'installed' } elseif ($state.version -ne $version) { "updated from $($state.version)" } else { 'repaired' }; $about = if ($state) { 'at most about' } else { 'about' }  # an update or a repair reuses what uv downloaded before
    Say "`nFluentfolio $version will be $what for this Windows user. No admin rights are needed.`n  Program, tools and models: $AppRoot`n  Your recordings and results: $data"
    Say ('  Download now: {3} {0:N1} GB (uv, Python 3.12 and the packages). Disk space needed: about {1} GB; free: {2:N1} GB.' -f ($CoreMB / 1000), $CoreDiskGB, ($free / $GB), $about)
    if ($speed -gt 0) { Say ('  Time: {2} {0:N0} minutes to download at the speed measured now ({1:N1} MB/s). Unpacking and slower servers can double or triple that.' -f [Math]::Max(1, $CoreMB * $MB / $speed / 60), ($speed / $MB), $about) }  # test installs took 0.8 to 2.3 times their download time (5.6 while a disk scan ran)
    if ($Device -ne 'cpu') { Say ('  If an NVIDIA graphics card can be used, GPU support adds {2} {0:N1} GB to download{1} and about {3} GB of disk space.' -f ($GpuMB / 1000), $(if ($speed -gt 0) { ' ({1} {0:N0} more minutes)' -f [Math]::Max(1, $GpuMB * $MB / $speed / 60), $about }), $about, $GpuDiskGB) }
    Say '  Then a page in your browser lists the speech models (several GB more). Untick what you do not want.'
    if (-not (Confirm-Step 'Continue?')) { Say 'Nothing was installed.'; exit 0 }
    Step 2 "Getting uv $UvVersion, the tool that installs Python and the packages"
    $uv = Join-Path (New-Item -ItemType Directory -Force -Path ($bin = Join-Path $AppRoot 'bin')) 'uv.exe'
    if ((Test-Path -LiteralPath $uv) -and ((& $uv --version) -like "uv $UvVersion *")) { Say '  It is here already.' } else {
        $zip = Join-Path $bin 'uv.download.zip'  # Expand-Archive takes only names that end in .zip
        try { Save-File $UvUrl $zip } catch { Stop-With 'E_NETWORK' 'The download of uv failed.' $Net }
        $good = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -eq $UvSha256
        if ($good) { Expand-Archive -LiteralPath $zip -DestinationPath $bin -Force }
        Remove-Item -LiteralPath $zip
        if (-not $good) { Stop-With 'E_HASH_MISMATCH' 'The downloaded uv file is damaged or was changed on the way.' $Retry }
    }
    $app = Join-Path $AppRoot "app\$version"
    if ($what -like 'updated*') { $script:Kept, $script:NewApp = $state.version, $app }
    Step 3 "Copying the program to $app"
    & (Join-Path $env:SystemRoot 'System32\robocopy.exe') $Source $app /MIR /XD .git .venv __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { Stop-With 'E_INSTALL_FAILED' "The program could not be copied to $app." $Retry }
    $settings = @{ UV_CACHE_DIR = "$AppRoot\cache\uv"; UV_PYTHON_INSTALL_DIR = "$AppRoot\python"; UV_PYTHON_PREFERENCE = 'only-managed'
        UV_PYTHON_INSTALL_REGISTRY = '0'; UV_PYTHON_INSTALL_BIN = '0'; FLUENTFOLIO_APP_ROOT = $AppRoot }
    foreach ($name in $settings.Keys) { Set-Item -Path "Env:$name" -Value $settings[$name] }  # no UV_NO_CONFIG: it drops the project's own uv settings, so the lock never matches
    Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
    $backend = if ($state.backend -eq 'cu130') { 'cu130' } else { 'cpu' }  # a choice stored by the device step, read back
    $chosen = @('--extra', $backend) + @(@($state.extras) | Where-Object { $_ -eq 'nc' } | ForEach-Object { '--extra', $_ })
    Step 4 "Installing Python 3.12 and the packages ($backend)"
    $sync = @('sync', '--locked', '--no-dev', '--project', $app) + $chosen
    Invoke-Tool $uv ($sync + @('--no-build', '--no-install-project'))  # wheels only: no package's build script runs
    Invoke-Tool $uv $sync  # then Fluentfolio itself, which uv builds with its own backend
    $python, $exe = (Join-Path $app '.venv\Scripts\python.exe'), (Join-Path $app '.venv\Scripts\fluentfolio.exe')
    if (-not ((Test-Path -LiteralPath $python) -and (Test-Path -LiteralPath $exe))) { Stop-With 'E_FILE_BLOCKED' 'A file of the install is missing right after it was made. A security program may have removed it.' $Shield }
    Step 5 $(if ($script:Kept) { 'Starting the new version once, and finishing' } else { 'Choosing between the graphics card (GPU) and the processor (CPU), and finishing' })
    if ($script:Kept) {  # keeps the backend chosen before; the self-test checks the GPU again
        if ((& $python -I -m fluentfolio --version) -notlike "*$version*") { Stop-With 'E_INSTALL_FAILED' "Version $version did not start." $Retry }
    } else {
        & $python -I -m fluentfolio.setup device --choice $Device
        if ($LASTEXITCODE -ne 0) { Stop-With 'E_INSTALL_FAILED' 'The device step failed.' $Retry }
    }
    & $python -I -m fluentfolio.setup finish  # the switch: state.json names the new version
    if ($LASTEXITCODE -ne 0) { Stop-With 'E_INSTALL_FAILED' 'The last install step failed.' $Retry }
    $script:Kept = $null
    Copy-Item -LiteralPath (Join-Path $app 'install.ps1'), (Join-Path $app 'uninstall.ps1'), (Join-Path $app 'Update.cmd') -Destination $AppRoot -Force
    New-Link 'Fluentfolio' $exe 'serve' 'Start Fluentfolio. Keep its window open while you use it.'
    New-Link 'Uninstall Fluentfolio' (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') "-NoProfile -ExecutionPolicy Bypass -File `"$AppRoot\uninstall.ps1`" -FromShortcut" 'Remove Fluentfolio.'
    Get-ChildItem -LiteralPath (Join-Path $AppRoot 'app') -Directory | Where-Object { $_.Name -ne $version } | Remove-Item -Recurse -Force
    Say "`nReady: Fluentfolio $version is $what. Start it from the Start menu: Fluentfolio." 'Green'
    if (-not $Yes) { Start-Process -FilePath $exe -ArgumentList 'serve' -WorkingDirectory $AppRoot }
}

if ($MyInvocation.InvocationName -ne '.') { Main }
