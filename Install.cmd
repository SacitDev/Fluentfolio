@echo off
setlocal
rem Installs, repairs or updates Fluentfolio from this folder. Read the result before you close this window.
rem Windows PowerShell builds its own module path; one from a PowerShell 7 window would load modules it
rem cannot use.
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
pause
