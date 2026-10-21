@echo off
setlocal
rem Shows how to update Fluentfolio and opens the page with the newest version.
rem Windows PowerShell builds its own module path; one from a PowerShell 7 window would load modules it
rem cannot use.
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Update
pause
