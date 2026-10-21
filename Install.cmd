@echo off
rem Installs, repairs or updates Fluentfolio from this folder. Read the result before you close this window.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
pause
