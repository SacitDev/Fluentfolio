@echo off
rem Shows how to update Fluentfolio and opens the page with the newest version.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Update
pause
