@echo off
rem shop-tour: double-click to open the upload page on this computer (run setup.bat once first).
cd /d "%~dp0"
set "PATH=%PATH%;%LOCALAPPDATA%\Microsoft\WinGet\Links;%ProgramFiles%\Git\cmd"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first: double-click it, wait for "All set", then double-click start.bat again.
  pause
  exit /b 1
)
title shop-tour - close this window to stop
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 pause
