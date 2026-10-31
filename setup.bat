@echo off
rem shop-tour setup for Windows: double-click it. Run it again any time: it skips what is already done.
rem   setup.bat --dry-run    says what it would do and changes nothing
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "DRY="
if /i "%~1"=="--dry-run" set "DRY=1"
set "PATH=%PATH%;%LOCALAPPDATA%\Microsoft\WinGet\Links;%ProgramFiles%\Git\cmd"
echo.
echo  shop-tour setup
echo  The first time takes 10 to 30 minutes and downloads 3 to 5 GB. Leave this window open.
if defined DRY echo  [dry-run] nothing will be installed or changed
echo.

rem ---- 1. Python 3.11 to 3.13 ----
set "PYEXE="
for %%v in (3.13 3.12 3.11) do if not defined PYEXE call :trypy py -%%v
if not defined PYEXE call :trypy python
if not defined PYEXE goto :nopython
echo  [ok] Python: !PYEXE!

rem ---- 2. this folder's own Python, .venv ----
set "VPY=%CD%\.venv\Scripts\python.exe"
if exist "%VPY%" (
  echo  [ok] .venv is there
) else (
  echo  [..] making .venv, this folder's own Python
  call :run "!PYEXE!" -m venv .venv || goto :fail
)
if not exist "%VPY%" set "VPY=!PYEXE!"

rem ---- 3. git and FFmpeg ----
where git >nul 2>&1 || call :winget Git.Git git
where git >nul 2>&1 || if not defined DRY goto :nogit
where ffmpeg >nul 2>&1 && where ffprobe >nul 2>&1 || call :winget Gyan.FFmpeg FFmpeg
where ffmpeg >nul 2>&1 && where ffprobe >nul 2>&1 || if not defined DRY goto :noffmpeg
echo  [ok] git and FFmpeg

rem ---- 4. PyTorch: the GPU build with an NVIDIA card (driver 580+), else the CPU build ----
set "TAG=cpu"
for /f %%t in ('call "%VPY%" tools\setup_help.py torch-tag') do set "TAG=%%t"
for /f "delims=" %%p in ('call "%VPY%" tools\setup_help.py torch-pins') do set "PINS=%%p"
"%VPY%" tools\setup_help.py torch-ok !TAG!
if not errorlevel 1 (
  echo  [ok] PyTorch is installed: !PINS! +!TAG!
) else (
  echo  [..] installing PyTorch, !PINS! +!TAG! - the big download, 1 to 3 GB
  call :run "%VPY%" -m pip install --upgrade pip || goto :fail
  call :run "%VPY%" -m pip install !PINS! --index-url https://download.pytorch.org/whl/!TAG! || goto :fail
)

rem ---- 5. everything else in requirements.txt, skipped when it has not changed ----
"%VPY%" tools\setup_help.py reqs-current
if not errorlevel 1 (
  echo  [ok] the other packages are installed
) else (
  echo  [..] installing the other packages
  call :run "%VPY%" -m pip install -r requirements.txt || goto :fail
  call :run "%VPY%" tools\setup_help.py reqs-stamp
)

rem ---- 6. the two blur models for plates and printed text, about 165 MB, once ----
call :run "%VPY%" tools\fetch_pii_models.py || goto :fail

echo.
echo  ============================================================
echo   All set. Next: double-click start.bat and choose your video.
echo   Not recorded yet? Read CAPTURE.md first - it takes 10 minutes.
echo  ============================================================
goto :end

:trypy
%* tools\setup_help.py pyok >nul 2>&1 || exit /b 0
for /f "delims=" %%i in ('%* -c "import sys; print(sys.executable)"') do set "PYEXE=%%i"
exit /b 0

:run
if defined DRY (
  echo  [dry-run] %*
  exit /b 0
)
%*
exit /b

:winget
echo  [..] %2 is missing: installing it with winget
where winget >nul 2>&1 || exit /b 1
if defined DRY (
  echo  [dry-run] winget install -e --id %1
  exit /b 0
)
winget install -e --id %1 --accept-source-agreements --accept-package-agreements
exit /b 0

:nopython
echo.
echo  [!!] Python 3.11, 3.12 or 3.13 is needed and was not found.
echo       Get Python 3.13 from https://www.python.org/downloads/ - on its first screen tick "Add python.exe to PATH".
echo       Or open PowerShell and run:  winget install -e --id Python.Python.3.13
echo       Then double-click setup.bat again.
goto :fail
:nogit
echo.
echo  [!!] git is needed: one package comes from GitHub. Get it from https://git-scm.com/download/win and run setup.bat again.
goto :fail
:noffmpeg
echo.
echo  [!!] FFmpeg is needed. If it was just installed, close this window and double-click setup.bat again.
echo       Otherwise get it from https://www.gyan.dev/ffmpeg/builds/ and put ffmpeg.exe and ffprobe.exe on your PATH.
goto :fail

:fail
echo.
echo  Setup did not finish. The lines above say why. Fix that and double-click setup.bat again: it carries on where it stopped.
if not defined NOPAUSE pause
exit /b 1

:end
if not defined NOPAUSE pause
exit /b 0
