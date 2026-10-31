#!/usr/bin/env bash
# shop-tour setup for Linux and macOS:  bash setup.sh   (run it again any time: it skips what is already done)
#   bash setup.sh --dry-run    says what it would do and changes nothing
set -u
cd "$(dirname "$0")"
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY=1
run() { if [ -n "$DRY" ]; then echo " [dry-run] $*"; else "$@"; fi; }
fail() { echo; echo " [!!] $*"; echo; echo " Setup did not finish. Fix that and run setup.sh again: it carries on where it stopped."; exit 1; }

echo
echo " shop-tour setup"
echo " The first time takes 10 to 30 minutes and downloads 3 to 5 GB."
[ -n "$DRY" ] && echo " [dry-run] nothing will be installed or changed"
echo
[ "$(uname)" = "Darwin" ] && echo " [!!] macOS is not tested: it has no NVIDIA graphics card, so tours build on the CPU (well over an hour)."

# 1. Python 3.11 to 3.13
PY=""
for c in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" tools/setup_help.py pyok 2>/dev/null; then PY="$(command -v "$c")"; break; fi
done
[ -n "$PY" ] || fail "Python 3.11, 3.12 or 3.13 is needed. Ubuntu: sudo apt install python3 python3-venv   macOS: brew install python@3.13   or https://www.python.org/downloads/"
echo " [ok] Python: $PY"

# 2. this folder's own Python, .venv
VPY="$PWD/.venv/bin/python"
if [ -x "$VPY" ]; then echo " [ok] .venv is there"
else
  echo " [..] making .venv, this folder's own Python"
  run "$PY" -m venv .venv || fail "could not make .venv. Ubuntu: sudo apt install python3-venv, then run setup.sh again."
fi
[ -x "$VPY" ] || VPY="$PY"

# 3. git and FFmpeg
command -v git >/dev/null 2>&1 || fail "git is needed (one package comes from GitHub). Ubuntu: sudo apt install git   macOS: xcode-select --install"
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  fail "FFmpeg is needed. Ubuntu: sudo apt install ffmpeg   macOS: brew install ffmpeg"
fi
echo " [ok] git and FFmpeg"

# 4. PyTorch: the GPU build with an NVIDIA card (driver 580+), else the CPU build
TAG="$("$VPY" tools/setup_help.py torch-tag)"
PINS="$("$VPY" tools/setup_help.py torch-pins)"
if "$VPY" tools/setup_help.py torch-ok "$TAG"; then echo " [ok] PyTorch is installed: $PINS +$TAG"
else
  echo " [..] installing PyTorch, $PINS +$TAG - the big download, 1 to 3 GB"
  run "$VPY" -m pip install --upgrade pip || fail "pip could not update itself (see above)"
  if [ "$TAG" = "pypi" ]; then run "$VPY" -m pip install $PINS || fail "PyTorch did not install (see above)"
  else run "$VPY" -m pip install $PINS --index-url "https://download.pytorch.org/whl/$TAG" || fail "PyTorch did not install (see above)"; fi
fi

# 5. everything else in requirements.txt, skipped when it has not changed
if "$VPY" tools/setup_help.py reqs-current; then echo " [ok] the other packages are installed"
else
  echo " [..] installing the other packages"
  run "$VPY" -m pip install -r requirements.txt || fail "a package did not install (see above)"
  run "$VPY" tools/setup_help.py reqs-stamp
fi

# 6. the two blur models for plates and printed text, about 165 MB, once
run "$VPY" tools/fetch_pii_models.py || fail "the blur models did not download (see above); check the internet connection"

echo
echo " ============================================================"
echo "  All set. Next: run  bash start.sh  and choose your video."
echo "  Not recorded yet? Read CAPTURE.md first - it takes 10 minutes."
echo " ============================================================"
