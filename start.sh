#!/usr/bin/env bash
# shop-tour: open the upload page on this computer.  bash start.sh   (run setup.sh once first)
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then echo "Run setup.sh first:  bash setup.sh"; exit 1; fi
exec .venv/bin/python app.py "$@"
