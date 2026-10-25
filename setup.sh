#!/usr/bin/env sh
# One-command setup for Mac and Linux: ./setup.sh
# Makes .venv, installs the pinned requirements, and starts the app. Safe to run again.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Creating .venv ..."
    python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
exec .venv/bin/python -m streamlit run app.py
