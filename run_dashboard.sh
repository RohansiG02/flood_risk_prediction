#!/bin/bash
# Start the dashboard at http://localhost:8501
cd "$(dirname "$0")"

# Use an existing environment if there is one (./.venv, or ../.venv in the original project folder),
# otherwise create ./.venv and install the requirements.
if [ -x .venv/bin/streamlit ]; then
  PY=.venv/bin
elif [ -x ../.venv/bin/streamlit ]; then
  PY=../.venv/bin
else
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  PY=.venv/bin
fi

$PY/streamlit run app.py
