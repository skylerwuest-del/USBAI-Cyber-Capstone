#!/usr/bin/env bash
set -euo pipefail
CAPSTONE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$CAPSTONE_ROOT"
CAPSTONE_PYTHON="$CAPSTONE_ROOT/hermes/python/bin/python3.12"
if [[ ! -x "$CAPSTONE_PYTHON" ]]; then
    # A host Python can report missing portable assets before starting the app.
    CAPSTONE_PYTHON=python3
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
if ! command -v "$CAPSTONE_PYTHON" >/dev/null 2>&1; then
    echo 'Cannot launch: Python is missing. Install python3, then run bash setup.sh.' >&2
    exit 1
fi
"$CAPSTONE_PYTHON" -s -B launch_check.py
if [[ $# -eq 0 ]]; then
    exec "$CAPSTONE_PYTHON" -s -B model_startup.py
fi
exec "$CAPSTONE_PYTHON" -s -B capstone.py "$@"
