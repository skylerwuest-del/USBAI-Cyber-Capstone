#!/usr/bin/env bash
set -euo pipefail
CAPSTONE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$CAPSTONE_ROOT"
CAPSTONE_PYTHON="$CAPSTONE_ROOT/hermes/python/bin/python3.12"
if [[ ! -x "$CAPSTONE_PYTHON" ]]; then
    # The standard-library startup menu can explain missing local assets offline.
    CAPSTONE_PYTHON=python3
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
if [[ $# -eq 0 ]]; then
    exec "$CAPSTONE_PYTHON" -s -B model_startup.py
fi
exec "$CAPSTONE_PYTHON" -s -B capstone.py "$@"
