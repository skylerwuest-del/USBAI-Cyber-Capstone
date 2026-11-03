#!/usr/bin/env bash
set -euo pipefail
CAPSTONE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$CAPSTONE_ROOT"
CAPSTONE_PYTHON="$CAPSTONE_ROOT/hermes/python/bin/python3.12"
if [[ ! -f "$CAPSTONE_PYTHON" ]]; then
    echo "Hermes is missing. Run: bash setup.sh"
    exit 1
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
if [[ $# -eq 0 ]]; then
    exec "$CAPSTONE_PYTHON" -s -B menu.py
fi
exec "$CAPSTONE_PYTHON" -s -B capstone.py "$@"
