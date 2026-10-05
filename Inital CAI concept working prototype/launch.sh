#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$ROOT"
if [[ $# -eq 0 || "${1:-}" == "--menu" ]]; then
    exec python3 "$ROOT/menu.py"
fi
if [[ "${1:-}" == "--run" ]]; then
    shift
fi
exec python3 "$ROOT/capstone.py" "$@"
