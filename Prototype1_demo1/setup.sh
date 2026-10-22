#!/usr/bin/env bash
# Public setup entry: online preparation by default; --offline reuses verified caches.
set -euo pipefail
CAPSTONE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$CAPSTONE_ROOT"
exec python3 -B "$CAPSTONE_ROOT/prepare.py" "$@"
