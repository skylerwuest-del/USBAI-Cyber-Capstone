#!/usr/bin/env bash
set -euo pipefail
CAPSTONE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$CAPSTONE_ROOT"
python3 setup.py "$@"
python3 setup_hermes.py --offline
