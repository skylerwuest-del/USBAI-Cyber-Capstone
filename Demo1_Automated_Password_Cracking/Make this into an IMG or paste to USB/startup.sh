#!/usr/bin/env bash
set -euo pipefail
CAPSTONE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# USB launcher only: no Ubuntu login hooks or services are installed.
exec bash "$CAPSTONE_ROOT/launch.sh" --chat "$@"
