#!/usr/bin/env bash
# Strict offline entry; original launch.sh remains the developer/preparation menu.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "$(uname -m)" in
    x86_64) ARCH=x86_64 ;;
    aarch64|arm64) ARCH=arm64 ;;
    *) echo 'Unsupported Linux architecture' >&2; exit 1 ;;
esac
PYTHON="$ROOT/hermes/linux/$ARCH/python/bin/python3.12"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$ROOT/hermes/python/bin/python3.12"; fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Bundle portable Python and Hermes dependencies for $ARCH during USB preparation." >&2
    exit 1
fi
if ! command -v unshare >/dev/null; then
    echo 'Strict offline startup requires Linux unshare (util-linux); prepare a compatible host image.' >&2
    exit 1
fi
# Inspect the interpreter before executing it (ELF64 little-endian, matching CPU).
case "$ARCH" in x86_64) ELF_MACHINE=62 ;; arm64) ELF_MACHINE=183 ;; esac
if [[ "$(od -An -tx1 -N6 "$PYTHON" | tr -d ' \n')" != '7f454c460201' ||
      "$(od -An -tu2 -j18 -N2 "$PYTHON" | tr -d ' \n')" != "$ELF_MACHINE" ]]; then
    echo "Bundled Python is not a matching ELF64 binary for $ARCH." >&2
    exit 1
fi
if ! unshare --user --map-root-user --net -- true; then
    echo 'Offline isolation unavailable: host policy must permit unprivileged user/network namespaces. No application was started.' >&2
    exit 1
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
# No inherited proxy, update, registry, or Internet access survives this namespace.
exec unshare --user --map-root-user --net -- "$PYTHON" -I -B "$ROOT/scripts/offline_launch.py" "$@"
