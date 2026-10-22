"""Run inside a fresh user/network namespace; enable only loopback."""
import fcntl
import os
from pathlib import Path
import socket
import struct
import sys

root = Path(__file__).resolve().parents[1]
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    request = struct.pack('16sH22x', b'lo', 1)  # IFF_UP
    fcntl.ioctl(sock, 0x8914, request)  # SIOCSIFFLAGS
home = root / 'state' / 'offline-home'
home.mkdir(parents=True, exist_ok=True)
os.environ.update(HOME=str(home), XDG_CACHE_HOME=str(home / '.cache'),
                  XDG_CONFIG_HOME=str(home / '.config'), XDG_DATA_HOME=str(home / '.local/share'),
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                  DO_NOT_TRACK='1', OTEL_SDK_DISABLED='true')
for key in list(os.environ):
    if key.lower().endswith('_proxy'):
        del os.environ[key]
args = sys.argv[1:] or ['--chat']
os.environ['CAPSTONE_STRICT_OFFLINE'] = '1'
if '--backend=airllm' in args or '--backend' in args and 'airllm' in args:
    raise SystemExit('Offline entry supports the existing llama.cpp + Hermes backend only.')
os.chdir(root)
sys.path.insert(0, str(root))
from scripts.import_models import sync
try:
    sync(root)
except (OSError, ValueError, RuntimeError) as exc:
    raise SystemExit('Model import failed: ' + str(exc))
os.execv(sys.executable, [sys.executable, '-s', '-B', str(root / 'capstone.py'), *args])
