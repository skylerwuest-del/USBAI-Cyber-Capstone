#!/usr/bin/env python3
"""Offline, standard-library-only GGUF selection. stdout is one JSON object."""
import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def architecture():
    machine = platform.machine().lower()
    try:
        return {'x86_64': 'x86_64', 'amd64': 'x86_64', 'aarch64': 'arm64', 'arm64': 'arm64'}[machine]
    except KeyError:
        raise ValueError('Unsupported Linux architecture: ' + machine) from None


def validate_elf(path, arch):
    with path.open('rb') as f:
        header = f.read(20)
    expected = {'x86_64': 62, 'arm64': 183}[arch]
    if len(header) != 20 or header[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', header, 18)[0] != expected:
        raise ValueError('Missing/incompatible ELF64 binary for ' + arch + ': ' + str(path))
    if not os.access(path, os.X_OK):
        raise ValueError('Binary is not executable: ' + str(path))


def deny_network():
    """Linux seccomp, inherited across exec and by descendants; fail closed.

    No Python preexec_fn: this runs in a fresh single-threaded interpreter.
    Also reject alternate ABIs so x32/compat syscalls cannot bypass the filter.
    """
    arch = architecture()
    audit, socket_calls = {'x86_64': (0xc000003e, (41, 53)), 'arm64': (0xc00000b7, (198, 199))}[arch]
    class Filter(ctypes.Structure):
        _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte), ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint)]
    class Program(ctypes.Structure):
        _fields_ = [('len', ctypes.c_ushort), ('filter', ctypes.POINTER(Filter))]
    rows = [(0x20, 0, 0, 4), (0x15, 1, 0, audit), (0x06, 0, 0, 0x80000000),
            (0x20, 0, 0, 0), (0x45, 0, 1, 0x40000000), (0x06, 0, 0, 0x80000000)]
    for call in socket_calls:
        rows += [(0x15, 0, 1, call), (0x06, 0, 0, 0x50000 | errno.EPERM)]
    rows += [(0x06, 0, 0, 0x7fff0000)]
    filters = (Filter * len(rows))(*(Filter(*row) for row in rows))
    program = Program(len(rows), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        raise OSError(ctypes.get_errno(), 'Cannot enforce offline seccomp policy')


def isolated_run(binary, arguments):
    with tempfile.TemporaryDirectory(prefix='capstone-llmfit-', dir='/tmp') as tmp:
        env = {'PATH': '/usr/bin:/bin', 'HOME': tmp, 'TMPDIR': tmp,
               'XDG_CACHE_HOME': tmp, 'XDG_CONFIG_HOME': tmp, 'XDG_DATA_HOME': tmp,
               'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
               'DO_NOT_TRACK': '1', 'NO_COLOR': '1'}
        command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                   '--isolated-exec', str(binary), *arguments]
        with tempfile.TemporaryFile(dir='/tmp') as out, tempfile.TemporaryFile(dir='/tmp') as err:
            proc = subprocess.Popen(command, cwd=tmp, env=env, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=err, start_new_session=True)
            try:
                proc.wait(timeout=20)
            finally:
                # Hardware helper descendants must not survive either success or timeout.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            out.seek(0)
            err.seek(0)
            data = out.read(1024 * 1024 + 1)
            if proc.returncode or len(data) > 1024 * 1024:
                raise ValueError('llmfit failed: ' + err.read(2048).decode(errors='replace'))
            return data.decode('utf-8')


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid positive number: ' + label)
    return value


def parse_hardware(data):
    if not isinstance(data, dict) or not isinstance(data.get('system'), dict):
        raise ValueError('llmfit JSON must contain a system object')
    system = data['system']
    total = positive(system['total_ram_gb'], 'total_ram_gb')
    available = positive(system['available_ram_gb'], 'available_ram_gb')
    return dict(system, total_ram_gb=total, available_ram_gb=min(total, available), source='llmfit')


def fallback_hardware():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            values[key] = int(value.split()[0]) * 1024 / GIB
    return {'total_ram_gb': positive(values['MemTotal'], 'MemTotal'),
            'available_ram_gb': positive(values['MemAvailable'], 'MemAvailable'),
            'cpu_cores': os.cpu_count() or 1, 'has_gpu': False,
            'backend': 'CPU', 'unified_memory': False, 'source': '/proc/meminfo fallback'}


def detect_hardware(root, arch):
    try:
        binary = root / 'llmfit' / 'linux' / arch / 'llmfit'
        if not binary.resolve().is_relative_to(root.resolve()):
            raise ValueError('llmfit binary escapes USB project')
        validate_elf(binary, arch)
        # Preparation records the exact tested binary version; never execute an unknown CLI.
        expected = (binary.parent / 'version.txt').read_text().strip()
        if not expected.startswith('llmfit ') or isolated_run(binary, ['--version']).strip() != expected:
            raise ValueError('llmfit version differs from preparation version.txt')
        return parse_hardware(json.loads(isolated_run(binary, ['--no-dashboard', 'system', '--json'])))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print('llmfit unavailable: ' + str(exc) + '; using CPU-only /proc/meminfo fallback. Bundle a tested binary during USB preparation.', file=sys.stderr)
        return fallback_hardware()


def load_models(root):
    store = root / 'models'
    if store.is_symlink() or store.resolve().parent != root.resolve():
        raise ValueError('models directory must reside on this USB project')
    manifest = store / 'models.json'
    if not manifest.resolve().is_relative_to(store.resolve()):
        raise ValueError('Manifest escapes models directory')
    doc = json.loads(manifest.read_text())
    if not isinstance(doc, dict) or not isinstance(doc.get('models'), list):
        raise ValueError('models.json must contain a models array')
    models = []
    names = set()
    for entry in doc['models']:
        try:
            if not isinstance(entry, dict):
                raise ValueError('Model entry must be an object')
            name = entry['name']
            if not isinstance(name, str) or not name or name in names:
                raise ValueError('Invalid or duplicate model name')
            names.add(name)
            relative = Path(entry['file'])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != ('models',):
                raise ValueError('file must be project-relative under models/')
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(store.resolve()) or not path.is_file() or path.suffix.lower() != '.gguf':
                raise ValueError('Model must be a regular GGUF inside models/')
            for field in ('parameters_b', 'estimated_ram_gb', 'kv_bytes_per_token', 'context_size'):
                positive(entry[field], field)
            if type(entry['context_size']) is not int or entry['context_size'] < 4096:
                raise ValueError('context_size must be an integer >= 4096')
            if type(entry['priority']) is not int or not isinstance(entry['quantization'], str) or not entry['quantization']:
                raise ValueError('Invalid priority or quantization')
            checksum = entry['sha256']
            if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
                raise ValueError('sha256 must be the preparation-verified file digest')
            with path.open('rb') as handle:
                header = handle.read(24)
                if len(header) < 24 or header[:4] != b'GGUF' or struct.unpack_from('<I', header, 4)[0] not in (2, 3) or not struct.unpack_from('<Q', header, 8)[0]:
                    raise ValueError('Invalid GGUF v2/v3 header or empty tensor table')
                handle.seek(0)
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != checksum:
                raise ValueError('GGUF checksum mismatch; revalidate during USB preparation')
            sys.path.insert(0, str(ROOT))
            from gguf_metadata import context_length
            from scripts.gguf_check import validate_structure
            validate_structure(path)
            native = context_length(path)
            if native is None or entry['context_size'] > native:
                raise ValueError('Manifest context exceeds or lacks GGUF native context')
            models.append(dict(entry, path=str(path), size_bytes=path.stat().st_size))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, struct.error) as exc:
            print('Rejected model ' + repr(entry) + ': ' + str(exc), file=sys.stderr)
    return models


def choose(models, hardware, requested=None):
    total = hardware['total_ram_gb']
    # MemAvailable already excludes occupied Ubuntu memory. Reserve future agent,
    # server, and OS growth, then leave 15% of remaining RAM untouched.
    budget = max(0, min(total, hardware['available_ram_gb']) - 2.0) * 0.85
    candidates = []
    for model in models:
        if requested and Path(requested).resolve() != Path(model['path']):
            continue
        weights_and_runtime = max(model['estimated_ram_gb'], model['size_bytes'] / GIB * 1.10 + 0.5)
        required = weights_and_runtime + model['context_size'] * model['kv_bytes_per_token'] / GIB
        if required <= budget:
            candidates.append((model, required))
        else:
            print(f"Rejected {model['name']}: needs {required:.2f} GiB, budget {budget:.2f} GiB", file=sys.stderr)
    if not candidates:
        raise ValueError('No safe manifest-listed local model fits available RAM (or requested model is ineligible)')
    model, required = sorted(candidates, key=lambda item: (-item[0]['priority'], item[1], item[0]['name']))[0]
    return {'model': model['path'], 'model_label': model['name'], 'context_size': model['context_size'],
            'n_gpu_layers': 0, 'threads': max(1, (os.cpu_count() or 2) // 2),
            'estimated_required_gib': required, 'memory_budget_gib': budget, 'hardware': hardware}


def safe_server_args(arguments):
    """Keep the existing reasoning setting; reject unbudgeted/server overrides."""
    if (not isinstance(arguments, list) or len(arguments) % 2 or
            any(arguments[i] != '--reasoning-budget' or
                not isinstance(arguments[i + 1], str) or
                not arguments[i + 1].isdigit() for i in range(0, len(arguments), 2))):
        raise ValueError('Offline server_args only supports --reasoning-budget INTEGER')
    return arguments


def select(root=ROOT, requested=None):
    arch = architecture()
    hardware = detect_hardware(root, arch)
    # Cross-check rounded llmfit values against the kernel's current available RAM.
    kernel = fallback_hardware()
    for key in ('total_ram_gb', 'available_ram_gb'):
        hardware[key] = min(hardware[key], kernel[key])
    result = choose(load_models(root), hardware, requested)
    server = root / 'runtime' / 'linux' / arch / 'llama-server'
    if not server.exists():
        server = root / 'runtime' / 'llama-server'  # existing layout, only if ELF matches
    if not server.resolve().is_relative_to(root.resolve()):
        raise ValueError('llama-server binary escapes USB project')
    validate_elf(server, arch)
    result['server'] = str(server.resolve())
    print(f"Selected {result['model_label']}: {result['estimated_required_gib']:.2f} GiB / {result['memory_budget_gib']:.2f} GiB budget; CPU-only", file=sys.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Require this exact manifest-listed local model')
    args = parser.parse_args()
    try:
        print(json.dumps(select(requested=args.model)))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('Offline model selection failed: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--isolated-exec':
        deny_network()
        os.execv(sys.argv[2], sys.argv[2:])
    raise SystemExit(main())
