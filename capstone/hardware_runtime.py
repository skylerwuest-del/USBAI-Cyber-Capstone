"""Select USB-local GPU runtimes on the existing Linux x86_64 target."""
import os
from pathlib import Path
import platform
import re
import struct
import subprocess

from portable_env import usb_environment

GPU_DEVICE = re.compile(r'^\s*((?:CUDA|Vulkan)\d+):', re.MULTILINE | re.IGNORECASE)


def cpu_threads():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def select_runtime(config, root):
    root = Path(root)
    configured = Path(config['server'])
    configured = configured if configured.is_absolute() else root / configured
    result = dict(server=str(configured), gpu=False, devices='', backend='CPU', probe_errors=[])
    policy = config.get('acceleration', 'auto')
    if policy not in ('auto', 'cpu', 'vulkan', 'cuda'):
        raise ValueError('acceleration must be auto, cpu, vulkan or cuda')
    if policy == 'cpu':
        return result
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        return result
    modes = ('cuda', 'vulkan') if policy == 'auto' else (policy,)
    for mode in modes:
        server = root / 'runtime' / mode / 'llama-server'
        if not server.is_file() or not os.access(server, os.X_OK):
            continue
        with server.open('rb') as handle:
            header = handle.read(20)
        if len(header) != 20 or header[:6] != b'\x7fELF\x02\x01' or struct.unpack('<H', header[18:20])[0] != 62:
            result['probe_errors'].append(mode + ': runtime ABI mismatch')
            continue
        env = runtime_environment(root, server)
        try:
            probe = subprocess.run([str(server), '--list-devices'], env=env, cwd=root,
                                   capture_output=True, text=True, timeout=15)
            output = (probe.stdout + probe.stderr)[-12000:]
            if probe.returncode == 0 and GPU_DEVICE.search(output):
                result.update(server=str(server), gpu=True, devices=output, backend=mode.upper())
                return result
            result['probe_errors'].append(mode + ': no usable device (exit ' + str(probe.returncode) + ')')
        except (OSError, subprocess.SubprocessError) as exc:
            result['probe_errors'].append(mode + ': ' + str(exc))
    if policy != 'auto':
        raise RuntimeError('Requested ' + policy + ' runtime unavailable; prepare it on USB and check host drivers')
    return result


def runtime_environment(root, server):
    env = usb_environment(root)
    # The application supplies launch settings, not inherited llama.cpp options.
    env = {key: value for key, value in env.items() if not key.startswith('LLAMA_ARG_')}
    env['LD_LIBRARY_PATH'] = os.pathsep.join(dict.fromkeys((str(Path(server).parent), str(Path(root) / 'runtime'))))
    return env


def acceleration_args(selection, config):
    if not selection['gpu']:
        return ['--n-gpu-layers', '0']
    # The pinned runtime fits offload to actual device memory. Keep context fixed
    # and retain a margin per GPU rather than combining RAM and VRAM budgets.
    return ['--n-gpu-layers', 'auto', '--fit', 'on',
            '--fit-target', str(config.get('gpu_reserve_mib', 1024))]
