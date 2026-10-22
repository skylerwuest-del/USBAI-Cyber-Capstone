"""Offline installed-GGUF fit analysis. No catalogs, downloads or network probes."""
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess

from gguf_metadata import read_metadata
from models_menu import selectable_names, shard_group
from usb_core import ROOT, resolve
from hardware_runtime import select_runtime, runtime_environment

GIB = 1024 ** 3
# Architectures whose full-attention KV upper bound is understood here.
DENSE_ARCHES = {'llama', 'qwen2', 'qwen3', 'gemma', 'gemma2', 'gemma3', 'mistral', 'phi3'}
QUANTS = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 7: 'Q8_0',
          8: 'Q5_0', 9: 'Q5_1', 10: 'Q2_K', 11: 'Q3_K_S', 12: 'Q3_K_M',
          13: 'Q3_K_L', 14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S',
          17: 'Q5_K_M', 18: 'Q6_K', 30: 'BF16'}


def runtime_compatibility(path, architecture):
    """Inspect CPU ABI before invoking a bundled Linux executable."""
    if platform.system() != 'Linux':
        return False, 'This bundled launch path requires Linux'
    machine = {'x86_64': 62, 'amd64': 62, 'aarch64': 183, 'arm64': 183}.get(architecture.lower())
    if machine is None:
        return False, 'Unsupported CPU architecture'
    if not path.is_file() or not os.access(path, os.X_OK):
        return False, 'Bundled llama.cpp server missing or not executable'
    with path.open('rb') as handle:
        header = handle.read(20)
    if len(header) != 20 or header[:6] != b'\x7fELF\x02\x01' or struct.unpack('<H', header[18:20])[0] != machine:
        return False, 'Bundled server ELF architecture does not match the host'
    return True, 'Matching Linux ELF64 ABI; instruction-set/runtime support checked on startup'


def detect_hardware(config, root=ROOT):
    memory, cpu = {}, {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            memory[key] = int(value.strip().split()[0]) * 1024
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                cpu[key.strip()] = value.strip()
    except (OSError, ValueError):
        pass
    total, available = memory.get('MemTotal', 0), memory.get('MemAvailable', 0)
    # Respect container memory ceilings as well as current host pressure.
    try:
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        used = int(Path('/sys/fs/cgroup/memory.current').read_text())
        if limit != 'max':
            total = min(total, int(limit))
            available = min(available, max(0, int(limit) - used))
    except (OSError, ValueError):
        pass
    architecture = platform.machine()
    selection = select_runtime(config, root)
    server = Path(selection['server'])
    compatible, reason = runtime_compatibility(server, architecture)
    gpus = []
    for device in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
        try:
            gpus.append({'vendor_id': (device / 'vendor').read_text().strip(),
                         'device_id': (device / 'device').read_text().strip()})
        except OSError:
            pass
    devices = ''
    if compatible:
        try:
            env = runtime_environment(root, server)
            probe = subprocess.run([str(server), '--list-devices'], env=env,
                                   capture_output=True, text=True, timeout=10)
            devices = (probe.stdout + probe.stderr)[-4000:]
            if probe.returncode:
                compatible, reason = False, 'Local server device probe failed (exit %s)' % probe.returncode
        except (OSError, subprocess.SubprocessError) as exc:
            compatible, reason = False, 'Local server device probe failed: ' + str(exc)
    gpu_supported = any(tag in devices.lower() for tag in ('cuda0:', 'vulkan0:', 'metal:', 'sycl0:', 'rocm0:'))
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    return dict(cpu=cpu.get('model name', cpu.get('Processor', platform.processor())),
                architecture=architecture, cores=cores, cpu_flags=cpu.get('flags', cpu.get('Features', '')),
                total_ram=total, available_ram=available, gpus=gpus,
                gpu_acceleration_supported=gpu_supported, runtime_devices=devices,
                backend='llama.cpp ' + selection['backend'] + (' (automatic VRAM offload)' if selection['gpu'] else ' (--n-gpu-layers 0)'), runtime_compatible=compatible,
                selected_server=str(server), runtime_probe_errors=selection['probe_errors'],
                runtime_reason=reason,
                gpu_policy='Automatic offload on a usable USB GPU runtime. Fit retains conservative system RAM bounds; VRAM is not added to RAM.')


def installed_models(root=ROOT):
    """Reuse split-file rules; confine symlinks and deduplicate canonical paths."""
    store = (root / 'models').resolve()
    paths = {}
    for candidate in sorted(store.rglob('*.gguf')):
        path = candidate.resolve()
        if path.is_file() and path.is_relative_to(store):
            paths[str(candidate.relative_to(store))] = path
    result, seen = [], set()
    for name in selectable_names(paths):
        path = paths[name]
        if path in seen:
            continue
        seen.add(path)
        try:
            group = shard_group(name, paths)
            error = 'Split GGUF memory estimation not supported in phase 1' if len(group) > 1 else None
        except ValueError as exc:
            error = str(exc)
        result.append((path, error))
    return result


def model_profile(path, root=ROOT):
    profiles_path = root / 'model-profiles.json'
    profiles = json.loads(profiles_path.read_text()) if profiles_path.exists() else {}
    matches = [value for key, value in profiles.items() if resolve(root, key) == path.resolve()]
    if len(matches) > 1 and any(item != matches[0] for item in matches[1:]):
        raise ValueError('Conflicting profiles for canonical model path')
    return matches[0] if matches else {}


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Missing/invalid ' + name)
    return value


def evaluate(path, error, config, hardware, root=ROOT):
    result = dict(path=str(path), name=path.name, classification='Unknown', eligible=False,
                  score=0, reasons=[], metadata={}, assumptions=[])
    try:
        if error:
            raise ValueError(error)
        meta = read_metadata(path)
        result['metadata'] = meta
        profile = model_profile(path, root)
        result['profile'] = profile
        result['hermes_compatibility'] = profile.get('llm_fit', {}).get('tool_calling', 'unknown')
        result['name'] = profile.get('model_label', meta.get('general.name', path.name))
        arch = meta.get('general.architecture')
        native = positive(meta.get(str(arch) + '.context_length'), 'native context')
        context = int(positive(profile.get('context_size', config['context_size']), 'configured context'))
        result.update(native_context=native, context=context, file_size=path.stat().st_size,
                      quantization=QUANTS.get(meta.get('general.file_type'), 'Unknown'))
        if context < 4096 or native < context:
            result.update(classification='Does Not Fit', reasons=['Native/configured context incompatible with Hermes requirement'])
            return result
        fit = profile.get('llm_fit', {})
        if meta.get('tensor_count', 0) <= 0:
            raise ValueError('No GGUF tensors')
        if meta.get('split.count', 0) > 1:
            raise ValueError('Split GGUF memory estimation not supported in phase 1')
        if arch not in DENSE_ARCHES and 'kv_bytes_per_token' not in fit:
            raise ValueError('Unsupported KV architecture; needs documented profile llm_fit.kv_bytes_per_token upper bound')
        if 'kv_bytes_per_token' in fit:
            if not fit.get('memory_evidence'):
                raise ValueError('Profile KV bound requires memory_evidence')
            kv_per_token = positive(fit['kv_bytes_per_token'], 'profile KV upper bound')
            result['assumptions'].append(fit['memory_evidence'])
        else:
            layers = positive(meta.get(arch + '.block_count'), 'block count')
            heads = positive(meta.get(arch + '.attention.head_count'), 'head count')
            kv_heads = positive(meta.get(arch + '.attention.head_count_kv', heads), 'KV head count')
            embedding = positive(meta.get(arch + '.embedding_length'), 'embedding length')
            key = positive(meta.get(arch + '.attention.key_length', embedding / heads), 'key length')
            value = positive(meta.get(arch + '.attention.value_length', embedding / heads), 'value length')
            kv_per_token = layers * kv_heads * (key + value) * 2  # default f16 K + V
            result['assumptions'].append('Full-attention f16 KV for every layer; no sliding-window discount')
        weights = path.stat().st_size
        overhead = max(GIB, weights * .20) + positive(fit.get('extra_overhead_bytes', 256 * 1024**2), 'extra runtime allowance')
        reserve = max(GIB, hardware['total_ram'] * .10)
        estimated = math.ceil(weights + context * kv_per_token + overhead)
        available = hardware['available_ram']
        result.update(estimated_memory=estimated, weight_memory=weights, kv_memory=context * kv_per_token,
                      runtime_overhead=overhead, safety_reserve=reserve, available_ram=available)
        result['assumptions'].append('All weights resident in RAM + max(1 GiB, 20% weights) runtime + extra allowance; no swap/VRAM credit')
        if not hardware['runtime_compatible']:
            result.update(classification='Does Not Fit', reasons=[hardware['runtime_reason']])
            return result
        if not available or not hardware['total_ram']:
            raise ValueError('Current available memory could not be determined')
        if estimated + reserve > available:
            result.update(classification='Does Not Fit', reasons=['Estimated RAM plus safety reserve exceeds current available RAM'])
            return result
        evidence = fit.get('tool_calling', 'unknown')
        result['hermes_compatibility'] = evidence
        if evidence == 'incompatible':
            result.update(classification='Does Not Fit', reasons=['Profile marks tool calling incompatible'])
            return result
        if evidence not in ('verified', 'experimental', 'unknown'):
            raise ValueError('Invalid tool compatibility evidence')
        if evidence != 'unknown' and not fit.get('capability_evidence'):
            raise ValueError('Tool suitability claim requires capability_evidence')
        if not meta.get('has_chat_template') and path.name not in config.get('chat_templates', {}):
            raise ValueError('No installed chat template evidence')
        if result['quantization'] == 'Unknown':
            raise ValueError('Unknown quantization')
        parameters = positive(meta.get('parameter_count_estimate'), 'tensor parameter estimate') / 1e9
        # Explainable heuristic points, not measured benchmarks or tok/s.
        factors = dict(capacity=min(25, 8 * math.log2(1 + parameters)),
                       tools={'verified': 35, 'experimental': 12, 'unknown': 0}[evidence],
                       context=min(10, 2 * math.log2(context / 4096 + 1)),
                       memory_headroom=15 * max(0, 1 - (estimated + reserve) / available),
                       cpu_responsiveness=10 / (1 + parameters / max(1, hardware['cores'] / 4)),
                       quantization=5 if result['quantization'] in ('F32', 'F16', 'BF16', 'Q8_0') else 3)
        result.update(classification='Good Fit' if estimated + 2 * reserve <= available else 'Marginal Fit',
                      eligible=True, score=round(sum(factors.values()), 6), factors=factors,
                      reasons=['Heuristic quality/context/tool evidence balanced against CPU cost and RAM headroom',
                               'Tool execution reliability is not established by the filename or template'])
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        result['reasons'].append(str(exc))
    return result


def rank_models(config, hardware, root=ROOT):
    rows = [evaluate(path, error, config, hardware, root) for path, error in installed_models(root)]
    from fit_performance import PerformanceStore, measurement_identity
    store = PerformanceStore(root)
    for row in rows:
        if not row['eligible']:
            continue
        try:
            active = dict(config, **row.get('profile', {}))
            active.update(model=row['path'], context_size=row['context'])
            observed = store.adjustment(measurement_identity(row['path'], active, hardware, root))
            row['performance'] = observed
            row['base_score'] = row['score']
            row['score'] = round(row['score'] + observed['points'], 6)
            row['factors']['measured_performance'] = observed['points']
        except (OSError, ValueError, TypeError, KeyError) as exc:
            row['performance'] = {'points': 0, 'warning': str(exc)}
    return sorted(rows, key=lambda row: (not row['eligible'], -row['score'], row['path']))
