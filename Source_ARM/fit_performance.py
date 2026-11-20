"""Local, numeric-only observations. Never read prompts, responses, or tool outputs."""
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading
import time

from usb_core import ROOT, atomic_json, resolve
from hardware_runtime import cpu_threads

MAX_SAMPLES = 64
MAX_KEYS = 128
MAX_AGE = 30 * 86400
TIMING = re.compile(r'^(?:[\d:.]+\s+[A-Z]\s+)?slot\s+print_timings?\s*:[^\r\n]*?'
                    r'(prompt eval time|(?<!prompt )eval time)\s*=\s*([\d.]+) ms\s*/\s*(\d+) (?:tokens|runs)\b'
                    r'(?:\s*\([^\r\n]*?,\s*([\d.]+) tokens per second\))?', re.MULTILINE)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_identity(path):
    """Cheap cache invalidation, not a full model integrity checksum."""
    path = Path(path)
    stat = path.stat()
    with path.open('rb') as handle:
        header = hashlib.sha256(handle.read(65536)).hexdigest()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, header_sha256=header)


def hardware_identity(hardware):
    # Available RAM is intentionally excluded; fit always reevaluates it separately.
    if not hardware.get('cpu') or not hardware.get('architecture'):
        raise ValueError('Hardware identity unavailable; performance reuse disabled')
    return digest({key: hardware.get(key) for key in
                   ('cpu', 'architecture', 'cores', 'cpu_flags', 'total_ram', 'backend', 'gpus')})


def measurement_identity(path, config, hardware, root=ROOT):
    template = config.get('chat_templates', {}).get(Path(path).name)
    server = resolve(root, hardware.get('selected_server', config['server']))
    return dict(hardware=hardware_identity(hardware), model=file_identity(path),
                runtime=file_identity(server) if server.is_file() else None,
                context=int(config['context_size']),
                threads=config.get('threads') or cpu_threads(),
                server_args=config.get('server_args', []),
                batch_size=config.get('batch_size', 512), ubatch_size=config.get('ubatch_size', 512),
                prompt_policy='direct-compact-interactive-tools-v2',
                template=file_identity(resolve(root, template)) if template else None,
                acceleration=config.get('acceleration', 'auto'),
                gpu_reserve_mib=config.get('gpu_reserve_mib', 1024),
                backend=hardware.get('backend', 'llama.cpp CPU (--n-gpu-layers 0)'))


class PerformanceStore:
    def __init__(self, root=ROOT):
        self.path = Path(root) / 'state/llm-fit-performance.json'

    def read(self):
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text())
        if not isinstance(value, dict) or value.get('version') != 1 or not isinstance(value.get('records'), dict):
            raise ValueError('Unsupported or corrupt performance cache; no measurements reused')
        if any(not isinstance(entry, dict) or not isinstance(entry.get('events'), list)
               for entry in value['records'].values()):
            raise ValueError('Corrupt performance record; no measurements reused')
        return value['records']

    def append(self, identity, events):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = self.read()  # Preserve corrupt data rather than overwrite it.
            key = digest(identity)
            entry = records.setdefault(key, {'identity': identity, 'events': []})
            if not isinstance(entry, dict) or not isinstance(entry.get('events'), list):
                raise ValueError('Corrupt performance record')
            entry['events'] = (entry['events'] + events)[-MAX_SAMPLES:]
            entry['updated'] = time.time()
            if len(records) > MAX_KEYS:
                records = dict(sorted(records.items(), key=lambda item: item[1].get('updated', 0))[-MAX_KEYS:])
            atomic_json(self.path, {'version': 1, 'records': records})

    def adjustment(self, identity, now=None):
        now = time.time() if now is None else now
        entry = self.read().get(digest(identity), {})
        if not isinstance(entry, dict) or not isinstance(entry.get('events', []), list):
            raise ValueError('Corrupt performance record')
        events = [event for event in entry.get('events', [])
                  if isinstance(event, dict) and isinstance(event.get('timestamp'), (int, float))
                  and 0 <= now - event['timestamp'] <= MAX_AGE]
        def numeric(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
        speeds = [event['tokens_per_second'] for event in events
                  if event.get('kind') == 'generation' and numeric(event.get('tokens')) and event['tokens'] >= 8
                  and numeric(event.get('tokens_per_second'))]
        loads = [event['seconds'] for event in events if event.get('kind') == 'load' and numeric(event.get('seconds'))]
        failure_streak = 0
        for event in reversed(events):
            if event.get('kind') == 'load':
                break
            if event.get('kind') in ('server_failure', 'runtime_failure'):
                failure_streak += 1
        # Small bounded preference: never changes eligibility or claims agent quality.
        speed = max(-6, min(6, 3 * math.log2(statistics.median(speeds) / 5))) if len(speeds) >= 3 else 0
        load = -min(2, statistics.median(loads) / 300) if len(loads) >= 2 else 0
        failure = -min(6, failure_streak * 2)
        return dict(points=round(max(-10, min(6, speed + load + failure)), 6),
                    generation_samples=len(speeds), load_samples=len(loads),
                    median_tokens_per_second=statistics.median(speeds) if speeds else None,
                    median_load_seconds=statistics.median(loads) if loads else None,
                    consecutive_failures=failure_streak,
                    note='Same hardware/model/runtime/context/settings; speed needs 3 samples of >=8 tokens; no tool-quality inference')


def timing_samples(text):
    """Extract only counts/durations from llama.cpp timing lines; discard all text."""
    result = []
    for match in TIMING.finditer(text):
        milliseconds, tokens = float(match[2]), int(match[3])
        if not math.isfinite(milliseconds) or milliseconds <= 0 or tokens <= 0:
            continue
        result.append(dict(kind='prompt_processing' if match[1].startswith('prompt') else 'generation',
                           tokens=tokens, seconds=milliseconds / 1000,
                           tokens_per_second=float(match[4]) if match[4] else tokens * 1000 / milliseconds,
                           timestamp=time.time()))
    return result


class SessionPerformance:
    """Passive owned-server log sampler; no benchmark requests during normal startup."""
    def __init__(self, path, config, hardware, report, root=ROOT):
        self.root, self.report = Path(root), report
        self.identity = measurement_identity(path, config, hardware, root)
        self.store = PerformanceStore(root)
        self.log = self.root / 'logs/server.log'
        self.offset = self.log.stat().st_size if self.log.exists() else 0
        self.inode = self.log.stat().st_ino if self.log.exists() else None
        self.pending = b''
        self.stop = threading.Event()
        self.thread = None

    def event(self, kind, **numbers):
        event = dict(kind=kind, timestamp=time.time(), **numbers)
        self.record([event])

    def record(self, events):
        try:
            self.store.append(self.identity, events)
            self.report.write('MEASURED PERFORMANCE', identity=self.identity, observations=events,
                              note='Numeric runtime observations; no prompts/responses or tool reliability claims')
        except (OSError, ValueError, TypeError) as exc:
            with contextlib.suppress(OSError, ValueError, TypeError):
                self.report.write('PERFORMANCE WARNING', reason=str(exc), effect='Observation not reused')

    def poll(self):
        try:
            stat = self.log.stat()
            if self.inode is not None and (stat.st_ino != self.inode or stat.st_size < self.offset):
                # Do not attribute a replacement/truncated log to this server.
                return
            self.inode = stat.st_ino
            with self.log.open('rb') as handle:
                handle.seek(self.offset)
                block = handle.read(1024 * 1024)
                self.offset = handle.tell()
            lines = (self.pending + block).split(b'\n')
            self.pending = lines.pop()[-65536:]
            events = timing_samples(b'\n'.join(lines).decode('utf-8', errors='replace'))
            if events:
                self.record(events)
        except (OSError, ValueError, TypeError):
            pass  # Telemetry must not interfere with chat.

    def start(self):
        # Exclude loader/warmup timing lines: only observe work after readiness.
        if self.log.exists():
            stat = self.log.stat()
            self.offset, self.inode, self.pending = stat.st_size, stat.st_ino, b''
        def sample():
            while not self.stop.wait(2):
                self.poll()
        self.thread = threading.Thread(target=sample, name='fit-performance', daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join()
        self.poll()


def create_tracker(path, config, hardware, report, root=ROOT):
    try:
        return SessionPerformance(path, config, hardware, report, root)
    except (OSError, ValueError, TypeError) as exc:
        with contextlib.suppress(OSError, ValueError, TypeError):
            report.write('PERFORMANCE WARNING', reason=str(exc), effect='Tracking unavailable; chat unaffected')
        return None
