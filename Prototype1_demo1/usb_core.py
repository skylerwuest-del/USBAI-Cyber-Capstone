import argparse

import contextlib

import datetime

import fcntl

import hashlib

import json

import os

from pathlib import Path

import signal

import socket

import subprocess

import sys

import tempfile

import threading

import time

import urllib.error

import urllib.request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'hermes' / 'site-packages'))

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.checkpoint-', delete=False) as handle:
        temp = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)

def resolve(root, path):
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()

@contextlib.contextmanager
def instance_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another CapStone runner is already using this USB') from None
        yield

@contextlib.contextmanager
def shutdown_signals():
    """Repeated Ctrl+C must not interrupt cleanup between worker and server shutdown."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    saved = {number: signal.signal(number, signal.SIG_IGN)
             for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for number, handler in saved.items():
            signal.signal(number, handler)

def stop_process(process):
    if process is None:
        return
    descendants = []
    if process.poll() is None:
        try:
            import psutil
            descendants = psutil.Process(process.pid).children(recursive=True)
        except Exception:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    for child in reversed(descendants):
        try:
            child.kill()
        except Exception:
            pass

def run_shell(command, cwd, timeout, limit):
    process = subprocess.Popen(['bash', '-c', command], cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    tail = bytearray()
    total = 0
    capacity = limit * 4

    def consume():
        nonlocal total
        try:
            while True:
                block = os.read(process.stdout.fileno(), 8192)
                if not block:
                    break
                total += len(block)
                tail.extend(block)
                if len(tail) > capacity:
                    del tail[:-capacity]
        except (OSError, ValueError):
            pass
    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        stop_process(process)
        reader.join(timeout=3)
        process.stdout.close()
    decoded = bytes(tail).decode('utf-8', errors='replace')
    return {'exit_code': process.returncode, 'timed_out': timed_out, 'output': decoded[-limit:], 'output_truncated': total > capacity or len(decoded) > limit}

def chat_template_args(config):
    templates = config.get('chat_templates', {})
    if not isinstance(templates, dict) or not all((isinstance(k, str) and isinstance(v, str) for k, v in templates.items())):
        raise ValueError('chat_templates must map GGUF filenames to template paths')
    template = templates.get(Path(config['model']).name)
    if not template:
        return []
    path = resolve(ROOT, template)
    if not path.is_file():
        raise FileNotFoundError('Chat template missing: ' + str(path))
    return ['--chat-template-file', str(path)]

class Server:

    def __init__(self, config):
        self.config = config
        self.process = None
        self.log = None
        self.base = 'http://127.0.0.1:' + str(config['port'])
        self.client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def healthy(self):
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            with self.client.open(self.base + '/health', timeout=3) as r:
                return r.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def ensure(self):
        if self.healthy():
            return
        self.close()
        cfg = self.config
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('127.0.0.1', cfg['port']))
            except OSError:
                raise RuntimeError('Configured server port is occupied; choose a free port in config.json') from None
        server = resolve(ROOT, cfg['server'])
        model = resolve(ROOT, cfg['model'])
        threads = cfg['threads'] or max(1, (os.cpu_count() or 2) // 2)
        airllm = cfg.get('backend') == 'airllm'
        if airllm:
            args = [str(ROOT / 'hermes/python/bin/python3.12'), '-s', '-B', str(ROOT / 'airllm_server.py'),
                    '--model', str(model), '--port', str(cfg['port']),
                    '--context', str(cfg['context_size']), '--max-tokens', str(cfg['max_tokens']),
                    '--cache-gb', str(cfg.get('cache_gb', 8))]
        else:
            args = [str(server), '--model', str(model), '--host', '127.0.0.1', '--port', str(cfg['port']), '--ctx-size', str(cfg['context_size']), '--threads', str(threads), '--n-gpu-layers', '0', '--parallel', '1', '--jinja', '--alias', 'capstone'] + chat_template_args(cfg) + cfg['server_args'] + ['--no-webui']
        env = dict(os.environ, LLAMA_CACHE=str(ROOT / 'state' / 'llama-cache'))
        env['LD_LIBRARY_PATH'] = str(server.parent) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        log_name = 'airllm-server.log' if airllm else 'server.log'
        self.log = (ROOT / 'logs' / log_name).open('ab')
        try:
            self.process = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log, start_new_session=True)
        except BaseException:
            self.close()
            raise
        print('Starting USB ' + ('AirLLM' if airllm else 'llama.cpp') + ' model server...', flush=True)
        end = time.monotonic() + cfg['startup_timeout']
        while time.monotonic() < end:
            if self.healthy():
                print('Model ready at ' + self.base, flush=True)
                return
            if self.process.poll() is not None:
                raise RuntimeError('Model server exited; see logs/' + log_name)
            time.sleep(1)
        self.close()
        raise RuntimeError('Model startup timed out; see logs/' + log_name)

    def close(self):
        stop_process(self.process)
        self.process = None
        if self.log:
            self.log.close()
            self.log = None
