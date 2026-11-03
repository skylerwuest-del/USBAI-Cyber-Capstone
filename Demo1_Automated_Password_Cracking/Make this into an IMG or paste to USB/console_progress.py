"""Owned, line-based progress for a terminal SDK session. No animation threads."""
import json
from pathlib import Path
import threading
import time
import urllib.request


class StopRequested(BaseException):
    """A process stop must escape the SDK's ordinary retry/interrupt handling."""


class ConsoleProgress:
    def __init__(self, base_url, backend, root, interval=15, emit=None):
        self.base = base_url.removesuffix('/v1')
        self.backend, self.root, self.interval = backend, Path(root), interval
        self.emit = emit or (lambda value: print(value, flush=True))
        self.done, self.lock = threading.Event(), threading.Lock()
        self.waiting, self.started, self.thread = False, 0, None
        self.client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def thinking(self, message):
        with self.lock:
            if self.done.is_set():
                return
            if message and not self.waiting:
                self.started = time.monotonic()
                self.emit('[model] Processing on CPU; Ctrl+C stops the session.')
            self.waiting = bool(message)

    def detail(self):
        if self.backend == 'airllm':
            path = self.root / 'logs/airllm-server.log'
            with path.open('rb') as handle:
                handle.seek(max(0, path.stat().st_size - 8192))
                lines = handle.read().decode(errors='replace').splitlines()
            for line in reversed(lines):
                if 'AirLLM generated ' in line:
                    return line.strip()
                if 'AirLLM request:' in line:
                    return 'AirLLM is processing the prompt'
            return ''
        with self.client.open(self.base + '/slots', timeout=1) as response:
            slots = json.load(response)
        for slot in slots:
            if slot.get('is_processing'):
                processed = slot.get('n_prompt_tokens_processed')
                if processed is not None:
                    return str(processed) + ' prompt tokens processed'
                return 'model server is processing the request'
        return ''

    def _run(self):
        while not self.done.wait(self.interval):
            with self.lock:
                if not self.waiting:
                    continue
                started = self.started
                if time.monotonic() - started < self.interval:
                    continue
            try:
                detail = self.detail()
            except (OSError, ValueError, KeyError, TypeError):
                detail = ''
            with self.lock:
                # A late poll must not print after cancellation or the next turn.
                if not self.done.is_set() and self.waiting and self.started == started:
                    elapsed = int(time.monotonic() - started)
                    self.emit('[model] ' + str(elapsed) + 's elapsed' + ('; ' + detail if detail else ''))

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def __exit__(self, *args):
        self.close()
