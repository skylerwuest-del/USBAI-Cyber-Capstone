"""Append-only local diagnostics. Never pass prompts or tool/session content here."""
import datetime
import fcntl
import json
import os
import time
import uuid
from pathlib import Path

from usb_core import ROOT


class DiagnosticRun:
    @classmethod
    def attach(cls, run_id, root=ROOT):
        result = cls.__new__(cls)
        result.path = Path(root) / 'logs/llm-fit-diagnostics.txt'
        result.run_id = run_id
        return result

    def __init__(self, category='LLM FIT DIAGNOSTIC RUN', root=ROOT):
        self.path = Path(root) / 'logs/llm-fit-diagnostics.txt'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.write(category, run_id=self.run_id,
                   timestamp=datetime.datetime.now().astimezone().isoformat(),
                   application_version='CapStone prototype1 / LLM Fit phase 1')

    def write(self, category, **values):
        text = '\n' + '=' * 60 + '\n[' + category + '] Run ID: ' + self.run_id + '\n'
        text += json.dumps(values, indent=2, ensure_ascii=False, default=str, allow_nan=False) + '\n'
        with self.path.open('a', encoding='utf-8') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    def finish(self, outcome, **values):
        self.write('FINAL RESULT / END OF RUN', outcome=outcome,
                   elapsed_seconds=round(time.monotonic() - self.started, 3), **values)
