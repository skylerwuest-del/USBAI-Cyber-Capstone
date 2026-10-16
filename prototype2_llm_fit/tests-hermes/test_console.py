import io
import os
from pathlib import Path
import signal
import struct
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from console_progress import ConsoleProgress, StopRequested
from gguf_metadata import context_length
from usb_core import shutdown_signals


class ConsoleTests(unittest.TestCase):
    def test_repeated_interrupts_do_not_abort_cleanup_and_handlers_are_restored(self):
        received = []
        signals = (signal.SIGINT, signal.SIGTERM)
        saved = {number: signal.signal(number, lambda number, frame: received.append(number)) for number in signals}
        try:
            with self.assertRaises(ValueError):
                with shutdown_signals():
                    for number in signals:
                        os.kill(os.getpid(), number)
                    raise ValueError('cleanup error still restores handlers')
            self.assertEqual(received, [])
            os.kill(os.getpid(), signal.SIGINT)
            self.assertEqual(received, [signal.SIGINT])
        finally:
            for number, handler in saved.items():
                signal.signal(number, handler)

    def test_cleanup_in_a_worker_thread_does_not_try_to_replace_signal_handlers(self):
        completed = []
        def cleanup():
            with shutdown_signals():
                completed.append(True)
        thread = threading.Thread(target=cleanup)
        thread.start()
        thread.join(timeout=1)
        self.assertEqual(completed, [True])

    def test_status_thread_stops_and_late_callbacks_cannot_restart_output(self):
        lines, heartbeat = [], threading.Event()
        def emit(line):
            lines.append(line)
            if 'elapsed' in line:
                heartbeat.set()
        progress = ConsoleProgress('http://127.0.0.1:1/v1', 'llama.cpp', '.', interval=0.01, emit=emit)
        progress.detail = lambda: 'server working'
        with progress:
            progress.thinking('processing')
            self.assertTrue(heartbeat.wait(1))
        count = len(lines)
        self.assertFalse(progress.thread.is_alive())
        progress.thinking('initializing again')
        self.assertEqual(len(lines), count)
        self.assertTrue(all('\r' not in line and '🌑' not in line for line in lines))

    def test_stop_escapes_ordinary_sdk_exception_handlers(self):
        caught = []
        with self.assertRaises(StopRequested):
            try:
                raise StopRequested()
            except (Exception, KeyboardInterrupt):
                caught.append(True)
        self.assertEqual(caught, [])


class GGUFTests(unittest.TestCase):
    def fixture(self, value, truncate=False):
        def string(text):
            raw = text.encode()
            return struct.pack('<Q', len(raw)) + raw
        data = b'GGUF' + struct.pack('<IQQ', 3, 0, 2)
        data += string('tokenizer.tokens') + struct.pack('<IIQ', 9, 8, 2) + string('one') + string('two')
        data += string('qwen2.context_length') + struct.pack('<II', 4, value)
        return data[:-1] if truncate else data

    def test_native_context_is_read_past_variable_length_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.gguf'
            path.write_bytes(self.fixture(8192))
            self.assertEqual(context_length(path), 8192)

    def test_truncated_native_context_and_invalid_headers_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.gguf'
            for data in (self.fixture(8192, truncate=True), b'not a GGUF file'):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    context_length(path)
