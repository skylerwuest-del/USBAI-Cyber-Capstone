import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from setup import download


class DownloadTests(unittest.TestCase):
    def exercise(self, honor_range):
        data = bytes(range(256)) * 4096
        ranges = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                ranges.append(self.headers.get('Range'))
                offset = int(self.headers['Range'].partition('=')[2].partition('-')[0]) if self.headers.get('Range') and honor_range else 0
                self.send_response(206 if offset else 200)
                self.send_header('Content-Length', str(len(data) - offset))
                if offset:
                    self.send_header('Content-Range', 'bytes %d-%d/%d' % (offset, len(data)-1, len(data)))
                self.end_headers()
                self.wfile.write(data[offset:])
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                destination = Path(folder) / 'model.gguf'
                destination.with_suffix('.gguf.part').write_bytes(data[:65536])
                asset = {'url': 'http://127.0.0.1:%d/weights' % server.server_port,
                         'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                with patch('setup.shutil.which', return_value=None):
                    download(asset, destination)
                    self.assertEqual(destination.read_bytes(), data)
                    count = len(ranges)
                    download(asset, destination)
                    self.assertEqual(len(ranges), count, 'Verified cached file should require no request')
                self.assertEqual(ranges[0], 'bytes=65536-')
                self.assertFalse(destination.with_suffix('.gguf.part').exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_resumes_without_host_curl(self):
        self.exercise(True)

    def test_server_ignores_range_without_corrupting_download(self):
        self.exercise(False)


if __name__ == '__main__':
    unittest.main()
