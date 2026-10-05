import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import setup


class DownloadDurabilityTests(unittest.TestCase):
    def test_complete_partial_is_verified_and_published_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            partial = path.with_name(path.name + ".part")
            content = b"complete-model-data"
            partial.write_bytes(content)
            asset = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest(), "url": "unused"}
            with patch.object(setup.subprocess, "run") as curl, contextlib.redirect_stdout(io.StringIO()):
                setup.download(asset, path)
            curl.assert_not_called()
            self.assertEqual(path.read_bytes(), content)
            self.assertFalse(partial.exists())

    def test_flush_error_preserves_existing_destination_and_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            previous = b"previous"
            path.write_bytes(previous)
            partial = path.with_name(path.name + ".part")
            content = b"complete-model-data"
            partial.write_bytes(content)
            asset = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest(), "url": "unused"}
            with patch.object(setup.os, "fsync", side_effect=OSError("writeback failed")):
                with self.assertRaisesRegex(OSError, "writeback failed"):
                    setup.download(asset, path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(partial.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
