import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import import_models as importer
from scripts.select_model import load_models


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.drop = self.root / 'models/drop-in'
        self.drop.mkdir(parents=True)
        self.manifest = self.root / 'models/models.json'
        self.manifest.write_text('{"models": []}')

    def model(self, arch='llama'):
        def string(value):
            raw = value.encode()
            return struct.pack('<Q', len(raw)) + raw
        fields = {'general.architecture': arch, arch + '.context_length': 8192,
                  arch + '.block_count': 32, arch + '.embedding_length': 4096,
                  'general.file_type': 15}
        raw = b'GGUF' + struct.pack('<IQQ', 3, 1, len(fields))
        for key, value in fields.items():
            raw += string(key)
            raw += struct.pack('<I', 8) + string(value) if isinstance(value, str) else struct.pack('<II', 4, value)
        raw += string('weight') + struct.pack('<IQIQ', 1, 1, 0, 0)
        raw += b'\0' * (-len(raw) % 32) + struct.pack('<f', 1)
        path = self.drop / 'example.gguf'
        path.write_bytes(raw)
        return path

    def test_import_is_eligible_and_idempotent(self):
        self.model()
        importer.sync(self.root)
        entries = load_models(self.root)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['context_size'], 4096)
        self.assertEqual(entries[0]['kv_bytes_per_token'], 32 * 4096 * 16)
        before = self.manifest.read_bytes()
        importer.sync(self.root)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_removed_file_removes_generated_entry(self):
        path = self.model()
        importer.sync(self.root)
        path.unlink()
        importer.sync(self.root)
        self.assertEqual(json.loads(self.manifest.read_text())['models'], [])

    def test_manual_entry_preserved(self):
        entry = {'name': 'manual', 'file': 'models/manual.gguf', 'priority': 100}
        self.manifest.write_text(json.dumps({'models': [entry]}))
        importer.sync(self.root)
        self.assertEqual(json.loads(self.manifest.read_text())['models'], [entry])

    def test_unsupported_and_corrupt_files_skipped(self):
        path = self.model('qwen35')
        importer.sync(self.root)
        self.assertEqual(json.loads(self.manifest.read_text())['models'], [])
        path.write_bytes(b'GGUF')
        importer.sync(self.root)
        self.assertEqual(json.loads(self.manifest.read_text())['models'], [])

    def test_symlink_and_malformed_manifest(self):
        (self.drop / 'escape.gguf').symlink_to(self.root / 'outside.gguf')
        importer.sync(self.root)
        self.assertEqual(json.loads(self.manifest.read_text())['models'], [])
        self.manifest.write_text('{')
        with self.assertRaises(ValueError):
            importer.sync(self.root)
        self.assertEqual(self.manifest.read_text(), '{')


if __name__ == '__main__':
    unittest.main()
