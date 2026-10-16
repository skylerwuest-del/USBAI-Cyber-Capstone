import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import select_model as s


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'models').mkdir()
        self.entries = []
        for name, ram, priority in [('small', 1, 1), ('medium', 5, 2), ('large', 12, 3)]:
            key = b'llama.context_length'
            gguf = b'GGUF' + struct.pack('<IQQQ', 3, 1, 1, len(key)) + key + struct.pack('<II', 4, 8192)
            gguf += struct.pack('<Q', 6) + b'weight' + struct.pack('<IQIQ', 1, 1, 0, 0)
            gguf += b'\0' * (-len(gguf) % 32) + struct.pack('<f', 1.0)
            path = self.root / 'models' / (name + '.gguf')
            path.write_bytes(gguf)
            self.entries.append(dict(name=name, file='models/' + path.name, parameters_b=ram,
                                     quantization='Q4_K_M', context_size=4096, estimated_ram_gb=ram,
                                     kv_bytes_per_token=65536, priority=priority,
                                     sha256=hashlib.sha256(gguf).hexdigest()))
        self.save()

    def save(self):
        (self.root / 'models/models.json').write_text(json.dumps({'models': self.entries}))

    def hardware(self, ram):
        # Fixture mirrors upstream display_json_system_with_providers, not help prose.
        return s.parse_hardware({'system': {'total_ram_gb': ram, 'available_ram_gb': ram,
                 'cpu_cores': 4, 'cpu_name': 'fixture', 'has_gpu': False, 'gpu_vram_gb': None,
                 'gpu_name': None, 'gpu_count': 0, 'unified_memory': False, 'backend': 'CPU', 'gpus': []},
                 'providers': {'llama.cpp': {'available': False}}})

    def test_memory_tiers(self):
        for ram, expected in [(4, 'small'), (12, 'medium'), (32, 'large')]:
            with self.subTest(ram=ram):
                result = s.choose(s.load_models(self.root), self.hardware(ram))
                self.assertEqual(result['model_label'], expected)
                self.assertLessEqual(result['estimated_required_gib'], result['memory_budget_gib'])

    def test_missing_file(self):
        (self.root / 'models/large.gguf').unlink()
        self.assertEqual(len(s.load_models(self.root)), 2)

    def test_malformed_json(self):
        (self.root / 'models/models.json').write_text('{')
        with self.assertRaises(ValueError):
            s.load_models(self.root)

    def test_invalid_metadata_and_checksum(self):
        (self.root / 'models/large.gguf').write_bytes(b'not gguf')
        self.assertEqual(len(s.load_models(self.root)), 2)
        self.entries[0]['sha256'] = '0' * 64
        self.save()
        self.assertEqual(len(s.load_models(self.root)), 1)

    def test_truncated_tensor_table_even_with_matching_digest(self):
        path = self.root / 'models/small.gguf'
        data = path.read_bytes()[:70]
        path.write_bytes(data)
        self.entries[0]['sha256'] = hashlib.sha256(data).hexdigest()
        self.save()
        self.assertEqual(len(s.load_models(self.root)), 2)

    def test_main_reports_clean_failure(self):
        with patch.object(s, 'select', side_effect=ValueError('No safe local model')), patch.object(sys, 'argv', ['select_model.py']):
            self.assertEqual(s.main(), 1)

    def test_wrong_architecture(self):
        binary = self.root / 'llmfit'
        binary.write_bytes(b'\x7fELF\x02\x01' + b'\0' * 12 + struct.pack('<H', 183))
        binary.chmod(0o755)
        with self.assertRaises(ValueError):
            s.validate_elf(binary, 'x86_64')
        s.validate_elf(binary, 'arm64')

    def test_no_fit(self):
        with self.assertRaisesRegex(ValueError, 'No safe'):
            s.choose(s.load_models(self.root), self.hardware(2))

    def test_failure_fallback(self):
        with patch.object(s, 'validate_elf', side_effect=ValueError('test execution failure')):
            hw = s.detect_hardware(self.root, 'x86_64')
        self.assertEqual(hw['source'], '/proc/meminfo fallback')
        self.assertFalse(hw['has_gpu'])

    def test_execution_failure_fallback(self):
        binary_dir = self.root / 'llmfit/linux/x86_64'
        binary_dir.mkdir(parents=True)
        (binary_dir / 'version.txt').write_text('llmfit test')
        with patch.object(s, 'validate_elf'), patch.object(s, 'isolated_run', side_effect=subprocess.TimeoutExpired('llmfit', 20)):
            self.assertFalse(s.detect_hardware(self.root, 'x86_64')['has_gpu'])

    def test_traversal_and_symlinks(self):
        for bad in ['../outside.gguf', '/tmp/outside.gguf', 'models/../../outside.gguf']:
            self.entries[0]['file'] = bad
            self.save()
            self.assertEqual(len(s.load_models(self.root)), 2)
        outside = self.root / 'outside.gguf'
        outside.write_bytes(b'GGUF')
        (self.root / 'models/escape.gguf').symlink_to(outside)
        self.entries[0]['file'] = 'models/escape.gguf'
        self.save()
        self.assertEqual(len(s.load_models(self.root)), 2)

    def test_no_gpu_memory_added(self):
        hw = self.hardware(4)
        hw.update(has_gpu=True, gpu_vram_gb=80, unified_memory=True)
        self.assertEqual(s.choose(s.load_models(self.root), hw)['model_label'], 'small')

    def test_explicit_request_never_substituted(self):
        with self.assertRaises(ValueError):
            s.choose(s.load_models(self.root), self.hardware(4), self.root / 'models/large.gguf')

    def test_selection_does_not_use_sockets(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('network attempted')):
            s.choose(s.load_models(self.root), self.hardware(12))

    def test_real_seccomp_blocks_network_across_exec(self):
        # Real kernel test: both direct and descendant sockets must fail with EPERM.
        code = '''import socket, subprocess, sys
try:
 socket.socket()
except PermissionError:
 pass
else:
 raise SystemExit('socket policy bypassed')
if len(sys.argv) == 1:
 subprocess.run([sys.executable, '-c', sys.argv[0] if False else "import socket; socket.socket()"], check=False)
'''
        out = s.isolated_run(Path(sys.executable), ['-c', code, 'child'])
        self.assertEqual(out, '')

    def test_descendant_network_denied(self):
        child = "import socket; socket.socket(socket.AF_INET, socket.SOCK_DGRAM)"
        code = ('import subprocess, sys; p = subprocess.run([sys.executable, "-c", ' +
                repr(child) + '], capture_output=True); '
                'assert p.returncode != 0 and b"PermissionError" in p.stderr')
        self.assertEqual(s.isolated_run(Path(sys.executable), ['-c', code]), '')

    def test_invalid_hardware_schema(self):
        for data in [None, [], {}, {'system': []}, {'system': {'total_ram_gb': True}},
                     {'system': {'total_ram_gb': 8, 'available_ram_gb': float('nan')}}]:
            with self.subTest(data=data), self.assertRaises((ValueError, KeyError)):
                s.parse_hardware(data)

    def test_server_arguments_cannot_change_selection(self):
        self.assertEqual(s.safe_server_args(['--reasoning-budget', '0']), ['--reasoning-budget', '0'])
        for args in [['-m', 'other.gguf'], ['--model=other.gguf'], ['--ctx-size', '999999'],
                     ['--hf-repo', 'remote'], ['--gpu-layers', '99'], None]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                s.safe_server_args(args)

    def test_manifest_symlink_escape(self):
        manifest = self.root / 'models/models.json'
        outside = self.root / 'outside.json'
        manifest.rename(outside)
        manifest.symlink_to(outside)
        with self.assertRaises(ValueError):
            s.load_models(self.root)

    def test_unsupported_architecture(self):
        with patch.object(s.platform, 'machine', return_value='riscv64'):
            with self.assertRaises(ValueError):
                s.architecture()

    def test_llmfit_json_command(self):
        directory = self.root / 'llmfit/linux/x86_64'
        directory.mkdir(parents=True)
        (directory / 'version.txt').write_text('llmfit fixture')
        raw = json.dumps({'system': {'total_ram_gb': 12, 'available_ram_gb': 10}})
        with patch.object(s, 'validate_elf'), patch.object(s, 'isolated_run', side_effect=['llmfit fixture', raw]) as run:
            self.assertEqual(s.detect_hardware(self.root, 'x86_64')['source'], 'llmfit')
            self.assertEqual(run.call_args.args[1], ['--no-dashboard', 'system', '--json'])


if __name__ == '__main__':
    unittest.main()
