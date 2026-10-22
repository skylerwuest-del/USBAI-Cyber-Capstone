"""Preparation tests use temporary archives and mocked downloads/processes."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup
import setup_hermes
import prepare


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'state').mkdir()
        (self.root / 'logs').mkdir()
        (self.root / 'config.json').write_text('{}')

    def test_unreadable_settings_fail_before_downloads_with_ownership_guidance(self):
        with patch.object(Path, 'open', side_effect=PermissionError('private file')), \
             patch.object(prepare.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Restore ownership with: sudo chown'):
                prepare.check_settings_access(self.root)
        run.assert_not_called()

    def test_settings_preflight_preserves_existing_content(self):
        config = self.root / 'config.json'
        config.write_text('{"model": "models/keep.gguf"}\n')
        before = config.read_bytes()
        prepare.check_settings_access(self.root)
        self.assertEqual(config.read_bytes(), before)

    def test_missing_legacy_library_archive_is_created_and_reused_offline(self):
        runtime = self.root / 'runtime'
        runtime.mkdir()
        (runtime / 'llama-server').write_text('fixture')
        source = self.root / 'host'
        source.mkdir()
        libs = ['libstdc++.so.6', 'libgcc_s.so.1', 'libgomp.so.1', 'libssl.so.3', 'libcrypto.so.3']
        for name in libs:
            (source / name).write_bytes(('synthetic ' + name).encode())
        inspection = '\n'.join(name + ' => ' + str(source / name) + ' (0x1234)' for name in libs)
        legacy = dict(file='state/ubuntu-runtime-libs.tar.gz', size=1, sha256='unknown')
        with patch.object(setup, 'ROOT', self.root), \
             patch.object(setup.subprocess, 'run', return_value=Mock(stdout=inspection)), \
             contextlib.redirect_stdout(io.StringIO()):
            setup.prepare_runtime_libraries(legacy)
        descriptor = json.loads((self.root / 'state/ubuntu-runtime-libs.local.json').read_text())
        self.assertNotEqual(descriptor['sha256'], legacy['sha256'])
        for name in libs:
            self.assertEqual((runtime / name).read_bytes(), (source / name).read_bytes())
        with patch.object(setup, 'ROOT', self.root), \
             patch.object(setup.subprocess, 'run', side_effect=AssertionError('offline repair must use verified cache')), \
             patch.object(setup, 'download', side_effect=AssertionError('network forbidden')), \
             contextlib.redirect_stdout(io.StringIO()):
            setup.prepare_runtime_libraries(legacy, offline=True)

    def test_missing_or_corrupt_offline_library_cache_has_actionable_error(self):
        asset = dict(file='state/missing.tar.gz', size=1, sha256='invalid')
        with patch.object(setup, 'ROOT', self.root), self.assertRaisesRegex(RuntimeError, 'bash setup.sh online'):
            setup.prepare_runtime_libraries(asset, offline=True)

    def test_unsafe_library_archive_rejected(self):
        archive = self.root / 'state/libraries.tar.gz'
        with tarfile.open(archive, 'w:gz') as handle:
            member = tarfile.TarInfo('../unsafe')
            member.size = 1
            handle.addfile(member, io.BytesIO(b'x'))
        asset = dict(file='state/libraries.tar.gz', size=archive.stat().st_size, sha256=setup.sha256(archive))
        with patch.object(setup, 'ROOT', self.root), self.assertRaises(ValueError):
            setup.restore_ubuntu_libraries(asset)
        self.assertFalse((self.root.parent / 'unsafe').exists())

    def test_offline_asset_verification_never_downloads(self):
        path = self.root / 'cache'
        path.write_bytes(b'cached')
        asset = dict(size=6, sha256=setup.sha256(path))
        with patch.object(setup, 'download', side_effect=AssertionError('network forbidden')):
            setup.ensure_asset(asset, path, offline=True)
            path.write_bytes(b'bad')
            with self.assertRaisesRegex(RuntimeError, 'run bash setup.sh online'):
                setup.ensure_asset(asset, path, offline=True)

    def test_single_setup_preserves_installed_models_and_verifies_real_startup(self):
        model = self.root / 'models/installed.gguf'
        model.parent.mkdir()
        model.write_bytes(b'unchanged synthetic model')
        before = model.read_bytes()
        python = self.root / 'hermes/python/bin/python3.12'
        python.parent.mkdir(parents=True)
        python.touch()
        with patch.object(prepare, 'ROOT', self.root), \
             patch.object(prepare.platform, 'system', return_value='Linux'), \
             patch.object(prepare.platform, 'machine', return_value='x86_64'), \
             patch.object(prepare.subprocess, 'run', return_value=Mock(returncode=0)) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prepare.main([]), 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn('--runtime-only', commands[0])
        self.assertNotIn('--offline', commands[1])
        self.assertIn('--check', commands[2])
        self.assertIn('--verify-session', commands[3])
        self.assertEqual(run.call_args_list[3].kwargs['input'], '/quit\n')
        self.assertEqual(model.read_bytes(), before)
        self.assertTrue(json.loads((self.root / 'state/setup-ready.json').read_text())['ready'])

    def test_offline_master_passes_offline_to_both_helpers(self):
        model = self.root / 'models/test.gguf'
        model.parent.mkdir()
        model.touch()
        with patch.object(prepare, 'ROOT', self.root), \
             patch.object(prepare.platform, 'system', return_value='Linux'), \
             patch.object(prepare.platform, 'machine', return_value='x86_64'), \
             patch.object(prepare.subprocess, 'run', return_value=Mock(returncode=0)) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            prepare.main(['--offline'])
        self.assertIn('--offline', run.call_args_list[0].args[0])
        self.assertIn('--offline', run.call_args_list[1].args[0])

    def test_failed_setup_never_records_ready(self):
        with patch.object(prepare, 'ROOT', self.root), \
             patch.object(prepare.platform, 'system', return_value='Linux'), \
             patch.object(prepare.platform, 'machine', return_value='x86_64'), \
             patch.object(prepare.subprocess, 'run', return_value=Mock(returncode=1)), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'logs/setup.log'):
            prepare.main([])
        self.assertFalse(json.loads((self.root / 'state/setup-ready.json').read_text())['ready'])

    def test_online_hermes_downloads_pinned_wheels_before_offline_install(self):
        bundle = self.root / 'hermes'
        python = bundle / 'python/bin/python3.12'
        python.parent.mkdir(parents=True)
        python.touch()
        (bundle / 'src').mkdir()
        (bundle / 'src/run_agent.py').touch()
        (bundle / 'site-packages.pending').mkdir()
        (self.root / 'hermes-assets.json').write_text(json.dumps(dict(python={}, hermes=dict(version='fixture',revision='fixture'))))
        def process(*args, **kwargs):
            command = args[0]
            if 'install' in command:
                (bundle / 'site-packages.pending').mkdir(exist_ok=True)
            return Mock(returncode=0)
        with patch.object(setup_hermes, 'ROOT', self.root), patch.object(setup_hermes, 'BUNDLE', bundle), \
             patch.object(setup_hermes, 'PYTHON', python), patch.object(setup_hermes, 'download'), \
             patch.object(setup_hermes.subprocess, 'run', side_effect=process) as run, \
             patch.object(setup_hermes, 'sha256', return_value='fixture'), \
             patch.object(sys, 'argv', ['setup_hermes.py']), contextlib.redirect_stdout(io.StringIO()):
            setup_hermes.main()
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn('download', commands[0])
        self.assertIn('--require-hashes', commands[0])
        self.assertIn('--only-binary=:all:', commands[0])
        self.assertIn('--no-index', commands[1])


if __name__ == '__main__':
    unittest.main()
