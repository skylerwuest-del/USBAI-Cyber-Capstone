import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cai_bridge as bridge
import setup_cai
from setup import sha256


class CaiBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cai = self.root / 'cai'
        self.cai.mkdir()
        self.python = self.cai / 'python' / 'bin' / 'python3.12'
        self.patches = [patch.object(bridge, 'ROOT', self.root), patch.object(bridge, 'CAI', self.cai), patch.object(bridge, 'PYTHON', self.python)]
        for p in self.patches: p.start()
        self.config = {'port': 12345, 'context_size': 8192, 'max_tokens': 1200, 'request_timeout': 600, 'retry_delay': 1}

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def make_bundle(self):
        self.python.parent.mkdir(parents=True)
        self.python.write_text('# runtime placeholder')
        self.python.chmod(0o755)
        cli = self.cai / 'site-packages' / 'cai' / 'cli.py'
        cli.parent.mkdir(parents=True)
        cli.write_text('# CAI placeholder')
        lock = self.root / 'cai-requirements.lock'
        lock.write_text('cai-framework==0.5.10')
        (self.cai / 'installed.json').write_text(json.dumps({'cai_version': '0.5.10', 'lock_sha256': sha256(lock)}))

    def test_host_cloud_configuration_is_overridden_for_the_local_model(self):
        with patch.dict(os.environ, {'OPENAI_BASE_URL': 'https://remote.example/v1', 'OPENAI_API_KEY': 'host-secret', 'CAI_MODEL': 'alias1', 'PYTHONPATH': '/host/project'}):
            env, cwd = bridge.prepare_environment(self.config)
        self.assertEqual(env['OPENAI_BASE_URL'], 'http://127.0.0.1:12345/v1')
        self.assertEqual(env['OPENAI_API_KEY'], 'capstone-local')
        self.assertEqual(env['CAI_MODEL'], 'openai/capstone')
        self.assertEqual(env['CAI_MAX_TURNS'], 'inf')
        self.assertEqual(env['CAI_TELEMETRY'], 'false')
        self.assertEqual(env['PYTHON_DOTENV_DISABLED'], '1')
        self.assertNotIn('PYTHONPATH', env)
        self.assertTrue(cwd.is_relative_to(self.root))
        self.assertNotIn('host-secret', (self.cai / 'local-config.json').read_text())
        self.assertTrue((cwd / 'pricing.json').is_file())

    def test_missing_or_changed_bundle_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'not installed'): bridge.check_bundle()
        self.make_bundle()
        bridge.check_bundle()
        (self.root / 'cai-requirements.lock').write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'does not match'): bridge.check_bundle()

    def test_first_launch_installs_once_before_opening_a_session(self):
        with patch.object(bridge, 'check_bundle') as check, patch.object(bridge.subprocess, 'run') as install:
            bridge.ensure_bundle()
        install.assert_called_once()
        check.assert_called_once()
        (self.cai / 'installed.json').write_text('{}')
        with patch.object(bridge, 'check_bundle'), patch.object(bridge.subprocess, 'run') as install:
            bridge.ensure_bundle()
        install.assert_not_called()

    def test_model_readiness_precedes_cai_launch_and_prompt_is_one_argument(self):
        events = []
        server = Mock()
        server.ensure.side_effect = lambda: events.append('ready')
        child = Mock()
        child.poll.return_value = 0
        child.returncode = 0
        def spawn(args, **kwargs):
            events.append('launch')
            self.assertEqual(args[-2:], ['--prompt', 'one prompt with spaces'])
            self.assertNotIn('start_new_session', kwargs)
            return child
        with patch.object(bridge, 'check_bundle'), patch.object(bridge.subprocess, 'Popen', side_effect=spawn):
            self.assertEqual(bridge.run_cai(server, self.config, prompt='one prompt with spaces'), 0)
        self.assertEqual(events, ['ready', 'launch'])

    def test_server_crash_is_restarted_while_cai_stays_open(self):
        server = Mock()
        server.healthy.return_value = False
        child = Mock()
        child.poll.side_effect = [None, 0, 0]
        child.returncode = 0
        with patch.object(bridge, 'check_bundle'), patch.object(bridge.subprocess, 'Popen', return_value=child), patch.object(bridge.time, 'sleep'):
            bridge.run_cai(server, self.config)
        self.assertEqual(server.ensure.call_count, 2)
        child.terminate.assert_not_called()

    def test_interrupt_terminates_the_owned_cai_child(self):
        server = Mock()
        server.healthy.return_value = True
        child = Mock()
        child.poll.return_value = None
        with patch.object(bridge, 'check_bundle'), patch.object(bridge.subprocess, 'Popen', return_value=child), patch.object(bridge.time, 'sleep', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): bridge.run_cai(server, self.config)
        child.terminate.assert_called_once()
        child.wait.assert_called_once()

    def test_setup_does_not_inherit_host_python_or_package_indexes(self):
        with patch.dict(os.environ, {'PIP_INDEX_URL': 'https://private.example', 'PYTHONPATH': '/private/lib', 'PYTHONHOME': '/other/python', 'VIRTUAL_ENV': '/host/venv'}):
            env = setup_cai.pip_env()
        for key in ('PIP_INDEX_URL', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
            self.assertNotIn(key, env)
        self.assertEqual(env['PIP_CONFIG_FILE'], os.devnull)


if __name__ == '__main__': unittest.main()
