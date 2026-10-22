"""Offline fit tests: synthetic GGUFs, mocked hardware, existing lifecycle integration."""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capstone
from fit_diagnostics import DiagnosticRun
from gguf_metadata import read_metadata
import llm_fit as fit
import model_startup as startup
import usb_core


def fixture(path, native=16384, parameters=1024**3, template=True, arch='llama'):
    def string(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw
    metadata = {'general.architecture': arch, 'general.name': path.stem,
                'general.file_type': 15, arch + '.context_length': native,
                arch + '.block_count': 16, arch + '.embedding_length': 1024,
                arch + '.attention.head_count': 16, arch + '.attention.head_count_kv': 4}
    if template:
        metadata['tokenizer.chat_template'] = 'synthetic template'
    data = b'GGUF' + struct.pack('<IQQ', 3, 1, len(metadata))
    for key, value in metadata.items():
        data += string(key)
        data += struct.pack('<I', 8) + string(value) if isinstance(value, str) else struct.pack('<II', 4, value)
    data += string('weights') + struct.pack('<IQIQ', 1, parameters, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data + bytes(256))
    return path


class FitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'models').mkdir()
        self.config = {'context_size': 8192, 'server': 'runtime/llama-server',
                       'model': 'models/a.gguf', 'chat_templates': {}}
        self.hardware = dict(total_ram=32 * fit.GIB, available_ram=24 * fit.GIB,
                             runtime_compatible=True, runtime_reason='CPU supported', cores=8)
        (self.root / 'config.json').write_text(json.dumps(self.config))
        self.a = fixture(self.root / 'models/a.gguf')

    def evaluate(self, path=None, **hardware):
        return fit.evaluate(path or self.a, None, self.config, dict(self.hardware, **hardware), self.root)

    def profiles(self, data):
        (self.root / 'model-profiles.json').write_text(json.dumps(data))

    def test_sufficient_ram_recommends_highest_score(self):
        b = fixture(self.root / 'models/b.gguf', parameters=4 * 1024**3)
        rows = fit.rank_models(self.config, self.hardware, self.root)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['path'], str(b))
        self.assertTrue(rows[0]['eligible'])

    def test_oversized_model_rejected(self):
        with self.a.open('r+b') as handle:
            handle.truncate(30 * fit.GIB)  # sparse file; no weights loaded
        self.assertEqual(self.evaluate()['classification'], 'Does Not Fit')

    def test_low_available_ram_even_when_total_is_large(self):
        row = self.evaluate(available_ram=fit.GIB)
        self.assertFalse(row['eligible'])
        self.assertIn('current available', row['reasons'][0])

    def test_marginal_respects_safety_reserve(self):
        base = self.evaluate()
        row = self.evaluate(available_ram=base['estimated_memory'] + base['safety_reserve'] + 1)
        self.assertEqual(row['classification'], 'Marginal Fit')
        self.assertTrue(row['eligible'])

    def test_cpu_only(self):
        self.assertTrue(self.evaluate(gpus=[], gpu_acceleration_supported=False)['eligible'])

    def test_gpu_detected_but_unsupported_does_not_add_memory(self):
        row = self.evaluate(available_ram=fit.GIB, gpus=['GPU'], gpu_acceleration_supported=False)
        self.assertFalse(row['eligible'])

    def test_missing_corrupt_and_missing_tensor_metadata(self):
        for contents in (b'garbage', b'GGUF', b'GGUF' + struct.pack('<IQQ', 3, 0, 0)):
            self.a.write_bytes(contents)
            self.assertFalse(self.evaluate()['eligible'])
        self.a.unlink()
        self.assertEqual(self.evaluate()['classification'], 'Unknown')

    def test_multiple_models_deduplicate_symlinks_and_reject_external(self):
        (self.root / 'models/alias.gguf').symlink_to(self.a)
        fixture(self.root / 'models/drop-in/b.gguf')
        outside = fixture(self.root / 'outside.gguf')
        (self.root / 'models/outside.gguf').symlink_to(outside)
        self.assertEqual(len(fit.installed_models(self.root)), 2)

    def test_no_models(self):
        self.a.unlink()
        self.assertEqual(fit.rank_models(self.config, self.hardware, self.root), [])

    def test_native_context_incompatible(self):
        fixture(self.a, native=4096)
        self.assertEqual(self.evaluate()['classification'], 'Does Not Fit')

    def test_hermes_capability_evidence_changes_ranking(self):
        b = fixture(self.root / 'models/b.gguf', parameters=4 * 1024**3)
        self.profiles({'models/a.gguf': {'llm_fit': {'tool_calling': 'verified',
                       'capability_evidence': 'Synthetic test evidence, not a production claim'}}})
        rows = fit.rank_models(self.config, self.hardware, self.root)
        self.assertEqual(rows[0]['path'], str(self.a))
        self.assertGreater(rows[0]['score'], rows[1]['score'])
        self.assertNotEqual(rows[0]['path'], str(b))

    def test_unsubstantiated_capability_claim_is_unknown(self):
        self.profiles({'models/a.gguf': {'llm_fit': {'tool_calling': 'verified'}}})
        self.assertEqual(self.evaluate()['classification'], 'Unknown')

    def test_deterministic_scores(self):
        self.assertEqual(self.evaluate(), self.evaluate())

    def test_missing_template_is_unknown(self):
        fixture(self.a, template=False)
        self.assertEqual(self.evaluate()['classification'], 'Unknown')

    def test_unknown_architecture_needs_documented_memory_bound(self):
        fixture(self.a, arch='hybrid')
        self.assertEqual(self.evaluate()['classification'], 'Unknown')
        self.profiles({'models/a.gguf': {'llm_fit': {'kv_bytes_per_token': 65536,
                       'memory_evidence': 'synthetic upper bound'}}})
        self.assertTrue(self.evaluate()['eligible'])

    def test_runtime_incompatible(self):
        self.assertFalse(self.evaluate(runtime_compatible=False)['eligible'])

    def test_profile_matches_canonical_drop_in_path(self):
        (self.root / 'models/original.gguf').symlink_to(self.a)
        self.profiles({'models/original.gguf': {'context_size': 4096}})
        self.assertEqual(fit.model_profile(self.a, self.root)['context_size'], 4096)
        with patch.object(capstone, 'ROOT', self.root):
            active = capstone.runtime_config(self.config, 'llama.cpp', str(self.a))
        self.assertEqual(active['context_size'], 4096)
        self.assertEqual(self.config['context_size'], 8192)

    def test_scanner_and_ranking_never_request_network(self):
        with patch('socket.socket', side_effect=AssertionError('network forbidden')), \
             patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            self.assertTrue(fit.rank_models(self.config, self.hardware, self.root)[0]['eligible'])

    def test_diagnostics_append_and_attach(self):
        first = DiagnosticRun(root=self.root)
        first.write('MODEL LOAD TIME', measured_seconds=.5)
        previous = first.path.read_text()
        second = DiagnosticRun(root=self.root)
        DiagnosticRun.attach(second.run_id, self.root).write('FUTURE TEST', estimated=True)
        text = first.path.read_text()
        self.assertTrue(text.startswith(previous))
        self.assertIn(first.run_id, text)
        self.assertIn(second.run_id, text)
        self.assertIn('timestamp', text)

    def test_auto_selects_without_confirmation_then_opens_original_menu(self):
        b = fixture(self.root / 'models/b.gguf', parameters=2 * 1024**3)
        before = (self.root / 'config.json').read_bytes()
        with patch.object(startup, 'detect_hardware', return_value=self.hardware), \
             patch('menu.main', return_value=0) as original, patch.object(startup, 'Server') as server, \
             patch('builtins.input', side_effect=AssertionError('confirmation forbidden')), contextlib.redirect_stdout(io.StringIO()):
            startup.startup_cycle('auto', self.root)
        original.assert_called_once_with(active_model=str(b))
        server.assert_not_called()
        self.assertEqual(before, (self.root / 'config.json').read_bytes())
        self.assertIn('opening original application menu', (self.root / 'logs/llm-fit-diagnostics.txt').read_text())

    def test_no_eligible_returns_concise_error(self):
        for rows in ([], [dict(path=str(self.a), eligible=False, classification='Unknown', score=0, reasons=[])]):
            with patch.object(startup, 'detect_hardware', return_value=self.hardware), \
                 patch.object(startup, 'rank_models', return_value=rows), patch('menu.main') as original, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                startup.startup_cycle('auto', self.root)
            original.assert_not_called()
            self.assertIn('No compatible AI', output.getvalue())

    def test_default_bypasses_ranking_and_has_no_stale_override(self):
        with patch.object(startup, 'rank_models', side_effect=AssertionError('ranking forbidden')), \
             patch.object(startup, 'runtime_compatibility', return_value=(True, 'ok')), \
             patch('menu.main', return_value=0) as original, contextlib.redirect_stdout(io.StringIO()):
            startup.startup_cycle('default', self.root)
        original.assert_called_once_with(active_model=str(self.a))

    def test_manual_selection_opens_original_menu_without_saving_default(self):
        b = fixture(self.root / 'models/b.gguf')
        before = (self.root / 'config.json').read_bytes()
        with patch('builtins.input', return_value='2'), \
             patch.object(startup, 'runtime_compatibility', return_value=(True, 'ok')), \
             patch('menu.main', return_value=0) as original, contextlib.redirect_stdout(io.StringIO()):
            startup.startup_cycle('manual', self.root)
        original.assert_called_once_with(active_model=str(b))
        self.assertEqual(before, (self.root / 'config.json').read_bytes())

    def test_server_ready_precedes_hermes_and_direct_prompt(self):
        events = []
        server = Mock()
        server.ensure.side_effect = lambda: events.append('server ready')
        server.verify_model.return_value = str(self.a)
        session = Mock()
        def chat(*args, **kwargs):
            self.assertTrue(kwargs['interactive'])
            self.assertEqual(events, ['server ready'])
            print('You>')
            return {'failed': False, 'hermes_ready': True}
        session.run.side_effect = chat
        config = dict(self.config, model_label='fixture', workdir='workspace')
        with patch.object(capstone, 'runtime_config', return_value=config), \
             patch.object(capstone, 'configure_hermes'), patch.object(startup, 'validate_assets'), \
             patch.object(startup, 'Server', return_value=server), \
             patch.object(startup, 'Sessions', return_value=session), \
             patch.object(startup.subprocess, 'run'), contextlib.redirect_stdout(io.StringIO()) as output:
            started, outcome = startup.launch_candidate(self.a, self.config, DiagnosticRun(root=self.root), self.root)
        self.assertTrue(started)
        self.assertIn('You>', output.getvalue())
        server.close.assert_called_once()
        self.assertEqual(outcome, 'interactive session completed')

    def test_failed_server_is_closed_logged_and_hermes_not_started(self):
        server = Mock()
        server.ensure.side_effect = RuntimeError('synthetic failure')
        config = dict(self.config, model_label='fixture', workdir='workspace')
        with patch.object(capstone, 'runtime_config', return_value=config), \
             patch.object(capstone, 'configure_hermes'), patch.object(startup, 'validate_assets'), \
             patch.object(startup, 'Server', return_value=server), patch.object(startup, 'Sessions') as sessions, \
             contextlib.redirect_stdout(io.StringIO()):
            started, _ = startup.launch_candidate(self.a, self.config, DiagnosticRun(root=self.root), self.root)
        self.assertFalse(started)
        sessions.assert_not_called()
        server.close.assert_called_once()
        self.assertIn('synthetic failure', (self.root / 'logs/llm-fit-diagnostics.txt').read_text())

    def test_hermes_failure_does_not_fallback_after_interaction(self):
        config = dict(self.config, model_label='fixture', workdir='workspace')
        server, sessions = Mock(), Mock()
        server.verify_model.return_value = str(self.a)
        sessions.run.return_value = {'failed': True, 'hermes_ready': True}
        with patch.object(capstone, 'runtime_config', return_value=config), \
             patch.object(capstone, 'configure_hermes'), patch.object(startup, 'validate_assets'), \
             patch.object(startup, 'Server', return_value=server), \
             patch.object(startup, 'Sessions', return_value=sessions), \
             patch.object(startup.subprocess, 'run'), contextlib.redirect_stdout(io.StringIO()):
            started, outcome = startup.launch_candidate(self.a, self.config, DiagnosticRun(root=self.root), self.root)
        self.assertTrue(started)
        self.assertIn('failure', outcome)
        server.close.assert_called_once()

    def test_hermes_init_failure_before_prompt_allows_fallback(self):
        config = dict(self.config, model_label='fixture', workdir='workspace')
        server, sessions = Mock(), Mock()
        server.verify_model.return_value = str(self.a)
        sessions.run.return_value = {'failed': True, 'hermes_ready': False}
        with patch.object(capstone, 'runtime_config', return_value=config), \
             patch.object(capstone, 'configure_hermes'), patch.object(startup, 'validate_assets'), \
             patch.object(startup, 'Server', return_value=server), \
             patch.object(startup, 'Sessions', return_value=sessions), \
             patch.object(startup.subprocess, 'run'), contextlib.redirect_stdout(io.StringIO()):
            started, _ = startup.launch_candidate(self.a, self.config, DiagnosticRun(root=self.root), self.root)
        self.assertFalse(started)
        server.close.assert_called_once()

    def test_exit_and_return_menu_and_original_menu(self):
        with patch('builtins.input', side_effect=['1', '2', '3', '4', '0']), \
             patch.object(startup, 'startup_cycle') as cycle, patch('menu.main') as original, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(startup.main(), 0)
        self.assertEqual([call.args[0] for call in cycle.call_args_list], ['auto', 'default', 'manual'])
        original.assert_not_called()
        self.assertNotIn('4. Original', startup.MENU)

    def test_missing_asset_fails_without_installation(self):
        with self.assertRaisesRegex(RuntimeError, 'asset missing'):
            startup.validate_assets(self.config, self.root)

    def test_unsafe_server_flags_rejected(self):
        for name in ('hermes/python/bin/python3.12', 'hermes/src/run_agent.py', 'hermes/site-packages/openai/__init__.py'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for args in (['--hf-repo', 'remote'], ['--model', '/other.gguf'], ['--ctx-size', '1000000']):
            with self.assertRaises(ValueError):
                startup.validate_assets(dict(self.config, server_args=args), self.root)


class HardwareTests(unittest.TestCase):
    def test_real_host_detection_is_offline_with_missing_runtime(self):
        with tempfile.TemporaryDirectory() as folder, patch('socket.socket', side_effect=AssertionError('network forbidden')):
            hardware = fit.detect_hardware({'server': 'missing'}, Path(folder))
        self.assertGreater(hardware['total_ram'], 0)
        self.assertGreater(hardware['available_ram'], 0)
        self.assertFalse(hardware['runtime_compatible'])
        self.assertIn('CPU', hardware['backend'])

    def test_runtime_elf_architecture(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(fit.platform, 'system', return_value='Linux'):
            path = Path(folder) / 'server'
            path.write_bytes(b'\x7fELF\x02\x01' + bytes(12) + struct.pack('<H', 62))
            path.chmod(0o755)
            self.assertTrue(fit.runtime_compatibility(path, 'x86_64')[0])
            self.assertFalse(fit.runtime_compatibility(path, 'aarch64')[0])
            self.assertFalse(fit.runtime_compatibility(path, 'unknown')[0])

    def test_gpu_probe_distinguishes_detected_from_supported(self):
        with patch.object(fit, 'runtime_compatibility', return_value=(True, 'ok')), \
             patch.object(fit.subprocess, 'run', return_value=Mock(returncode=0, stdout='Available devices:\n', stderr='')):
            hardware = fit.detect_hardware({'server': 'missing', 'acceleration': 'cpu'})
            self.assertFalse(hardware['gpu_acceleration_supported'])
        with patch.object(fit, 'runtime_compatibility', return_value=(True, 'ok')), \
             patch.object(fit.subprocess, 'run', return_value=Mock(returncode=0, stdout='CUDA0: local GPU', stderr='')):
            hardware = fit.detect_hardware({'server': 'missing', 'acceleration': 'cpu'})
            self.assertTrue(hardware['gpu_acceleration_supported'])
            self.assertIn('CPU', hardware['backend'])


class ExistingLifecycleTests(unittest.TestCase):
    def test_correct_llama_model_argument_and_health_readiness(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'logs').mkdir()
            model = fixture(root / 'models/selected.gguf')
            config = dict(model=str(model), server='runtime/llama-server', port=12345,
                          context_size=8192, threads=2, server_args=[], startup_timeout=1)
            server = usb_core.Server(config)
            process = Mock()
            process.poll.return_value = None
            with patch.object(usb_core, 'ROOT', root), patch.object(usb_core.socket, 'socket'), \
                 patch.object(usb_core.subprocess, 'Popen', return_value=process) as popen, \
                 patch.object(server, 'healthy', side_effect=[False, True]):
                server.ensure()
            args = popen.call_args.args[0]
            self.assertEqual(args[args.index('--model') + 1], str(model))
            self.assertIn('--no-webui', args)
            self.assertEqual(args[args.index('--host') + 1], '127.0.0.1')
            with patch.object(usb_core, 'stop_process'):
                server.close()

    def test_alive_process_is_not_readiness_and_mismatched_model_rejected(self):
        server = usb_core.Server({'port': 12345, 'model': '/selected.gguf'})
        server.process = Mock()
        server.process.poll.return_value = None
        server.client = Mock()
        server.client.open.side_effect = OSError('not ready')
        self.assertFalse(server.healthy())
        response = io.StringIO(json.dumps({'model_path': '/different.gguf'}))
        with patch.object(server, 'healthy', return_value=True), \
             patch.object(server.client, 'open', return_value=contextlib.closing(response)):
            with self.assertRaisesRegex(RuntimeError, 'selected model'):
                server.verify_model()

    def test_dead_fit_server_not_restarted(self):
        server = usb_core.Server({'port': 12345, '_fit_single_attempt': True})
        server.start_attempted = True
        with patch.object(server, 'healthy', return_value=False), patch.object(usb_core.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'restart disabled'):
                server.ensure()
        process.assert_not_called()

    def test_cli_status_without_runtime_or_ranking(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(capstone, 'ROOT', Path(folder)), \
             patch.object(capstone, 'runtime_config', side_effect=AssertionError('not needed')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(capstone.main(['--status']), 0)

    def test_cli_model_override_and_queue_still_delegate(self):
        config = dict(model='saved.gguf', workdir='workspace')
        server = Mock()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps(config))
            with patch.object(capstone, 'ROOT', root), \
                 patch.object(capstone, 'runtime_config', return_value=config) as runtime, \
                 patch.object(capstone, 'configure_hermes'), patch.object(capstone, 'Server', return_value=server), \
                 patch.object(capstone, 'Sessions'), patch.object(capstone, 'load_tasks', return_value=['fixture']), \
                 patch.object(capstone, 'run_queue', return_value=0) as queue, \
                 patch.object(capstone.subprocess, 'run'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(capstone.main(['--run', '--tasks', 'safe.txt', '--model', 'override.gguf']), 0)
            self.assertEqual(runtime.call_args.args[2], 'override.gguf')
            queue.assert_called_once()
            server.close.assert_called_once()

    def test_airllm_config_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'airllm/models/local'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'config.json').write_text(json.dumps({'max_position_embeddings': 16384}))
            (root / 'airllm-config.json').write_text(json.dumps({'model': str(checkpoint)}))
            with patch.object(capstone, 'ROOT', root), patch.object(fit, 'rank_models', side_effect=AssertionError('GGUF only')):
                result = capstone.runtime_config({}, 'airllm')
            self.assertEqual(result['backend'], 'airllm')
            self.assertEqual(result['model'], str(checkpoint))


    def test_cli_check_delegates_without_starting_server(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = dict(server='runtime/server', model='models/model.gguf')
            (root / 'config.json').write_text(json.dumps(config))
            for name in ('runtime/server', 'models/model.gguf', 'hermes/python/bin/python3.12',
                         'hermes/src/run_agent.py', 'hermes/site-packages/openai/__init__.py'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with patch.object(capstone, 'ROOT', root), \
                 patch.object(capstone, 'runtime_config', return_value=config), \
                 patch.object(capstone, 'load_tasks', return_value=[]), patch.object(capstone, 'Server') as server, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(capstone.main(['--check']), 0)
            server.assert_not_called()

    def test_durable_job_status_is_not_changed_by_session_cleanup(self):
        import long_job
        with tempfile.TemporaryDirectory() as folder:
            jobs = Path(folder) / 'jobs'
            job = jobs / 'fixture'
            job.mkdir(parents=True)
            state = job / 'job.json'
            state.write_text(json.dumps({'name': 'fixture', 'status': 'completed', 'exit_code': 0}))
            before = state.read_bytes()
            with patch.object(long_job, 'JOBS', jobs), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(long_job.main(['status', 'fixture']), 0)
            server = usb_core.Server({'port': 12345})
            with patch.object(usb_core, 'stop_process') as stop:
                server.close()
            stop.assert_called_once_with(None)
            self.assertEqual(state.read_bytes(), before)

    def test_actual_local_server_readiness_and_model_identity(self):
        # Synthetic server exercises real Popen/loopback/cleanup, not real inference.
        import socket
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'logs').mkdir()
            model = fixture(root / 'models/local.gguf')
            script = root / 'server'
            script.write_text('#!' + sys.executable + '\n' + """
import argparse, json
from http.server import BaseHTTPRequestHandler, HTTPServer
parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--port', type=int)
args, _ = parser.parse_known_args()
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        payload = {'model_path': args.model} if self.path == '/props' else {'status': 'ok'}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(data)
HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
""")
            script.chmod(0o755)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            config = dict(model=str(model), server=str(script), port=port, context_size=8192,
                          threads=1, server_args=[], startup_timeout=5,
                          _fit_single_attempt=True)
            server = usb_core.Server(config)
            with patch.object(usb_core, 'ROOT', root):
                try:
                    server.ensure()
                    self.assertTrue(server.healthy())
                    self.assertEqual(server.verify_model(), str(model))
                    process = server.process
                finally:
                    server.close()
                self.assertIsNotNone(process.poll())


if __name__ == '__main__':
    unittest.main()
