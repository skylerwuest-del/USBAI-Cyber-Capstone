"""Synthetic telemetry/selection/benchmark checks. No installed weights are modified."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit_diagnostics import DiagnosticRun
import fit_performance as perf
import fit_benchmark as bench
import llm_fit
import model_startup
import capstone
from test_llm_fit import fixture


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = fixture(self.root / 'models/a.gguf')
        self.server = self.root / 'runtime/server'
        self.server.parent.mkdir()
        self.server.write_bytes(b'synthetic runtime')
        self.config = dict(model=str(self.model), server=str(self.server), context_size=8192,
                           threads=2, server_args=[], chat_templates={})
        self.hardware = dict(cpu='Synthetic CPU', architecture='x86_64', cores=8,
                             total_ram=32 * 1024**3, available_ram=24 * 1024**3,
                             backend='llama.cpp CPU (--n-gpu-layers 0)', runtime_compatible=True)
        self.identity = perf.measurement_identity(self.model, self.config, self.hardware, self.root)
        self.store = perf.PerformanceStore(self.root)
        self.report = DiagnosticRun(root=self.root)

    def generation(self, speed=20, tokens=64, timestamp=None):
        return dict(kind='generation', tokens=tokens, tokens_per_second=speed,
                    timestamp=time.time() if timestamp is None else timestamp)

    def test_reuses_matching_hardware_and_settings_after_three_samples(self):
        self.store.append(self.identity, [self.generation() for _ in range(3)])
        summary = self.store.adjustment(self.identity)
        self.assertEqual(summary['generation_samples'], 3)
        self.assertEqual(summary['median_tokens_per_second'], 20)
        self.assertGreater(summary['points'], 0)

    def test_fewer_samples_or_short_generations_do_not_affect_speed_score(self):
        for events in ([self.generation()] * 2, [self.generation(tokens=2)] * 5):
            identity = dict(self.identity, variant=len(events))
            self.store.append(identity, events)
            self.assertEqual(self.store.adjustment(identity)['points'], 0)

    def test_hardware_change_cannot_reuse_speed(self):
        self.store.append(self.identity, [self.generation()] * 3)
        for key, value in [('cpu', 'Different CPU'), ('cores', 4), ('total_ram', 16 * 1024**3)]:
            hardware = dict(self.hardware, **{key: value})
            other = perf.measurement_identity(self.model, self.config, hardware, self.root)
            self.assertEqual(self.store.adjustment(other)['generation_samples'], 0)

    def test_available_memory_change_keeps_identity_but_still_gates_selection(self):
        self.store.append(self.identity, [self.generation()] * 3)
        low = dict(self.hardware, available_ram=1024**3)
        self.assertEqual(perf.hardware_identity(low), perf.hardware_identity(self.hardware))
        rows = llm_fit.rank_models(self.config, low, self.root)
        self.assertFalse(rows[0]['eligible'])
        self.assertEqual(rows[0]['score'], 0)

    def test_settings_change_invalidates_history(self):
        self.store.append(self.identity, [self.generation()] * 3)
        for overrides in ({'context_size': 4096}, {'threads': 1}, {'server_args': ['--reasoning-budget', '0']}):
            other = perf.measurement_identity(self.model, dict(self.config, **overrides), self.hardware, self.root)
            self.assertEqual(self.store.adjustment(other)['generation_samples'], 0)

    def test_model_and_runtime_change_invalidates_history(self):
        self.store.append(self.identity, [self.generation()] * 3)
        for path in (self.model, self.server):
            with path.open('ab') as handle:
                handle.write(b'changed synthetic fixture')
            changed = perf.measurement_identity(self.model, self.config, self.hardware, self.root)
            self.assertEqual(self.store.adjustment(changed)['generation_samples'], 0)

    def test_old_and_future_samples_ignored(self):
        now = time.time()
        self.store.append(self.identity, [self.generation(timestamp=now - perf.MAX_AGE - 1)] * 3 +
                          [self.generation(timestamp=now + 60)] * 3)
        self.assertEqual(self.store.adjustment(self.identity, now)['generation_samples'], 0)

    def test_adjustment_is_small_and_cannot_promote_ineligible_model(self):
        self.store.append(self.identity, [self.generation(1e6)] * 3)
        self.assertLessEqual(self.store.adjustment(self.identity)['points'], 6)
        rows = llm_fit.rank_models(self.config, self.hardware, self.root)
        self.assertEqual(rows[0]['performance']['points'], 6)
        self.assertEqual(rows[0]['score'], rows[0]['base_score'] + 6)
        blocked = llm_fit.rank_models(self.config, dict(self.hardware, runtime_compatible=False, runtime_reason='unsupported'), self.root)
        self.assertFalse(blocked[0]['eligible'])

    def test_observations_can_change_ranking_without_erasing_quality(self):
        b = fixture(self.root / 'models/b.gguf')
        other = perf.measurement_identity(b, self.config, self.hardware, self.root)
        self.store.append(other, [self.generation(20)] * 3)
        ranked = llm_fit.rank_models(self.config, self.hardware, self.root)
        self.assertEqual(ranked[0]['path'], str(b))
        self.assertIn('tools', ranked[0]['factors'])
        self.assertEqual(ranked[0]['hermes_compatibility'], 'unknown')

    def test_failures_recorded_and_successful_load_resets_streak(self):
        events = [dict(kind='server_failure', timestamp=time.time())] * 3
        self.store.append(self.identity, events)
        self.assertLess(self.store.adjustment(self.identity)['points'], 0)
        self.store.append(self.identity, [dict(kind='load', seconds=5, timestamp=time.time())])
        self.assertEqual(self.store.adjustment(self.identity)['consecutive_failures'], 0)

    def test_cancel_and_unknown_session_failure_do_not_penalize_model(self):
        self.store.append(self.identity, [dict(kind='session_failure', timestamp=time.time())])
        self.assertEqual(self.store.adjustment(self.identity)['points'], 0)

    def test_load_time_observation_needs_two_samples(self):
        self.store.append(self.identity, [dict(kind='load', seconds=600, timestamp=time.time())])
        self.assertEqual(self.store.adjustment(self.identity)['points'], 0)
        self.store.append(self.identity, [dict(kind='load', seconds=600, timestamp=time.time())])
        self.assertEqual(self.store.adjustment(self.identity)['points'], -2)

    def test_malformed_cache_not_overwritten_or_used(self):
        self.store.path.parent.mkdir()
        for content in ('broken JSON', '{"version":1,"records":{"broken":[]}}'):
            self.store.path.write_text(content)
            rows = llm_fit.rank_models(self.config, self.hardware, self.root)
            self.assertTrue(rows[0]['eligible'])
            self.assertEqual(rows[0]['performance']['points'], 0)
            tracker = perf.create_tracker(self.model, self.config, self.hardware, self.report, self.root)
            tracker.event('load', seconds=1)
            self.assertEqual(self.store.path.read_text(), content)

    def test_cache_bounded(self):
        self.store.append(self.identity, [self.generation()] * 100)
        self.assertEqual(len(next(iter(self.store.read().values()))['events']), perf.MAX_SAMPLES)

    def test_missing_hardware_identity_disables_reuse(self):
        with self.assertRaises(ValueError):
            perf.hardware_identity(dict(self.hardware, cpu=''))

    def test_timing_parser_keeps_only_numbers_and_not_sensitive_content(self):
        text = ('USER SECRET PROMPT\nslot print_timings: prompt eval time = 100.00 ms / 50 tokens (ignored)\n'
                'slot print_timings:        eval time = 2000.00 ms / 20 tokens (ignored)\nSECRET RESPONSE\n')
        events = perf.timing_samples(text)
        self.assertEqual([event['kind'] for event in events], ['prompt_processing', 'generation'])
        self.assertEqual(events[1]['tokens_per_second'], 10)
        self.assertNotIn('SECRET', json.dumps(events))
        self.assertEqual(perf.timing_samples('eval time = 0 ms / 0 tokens'), [])

    def test_installed_runtime_timestamped_log_uses_reported_generation_speed(self):
        line = ('0.06.318.601 I slot print_timing: id  0 | task 0 | '
                'eval time = 380.59 ms / 23 tokens (17.30 ms per token, 57.81 tokens per second)\n')
        samples = perf.timing_samples(line)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]['tokens_per_second'], 57.81)

    def test_live_sampler_reads_only_new_lines_once(self):
        log = self.root / 'logs/server.log'
        log.write_text('eval time = 100 ms / 100 tokens\nOLD SECRET\n')
        tracker = perf.SessionPerformance(self.model, self.config, self.hardware, self.report, self.root)
        with log.open('a') as handle:
            handle.write('SECRET PROMPT\nslot print_timings: eval time = 2000 ms / 20 tokens\n')
        with patch('socket.socket', side_effect=AssertionError('passive tracking must not connect')):
            tracker.poll()
            tracker.poll()
            tracker.close()
        summary = self.store.adjustment(self.identity)
        self.assertEqual(summary['generation_samples'], 1)
        self.assertNotIn('SECRET', self.report.path.read_text())
        self.assertNotIn('SECRET', self.store.path.read_text())

    def test_sampler_handles_partial_lines_and_truncation(self):
        log = self.root / 'logs/server.log'
        tracker = perf.SessionPerformance(self.model, self.config, self.hardware, self.report, self.root)
        log.write_text('slot print_timings: eval time = 2000 ms / ')
        tracker.poll()
        with log.open('a') as handle:
            handle.write('20 tokens\n')
        tracker.poll()
        self.assertEqual(self.store.adjustment(self.identity)['generation_samples'], 1)
        log.write_text('eval time = 1 ms / 100 tokens\n')
        tracker.poll()
        self.assertEqual(self.store.adjustment(self.identity)['generation_samples'], 1)

    def test_telemetry_disk_failure_does_not_break_chat(self):
        tracker = perf.SessionPerformance(self.model, self.config, self.hardware, self.report, self.root)
        with patch.object(tracker.store, 'append', side_effect=OSError('disk unavailable')), \
             patch.object(tracker.report, 'write', side_effect=OSError('disk unavailable')):
            tracker.event('load', seconds=1)
        tracker.close()

    def test_normal_selection_does_not_execute_benchmarks(self):
        (self.root / 'config.json').write_text(json.dumps(self.config))
        with patch.object(model_startup, 'detect_hardware', return_value=self.hardware), \
             patch('menu.main', return_value=0), \
             patch.object(bench, 'benchmark_candidate', side_effect=AssertionError('benchmark forbidden')), \
             contextlib.redirect_stdout(io.StringIO()):
            model_startup.startup_cycle('auto', self.root)

    def test_benchmark_is_explicit_cli_and_menu_option(self):
        with patch.object(bench, 'run_benchmarks', return_value=0) as run:
            self.assertEqual(capstone.main(['--benchmark-installed']), 0)
        run.assert_called_once()
        with patch('builtins.input', side_effect=['11', '0']), patch.object(bench, 'run_benchmarks') as run, \
             contextlib.redirect_stdout(io.StringIO()):
            from menu import main
            self.assertEqual(main(), 0)
        run.assert_called_once()

    def test_benchmark_conflicting_modes_rejected(self):
        for flag in ('--run', '--status', '--chat'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                capstone.main(['--benchmark-installed', flag])
            self.assertEqual(result.exception.code, 2)

    def test_benchmark_numeric_response_measurements_only(self):
        response = dict(content='SECRET generated text', timings=dict(predicted_n=64, predicted_ms=8000,
                                                                     prompt_n=12, prompt_ms=100))
        samples = bench.response_measurements(response)
        self.assertEqual(samples[1]['tokens_per_second'], 8)
        self.assertNotIn('SECRET', json.dumps(samples))
        self.assertEqual(bench.response_measurements({'timings': {'predicted_n': 8, 'predicted_ms': float('nan')}}), [])

    def test_benchmark_uses_fixed_harmless_requests_and_closes_server(self):
        server = Mock()
        server.base = 'http://127.0.0.1:12345'
        server.verify_model.return_value = str(self.model)
        payload = json.dumps(dict(content='discard me', timings=dict(predicted_n=64, predicted_ms=8000))).encode()
        server.client.open.side_effect = lambda *args, **kwargs: contextlib.closing(io.BytesIO(payload))
        with patch.object(bench.capstone, 'runtime_config', return_value=self.config), \
             patch.object(bench, 'Server', return_value=server), patch.object(bench, 'create_tracker') as tracker:
            self.assertTrue(bench.benchmark_candidate(self.model, self.config, self.hardware, self.report, self.root))
        self.assertEqual(server.client.open.call_count, 3)
        for call in server.client.open.call_args_list:
            request = call.args[0]
            self.assertEqual(request.full_url, server.base + '/completion')
            body = json.loads(request.data)
            self.assertEqual(body['prompt'], bench.PROMPT)
            self.assertEqual(body['n_predict'], 64)
            self.assertFalse(body['cache_prompt'])
            self.assertNotIn('tools', body)
        server.close.assert_called_once()
        tracker.return_value.start.assert_not_called()
        self.assertNotIn('discard me', self.report.path.read_text())

    def test_failed_benchmark_is_bounded_and_cleanup_always_runs(self):
        server = Mock()
        server.ensure.side_effect = RuntimeError('synthetic load failure')
        with patch.object(bench.capstone, 'runtime_config', return_value=self.config), \
             patch.object(bench, 'Server', return_value=server), patch.object(bench, 'create_tracker', return_value=None):
            self.assertFalse(bench.benchmark_candidate(self.model, self.config, self.hardware, self.report, self.root))
        server.close.assert_called_once()
        server.client.open.assert_not_called()
        self.assertIn('BENCHMARK FAILURE', self.report.path.read_text())

    def test_all_benchmarks_skip_when_ram_drops(self):
        (self.root / 'config.json').write_text(json.dumps(self.config))
        row = dict(path=str(self.model), eligible=True)
        with patch.object(bench, 'detect_hardware', side_effect=[self.hardware, dict(self.hardware, available_ram=1)]), \
             patch.object(bench, 'rank_models', return_value=[row]), patch.object(bench, 'benchmark_candidate') as run, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bench.run_benchmarks(self.root), 1)
        run.assert_not_called()


    def test_startup_tracks_only_post_readiness_requests(self):
        server, sessions = Mock(), Mock()
        server.process = None
        server.verify_model.return_value = str(self.model)
        log = self.root / 'logs/server.log'
        warmup = 'slot print_timings: eval time = 10 ms / 64 tokens\n'
        server.ensure.side_effect = lambda: log.write_text(warmup)
        def chat(*args, **kwargs):
            with log.open('a') as handle:
                handle.write('SECRET user content\n')
                handle.write('slot print_timings: eval time = 2000 ms / 20 tokens\n' * 3)
            return {'failed': False, 'hermes_ready': True}
        sessions.run.side_effect = chat
        config = dict(self.config, model_label='synthetic', workdir='workspace', _fit_hardware=self.hardware)
        with patch.object(capstone, 'runtime_config', return_value=config), \
             patch.object(capstone, 'configure_hermes'), patch.object(model_startup, 'validate_assets'), \
             patch.object(model_startup, 'Server', return_value=server), \
             patch.object(model_startup, 'Sessions', return_value=sessions), \
             patch.object(model_startup.subprocess, 'run'), contextlib.redirect_stdout(io.StringIO()):
            launched, _ = model_startup.launch_candidate(self.model, config, self.report, self.root)
        self.assertTrue(launched)
        summary = self.store.adjustment(self.identity)
        self.assertEqual(summary['load_samples'], 1)
        self.assertEqual(summary['generation_samples'], 3)
        self.assertEqual(summary['median_tokens_per_second'], 10)
        self.assertNotIn('SECRET', self.report.path.read_text())
        server.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
