"""Explicit, bounded, offline speed comparison. Never invokes Hermes or tool tasks."""
import json
import math
from pathlib import Path
import time
import urllib.request

import capstone
from fit_diagnostics import DiagnosticRun
from fit_performance import create_tracker
from llm_fit import detect_hardware, evaluate, rank_models
from usb_core import ROOT, Server, instance_lock, shutdown_signals

PROMPT = 'Write a short explanation of why the sky appears blue, using simple words.'
REPETITIONS = 3
OUTPUT_TOKENS = 64


def response_measurements(response):
    timings = response.get('timings', {}) if isinstance(response, dict) else {}
    result = []
    for kind, prefix in (('prompt_processing', 'prompt'), ('generation', 'predicted')):
        count, duration = timings.get(prefix + '_n'), timings.get(prefix + '_ms')
        if (not isinstance(count, int) or isinstance(count, bool) or count <= 0 or
                not isinstance(duration, (int, float)) or isinstance(duration, bool) or
                not math.isfinite(duration) or duration <= 0):
            continue
        speed = timings.get(prefix + '_per_second')
        if not isinstance(speed, (int, float)) or isinstance(speed, bool) or not math.isfinite(speed) or speed <= 0:
            speed = count * 1000 / duration
        result.append(dict(kind=kind, tokens=count, seconds=duration / 1000,
                           tokens_per_second=speed, timestamp=time.time()))
    return result


def benchmark_candidate(path, config, hardware, report, root=ROOT):
    from model_startup import validate_server_args
    active = capstone.runtime_config(config, 'llama.cpp', str(path))
    validate_server_args(active)
    active['_fit_single_attempt'] = True
    server = Server(active)
    tracker = create_tracker(path, active, hardware, report, root)
    try:
        started = time.monotonic()
        server.ensure()
        actual = server.verify_model()
        if tracker:
            tracker.event('load', seconds=time.monotonic() - started)
        report.write('BENCHMARK SERVER READY', model=path, reported_model_path=actual)
        measurements = []
        for iteration in range(REPETITIONS):
            request = urllib.request.Request(server.base + '/completion',
                data=json.dumps(dict(prompt=PROMPT, n_predict=OUTPUT_TOKENS, temperature=0,
                                     seed=42, stream=False, cache_prompt=False, ignore_eos=True)).encode(),
                headers={'Content-Type': 'application/json'})
            with server.client.open(request, timeout=min(120, int(active.get('request_timeout', 120)))) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError('Benchmark response exceeded size limit')
            samples = response_measurements(json.loads(raw))
            if not any(sample['kind'] == 'generation' for sample in samples):
                raise ValueError('Server returned no valid generation timing measurements')
            for sample in samples:
                sample.update(source='explicit_benchmark', repetition=iteration + 1)
            measurements.extend(samples)
            if tracker:
                tracker.record(samples)
        report.write('BENCHMARK RESULT', model=path, repetitions=REPETITIONS,
                     max_output_tokens=OUTPUT_TOKENS, measurements=measurements,
                     tool_reliability='Not tested; no tools or tasks executed')
        return True
    except (OSError, ValueError, RuntimeError) as exc:
        report.write('BENCHMARK FAILURE', model=path, error=str(exc))
        if tracker:
            process = server.process
            tracker.event('server_failure' if process is not None and process.poll() is not None else 'benchmark_failure')
        return False
    finally:
        with shutdown_signals():
            server.close()
        # Benchmark response timings are already recorded; do not also ingest log timings.


def run_benchmarks(root=ROOT):
    report = DiagnosticRun('EXPLICIT INSTALLED MODEL BENCHMARK', root)
    try:
        config = json.loads((root / 'config.json').read_text())
        hardware = detect_hardware(config, root)
        report.write('HOST HARDWARE', **hardware)
        rows = rank_models(config, hardware, root)
        report.write('BENCHMARK INVENTORY', inventory=rows)
        candidates = [Path(row['path']) for row in rows if row['eligible']]
        if not candidates:
            report.finish('No eligible installed model available for benchmarking')
            print('No compatible models available for benchmarking. Diagnostics: logs/llm-fit-diagnostics.txt', flush=True)
            return 1
        passed = failed = skipped = 0
        for folder in ('logs', 'state'):
            (root / folder).mkdir(exist_ok=True)
        with instance_lock(root / 'state/runner.lock'):
            for path in candidates:
                # Explicit tests can be lengthy; recheck current RAM before each load.
                current = detect_hardware(config, root)
                checked = evaluate(path, None, config, current, root)
                if not checked['eligible']:
                    report.write('BENCHMARK SKIPPED', model=path, reasons=checked['reasons'])
                    skipped += 1
                    continue
                print('Testing installed model: ' + checked['name'], flush=True)
                if benchmark_candidate(path, config, current, report, root):
                    passed += 1
                else:
                    failed += 1
        report.finish('benchmark completed', tested=passed + failed, passed=passed, failed=failed, skipped=skipped)
        print('Benchmark: %s completed, %s failed, %s skipped. Diagnostics: logs/llm-fit-diagnostics.txt' %
              (passed, failed, skipped), flush=True)
        return 0 if passed and not failed else 1
    except KeyboardInterrupt:
        report.finish('benchmark interrupted; owned server closed')
        print('\nBenchmark stopped. Diagnostics: logs/llm-fit-diagnostics.txt')
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        report.write('BENCHMARK FAILURE', error=str(exc))
        report.finish('benchmark unavailable')
        print('Benchmark unavailable. Diagnostics: logs/llm-fit-diagnostics.txt')
        return 1
