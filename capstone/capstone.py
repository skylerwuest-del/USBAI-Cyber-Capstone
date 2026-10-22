"""Terminal-only CapStone launcher: local model + real Hermes tools + persistent task loop."""
import argparse
import json
import math
import os
import signal
import subprocess
import time

from usb_core import ROOT, Server, atomic_json, instance_lock, resolve, shutdown_signals
from task_loop import Sessions, load_tasks, run_queue


def runtime_config(config, backend, model=None):
    config = dict(config, backend=backend)
    if backend != 'airllm':
        strict = os.environ.get('CAPSTONE_STRICT_OFFLINE') == '1'
        if strict:
            from scripts.select_model import select
            selection = select(ROOT, str(resolve(ROOT, model)) if model else None)
            config.update(selection, _strict_offline=True, acceleration='cpu')
        if model:
            config['model'] = model
        checkpoint = resolve(ROOT, config['model'])
        from llm_fit import model_profile
        profile = model_profile(checkpoint, ROOT)
        config.update(profile)
        # Profiles may supply tool settings, but must never enlarge the tested context.
        if strict:
            config.update({key: selection[key] for key in ('model', 'server', 'context_size', 'threads', 'n_gpu_layers')})
            config.update(_strict_offline=True, acceleration='cpu')
        from gguf_metadata import context_length
        native = context_length(checkpoint)
        config['context_size'] = min(int(config['context_size']), native) if native else int(config['context_size'])
        if config['context_size'] < 4096:
            raise ValueError('Model context is too small for the USB agent; choose at least 4096 tokens')
        config['local_context_compat'] = True
        config['compact_system_prompt'] = True
        config['chat_max_tokens'] = int(profile.get('chat_max_tokens', 512))
        config['chat_toolsets'] = profile.get('chat_toolsets', ['terminal'])
        config.setdefault('batch_size', 512)
        config.setdefault('ubatch_size', 512)
        config['model_label'] = profile.get('model_label', checkpoint.name)
        return config
    path = ROOT / 'airllm-config.json'
    air = json.loads(path.read_text()) if path.exists() else {}
    selected = model or air.get('model')
    if not selected:
        raise ValueError('Select a Transformers model in menu 9 first')
    checkpoint = resolve(ROOT, selected)
    if not checkpoint.is_dir() or not (checkpoint / 'config.json').is_file():
        raise ValueError('AirLLM needs an installed Transformers model directory; use menu 9 to download one')
    metadata = json.loads((checkpoint / 'config.json').read_text())
    native = metadata.get('text_config', metadata).get('max_position_embeddings', 16384)
    context = min(int(air.get('context_size', 16384)), int(native))
    if context < 8192:
        raise ValueError('This model context is too small for Hermes tools; choose an instruction model with at least 8192 tokens')
    config.update(model=str(checkpoint), context_size=context,
                  local_context_compat=True, compact_system_prompt=True,
                  model_label=checkpoint.name, chat_max_tokens=int(air.get('chat_max_tokens', 256)),
                  toolsets=air.get('toolsets', ['terminal']),
                  cache_gb=float(air.get('cache_gb', 8)),
                  max_tokens=int(air.get('max_tokens', 1024)),
                  request_timeout=int(air.get('request_timeout', 3600)),
                  startup_timeout=int(air.get('startup_timeout', 1800)),
                  pass_timeout=int(air.get('pass_timeout', 14400)))
    if not 1 <= config['max_tokens'] < context:
        raise ValueError('AirLLM max_tokens must be positive and smaller than its context window')
    if not math.isfinite(config['cache_gb']) or config['cache_gb'] < 0:
        raise ValueError('AirLLM cache_gb must be a nonnegative number')
    if not isinstance(config['toolsets'], list) or not config['toolsets'] or any(
            tool not in ('terminal', 'file', 'todo') for tool in config['toolsets']):
        raise ValueError('AirLLM toolsets must select terminal, file and/or todo')
    return config


def configure_hermes(config):
    # JSON is also valid YAML. Paths are regenerated for the current USB mount.
    home = ROOT / 'state' / 'hermes'
    home.mkdir(parents=True, exist_ok=True)
    settings = {
        'model': {'default': 'capstone', 'provider': 'custom',
                  'base_url': 'http://127.0.0.1:' + str(config['port']) + '/v1',
                  'context_length': config['context_size']},
        'providers': {'custom': {'base_url': 'http://127.0.0.1:' + str(config['port']) + '/v1',
                                'request_timeout_seconds': config['request_timeout']}},
        'terminal': {'backend': 'local', 'cwd': str(resolve(ROOT, config['workdir'])),
                     'timeout': config['command_timeout'], 'home_mode': 'real'},
        'memory': {'memory_enabled': False, 'user_profile_enabled': False},
        'compression': {'enabled': True, 'threshold': 0.70, 'target_ratio': 0.3},
        'agent': {'max_turns': config['pass_turns'], 'verify_on_stop': False,
                  'api_max_retries': 2},
        'tool_loop_guardrails': {'warnings_enabled': True, 'hard_stop_enabled': False,
                                 'non_interactive_hard_stop_enabled': False},
        'fallback_model': None,
    }
    if config.get('_interactive_chat') and config.get('backend') != 'airllm':
        settings['tools'] = {'tool_search': {'enabled': 'off'}}
    atomic_json(home / 'config.yaml', settings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--tasks', default='tasks.txt')
    parser.add_argument('--state')
    parser.add_argument('--backend', choices=['llama.cpp', 'airllm'], default='llama.cpp')
    parser.add_argument('--model', help='Override the selected local model for this run')
    parser.add_argument('--airllm-model', help='Override the selected AirLLM local Transformers directory')
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--query', help='Execute one task through the same autonomous loop')
    parser.add_argument('--serve-only', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--benchmark-installed', action='store_true', help='Explicit offline speed test of compatible installed GGUFs')
    parser.add_argument('--fit-session', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.benchmark_installed:
        if args.backend != 'llama.cpp' or any((args.run, args.query, args.chat, args.serve_only, args.check, args.status, args.model, args.airllm_model)):
            parser.error('--benchmark-installed must be used alone with the GGUF backend')
        from fit_benchmark import run_benchmarks
        return run_benchmarks(ROOT)
    args.state = args.state or ('state/airllm-task-loop.json' if args.backend == 'airllm' else 'state/task-loop.json')
    if args.status:
        path = resolve(ROOT, args.state)
        print(path.read_text() if path.exists() else 'No Hermes task-loop progress yet.')
        return 0
    config = json.loads((ROOT / 'config.json').read_text())
    config = runtime_config(config, args.backend, args.airllm_model or args.model)
    config['_interactive_chat'] = args.backend == 'llama.cpp' and not any((args.run, args.query, args.serve_only, args.check))
    for folder in ('state', 'logs', 'workspace'):
        (ROOT / folder).mkdir(exist_ok=True)
    if args.check:
        from usb_core import HERMES
        for name in (config['server'], str(HERMES / 'python/bin/python3.12'),
                     'hermes/src/run_agent.py', str(HERMES / 'site-packages/openai/__init__.py')):
            if not resolve(ROOT, name).is_file():
                raise RuntimeError('Missing ' + name + '; run bash setup.sh')
        model = resolve(ROOT, config['model'])
        if not (model.is_dir() if args.backend == 'airllm' else model.is_file()):
            raise RuntimeError('Selected model is missing')
        if args.backend == 'airllm' and not (ROOT / 'airllm/site-packages/airllm').is_dir():
            raise RuntimeError('AirLLM dependencies are missing; repair them in menu 9')
        load_tasks(resolve(ROOT, args.tasks))
        print('USB files and plain-English task queue are ready.')
        return 0
    tasks = load_tasks(resolve(ROOT, args.tasks)) if args.run else None
    if args.query:
        tasks = [{'prompt': args.query, 'id': 'query-' + __import__('hashlib').sha256(args.query.encode()).hexdigest()[:16]}]
    if args.fit_session and args.backend == 'llama.cpp' and tasks is None and not args.serve_only:
        from fit_diagnostics import DiagnosticRun
        from model_startup import launch_candidate
        report = DiagnosticRun('SELECTED MODEL APPLICATION MENU CHAT')
        try:
            with instance_lock(ROOT / 'state/runner.lock'):
                launched, outcome = launch_candidate(resolve(ROOT, config['model']), config, report)
            report.finish(outcome)
            return 0 if launched and outcome == 'interactive session completed' else 1
        except KeyboardInterrupt:
            report.finish('Interrupted; owned session resources closed')
            raise
    with instance_lock(ROOT / 'state' / 'runner.lock'):
        configure_hermes(config)
        server = Server(config)
        try:
            server.ensure()
            sessions = Sessions(config, server)
            if tasks is not None:
                return run_queue(tasks, resolve(ROOT, args.state), config, sessions)
            if args.serve_only:
                print('Terminal-only model service running; Ctrl+C stops it.', flush=True)
                while True:
                    time.sleep(5)
                    server.ensure()
            result = sessions.run('', resolve(ROOT, config['workdir']), interactive=True)
            return 1 if result.get('failed') else 0
        finally:
            with shutdown_signals():
                server.close()
                # Flush this filesystem before the launcher announces shutdown.
                subprocess.run(['sync', '-f', str(ROOT)], check=True)
                print('CapStone processes closed; USB writes flushed.', flush=True)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print('\nStopped. Progress saved; owned processes closed.', flush=True)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print('CapStone: ' + str(exc), flush=True)
        raise SystemExit(1)
