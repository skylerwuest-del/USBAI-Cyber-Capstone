"""Model startup menu; delegates inference and interactive chat to existing owners."""
import json
from pathlib import Path
import signal
import subprocess
import time

import capstone
from fit_diagnostics import DiagnosticRun
from fit_performance import create_tracker
from llm_fit import (detect_hardware, installed_models, rank_models,
                     runtime_compatibility)
from models_menu import select
from task_loop import Sessions
from usb_core import ROOT, Server, instance_lock, resolve, shutdown_signals


MENU = '''
=============================================
           USB AI - MODEL STARTUP
=============================================

1. LLM Fit - Automatically Select Best AI
2. Use Default AI
3. Manually Select Installed GGUF AI
0. Exit
'''


def manual_model(root=ROOT):
    paths = [path for path, error in installed_models(root) if not error]
    if not paths:
        raise RuntimeError('No complete installed GGUF models')
    choice = select([str(path.relative_to((root / 'models').resolve())) for path in paths], 'Installed GGUF models')
    return paths[choice] if choice is not None else None


def validate_assets(config, root=ROOT):
    """No setup/repair/download calls. CLI paths keep their previous behavior."""
    for name in ('hermes/python/bin/python3.12', 'hermes/src/run_agent.py',
                 'hermes/site-packages/openai/__init__.py'):
        if not (root / name).is_file():
            raise RuntimeError('Required local Hermes asset missing: ' + name)
    validate_server_args(config)


def validate_server_args(config):
    # Restrict fit launches to known local settings whose resource use is budgeted.
    args = list(config.get('server_args', []))
    while args:
        flag = args.pop(0)
        if flag != '--reasoning-budget' or not args:
            raise ValueError('Unsupported server_args for offline fit startup: ' + str(flag))
        int(args.pop(0))


def launch_candidate(path, config, report, root=ROOT):
    """Return (started, outcome); only pre-chat server failures permit fallback."""
    started = time.monotonic()
    server = None
    tracker = None
    try:
        selected = path.resolve()
        if not selected.is_file() or selected.suffix.lower() != '.gguf':
            raise ValueError('Selected local GGUF is missing')
        if not selected.is_relative_to((root / 'models').resolve()):
            raise ValueError('Interactive selection requires a model inside the managed models folder')
        active = capstone.runtime_config(config, 'llama.cpp', str(selected))
        # runtime_config copies settings; never persist a selected default.
        active.update(_fit_run_id=report.run_id, _fit_single_attempt=True, _interactive_chat=True)
        validate_assets(active, root)
        hardware = config.get('_fit_hardware') or detect_hardware(active, root)
        tracker = create_tracker(selected, active, hardware, report, root)
        report.write('SELECTION', selected_model=active['model_label'], selected_path=selected,
                     context=active['context_size'], session_only_override=True)
        print('Selected: ' + active['model_label'], flush=True)
        capstone.configure_hermes(active)
        server = Server(active)
        report.write('STARTUP ATTEMPT', model=selected, server_started=False, server_ready=False)
        print('Starting local model server...', flush=True)
        load_started = time.monotonic()
        server.ensure()
        actual = server.verify_model()
        if tracker:
            tracker.event('load', seconds=time.monotonic() - load_started)
            tracker.start()
        report.write('STARTUP VALIDATION', server_started=True, server_ready=True,
                     reported_model_path=actual, hermes_ready='pending SDK initialization',
                     startup_duration_seconds=round(time.monotonic() - started, 3))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        if tracker:
            process = getattr(server, 'process', None)
            kind = 'server_failure' if process is not None and process.poll() is not None else 'startup_failure'
            tracker.event(kind)
        report.write('STARTUP FAILURE', model=path, error=str(exc), server_ready=False,
                     startup_duration_seconds=round(time.monotonic() - started, 3))
        if server is not None:
            with shutdown_signals():
                server.close()
        if tracker:
            tracker.close()
        return False, 'server or preflight failure'
    except BaseException:
        if server is not None:
            with shutdown_signals():
                server.close()
        if tracker:
            tracker.close()
        raise
    try:
        print('Starting Hermes...', flush=True)
        print('Diagnostics: logs/llm-fit-diagnostics.txt', flush=True)
        result = Sessions(active, server).run('', resolve(root, active['workdir']), interactive=True)
        report.write('SESSION RESULT', hermes_ready=result.get('hermes_ready', False),
                     failed=result.get('failed', False))
        if not result.get('hermes_ready'):
            report.write('STARTUP FAILURE', model=path, stage='Hermes initialization',
                         hermes_ready=False, error='Worker exited before reaching the interactive prompt')
            return False, 'Hermes initialization failed before interactive prompt'
        if result.get('failed'):
            if tracker:
                tracker.event('session_failure')
            print('Hermes did not complete normally. Diagnostics: logs/llm-fit-diagnostics.txt')
            return True, 'Hermes session failure after interactive prompt'
        return True, 'interactive session completed'
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        report.write('STARTUP FAILURE', model=path, stage='Hermes worker startup', error=str(exc))
        return False, 'Hermes worker could not start'
    finally:
        with shutdown_signals():
            if tracker:
                if server.process is not None and server.process.poll() is not None:
                    tracker.event('runtime_failure')
            server.close()
            if tracker:
                tracker.close()
            subprocess.run(['sync', '-f', str(root)], check=True)


def startup_cycle(mode, root=ROOT):
    report = DiagnosticRun(root=root)
    try:
        config = json.loads((root / 'config.json').read_text())
        for folder in ('state', 'logs', 'workspace'):
            (root / folder).mkdir(exist_ok=True)
        if mode == 'auto':
            print('Detecting hardware...', flush=True)
            hardware = detect_hardware(config, root)
            config['_fit_hardware'] = hardware
            report.write('HOST HARDWARE', **hardware)
            print('Scanning installed GGUF models...', flush=True)
            rows = rank_models(config, hardware, root)
            report.write('INSTALLED GGUF MODELS', inventory=rows)
            report.write('MODEL RANKING', ranking=[dict(rank=i, path=row['path'],
                         classification=row['classification'], score=row['score'], reasons=row['reasons'])
                         for i, row in enumerate(rows, 1)])
            candidates = [Path(row['path']) for row in rows if row['eligible']]
            print('Selecting best compatible AI...', flush=True)
            report.write('SELECTION RATIONALE', policy='Highest deterministic score among eligible models; each tried once',
                         candidate_count=len(candidates))
        else:
            path = resolve(root, config['model']) if mode == 'default' else manual_model(root)
            if path is None:
                report.finish('manual selection cancelled')
                return
            candidates = [path]
            report.write('SELECTION RATIONALE', mode=mode, ranking_bypassed=True, selected_path=path)
            import platform
            compatible, reason = runtime_compatibility(resolve(root, config['server']), platform.machine())
            if not compatible:
                raise RuntimeError(reason)
            config['_fit_hardware'] = detect_hardware(config, root)
        if not candidates:
            report.finish('No eligible installed model available')
            print('No compatible AI available. Diagnostics: logs/llm-fit-diagnostics.txt')
            return
        # Selection does not start a server. The original menu owns the next action.
        selected = candidates[0]
        report.write('SELECTION', selected_path=selected, session_only_override=True,
                     rationale='Highest eligible score' if mode == 'auto' else mode,
                     next_screen='Original application menu; no inference started')
        print('Selected: ' + selected.name, flush=True)
        report.finish('Model selected; opening original application menu')
        from menu import main as original_menu
        return original_menu(active_model=str(selected))
    except KeyboardInterrupt:
        report.finish('interrupted; owned startup/session resources closed')
        print('\nReturned to startup menu. Diagnostics: logs/llm-fit-diagnostics.txt')
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        report.write('WARNINGS AND ERRORS', error=str(exc))
        report.finish('startup failed')
        print('AI startup failed. Diagnostics: logs/llm-fit-diagnostics.txt', flush=True)


def main():
    while True:
        print(MENU)
        try:
            choice = input('Selection: ').strip()
            if choice == '0':
                return 0
            if choice in ('1', '2', '3'):
                startup_cycle({'1': 'auto', '2': 'default', '3': 'manual'}[choice])
            else:
                print('Choose 0, 1, 2 or 3.')
        except KeyboardInterrupt:
            print('\nReturned to startup menu.')
        except EOFError:
            return 0
        except (OSError, ValueError, RuntimeError) as exc:
            print('Startup unavailable: ' + str(exc))


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
