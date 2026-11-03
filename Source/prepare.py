"""Single preparation workflow: online first use, then verified offline repairs."""
import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

from usb_core import ROOT, atomic_json, instance_lock


def check_settings_access(root):
    """Fail before downloads when existing private settings belong to another user."""
    for relative in ('config.json', 'state/hermes/config.yaml', 'state/llm-fit-performance.json'):
        path = root / relative
        if relative != 'config.json' and not path.exists():
            continue
        try:
            # Both startup and later settings/telemetry updates must work. Opening
            # r+ does not truncate or modify the user's configuration.
            with path.open('r+', encoding='utf-8') as handle:
                if relative == 'config.json':
                    config = json.load(handle)
                    if not isinstance(config, dict):
                        raise ValueError('config.json must contain a JSON object')
        except PermissionError as exc:
            command = 'sudo chown ' + str(os.getuid()) + ':' + str(os.getgid()) + ' ' + shlex.quote(str(path))
            raise RuntimeError('Cannot read/write project settings: ' + str(path)
                               + '. Owner UID: ' + str(path.stat().st_uid)
                               + '; current UID: ' + str(os.getuid())
                               + '. Restore ownership with: ' + command
                               + '; then rerun bash setup.sh. Do not run setup with sudo.') from exc


def verify_session():
    from fit_diagnostics import DiagnosticRun
    from llm_fit import detect_hardware, rank_models
    from model_startup import launch_candidate
    config = json.loads((ROOT / 'config.json').read_text())
    hardware = detect_hardware(config)
    rows = rank_models(config, hardware)
    eligible = [row for row in rows if row['eligible']]
    if not eligible:
        raise RuntimeError('No installed GGUF currently fits this host; see model startup diagnostics')
    selected = min(eligible, key=lambda row: (row['file_size'], row['path']))
    report = DiagnosticRun('SETUP REAL SERVER AND HERMES SMOKE TEST')
    config['_fit_hardware'] = hardware
    with instance_lock(ROOT / 'state/runner.lock'):
        started, outcome = launch_candidate(Path(selected['path']), config, report)
    report.finish(outcome, purpose='Initialize real server and SDK, reach You>, then quit; no user task executed')
    if not started or outcome != 'interactive session completed':
        raise RuntimeError('Real Hermes startup verification failed; see logs/setup.log and llm-fit-diagnostics.txt')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Repair exclusively from an already prepared cache')
    parser.add_argument('--verify-session', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.verify_session:
        verify_session()
        return 0
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('This pinned setup supports Linux x86_64; a matching bundle is needed for other hosts')
    check_settings_access(ROOT)
    (ROOT / 'logs').mkdir(exist_ok=True)
    (ROOT / 'state').mkdir(exist_ok=True)
    offline = ['--offline'] if args.offline else []
    model_files = list((ROOT / 'models').rglob('*.gguf'))
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    for key in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV'):
        env.pop(key, None)
    def run(label, command, **kwargs):
        print(label, flush=True)
        with (ROOT / 'logs/setup.log').open('a') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, **kwargs)
        if result.returncode:
            raise RuntimeError(label + ' failed. See logs/setup.log; rerun bash setup.sh to resume.')
    with instance_lock(ROOT / 'state/setup.lock'):
        atomic_json(ROOT / 'state/setup-ready.json', dict(ready=False, stage='preparing'))
        with instance_lock(ROOT / 'state/runner.lock'):
            run('Preparing llama.cpp and cached runtime libraries...',
                [sys.executable, '-B', str(ROOT / 'setup.py'), *(['--runtime-only'] if model_files else []), *offline])
        run('Preparing portable Python, Hermes SDK and verified dependency wheels...',
            [sys.executable, '-B', str(ROOT / 'setup_hermes.py'), *offline])
        python = ROOT / 'hermes/python/bin/python3.12'
        run('Checking local application assets...', [str(python), '-s', '-B', str(ROOT / 'capstone.py'), '--check'])
        run('Verifying real model-server readiness and Hermes interactive startup (no tasks executed)...',
            [str(python), '-s', '-B', str(ROOT / 'prepare.py'), '--verify-session'], input='/quit\n', text=True, timeout=900)
        atomic_json(ROOT / 'state/setup-ready.json', dict(ready=True, offline_cache=True,
                    host=platform.platform(), verification='Real installed GGUF server and Hermes interactive prompt initialized'))
        subprocess.run(['sync', '-f', str(ROOT)], check=True)
    print('Setup complete. Your installed models were preserved.\nStart the application: bash launch.sh\nFuture offline repair: bash setup.sh --offline', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nSetup interrupted; rerun bash setup.sh to resume.', flush=True)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print('Setup: ' + str(exc), flush=True)
        raise SystemExit(1)
