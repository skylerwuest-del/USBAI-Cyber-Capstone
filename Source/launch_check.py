"""Offline launch preflight. Model availability is deliberately not a prerequisite."""
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def package_key(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def check_packages(root):
    """Check every pinned dependency using only the bundled installation."""
    distributions = {
        package_key(dist.metadata['Name']): dist.version
        for dist in importlib.metadata.distributions(path=[str(root / 'hermes/site-packages')])
        if dist.metadata['Name']
    }
    errors = []
    for line in (root / 'hermes-requirements.lock').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, version = line.split()[0].split('==', 1)
        installed = distributions.get(package_key(name))
        if installed != version:
            errors.append(name + ' requires ' + version + ' (installed: ' + (installed or 'missing') + ')')
    return errors


def check(root=ROOT):
    errors = []
    python = root / 'hermes/python/bin/python3.12'
    for path in (python, root / 'hermes/src/run_agent.py'):
        if not path.is_file():
            errors.append('Missing ' + str(path.relative_to(root)))
    if python.is_file() and not os.access(python, os.X_OK):
        errors.append('Portable Python is not executable')
    try:
        config = json.loads((root / 'config.json').read_text())
        server = root / config['server']
        if not server.is_file() or not os.access(server, os.X_OK):
            errors.append('Missing or non-executable model server: ' + str(server))
        else:
            env = dict(os.environ)
            env['LD_LIBRARY_PATH'] = os.pathsep.join((str(server.parent), str(root / 'runtime')))
            probe = subprocess.run([str(server), '--version'], cwd=root, env=env,
                                   capture_output=True, text=True, timeout=15)
            if probe.returncode:
                errors.append('Model server runtime cannot load: ' + (probe.stderr or probe.stdout).strip()[-2000:])
        errors.extend(check_packages(root))
        if not errors:
            env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root / 'hermes/site-packages'),
                                                            str(root / 'hermes/src'))))
            probe = subprocess.run([str(python), '-s', '-B', '-c',
                                    'from run_agent import AIAgent; import psutil'],
                                   cwd=root, env=env, capture_output=True, text=True, timeout=60)
            if probe.returncode:
                errors.append('Hermes dependencies cannot load:\n' + probe.stderr.strip()[-3000:])
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        errors.append(str(exc))
    return errors


def main():
    errors = check()
    if errors:
        print('Cannot launch: setup is incomplete or required dependencies are unavailable.', file=sys.stderr)
        for error in errors:
            print('  - ' + error, file=sys.stderr)
        print('Run: bash setup.sh\nFor repair using an existing offline cache: bash setup.sh --offline', file=sys.stderr)
        return 1
    print('Launch dependency check passed. Model loading is checked when you select a model.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
