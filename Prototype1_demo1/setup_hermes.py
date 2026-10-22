"""Repair/reinstall the pinned Hermes bundle using USB archives and binary wheels."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from portable_env import activate

from usb_core import ROOT, atomic_json, instance_lock
from setup import download, sha256
activate(ROOT)

BUNDLE = ROOT / 'hermes'
PYTHON = BUNDLE / 'python' / 'bin' / 'python3.12'
EXCLUDE = {'tests', 'tests-js', 'website', 'web', 'apps', 'ui-tui', 'evals',
           'contributors', 'nix', 'docker', '.github', '.git'}


def extract(archive, destination, source=False):
    with tempfile.TemporaryDirectory(prefix='capstone-extract-') as temp:
        base = Path(temp)
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar.getmembers():
                target = (base / member.name).resolve()
                if not target.is_relative_to(base) or not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise ValueError('Unsafe archive member')
                if member.issym() or member.islnk():
                    link = (target.parent / member.linkname).resolve() if member.issym() else (base / member.linkname).resolve()
                    if not link.is_relative_to(base):
                        raise ValueError('Archive link escapes extraction directory')
            # Scratch is on USB too: exFAT cannot create the archive's symlinks.
            # Materialize linked files using tar's link resolver instead.
            links = []
            for member in tar.getmembers():
                target = base / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile() or member.issym() or member.islnk():
                    if member.issym() or member.islnk():
                        links.append(member)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, target.open('wb') as out:
                        shutil.copyfileobj(src, out)
                    target.chmod(member.mode & 0o777)
            for member in links:
                target = base / member.name
                linked = ((target.parent / member.linkname) if member.issym()
                          else (base / member.linkname)).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                if linked.is_dir():
                    shutil.copytree(linked, target, dirs_exist_ok=True)
                else:
                    with tar.extractfile(member) as src, target.open('wb') as out:
                        shutil.copyfileobj(src, out)
                    target.chmod(linked.stat().st_mode & 0o777 if linked.exists() else 0o755)
        roots = list(base.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError('Expected a single archive root')
        root = roots[0]
        ignore = (lambda directory, names: [name for name in names if name in {'__pycache__', 'node_modules'}
                                           or source and Path(directory) == root and name in EXCLUDE])
        # exFAT cannot create symlinks; preserve their content as regular files.
        shutil.copytree(root, destination, symlinks=False, dirs_exist_ok=True, ignore=ignore)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    assets = json.loads((ROOT / 'hermes-assets.json').read_text())
    with instance_lock(ROOT / 'state' / 'runner.lock'):
        for key, filename, dest in [('python', 'python.tar.gz', 'python'), ('hermes', 'source.tar.gz', 'src')]:
            archive = BUNDLE / 'cache' / filename
            if args.offline:
                if not archive.is_file() or sha256(archive) != assets[key]['sha256']:
                    raise RuntimeError('Missing or corrupt offline archive: ' + str(archive))
            else:
                download(assets[key], archive)
            required = BUNDLE / dest / ('bin/python3.12' if key == 'python' else 'run_agent.py')
            if not required.is_file():
                extract(archive, BUNDLE / dest, source=key == 'hermes')
        PYTHON.chmod(0o755)
        pending = BUNDLE / 'site-packages.pending'
        if pending.exists():
            shutil.rmtree(pending)
        env = {key: value for key, value in os.environ.items() if not key.startswith('PIP_') and key != 'VIRTUAL_ENV'}
        env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                   PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1',
                   HERMES_HOME=str(ROOT / 'state' / 'hermes'))
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)
        if not args.offline:
            # First preparation must populate the exact wheel cache used by offline repair.
            wheels = BUNDLE / 'wheels'
            wheels.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(PYTHON), '-m', 'pip', 'download', '--disable-pip-version-check',
                            '--index-url', 'https://pypi.org/simple', '--only-binary=:all:',
                            '--require-hashes', '--no-deps', '--dest', str(wheels),
                            '-r', str(ROOT / 'hermes-requirements.lock')], env=env, check=True)
        subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--no-index', '--no-cache-dir',
                        '--find-links', str(BUNDLE / 'wheels'), '--require-hashes', '--no-deps',
                        '--no-compile', '--target', str(pending), '-r', str(ROOT / 'hermes-requirements.lock')],
                       env=env, check=True)
        env['PYTHONPATH'] = str(pending) + os.pathsep + str(BUNDLE / 'src')
        subprocess.run([str(PYTHON), '-s', '-B', '-c', 'from run_agent import AIAgent; import psutil; print("Hermes SDK imports passed")'], env=env, check=True)
        previous = BUNDLE / 'site-packages.previous'
        if previous.exists():
            shutil.rmtree(previous)
        subprocess.run(['sync', '-f', str(ROOT)], check=True)
        packages = BUNDLE / 'site-packages'
        if packages.exists():
            packages.rename(previous)
        pending.rename(packages)
        atomic_json(BUNDLE / 'installed.json', {'version': assets['hermes']['version'],
                    'revision': assets['hermes']['revision'], 'lock_sha256': sha256(ROOT / 'hermes-requirements.lock')})
        if previous.exists():
            shutil.rmtree(previous)
        subprocess.run(['sync', '-f', str(ROOT)], check=True)
    print('Portable Hermes ready. Start with bash launch.sh')


if __name__ == '__main__':
    main()
