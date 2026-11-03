"""Persistent USB job supervisor used by autonomous CapStone tasks.

The visible CLI exits after starting a detached supervisor.  The supervisor owns the
real command, records its exit status, and keeps running when a Hermes turn ends.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / 'state' / 'jobs'
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def job_dir(name):
    if not NAME.fullmatch(name):
        raise ValueError('Job names may contain letters, numbers, dot, underscore and dash')
    return JOBS / name


def read_state(name):
    path = job_dir(name) / 'job.json'
    if not path.is_file():
        raise FileNotFoundError('No durable job named ' + name)
    return json.loads(path.read_text(encoding='utf-8'))


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        stat = Path('/proc') / str(pid) / 'stat'
        return not stat.exists() or stat.read_text().split()[2] != 'Z'
    except (OSError, ValueError, IndexError):
        return False


def refreshed(name):
    state = read_state(name)
    if (state.get('status') in {'starting', 'running'} and state.get('supervisor_pid')
            and not pid_alive(state.get('supervisor_pid', 0))):
        state.update(status='lost', finished_at=now(), error='Supervisor disappeared before recording command completion')
        atomic_json(job_dir(name) / 'job.json', state)
    state['alive'] = state.get('status') in {'starting', 'running'} and pid_alive(state.get('supervisor_pid', 0))
    return state


def supervise(name, command, cwd):
    folder = job_dir(name)
    path = folder / 'job.json'
    state = json.loads(path.read_text(encoding='utf-8'))
    state.update(status='running', supervisor_pid=os.getpid(), started_at=now())
    atomic_json(path, state)
    code = 125
    error = ''
    try:
        with (folder / 'output.log').open('ab', buffering=0) as output:
            output.write(('\n=== CapStone durable job started ' + now() + ' ===\n').encode())
            process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
            state.update(command_pid=process.pid)
            atomic_json(path, state)
            code = process.wait()
            output.write(('\n=== CapStone durable job exited ' + str(code) + ' at ' + now() + ' ===\n').encode())
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    state = json.loads(path.read_text(encoding='utf-8'))
    state.update(status='completed' if code == 0 else 'failed', exit_code=code,
                 finished_at=now(), error=error)
    atomic_json(path, state)
    return 0 if code == 0 else 1


def start(args):
    folder = job_dir(args.name)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / '.starting'
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError('Another start operation is already in progress for ' + args.name)
    try:
        state_path = folder / 'job.json'
        if state_path.exists():
            previous = refreshed(args.name)
            if previous.get('status') in {'starting', 'running'}:
                raise RuntimeError('Job is already running; inspect it instead of starting a duplicate')
            if not args.restart:
                raise RuntimeError('Job already has a recorded result; use restart only when resuming a failed job')
            history = folder / 'history'
            history.mkdir(exist_ok=True)
            atomic_json(history / (str(time.time_ns()) + '.json'), previous)
        command = list(args.command)
        if command and command[0] == '--':
            command.pop(0)
        if not command:
            raise ValueError('A command is required after --')
        cwd = str(Path(args.cwd or os.getcwd()).resolve())
        state = {'version': 1, 'name': args.name, 'status': 'starting', 'created_at': now(),
                 'cwd': cwd, 'command': command,
                 'queue_task_id': os.environ.get('CAPSTONE_QUEUE_TASK_ID', ''),
                 'output_log': str(folder / 'output.log')}
        atomic_json(state_path, state)
        process = subprocess.Popen([sys.executable, '-s', '-B', str(Path(__file__).resolve()),
                                    '_supervise', args.name, cwd, '--', *command],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            current = json.loads(state_path.read_text(encoding='utf-8'))
            if current.get('status') != 'starting':
                current['alive'] = pid_alive(current.get('supervisor_pid', 0))
                print(json.dumps(current, sort_keys=True))
                return 0
            if process.poll() is not None:
                raise RuntimeError('Durable supervisor exited before it started the command')
            time.sleep(0.05)
        raise RuntimeError('Durable supervisor did not start')
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


def status(args):
    print(json.dumps(refreshed(args.name), indent=2, sort_keys=True))
    return 0


def wait_job(args):
    while True:
        state = refreshed(args.name)
        if state.get('status') not in {'starting', 'running'}:
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0 if state.get('status') == 'completed' else 1
        print('Durable job ' + args.name + ' is still running (pid ' + str(state.get('command_pid', '?')) + ')', flush=True)
        time.sleep(args.poll)


def stop(args):
    state = refreshed(args.name)
    if state.get('status') not in {'starting', 'running'}:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    pid = int(state.get('command_pid') or 0)
    if pid <= 1:
        raise RuntimeError('Job has no command PID yet; retry status shortly')
    os.killpg(pid, signal.SIGTERM)
    print('Stop requested for ' + args.name)
    return 0


def list_jobs(_args):
    values = []
    if JOBS.exists():
        for path in sorted(JOBS.iterdir()):
            if (path / 'job.json').is_file():
                try:
                    values.append(refreshed(path.name))
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
    print(json.dumps(values, indent=2, sort_keys=True))
    return 0


def show_log(args):
    lines = (job_dir(args.name) / 'output.log').read_text(encoding='utf-8', errors='replace').splitlines()
    print('\n'.join(lines[-args.lines:]))
    return 0


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest='action', required=True)
    item = commands.add_parser('start')
    item.add_argument('name')
    item.add_argument('--cwd')
    item.add_argument('--restart', action='store_true')
    item.add_argument('command', nargs=argparse.REMAINDER)
    item.set_defaults(function=start)
    item = commands.add_parser('status'); item.add_argument('name'); item.set_defaults(function=status)
    item = commands.add_parser('wait'); item.add_argument('name'); item.add_argument('--poll', type=float, default=60); item.set_defaults(function=wait_job)
    item = commands.add_parser('stop'); item.add_argument('name'); item.set_defaults(function=stop)
    item = commands.add_parser('log'); item.add_argument('name'); item.add_argument('--lines', type=int, default=80); item.set_defaults(function=show_log)
    commands.add_parser('list').set_defaults(function=list_jobs)
    item = commands.add_parser('_supervise'); item.add_argument('name'); item.add_argument('cwd'); item.add_argument('command', nargs=argparse.REMAINDER)
    item.set_defaults(function=lambda a: supervise(a.name, a.command[1:] if a.command[:1] == ['--'] else a.command, a.cwd))
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if getattr(args, 'poll', 1) <= 0:
        raise ValueError('Poll interval must be positive')
    return args.function(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print('capstone-job: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
