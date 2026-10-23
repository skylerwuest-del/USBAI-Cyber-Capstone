"""The USB's terminal menu. No web frontend."""
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys

from usb_core import ROOT, instance_lock
from task_loop import load_tasks


def edit(name):
    path = ROOT / name
    with instance_lock(ROOT / 'state' / 'runner.lock'):
        if not path.exists():
            path.write_text('# Put your instructions here.\n')
        shutil.copy2(path, path.with_suffix(path.suffix + '.bak'))
        selected = os.getenv('CAPSTONE_EDITOR') or os.getenv('EDITOR')
        editor = shlex.split(selected) if selected else [next((x for x in ('nano', 'vim', 'vi') if shutil.which(x)), '')]
        if not editor or not editor[0]:
            raise RuntimeError('No terminal editor found; set CAPSTONE_EDITOR or edit ' + str(path))
        subprocess.run(editor + [str(path)], check=True)
        with path.open('rb') as handle:
            os.fsync(handle.fileno())
        if name == 'tasks.txt':
            tasks = load_tasks(path)
            print('Saved ' + str(len(tasks)) + ' task(s). VERIFY lines are optional.')


def run(*args, active_model=None):
    if active_model and '--backend' not in args:
        args = (*args, '--model', str(active_model))
        if '--chat' in args:
            args = (*args, '--fit-session')
    process = subprocess.Popen([sys.executable, '-s', '-B', str(ROOT / 'capstone.py'), *args], cwd=ROOT, start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        # Give the supervisor time to terminate its separately grouped model and
        # agents. subprocess.call's short interrupt grace can orphan that server.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                return process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                from usb_core import stop_process
                stop_process(process)
                return 130
        finally:
            signal.signal(signal.SIGINT, previous)


def main(active_model=None):
    while True:
        if active_model:
            print('\nSession GGUF: ' + os.path.basename(str(active_model)))
        print('''
CapStone — local Hermes, terminal only
1) Start interactive Hermes (model + terminal/file tools)
2) Edit preordained tasks (tasks.txt)
3) Run/resume tasks, then close all CapStone processes and exit
4) Show task progress
5) Pull a GGUF model from a Hugging Face URL
6) Select an installed GGUF model
7) Edit the AI's standing instructions (prompt.txt)
8) Serve the local model only
9) AirLLM + Hermes (interactive tools or autonomous tasks)
10) USB toolkit (tools, local self-test, offline repair)
11) Test installed GGUF models (optional performance benchmark)
0) Exit''')
        try:
            choice = input('Choose: ').strip()
            if choice == '0':
                return 0
            if choice == '1':
                run('--chat', active_model=active_model)
            elif choice == '2':
                edit('tasks.txt')
            elif choice == '3':
                code = run('--run', '--tasks', 'tasks.txt', active_model=active_model)
                if code == 0:
                    return 0
                print('Run interrupted or failed; progress retained. Exit code: ' + str(code))
            elif choice == '4':
                run('--status')
            elif choice in ('5', '6'):
                from models_menu import pull, choose_installed
                selected = (pull if choice == '5' else choose_installed)()
                if selected is not None:
                    active_model = str(selected)
            elif choice == '7':
                edit('prompt.txt')
            elif choice == '8':
                run('--serve-only', active_model=active_model)
            elif choice == '11':
                from fit_benchmark import run_benchmarks
                run_benchmarks()
            elif choice == '10':
                from toolkit_tools import menu as toolkit_menu
                toolkit_menu()
            elif choice == '9':
                from airllm_menu import main as airllm_menu
                if airllm_menu() == 'completed':
                    return 0
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            print(str(exc))
        except KeyboardInterrupt:
            print('\nReturned to menu.')
        except EOFError:
            return 0


if __name__ == '__main__':
    raise SystemExit(main())
