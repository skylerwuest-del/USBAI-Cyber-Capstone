"""Terminal menu for Hermes using the CPU AirLLM model service."""
import json
from pathlib import Path
import subprocess
import sys

from usb_core import ROOT, atomic_json, instance_lock, resolve


def select_model(path):
    from capstone import runtime_config
    config_path = ROOT / 'airllm-config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    # Validate before committing the selection.
    runtime_config(json.loads((ROOT / 'config.json').read_text()), 'airllm', str(path))
    path = path.resolve()
    config['model'] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    with instance_lock(ROOT / 'state/runner.lock'):
        atomic_json(config_path, config)
    print('Selected AirLLM model: ' + config['model'])


def main():
    from menu import edit, run
    while True:
        path = ROOT / 'airllm-config.json'
        config = json.loads(path.read_text()) if path.exists() else {}
        print('''
AirLLM + Hermes — CPU, terminal only
1) Start interactive Hermes with AirLLM
2) Run/resume tasks.txt with AirLLM, then close processes and exit
3) Download/select a Transformers model from a Hugging Face URL
4) Select an installed Transformers model
5) Edit tasks.txt
6) Show AirLLM task progress
7) Repair/install cached AirLLM dependencies
Enter) Back''')
        print('Selected: ' + str(config.get('model') or '(choose a model first)'))
        try:
            choice = input('Choose: ').strip()
            if not choice:
                return None
            if choice == '1':
                run('--backend', 'airllm', '--chat')
            elif choice == '2':
                if run('--backend', 'airllm', '--run', '--tasks', 'tasks.txt') == 0:
                    return 'completed'
                print('Run stopped; unfinished progress retained.')
            elif choice == '3':
                value = input('Hugging Face Transformers repository URL: ').strip()
                if not value:
                    continue
                code = subprocess.call([sys.executable, '-s', '-B', str(ROOT / 'airllm_experiment.py'),
                                        '--prepare', '--model', value], cwd=ROOT)
                if code:
                    print('Model preparation failed; previous selection retained.')
            elif choice == '4':
                models = sorted(p.parent for p in (ROOT / 'airllm/models').glob('*/config.json'))
                if not models:
                    print('No downloaded Transformers models; use option 3.')
                    continue
                for index, model in enumerate(models, 1):
                    print(str(index) + ') ' + model.name)
                value = input('Number, or path to a local Transformers directory: ').strip()
                if value:
                    select_model(models[int(value)-1] if value.isdigit() and 1 <= int(value) <= len(models)
                                 else resolve(ROOT, value))
            elif choice == '5':
                edit('tasks.txt')
            elif choice == '6':
                run('--backend', 'airllm', '--status')
            elif choice == '7':
                subprocess.call([sys.executable, '-s', '-B', str(ROOT / 'airllm_experiment.py'), '--install'], cwd=ROOT)
        except (OSError, ValueError, RuntimeError) as exc:
            print('AirLLM: ' + str(exc))
        except KeyboardInterrupt:
            print('\nStopped; returned to AirLLM menu.')


if __name__ == '__main__':
    main()
