"""Optional, separate CPU layer-streaming experiment. Not the Hermes model server."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
from portable_env import activate
activate(ROOT)
AIR = ROOT / 'airllm'
os.environ.setdefault('SSL_CERT_FILE', str(ROOT / 'hermes/site-packages/certifi/cacert.pem'))


@contextlib.contextmanager
def experiment_lock():
    (ROOT / 'state').mkdir(exist_ok=True)
    with (ROOT / 'state' / 'runner.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another CapStone run is using this USB') from None
        try:
            yield
        finally:
            subprocess.run(['sync', '-f', str(ROOT)], check=True)


def install():
    python = ROOT / 'hermes' / 'python' / 'bin' / 'python3.12'
    pending = AIR / 'site-packages.pending'
    if pending.exists():
        shutil.rmtree(pending)
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PIP_CONFIG_FILE=os.devnull)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--no-cache-dir',
                    '--find-links', str(AIR / 'wheels'), '--require-hashes', '--no-deps', '--no-compile',
                    '--target', str(pending), '-r', str(ROOT / 'airllm-requirements.lock')], env=env, check=True)
    env['PYTHONPATH'] = str(pending)
    subprocess.run([str(python), '-s', '-B', '-c', 'import torch,airllm; assert torch.version.cuda is None; print("AirLLM CPU imports passed")'], env=env, check=True)
    packages = AIR / 'site-packages'
    if packages.exists():
        shutil.rmtree(packages)
    pending.rename(packages)


def model_info(value):
    if value.startswith('https://'):
        url = urllib.parse.urlparse(value)
        if url.hostname != 'huggingface.co':
            raise ValueError('Use a Hugging Face repository URL')
        parts = url.path.strip('/').split('/')
        if len(parts) != 2:
            raise ValueError('Use https://huggingface.co/owner/repository (Transformers weights, not GGUF)')
        value = '/'.join(parts)
    if len(value.split('/')) != 2 or any(x in ('', '.', '..') for x in value.split('/')):
        raise ValueError('Use owner/repository or a Hugging Face repository URL')
    endpoint = 'https://huggingface.co/api/models/' + urllib.parse.quote(value, safe='/') + '?blobs=true'
    from setup import tls_context
    with urllib.request.urlopen(endpoint, context=tls_context(), timeout=30) as response:
        info = json.load(response)
    weights = [f for f in info['siblings'] if f['rfilename'].endswith('.safetensors')]
    if not weights:
        weights = [f for f in info['siblings'] if f['rfilename'].endswith('.bin') and 'pytorch_model' in f['rfilename']]
    if not weights:
        raise ValueError('This repository has no Transformers weights; AirLLM does not load GGUF')
    total = sum(f.get('size', 0) for f in weights)
    if total <= 0:
        raise ValueError('Model size is unknown; cannot assess USB storage requirements')
    return value, info['sha'], weights, total


def fetch_model(value):
    repo, revision, weights, total = model_info(value)
    # Originals and layer files both exist; deliberately keep originals.
    needed = total * 2 + 1024 ** 3
    available = shutil.disk_usage(ROOT).free
    print('Weights: %.2f GB; originals + layers + reserve: %.2f GB; free USB: %.2f GB'
          % (total / 1e9, needed / 1e9, available / 1e9), flush=True)
    if needed > available:
        raise RuntimeError('This model does not fit on this USB with its layer files. Select a smaller model.')
    from huggingface_hub import snapshot_download
    destination = AIR / 'models' / (repo.replace('/', '--') + '--' + revision[:12])
    filenames = [f['rfilename'] for f in weights]
    patterns = filenames + ['*.json', '*.model', '*.txt', '*.tiktoken', '*.jinja']
    snapshot_download(repo, revision=revision, local_dir=str(destination), allow_patterns=patterns)
    for entry in weights:
        path = destination / entry['rfilename']
        expected = entry.get('lfs', {}).get('sha256')
        if not expected:
            raise RuntimeError('Missing published weight checksum: ' + entry['rfilename'])
        with path.open('rb') as handle:
            os.fsync(handle.fileno())
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != expected:
            raise RuntimeError('Weight checksum mismatch: ' + str(path))
    (destination / 'capstone-source.json').write_text(json.dumps({'repository': repo, 'revision': revision, 'weights': weights}, indent=2))
    return destination


def load_streaming_model(model_path, context=512):
    import torch
    from airllm import AutoModel
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    # Distinct local checkpoints with the same directory name must not share layers.
    model_path = model_path.resolve()
    identity = str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path)
    stamps = [(str(p.relative_to(model_path)), p.stat().st_size, p.stat().st_mtime_ns)
              for pattern in ('*.safetensors', '*.bin', 'config.json') for p in sorted(model_path.rglob(pattern))]
    cache_key = model_path.name + '-' + hashlib.sha256(json.dumps([identity, stamps]).encode()).hexdigest()[:16]
    return AutoModel.from_pretrained(str(model_path), device='cpu', dtype=torch.float32,
                                      max_seq_len=context, prefetching=False,
                                      layer_shards_saving_path=str(AIR / 'layers' / cache_key),
                                      delete_original=False)


def infer(model_path, prompt, tokens=16, context=512):
    started = time.monotonic()
    model = load_streaming_model(model_path, context)
    tokenizer = model.tokenizer
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tokenizer(text, return_tensors='pt', return_attention_mask=False)['input_ids']
    if inputs.shape[1] + tokens > context:
        raise ValueError('Prompt plus generated tokens exceeds experimental context size')
    result = model.generate(inputs, max_new_tokens=tokens, use_cache=True, return_dict_in_generate=True)
    generated = result.sequences[0, inputs.shape[1]:]
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print('Generated %d tokens in %.1f seconds including model loading.' % (len(generated), time.monotonic() - started))
    return {'tokens': len(generated), 'seconds': time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--menu', action='store_true')
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--prepare', action='store_true', help='Download and select a Transformers model for Hermes')
    parser.add_argument('--model', help='Hugging Face repository URL, owner/repo, or existing local model directory')
    parser.add_argument('--prompt', default='Explain what a USB drive is in one sentence.')
    parser.add_argument('--tokens', type=int, default=16)
    parser.add_argument('--context', type=int, default=512)
    args = parser.parse_args()
    if args.menu:
        print('Standalone AirLLM CPU diagnostic. For AirLLM + Hermes, use launch menu 9.\n'
              'It needs Transformers weights and a second set of layer files. GGUF uses llama.cpp.\n'
              'Large models may not fit this 59 GB USB and can be very slow without a GPU.\n'
              '1) Repair/install the cached CPU environment\n2) Download a model and run a short prompt\n'
              '3) Test CPU layer streaming with the tiny synthetic model\nEnter) Back')
        choice = input('Choose: ').strip()
        args.install = choice == '1'
        if choice == '2':
            args.model = input('Hugging Face repository URL or local model directory: ').strip()
            args.prompt = input('Prompt: ').strip() or args.prompt
        if choice == '3':
            args.model, args.prompt, args.tokens, args.context = str(AIR / 'smoke-model'), 'hello world', 4, 128
        if not args.install and not args.model:
            return 0
    if args.install:
        with experiment_lock():
            install()
        return 0
    if not (AIR / 'site-packages' / 'airllm').exists():
        raise RuntimeError('Repair AirLLM dependencies using menu option 9, then 7')
    # Keep its torch/transformers versions completely separate from Hermes.
    sys.path[:0] = [str(AIR / 'site-packages')]
    os.environ.update(HF_HOME=str(AIR / 'cache'), HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_SYMLINKS_WARNING='1')
    if args.model:
        with experiment_lock():
            path = Path(args.model).expanduser()
            if not path.is_dir():
                path = fetch_model(args.model)
            if args.prepare:
                metadata = json.loads((path / 'config.json').read_text())
                if int(metadata.get('text_config', metadata).get('max_position_embeddings', 16384)) < 8192:
                    raise ValueError('Choose an instruction model with at least 8192 tokens of context')
                settings_path = ROOT / 'airllm-config.json'
                settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
                path = path.resolve()
                settings['model'] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                # Import only the stdlib persistence helper; AirLLM packages remain first.
                from usb_core import atomic_json
                atomic_json(settings_path, settings)
                print('Selected AirLLM model for Hermes: ' + settings['model'])
            else:
                infer(path.resolve(), args.prompt, args.tokens, args.context)
    else:
        parser.print_help()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print('AirLLM: ' + str(exc))
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
