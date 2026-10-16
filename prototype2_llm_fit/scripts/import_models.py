#!/usr/bin/env python3
"""Register complete local GGUF files from models/drop-in; never use the network."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gguf_check import validate_structure


def build_entry(path, root):
    before = path.stat()
    metadata, parameters = validate_structure(path)
    arch = metadata.get('general.architecture')
    # Hybrid/recurrent/MoE architectures need a separate memory model.
    if arch not in ('llama', 'qwen2', 'qwen3') or any(
            'ssm' in key or 'expert' in key and value != 0 for key, value in metadata.items()):
        raise ValueError('Automatic memory estimation supports dense llama/qwen2/qwen3 only; use a reviewed manual entry for this architecture')
    if metadata.get('split.count', 1) != 1 or metadata.get('general.type', 'model') != 'model':
        raise ValueError('Only complete, single-file models can be imported')
    def integer(key):
        value = metadata.get(arch + '.' + key)
        if type(value) is not int or value <= 0:
            raise ValueError('Missing positive GGUF metadata: ' + arch + '.' + key)
        return value
    context = min(4096, integer('context_length'))
    if context < 4096:
        raise ValueError('Agent requires at least 4096 tokens of native context')
    layers, embedding = integer('block_count'), integer('embedding_length')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('File changed while being imported; finish copying then retry')
    return dict(name=path.stem, file=path.relative_to(root).as_posix(),
                parameters_b=parameters / 1e9,
                quantization='GGUF file_type ' + str(metadata.get('general.file_type', 'unspecified')),
                context_size=context, estimated_ram_gb=before.st_size / 1024**3 * 1.25 + 1,
                kv_bytes_per_token=layers * embedding * 8 * 2,
                priority=max(1, round(parameters / 1e7)), sha256=digest.hexdigest(),
                auto_imported=True,
                estimate_note='Unmeasured conservative dense-attention estimate; FP32 full-width K+V with 2x margin')


def sync(root=ROOT):
    store = root / 'models'
    drop = store / 'drop-in'
    if store.is_symlink() or drop.is_symlink() or not drop.resolve().is_relative_to(root.resolve()):
        raise ValueError('Model drop folder must stay inside this USB project')
    drop.mkdir(exist_ok=True)
    manifest = store / 'models.json'
    if manifest.is_symlink():
        raise ValueError('Manifest must not be a symbolic link')
    with (store / '.import.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        doc = json.loads(manifest.read_text())
        if not isinstance(doc, dict) or not isinstance(doc.get('models'), list) or any(not isinstance(e, dict) for e in doc['models']):
            raise ValueError('Invalid models.json; refusing to overwrite it')
        existing = {e.get('file'): e for e in doc['models']}
        entries = [e for e in doc['models'] if not e.get('auto_imported')]
        for path in sorted(drop.glob('*.gguf')):
            try:
                if path.is_symlink() or not path.is_file():
                    raise ValueError('Only regular files are accepted')
                relative = path.relative_to(root).as_posix()
                old = existing.get(relative)
                if old and not old.get('auto_imported'):
                    continue  # Preserve all manually reviewed settings.
                entry = build_entry(path, root)
                if old and old.get('sha256') == entry['sha256']:
                    entry = old
                entries.append(entry)
                if entry != old:
                    print('Registered dropped model: ' + path.name, file=sys.stderr)
            except (OSError, ValueError, RuntimeError) as exc:
                print('Skipped ' + path.name + ': ' + str(exc), file=sys.stderr)
        if entries == doc['models']:
            return
        doc['models'] = entries
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=store, prefix='.models-', delete=False) as handle:
                temp = Path(handle.name)
                json.dump(doc, handle, indent=2)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, manifest)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        sync()
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit('Model import failed: ' + str(exc))
