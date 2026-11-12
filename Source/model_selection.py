"""Resolve a saved model file or a folder whose contents can change offline."""
from pathlib import Path


def selected_model(root, selection):
    path = Path(selection)
    path = path if path.is_absolute() else Path(root) / path
    # Recognize the configured folder even before it has been populated.
    if path.is_dir() or path.suffix.lower() != '.gguf':
        from models_menu import selectable_names, shard_group
        files = {str(item.relative_to(path)): item for item in path.rglob('*.gguf')
                 if item.is_file() and item.resolve().is_relative_to(path.resolve())}
        for name in selectable_names(files):
            try:
                shard_group(name, files)
            except ValueError:
                continue
            return files[name].resolve()
        raise RuntimeError('No complete GGUF model in ' + str(path)
                           + '; place a model in this folder, then retry.')
    return path.resolve()
