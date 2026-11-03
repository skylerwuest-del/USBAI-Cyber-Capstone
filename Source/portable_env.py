"""Keep application caches, downloads and scratch files alongside the USB app."""
import os
from pathlib import Path
import tempfile


def usb_environment(root, source=None):
    root = Path(root).resolve()
    env = dict(os.environ if source is None else source)
    locations = {
        'TMPDIR': 'state/tmp', 'TMP': 'state/tmp', 'TEMP': 'state/tmp',
        'XDG_CACHE_HOME': 'state/cache', 'XDG_CONFIG_HOME': 'state/config',
        'XDG_DATA_HOME': 'state/data', 'XDG_STATE_HOME': 'state',
        'HF_HOME': 'state/cache/huggingface', 'HUGGINGFACE_HUB_CACHE': 'state/cache/huggingface/hub',
        'TORCH_HOME': 'state/cache/torch', 'PIP_CACHE_DIR': 'state/cache/pip',
        'CUDA_CACHE_PATH': 'state/cache/cuda', 'CUDA_CACHE_DISABLE': '1',
        'MESA_SHADER_CACHE_DIR': 'state/cache/mesa',
        '__GL_SHADER_DISK_CACHE_PATH': 'state/cache/nvidia',
        'LLAMA_CACHE': 'state/llama-cache', 'HERMES_HOME': 'state/hermes',
        'HISTFILE': 'state/shell-history',
    }
    for key, relative in locations.items():
        if key == 'CUDA_CACHE_DISABLE':
            env[key] = relative
            continue
        path = root / relative
        if key == 'HISTFILE':
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CONFIG_FILE=os.devnull)
    return env


def activate(root):
    os.environ.update(usb_environment(root))
    # tempfile may already have cached the host's /tmp before activation.
    tempfile.tempdir = os.environ['TMPDIR']
