import io
import os
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hardware_runtime as runtime
import portable_env
import setup
import setup_hermes


class HardwareRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.config = {'server': 'runtime/llama-server'}

    def server(self, backend, machine=62):
        path = self.root / 'runtime' / backend / 'llama-server'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'\x7fELF\x02\x01' + bytes(12) + struct.pack('<H', machine))
        path.chmod(0o755)
        return path

    def test_cuda_selected_and_offload_enabled(self):
        server = self.server('cuda')
        with patch.object(runtime.platform, 'machine', return_value='x86_64'), \
             patch.object(runtime.platform, 'system', return_value='Linux'), \
             patch.object(runtime.subprocess, 'run', return_value=Mock(returncode=0, stdout='CUDA0: GPU (8192 MiB, 7000 MiB free)', stderr='')):
            selected = runtime.select_runtime(self.config, self.root)
        self.assertEqual(selected['server'], str(server))
        self.assertTrue(selected['gpu'])
        self.assertIn('auto', runtime.acceleration_args(selected, self.config))

    def test_failed_cuda_probe_falls_back_to_vulkan(self):
        self.server('cuda')
        vulkan = self.server('vulkan')
        with patch.object(runtime.platform, 'machine', return_value='x86_64'), \
             patch.object(runtime.platform, 'system', return_value='Linux'), \
             patch.object(runtime.subprocess, 'run', side_effect=[OSError('missing driver'), Mock(returncode=0, stdout='Vulkan0: GPU', stderr='')]):
            selected = runtime.select_runtime(self.config, self.root)
        self.assertEqual(selected['server'], str(vulkan))
        self.assertTrue(selected['probe_errors'])

    def test_cpu_fallback_for_no_devices_and_wrong_abi(self):
        self.server('cuda', machine=183)
        self.server('vulkan')
        with patch.object(runtime.platform, 'machine', return_value='x86_64'), \
             patch.object(runtime.platform, 'system', return_value='Linux'), \
             patch.object(runtime.subprocess, 'run', return_value=Mock(returncode=0, stdout='Available devices:', stderr='')) as probe:
            selected = runtime.select_runtime(self.config, self.root)
        self.assertFalse(selected['gpu'])
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(runtime.acceleration_args(selected, {}), ['--n-gpu-layers', '0'])

    def test_forced_cpu_does_not_probe(self):
        with patch.object(runtime.subprocess, 'run') as probe:
            selected = runtime.select_runtime(dict(self.config, acceleration='cpu'), self.root)
        self.assertFalse(selected['gpu'])
        probe.assert_not_called()

    def test_strict_offline_entry_retains_manifest_context_and_cpu_policy(self):
        import capstone
        from scripts import select_model
        selection = dict(model=str(self.root / 'model.gguf'), server='runtime/llama-server',
                         context_size=4096, threads=2, n_gpu_layers=0)
        with patch.dict(os.environ, {'CAPSTONE_STRICT_OFFLINE': '1'}), \
             patch.object(select_model, 'select', return_value=selection) as select, \
             patch('llm_fit.model_profile', return_value={'context_size': 16384, 'acceleration': 'auto'}), \
             patch('gguf_metadata.context_length', return_value=8192):
            config = capstone.runtime_config({}, 'llama.cpp')
        self.assertEqual(config['context_size'], 4096)
        self.assertEqual(config['acceleration'], 'cpu')
        self.assertTrue(config['_strict_offline'])
        select.assert_called_once()

    def test_automatic_threads_respect_affinity(self):
        with patch.object(os, 'sched_getaffinity', return_value={1, 2, 3}):
            self.assertEqual(runtime.cpu_threads(), 3)

    def test_caches_and_scratch_are_usb_local_without_changing_home(self):
        source = {'HOME': '/host/home', 'TMPDIR': '/host/tmp', 'HF_HOME': '/host/hf'}
        env = portable_env.usb_environment(self.root, source)
        self.assertEqual(env['HOME'], '/host/home')
        for key in ('TMPDIR', 'HF_HOME', 'XDG_CACHE_HOME', 'PIP_CACHE_DIR', 'CUDA_CACHE_PATH', 'MESA_SHADER_CACHE_DIR'):
            self.assertTrue(Path(env[key]).is_relative_to(self.root))
            self.assertTrue(Path(env[key]).is_dir())
        self.assertEqual(source['TMPDIR'], '/host/tmp')

    def test_runtime_env_drops_inherited_launch_overrides(self):
        with patch.dict(os.environ, {'LLAMA_ARG_MODEL': '/host/model.gguf'}):
            env = runtime.runtime_environment(self.root, self.root / 'runtime/cuda/llama-server')
        self.assertNotIn('LLAMA_ARG_MODEL', env)
        self.assertEqual(env['LD_LIBRARY_PATH'].split(os.pathsep),
                         [str(self.root / 'runtime/cuda'), str(self.root / 'runtime')])

    def test_gpu_companion_libraries_extract_to_usb(self):
        archive = self.root / 'fixture.tar.gz'
        with tarfile.open(archive, 'w:gz') as handle:
            member = tarfile.TarInfo('lib/libcudart.so.12')
            member.size = 4
            handle.addfile(member, io.BytesIO(b'test'))
        destination = self.root / 'runtime/cuda'
        setup.extract_runtime(archive, destination, libraries_only=True)
        self.assertEqual((destination / 'libcudart.so.12').read_bytes(), b'test')

    def test_python_archive_links_materialize_without_host_scratch_or_symlinks(self):
        archive = self.root / 'python.tar.gz'
        with tarfile.open(archive, 'w:gz') as handle:
            member = tarfile.TarInfo('python/bin/python3.12')
            member.size = 4
            member.mode = 0o755
            handle.addfile(member, io.BytesIO(b'test'))
            link = tarfile.TarInfo('python/bin/python3')
            link.type = tarfile.SYMTYPE
            link.linkname = 'python3.12'
            handle.addfile(link)
        destination = self.root / 'hermes/python'
        setup_hermes.extract(archive, destination)
        self.assertEqual((destination / 'bin/python3').read_bytes(), b'test')
        self.assertFalse((destination / 'bin/python3').is_symlink())


if __name__ == '__main__':
    unittest.main()
