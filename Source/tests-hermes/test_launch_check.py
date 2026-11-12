import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch_check


class LaunchCheckTests(unittest.TestCase):
    def fixture(self, root):
        for name in ('hermes/python/bin/python3.12', 'hermes/src/run_agent.py', 'runtime/llama-server'):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('')
            path.chmod(0o755)
        (root / 'config.json').write_text(json.dumps({'server': 'runtime/llama-server',
                                                    'model': 'models/broken.gguf'}))
        (root / 'hermes-requirements.lock').write_text('demo==1.0 --hash=sha256:example\n')

    def test_model_failure_and_failed_setup_marker_do_not_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / 'state').mkdir()
            (root / 'state/setup-ready.json').write_text('{"ready": false}')
            with patch.object(launch_check, 'check_packages', return_value=[]), \
                 patch.object(launch_check.subprocess, 'run') as run:
                run.return_value.returncode = 0
                self.assertEqual(launch_check.check(root), [])
                self.assertEqual(run.call_count, 2)
                self.assertEqual(run.call_args_list[0].args[0][-1], '--version')

    def test_missing_dependency_blocks_before_sdk_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with patch.object(launch_check.subprocess, 'run') as run:
                run.return_value.returncode = 0
                self.assertIn('demo requires 1.0 (installed: missing)', launch_check.check(root))
                self.assertEqual(run.call_count, 1)

    def test_missing_setup_assets_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            errors = launch_check.check(Path(directory))
            self.assertIn('Missing hermes/python/bin/python3.12', errors)

    def test_distribution_name_normalization(self):
        self.assertEqual(launch_check.package_key('Ruamel.yaml'), 'ruamel-yaml')


if __name__ == '__main__':
    unittest.main()
