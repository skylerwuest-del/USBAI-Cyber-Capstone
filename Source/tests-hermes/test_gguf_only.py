"""Regression coverage for the public launcher/menu after backend simplification."""
import contextlib
import io
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capstone
import menu


class GGUFOnlyLauncherTests(unittest.TestCase):
    def test_menu_numbers_remain_stable(self):
        output = io.StringIO()
        with patch('builtins.input', return_value='0'), contextlib.redirect_stdout(output):
            self.assertEqual(menu.main(), 0)
        choices = re.findall(r'^(\d+)\)', output.getvalue(), re.MULTILINE)
        self.assertEqual(choices, ['1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12', '0'])

    def test_option_three_still_runs_selected_model_queue_and_exits(self):
        with patch('builtins.input', return_value='3'), patch.object(menu, 'run', return_value=0) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(menu.main(active_model='/models/selected.gguf'), 0)
        run.assert_called_once_with('--run', '--tasks', 'tasks.txt', active_model='/models/selected.gguf')

    def test_unsupported_cli_backend_fails_before_reading_configuration(self):
        with patch.object(capstone.json, 'loads', side_effect=AssertionError('must not load config')), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            capstone.main(['--backend', 'unsupported', '--chat'])
        self.assertEqual(stopped.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
