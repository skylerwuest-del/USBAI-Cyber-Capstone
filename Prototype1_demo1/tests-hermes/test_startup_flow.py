import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import menu
from chat_tools import compact_tool_descriptions
from console_progress import ConsoleProgress
import capstone
from test_llm_fit import fixture
import tempfile


class StartupFlowTests(unittest.TestCase):
    def test_chat_and_queue_and_serve_receive_session_model(self):
        process = Mock()
        process.wait.return_value = 0
        for args in (('--chat',), ('--run', '--tasks', 'safe.txt'), ('--serve-only',)):
            with patch.object(menu.subprocess, 'Popen', return_value=process) as launch:
                menu.run(*args, active_model='/models/selected.gguf')
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index('--model') + 1], '/models/selected.gguf')
            self.assertEqual('--fit-session' in command, '--chat' in args)

    def test_airllm_run_never_gets_gguf_override(self):
        process = Mock()
        with patch.object(menu.subprocess, 'Popen', return_value=process) as launch:
            menu.run('--backend', 'airllm', '--chat', active_model='/models/gguf')
        self.assertNotIn('--model', launch.call_args.args[0])
        self.assertNotIn('--fit-session', launch.call_args.args[0])

    def test_original_menu_retains_selection_after_chat(self):
        with patch('builtins.input', side_effect=['1', '1', '0']), patch.object(menu, 'run') as run, \
             contextlib.redirect_stdout(io.StringIO()):
            menu.main(active_model='/models/current.gguf')
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs['active_model'], '/models/current.gguf')

    def test_explicit_saved_model_selection_replaces_session_override(self):
        with patch('builtins.input', side_effect=['6', '1', '0']), \
             patch('models_menu.choose_installed', return_value=Path('/models/saved.gguf')), \
             patch.object(menu, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            menu.main(active_model='/models/previous.gguf')
        self.assertEqual(run.call_args.kwargs['active_model'], '/models/saved.gguf')

    def test_compact_descriptions_preserve_schema_and_original(self):
        tool = {'type': 'function', 'function': {'name': 'terminal', 'description': 'long ' * 500,
                'parameters': {'type': 'object', 'required': ['command'], 'properties': {
                    'command': {'type': 'string', 'description': 'long ' * 50},
                    'timeout': {'type': 'integer', 'minimum': 1, 'description': 'long ' * 50},
                    'background': {'type': 'boolean', 'default': False}}}}}
        original = copy.deepcopy(tool)
        compact = compact_tool_descriptions([tool])[0]
        self.assertEqual(tool, original)
        self.assertEqual(compact['function']['parameters']['required'], ['command'])
        self.assertEqual(compact['function']['parameters']['properties']['timeout']['minimum'], 1)
        self.assertEqual(compact['function']['parameters']['properties']['background'],
                         original['function']['parameters']['properties']['background'])
        self.assertLess(len(json.dumps(compact)), len(json.dumps(original)))

    def progress(self, slot):
        progress = ConsoleProgress('http://127.0.0.1:1/v1', 'llama.cpp', '.')
        response = io.StringIO(json.dumps([slot]))
        with patch.object(progress.client, 'open', return_value=contextlib.closing(response)):
            return progress.detail()

    def test_first_prompt_batch_never_reports_false_zero_progress(self):
        detail = self.progress({'is_processing': True, 'n_prompt_tokens_processed': 0})
        self.assertIn('first prompt batch', detail)
        self.assertNotIn('0 prompt tokens', detail)

    def test_progress_reports_actual_processed_and_generated_counts(self):
        self.assertIn('512 prompt tokens', self.progress({'is_processing': True, 'n_prompt_tokens_processed': 512}))
        self.assertIn('12 tokens generated', self.progress({'is_processing': True, 'next_token': {'n_decoded': 12}}))
        self.assertIn('12 tokens generated', self.progress({'is_processing': True, 'next_token': [{'n_decoded': 12}]}))
        self.assertIn('first prompt batch', self.progress({'is_processing': True, 'next_token': []}))

    def test_interactive_discovery_overhead_disabled_without_changing_queue_or_airllm(self):
        original = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for backend, interactive in [('llama.cpp', True), ('llama.cpp', False), ('airllm', True)]:
                config = dict(original, backend=backend, _interactive_chat=interactive)
                with patch.object(capstone, 'ROOT', root):
                    capstone.configure_hermes(config)
                settings = json.loads((root / 'state/hermes/config.yaml').read_text())
                if backend == 'llama.cpp' and interactive:
                    self.assertEqual(settings['tools']['tool_search']['enabled'], 'off')
                else:
                    self.assertNotIn('tools', settings)

    def test_chat_configuration_is_lean_without_changing_queue_tools(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = fixture(root / 'models/test.gguf')
            config = dict(context_size=8192, model=str(model), toolsets=['terminal', 'file', 'todo'])
            with patch.object(capstone, 'ROOT', root):
                active = capstone.runtime_config(config, 'llama.cpp')
            self.assertEqual(active['chat_toolsets'], ['terminal'])
            self.assertEqual(active['toolsets'], ['terminal', 'file', 'todo'])
            self.assertEqual(active['batch_size'], 512)
            self.assertEqual(active['ubatch_size'], 512)


if __name__ == '__main__':
    unittest.main()
