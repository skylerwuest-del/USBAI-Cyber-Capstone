import unittest
from cai_tool_adapter import extract_call


class LocalToolFormatTests(unittest.TestCase):
    def setUp(self):
        self.tools = [{'type': 'function', 'function': {'name': 'run', 'parameters': {
            'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]

    def test_whole_json_for_a_declared_tool_is_recognized(self):
        name, args, schema = extract_call('{"name":"run","arguments":{"command":"echo ok"}}', self.tools)
        self.assertEqual(name, 'run')
        self.assertEqual(args, {'command': 'echo ok'})
        self.assertEqual(schema['required'], ['command'])

    def test_prose_markdown_and_malformed_json_remain_text(self):
        for content in ['Run this: {"name":"run","arguments":{}}', '```json\n{"name":"run","arguments":{}}\n```', '{{"name":"run","arguments":{}}}', '{"name":"run"', None]:
            self.assertIsNone(extract_call(content, self.tools))

    def test_undeclared_tools_are_never_recognized(self):
        self.assertIsNone(extract_call('{"name":"other","arguments":{}}', self.tools))
        self.assertIsNone(extract_call('{"name":"run","arguments":{}}', []))

    def test_extra_fields_arrays_and_nonobject_arguments_are_rejected(self):
        for content in ['{"name":"run","arguments":{},"extra":"yes"}', '[{"name":"run","arguments":{}}]', '{"name":"run","arguments":"echo ok"}']:
            self.assertIsNone(extract_call(content, self.tools))


if __name__ == '__main__': unittest.main()
