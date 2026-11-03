import json
from pathlib import Path
import sys
import socket
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airllm_protocol import ProtocolError, completion, parse_output, prepare_messages, validate
from airllm_server import CPUCleanup, CPUWeightCache, GenerationCancelled, Handler

TOOLS = [{'type': 'function', 'function': {'name': 'terminal', 'parameters': {
    'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]
CALL = '<tool_call>{"name":"terminal","arguments":{"command":"pwd"}}</tool_call>'


class ProtocolTests(unittest.TestCase):
    def test_weight_cache_bounds_bytes_and_retains_values_after_meta_unload(self):
        import torch
        loads = []
        def load(name):
            loads.append(name)
            return {'weight': torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)}
        cache = CPUWeightCache(load, torch.float32, 16)
        state = cache('first')
        layer = torch.nn.Linear(2, 2, bias=False)
        layer.weight = torch.nn.Parameter(state['weight'])
        layer.to('meta')
        self.assertEqual(cache('first')['weight'].device.type, 'cpu')
        self.assertTrue(torch.equal(cache('first')['weight'], torch.arange(4).reshape(2, 2).float()))
        cache('second')
        cache('second')
        self.assertEqual(loads, ['first', 'second', 'second'])
        self.assertEqual(cache.used, 16)
        self.assertLessEqual(cache.used, cache.budget)

    def test_cpu_cleanup_is_bounded_and_flushes_partial_forward_on_error(self):
        calls = []
        cleanup = CPUCleanup(lambda: calls.append('collected'), 31)
        for _ in range(31):
            cleanup()
        self.assertEqual(len(calls), 1)
        for _ in range(5):
            cleanup()
        cleanup.flush()
        self.assertEqual(len(calls), 2)
        self.assertEqual(cleanup.calls, 0)

    def test_delimiters_inside_command_strings_are_preserved(self):
        command = "printf '%s' '</tool_call><think>file content</think>'"
        text = '<tool_call>' + json.dumps({'name': 'terminal', 'arguments': {'command': command}}) + '</tool_call>'
        message, reason = parse_output(text, TOOLS)
        self.assertEqual(reason, 'tool_calls')
        self.assertEqual(json.loads(message['tool_calls'][0]['function']['arguments'])['command'], command)

    def test_native_call_is_translated_to_sdk_tool_call(self):
        message, reason = parse_output(CALL, TOOLS)
        self.assertEqual(reason, 'tool_calls')
        call = message['tool_calls'][0]
        self.assertEqual(call['function']['name'], 'terminal')
        self.assertEqual(json.loads(call['function']['arguments']), {'command': 'pwd'})

    def test_tool_history_arguments_are_objects_for_native_template(self):
        message, _ = parse_output(CALL, TOOLS)
        body = {'messages': [{'role': 'user', 'content': 'Where am I?'}, message,
                             {'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'], 'content': '/tmp'}],
                'tools': TOOLS}
        messages, _ = prepare_messages(body)
        self.assertIsInstance(messages[2]['tool_calls'][0]['function']['arguments'], dict)
        self.assertIsInstance(body['messages'][1]['tool_calls'][0]['function']['arguments'], str)

    def test_unknown_malformed_or_incomplete_calls_never_execute(self):
        for value in (CALL.replace('terminal', 'unknown'), CALL[:-12], '<tool_call>not json</tool_call>',
                      CALL + '<tool_call>', CALL.replace('"pwd"', 'false')):
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                parse_output(value, TOOLS)

    def test_calls_in_reasoning_are_ignored(self):
        message, reason = parse_output('<think>' + CALL + '</think>Finished', TOOLS)
        self.assertEqual(reason, 'stop')
        self.assertNotIn('tool_calls', message)

    def test_truncated_output_never_executes_even_a_closed_call(self):
        message, reason = parse_output(CALL, TOOLS, truncated=True)
        self.assertEqual(reason, 'length')
        self.assertNotIn('tool_calls', message)

    def test_required_tool_is_enforced(self):
        with self.assertRaises(ProtocolError):
            parse_output('Run pwd yourself', TOOLS, 'required')

    def test_tool_choice_none_removes_tools(self):
        _, tools = prepare_messages({'messages': [{'role': 'user', 'content': 'Hi'}],
                                    'tools': TOOLS, 'tool_choice': 'none'})
        self.assertEqual(tools, [])
        with self.assertRaises(ProtocolError):
            parse_output(CALL, tools, 'none')

    def test_structured_verdict_types_are_checked(self):
        schema = {'type': 'object', 'properties': {'complete': {'type': 'boolean'}},
                  'required': ['complete'], 'additionalProperties': False}
        validate({'complete': True}, schema)
        for value in ({'complete': 'true'}, {}, {'complete': True, 'invented': 1}):
            with self.assertRaises(ProtocolError):
                validate(value, schema)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Engine:
            context = 16384
            max_tokens = 1024
            lock = threading.Lock()
            cancel_seen = threading.Event()
            def generate(self, body, cancelled=None):
                if body.get('wait_for_disconnect'):
                    if cancelled.wait(3):
                        self.cancel_seen.set()
                        raise GenerationCancelled('test cancellation')
                    raise RuntimeError('Disconnected client was not detected')
                message, reason = parse_output(CALL, TOOLS)
                return completion(message, reason, 10, 8)
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.engine = Engine()
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, body=None):
        request = urllib.request.Request(self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(request, timeout=5)

    def test_health_and_metadata_report_real_context(self):
        with self.request('/health') as response:
            self.assertEqual(json.load(response)['context_length'], 16384)
        with self.request('/v1/models') as response:
            self.assertEqual(json.load(response)['data'][0]['context_length'], 16384)

    def test_nonstream_response_contains_tool_call(self):
        with self.request('/v1/chat/completions', {'stream': False}) as response:
            self.assertEqual(json.load(response)['choices'][0]['finish_reason'], 'tool_calls')

    def test_stream_response_contains_sdk_tool_indexes_and_finish(self):
        with self.request('/v1/chat/completions', {'stream': True}) as response:
            lines = response.read().decode().splitlines()
        chunks = [json.loads(line[6:]) for line in lines if line.startswith('data: {')]
        self.assertEqual(chunks[0]['choices'][0]['delta']['tool_calls'][0]['index'], 0)
        self.assertEqual(chunks[1]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertIn('data: [DONE]', lines)

    def test_busy_requests_fail_but_health_remains_available(self):
        self.server.engine.lock.acquire()
        try:
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/v1/chat/completions', {})
            self.assertEqual(error.exception.code, 503)
            error.exception.close()
            with self.request('/health') as response:
                self.assertEqual(response.status, 200)
        finally:
            self.server.engine.lock.release()

    def test_disconnected_client_cancels_generation_and_releases_model(self):
        body = json.dumps({'wait_for_disconnect': True}).encode()
        sock = socket.create_connection(self.server.server_address, timeout=5)
        sock.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: '
                     + str(len(body)).encode() + b'\r\nContent-Type: application/json\r\n\r\n' + body)
        sock.close()
        self.assertTrue(self.server.engine.cancel_seen.wait(3))
        # A following request may arrive during the handler's brief final cleanup.
        self.assertTrue(self.server.engine.lock.acquire(timeout=3))
        self.server.engine.lock.release()
        with self.request('/v1/chat/completions', {}) as response:
            self.assertEqual(response.status, 200)


if __name__ == '__main__':
    unittest.main()
