"""Exercise typed CAI tool conversion with the USB's installed dependencies."""
import asyncio
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'cai' / 'site-packages')]
os.environ.update(json.loads((ROOT / 'cai' / 'local-config.json').read_text()))
from openai.types.chat import ChatCompletion
from cai.sdk.agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
import cai_tool_adapter


class TypedToolAdapterTests(unittest.TestCase):
    def response(self, content, reason='stop', tools=None):
        message = {'role':'assistant','content':content}
        if tools: message['tool_calls'] = tools
        return ChatCompletion(id='check', created=0, model='capstone', object='chat.completion',
                              choices=[{'index':0,'finish_reason':reason,'message':message}])

    def convert(self, response, model='openai/capstone'):
        async def original(*args): return response
        with patch.object(OpenAIChatCompletionsModel, '_fetch_response_litellm_openai', original):
            cai_tool_adapter.install()
            method = OpenAIChatCompletionsModel._fetch_response_litellm_openai
            kwargs = {'model':model,'tools':[{'type':'function','function':{'name':'run','parameters':{
                'type':'object','properties':{'command':{'type':'string'}},'required':['command'],'additionalProperties':False}}}]}
            return asyncio.run(method(None, kwargs, None, 'auto', False, False))

    def test_valid_json_becomes_an_openai_typed_tool_call(self):
        response = self.convert(self.response('{"name":"run","arguments":{"command":"echo ok"}}'))
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, 'tool_calls')
        self.assertIsNone(choice.message.content)
        call = choice.message.tool_calls[0]
        self.assertEqual(call.function.name, 'run')
        self.assertEqual(json.loads(call.function.arguments), {'command':'echo ok'})
        self.assertTrue(call.id.startswith('call_'))

    def test_arguments_that_fail_schema_validation_remain_text(self):
        for args in ('{"command":123}', '{}', '{"command":"ok","extra":true}'):
            response = self.convert(self.response('{"name":"run","arguments":'+args+'}'))
            self.assertFalse(response.choices[0].message.tool_calls)
            self.assertIsNotNone(response.choices[0].message.content)

    def test_existing_native_tool_calls_are_preserved(self):
        response = self.response(None, 'tool_calls', [{'id':'native','type':'function','function':{'name':'run','arguments':'{}'}}])
        converted = self.convert(response)
        self.assertEqual(converted.choices[0].message.tool_calls[0].id, 'native')

    def test_other_models_and_truncated_responses_are_not_converted(self):
        text = '{"name":"run","arguments":{"command":"echo ok"}}'
        response = self.convert(self.response(text), model='openai/other')
        self.assertFalse(response.choices[0].message.tool_calls)
        response = self.convert(self.response(text, 'length'))
        self.assertFalse(response.choices[0].message.tool_calls)


if __name__ == '__main__': unittest.main(verbosity=2)
