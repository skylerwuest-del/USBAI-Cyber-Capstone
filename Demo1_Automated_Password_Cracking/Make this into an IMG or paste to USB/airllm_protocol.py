"""Pure protocol conversion for the local AirLLM/Hermes adapter."""
import copy
import json
import re
import time
import uuid


class ProtocolError(ValueError):
    pass


def validate(value, schema):
    """Check the JSON types/required fields used by Hermes tool and review schemas."""
    if 'anyOf' in schema:
        for option in schema['anyOf']:
            try:
                validate(value, option)
                return
            except ProtocolError:
                pass
        raise ProtocolError('Value does not match any allowed schema')
    types = schema.get('type')
    if isinstance(types, str):
        types = [types]
    checks = {'object': isinstance(value, dict), 'array': isinstance(value, list),
              'string': isinstance(value, str), 'boolean': isinstance(value, bool),
              'integer': isinstance(value, int) and not isinstance(value, bool),
              'number': isinstance(value, (int, float)) and not isinstance(value, bool),
              'null': value is None}
    if types and not any(checks.get(t, False) for t in types):
        raise ProtocolError('Unexpected argument type: expected ' + str(types))
    if 'enum' in schema and value not in schema['enum']:
        raise ProtocolError('Value is not in the allowed enum')
    if isinstance(value, dict):
        if any(key not in value for key in schema.get('required', [])):
            raise ProtocolError('Missing a required field')
        properties = schema.get('properties', {})
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key])
            elif schema.get('additionalProperties') is False:
                raise ProtocolError('Unexpected field: ' + key)
    if isinstance(value, list) and isinstance(schema.get('items'), dict):
        for item in value:
            validate(item, schema['items'])


def prepare_messages(body):
    messages = copy.deepcopy(body.get('messages'))
    if not isinstance(messages, list) or not messages:
        raise ProtocolError('messages must be a nonempty list')
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant', 'tool'):
            raise ProtocolError('Invalid message role')
        content = message.get('content')
        if isinstance(content, list):
            if any(part.get('type') != 'text' for part in content):
                raise ProtocolError('This CPU server supports text messages only')
            message['content'] = '\n'.join(part['text'] for part in content)
        elif content is None:
            message['content'] = ''
        elif not isinstance(content, str):
            raise ProtocolError('Message content must be text')
        for call in message.get('tool_calls', []):
            function = call['function']
            if isinstance(function.get('arguments'), str):
                function['arguments'] = json.loads(function['arguments'])
    tools = body.get('tools') or []
    if not isinstance(tools, list) or any(t.get('type') != 'function' for t in tools):
        raise ProtocolError('Only function tools are supported')
    choice = body.get('tool_choice', 'auto')
    if choice == 'none':
        tools = []
    instruction = ''
    if tools:
        instruction = ('\nUse the supplied tools to perform actions. To call a tool, output '
                       '<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>. '
                       'Do not present executable commands as a substitute for a tool call. '
                       'After a tool result, continue working or give your final response.')
        if choice == 'required':
            instruction += '\nYou must call at least one supplied tool.'
        elif isinstance(choice, dict):
            name = choice.get('function', {}).get('name')
            if name not in {t['function']['name'] for t in tools}:
                raise ProtocolError('Requested tool is unavailable')
            instruction += '\nCall the tool named ' + name + '.'
    response_format = body.get('response_format') or {}
    if response_format.get('type') in ('json_object', 'json_schema'):
        instruction += '\nReturn only valid JSON, without Markdown fences.'
        if response_format['type'] == 'json_schema':
            instruction += '\nRequired JSON schema: ' + json.dumps(response_format['json_schema']['schema'])
    if instruction:
        if messages[0]['role'] == 'system':
            messages[0]['content'] += instruction
        else:
            messages.insert(0, {'role': 'system', 'content': instruction.strip()})
    return messages, tools


def parse_output(text, tools, choice='auto', truncated=False):
    if truncated:
        return {'role': 'assistant', 'content': text}, 'length'
    # Decode JSON before recognizing closing tags. File contents and shell
    # arguments may legitimately contain literal tool or reasoning delimiters.
    definitions = {t['function']['name']: t['function'] for t in tools}
    blocks, content, position = [], [], 0
    decoder = json.JSONDecoder()
    while position < len(text):
        match = re.search(r'<think>|<tool_call>|</tool_call>', text[position:])
        if not match:
            content.append(text[position:])
            break
        start = position + match.start()
        content.append(text[position:start])
        position += match.end()
        if match.group() == '<think>':
            end = text.find('</think>', position)
            if end < 0:
                return {'role': 'assistant', 'content': ''}, 'length'
            position = end + len('</think>')
            continue
        if match.group() == '</tool_call>':
            raise ProtocolError('Unexpected tool closing tag; no command was executed')
        position += len(text[position:]) - len(text[position:].lstrip())
        try:
            value, end = decoder.raw_decode(text, position)
        except ValueError as exc:
            raise ProtocolError('Malformed tool call; no command was executed') from exc
        tail = text[end:]
        position = end + len(tail) - len(tail.lstrip())
        if not text.startswith('</tool_call>', position):
            raise ProtocolError('Incomplete tool call; no command was executed')
        position += len('</tool_call>')
        blocks.append(value)
    calls = []
    for block in blocks:
        try:
            value = block
            name, arguments = value['name'], value['arguments']
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
        except (ValueError, KeyError, TypeError) as exc:
            raise ProtocolError('Malformed tool call; no command was executed') from exc
        if name not in definitions or not isinstance(arguments, dict):
            raise ProtocolError('Unknown tool or non-object arguments; no command was executed')
        if isinstance(choice, dict) and name != choice['function']['name']:
            raise ProtocolError('Model called a different tool than requested')
        validate(arguments, definitions[name].get('parameters', {}))
        calls.append({'id': 'call_' + uuid.uuid4().hex, 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}})
    if calls:
        return {'role': 'assistant', 'content': ''.join(content).strip() or None, 'tool_calls': calls}, 'tool_calls'
    if tools and (choice == 'required' or isinstance(choice, dict)):
        raise ProtocolError('Model did not call the required tool')
    return {'role': 'assistant', 'content': ''.join(content).strip()}, 'stop'


def completion(message, finish_reason, prompt_tokens, output_tokens):
    return {'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion',
            'created': int(time.time()), 'model': 'capstone',
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': output_tokens,
                      'total_tokens': prompt_tokens + output_tokens}}


def stream_chunks(response):
    base = {key: response[key] for key in ('id', 'created', 'model')}
    base['object'] = 'chat.completion.chunk'
    message = copy.deepcopy(response['choices'][0]['message'])
    for index, call in enumerate(message.get('tool_calls', [])):
        call['index'] = index
    yield dict(base, choices=[{'index': 0, 'delta': message, 'finish_reason': None}])
    yield dict(base, choices=[{'index': 0, 'delta': {}, 'finish_reason': response['choices'][0]['finish_reason']}],
               usage=response['usage'])
