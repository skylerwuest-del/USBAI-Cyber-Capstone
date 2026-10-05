"""Compatibility for a local model returning a complete JSON tool call as text."""
import json


def extract_call(content, tools):
    """Accept only one whole JSON object naming a tool declared in this request.

    Prose, markdown, unknown tools, extra fields, and malformed JSON stay text.
    Full JSON-schema validation is performed by the CAI runtime before conversion.
    """
    if not isinstance(content, str) or len(content) > 100000:
        return None
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        return None
    if not isinstance(value["name"], str) or not isinstance(value["arguments"], dict):
        return None
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function", {})
        if function.get("name") == value["name"]:
            return value["name"], value["arguments"], function.get("parameters", {})
    return None


def install():
    """Normalize non-streaming responses only for CapStone's local CAI model."""
    import jsonschema
    from openai.types.chat import ChatCompletionMessageToolCall
    from cai.sdk.agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from uuid import uuid4

    original = OpenAIChatCompletionsModel._fetch_response_litellm_openai
    if getattr(original, "_capstone_adapter", False):
        return

    async def fetch(self, kwargs, model_settings, tool_choice, stream, parallel_tool_calls):
        response = await original(self, kwargs, model_settings, tool_choice, stream, parallel_tool_calls)
        if stream or kwargs.get("model") != "openai/capstone" or tool_choice == "none":
            return response
        message = response.choices[0].message
        if message.tool_calls or response.choices[0].finish_reason == "length":
            return response
        call = extract_call(message.content, kwargs.get("tools"))
        if call is None:
            return response
        name, arguments, schema = call
        try:
            jsonschema.validate(arguments, schema)
        except (jsonschema.ValidationError, jsonschema.SchemaError):
            return response
        message.tool_calls = [ChatCompletionMessageToolCall(
            id="call_" + uuid4().hex[:24], type="function",
            function={"name": name, "arguments": json.dumps(arguments)})]
        message.content = None
        response.choices[0].finish_reason = "tool_calls"
        return response

    fetch._capstone_adapter = True
    OpenAIChatCompletionsModel._fetch_response_litellm_openai = fetch
