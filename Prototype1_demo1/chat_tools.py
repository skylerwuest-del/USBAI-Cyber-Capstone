"""Concise descriptions for interactive local tools; SDK schemas/dispatch stay intact."""
from copy import deepcopy

DESCRIPTIONS = {
    'terminal': ('Execute a shell command on the local computer and return its output and exit code. '
                 'Use this tool to actually read/write files (cat, printf, Python), inspect or run commands. '
                 'Use absolute paths when needed. Bounded commands may use a generous timeout; '
                 'use capstone-job for durable work that must survive this session.'),
    'process_manage': ('Manage an existing background terminal process by its session ID: '
                       'inspect output/status, send input, or stop it. Do not start duplicate jobs.'),
}
PARAMETERS = {
    'command': 'Shell command to execute.', 'action': 'Operation on the existing process.',
    'session_id': 'Existing process session ID.', 'workdir': 'Absolute working directory.',
    'timeout': 'Foreground wait in seconds; a larger value does not delay a completed command.',
    'background': 'Start in background, returning a session ID.',
    'notify': 'For background work: true notifies on completion; a list matches readiness patterns.',
    'pty': 'Use a pseudo-terminal for an interactive background program.',
    'heartbeat': 'Background progress interval in seconds; minimum 60.',
}


def compact_tool_descriptions(tools):
    """Copy descriptions only; preserve names, all properties, types and constraints."""
    result = deepcopy(tools)
    for tool in result:
        function = tool.get('function', {})
        if function.get('name') not in DESCRIPTIONS:
            continue
        function['description'] = DESCRIPTIONS[function['name']]
        for name, schema in function.get('parameters', {}).get('properties', {}).items():
            if 'description' in schema and name in PARAMETERS:
                schema['description'] = PARAMETERS[name]
    return result
