"""One Hermes session, owned by the USB supervisor. Uses the real Hermes SDK."""
import contextlib
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'hermes' / 'site-packages'), str(ROOT / 'hermes' / 'src')]
os.environ.update(HERMES_HOME=str(ROOT / 'state' / 'hermes'),
                  PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                  HERMES_YOLO_MODE='1', DO_NOT_TRACK='1',
                  HERMES_CONCURRENT_TOOL_TIMEOUT_S='0',
                  CAPSTONE_ROOT=str(ROOT))
os.environ['PATH'] = os.pathsep.join((str(ROOT / 'toolkit/bin'), str(ROOT / 'hermes/python/bin'), os.environ.get('PATH', '')))
os.environ['CAPSTONE_TOOLKIT'] = str(ROOT / 'toolkit')
os.environ.setdefault('SSL_CERT_FILE', str(ROOT / 'hermes/site-packages/certifi/cacert.pem'))
for proxy_key in ('NO_PROXY', 'no_proxy'):
    os.environ[proxy_key] = os.environ.get(proxy_key, '') + ',127.0.0.1,localhost,::1'
from usb_core import atomic_json
from console_progress import ConsoleProgress, StopRequested


def needs_continuation(result):
    reason = str(result.get('turn_exit_reason') or '') if isinstance(result, dict) else ''
    return (reason.startswith('max_iterations_reached(')
            and not result.get('failed') and not result.get('interrupted'))


def main():
    request_path, result_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text())
    os.environ['CAPSTONE_QUEUE_TASK_ID'] = str(request.get('queue_task_id') or '')
    if (ROOT / 'toolkit/bin/nmap').is_file():
        request['system'] += ('\nBundled USB tools are on PATH: nmap/ncat/netcat, Hashcat with CPU OpenCL, '
                              'John, Hydra, Crunch, LDAP/SMB/RPC/NetBIOS/SNMP clients, Gobuster, FFUF, '
                              'getcap/capsh, lsof, file, binutils, jq and basic network tools. RockYou is '
                              'at $CAPSTONE_TOOLKIT/share/wordlists/rockyou.txt. Their dependencies and '
                              'data are bundled; no apt install is needed. Use them only within the task\'s '
                              'authorized scope. Use nmap -sT for ordinary-user TCP scanning.')
    if request.get('toolsets') == ['terminal']:
        request['system'] += ('\nUse standard Ubuntu shell commands through terminal: cat or Python to read '
                              'files, mkdir -p for folders, printf for exact text and newlines, and Python '
                              'for edits and byte assertions. read_file, write_file and patch are not shell '
                              'commands. Use the task\'s full relative file path, not just its basename. '
                              'Keep the final report brief, about three sentences.')
    if request.get('review'):
        request['system'] += ('\nFor exact file content requirements, compare raw bytes (including newline counts) '
                              'using a terminal Python assertion or equivalent. Printed terminal output may '
                              'strip whitespace and is not enough to establish an exact match. If a check fails, '
                              'report complete=false with the observed mismatch; do not repair files in review.')
        request['prompt'] += ('\nREVIEW ACTION REQUIRED: Use your inspection tools now to check the actual '
                              'result independently. The worker report is not evidence. For exact file '
                              'contents, run a raw-byte comparison including newlines. Only after inspecting, '
                              'return JSON with complete, evidence (a string), and remaining '
                              '(an empty string only when every requirement is met).')
        atomic_json(request_path, request)
    cwd = Path(request['cwd'])
    os.chdir(cwd)
    os.environ.update(TERMINAL_ENV='local', TERMINAL_CWD=str(cwd),
                      TERMINAL_TIMEOUT=str(request['command_timeout']),
                      TERMINAL_MAX_FOREGROUND_TIMEOUT=str(request['command_timeout']))
    if request.get('local_context_compat', request.get('backend') == 'airllm'):
        # Opt in only for this local adapter. Keep the real configured context;
        # do not advertise 64K for a 16K/32K model or change vendored SDK files.
        from urllib.parse import urlparse
        if urlparse(request['base_url']).hostname != '127.0.0.1':
            raise ValueError('Local context compatibility requires the local USB server')
        from agent import model_metadata
        model_metadata.MINIMUM_CONTEXT_LENGTH = 1024
    from run_agent import AIAgent
    class USBHermesAgent(AIAgent):
        def _should_start_quiet_spinner(self):
            return False

        def _build_system_prompt(self, system_message=None):
            # Keep the role explicit for small models: the SDK's general
            # execution brief must not turn the independent reviewer into a worker.
            if not request.get('compact_system_prompt', request.get('backend') == 'airllm'):
                return super()._build_system_prompt(system_message)
            self._cached_system_prompt_static = 'You are a local Hermes agent. Follow the supplied role and task instructions.'
            return self._cached_system_prompt_static + '\n' + (system_message or '')
    from tools.terminal_tool_lifecycle import cleanup_all_environments
    from tools.process_registry import process_registry
    agent = None
    progress = ConsoleProgress(request['base_url'], request.get('backend'), ROOT)
    events = []
    def started(call_id, name, arguments):
        print('[tool] ' + name + ' ' + json.dumps(arguments, ensure_ascii=False)[:500], flush=True)
    def finished(call_id, name, arguments, result):
        # Save actual tool results as they arrive, including before a crash.
        event = {'tool': name, 'arguments': arguments, 'result': str(result)[-12000:]}
        events.append(event)
        if agent is not None and request.get('review'):
            agent.request_overrides.pop('tool_choice', None)
        atomic_json(result_path.with_suffix('.tools.json'), events)
        print('[result] ' + str(result)[-1200:], flush=True)
    def interrupted(signum, frame):
        raise StopRequested
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        overrides = {'temperature': request['temperature']}
        if request.get('review') and request.get('backend') == 'airllm':
            overrides['tool_choice'] = ({'type': 'function', 'function': {'name': 'terminal'}}
                                        if request.get('toolsets') == ['terminal'] else 'required')
        agent = USBHermesAgent(model='capstone', provider='custom', api_mode='chat_completions',
                        base_url=request['base_url'], api_key='usb-local',
                        enabled_toolsets=request.get('toolsets', ['terminal', 'file', 'todo']),
                        max_iterations=sys.maxsize if request.get('interactive') else request['turns'],
                        max_tokens=request['max_tokens'],
                        quiet_mode=True, skip_memory=True, skip_background_review=True,
                        skip_context_files=True, load_soul_identity=False,
                        checkpoints_enabled=False, cwd=str(cwd),
                        tool_start_callback=started, tool_complete_callback=finished,
                        thinking_callback=progress.thinking,
                        tool_progress_callback=lambda *args, **kwargs: None,
                        request_overrides=overrides)
        progress.__enter__()
        history = None
        if request.get('interactive'):
            print('Hermes SDK console | YOLO ON | local Ubuntu user permissions', flush=True)
            print('Model: ' + str(request.get('model_label') or 'capstone') + ' | context: ' + str(request.get('context_size')), flush=True)
            print('/quit exits; Ctrl+C stops this session and its model.', flush=True)
            while True:
                try:
                    prompt = input('\nYou> ').strip()
                except EOFError:
                    return 0
                if prompt in ('/quit', '/exit'):
                    return 0
                if not prompt:
                    continue
                result = agent.run_conversation(prompt, system_message=request['system'], conversation_history=history)
                history = result.get('messages')
                if result.get('interrupted'):
                    return 130
                if result.get('failed'):
                    print('[model] Request failed: ' + str(result.get('final_response') or result.get('turn_exit_reason') or 'See session logs'), flush=True)
                    continue
                print('\nHermes> ' + str(result.get('final_response', '')), flush=True)
        task_id = str(request.get('task_id') or request_path.parent.name)
        result = agent.run_conversation(request['prompt'], system_message=request['system'], task_id=task_id)
        if not request.get('review'):
            for segment in range(1, max(1, int(request.get('continuation_segments', 1)))):
                if not needs_continuation(result):
                    break
                atomic_json(result_path, {'final_response': result.get('final_response', ''),
                                          'failed': False, 'interrupted': False,
                                          'exit_reason': result.get('turn_exit_reason'),
                                          'partial': True, 'tools': events})
                print('[continuation] Turn budget reached; keeping this worker and its tracked processes alive.', flush=True)
                result = agent.run_conversation(
                    'Continue the same task from the current conversation and tool state. Do not restart completed work. '
                    'If a bounded command is still running, poll or wait for that exact process until it finishes, then '
                    'inspect its result. Do not provide a final report while required work is still running.',
                    system_message=request['system'], conversation_history=result.get('messages'), task_id=task_id)
        if request.get('review') or request['system'].startswith('You are a fresh reviewer'):
            # Some local models give prose despite the JSON instruction. Ask for a
            # schema-constrained verdict using the review's actual observations.
            # This never supplies fabricated tool evidence or turns an SDK failure
            # into success; the supervisor checks both independently.
            recovered_review = False
            try:
                verdict = json.loads(str(result.get('final_response', '')))
                if not isinstance(verdict, dict) or not isinstance(verdict.get('complete'), bool) or not all(
                        isinstance(verdict.get(key), str) for key in ('evidence', 'remaining')):
                    raise ValueError('Review fields need their required JSON types')
                if str(result.get('turn_exit_reason') or '').startswith('max_iterations_reached('):
                    raise ValueError('Review needs separate finalization after exhausting its tool-turn budget')
            except (TypeError, ValueError):
                from openai import OpenAI
                schema = {'type': 'object', 'properties': {
                    'complete': {'type': 'boolean'}, 'evidence': {'type': 'string'},
                    'remaining': {'type': 'string'}},
                    'required': ['complete', 'evidence', 'remaining'], 'additionalProperties': False}
                with OpenAI(base_url=request['base_url'], api_key='usb-local', timeout=request.get('request_timeout', 900), max_retries=1) as client:
                    response = client.chat.completions.create(model='capstone', temperature=0,
                        max_tokens=1024, messages=[{'role': 'system', 'content': request['system']},
                            {'role': 'user', 'content': request['prompt'] + '\nACTUAL REVIEW OBSERVATIONS:\n'
                             + json.dumps(events, ensure_ascii=False)[-24000:]
                             + '\nREVIEW FINDINGS:\n' + str(result.get('final_response', ''))
                             + '\nReturn the verdict as JSON. If evidence is insufficient, complete must be false. '
                             'When complete is true, remaining must be an empty string.'}],
                        response_format={'type': 'json_schema', 'json_schema': {'name': 'review_verdict', 'strict': True, 'schema': schema}},
                        extra_body={'chat_template_kwargs': {'enable_thinking': False}})
                if response.choices[0].finish_reason != 'length':
                    result['final_response'] = response.choices[0].message.content
                    recovered_review = True
        atomic_json(result_path, {'final_response': result.get('final_response', ''),
                                 'failed': result.get('failed', False),
                                 'interrupted': result.get('interrupted', False),
                                 'exit_reason': result.get('turn_exit_reason'),
                                 'review_recovered': recovered_review if request.get('review') else False,
                                 'tools': events})
        print(result.get('final_response', ''), flush=True)
        return 1 if result.get('failed') or result.get('interrupted') else 0
    finally:
        progress.close()
        if agent is not None:
            with contextlib.suppress(Exception):
                agent.close()
        with contextlib.suppress(Exception):
            process_registry.kill_all()
        with contextlib.suppress(Exception):
            cleanup_all_environments()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, StopRequested):
        raise SystemExit(130)
