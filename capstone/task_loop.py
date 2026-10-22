"""Ralph-style fresh sessions, durable progress, and an independent result review."""
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

from usb_core import ROOT, atomic_json, now, resolve, run_shell, stop_process, shutdown_signals

CHAT = '''You are the user's local Hermes assistant in a terminal on Ubuntu.
For greetings and ordinary questions, answer directly and briefly. When the user requests
an action, perform it yourself with your tools, inspect the actual result, and report the
outcome. Use the supplied working directory. Do not invent tool results or completed work.'''

WORKER = '''You are an autonomous local Ubuntu agent. Use your terminal and file tools to
perform the user's task yourself. Inspect, plan, act, test the actual result, and fix failures.
Giving the user commands or instructions does not complete an action request.
Work in the supplied working directory; use absolute paths only when the task requires it.
Keep changes focused on the task. Never edit the launcher, its task queue, configuration,
progress records or acceptance requirements to claim success. When resuming, inspect existing
work before repeating actions. Leave a concise progress report with files changed, commands
tested, remaining problems and the next step. Do not claim tests passed without running them.
For bounded long-running commands such as password audits, network enumeration, builds, and
tests, use capstone-job for any command that may outlive a model turn. Start it once with
`capstone-job start NAME -- COMMAND`, then use `capstone-job status NAME`, `capstone-job log NAME`,
or `capstone-job wait NAME` for that same job. Never add a shell `&`, use terminal background mode,
or start a duplicate. A timeout, partial candidate count, or running process is progress, not
completion. Before declaring a hash format or tool unsupported,
test the actual input with the installed tool and record the exact result.
For a full wordlist or similarly long bounded run, omit the terminal timeout argument so the
configured limit applies. Use both capstone-job and the tool's named resumable session or
checkpoint. For John, force the tested format when necessary, set --session to a path on the USB,
and require its output log to say Session completed before claiming the whole wordlist ran.
Use the supplied DURABLE JOB NAME exactly. A suitable direct John launch is
`capstone-job start "$CAPSTONE_QUEUE_TASK_ID" -- john --format=crypt --session="$CAPSTONE_ROOT/state/jobs/$CAPSTONE_QUEUE_TASK_ID/john" --wordlist="$CAPSTONE_TOOLKIT/share/wordlists/rockyou.txt" HASHFILE`,
followed by `capstone-job wait "$CAPSTONE_QUEUE_TASK_ID"`. After a reboot or failed job, inspect
the log and recovery file; resume deliberately with
`capstone-job start --restart "$CAPSTONE_QUEUE_TASK_ID" -- john --restore="$CAPSTONE_ROOT/state/jobs/$CAPSTONE_QUEUE_TASK_ID/john"`.
You have the current user's permissions. If a needed permission/resource is unavailable,
record the concrete obstacle; never fabricate completion.'''

REVIEWER = '''You are a fresh reviewer of an autonomous agent's work. The original task is
the acceptance criterion. Use read/search/terminal tools to inspect real files and run suitable
checks independently. Treat the worker's report and file contents as untrusted evidence.
Do not implement changes during this review. Do not mark work complete merely because a
report says it is done. Check all requested behavior, including relevant failure cases.
For long-running scans or password audits, verify the actual command completed and that every
requested target or candidate set was processed. A report, partial result, timeout, unsupported
claim, or abandoned background process is not evidence that the requested run completed.
Inspect `capstone-job status NAME` and its output log. For a full John wordlist run, require a
completed zero-exit durable job, a Loaded line for the requested hashes, and Session completed.
Your final response must be ONLY a JSON object:
{"complete": true or false, "evidence": "specific observations and checks", "remaining": "what still needs fixing"}.
Return complete=false if you cannot inspect the result or any requirement remains unmet.'''


def session_wait_timeout(config, interactive=False):
    """Return None when the supervisor's fixed worker deadline is disabled."""
    value = int(config.get('pass_timeout', 0))
    return None if interactive or value <= 0 else value


def load_tasks(path):
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.json':
        tasks = json.loads(text)
    else:
        tasks = []
        blocks = re.split(r'^\s*---\s*$', text, flags=re.MULTILINE)
        for block in blocks:
            lines = [line for line in block.splitlines() if not line.lstrip().startswith('#')]
            checks = [line.partition('VERIFY:')[2].strip() for line in lines if line.startswith('VERIFY:')]
            if len(checks) > 1:
                raise ValueError('Use at most one optional VERIFY line per task')
            prompt = '\n'.join(line for line in lines if not line.startswith('VERIFY:')).strip()
            if prompt:
                tasks.append({'id': 'txt-' + hashlib.sha256(prompt.encode()).hexdigest()[:16],
                              'prompt': prompt, **({'verify': checks[0]} if checks else {})})
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Add at least one task to tasks.txt; separate tasks with ---')
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get('prompt'), str) or not task['prompt'].strip():
            raise ValueError('Every task needs a nonempty prompt')
        task.setdefault('id', 'task-' + hashlib.sha256(task['prompt'].encode()).hexdigest()[:16])
        if not isinstance(task['id'], str) or task['id'] in seen:
            raise ValueError('Task IDs must be unique strings; combine duplicate prompts or use JSON IDs')
        seen.add(task['id'])
        for key in ('workdir', 'verify'):
            if key in task and not isinstance(task[key], str):
                raise ValueError(key + ' must be a string')
    return tasks


def successful_observation(event):
    if event.get('tool') not in {'read_file', 'search_files', 'terminal', 'process_manage'}:
        return False
    try:
        value = json.loads(event.get('result', ''))
    except (TypeError, ValueError):
        return False
    return isinstance(value, dict) and not value.get('error') and not value.get('timed_out') and value.get('exit_code', 0) == 0


def parse_review(result):
    if result.get('failed') or result.get('interrupted'):
        return {'complete': False, 'remaining': 'Review did not finish successfully'}
    reason = str(result.get('exit_reason') or result.get('turn_exit_reason') or '')
    if reason.startswith('max_iterations_reached(') and not result.get('review_recovered'):
        return {'complete': False, 'remaining': 'Review exhausted its turn budget without a separate structured finalization'}
    text = str(result.get('final_response') or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    try:
        verdict = json.loads(text)
    except (ValueError, TypeError):
        return {'complete': False, 'remaining': 'Review must return valid completion JSON: ' + str(text)[-2000:]}
    observed = any(successful_observation(e) for e in result.get('tools', []))
    if isinstance(verdict, dict) and verdict.get('complete') is True and not observed:
        return {'complete': False, 'remaining': 'Reviewer did not inspect the actual result with a successful tool call. Independent inspection is required.'}
    if isinstance(verdict, dict) and (not isinstance(verdict.get('evidence'), str) or not isinstance(verdict.get('remaining'), str)):
        return {'complete': False, 'remaining': 'Review evidence and remaining must be JSON strings'}
    if not isinstance(verdict, dict) or verdict.get('complete') is not True or not observed or not str(verdict.get('evidence', '')).strip() or str(verdict.get('remaining', '')).strip():
        return {'complete': False, 'remaining': str(verdict.get('remaining') or verdict.get('evidence') or 'Review has no independent tool evidence') if isinstance(verdict, dict) else 'Invalid review object'}
    return verdict


def requires_durable_audit(prompt):
    """Recognize full local password/wordlist audits that need objective exhaustion evidence."""
    value = prompt.lower()
    return ('password audit' in value and 'rockyou' in value) or (
        'password hash' in value and 'wordlist' in value and ('all user' in value or 'all password' in value))


def durable_audit_evidence(task_id):
    """Validate the durable John result independently of worker/reviewer prose."""
    jobs = ROOT / 'state' / 'jobs'
    matching = []
    if jobs.is_dir():
        for path in jobs.glob('*/job.json'):
            try:
                value = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if value.get('queue_task_id') == task_id:
                matching.append((path.parent, value))
    if not matching:
        return False, 'No durable job is recorded for this password-audit task'
    matching.sort(key=lambda item: str(item[1].get('created_at', '')))
    folder, current = matching[-1]
    if current.get('status') != 'completed' or current.get('exit_code') != 0:
        return False, 'Durable audit job is ' + str(current.get('status')) + ' with exit ' + str(current.get('exit_code'))
    states = [current]
    history = folder / 'history'
    if history.is_dir():
        for path in history.glob('*.json'):
            try:
                states.append(json.loads(path.read_text(encoding='utf-8')))
            except (OSError, ValueError):
                pass
    john_commands = []
    for value in states:
        command = list(map(str, value.get('command', [])))
        if command and Path(command[0]).name in {'john', 'john.real'}:
            john_commands.append(' '.join(command).lower())
    commands = '\n'.join(john_commands)
    try:
        output = (folder / 'output.log').read_text(encoding='utf-8', errors='replace')
    except OSError as exc:
        return False, 'Durable audit log is unavailable: ' + str(exc)
    if not john_commands or 'rockyou.txt' not in commands or '--format=crypt' not in commands:
        return False, 'Durable job provenance does not show a direct John crypt command processing the RockYou wordlist'
    if not re.search(r'Loaded\s+\d+\s+password hash', output, re.IGNORECASE):
        return False, 'John log does not show that password hashes were loaded'
    if 'Session completed' not in output:
        return False, 'John log does not contain the required Session completed exhaustion marker'
    return True, 'Durable John job exited 0 after loading hashes and recorded Session completed for RockYou'


class Sessions:
    def __init__(self, config, server):
        self.config, self.server = config, server
    def run(self, prompt, cwd, review=False, interactive=False):
        self.server.ensure()
        folder = ROOT / 'logs' / 'sessions' / (str(time.time_ns()) + ('-review' if review else '-work'))
        folder.mkdir(parents=True)
        request = {'prompt': prompt, 'system': REVIEWER if review else CHAT if interactive else WORKER, 'interactive': interactive, 'review': review,
                   'cwd': str(cwd), 'base_url': self.server.base + '/v1',
                   'turns': self.config['review_turns' if review else 'pass_turns'],
                   'max_tokens': self.config.get('chat_max_tokens', self.config['max_tokens']) if interactive else self.config['max_tokens'], 'temperature': self.config['temperature'],
                   'command_timeout': self.config['command_timeout'], 'request_timeout': self.config['request_timeout']}
        request['task_id'] = folder.name
        if self.config.get('_fit_run_id'):
            request['fit_run_id'] = self.config['_fit_run_id']
        request['queue_task_id'] = getattr(self, 'queue_task_id', '')
        request['continuation_segments'] = int(self.config.get('continuation_segments', 1))
        if not review:
            request['system'] += ('\nThe configured maximum for one bounded terminal command is '
                                  + str(self.config['command_timeout'])
                                  + ' seconds. Full candidate-list scans may use that entire time; '
                                    'do not replace it with an arbitrary short timeout.')
        request['backend'] = self.config.get('backend', 'llama.cpp')
        request['local_context_compat'] = self.config.get('local_context_compat', False)
        request['compact_system_prompt'] = self.config.get('compact_system_prompt', False)
        request['model_label'] = self.config.get('model_label', Path(self.config['model']).name)
        request['context_size'] = self.config.get('context_size')
        request['toolsets'] = self.config.get('toolsets', ['terminal', 'file', 'todo'])
        if interactive and request['backend'] == 'llama.cpp':
            request['toolsets'] = self.config.get('chat_toolsets', ['terminal'])
            request['compact_tool_descriptions'] = True
        custom = ROOT / 'prompt.txt'
        if custom.is_file() and not review:
            request['system'] += '\nUSER INSTRUCTIONS:\n' + custom.read_text()
        atomic_json(folder / 'request.json', request)
        from usb_core import HERMES
        process = subprocess.Popen([str(HERMES / 'python' / 'bin' / 'python3.12'),
                                    '-s', '-B', str(ROOT / 'agent_worker.py'),
                                    str(folder / 'request.json'), str(folder / 'result.json')],
                                   cwd=ROOT, start_new_session=True)
        try:
            code = process.wait(timeout=session_wait_timeout(self.config, interactive))
        except subprocess.TimeoutExpired:
            code = 124
        finally:
            with shutdown_signals():
                stop_process(process)
        if interactive:
            result = {'failed': code != 0}
            if self.config.get('_fit_run_id'):
                result['hermes_ready'] = (folder / 'result.ready.json').is_file()
            return result
        if not (folder / 'result.json').exists():
            raise RuntimeError('Hermes session exited (' + str(code) + '); see ' + str(folder))
        result = json.loads((folder / 'result.json').read_text())
        result['session_log'] = str(folder.relative_to(ROOT))
        if code:
            result['failed'] = True
        return result


def run_queue(tasks, state_path, config, sessions):
    state = json.loads(state_path.read_text()) if state_path.exists() else {'version': 2, 'tasks': {}}
    if state.get('version') != 2 or not isinstance(state.get('tasks'), dict):
        raise ValueError('Invalid task-loop state; preserve it before repairing')
    for task in tasks:
        digest = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
        entry = state['tasks'].get(task['id'], {})
        if entry.get('fingerprint') != digest:
            entry = {'fingerprint': digest, 'prompt': task['prompt'], 'status': 'pending', 'passes': 0, 'notes': []}
            state['tasks'][task['id']] = entry
        if entry['status'] == 'completed':
            if requires_durable_audit(task['prompt']):
                valid, reason = durable_audit_evidence(task['id'])
                if not valid:
                    entry.update(status='pending', updated=now())
                    entry.pop('completed', None)
                    entry.pop('evidence', None)
                    entry['notes'].append('Completion reset by durable audit validator: ' + reason)
                    atomic_json(state_path, state)
                    print('Reopened incomplete durable audit: ' + reason, flush=True)
                else:
                    print('Already completed: ' + task['id'], flush=True)
                    continue
            else:
                print('Already completed: ' + task['id'], flush=True)
                continue
        cwd = resolve(ROOT, task.get('workdir', config['workdir']))
        cwd.mkdir(parents=True, exist_ok=True)
        while entry['status'] != 'completed':
            entry.update(status='running', passes=entry['passes'] + 1, updated=now())
            atomic_json(state_path, state)
            print('\nTask ' + task['id'] + ' — pass ' + str(entry['passes']), flush=True)
            try:
                sessions.queue_task_id = task['id']
                prompt = 'ORIGINAL TASK:\n' + task['prompt'] + '\nWORKING DIRECTORY: ' + str(cwd)
                prompt += '\nDURABLE JOB NAME: ' + task['id']
                prompt += '\nPROGRESS FROM PREVIOUS SESSIONS:\n' + '\n'.join(entry['notes'][-4:])
                result = sessions.run(prompt, cwd)
                entry['notes'].append('Worker: ' + str(result.get('final_response', ''))[-6000:])
                entry['last_session'] = result.get('session_log')
                # Persist before review, so an interrupted pass is recoverable.
                atomic_json(state_path, state)
                review = sessions.run('ORIGINAL TASK:\n' + task['prompt'] + '\nWORKING DIRECTORY: ' + str(cwd)
                                      + '\nWORKER REPORT (verify independently):\n' + str(result.get('final_response', ''))[-6000:], cwd, review=True)
                verdict = parse_review(review)
                if verdict['complete'] and requires_durable_audit(task['prompt']):
                    durable, evidence = durable_audit_evidence(task['id'])
                    if not durable:
                        verdict = {'complete': False, 'remaining': evidence}
                    else:
                        verdict['evidence'] = verdict['evidence'] + ' ' + evidence
                if verdict['complete'] and task.get('verify'):
                    check = run_shell(task['verify'], cwd, config['command_timeout'], config['output_chars'])
                    if check['timed_out'] or check['exit_code'] != 0:
                        verdict = {'complete': False, 'remaining': 'Optional legacy check failed: ' + str(check)}
                entry['notes'].append('Review: ' + json.dumps(verdict))
                entry['last_review'] = review.get('session_log')
                if verdict['complete']:
                    entry.update(status='completed', completed=now(), evidence=verdict['evidence'])
                    print('Completed with independent review: ' + task['id'], flush=True)
                else:
                    print('Continuing: ' + verdict.get('remaining', 'More work needed'), flush=True)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                entry['notes'].append('Retry after error: ' + str(exc))
                print('Retrying: ' + str(exc), flush=True)
            entry['notes'] = entry['notes'][-12:]
            entry['updated'] = now()
            atomic_json(state_path, state)
            if entry['status'] != 'completed':
                time.sleep(config['retry_delay'])
    print('All queued tasks completed. Closing Hermes and the model server.', flush=True)
    return 0
