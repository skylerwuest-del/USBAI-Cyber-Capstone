import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import task_loop
from task_loop import (load_tasks, parse_review, run_queue, session_wait_timeout,
                       requires_durable_audit, durable_audit_evidence, WORKER, REVIEWER)
from agent_worker import needs_continuation
from usb_core import run_shell


def reviewed(complete=True, evidence='Read the actual output file', tools=True, failed=False):
    return {'final_response': json.dumps({'complete': complete, 'evidence': evidence,
                                         'remaining': '' if complete else 'Output is still wrong'}),
            'failed': failed,
            'tools': [{'tool': 'read_file', 'result': json.dumps({'content': 'correct'})}] if tools else []}


class QueueTests(unittest.TestCase):
    def test_worker_deadline_can_be_disabled_for_long_jobs(self):
        self.assertIsNone(session_wait_timeout({'pass_timeout': 0}))
        self.assertIsNone(session_wait_timeout({'pass_timeout': 60}, interactive=True))
        self.assertEqual(session_wait_timeout({'pass_timeout': 60}), 60)

    def test_max_iteration_handoff_stays_in_worker(self):
        self.assertTrue(needs_continuation({'turn_exit_reason': 'max_iterations_reached(32/32)'}))
        self.assertFalse(needs_continuation({'turn_exit_reason': 'text_response(finish_reason=stop)'}))
        self.assertFalse(needs_continuation({'turn_exit_reason': 'max_iterations_reached(32/32)', 'failed': True}))

    def test_long_job_prompts_require_completion_evidence(self):
        self.assertIn('capstone-job start', WORKER)
        self.assertIn('Session completed', WORKER)
        self.assertIn('requested target or candidate set was processed', REVIEWER)

    def test_exhausted_review_is_rejected_unless_recovered(self):
        value = reviewed()
        value['exit_reason'] = 'max_iterations_reached(12/12)'
        self.assertFalse(parse_review(value)['complete'])
        value['review_recovered'] = True
        self.assertTrue(parse_review(value)['complete'])

    def test_password_wordlist_audit_requires_durable_evidence(self):
        self.assertTrue(requires_durable_audit('Run a password audit for ALL USERS against rockyou.txt'))
        self.assertFalse(requires_durable_audit('Create a report about password policy'))

    def test_durable_audit_requires_john_exhaustion_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = root / 'state/jobs/audit'
            job.mkdir(parents=True)
            state = {'queue_task_id': 'audit-task', 'status': 'completed', 'exit_code': 0,
                     'created_at': '2026-01-01',
                     'command': ['john', '--format=crypt', '--wordlist=/usb/rockyou.txt', '/etc/shadow']}
            (job / 'job.json').write_text(json.dumps(state))
            (job / 'output.log').write_text('Loaded 5 password hashes\nSession aborted\n')
            original = task_loop.ROOT
            task_loop.ROOT = root
            try:
                self.assertFalse(durable_audit_evidence('audit-task')[0])
                (job / 'output.log').write_text('Loaded 5 password hashes\nSession completed\n')
                self.assertTrue(durable_audit_evidence('audit-task')[0])
                state['command'] = ['bash', '-c', 'john --format=crypt /usb/rockyou.txt; echo Session completed']
                (job / 'job.json').write_text(json.dumps(state))
                self.assertFalse(durable_audit_evidence('audit-task')[0])
            finally:
                task_loop.ROOT = original

    def test_plain_english_needs_no_verify(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tasks.txt'
            path.write_text('# comment\nWrite a file.\n---\nEdit another file.\n')
            tasks = load_tasks(path)
            self.assertEqual([t['prompt'] for t in tasks], ['Write a file.', 'Edit another file.'])
            self.assertNotIn('verify', tasks[0])

    def test_completion_without_tools_is_rejected(self):
        self.assertFalse(parse_review(reviewed(tools=False))['complete'])
        self.assertIn('Independent inspection', parse_review(reviewed(tools=False))['remaining'])

    def test_review_cannot_use_an_evidence_array(self):
        value = reviewed()
        value['final_response'] = json.dumps({'complete': True, 'evidence': ['looks done'], 'remaining': ''})
        self.assertFalse(parse_review(value)['complete'])

    def test_empty_model_response_is_rejected(self):
        value = reviewed()
        value['final_response'] = None
        self.assertFalse(parse_review(value)['complete'])

    def test_failed_observation_is_rejected(self):
        value = reviewed()
        value['tools'] = [{'tool': 'terminal', 'result': json.dumps({'exit_code': 1, 'output': 'failed'})}]
        self.assertFalse(parse_review(value)['complete'])

    def test_missing_file_is_rejected(self):
        value = reviewed()
        value['tools'] = [{'tool': 'read_file', 'result': json.dumps({'error': 'not found'})}]
        self.assertFalse(parse_review(value)['complete'])

    def test_interrupted_review_is_rejected(self):
        value = reviewed()
        value['interrupted'] = True
        self.assertFalse(parse_review(value)['complete'])

    def test_incomplete_review_is_rejected(self):
        self.assertFalse(parse_review(reviewed(complete=False))['complete'])

    def test_successful_review_requires_evidence(self):
        self.assertTrue(parse_review(reviewed())['complete'])
        self.assertFalse(parse_review(reviewed(evidence=''))['complete'])

    def test_fresh_pass_retries_and_resume_skips_completed(self):
        with tempfile.TemporaryDirectory() as folder:
            cwd = Path(folder)
            state = cwd / 'state.json'
            class Sessions:
                workers = 0
                reviews = 0
                def run(self, prompt, workdir, review=False):
                    if review:
                        self.reviews += 1
                        # Check actual file content independently of the worker's report.
                        return reviewed(complete=(workdir / 'output.txt').read_text() == 'correct')
                    self.workers += 1
                    (workdir / 'output.txt').write_text('wrong' if self.workers == 1 else 'correct')
                    return {'final_response': 'I am done', 'tools': []}
            sessions = Sessions()
            task = {'id': 'one', 'prompt': 'Write correct to output.txt', 'workdir': str(cwd)}
            config = {'workdir': str(cwd), 'retry_delay': 0, 'command_timeout': 5, 'output_chars': 1000}
            run_queue([task], state, config, sessions)
            self.assertEqual(sessions.workers, 2)
            self.assertEqual(sessions.reviews, 2)
            data = json.loads(state.read_text())['tasks']['one']
            self.assertEqual(data['status'], 'completed')
            self.assertIn('Output is still wrong', '\n'.join(data['notes']))
            run_queue([task], state, config, sessions)
            self.assertEqual(sessions.workers, 2)
            # Changing a task invalidates previous completion.
            task['prompt'] += ' and check it'
            run_queue([task], state, config, sessions)
            self.assertEqual(sessions.workers, 3)

    def test_retry_after_session_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            cwd = Path(folder)
            class Sessions:
                attempts = 0
                def run(self, prompt, workdir, review=False):
                    if review:
                        return reviewed()
                    self.attempts += 1
                    if self.attempts == 1:
                        raise RuntimeError('model restarted')
                    return {'final_response': 'Finished'}
            sessions = Sessions()
            run_queue([{'id': 'crash', 'prompt': 'Do work', 'workdir': str(cwd)}], cwd / 'state.json',
                      {'workdir': str(cwd), 'retry_delay': 0}, sessions)
            self.assertEqual(sessions.attempts, 2)

    def test_terminal_timeout_stops_background_children(self):
        with tempfile.TemporaryDirectory() as folder:
            cwd = Path(folder)
            result = run_shell('sleep 30 & echo $! > child.pid; wait', cwd, 0.2, 1000)
            self.assertTrue(result['timed_out'])
            child = int((cwd / 'child.pid').read_text())
            # A terminated child may briefly be a zombie waiting for the host to reap it.
            status = Path('/proc') / str(child) / 'stat'
            self.assertTrue(not status.exists() or status.read_text().split()[2] == 'Z')


if __name__ == '__main__':
    unittest.main()
