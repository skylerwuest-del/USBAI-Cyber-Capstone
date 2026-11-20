"""Isolated offline scratch lifecycle and hostile-path regression tests."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import temporary_workspace as tw
import menu
from portable_env import usb_environment


class TemporaryWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory()
        self.addCleanup(self.fixture.cleanup)
        self.base = Path(self.fixture.name)
        self.root = self.base / 'Source'
        self.root.mkdir()
        self.host = self.base / 'scratch'
        self.host.mkdir()
        self.managers = []
        self.addCleanup(self.close_all)

    def close_all(self):
        for manager in self.managers:
            manager.close()

    def create(self):
        manager = tw.Workspace(self.root, self.host)
        self.managers.append(manager)
        return manager

    def abandon(self, manager):
        os.close(manager.fd)
        manager.fd = None
        manager.closed = True

    def test_unique_private_creation(self):
        first, second = self.create(), self.create()
        self.assertNotEqual(first.path, second.path)
        self.assertEqual(first.path.stat().st_mode & 0o777, 0o700)
        self.assertNotEqual(first.record['session'], second.record['session'])

    def test_cleanup_unrelated_and_restart_no_setup(self):
        for relative in ('models/model.gguf', 'hermes/python/bin/python3.12', 'config.json'):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('persistent')
        unrelated = self.host / 'unrelated'
        unrelated.mkdir()
        (unrelated / 'keep').write_text('keep')
        manager = self.create()
        (manager.path / 'scratch').write_text('scratch')
        with patch('subprocess.run', side_effect=AssertionError('no setup or network')):
            manager.close()
            replacement = self.create()
        self.assertFalse(manager.path.exists())
        self.assertTrue(replacement.path.exists())
        self.assertEqual((unrelated / 'keep').read_text(), 'keep')
        for relative in ('models/model.gguf', 'hermes/python/bin/python3.12', 'config.json'):
            self.assertEqual((self.root / relative).read_text(), 'persistent')

    def test_missing_directory(self):
        manager = self.create()
        shutil.rmtree(manager.path)
        manager.close()
        manager.close()

    def test_symlink_and_substitution(self):
        manager = self.create()
        outside = self.host / 'outside'
        outside.mkdir()
        (outside / 'keep').write_text('keep')
        (manager.path / 'link').symlink_to(outside, target_is_directory=True)
        manager.close()
        self.assertTrue((outside / 'keep').exists())
        manager = self.create()
        original = manager.path.with_name('saved')
        manager.path.rename(original)
        manager.path.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OSError):
            manager.remove(manager.record)
        self.assertTrue((outside / 'keep').exists())

    def test_dangerous_path_and_marker_rejected(self):
        manager = self.create()
        for path in (self.host, self.root, Path('/'), Path.home()):
            with self.assertRaises(RuntimeError):
                manager.remove(dict(manager.record, path=str(path)))
        (manager.path / '.hermes-session.json').write_text('{}')
        with self.assertRaises(RuntimeError):
            manager.remove(manager.record)

    def test_active_and_abandoned_recovery(self):
        active = self.create()
        reviewer = self.create()
        self.assertTrue(any(error for _, _, error in reviewer.previous()))
        with self.assertRaisesRegex(RuntimeError, 'active'):
            reviewer.remove(active.record)
        self.abandon(active)
        rows = reviewer.previous()
        self.assertEqual(rows[0][0], active.record)
        reviewer.remove(rows[0][0])
        self.assertFalse(active.path.exists())

    def test_orphaned_child_retains_active_lock(self):
        owner = self.create()
        reviewer = self.create()
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                 stdin=subprocess.PIPE, pass_fds=(owner.fd,))
        try:
            self.abandon(owner)
            with self.assertRaisesRegex(RuntimeError, 'active'):
                reviewer.remove(owner.record)
        finally:
            child.communicate(timeout=5)
        reviewer.remove(owner.record)
        self.assertFalse(owner.path.exists())

    def test_termination_signal_runs_cleanup(self):
        real_workspace = tw.Workspace
        with patch.object(tw, 'Workspace', side_effect=lambda root: real_workspace(root, self.host)):
            with self.assertRaises(SystemExit):
                with tw.session(self.root) as manager:
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        self.assertFalse(manager.path.exists())

    def test_other_boot_and_unverified_protected(self):
        old = self.create()
        reviewer = self.create()
        self.abandon(old)
        record = dict(old.record, host='other-boot')
        old.record_path.write_text(json.dumps(record))
        self.assertIsNone(reviewer.previous()[0][0])
        old.record_path.write_text(json.dumps(dict(old.record, token='forged')))
        self.assertIsNone(reviewer.previous()[0][0])
        self.assertTrue(old.path.exists())

    def test_usage_does_not_follow_symlinks(self):
        manager = self.create()
        (manager.path / 'file').write_bytes(b'123')
        (manager.path / 'folder').mkdir()
        (manager.path / 'link').symlink_to(self.root, target_is_directory=True)
        self.assertEqual(manager.usage()['files'], 2)  # marker and scratch file
        self.assertEqual(manager.usage()['errors'], [])

    def test_menu_integration_and_active_deferral(self):
        manager = self.create()
        with patch.object(tw, '_current', manager), patch('builtins.input', side_effect=['12', '1', '2', '3', 'y', '4', '5', '0']), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(menu.main(), 0)
        self.assertIn('deferred to shutdown', output.getvalue())
        self.assertTrue(manager.path.exists())

    def test_shutdown_interrupt_and_offline(self):
        real_workspace = tw.Workspace
        def isolated(root):
            return real_workspace(root, self.host)
        for exception in (None, KeyboardInterrupt, RuntimeError, SystemExit):
            with patch.object(tw, 'Workspace', side_effect=isolated), patch('socket.socket', side_effect=AssertionError('network forbidden')):
                try:
                    with tw.session(self.root) as manager:
                        self.assertEqual(os.environ['TMPDIR'], str(manager.path))
                        if exception:
                            raise exception()
                except (KeyboardInterrupt, RuntimeError, SystemExit):
                    pass
            self.assertFalse(manager.path.exists())
            self.assertIsNone(tw._current)

    def test_shutdown_preserves_workspace_when_child_stop_fails(self):
        owner = self.create()
        owner.processes.append(Mock(poll=Mock(return_value=None)))
        owner.close()
        self.assertTrue(owner.path.exists())
        reviewer = self.create()
        reviewer.remove(owner.record)

    def test_creation_failure(self):
        with patch.object(tw.tempfile, 'mkdtemp', side_effect=PermissionError('read-only')):
            with self.assertRaises(PermissionError):
                self.create()

    def test_child_environment(self):
        # No host /tmp fixture: verify validation failure does not silently fallback.
        with self.assertRaises(RuntimeError):
            usb_environment(self.root, {'HERMES_SESSION_TMP': str(self.host)})


if __name__ == '__main__':
    unittest.main()
