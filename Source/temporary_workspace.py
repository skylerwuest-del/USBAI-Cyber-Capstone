"""Exact-path, locked Linux session scratch storage; no external dependencies."""
import contextlib
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import tempfile
import uuid

_current = None


def host_identity():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


class Workspace:
    def __init__(self, root, temp_root=None):
        self.root = Path(root).resolve()
        self.base = Path(temp_root or '/tmp').resolve(strict=True)
        if self.base == self.root or self.base.is_relative_to(self.root):
            raise RuntimeError('Temporary root must be separate from SOURCE')
        self.records = self.root / 'state' / 'temporary-sessions'
        self.records.mkdir(parents=True, exist_ok=True)
        self.record = dict(session=uuid.uuid4().hex, token=uuid.uuid4().hex,
                           created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           pid=os.getpid(), uid=os.getuid(), host=host_identity(), base=str(self.base))
        self.path = Path(tempfile.mkdtemp(prefix='hermes_session_', dir=self.base))
        self.fd = None
        self.closed = False
        self.processes = []
        try:
            os.chmod(self.path, 0o700)
            info = self.path.stat()
            self.record.update(path=str(self.path), device=info.st_dev, inode=info.st_ino)
            self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (self.path / '.hermes-session.json').open('x') as handle:
                json.dump(self.record, handle)
            self.record_path = self.records / (self.record['session'] + '.json')
            with self.record_path.open('x') as handle:
                json.dump(self.record, handle)
        except BaseException:
            if self.fd is not None:
                os.close(self.fd)
            # No deletion without a complete, verifiable registration.
            raise

    def validate(self, record):
        if len(record['session']) != 32 or any(c not in '0123456789abcdef' for c in record['session']):
            raise RuntimeError('Invalid session identifier')
        path = Path(record['path'])
        if record['host'] != host_identity() or record['uid'] != os.getuid():
            raise RuntimeError('Session belongs to another host boot or user; preserved')
        if record['base'] != str(self.base) or path.parent != self.base or path == self.base:
            raise RuntimeError('Unapproved temporary path')
        if path == self.root or path.is_relative_to(self.root) or path == Path.home():
            raise RuntimeError('Protected application path')
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700 or
                    (info.st_dev, info.st_ino) != (record['device'], record['inode'])):
                raise RuntimeError('Workspace ownership or directory identity changed')
            marker = os.open('.hermes-session.json', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            with os.fdopen(marker) as handle:
                if json.load(handle) != record:
                    raise RuntimeError('Workspace verification marker mismatch')
            return fd
        except BaseException:
            os.close(fd)
            raise

    def usage(self, record=None):
        record = record or self.record
        fd = self.validate(record)
        files = directories = size = 0
        errors = []
        try:
            def failed(error):
                errors.append(str(error))
            for _, dirs, names, parent in os.fwalk('.', dir_fd=fd, follow_symlinks=False, onerror=failed):
                directories += len(dirs)
                files += len(names)
                for name in names:
                    try:
                        size += os.stat(name, dir_fd=parent, follow_symlinks=False).st_size
                    except OSError as exc:
                        errors.append(str(exc))
            return dict(files=files, directories=directories, bytes=size, errors=errors)
        finally:
            os.close(fd)

    def remove(self, record, own=False):
        if own and (record != self.record or self.closed and self.fd is None):
            raise RuntimeError('Not the owned current session')
        # Retain records for missing or changed paths: absence alone cannot prove identity.
        try:
            fd = self.validate(record)
        except FileNotFoundError:
            if not Path(record['path']).exists() and not Path(record['path']).is_symlink():
                return 'Workspace already absent'
            raise
        try:
            if not own:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError('Session is active; cleanup refused') from None
            if not shutil.rmtree.avoids_symlink_attacks:
                raise RuntimeError('Safe descriptor-based cleanup unavailable')
            # Traverse only the opened, verified directory, never a substituted path.
            for name in os.listdir(fd):
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    shutil.rmtree(name, dir_fd=fd)
                else:
                    os.unlink(name, dir_fd=fd)
            path = Path(record['path'])
            info = path.lstat()
            if (info.st_dev, info.st_ino) != (record['device'], record['inode']):
                raise RuntimeError('Workspace path changed during cleanup')
            path.rmdir()
            if path.exists() or path.is_symlink():
                raise RuntimeError('Workspace removal verification failed')
            (self.records / (record['session'] + '.json')).unlink(missing_ok=True)
            return 'Verified temporary workspace removed'
        finally:
            os.close(fd)

    def previous(self):
        rows = []
        for path in sorted(self.records.glob('*.json')):
            try:
                if path.is_symlink():
                    raise RuntimeError('Symlink record rejected')
                record = json.loads(path.read_text())
                if path.name != record['session'] + '.json':
                    raise RuntimeError('Record identity mismatch')
                if record == self.record:
                    continue
                fd = self.validate(record)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    usage = self.usage(record)
                finally:
                    os.close(fd)
                rows.append((record, usage, None))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                rows.append((None, None, str(path) + ': protected/unavailable: ' + str(exc)))
        return rows

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if any(process.poll() is None for process in self.processes):
                raise RuntimeError('Owned subprocess still running; cleanup deferred for recovery')
            print('HERMES temporary cleanup: ' + self.remove(self.record, own=True), flush=True)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            print('HERMES temporary cleanup failed; workspace preserved for review: ' + str(exc), flush=True)
        finally:
            if self.fd is not None:
                os.close(self.fd)


@contextlib.contextmanager
def session(root):
    global _current
    if _current is not None:
        yield _current
        return
    manager = Workspace(root)
    _current = manager
    saved = {key: os.environ.get(key) for key in ('TMPDIR', 'TMP', 'TEMP', 'HERMES_SESSION_TMP')}
    cached = tempfile.tempdir
    handlers = {}
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    try:
        for key in saved:
            os.environ[key] = str(manager.path)
        tempfile.tempdir = str(manager.path)
        handlers = {number: signal.signal(number, terminate) for number in (signal.SIGTERM, signal.SIGHUP)}
        yield manager
    finally:
        # The callers' worker/server finally blocks have completed before this point.
        with __import__('usb_core').shutdown_signals():
            manager.close()
        for number, handler in handlers.items():
            signal.signal(number, handler)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = cached
        _current = None


def menu():
    manager = _current
    if manager is None:
        print('No managed temporary workspace. Restart through bash launch.sh.')
        return
    while True:
        print('\nHERMES TEMPORARY FILE MANAGEMENT\n1) View Current Session Temporary Directory\n2) View Temporary Storage Usage\n3) Clean HERMES Temporary Files\n4) Review Previous Session Temporary Files\n5) Return to Main Menu')
        choice = input('Choose: ').strip()
        try:
            if choice == '5':
                return
            if choice == '1':
                print('Session: ' + manager.record['session'] + '\nPath: ' + str(manager.path))
                print('Exists: ' + str(manager.path.exists()))
                fd = manager.validate(manager.record)
                os.close(fd)
                print('Ownership verified. Cleanup available at shutdown.')
            elif choice == '2':
                print(manager.usage())
            elif choice == '3':
                # The supervisor cannot prove that arbitrary tools left no open files.
                print('Active workspace may be needed by inference or child tools. Cleanup is deferred to shutdown.')
                if input('Clean automatically at shutdown? [y/N]: ').lower() == 'y':
                    print('Cleanup scheduled; exit HERMES normally to release processes first.')
            elif choice == '4':
                for record, usage, error in manager.previous():
                    if error:
                        print(error)
                        continue
                    print(str(record['path']) + '\nCreated: ' + record['created'] + '\n' + str(usage))
                    if input('Delete this verified abandoned workspace? [y/N]: ').lower() == 'y':
                        print(manager.remove(record))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            print('Temporary storage unavailable; preserved: ' + str(exc))


def inherited_lock():
    """Keep ownership locked if a supervisor dies while its child still runs."""
    return (_current.fd,) if _current is not None and _current.fd is not None else ()


def track_process(process):
    if _current is not None:
        _current.processes.append(process)
    return process
