"""Exercise editing and run-and-exit on exFAT using the actual USB model."""
import ast
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from capstone import instance_lock

env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
with tempfile.TemporaryDirectory(prefix=".menu-check-", dir=ROOT / "state") as directory:
    stage = Path(directory)
    for name in ("menu.py", "capstone.py", "models_menu.py", "setup.py"):
        shutil.copyfile(ROOT / name, stage / name)
        ast.parse((stage / name).read_text(), feature_version=(3, 10))
    for name in ("state", "logs", "workspace"):
        (stage / name).mkdir()
    cfg = json.loads((ROOT / "config.json").read_text())
    for key in ("model", "server"):
        cfg[key] = str(ROOT / cfg[key])
    cfg["chat_templates"] = {name: str(ROOT / path) for name, path in cfg["chat_templates"].items()}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        cfg["port"] = sock.getsockname()[1]
    (stage / "config.json").write_text(json.dumps(cfg))
    original = "Create old.txt containing old.\nVERIFY: test -s old.txt\n"
    (stage / "tasks.txt").write_text(original)
    (stage / "tasks.example.txt").write_text(original)
    queue = (
        "Create first.txt containing exactly first followed by a newline. Inspect the result.\n"
        "VERIFY: python3 -c \"from pathlib import Path; assert Path('first.txt').read_text() == 'first\\n'\"\n"
        "---\n"
        "Create second.txt containing exactly second followed by a newline. Inspect the result.\n"
        "VERIFY: python3 -c \"from pathlib import Path; assert Path('second.txt').read_text() == 'second\\n'\"\n"
    )
    editor = stage / "editor with spaces.py"
    editor.write_text("from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text(" + repr(queue) + ")\n")
    editor_env = dict(env, CAPSTONE_EDITOR=shlex.join([sys.executable, str(editor)]))
    edited = subprocess.run([sys.executable, "menu.py"], cwd=stage, input="7\n0\n", text=True,
                            capture_output=True, env=editor_env, timeout=30)
    assert edited.returncode == 0 and "2 valid task(s)" in edited.stdout, edited.stdout + edited.stderr
    assert (stage / "tasks.txt").read_text() == queue
    assert (stage / "tasks.txt.bak").read_text() == original
    print("PASS: menu option 7 opens the editor, preserves a backup, and validates both tasks.", flush=True)

    with instance_lock(stage / "state" / "runner.lock"):
        blocked = subprocess.run([sys.executable, "menu.py"], cwd=stage, input="7\n0\n", text=True,
                                 capture_output=True, env=editor_env, timeout=30)
    assert "Another CapStone runner" in blocked.stdout
    assert (stage / "tasks.txt.bak").read_text() == original
    print("PASS: editing is blocked while a run owns the session lock.", flush=True)

    (stage / "tasks.txt").write_text("This task has no completion check.\n")
    invalid = subprocess.run([sys.executable, "menu.py"], cwd=stage, input="8\n0\n", text=True,
                             capture_output=True, env=env, timeout=30)
    assert "exactly one VERIFY:" in invalid.stdout and "Starting USB model server" not in invalid.stdout
    assert not (stage / "logs" / "server.log").exists()
    print("PASS: an invalid task file is rejected before any model starts.", flush=True)
    (stage / "tasks.txt").write_text(queue)

    model_pids = set()
    with (ROOT / "logs" / "menu-smoke-run.txt").open("w") as log:
        process = subprocess.Popen([sys.executable, "menu.py"], cwd=stage, stdin=subprocess.PIPE,
                                   stdout=log, stderr=subprocess.STDOUT, text=True, env=env,
                                   start_new_session=True)
        process.stdin.write("8\n")
        process.stdin.close()
        deadline = time.monotonic() + 300
        try:
            while process.poll() is None:
                assert time.monotonic() < deadline, "Runtime check exceeded its 300-second test timeout"
                for entry in Path("/proc").iterdir():
                    if not entry.name.isdigit():
                        continue
                    try:
                        args = (entry / "cmdline").read_bytes().split(b"\0")
                    except (FileNotFoundError, PermissionError, ProcessLookupError):
                        continue
                    if args and args[0] == os.fsencode(cfg["server"]) and b"--port" in args:
                        index = args.index(b"--port")
                        if args[index + 1] == str(cfg["port"]).encode():
                            model_pids.add(int(entry.name))
                time.sleep(0.5)
            assert process.returncode == 0, "See logs/menu-smoke-run.txt"
            model_clean = bool(model_pids) and all(not Path("/proc", str(pid)).exists() for pid in model_pids)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for pid in model_pids:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    output = (ROOT / "logs" / "menu-smoke-run.txt").read_text()
    assert "Closing CapStone." in output and "For unattended execution" not in output
    assert output.count("CapStone — USB local AI") == 1
    state = json.loads((stage / "state" / "progress.json").read_text())
    assert len(state["tasks"]) == 2 and all(task["status"] == "completed" for task in state["tasks"].values())
    assert (stage / "workspace" / "first.txt").read_text() == "first\n"
    assert (stage / "workspace" / "second.txt").read_text() == "second\n"
    assert model_clean, "The menu exited before its model stopped"
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", cfg["port"]))
    shutil.copyfile(stage / "logs" / "actions.jsonl", ROOT / "logs" / "menu-smoke-actions.jsonl")
    print("PASS: option 8 completed both tasks with real CPU inference, stopped its model, released its port, and exited the menu.", flush=True)
    resumed = subprocess.run([sys.executable, "menu.py"], cwd=stage, input="8\n", text=True,
                             capture_output=True, env=env, timeout=30)
    assert resumed.returncode == 0 and "Closing CapStone." in resumed.stdout
    assert "Starting USB model server" not in resumed.stdout
    print("PASS: completed tasks are skipped on the next run, and the menu exits without starting a model.", flush=True)
