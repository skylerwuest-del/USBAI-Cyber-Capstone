#!/usr/bin/env python3
"""CapStone terminal menu."""
import json
import os
import shlex
import shutil
import subprocess
import sys

from capstone import ROOT, instance_lock, load_tasks
from models_menu import choose_installed, pull


def terminal_editor():
    configured = os.environ.get("CAPSTONE_EDITOR", "").strip()
    if configured:
        args = shlex.split(configured)
        if not args or not shutil.which(args[0]):
            raise RuntimeError("CAPSTONE_EDITOR must name an installed terminal editor")
        return args
    for name in ("nano", "vim", "vi"):
        editor = shutil.which(name)
        if editor:
            return [editor]
    raise RuntimeError("No terminal editor found. Install nano or set CAPSTONE_EDITOR to a terminal editor.")


def edit_preordained_tasks():
    editor = terminal_editor()
    path = ROOT / "tasks.txt"
    with instance_lock(ROOT / "state" / "runner.lock"):
        if not path.exists():
            shutil.copyfile(ROOT / "tasks.example.txt", path)
        backup = ROOT / "tasks.txt.bak"
        shutil.copyfile(path, backup)
        print("Editing " + str(path), flush=True)
        print("Previous contents saved in tasks.txt.bak. Each task needs a VERIFY: check.", flush=True)
        result = subprocess.run(editor + [str(path)], cwd=ROOT)
        if result.returncode != 0:
            print("Editor exited with code " + str(result.returncode) + "; review tasks.txt before running.")
            return
        try:
            tasks = load_tasks(path)
        except (OSError, ValueError) as exc:
            print("Task file needs correction: " + str(exc))
            print("Choose option 7 to edit it again. The previous contents are in tasks.txt.bak.")
            return
        print("Saved tasks.txt: " + str(len(tasks)) + " valid task(s). Choose option 8 to run them.")


def run_preordained_tasks():
    path = ROOT / "tasks.txt"
    load_tasks(path)  # Reject invalid queues before launching the worker/model.
    print("Running tasks.txt. CapStone will close after every task is verified.", flush=True)
    result = subprocess.run([sys.executable, str(ROOT / "capstone.py"), "--tasks", str(path)], cwd=ROOT)
    if result.returncode == 0:
        # capstone.py returns success only after its finally block shuts down
        # the owned model. Shell tools also clean up their process groups.
        print("All tasks verified; the owned AI processes have stopped. Closing CapStone.", flush=True)
        return True
    print("Task run stopped (exit code " + str(result.returncode) + "). Saved progress can be resumed with option 8.")
    return False


def main():
    while True:
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        print("\nCapStone — USB local AI")
        print("Model: " + cfg["model"])
        print("1) Start / resume task queue (choose a file)")
        print("2) Serve model only")
        print("3) Download & select model from Hugging Face URL")
        print("4) Select an installed model")
        print("5) Show task progress")
        print("6) Start local model + CAI harness")
        print("7) Edit preordained tasks (tasks.txt)")
        print("8) Run preordained tasks, then close CapStone")
        print("0) Exit")
        try:
            choice = input("Choose: ").strip()
            if choice == "0":
                return 0
            if choice == "1":
                taskfile = input("Task file [tasks.json]: ").strip() or "tasks.json"
                subprocess.run([sys.executable, str(ROOT / "capstone.py"), "--tasks", taskfile], cwd=ROOT)
            elif choice == "2":
                subprocess.run([sys.executable, str(ROOT / "capstone.py"), "--serve-only"], cwd=ROOT)
            elif choice == "3":
                pull()
            elif choice == "4":
                choose_installed()
            elif choice == "6":
                subprocess.run([sys.executable, str(ROOT / "capstone.py"), "--cai"], cwd=ROOT)
            elif choice == "5":
                subprocess.run([sys.executable, str(ROOT / "capstone.py"), "--status"], cwd=ROOT)
            elif choice == "7":
                edit_preordained_tasks()
            elif choice == "8":
                if run_preordained_tasks():
                    return 0
            else:
                print("Choose a listed option.")
        except EOFError:
            print("\nFor unattended execution use: bash launch.sh --run --tasks tasks.txt")
            return 0
        except KeyboardInterrupt:
            print("\nReturning to menu. Task progress is saved.")
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
            print("Menu error: " + str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
