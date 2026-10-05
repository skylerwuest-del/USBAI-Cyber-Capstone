import contextlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capstone as c
import models_menu as menu


def action(kind, **values):
    return dict({"action": kind, "command": "", "path": "", "content": "", "note": "progress"}, **values)


class FakeServer:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        a = next(self.actions)
        if isinstance(a, BaseException):
            raise a
        return a


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for directory in ("logs", "state", "workspace", "models"):
            (self.root / directory).mkdir()
        self.root_patch = patch.object(c, "ROOT", self.root)
        self.root_patch.start()
        self.cfg = {"workdir": "workspace", "command_timeout": 2, "output_chars": 1000,
                    "history_chars": 5000, "retry_delay": 0.01}
        self.state_path = self.root / "state" / "progress.json"
        self.task = {"id": "demo", "prompt": "Make result.txt contain yes", "verify":
                     "python3 -c \"from pathlib import Path; assert Path('result.txt').read_text() == 'yes'\""}

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def run_task(self, server, state=None, task=None):
        state = state or {"version": 1, "tasks": {}}
        with contextlib.redirect_stdout(io.StringIO()):
            c.run_task(task or self.task, state, self.state_path, server, self.cfg)
        return state

    def test_failed_verification_retries_then_only_verified_work_is_completed(self):
        server = FakeServer([action("finish"), action("write", path="result.txt", content="yes"), action("finish")])
        state = self.run_task(server)
        entry = state["tasks"]["demo"]
        self.assertEqual(entry["status"], "completed")
        self.assertEqual(entry["steps"], 3)
        self.assertNotEqual(entry["history"][0]["result"]["exit_code"], 0)
        self.assertEqual(c.read_state(self.state_path), state)
        self.run_task(FakeServer([]), state)
        self.assertEqual(entry["steps"], 3)

    def test_retry_after_model_error_and_tool_error(self):
        server = FakeServer([OSError("temporarily unavailable"), action("read", path="missing"),
                             action("write", path="result.txt", content="yes"), action("finish")])
        state = self.run_task(server)
        self.assertEqual(state["tasks"]["demo"]["status"], "completed")
        self.assertIn("error", state["tasks"]["demo"]["history"][0]["result"])

    def test_interrupted_action_is_checkpointed_and_resume_receives_uncertainty(self):
        with patch.object(c, "execute_action", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_task(FakeServer([action("write", path="result.txt", content="yes")]))
        saved = c.read_state(self.state_path)
        self.assertEqual(saved["tasks"]["demo"]["pending_action"]["action"], "write")
        resumed = FakeServer([action("write", path="result.txt", content="yes"), action("finish")])
        self.run_task(resumed, saved)
        self.assertIn("may have run already", resumed.messages[0][1]["content"])
        self.assertEqual(c.read_state(self.state_path)["tasks"]["demo"]["status"], "completed")

    def test_changed_task_with_same_id_reruns(self):
        state = self.run_task(FakeServer([action("write", path="result.txt", content="yes"), action("finish")]))
        changed = dict(self.task, prompt="Changed instructions")
        server = FakeServer([action("finish")])
        self.run_task(server, state, changed)
        self.assertEqual(len(server.messages), 1)
        self.assertEqual(state["tasks"]["demo"]["steps"], 1)

    def test_timeout_kills_process_group_and_preserves_output(self):
        result = c.run_shell("echo started; sleep 10", self.root, 0.1, 100)
        self.assertTrue(result["timed_out"])
        self.assertNotEqual(result["exit_code"], 0)
        self.assertIn("started", result["output"])

    def test_noisy_shell_has_bounded_output(self):
        result = c.run_shell("python3 -c \"print('x' * 1000000)\"", self.root, 2, 100)
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["output_truncated"])
        self.assertLessEqual(len(result["output"]), 100)

    def test_chat_template_override_only_matches_its_selected_model(self):
        template = self.root / "qwen.jinja"
        template.write_text("{{ messages }}")
        config = {"model": "models/qwen.gguf", "chat_templates": {"qwen.gguf": "qwen.jinja"}}
        self.assertEqual(c.chat_template_args(config), ["--chat-template-file", str(template)])
        config["model"] = "models/another-model.gguf"
        self.assertEqual(c.chat_template_args(config), [])
        config["model"] = "models/qwen.gguf"
        template.unlink()
        with self.assertRaises(FileNotFoundError): c.chat_template_args(config)

    def test_occupied_port_is_not_mistaken_for_owned_server(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            server = c.Server({"port": listener.getsockname()[1]})
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                server.ensure()
            self.assertIsNone(server.process)

    def test_duplicate_runner_lock_rejected(self):
        lock = self.root / "state" / "runner.lock"
        with c.instance_lock(lock):
            with self.assertRaisesRegex(RuntimeError, "already"):
                with c.instance_lock(lock):
                    pass

    def test_plain_text_queue_requires_verification_and_is_stable(self):
        path = self.root / "tasks.txt"
        path.write_text("# comment\nWrite result.txt\nVERIFY: test -f result.txt\n---\nWrite second\nVERIFY: test -f second\n")
        tasks = c.load_tasks(path)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["verify"], "test -f result.txt")
        self.assertEqual(tasks, c.load_tasks(path))
        path.write_text("Do work without a verification check")
        with self.assertRaisesRegex(ValueError, "VERIFY"):
            c.load_tasks(path)

    def test_context_keeps_whole_action_result_pairs(self):
        entry = {"history": [{"action": action("shell", command="pwd"), "result": {"output": "x" * 300}}] * 20}
        messages = c.messages_for(self.task, entry, self.root, 1000)
        self.assertLess(len(messages), 42)
        self.assertEqual((len(messages) - 2) % 2, 0)
        self.assertEqual(messages[0]["role"], "system")


class ModelMenuTests(unittest.TestCase):
    def test_huggingface_repository_blob_and_resolve_urls(self):
        for kind in ("blob", "resolve"):
            self.assertEqual(menu.parse_url("https://huggingface.co/owner/repo/" + kind + "/main/sub/model.gguf?download=true"),
                             ("owner/repo", "main", "sub/model.gguf"))
        self.assertEqual(menu.parse_url("https://huggingface.co/owner/repo"), ("owner/repo", "main", ""))
        for bad in ("http://huggingface.co/a/b", "https://example.com/a/b", "https://huggingface.co/a/b/blob/main/model.safetensors", "https://huggingface.co/a/b/blob/main/../model.gguf"):
            with self.assertRaises(ValueError):
                menu.parse_url(bad)

    def test_split_model_downloads_all_shards_even_if_url_is_second_shard(self):
        files = {"m-00001-of-00002.gguf": {}, "m-00002-of-00002.gguf": {}}
        self.assertEqual(menu.shard_group("m-00002-of-00002.gguf", files), list(files))
        self.assertEqual(menu.selectable_names(files), ["m-00001-of-00002.gguf"])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            menu.shard_group("m-00001-of-00002.gguf", {"m-00001-of-00002.gguf": {}})

    def test_pull_downloads_pinned_files_and_updates_selection_after_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "models").mkdir()
            (root / "config.json").write_text('{"model": "old.gguf"}')
            contents = b"GGUFexample"
            digest = __import__('hashlib').sha256(contents).hexdigest()
            info = {"sha": "abc123", "siblings": [{"rfilename": "model.gguf", "size": len(contents), "lfs": {"sha256": digest}}]}
            downloads = []
            def download(asset, path):
                downloads.append(asset)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            with patch.object(menu, "ROOT", root), patch.object(menu, "metadata", return_value=info), patch.object(menu, "download", side_effect=download), contextlib.redirect_stdout(io.StringIO()):
                menu.pull("https://huggingface.co/o/r/blob/main/model.gguf")
            self.assertIn("/resolve/abc123/", downloads[0]["url"])
            chosen = json.loads((root / "config.json").read_text())["model"]
            self.assertTrue((root / chosen).exists())


if __name__ == "__main__":
    unittest.main()
