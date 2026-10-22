import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models_menu as menu


class ModelPullTests(unittest.TestCase):
    def test_search_filters_other_models_and_prefers_original_publisher(self):
        results = [{"id": name} for name in (
            "community/Qwen3-14B-GGUF", "other/Qwen3-14B-abliterated-GGUF",
            "other/Qwen3-14B2-GGUF", "Qwen/Qwen3-14B-GGUF")]
        with patch.object(menu.urllib.request, "urlopen", return_value=io.StringIO(json.dumps(results))):
            self.assertEqual(menu.find_gguf_repositories("Qwen/Qwen3-14B"),
                             ["Qwen/Qwen3-14B-GGUF", "community/Qwen3-14B-GGUF"])

    def test_original_model_url_downloads_selected_conversion(self):
        payload = b"GGUFtest"
        import hashlib
        conversion = {"sha": "a" * 40, "siblings": [{
            "rfilename": "model-Q4_K_M.gguf", "size": len(payload),
            "lfs": {"sha256": hashlib.sha256(payload).hexdigest()}}]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text('{"model": "old.gguf"}')
            def download(asset, path):
                path.write_bytes(payload)
            with patch.object(menu, "ROOT", root), \
                 patch.object(menu, "metadata", side_effect=[{"siblings": [{"rfilename": "model.safetensors"}]}, conversion]) as metadata, \
                 patch.object(menu, "find_gguf_repositories", return_value=["community/Qwen3-14B-GGUF"]), \
                 patch("builtins.input", side_effect=["1", "1"]), \
                 patch.object(menu, "download", side_effect=download) as transfer:
                selected = menu.pull("https://huggingface.co/Qwen/Qwen3-14B")
            self.assertEqual(selected.read_bytes(), payload)
            self.assertEqual(json.loads((root / "config.json").read_text())["model"], str(selected.relative_to(root)))
            self.assertIn("/resolve/" + "a" * 40 + "/", transfer.call_args.args[0]["url"])
            self.assertEqual(metadata.call_args_list[0].args, ("Qwen/Qwen3-14B", "main"))

    def test_cancel_conversion_selection_does_not_download(self):
        with patch.object(menu, "metadata", return_value={"siblings": []}), \
             patch.object(menu, "find_gguf_repositories", return_value=["community/Qwen3-14B-GGUF"]), \
             patch("builtins.input", return_value=""), patch.object(menu, "download") as transfer:
            self.assertIsNone(menu.pull("https://huggingface.co/Qwen/Qwen3-14B"))
            transfer.assert_not_called()

    def test_empty_conversion_does_not_repeat_search(self):
        with patch.object(menu, "metadata", return_value={"siblings": []}), \
             patch.object(menu, "find_gguf_repositories") as search:
            with self.assertRaisesRegex(ValueError, "selected conversion"):
                menu.pull("https://huggingface.co/community/Qwen3-14B-GGUF", discover=False)
            search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
