from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_selection import selected_model


class ModelSelectionTests(unittest.TestCase):
    def test_folder_tracks_replacement_without_saved_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'models/3) medium'
            folder.mkdir(parents=True)
            first = folder / 'first.gguf'
            first.touch()
            self.assertEqual(selected_model(root, 'models/3) medium'), first.resolve())
            first.unlink()
            replacement = folder / 'replacement.gguf'
            replacement.touch()
            self.assertEqual(selected_model(root, 'models/3) medium'), replacement.resolve())

    def test_missing_folder_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'No complete GGUF model'):
                selected_model(Path(directory), 'models/3) medium')

    def test_split_model_skips_incomplete_and_uses_first_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'models/3) medium'
            folder.mkdir(parents=True)
            for name in ('a-00001-of-00002.gguf', 'b-00001-of-00002.gguf', 'b-00002-of-00002.gguf'):
                (folder / name).touch()
            self.assertEqual(selected_model(root, 'models/3) medium'),
                             (folder / 'b-00001-of-00002.gguf').resolve())

    def test_explicit_model_still_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(selected_model(root, 'models/custom.gguf'),
                             (root / 'models/custom.gguf').resolve())


if __name__ == '__main__':
    unittest.main()
