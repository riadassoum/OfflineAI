"""Platform-independent regression tests; native Windows/NVDA engines are stubbed."""
import builtins
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'addon/globalPlugins/offlineAI'


def load(name):
    spec = importlib.util.spec_from_file_location('offlineai_test_' + name, SOURCE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    # Translation is per-module in NVDA. Avoid importing the wx plugin package.
    module.__dict__['_'] = lambda text: text
    with patch.dict(sys.modules, {
        'addonHandler': types.SimpleNamespace(initTranslation=lambda: None),
        'logHandler': types.SimpleNamespace(log=logging.getLogger('test')),
    }):
        spec.loader.exec_module(module)
    return module


class TextTests(unittest.TestCase):
    def setUp(self):
        self.engine = load('inference')

    def test_long_arabic_keeps_punctuation_and_paragraphs(self):
        text = '؟!\n' + ('السلام عليكم. كيف الحال؟\n\n' * 100) + '!!!\n'
        with patch.object(self.engine, 'count_tokens', side_effect=len):
            chunks, budget = self.engine.split_for_context(text, '', {'n_ctx': 300})
            self.assertEqual(''.join(chunks), text)
            self.assertTrue(all(len(chunk) <= budget for chunk in chunks))

    def test_punctuation_only_selection_is_not_lost(self):
        text = '؟!.' * 100
        with patch.object(self.engine, 'count_tokens', side_effect=len):
            chunks, budget = self.engine.split_for_context(text, '', {'n_ctx': 300})
            self.assertEqual(''.join(chunks), text)
            self.assertTrue(all(len(chunk) <= budget for chunk in chunks))

    def test_single_character_over_budget(self):
        with patch.object(self.engine, 'count_tokens', side_effect=lambda s: len(s) * 5):
            self.assertEqual(self.engine.truncate_to_tokens('ع', 1), ('', True))

    def test_truncation_keeps_valid_unicode_prefix(self):
        text = 'مرحبًا بالعالم 🙂' * 30
        with patch.object(self.engine, 'count_tokens', side_effect=lambda s: len(s.encode('utf-8'))):
            result, cut = self.engine.truncate_to_tokens(text, 27)
            self.assertTrue(cut)
            self.assertTrue(text.startswith(result))
            self.assertLessEqual(len(result.encode('utf-8')), 27)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.engine = load('inference')
        self.closed = []
        self.engine._llm = types.SimpleNamespace(close=lambda: self.closed.append(True))

    def test_busy_model_not_unloaded(self):
        self.assertTrue(self.engine.try_acquire_busy())
        self.assertFalse(self.engine.unload())
        self.assertEqual(self.closed, [])
        self.engine.release_busy()
        self.assertEqual(self.closed, [])
        self.assertTrue(self.engine.unload())
        self.assertEqual(self.closed, [True])

    def test_shutdown_defers_close_until_request_releases(self):
        self.engine.try_acquire_busy()
        self.assertFalse(self.engine.unload(force=True))
        self.assertEqual(self.closed, [])
        self.engine.release_busy()
        self.assertEqual(self.closed, [True])
        self.assertFalse(self.engine.is_busy())
        self.assertFalse(self.engine.is_loaded())

    def test_cleanup_blocks_new_request_until_close_finishes(self):
        acquired = []
        self.engine._llm.close = lambda: acquired.append(self.engine.try_acquire_busy())
        self.assertTrue(self.engine.unload())
        self.assertEqual(acquired, [False])
        self.assertTrue(self.engine.try_acquire_busy())
        self.engine.release_busy()

    def test_delete_rejects_busy_model_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.gguf'
            path.write_bytes(b'model')
            self.engine._loaded_path = str(path)
            self.engine.try_acquire_busy()
            self.assertFalse(self.engine.delete_model_file(str(path)))
            self.assertTrue(path.exists())
            self.assertEqual(self.closed, [])
            self.engine.release_busy()
            self.assertTrue(self.engine.delete_model_file(str(path)))
            self.assertFalse(path.exists())
            self.assertEqual(self.closed, [True])

    def test_delete_error_releases_lock(self):
        with patch.object(self.engine.os, 'remove', side_effect=PermissionError('in use')):
            with self.assertRaises(PermissionError):
                self.engine.delete_model_file('missing.gguf')
        self.assertFalse(self.engine.is_busy())


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.backup = load('backup')

    def test_invalid_manifests_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.zip'
            for manifest in ([], None, {}, {'version': 999, 'contents': []},
                             {'version': 1, 'contents': 'config'},
                             {'version': 1, 'contents': [{}]}):
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr(self.backup.MANIFEST_NAME, json.dumps(manifest))
                result, error = self.backup.inspect_backup(path)
                self.assertIsNone(result)
                self.assertTrue(error)

    def test_restore_into_new_config_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'config.json'
            source.write_text('{"settings": {"translate_lang_1": "ar"}}')
            archive = root / 'backup.zip'
            self.assertEqual(self.backup.create_backup(archive, source, None, None), (True, None))
            target = root / 'new' / 'config.json'
            restored, error = self.backup.restore_backup(archive, str(target), None, None)
            self.assertIsNone(error)
            self.assertEqual(restored, ['config'])
            self.assertEqual(json.loads(target.read_text())['settings']['translate_lang_1'], 'ar')


if __name__ == '__main__':
    unittest.main()
