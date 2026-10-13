"""Exercise archive selection without inference or canister access."""
import importlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import assess_proposal_range as assessment
import run_fresh_proposal_assessment as fresh


class ReachedSnapshot(Exception):
    pass


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / 'relocated source'
        self.archive = self.source / 'snapshots'
        (self.archive / 'snapshots').mkdir(parents=True)
        (self.source / 'evaluation').mkdir()
        self.record = dict(proposal_id=500, snapshot='/historical/snapshots/proposal-500.json', sha256='hash')
        self.manifest = dict(complete=True, all_fetched=True, first=500, last=500,
                             sns_root='root', records=[self.record])
        (self.archive / 'manifest.json').write_text(json.dumps(self.manifest))
        (self.source / 'evaluation/report.json').write_text('{"proposals": []}')

    def test_range_reads_explicit_archive(self):
        output = self.root / 'range output'
        (output / 'snapshots').mkdir(parents=True)
        (output / 'snapshots/manifest.json').write_bytes((self.archive / 'manifest.json').read_bytes())
        with patch.object(assessment, 'TextPreparer'), \
                patch.object(assessment, 'read_snapshot', side_effect=ReachedSnapshot) as read:
            with self.assertRaises(ReachedSnapshot):
                assessment.prepare(output, reuse=False, snapshot_archive=self.archive)
        read.assert_called_once_with(self.archive, self.record, 'root')

    def test_range_rejects_different_manifest_before_tokenizer(self):
        output = self.root / 'range output'
        (output / 'snapshots').mkdir(parents=True)
        (output / 'snapshots/manifest.json').write_text('{}')
        with patch.object(assessment, 'TextPreparer') as tokenizer:
            with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
                assessment.prepare(output, snapshot_archive=self.archive)
        tokenizer.assert_not_called()

    def test_fresh_manifest_copy_keeps_explicit_source_archive(self):
        output = self.root / 'fresh output'
        with patch.object(fresh, 'D', output), patch.object(fresh, 'E', output / 'evaluation'), \
                patch.object(fresh.assessment, 'prepare', side_effect=ReachedSnapshot) as prepare:
            with self.assertRaises(ReachedSnapshot):
                fresh.prepare(self.archive)
        self.assertEqual((output / 'snapshots/manifest.json').read_bytes(),
                         (self.archive / 'manifest.json').read_bytes())
        prepare.assert_called_once_with(output, reuse=False, snapshot_archive=self.archive)

    def test_all_window_preparers_select_relocated_source(self):
        for name, prefix in [('windows', 42), ('query32', 27), ('compact86', 27)]:
            with self.subTest(name=name):
                module = importlib.import_module('prepare_all_proposal_' + name)
                output = self.root / name
                tokenizer = types.SimpleNamespace(encode=lambda text, add_special_tokens: [1] * prefix + [
                    8 if 'another' in text else 7])
                preparer = types.SimpleNamespace(tokenizer=tokenizer, render=lambda text: text)
                with patch.object(module, 'D', output), patch.object(module, 'OLD', self.source), \
                        patch.object(module, 'TextPreparer', return_value=preparer), \
                        patch.object(module, 'read_snapshot', side_effect=ReachedSnapshot) as read, \
                        patch('sys.argv', ['prepare', '--source-directory', str(self.source), '--directory', str(output)]):
                    with self.assertRaises(ReachedSnapshot):
                        module.main()
                read.assert_called_once_with(self.archive, self.record, 'root')


if __name__ == '__main__':
    unittest.main()
