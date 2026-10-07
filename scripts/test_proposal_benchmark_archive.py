"""The 600–660 preparer reads and verifies the selected copied archive."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.proposal_assessment import benchmark_600_660_binary as benchmark


class ReachedTokenizer(Exception):
    pass


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / 'selected archive'
        snapshots = self.archive / 'snapshots'
        snapshots.mkdir(parents=True)
        records, proposals = [], []
        for proposal_id in range(600, 661):
            path = snapshots / f'proposal-{proposal_id}.json'
            path.write_text(json.dumps(dict(id=proposal_id, root_canister_id='sns-root')))
            digest = benchmark.sha(path)
            records.append(dict(proposal_id=proposal_id,
                                snapshot=f'/historical/snapshots/proposal-{proposal_id}.json', sha256=digest))
            proposals.append(dict(proposal_id=proposal_id, snapshot_sha256=digest,
                                  task={}, prediction=dict(label='hold')))
        self.manifest = dict(complete=True, sns_root='sns-root', records=records)
        (self.archive / 'manifest.json').write_text(json.dumps(self.manifest))
        reference = self.root / 'tools/proposal_assessment/gpt-6.1-sol-medium-20261005-1509'
        reference.mkdir(parents=True)
        (reference / 'vote-results.json').write_text(json.dumps(dict(complete=True, proposals=proposals)))
        self.output = self.root / 'new evaluation'

    def prepare(self):
        with patch.object(benchmark, 'ROOT', self.root), \
                patch('prepare_text.TextPreparer', side_effect=ReachedTokenizer):
            benchmark.prepare(self.output, snapshot_archive=self.archive)

    def test_all_copied_snapshots_are_verified_before_tokenizer(self):
        with patch.object(benchmark, 'read_snapshot', wraps=benchmark.read_snapshot) as read:
            with self.assertRaises(ReachedTokenizer):
                self.prepare()
        self.assertEqual(read.call_count, 61)
        self.assertEqual([call.args[1]['proposal_id'] for call in read.call_args_list], list(range(600, 661)))
        self.assertTrue(all(call.args[0] == self.archive and call.args[2] == 'sns-root'
                            for call in read.call_args_list))
        self.assertFalse(self.output.exists())

    def test_missing_selected_copy_does_not_fall_back_to_existing_original(self):
        original = self.root / 'original/snapshots/proposal-600.json'
        original.parent.mkdir(parents=True)
        selected = self.archive / 'snapshots/proposal-600.json'
        original.write_bytes(selected.read_bytes())
        selected.unlink()
        self.manifest['records'][0]['snapshot'] = str(original)
        (self.archive / 'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertTrue(original.exists())
        self.assertFalse(self.output.exists())

    def test_modified_selected_copy_is_rejected(self):
        (self.archive / 'snapshots/proposal-600.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_cli_passes_explicit_archive_to_preparer(self):
        with patch.object(benchmark, 'prepare') as prepare, \
                patch('sys.argv', ['benchmark', 'prepare', '--output', str(self.output),
                                   '--snapshot-archive', str(self.archive), '--compact-ratio']):
            benchmark.main()
        prepare.assert_called_once_with(self.output.resolve(), True, True,
                                        snapshot_archive=self.archive.resolve())


if __name__ == '__main__':
    unittest.main()
