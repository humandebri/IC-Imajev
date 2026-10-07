"""Fresh retries preserve failed evidence without replaying query journals."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_fresh_proposal_assessment as fresh


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.evaluation = self.root / 'evaluation'
        self.run = self.evaluation / 'runs/000'
        self.prefix = self.root / 'prefixes/00'
        self.packets = self.root / 'packets/00'
        for path, marker in [(self.prefix, 'report.json'), (self.packets, 'cache.json'),
                             (self.packets, 'report.json')]:
            path.mkdir(parents=True, exist_ok=True)
            (path / marker).write_text('{}')
        (self.root / 'plan.json').write_text(json.dumps(dict(
            records=[dict(record=0, bank=0, prefix=27, suffix=1, route='query32')],
            banks=[dict(record=0)])))
        self.calls = []

    def partial(self, directory):
        journal = directory / 'queries/18000000.response.bin'
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_bytes(b'previous response')
        return journal

    def subprocess_run(self, command, **kwargs):
        destination = Path(command[command.index('--directory') + 1])
        self.assertFalse(any((destination / 'queries').glob('*.response.bin')))
        self.calls.append(destination)
        (destination / 'report.json').write_text('{}')
        if destination == self.packets:
            (destination / 'cache.json').write_text('{}')

    def execute(self):
        def validate(index):
            self.assertEqual(index, 0)
            self.assertFalse((self.run / 'queries/18000000.response.bin').exists())
            (self.run / 'verified.json').write_text('{}')
            return {}
        with patch.object(fresh, 'D', self.root), patch.object(fresh, 'E', self.evaluation), \
                patch.object(fresh, 'frozen'), patch.object(fresh, 'command', return_value=['fake']), \
                patch.object(fresh, 'validate', side_effect=validate), patch.object(fresh, 'summarize', return_value={}), \
                patch.object(fresh.subprocess, 'run', side_effect=self.subprocess_run):
            fresh.execute()

    def test_unverified_run_is_archived_and_restarted_without_old_responses(self):
        self.partial(self.run)
        (self.run / 'report.json').write_text('{"replayed_queries": 1}')
        self.execute()
        archive = self.root / 'aborted/evaluation/runs/000-attempt-001'
        self.assertEqual((archive / 'queries/18000000.response.bin').read_bytes(), b'previous response')
        self.assertEqual((archive / 'report.json').read_text(), '{"replayed_queries": 1}')
        self.assertEqual(self.calls, [self.run])

    def test_retry_preserves_existing_archived_attempts(self):
        previous = self.root / 'aborted/evaluation/runs/000-attempt-001'
        previous.mkdir(parents=True)
        (previous / 'evidence.txt').write_text('first attempt')
        self.partial(self.run)
        self.execute()
        self.assertEqual((previous / 'evidence.txt').read_text(), 'first attempt')
        self.assertTrue((previous.parent / '000-attempt-002/queries/18000000.response.bin').exists())

    def test_verified_run_is_preserved_without_new_calls(self):
        self.partial(self.run)
        (self.run / 'verified.json').write_text('{}')
        self.execute()
        self.assertEqual(self.calls, [])
        self.assertTrue((self.run / 'queries/18000000.response.bin').exists())
        self.assertFalse((self.root / 'aborted').exists())

    def test_incomplete_prefix_restarts_and_invalidates_its_codec_packets(self):
        (self.prefix / 'report.json').unlink()
        self.partial(self.prefix)
        self.partial(self.packets)
        self.execute()
        self.assertEqual(self.calls, [self.prefix, self.packets, self.run])
        for category in ['prefixes', 'packets']:
            self.assertEqual((self.root / 'aborted' / category / '00-attempt-001/queries/18000000.response.bin').read_bytes(),
                             b'previous response')

    def test_codec_cache_without_terminal_report_is_rebuilt(self):
        (self.packets / 'report.json').unlink()
        self.partial(self.packets)
        self.execute()
        self.assertEqual(self.calls, [self.packets, self.run])
        self.assertTrue((self.root / 'aborted/packets/00-attempt-001/cache.json').exists())
        self.assertFalse((self.root / 'aborted/prefixes').exists())

    def test_failed_subprocess_evidence_survives_next_call(self):
        with patch.object(fresh, 'D', self.root), patch.object(fresh, 'frozen'), \
                patch.object(fresh.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['fake'])):
            with self.assertRaises(subprocess.CalledProcessError):
                fresh.call(['fake'], self.run)
        self.execute()
        archive = self.root / 'aborted/evaluation/runs/000-attempt-001'
        self.assertEqual(json.loads((archive / 'command.json').read_text()), ['fake'])
        self.assertTrue((archive / 'run.log').exists())


if __name__ == '__main__':
    unittest.main()
