"""Exercise retries and report acceptance without calling a canister."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import benchmark_proposal_assessment as b


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.target = self.directory / 'runs/000'
        self.target.mkdir(parents=True)
        self.record = dict(task_id='rationale', token_ids=[1], input_sha256='input', options=['yes', 'no'])
        b.write(self.directory / 'inputs.json', {'records': [self.record]})
        self.session = dict(input_sha256=b.sha(self.directory / 'inputs.json'), source_hashes={}, wasm_sha256=b.MODULE)
        b.write(self.directory / 'session.json', self.session)
        b.write(self.directory / 'prepared.json', {'tasks': {'task': {'status': 'pending', 'record_index': 0}}})
        self.report = dict(wasm_sha256=b.MODULE, deployed_wasm_sha256=b.MODULE, input_hash='input',
                           comparison={'typed_output_valid': True, 'value': 'yes'}, replayed_queries=0,
                           executed_query_count=2, query_count=2, total_instructions=3,
                           decision_query={'ok': {'decision': dict(value='yes', abstained=False,
                               probabilities=[.7, .2], unknown_probability=.1, raw_logits=[2., 1., 0.],
                               calibration_version='p3-r2-s000291-authored', instructions=1)}})

    def execute(self, report=None):
        def subprocess_run(*args, **kwargs):
            self.assertFalse((self.target / 'old.response.bin').exists())
            b.write(self.target / 'report.json', report or self.report)
            return types.SimpleNamespace(returncode=0)
        with patch.object(b, 'command', return_value=['fake']), patch.object(b.subprocess, 'run', side_effect=subprocess_run) as process, \
                patch.object(b, 'score') as score, patch.object(b, 'compact_artifacts'):
            b.run(self.directory, retry_errors=True)
        return process, score

    def test_retry_preserves_failed_journal_and_starts_fresh(self):
        (self.target / 'old.response.bin').write_bytes(b'completed query')
        b.write(self.target / 'error.json', {'returncode': 1})
        process, score = self.execute()
        self.assertEqual(process.call_count, 1)
        self.assertTrue(score.called)
        self.assertEqual((self.directory / 'aborted/000-attempt-001/old.response.bin').read_bytes(), b'completed query')
        self.assertFalse((self.target / 'error.json').exists())

    def test_interrupted_journal_also_starts_fresh(self):
        (self.target / 'old.response.bin').write_bytes(b'partial')
        self.execute()
        self.assertTrue((self.directory / 'aborted/000-attempt-001/old.response.bin').exists())

    def test_invalid_existing_report_cannot_bypass_validation_or_scoring(self):
        invalid = dict(self.report, replayed_queries=1, executed_query_count=1)
        b.write(self.target / 'report.json', invalid)
        with self.assertRaisesRegex(ValueError, 'invalid canister run report'):
            b.run(self.directory)
        with self.assertRaisesRegex(ValueError, 'invalid canister run report'):
            b.score(self.directory)
        self.execute()
        self.assertEqual(json.loads((self.directory / 'aborted/000-attempt-001/report.json').read_text()), invalid)

    def test_new_invalid_report_is_rejected_again_on_next_invocation(self):
        invalid = dict(self.report, replayed_queries=1)
        with self.assertRaises(ValueError):
            self.execute(invalid)
        with self.assertRaises(ValueError):
            b.run(self.directory)

    def test_valid_completed_run_is_reused_without_inference(self):
        b.write(self.target / 'report.json', self.report)
        with patch.object(b.subprocess, 'run') as process, patch.object(b, 'score'):
            b.run(self.directory)
        process.assert_not_called()

    def test_module_input_fallback_and_decision_are_checked(self):
        invalids = [dict(self.report, deployed_wasm_sha256='different'),
                    dict(self.report, input_hash='different'), dict(self.report, fallback=True)]
        invalid = copy.deepcopy(self.report)
        invalid['decision_query']['ok']['decision']['probabilities'] = [float('nan'), .2]
        invalids.append(invalid)
        for report in invalids:
            with self.subTest(report=report):
                # Corrupt saved JSON may contain NaN even though our writer rejects it.
                (self.target / 'report.json').write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    b.verified_report(self.target, self.record, self.session)


if __name__ == '__main__':
    unittest.main()
