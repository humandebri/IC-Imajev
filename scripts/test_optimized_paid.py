"""Paid validation keeps new measurements separate from receipts and checkpoints."""
import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import verify_optimized_paid
from verify_optimized_paid import request_id_for_run, checkpoint_comparison, verification_summary


class OptimizedPaidTests(unittest.TestCase):
    def test_cli_repeated_runs_use_fresh_receipts_and_leave_performance_unverified(self):
        receipts, executed = {}, []

        class Wire:
            def __init__(self, directory, target, *args, **kwargs):
                self.target = target

            def call(self, kind, value=None, relay=None, cycles=0):
                fee = 123
                refund = cycles
                if kind == 'configure_paid':
                    result = {'transport_error': 'owner only'} if relay else {'Ok': None}
                elif kind == 'paid_config':
                    result = dict(base_fee=10_000_000_000, fee_per_token=3_000_000_000, reserve_cycles=2_000_000_000_000)
                elif kind == 'inference_step':
                    result = {'Err': 'self only'}
                elif kind == 'quote':
                    result = {'Ok': dict(fee=fee, max_steps=8)}
                elif kind == 'inference_status':
                    receipt = receipts.get((self.target, value)) if relay else None
                    result = None if receipt is None else dict(state={'Completed': receipt[1]}, refund='None')
                elif kind == 'infer':
                    key = (self.target, value['request_id'])
                    receipt = receipts.get(key)
                    if receipt:
                        result = {'Ok': receipt[1]} if receipt[0] == value['request'] else {'Err': 'IdConflict'}
                    elif cycles < fee:
                        result = {'Err': {'InsufficientCycles': {'required': fee}}}
                    else:
                        executed.append(key)
                        output = dict(job_id=len(executed), paid_cycles=fee, prefix_tokens=5, suffix_tokens=4,
                                      workers=[dict(instructions=100)], decision=dict(value='yes', abstained=False,
                                      calibration_version='fixture', raw_logits=[1.0], probabilities=[1.0], unknown_probability=0.0))
                        receipts[key] = (value['request'], output)
                        result, refund = {'Ok': output}, cycles-fee
                else:
                    raise AssertionError(kind)
                return dict(result=result, forward=dict(refunded=refund), request_bytes=100, reply_bytes=100)

        def module_hash(command, **kwargs):
            return json.dumps({'ok': {'module_hash': command[2] + '-module'}})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root/'fixture.json'
            fixture.write_text(json.dumps({'cases': [dict(total=n, request=dict(token_ids=[1]*n, options=['yes'])) for n in (9,96,128,256,512,1024)]}))
            reports = []
            for i in range(2):
                argv = ['verify', '--reference-canister', 'reference', '--candidate-canister', 'candidate',
                        '--reference-module', 'reference-module', '--candidate-module', 'candidate-module',
                        '--relay', 'relay', '--url', 'http://localhost:8002', '--identity-pem', str(root/'test.pem'),
                        '--directory', str(root/f'run{i}'), '--helper', str(root/'helper'), '--call-helper', str(root/'call'),
                        '--bridge', str(root/'bridge'), '--fixture', str(fixture), '--lengths', '9']
                with patch('sys.argv', argv), patch.object(verify_optimized_paid, 'PaidTransport', Wire), patch.object(verify_optimized_paid.subprocess, 'check_output', module_hash), contextlib.redirect_stdout(io.StringIO()):
                    verify_optimized_paid.main()
                reports.append(json.loads((root/f'run{i}'/'report.json').read_text()))
        self.assertEqual(len(executed), 4, 'Both canisters must execute again on the second run')
        self.assertEqual(len(set(executed)), 4)
        for report in reports:
            self.assertEqual(report['failures'], [])
            self.assertFalse(report['paid_instruction_gate_passed'])
            self.assertFalse(report['whole_message_accounting_complete'])

    def test_new_runs_cannot_replay_previous_receipts(self):
        first = verification_summary('reference', 'candidate')
        second = verification_summary('reference', 'candidate')
        for total in (9, 96, 128, 256, 512, 1024):
            a = request_id_for_run(first['run_id'], total)
            b = request_id_for_run(second['run_id'], total)
            self.assertNotEqual(a, b)
            self.assertLessEqual(len(a), 64)
            self.assertEqual(a, request_id_for_run(first['run_id'], total))

    def test_passing_checkpoints_cannot_complete_instruction_gate(self):
        baseline = dict(workers=[dict(instructions=100), dict(instructions=200)])
        actual = dict(workers=[dict(instructions=105), dict(instructions=210)])
        metrics = checkpoint_comparison(baseline, actual)
        self.assertTrue(metrics['worker_checkpoint_within_105_percent'])
        report = verification_summary('reference', 'candidate')
        report['complete'] = report['paid_numerical_gate_passed'] = True
        report['inputs'].append(dict(total=9, **metrics))
        self.assertFalse(report['paid_instruction_gate_passed'])
        self.assertFalse(report['whole_message_accounting_complete'])
        self.assertTrue(any('Whole-message' in requirement for requirement in report['remaining']))
        self.assertNotIn('reference', metrics)
        self.assertNotIn('candidate', metrics)

    def test_checkpoint_regression_is_explicitly_diagnostic(self):
        row = checkpoint_comparison(dict(workers=[dict(instructions=100)]), dict(workers=[dict(instructions=106)]))
        self.assertFalse(row['worker_checkpoint_within_105_percent'])


if __name__ == '__main__':
    unittest.main()
