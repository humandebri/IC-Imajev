import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from paid_prefix import COMMON_PREFIX, MAX_TOKENS, length_fixture
from prove_paid_prefix5_local import verify_cases
from measure_paid_prefix5 import calculate

ROOT = Path(__file__).resolve().parents[1]
RECORD = dict(token_ids=COMMON_PREFIX + [220, 16, 198, 1, 2, 3, 4], options=['yes', 'no'])


class Wire:
    def __init__(self, fail=False):
        self.calls = []
        self.receipts = {}
        self.fail = fail

    def call(self, kind, value=None, relay=None, cycles=0):
        self.calls.append((kind, copy.deepcopy(value), relay, cycles))
        if kind == 'quote':
            if len(value['token_ids']) > MAX_TOKENS:
                return {'result': {'Err': {'Invalid': 'input bounds'}}}
            return {'result': {'Ok': {'fee': 1000, 'prefix_tokens': 5, 'suffix_tokens': len(value['token_ids']) - 5}}}
        if kind != 'infer':
            raise AssertionError(kind)
        assert set(value) == {'request', 'request_id'}
        count = len(value['request']['token_ids'])
        if count > MAX_TOKENS:
            result = {'Err': {'Invalid': 'input bounds'}}
            refunded = cycles
        elif self.fail:
            result = {'Err': {'Failed': {'refund': 'Done'}}}
            refunded = cycles - 1000
        elif value['request_id'] in self.receipts:
            result = self.receipts[value['request_id']]
            refunded = cycles
        else:
            result = {'Ok': {'paid_cycles': 1000, 'prefix_tokens': 5, 'suffix_tokens': count - 5,
                            'workers': [{'instructions': 100000000000 // 4, 'stage': 1, 'heap_pages': 100}], 'decision': {'value': 'yes'}}}
            self.receipts[value['request_id']] = result
            refunded = cycles - 1000
        return {'request': value, 'result': result, 'forward': {'refunded': refunded}, 'seconds': 1.0}


class CurrentPaidProofTests(unittest.TestCase):
    def test_fault_recovery_rejects_a_second_refund(self):
        class FaultWire(Wire):
            def __init__(self, double_refund=False):
                super().__init__()
                self.fault = False
                self.refund = 'Pending'
                self.double_refund = double_refund

            def call(self, kind, value=None, relay=None, cycles=0):
                if kind == 'paid_fault':
                    self.fault = value['trap']
                    return {'result': None}
                if kind == 'retry_inference_refund':
                    repeated = self.refund == 'Done'
                    self.refund = 'Done'
                    return {'result': {'Ok': 'Done'}, 'forward': {
                        'balance_before': 10000,
                        'balance_after': 11000 if not repeated or self.double_refund else 9999}}
                if kind == 'infer' and value['request_id'].endswith('-fault'):
                    return {'request': value, 'result': {'Err': {'Failed': {'refund': self.refund}}},
                            'forward': {'refunded': 0 if self.fault else cycles}}
                return super().call(kind, value, relay, cycles)

        rows = verify_cases(FaultWire(), 'relay', RECORD, 'a'*64, [491], fault_recovery=True)
        self.assertTrue(rows[-1]['fault_refund_retry_no_double_refund'])
        with self.assertRaises(AssertionError):
            verify_cases(FaultWire(double_refund=True), 'relay', RECORD, 'a'*64, [491], fault_recovery=True)

    def test_boundary_fixtures_preserve_prefix_and_chat_tail(self):
        for count in [9, 94, 490, 491, 512, 513, 768, 1024, 1025]:
            ids = length_fixture(RECORD, count)
            self.assertEqual(len(ids), count)
            self.assertEqual(ids[:5], COMMON_PREFIX)
            self.assertEqual(ids[-4:], RECORD['token_ids'][-4:])
        with self.assertRaisesRegex(ValueError, 'five-token'):
            length_fixture(dict(RECORD, token_ids=[248045, 846, 198, 56555] + [1]*40), 512)

    def test_new_boundary_proof_requires_success_and_replays_same_request(self):
        wire = Wire()
        rows = verify_cases(wire, 'relay', RECORD, 'a'*64, [490, 491, 1024, 1025])
        self.assertEqual([r['tokens'] for r in rows], [490, 491, 1024, 1025])
        self.assertTrue(all(r['duplicate_no_charge'] for r in rows[:3]))
        self.assertTrue(rows[-1]['rejected_before_payment'])
        self.assertEqual(len(wire.receipts), 3)
        self.assertTrue(all(call[2] == 'relay' for call in wire.calls if call[0] == 'infer'))

    def test_refunded_accepted_request_does_not_count_as_success(self):
        observed = []
        with self.assertRaises(KeyError):
            verify_cases(Wire(fail=True), 'relay', RECORD, 'a'*64, [491], observe=observed.append)
        self.assertEqual(observed, [])

    def test_tariff_uses_only_current_successful_measurements(self):
        rows = verify_cases(Wire(), 'relay', RECORD, 'a'*64, [491, 1025])
        report = dict(complete=True, prefix_tokens=5, network='local', module='test', cases=rows)
        value = calculate(report)
        self.assertEqual(len(value['measurements']), 1)
        config = value['config']
        for row in value['measurements']:
            self.assertGreaterEqual(config['base_fee'] + config['fee_per_token']*row['suffix_tokens'], row['estimated_cycles'])
        with self.assertRaises(ValueError):
            calculate(dict(report, prefix_tokens=27))

    def test_historical_entrypoints_stop_before_loading_assets_or_touching_network(self):
        for name in ['prove_paid_update.py', 'prove_paid_update_candid.py', 'prove_paid_update_faults.py',
                     'prove_paid_token_limit_local.py', 'measure_paid_pricing_poc.py', 'measure_paid_pricing_poc_final.py']:
            result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts'/name)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Historical prefix27 proof', result.stderr)


if __name__ == '__main__':
    unittest.main()
