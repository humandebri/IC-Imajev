import unittest
from pathlib import Path
import tempfile
import struct
from unittest.mock import patch
from paid_instruction_accounting import account, decode_snapshot, protobuf_fields, principal_hex, CheckpointMeter


def varint(value):
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128); value >>= 7
    out.append(value)
    return bytes(out)


def integer(number, value):
    return varint(number << 3) + varint(value)


def blob(number, value):
    return varint(number << 3 | 2) + varint(len(value)) + value


class AccountingTests(unittest.TestCase):
    def test_only_proven_self_worker_retries_explain_extra_attempts(self):
        before, after = self.snapshots()
        before['canister'] = after['canister'] = '4caro-hl777-77775-aaaba-cai'
        after.update(height=3, local=6)
        target = principal_hex(before['canister'])
        request = dict(receiver=target, sender=target, callback=7, method='runPaidInferenceWorker',
                       payload=(b'DIDL\x00\x02\x78\x78'+struct.pack('<QQ',9,0)).hex())
        retry = dict(before, height=2, canister_version=11, local=2, instructions=150, aborted_request=request)
        workers = [dict(stage=1,instructions=100), dict(stage=2,instructions=200)]
        measured = account(before, after, workers, job_id=9, retries=[retry, retry])
        self.assertEqual(measured['aborted_worker_attempts'], 1)
        for change in [dict(sender='00'), dict(method='other'),
                       dict(payload=(b'DIDL\x00\x02\x78\x78'+struct.pack('<QQ',10,0)).hex()),
                       dict(payload=(b'DIDL\x00\x02\x78\x78'+struct.pack('<QQ',9,99)).hex())]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                account(before, after, workers, job_id=9, retries=[dict(retry,aborted_request=dict(request,**change))])
        with self.assertRaises(ValueError):
            account(before, after, workers)

    def test_decoder_retains_original_aborted_worker_request(self):
        target = bytes.fromhex(principal_hex('4caro-hl777-77775-aaaba-cai'))
        canister = blob(1,blob(1,target))
        payload = b'DIDL\x00\x02\x78\x78'+struct.pack('<QQ',9,0)
        request = blob(1,canister)+blob(2,canister)+integer(3,7)+blob(5,b'runPaidInferenceWorker')+blob(6,payload)
        task = blob(54,blob(1,blob(1,blob(1,request))))
        base = blob(7,blob(6,bytes.fromhex('a'*64)))+integer(63,100)
        decoded = decode_snapshot(base+task,1,'target','a'*64)['aborted_request']
        self.assertEqual(decoded['sender'],target.hex())
        self.assertEqual(decoded['payload'],payload.hex())
        with self.assertRaises(ValueError):
            decode_snapshot(base+blob(54,blob(1,blob(2,b''))),1,'target','a'*64)

    def test_completed_checkpoint_waits_for_all_workers_and_callbacks(self):
        meter = CheckpointMeter('.', '4caro-hl777-77775-aaaba-cai', 'a'*64)
        before = dict(height=1, canister_version=10)
        partial = dict(height=2, canister_version=12)
        complete = dict(height=3, canister_version=13)
        with tempfile.TemporaryDirectory() as tmp, patch.object(meter, 'latest', side_effect=[(b'partial', partial), (b'complete', complete)]), patch('paid_instruction_accounting.time.sleep'):
            destination = Path(tmp)/'after.pbuf'
            self.assertEqual(meter.fresh(destination, after=before, worker_count=1), complete)
            self.assertEqual(destination.read_bytes(), b'complete')
        with patch.object(meter, 'latest', return_value=(b'unrelated', dict(height=3, canister_version=14))):
            with self.assertRaisesRegex(ValueError, 'Unexpected canister mutation'):
                meter.fresh('unused.pbuf', after=before, worker_count=1)

    def snapshots(self, instructions=315):
        before = dict(canister='target', module='a'*64, canister_version=10,
                      height=1, ingress=0, remote=0, local=0, http=0, timers=0, instructions=100)
        after = dict(before, canister_version=15, height=2, local=5, instructions=100+instructions)
        return before, after

    def test_total_and_conservative_message_bound_include_all_omitted_work(self):
        before, after = self.snapshots()
        result = account(before, after, [dict(instructions=100), dict(instructions=200)])
        self.assertEqual(result['total_instructions'], 315)
        self.assertEqual(result['unmeasured_instructions'], 15)
        self.assertEqual(result['max_message_instructions_upper_bound'], 215)
        self.assertTrue(result['message_budget_passed'])

    def test_passing_checkpoints_do_not_imply_passing_whole_message_budget(self):
        before, after = self.snapshots(40_000_000_101)
        result = account(before, after, [dict(instructions=100), dict(instructions=200)])
        self.assertFalse(result['message_budget_passed'])

    def test_scheduler_threshold_is_not_the_message_limit(self):
        before, after = self.snapshots(31_400_000_100)
        result = account(before, after, [dict(instructions=100), dict(instructions=31_300_000_000)])
        self.assertTrue(result['message_budget_passed'])
        before, after = self.snapshots(40_000_000_100)
        result = account(before, after, [dict(instructions=100), dict(instructions=200)])
        self.assertEqual(result['max_message_instructions_upper_bound'], 40_000_000_000)
        self.assertTrue(result['message_budget_passed'])

    def test_rejects_stale_unrelated_execution_reset_and_changed_module(self):
        before, after = self.snapshots()
        for edit in [dict(height=1), dict(local=6), dict(ingress=1), dict(timers=1),
                     dict(instructions=99), dict(module='b'*64), dict(canister_version=16)]:
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                account(before, dict(after, **edit), [dict(instructions=100), dict(instructions=200)])
        with self.assertRaises(ValueError):
            account(before, after, [dict(instructions=200), dict(instructions=200)])

    def test_schema_decoder_checks_module_and_retains_u64_precision(self):
        module = bytes.fromhex('a'*64)
        raw = blob(7, blob(6, module)) + integer(34, 9) + integer(63, (1<<63)+17)
        decoded = decode_snapshot(raw, 1, 'target', module.hex())
        self.assertEqual(decoded['instructions'], (1<<63)+17)
        with self.assertRaises(ValueError):
            decode_snapshot(raw, 1, 'target', 'b'*64)
        with self.assertRaises(ValueError):
            decode_snapshot(raw+integer(63,1), 1, 'target', module.hex())
        for bad in (b'\x80', b'\x0a\x02\x00', b'\x00'):
            with self.assertRaises(ValueError):
                protobuf_fields(bad)

    def test_principal_identity_is_checked(self):
        self.assertEqual(principal_hex('4caro-hl777-77775-aaaba-cai'), '7fffffffffa000020101')
        with self.assertRaises(ValueError):
            principal_hex('5caro-hl777-77775-aaaba-cai')


if __name__ == '__main__':
    unittest.main()
