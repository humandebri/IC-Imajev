import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from check_paid_instruction_evidence import validate
from paid_instruction_accounting import decode_snapshot, account
from test_paid_instruction_accounting import integer, blob


class EvidenceTests(unittest.TestCase):
    def fixture(self, root):
        summary = dict(complete=True, paid_numerical_gate_passed=True, paid_instruction_gate_passed=True,
                       whole_message_accounting_complete=True, failures=[], run_id='fixture',
                       reference_module='a'*64, candidate_module='b'*64, inputs=[])
        for n in (9,96,128,256,512,1024):
            row = dict(total=n, request_id=f'source-migration-fixture-{n}', whole_message_accounting={})
            for label, instructions in [('reference',1000),('candidate',1050)]:
                module = summary[label+'_module']
                snapshots = []
                for point, version, local, counter, height in [('before',10,0,10000,1),('after',13,3,10000+instructions,2)]:
                    raw = blob(7,blob(6,bytes.fromhex(module)))+integer(34,version)+integer(60,local)+integer(63,counter)
                    (root/f'n{n}-{label}-{point}.pbuf').write_bytes(raw)
                    snapshots.append(decode_snapshot(raw,height,label,module))
                result = dict(workers=[dict(instructions=900)], decision=dict(value='yes', abstained=False,
                              calibration_version='fixture',raw_logits=[1.0],probabilities=[1.0],unknown_probability=0.0))
                wire = dict(kind='infer',request=dict(request_id=row['request_id'],request=dict(token_ids=[1]*n,options=['yes'])),result={'Ok':result},request_bytes=100,reply_bytes=100)
                raw = json.dumps(wire).encode(); name=f'n{n}-{label}-result.json'; (root/name).write_bytes(raw)
                recorded = account(*snapshots,result['workers'])
                recorded['result_evidence'] = dict(file=name,sha256=hashlib.sha256(raw).hexdigest())
                row['whole_message_accounting'][label] = recorded
            summary['inputs'].append(row)
        path=root/'report.json'; path.write_text(json.dumps(summary)); return path,summary

    def test_complete_original_evidence_passes_at_105_percent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,summary=self.fixture(Path(tmp))
            self.assertEqual(validate(path),summary)

    def test_changed_original_result_and_checkpoint_are_rejected(self):
        for name in ('n9-candidate-result.json','n1024-reference-before.pbuf'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); path,_=self.fixture(root)
                (root/name).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    validate(path)

    def test_original_wire_sizes_must_meet_communication_budget(self):
        for field in ('request_bytes', 'reply_bytes'):
            for size in (None, False, 0, -1, 1.5, 1_990_000, 2_000_000):
                with self.subTest(field=field, size=size), tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp); path,summary=self.fixture(root)
                    evidence=summary['inputs'][0]['whole_message_accounting']['candidate']['result_evidence']
                    original=root/evidence['file']; wire=json.loads(original.read_text())
                    wire[field]=size
                    original.write_text(json.dumps(wire))
                    evidence['sha256']=hashlib.sha256(original.read_bytes()).hexdigest()
                    path.write_text(json.dumps(summary))
                    with self.assertRaisesRegex(ValueError, 'communication budget'):
                        validate(path)

    def test_stale_counts_coverage_gate_and_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,summary=self.fixture(Path(tmp))
            original=json.dumps(summary)
            for edit in [lambda s:s.update(whole_message_accounting_complete=False),
                         lambda s:s['inputs'].pop(),
                         lambda s:s['inputs'][0]['whole_message_accounting']['candidate'].update(total_instructions=1),
                         lambda s:s['inputs'][0]['whole_message_accounting']['candidate']['result_evidence'].update(file='../outside')]:
                changed=json.loads(original); edit(changed);path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):validate(path)


if __name__ == '__main__':
    unittest.main()
