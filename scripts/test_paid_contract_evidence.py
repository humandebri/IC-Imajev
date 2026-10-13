import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from verify_optimized_paid_contract import validate_contract


class ContractEvidenceTests(unittest.TestCase):
    def fixture(self, root):
        failed = {'job_id':1,'refund':'Done','reason':'local canister stopped'}
        receipt = {'job_id':1,'refund':'Done','state':{'Failed':failed['reason']}}
        request = {'request_id':'fixture','request':{'token_ids':[1]*1024}}
        completed = {'state':{'Completed':{}},'refund':'NotRequired'}
        responses = {
            'quote': {'Ok':{'fee':100_000_000}}, 'active':{'state':'Running'},
            'failed':{'Err':{'Failed':failed}}, 'failed_receipt':receipt,
            'scope_owner':None, 'post_scope_owner':None, 'retry_owner':{'Err':'missing receipt'},
            'retry':{'Ok':'Done'}, 'completed':completed, 'post_completed':completed,
            'config_owner':{'enabled':True}, 'post_config_owner':{'enabled':True},
            'post_failed_receipt':receipt, 'replay':{'Err':{'Failed':failed}},
        }
        observations, files = {}, {}
        for name,result in responses.items():
            forward = {'balance_before':1000000000,'balance_after':999999999,'refunded':0}
            forward['refunded'] = 1_000_000 if name=='failed' else 100_000_000 if name=='replay' else 0
            row = dict(result=result,forward=forward,request=request if name in ('failed','replay') else request['request'] if name=='quote' else 'fixture',
                       attached_cycles=101_000_000 if name=='failed' else 100_000_000 if name=='replay' else 0,
                       relay=None if name.endswith('_owner') or name=='quote' else 'relay')
            file=name+'.json'; raw=json.dumps(row).encode(); (root/file).write_bytes(raw)
            observations[name]=file; files[file]=hashlib.sha256(raw).hexdigest()
        report=dict(complete=True,live_fault_refund_passed=True,upgrade_gate_passed=True,failures=[],
                    candidate_module='a'*64,url='http://localhost:8002',relay='relay',request_id='fixture',
                    completed_request_id='fixture',
                    observations=observations,files=files,module_observations=['a'*64]*4,
                    actions=[dict(action=a,returncode=1 if i==0 else 0,stderr='paid inference active' if i==0 else '')
                             for i,a in enumerate(('install','stop','start','install'))])
        path=root/'report.json'; path.write_text(json.dumps(report));return path,report

    def test_live_contract_evidence_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,report=self.fixture(Path(tmp))
            self.assertEqual(validate_contract(path,'a'*64),report)

    def test_changed_calls_modules_and_failed_actions_rejected(self):
        for kind in ('bytes','module','action','endpoint','scope'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); path,report=self.fixture(root)
                if kind=='bytes': (root/'failed.json').write_text('{}')
                if kind=='module': report['module_observations'][-1]='b'*64
                if kind=='action': report['actions'][-1]['returncode']=1
                if kind=='endpoint': report['url']='https://ic0.app'
                if kind=='scope': report['observations']['scope_owner']='active.json'
                path.write_text(json.dumps(report))
                with self.assertRaises(AssertionError): validate_contract(path,'a'*64)

    def test_optimized_python_cannot_skip_contract_checks(self):
        result=subprocess.run([sys.executable,'-O','-B','-c',
            'from pathlib import Path; from verify_optimized_paid_contract import validate_contract; validate_contract(Path("unused"),"a"*64)'],
            cwd=Path(__file__).parent,text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('requires Python assertions enabled',result.stderr)


if __name__=='__main__':
    unittest.main()
