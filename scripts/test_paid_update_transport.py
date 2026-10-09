import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from paid_update_transport import PaidTransport, ROOT, evidence_path


class PaidTransportEvidenceTests(unittest.TestCase):
    def test_successful_update_outside_repository_is_recorded_once(self):
        with tempfile.TemporaryDirectory(prefix='paid proof ') as tmp:
            d=Path(tmp)
            helper=d/'helper';helper.touch()
            wire=PaidTransport(d/'calls','local-model',helper=helper,call_helper=helper)
            arg=wire.d/'args.bin';arg.write_bytes(b'DIDL')
            with patch.object(wire,'args',return_value=arg),patch.object(wire,'decode',return_value={'Ok':None}),patch('paid_update_transport.subprocess.check_output',return_value='00') as call:
                result=wire.call('configure_paid',{'base_fee':1,'fee_per_token':0,'reserve_cycles':0})
            self.assertEqual(call.call_count,1)
            self.assertEqual(result['result'],{'Ok':None})
            self.assertEqual(result['args_path'],str(arg))
            self.assertEqual(json.loads((wire.d/'0001-configure_paid.json').read_text()),result)
            self.assertTrue(Path(result['reply_path']).is_file())

    def test_repository_paths_remain_relative(self):
        self.assertEqual(evidence_path(ROOT/'artifacts/proof/args.bin'),'artifacts/proof/args.bin')

    def test_explicit_network_and_identity_are_forwarded_without_shell_splitting(self):
        with tempfile.TemporaryDirectory(prefix='paid proof ') as tmp:
            d=Path(tmp);helper=d/'helper';helper.touch()
            identity=d/'local identity.pem'
            wire=PaidTransport(d/'calls','local-model',helper=helper,call_helper=helper,
                               url='http://localhost:8100/',identity_pem=identity)
            arg=wire.d/'request file.bin';arg.write_bytes(b'DIDL')
            with patch.object(wire,'args',return_value=arg),patch.object(wire,'decode',return_value={'Ok':None}),patch('paid_update_transport.subprocess.check_output',return_value='00') as call:
                wire.call('configure_paid',{'base_fee':1,'fee_per_token':0,'reserve_cycles':0})
            self.assertEqual(call.call_args.args[0], [str(helper),'--url','http://localhost:8100/',
                             '--identity-pem',str(identity),'local-model','configurePaidInference',str(arg)])
            self.assertEqual(call.call_count,1)

    def test_paid_relay_response_outside_repository_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);helper=d/'helper';helper.touch()
            wire=PaidTransport(d/'calls','local-model',helper=helper,call_helper=helper)
            arg=wire.d/'args.bin';arg.write_bytes(b'DIDL')
            forward={'response':{'Ok':[0]},'refunded':12345}
            result={'Ok':{'paid_cycles':100}}
            with patch.object(wire,'args',return_value=arg),patch.object(wire,'decode',side_effect=[forward,result]),patch('paid_update_transport.subprocess.check_output',return_value='00') as call:
                row=wire.call('infer',{'request_id':'test'},relay='local-relay',cycles=12445)
            self.assertEqual(call.call_count,1)
            self.assertEqual(row['result'],result)
            self.assertEqual(row['forward'],forward)
            self.assertEqual(json.loads((wire.d/'0001-infer.json').read_text()),row)


if __name__=='__main__':unittest.main()
