import json,pathlib,sys,tempfile,types,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from limit_fallback import eligible,restart,standard_command
class Tests(unittest.TestCase):
 def test_only_limits_and_strip_all_scheduling_overrides(self):
  self.assertTrue(eligible(ValueError('pair continuation frame bounds')));self.assertTrue(eligible(RuntimeError('IC0522 exceeded')))
  for e in [ValueError('invalid dedicated decision result'),RuntimeError('timeout'),ValueError('checkpoint input mismatch')]:self.assertFalse(eligible(e))
  cmd=standard_command(['--tail-start','--join-start','--roll-start','--directory=x','--tail-heads28=8','--tail-front30','512','--cache','cache','--frame-checksum','host'],pathlib.Path('/tmp/child'))
  self.assertNotIn('--tail-start',cmd);self.assertNotIn('--tail-heads28=8',cmd);self.assertIn('--frame-checksum',cmd);self.assertEqual(cmd[-2:],['--directory','/tmp/child'])
 def test_valid_high_entropy_carry_hits_frame_limit(self):
  sys.path.insert(0,str(ROOT/'scripts'));import numpy as np
  from test_mlp_delta_stream_codec import reply
  from test_tail_stream_codec import header
  from mlp_delta_stream_codec import encode_continue_request,NAME
  n=87;k=6;C=2560;_,v=reply(n,k);rng=np.random.default_rng(4)
  def bf(size):return (rng.integers(0,0x7f00,size,dtype=np.uint32)<<16).view(np.float32)
  v[:n*C]=bf(n*C);v[n*C:2*n*C]=rng.integers(-127,128,n*C)
  prep=n*(C//256+128+64);v[2*n*C+prep:2*n*C+prep+n*C]=rng.normal(size=n*C).astype(np.float32)
  p=45;remaining=26;cv=bf(3*remaining*256)
  lv=np.concatenate([bf(p*remaining//2*128),rng.normal(size=p*remaining*128).astype(np.float32),rng.random(p*remaining).astype(np.float32)])
  h=header('delta_partial_mlp_front',NAME,n,27,[n,0,9216,k,p,5888])
  with self.assertRaisesRegex(ValueError,'pair continuation frame bounds')as error:encode_continue_request(h,v,cv,lv,compress_base=True,compress_prefix=True,compress_hidden=True,compress_input=True)
  self.assertTrue(eligible(error.exception))
 def test_restart_preserves_evidence_accounts_failure_and_resumes_child(self):
  for error,failed in [(ValueError('pair continuation frame bounds'),0),(RuntimeError('IC0522'),1)]:
   with tempfile.TemporaryDirectory()as temp:
    d=pathlib.Path(temp);session=dict(model='a',pack_hash='b',input_hash='c',wasm_sha256='d',prefix_identity='e');(d/'session.json').write_text(json.dumps(session));q=d/'queries';q.mkdir();(q/'000001.request.bin').write_bytes(b'failed-request')
    t=types.SimpleNamespace(directory=q,index=1,decision_options=['yes','no'],measurements=[dict(ok=dict(instructions=11,request_bytes=5,reply_bytes=7,heap_pages=2))])
    child_report=dict(**session,decision_query=dict(ok=dict(decision=dict(value='yes',abstained=False,probabilities=[.8,.1],unknown_probability=.1,raw_logits=[1.,0.,-1.],instructions=5,calibration_version='p3-r2-s000291-authored'))),queries=[dict(decision_options=['yes','no'])],query_count=62,total_instructions=100,total_candid_bytes=200,executed_query_count=62,executed_instructions=100,executed_candid_bytes=200,max_query_instructions=9,max_observed_heap_bytes=65536)
    calls=[]
    def run(cmd,**kwargs):
     calls.append(cmd);child=d/'standard-fallback';child.mkdir(exist_ok=True);(child/'report.json').write_text(json.dumps(child_report))
    with patch('limit_fallback.subprocess.run',side_effect=run):
     r=restart(d,t,['--tail-start','--directory',temp],error)
     self.assertEqual(r['query_count'],63+failed);self.assertEqual(r['successful_query_instructions'],111);self.assertEqual(r['successful_query_candid_bytes'],212)
     self.assertEqual(r['total_instructions'],None if failed else 111);self.assertEqual(r['unmeasured_failed_queries'],failed)
     self.assertEqual((q/'000001.request.bin').read_bytes(),b'failed-request')
     child_report.update(executed_query_count=0,executed_instructions=0,executed_candid_bytes=0)
     r=restart(d,t,['--tail-start','--directory',temp]);self.assertEqual(r['executed_query_count'],0)
     self.assertEqual(len(calls),2)
     child_report['input_hash']='wrong'
     with self.assertRaisesRegex(ValueError,'result identity'):restart(d,t,['--tail-start','--directory',temp])
     child_report['input_hash']=session['input_hash']
     child_report['queries']=[dict(decision_options=['no','yes'])]
     with self.assertRaisesRegex(ValueError,'options'):restart(d,t,['--tail-start','--directory',temp])
     child_report['queries']=[dict(decision_options=['yes','no'])]
     failure_path=d/'standard-fallback/queries/failures.jsonl';failure_path.parent.mkdir(exist_ok=True)
     def run_with_failure(cmd,**kwargs):
      run(cmd,**kwargs);failure_path.write_text('{"index":3,"error":"IC0522"}\n')
     with patch('limit_fallback.subprocess.run',side_effect=run_with_failure):r=restart(d,t,['--tail-start','--directory',temp])
     self.assertEqual(r['unmeasured_failed_queries'],failed+1);self.assertEqual(r['executed_query_count'],1);self.assertIsNone(r['total_instructions']);self.assertIsNone(r['max_query_instructions'])
     (d/'session.json').write_text('{"changed":true}')
     with self.assertRaisesRegex(ValueError,'identity'):restart(d,t,['--tail-start','--directory',temp])
if __name__=='__main__':unittest.main()
