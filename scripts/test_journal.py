#!/usr/bin/env python3
"""Client resume tests independent of model arithmetic and replica availability."""
import json,pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import JournalTransport
from transport import decode,encode,atomic
class FakeJournal(JournalTransport):
    def __init__(self,path):
        self.directory=path;self.index=0;self.model='a'*64;self.pack_hash='b'*64;self.input_hash='c'*64;self.measurements=[];self.replayed=0;self.calls=0
    def command(self,cmd):
        self.calls+=1;h,v=decode(pathlib.Path(cmd['input']).read_bytes());h['step']+=1;atomic(cmd['output'],encode(h,v))
        return dict(ok=dict(instructions=7,request_bytes=100,reply_bytes=100,heap_pages=1,stable_pages=1,stable_read_bytes=0),wall_seconds=0.)
class LimitedProjectionJournal(FakeJournal):
    def __init__(self,path):
        super().__init__(path);self.wire_codec='bf16-exact';self.max_floats=900000
    def command(self,cmd):
        self.calls+=1;h,v=decode(pathlib.Path(cmd['input']).read_bytes());n,width,cols,row=h['dims']
        if width>3:raise RuntimeError('instruction limit')
        h['step']+=1;out=np.repeat(v.reshape(n,cols)[:,0,None],width,axis=1)+np.arange(row,row+width)[None,:]
        atomic(cmd['output'],encode(h,out))
        return dict(ok=dict(instructions=7,request_bytes=100,reply_bytes=100,heap_pages=1,stable_pages=1,stable_read_bytes=0),wall_seconds=0.)
class JournalTests(unittest.TestCase):
    def test_resume_reuses_only_verified_matching_reply(self):
        with tempfile.TemporaryDirectory(prefix='imajev-journal-') as name:
            path=pathlib.Path(name);first=FakeJournal(path);expected=first.run('bf16',[1.,2.]);self.assertEqual(first.calls,1)
            resumed=FakeJournal(path);np.testing.assert_array_equal(resumed.run('bf16',[1.,2.]),expected);self.assertEqual(resumed.calls,0);self.assertEqual(resumed.replayed,1)
            changed=FakeJournal(path)
            with self.assertRaisesRegex(ValueError,'input mismatch'):changed.run('bf16',[3.,4.])
            self.assertEqual(changed.calls,0)
    def test_corrupt_reply_is_rejected_before_reuse(self):
        with tempfile.TemporaryDirectory(prefix='imajev-journal-') as name:
            path=pathlib.Path(name);first=FakeJournal(path);first.run('bf16',[1.,2.]);response=path/'000000.response.bin';b=bytearray(response.read_bytes());b[-1]^=1;response.write_bytes(b)
            with self.assertRaisesRegex(ValueError,'checksum'):FakeJournal(path).run('bf16',[1.,2.])
    def test_reply_without_metric_is_safely_requeried(self):
        with tempfile.TemporaryDirectory(prefix='imajev-journal-') as name:
            path=pathlib.Path(name);first=FakeJournal(path);first.run('bf16',[1.,2.]);(path/'000000.metric.json').unlink();resumed=FakeJournal(path);resumed.run('bf16',[1.,2.]);self.assertEqual(resumed.calls,1)
    def test_missing_request_cannot_replay_different_values(self):
        with tempfile.TemporaryDirectory(prefix='imajev-journal-') as name:
            path=pathlib.Path(name);FakeJournal(path).run('bf16',[1.,2.]);(path/'000000.request.bin').unlink();resumed=FakeJournal(path)
            with self.assertRaisesRegex(ValueError,'no matching request'):resumed.run('bf16',[3.,4.])
            self.assertEqual(resumed.calls,0);self.assertEqual(resumed.index,0)
    def test_metric_from_another_query_cannot_be_replayed(self):
        for field,value in [('index',1),('op','embed'),('tensor','other')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory(prefix='imajev-journal-') as name:
                path=pathlib.Path(name);FakeJournal(path).run('bf16',[1.,2.]);metric=path/'000000.metric.json';saved=json.loads(metric.read_text());saved[field]=value;metric.write_text(json.dumps(saved));resumed=FakeJournal(path)
                with self.assertRaisesRegex(ValueError,'metric identity'):resumed.run('bf16',[1.,2.])
                self.assertEqual(resumed.calls,0);self.assertEqual(resumed.index,0)
    def test_projection_limit_retries_and_remembers_width_on_resume(self):
        with tempfile.TemporaryDirectory(prefix='imajev-journal-') as name:
            path=pathlib.Path(name);x=np.arange(35,dtype=np.float32).reshape(7,5)
            first=LimitedProjectionJournal(path);y=first.project(x,9,5,2,'base','a','b',row_cap=8,token_cap=4)
            np.testing.assert_array_equal(y,x[:,0,None]+np.arange(9)[None,:])
            self.assertTrue(list(path.glob('*.failed-width-*.request.bin')))
            resumed=LimitedProjectionJournal(path);np.testing.assert_array_equal(resumed.project(x,9,5,2,'base','a','b',row_cap=8,token_cap=4),y);self.assertEqual(resumed.calls,0)
            changed=LimitedProjectionJournal(path)
            with self.assertRaises(ValueError):changed.project(x+1,9,5,2,'base','a','b',row_cap=8,token_cap=4)
if __name__=='__main__':unittest.main()
