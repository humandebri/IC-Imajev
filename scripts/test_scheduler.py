#!/usr/bin/env python3
"""Exercise failed-query recovery without a replica accepting over-budget work."""
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport
class Fake:
    project=Transport.project
    def __init__(self):self.index=0;self.failures=[];self.completed=[]
    def run(self,op,values,dims,scalars,tensor,aux):
        n,width,cols,row=dims
        if width>3:
            self.failures.append((self.index,n,row,width));raise RuntimeError('Canister exceeded the limit of 5000000000 instructions for single message execution. IC0522')
        self.completed.append((self.index,n,row,width));self.index+=1
        return np.repeat(np.asarray(values)[:,0,None],width,axis=1).ravel()+np.tile(np.arange(row,row+width),n)
class SchedulerTests(unittest.TestCase):
    def test_failure_does_not_advance_and_tiles_cover_all_values(self):
        client=Fake();x=np.arange(35,dtype=np.float32).reshape(7,5)
        y=client.project(x,9,5,2,'base','a','b',row_cap=8,token_cap=4)
        np.testing.assert_array_equal(y,x[:,0,None]+np.arange(9)[None,:])
        self.assertEqual(client.failures[0],(0,4,0,8))
        self.assertEqual(client.completed[0],(0,4,0,2))
        self.assertEqual(client.index,len(client.completed))
    def test_only_instruction_failures_are_split(self):
        client=Fake()
        client.run=lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError('model mismatch'))
        with self.assertRaisesRegex(RuntimeError,'model mismatch'):client.project(np.ones((3,5)),9,5,2,'base','a','b')
class IntegerFake:
    project_integer=Transport.project_integer
    wire_codec='bf16-exact'
    max_floats=900000
    def __init__(self):self.completed=[]
    def run(self,op,values,dims,scalars,tensor,aux):
        n,width,cols,row=dims
        self.completed.append((n,width,cols,row))
        return (np.asarray(values)[:,0,None]+np.arange(row,row+width,dtype=np.float32)[None,:]).ravel()
class IntegerSchedulerTests(unittest.TestCase):
    def test_blob_split_avoids_padded_integer_rows_and_keeps_all_tokens(self):
        client=IntegerFake();x=np.repeat(np.arange(132,dtype=np.float32)[:,None],9216,axis=1)
        y=client.project_integer(x,16,9216,0,'base',row_cap=16,token_cap=512)
        self.assertEqual([q[0] for q in client.completed],[96,36])
        np.testing.assert_array_equal(y,x[:,0,None]+np.arange(16,dtype=np.float32)[None,:])
    def test_small_cap_preserves_tail_and_nonzero_output_row(self):
        client=IntegerFake();x=np.repeat(np.arange(11,dtype=np.float32)[:,None],256,axis=1)
        y=client.project_integer(x,24,256,0,'base',row_cap=8,token_cap=5)
        self.assertEqual([q[0] for q in client.completed],[5,5,5,5,5,5,1,1,1])
        np.testing.assert_array_equal(y,x[:,0,None]+np.arange(24,dtype=np.float32)[None,:])
class MLPSchedulerTests(unittest.TestCase):
    def test_fused_projection_covers_each_row_and_token(self):
        client=IntegerFake();client.project_mlp_integer=Transport.project_mlp_integer.__get__(client,IntegerFake)
        x=np.repeat(np.arange(132,dtype=np.float32)[:,None],2560,axis=1)
        y=client.project_mlp_integer(x,9216,2560,64,'gate','up')
        self.assertEqual([q[1] for q in client.completed],[4096,4096,1024])
        np.testing.assert_array_equal(y,x[:,0,None]+np.arange(9216,dtype=np.float32)[None,:])
    def test_wide_fusion_respects_combined_work_budget(self):
        client=IntegerFake();client.project_mlp_integer=Transport.project_mlp_integer.__get__(client,IntegerFake)
        x=np.repeat(np.arange(132,dtype=np.float32)[:,None],2560,axis=1)
        y=client.project_mlp_integer(x,9216,2560,64,'gate','up',row_cap=8192)
        self.assertEqual([q[1] for q in client.completed],[5704,3512])
        self.assertTrue(all(2*n*(width*cols+64*(cols+width))<=4000000000 for n,width,cols,row in client.completed))
        np.testing.assert_array_equal(y,x[:,0,None]+np.arange(9216,dtype=np.float32)[None,:])
    def test_fused_instruction_recovery_does_not_skip_rows(self):
        client=IntegerFake();client.project_mlp_integer=Transport.project_mlp_integer.__get__(client,IntegerFake);base_run=client.run
        def run(op,values,dims,scalars,tensor,aux):
            if dims[1]>16:raise RuntimeError('Canister exceeded the limit of 5000000000 instructions for single message execution. IC0522')
            return base_run(op,values,dims,scalars,tensor,aux)
        client.run=run;x=np.ones((11,256),dtype=np.float32)
        y=client.project_mlp_integer(x,24,256,1,'gate','up',row_cap=24,token_cap=5)
        np.testing.assert_array_equal(y,x[:,0,None]+np.arange(24,dtype=np.float32)[None,:])
        self.assertTrue(all(q[1]<=16 for q in client.completed))
if __name__=='__main__':unittest.main()
