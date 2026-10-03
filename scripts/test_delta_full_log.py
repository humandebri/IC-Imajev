#!/usr/bin/env python3
import pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import TextGraph,delta_log_prefix_fits,delta_full_log_enabled
from transport import encode,decode
class FullDeltaTests(unittest.TestCase):
 def graph(self,t,keep=False):return TextGraph(t,dict(tensors=[]),arithmetic='int8',fuse_delta=True,fuse_delta_projected=True,fuse_delta_full_log=True,compact_heads=True,retain_terminal_state=keep)
 def test_wire_history_log_and_one_query_route(self):
  for n,keep,p in [(45,True,0),(87,False,45),(89,False,45)]:
   with self.subTest(n=n),tempfile.TemporaryDirectory()as directory:
    class Fake:
     wire_codec='bf16-block256-exact-v1';max_floats=900000
     def __init__(self):self.directory=pathlib.Path(directory);self.calls=[]
     def run(self,op,x,dims,tensor):
      self.calls.append(op);self_outer.assertEqual(dims,[n,32,p,int(keep)])
      h,y=decode(encode(dict(op=op,dims=dims,tensor=tensor,encoding=self.wire_codec),x));np.testing.assert_array_equal(y.view(np.uint32),x.view(np.uint32))
      self_outer.assertEqual(x.size,n*2560+24576+p*6176)
      return np.concatenate([np.full(n*2560,3.,np.float32),np.full(24576,2.,np.float32),np.full(n*6176,0.5,np.float32) if keep else np.empty(0,np.float32)])
    self_outer=self;t=Fake();g=self.graph(t,keep);g.initial_delta_log=lambda:np.full(p*6176,0.5,np.float32)
    g.initial_delta=lambda *_:self.fail('F32 state path must not run');out=g.delta(np.ones((n,2560),np.float32),'model.linear_attn',0)
    np.testing.assert_array_equal(out,3.);self.assertEqual(t.calls,['delta_full_log_integer'])
    with np.load(t.directory/'states/layer-00.npz') as f:
     self.assertEqual(set(f.files),{'conv','delta_log'} if keep else {'conv'});np.testing.assert_array_equal(f['conv'],2.)
     if keep:self.assertEqual(f['delta_log'].shape,(n*6176,))
 def test_cold_fallback_and_logged_prefix_bound(self):
  class Fake:wire_codec='bf16-block256-exact-v1'
  g=self.graph(Fake());g.delta_projected=lambda *_:'cold-fallback';self.assertEqual(g.delta(np.zeros((132,2560),np.float32),'model.linear_attn',0),'cold-fallback')
  g.initial_delta_log=lambda:np.zeros(45*6176,np.float32)
  with self.assertRaisesRegex(ValueError,'at most 90'):g.delta(np.zeros((91,2560),np.float32),'model.linear_attn',0)
 def test_flag_requires_projected_path(self):
  with self.assertRaisesRegex(ValueError,'requires projected Delta'):TextGraph(object(),dict(tensors=[]),fuse_delta_full_log=True)
 def test_prefix_reply_bound_selects_all_layers_before_queries(self):
  for codec,limit in [('bf16-exact',72),('bf16-block256-exact-v1',75)]:
   class Fake:wire_codec=codec
   for n in [1,limit,limit+1,90,91,132]:
    with self.subTest(codec=codec,n=n):
     g=self.graph(Fake(),keep=True)
     calls=[]
     g.delta_full_log=lambda *args:calls.append('log')
     g.delta_projected=lambda *args:calls.append('dense')
     for layer in [0,22,30]:g.delta(np.zeros((n,2560),np.float32),'layer.linear_attn',layer)
     self.assertEqual(calls,['log' if n<=limit else 'dense']*3)
     self.assertEqual(delta_log_prefix_fits(n,codec),n<=limit)
     self.assertEqual(delta_full_log_enabled(True,n,codec,True),n<=limit)
   # Use worst-case non-BF16 innovations and decays to test the real encoder.
   n=limit;output=np.concatenate([np.ones(n*4608+24576,np.float32),np.full(n*4128,1.0000001,np.float32)])
   frame=encode(dict(op='delta_full_log_integer',dims=[n,32,0,1],encoding=codec),output)
   self.assertLessEqual(len(frame)+16384,2000000)
 def test_terminal_bound_is_independent_of_retained_log_reply(self):
  for codec in ['bf16-exact','bf16-block256-exact-v1']:
   self.assertTrue(delta_full_log_enabled(True,90,codec,False))
   self.assertFalse(delta_full_log_enabled(True,91,codec,False))
   self.assertFalse(delta_full_log_enabled(True,45,codec,False,1))
   self.assertTrue(delta_full_log_enabled(True,90,codec,False,2))
   with self.assertRaisesRegex(ValueError,'at most 90'):delta_full_log_enabled(True,91,codec,False,2)
   with self.assertRaisesRegex(ValueError,'representation mismatch'):delta_full_log_enabled(False,1,codec,False,2)
if __name__=='__main__':unittest.main()
