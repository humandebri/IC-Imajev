#!/usr/bin/env python3
"""Check compact tail export, stale-file removal, final KV and replay."""
import pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));sys.path.insert(0,str(ROOT/'scripts'))
from full_inference import TextGraph
from test_terminal_decision_transport import TerminalDecisionTests

class TailDecisionReplayTests(TerminalDecisionTests):
 def run_terminal(self,t):
  return t.run('terminal_tail_integer',np.zeros(5120,np.float32),[1,0],tensor='model.language_model.layers.30.post_attention_layernorm.weight')

class TailGraphTests(unittest.TestCase):
 def test_supplied_terminal_attention_still_runs_mlp(self):
  with tempfile.TemporaryDirectory()as directory:
   class FakeTransport:
    def __init__(self):self.directory=pathlib.Path(directory);self.measurements=[];self.calls=[]
    def run(self,op,values,dims=(),**kwargs):
     self.calls.append(op);self.measurements.append(dict(ok=dict(instructions=1,request_bytes=1,reply_bytes=1)))
     if op!='terminal_mlp_integer':raise AssertionError(op)
     np.testing.assert_array_equal(values[:2560],np.ones(2560,np.float32))
     np.testing.assert_array_equal(values[2560:],np.full(2560,2.,np.float32))
     return np.concatenate([np.full(2560,3.,np.float32),np.full(2560,4.,np.float32)])
   t=FakeTransport();g=TextGraph.__new__(TextGraph);g.t=t;g.layers=[];g.fuse_terminal_tail=False;g.fuse_terminal_attention=True;g.terminal_readout=True;g.fuse_add_norm=True;g.arithmetic='int8';g.recorded_hidden=lambda x,l:x
   g.norm=lambda *args:(_ for _ in ()).throw(AssertionError('redundant norm'))
   g.terminal_attention_mlp=lambda *args:(_ for _ in ()).throw(AssertionError('repeated attention'))
   out=g.forward([1],initial_hidden=np.ones((1,2560),np.float32),start_layer=31,first_attention=np.full((1,2560),2.,np.float32))
   self.assertEqual(t.calls,['terminal_mlp_integer']);self.assertTrue(np.all(out==4.))
 def test_final_two_layers_keep_distinct_hidden_and_kv(self):
  with tempfile.TemporaryDirectory()as directory:
   class FakeTransport:
    def __init__(self):self.directory=pathlib.Path(directory);self.measurements=[];self.calls=[]
    def run(self,op,values,dims=(),scalars=(),tensor='',**kwargs):
     self.calls.append(op);self.measurements.append(dict(ok=dict(instructions=10,request_bytes=20,reply_bytes=30)))
     if op=='embed':return np.zeros(len(values)*2560,np.float32)
     if op=='terminal_tail_integer':
      n,p,compact=dims;self.assert_compact=compact;return np.concatenate([np.full(2560,31.,np.float32),np.full(2560,32.,np.float32),np.full(n*2048,42.,np.float32)])
     raise AssertionError(op)
   # A previous diagnostic run may have exported layer30 in this directory.
   np.save(pathlib.Path(directory)/'layer-30.npy',np.full((89,2560),-999.,np.float32))
   t=FakeTransport();g=TextGraph.__new__(TextGraph);g.t=t;g.layers=[];g.position_offset=0;g.fuse_terminal_tail=True;g.fuse_terminal_attention=True;g.terminal_readout=True;g.fuse_mlp_full=True;g.mlp_full_token_cap=89;g.fuse_add_norm=True;g.arithmetic='int8';g.recorded_hidden=lambda x,l:x;g.norm=lambda x,n:x;g.delta=lambda x,n,l:np.zeros_like(x);g.attention=g.delta
   def mlp(hidden,attention,prefix,layer):return np.zeros_like(hidden),np.zeros_like(hidden)
   g.full_mlp=mlp;g.attention_history=lambda k,v,layer:(k,v)
   out=g.forward(list(range(89)))
   self.assertEqual(t.calls,['embed','terminal_tail_integer'])
   self.assertFalse((t.directory/'layer-30.npy').exists());self.assertFalse(g.layers[-2]['hidden_exported']);self.assertEqual(t.assert_compact,0)
   self.assertTrue(np.all(np.load(t.directory/'layer-31.npy')==31.));self.assertTrue(np.all(out==32.))
   with np.load(t.directory/'states/layer-31.npz')as state:self.assertTrue(np.all(state['keys']==42.));self.assertTrue(np.all(state['values']==42.));self.assertEqual(state['keys'].shape,(89,4,256))
   self.assertEqual(g.layers[-1]['queries'],0);self.assertEqual(g.layers[-1]['included_in_layer'],30)

if __name__=='__main__':unittest.main()
