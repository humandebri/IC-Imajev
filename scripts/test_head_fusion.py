#!/usr/bin/env python3
"""Fused payload bounds and exact state/gate transport are separate from accuracy tests."""
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import encode,decode,int8_prefix
H=dict(version=1,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,op='delta_heads_bf16',tensor='',dims=[132,128,128,16],scalars=[],encoding='int8-block256-v1')
class FusionTests(unittest.TestCase):
 def test_large_input_keeps_f32_gate_and_state_tail(self):
  n,k,v,heads=H['dims'];prefix=heads*n*(2*k+v);count=heads*(n*(2*k+v+2)+k*v);x=np.zeros(count,dtype=np.float32);x[prefix:]=np.float32(.1234567)
  blob=encode(H,x);self.assertLess(len(blob),2000000);_,y=decode(blob);np.testing.assert_array_equal(y[prefix:].view(np.uint32),x[prefix:].view(np.uint32))
  self.assertEqual(int8_prefix(H,count),prefix)
  with self.assertRaises(ValueError):encode({**H,'op':'bf16'},x)
 def test_head_limit_and_wrong_payload_rejected(self):
  with self.assertRaises(ValueError):int8_prefix({**H,'dims':[132,128,128,17]},1)
  with self.assertRaises(ValueError):int8_prefix(H,500)
 def test_reply_tail_is_exact_state(self):
  n,k,v,heads=H['dims'];count=heads*(n*v+k*v);prefix=heads*n*v;x=np.ones(count,dtype=np.float32);x[prefix:]=np.float32(.00001234567)
  _,y=decode(encode(H,x));np.testing.assert_array_equal(x[prefix:].view(np.uint32),y[prefix:].view(np.uint32))
if __name__=='__main__':unittest.main()
