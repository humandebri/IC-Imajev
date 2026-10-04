#!/usr/bin/env python3
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));sys.path.insert(0,str(ROOT/'scripts'))
from delta_mlp_start_codec import NAME,CONV,layout,encode_request,decode_reply
from mlp_stream_codec import NAME as MLP_NAME,encode_payload,C
from test_mlp_stream_codec import carry
class StartCodecTests(unittest.TestCase):
 def header(self,n=89,b=5120):return dict(version=3,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,op='delta_mlp_stream_prepare',encoding=NAME,tensor='model.language_model.layers.0.post_attention_layernorm.weight',aux=['model.language_model.layers.1.input_layernorm.weight'],dims=[n,b,45],scalars=[2.,1e-6])
 def test_exact_carry_and_maximum_input_fit(self):
  for n in [1,87,89]:
   h=self.header(n);hidden=np.zeros(n*C,np.float32);norm=hidden.copy();cv=np.zeros(CONV,np.float32);log=np.zeros(45*6176,np.float32)
   self.assertLess(len(encode_request(h,hidden,norm,cv,log)),2_000_000)
   state=carry(n,5120);inner=dict(h,op='mlp_stream_prepare',encoding=MLP_NAME,dims=[n,0,5120]);payload=b'\0'+encode_payload(inner,state)+(cv.view('<u4')>>16).astype('<u2').tobytes()
   self.assertEqual(decode_reply(h,payload).tobytes(),np.concatenate([state,cv]).tobytes())
 def test_ids_are_layer_zero_only_and_canonical(self):
  from delta_mlp_start_codec import encode_ids
  h=self.header();h['op']='delta_mlp_stream_start_ids'
  cv=np.zeros(CONV,np.float32);log=np.zeros(45*6176,np.float32)
  self.assertLess(len(encode_ids(h,[123]*89,cv,log)),1_000_000)
  for ids in [[True]*89,[16_777_217]*89,[0]*88]:
   with self.assertRaises(ValueError):encode_ids(h,ids,cv,log)
  h['tensor']=h['tensor'].replace('.0.','.1.');h['aux']=[h['aux'][0].replace('.1.','.2.')]
  with self.assertRaises(ValueError):layout(h)
 def test_scope_precision_direction_and_bounds(self):
  h=self.header(1)
  for d in [[],[True,5120,45],[1,5121,45],[1,5120,0],[90,5120,45]]:
   h['dims']=d
   with self.assertRaises(ValueError):layout(h)
  h=self.header(1);hidden=np.zeros(C,np.float32);hidden[0]=.1234567
  with self.assertRaises(ValueError):encode_request(h,hidden,np.zeros(C,np.float32),np.zeros(CONV,np.float32),np.zeros(45*6176,np.float32))
  with self.assertRaises(ValueError):decode_reply(h,b'\1')
if __name__=='__main__':unittest.main()
