#!/usr/bin/env python3
"""Wire boundaries of joins needed for the 50-query schedule."""
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));sys.path.insert(0,str(ROOT/'scripts'))
from test_mlp_stream_codec import carry
from mlp_stream_codec import NAME as STREAM_NAME,encode_payload
from mlp_delta_carry import HUFFMAN_NAME,decode_front_reply
from mlp_delta_stream_codec import NAME,decode_reply,reply_count
class Tests(unittest.TestCase):
 def h(self,n):return dict(version=3,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,op='mlp_finish_delta_log_mlp_front',encoding=HUFFMAN_NAME,tensor='model.language_model.layers.1.post_attention_layernorm.weight',aux=['model.language_model.layers.2.input_layernorm.weight'],dims=[n,45,768,1792],scalars=[2.,1e-6])
 def test_front_reply_keeps_original_hidden_and_unrounded_state(self):
  for n in [1,80,87,89]:
   h=self.h(n);inner=dict(h,encoding=STREAM_NAME,op='mlp_stream_prepare',tensor=h['tensor'].replace('.1.','.2.'),aux=['model.language_model.layers.3.input_layernorm.weight'],dims=[n,0,1792]);state=carry(n,1792);hidden=np.full(n*2560,-0.,np.float32);conv=np.zeros(24576,np.float32)
   raw=b'\2'+(hidden.view('<u4')>>16).astype('<u2').tobytes()+encode_payload(inner,state)+(conv.view('<u4')>>16).astype('<u2').tobytes();self.assertLess(len(raw)+16424,2_000_000);self.assertEqual(decode_front_reply(h,raw).tobytes(),np.concatenate([hidden,state,conv]).tobytes())
   with self.assertRaises(ValueError):decode_front_reply(h,b'\0'+raw[1:])
 def test_full_follow_reply_is_bf16_and_direction_checked(self):
  for n in [1,80,87,89]:
   h=self.h(n);h.update(encoding=NAME,op='delta_partial_mlp_full',tensor='model.language_model.layers.3.post_attention_layernorm.weight',aux=['model.language_model.layers.4.input_layernorm.weight'],dims=[n,5376,3840,24,45]);v=np.full(reply_count(h),-0.,np.float32);raw=b'\5'+(v.view('<u4')>>16).astype('<u2').tobytes();self.assertEqual(decode_reply(h,raw).tobytes(),v.tobytes());self.assertLess(len(raw)+16424,2_000_000)
   with self.assertRaises(ValueError):decode_reply(h,b'\3'+raw[1:])
   with self.assertRaises(ValueError):decode_reply(h,raw[:-1])
if __name__=='__main__':unittest.main()
