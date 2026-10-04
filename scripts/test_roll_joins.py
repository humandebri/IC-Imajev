#!/usr/bin/env python3
"""Wire boundaries of joins needed for the 50-query schedule."""
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));sys.path.insert(0,str(ROOT/'scripts'))
from test_mlp_stream_codec import carry
from mlp_stream_codec import NAME as STREAM_NAME,encode_payload
from mlp_delta_carry import HUFFMAN_NAME,decode_front_reply
from mlp_delta_stream_codec import NAME,decode_reply,reply_count,encode_request
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
 def test_residual_compression_keeps_unrounded_stream_tail(self):
  import struct
  for n in [1,80,87,89]:
   h=self.h(n);h.update(encoding=NAME,op='mlp_complete_delta_partial',tensor='model.language_model.layers.3.post_attention_layernorm.weight',aux=['model.language_model.layers.4.input_layernorm.weight'],dims=[n,6144,3072,26,45])
   state=carry(n,6144);cv=np.zeros(3*26*256,np.float32);log=np.zeros(45*(26//2*128+26*128+26),np.float32)
   frame=encode_request(h,state,cv,log,compress_residual=True);self.assertLess(len(frame),2_000_000);size=struct.unpack('<I',frame[:4])[0];p=frame[4+size:-32];self.assertEqual(p[0],6)
   length=struct.unpack('<I',p[1:5])[0];inner=p[5:5+length];planes=struct.unpack('<I',inner[5:9])[0]
   self.assertEqual(inner[9:9+planes],b'\2\1\0\0\0\0\2\1\0\0\0\x80')
   original=encode_payload(dict(h,encoding=STREAM_NAME,op='mlp_stream_complete',dims=[n,6144,3072]),state)
   self.assertEqual(inner[:5],original[:5]);self.assertEqual(inner[9+planes:],original[5+2*n*2560:])
if __name__=='__main__':unittest.main()
