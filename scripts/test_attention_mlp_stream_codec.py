#!/usr/bin/env python3
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));sys.path.insert(0,str(ROOT/'scripts'))
from attention_mlp_stream_codec import NAME,C,KV,layout,encode_request,decode_reply
from test_mlp_stream_codec import carry
class Tests(unittest.TestCase):
 def h(self,n=87):return dict(version=3,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,encoding=NAME,op='mlp_complete_attention_kv',tensor='model.language_model.layers.2.post_attention_layernorm.weight',aux=['model.language_model.layers.3.input_layernorm.weight'],dims=[n,1024,8192,45],scalars=[2.,1e-6])
 def test_prepare_carry_and_next_frame_fit_and_preserve_bits(self):
  for n in [1,80,87,89]:
   h=self.h(n);self.assertLess(len(encode_request(h,carry(n,1024))),2_000_000)
   bf=n*(2*C+KV);v=np.concatenate([np.full(bf,-0.,np.float32),np.full(n*C,-127.,np.float32),np.full(n*C//256,.0123,np.float32)])
   raw=b'\1'+(v[:bf].view('<u4')>>16).astype('<u2').tobytes()+v[bf:bf+n*C].astype(np.int8).tobytes()+v[bf+n*C:].tobytes();self.assertEqual(decode_reply(h,raw).tobytes(),v.tobytes())
   h.update(op='attention_finish_mlp_front',tensor='model.language_model.layers.3.post_attention_layernorm.weight',aux=['model.language_model.layers.4.input_layernorm.weight'],dims=[n,4608,45]);self.assertLess(len(encode_request(h,v,np.zeros(45*KV,np.float32))),2_000_000)
 def test_invalid_bound_progress_and_precision(self):
  h=self.h(1)
  with self.assertRaises(ValueError):encode_request(h,carry(1,2048))
  for dims in [[],[True,1024,8192,45],[1,1024,8193,45],[90,1024,8192,45],[1,1024,8192,133]]:
   h['dims']=dims
   with self.assertRaises(ValueError):layout(h)
 def test_q4_frames_preserve_f32_a_and_existing_bf16_gate(self):
  for n in [1,80,87,89]:
   h=self.h(n);h['op']='mlp_complete_attention_kv_q4';prefix=np.zeros(45*KV,np.float32)
   self.assertLess(len(encode_request(h,carry(n,1024),prefix)),2_000_000)
   bf=n*(2*C+KV);v=np.full(bf,-0.,np.float32);q=np.full(n*C,-127,np.float32);sx=np.full(n*C//256,.0123,np.float32);g=np.full(n*1024,-0.,np.float32);ax=np.full(n*64,.01234567,np.float32)
   raw=b'\4'+(np.concatenate([v,g]).view('<u4')>>16).astype('<u2').tobytes()+q.astype(np.int8).tobytes()+sx.tobytes()+ax.tobytes();expected=np.concatenate([v,q,sx,g,ax]);self.assertEqual(decode_reply(h,raw).tobytes(),expected.tobytes())
   h.update(op='attention_finish_mlp_front_q4',tensor='model.language_model.layers.3.post_attention_layernorm.weight',aux=['model.language_model.layers.4.input_layernorm.weight'],dims=[n,5120,45]);self.assertLess(len(encode_request(h,expected,prefix)),2_000_000)
   bad=expected.copy();bad[-1]=np.nan
   with self.assertRaises(ValueError):encode_request(h,bad,prefix)
   bad=expected.copy();bad[bf+n*C+n*C//256]=.1234567
   with self.assertRaisesRegex(ValueError,'BF16'):encode_request(h,bad,prefix)
 def test_compact_q4_omits_only_norm_and_duplicate_kv(self):
  from mlp_stream_codec import encode_payload,NAME as MLP
  import json,struct
  for n in [1,80,87,89]:
   h=self.h(n);h.update(op='attention_finish_mlp_front_q4',tensor='model.language_model.layers.3.post_attention_layernorm.weight',aux=['model.language_model.layers.4.input_layernorm.weight'],dims=[n,6144,45])
   bf=n*(2*C+KV);v=np.concatenate([np.full(bf,-0.,np.float32),np.full(n*C,-127.,np.float32),np.full(n*C//256,.0123,np.float32),np.full(n*1024,-0.,np.float32),np.full(n*64,.01234567,np.float32)])
   prefix=np.zeros(45*KV,np.float32)
   def payload(frame):
    size=struct.unpack('<I',frame[:4])[0];return frame[4+size:-32]
   old=payload(encode_request(h,v,prefix));h['op']='attention_finish_mlp_front_q4_compact';new=payload(encode_request(h,v,prefix))
   self.assertEqual(new,b'\6'+old[1:1+2*n*C]+old[1+4*n*C:]);self.assertEqual(len(old)-len(new),2*n*C)
   inner=dict(h,encoding=MLP,op='mlp_stream_prepare',dims=[n,0,6144]);state=carry(n,6144);raw=b'\7'+encode_payload(inner,state)
   self.assertEqual(decode_reply(h,raw).tobytes(),state.tobytes())
   with self.assertRaises(ValueError):decode_reply(h,b'\3'+raw[1:])
if __name__=='__main__':unittest.main()
