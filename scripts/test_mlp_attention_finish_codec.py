#!/usr/bin/env python3
import pathlib,sys,struct,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from mlp_attention_finish_codec import NAME,C,H,KV,encode_request,decode_reply
from mlp_codec import decode_payload,NAME as MLP_NAME
class Tests(unittest.TestCase):
 def h(self,n):return dict(version=3,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,op='mlp_finish_attention_full',encoding=NAME,tensor='model.language_model.layers.6.post_attention_layernorm.weight',aux=['model.language_model.layers.7.input_layernorm.weight'],dims=[n,45,1280],scalars=[2.,1e-6])
 def carry(self,n):return np.concatenate([np.full(n*C,-0.,np.float32),np.full(n*H,127.,np.float32),np.full(n*36,.0123,np.float32),np.full(n*64,.1234567,np.float32)])
 def test_inner_carry_and_outer_reply_preserve_bits(self):
  for n in [1,7,87,89]:
   h=self.h(n);v=self.carry(n);kv=np.zeros(45*KV,np.float32);frame=encode_request(h,v,kv);offset=4+struct.unpack('<I',frame[:4])[0];payload=frame[offset:-32];end=2+n*(2*C+H+400);inner=dict(h,encoding=MLP_NAME,op='mlp_down_norm_partial_prepared',dims=[n,C,1280]);self.assertEqual(decode_payload(inner,payload[1:end]).tobytes(),v.tobytes());self.assertEqual(payload[end:],b'\0'*(2*kv.size));self.assertLess(len(frame),2_000_000)
   y=np.full(n*(3*C+KV),-0.,np.float32);raw=b'\0'+(y.view('<u4')>>16).astype('<u2').tobytes();self.assertEqual(decode_reply(h,raw).tobytes(),y.tobytes());self.assertLess(len(raw)+16424,2_000_000)
   for bad in [raw[:-1],b'\1'+raw[1:],raw+b'\0']:
    with self.assertRaises(ValueError):decode_reply(h,bad)
 def test_compact_reply_keeps_legacy_separate(self):
  for n in [1,87,89]:
   h=dict(self.h(n),op='mlp_finish_attention_full_compact');v=np.full(n*(2*C+KV),-0.,np.float32);raw=b'\0'+(v.view('<u4')>>16).astype('<u2').tobytes()
   self.assertEqual(decode_reply(h,raw).tobytes(),v.tobytes())
   with self.assertRaises(ValueError):decode_reply(self.h(n),raw)
   encode_request(h,self.carry(n),np.zeros(45*KV,np.float32))
 def test_delta_finish_only_reply_tag_and_shape(self):
  from mlp_delta_stream_codec import NAME as PAIR,reply_count,decode_reply as pair_decode
  for n in [1,87,89]:
   h=dict(self.h(n),encoding=PAIR,op='delta_partial_finish',dims=[n,4352,4864,20,45],tensor='model.language_model.layers.29.post_attention_layernorm.weight',aux=['model.language_model.layers.30.input_layernorm.weight']);v=np.full(reply_count(h),-0.,np.float32);raw=b'\10'+(v.view('<u4')>>16).astype('<u2').tobytes()
   self.assertEqual(pair_decode(h,raw).tobytes(),v.tobytes())
   with self.assertRaises(ValueError):pair_decode(dict(h,op='delta_partial_mlp_full'),raw)
 def test_metadata_nonfinite_and_precision_rejected(self):
  h=self.h(1);v=self.carry(1);kv=np.zeros(45*KV,np.float32)
  for dims in [[0,45,1280],[90,45,1280],[1,133,1280],[1,45,31],[1,45,C],[True,45,1280]]:
   with self.assertRaises(ValueError):encode_request(dict(h,dims=dims),v,kv)
  for position in [0,C,C+H,C+H+36]:
   for value in [np.nan,np.inf]:
    bad=v.copy();bad[position]=value
    with self.assertRaises(ValueError):encode_request(h,bad,kv)
  for value in [.1234567,np.nan,np.inf]:
   bad=kv.copy();bad[0]=value
   with self.assertRaises(ValueError):encode_request(h,v,bad)
  for layer in [3,30,'06']:
   with self.assertRaises(ValueError):encode_request(dict(h,tensor=f'model.language_model.layers.{layer}.post_attention_layernorm.weight'),v,kv)
if __name__=='__main__':unittest.main()
