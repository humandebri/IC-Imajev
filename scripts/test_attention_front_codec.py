#!/usr/bin/env python3
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from mlp_attention_finish_codec import NAME,FRONT,C,KV,next_mlp,encode_request,decode_reply
from mlp_stream_codec import encode_payload,length
class Tests(unittest.TestCase):
 def header(self,n,b=512):return dict(version=3,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=7,op=FRONT,encoding=NAME,tensor='model.language_model.layers.22.post_attention_layernorm.weight',aux=['model.language_model.layers.23.input_layernorm.weight'],dims=[n,45,1280,b],scalars=[2.,1e-6])
 def state(self,n,b):return np.concatenate([np.full(n*C,-0.,np.float32),np.ones(n*C,np.float32),np.full(n*10,.25,np.float32),np.full(n*128,.1234567,np.float32),np.full(n*b,127.,np.float32),np.full(n*(b//256),.0123,np.float32),np.full(n*64,.56789,np.float32)])
 def test_front_reply_preserves_typed_lanes(self):
  for n in [1,87,89]:
   h=self.header(n);v=self.state(n,512);bf=np.full(n*(C+KV),-0.,np.float32);raw=(bf.view('<u4')>>16).astype('<u2').tobytes();p=b'\2'+raw[:2*n*C]+encode_payload(next_mlp(h),v)+raw[2*n*C:];expected=np.concatenate([bf[:n*C],v,bf[n*C:]])
   self.assertEqual(decode_reply(h,p).tobytes(),expected.tobytes());self.assertLess(len(p)+16424,2_000_000)
   for bad in [p[:-1],p+b'\0',b'\0'+p[1:]]:
    with self.assertRaises(ValueError):decode_reply(h,bad)
   for b in [0,1,513,9216]:
    with self.assertRaises(ValueError):next_mlp(self.header(n,b))
 def test_continuation_front_progress_and_shape(self):
  from mlp_delta_stream_codec import NAME as PAIR,decode_reply as decode_pair
  for n in [1,87,89]:
   h=dict(self.header(n),op='delta_partial_mlp_front',encoding=PAIR,tensor='model.language_model.layers.23.post_attention_layernorm.weight',aux=['model.language_model.layers.24.input_layernorm.weight'],dims=[n,512,8704,12,45,6912]);inner=dict(h,op='mlp_stream_prepare',encoding='mlp-stream-exact-v1',tensor='model.language_model.layers.24.post_attention_layernorm.weight',aux=['model.language_model.layers.25.input_layernorm.weight'],dims=[n,0,6912]);v=self.state(n,6912);p=b'\11'+encode_payload(inner,v)+bytes(2*3*20*256)
   self.assertEqual(decode_pair(h,p)[:length(n,6912)].tobytes(),v.tobytes());self.assertLess(len(p)+16424,2_000_000)
   for bad in [p[:-1],p+b'\0',b'\5'+p[1:]]:
    with self.assertRaises(ValueError):decode_pair(h,bad)
 def test_compressed_prefix_frame_fits_and_option_is_explicit(self):
  from mlp_delta_stream_codec import NAME as PAIR,encode_continue_request
  for n in [1,87,89]:
   k=8;h=dict(self.header(n),op='delta_partial_mlp_front',encoding=PAIR,tensor='model.language_model.layers.23.post_attention_layernorm.weight',aux=['model.language_model.layers.24.input_layernorm.weight'],dims=[n,512,8704,k,45,5120]);v=np.concatenate([np.zeros(n*C,np.float32),np.ones(n*C,np.float32),np.ones(n*10,np.float32),np.zeros(n*(128+64+C+64),np.float32),np.zeros(3*k*256,np.float32)]);history=np.zeros(3*(32-k)*256,np.float32);log=np.zeros(45*((32-k)//2*128+(32-k)*128+(32-k)),np.float32)
   frame=encode_continue_request(h,v,history,log,compress_base=True,compress_prefix=True);self.assertLess(len(frame),2_000_000)
   import struct
   off=4+struct.unpack('<I',frame[:4])[0];self.assertEqual(frame[off],10)
   with self.assertRaises(ValueError):encode_continue_request(h,v,history,log,compress_prefix=True)
if __name__=='__main__':unittest.main()
