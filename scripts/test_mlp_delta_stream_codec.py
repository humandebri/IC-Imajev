#!/usr/bin/env python3
"""Fusion wire bounds, bit preservation and rejected progress/precision."""
import pathlib
import struct
import sys
import unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));sys.path.insert(0,str(ROOT/'scripts'))
from mlp_delta_stream_codec import NAME,encode_request,encode_continue_request,decode_reply,layout
from mlp_stream_codec import C,H,R
from test_mlp_stream_codec import carry

def header(n=89,k=24,p=45):
 return dict(version=3,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,encoding=NAME,op='mlp_complete_delta_partial',tensor='model.language_model.layers.0.post_attention_layernorm.weight',dims=[n,5120,4096,k,p],scalars=[2.,1e-6],aux=['model.language_model.layers.1.input_layernorm.weight'])

def reply(n,k):
 residual=np.full(n*C,-0.,np.float32);integer=np.full(n*C,-127,np.int8);rest=np.concatenate([np.full(n*C//256,.0123,np.float32),np.full(n*2*R,.1234567,np.float32),np.full(n*64,.875,np.float32),np.full(n*C,-.234567,np.float32),np.full(n*R,.345678,np.float32)]);cv=np.full(3*k*256,.5,np.float32)
 payload=b'\0'+(residual.view('<u4')>>16).astype('<u2').tobytes()+integer.tobytes()+rest.astype('<f4').tobytes()+(cv.view('<u4')>>16).astype('<u2').tobytes()
 return payload,np.concatenate([residual,integer.astype(np.float32),rest,cv])

class PairCodecTests(unittest.TestCase):
 def test_largest_normal_request_and_reply_are_bounded_lossless(self):
  for n in [1,7,87,89]:
   for k in [22,24]:
    h=header(n,k);p=45;cv=np.zeros(3*k*256,np.float32);log=np.zeros(p*(k//2*128+k*128+k),np.float32);packet=encode_request(h,carry(n,5120),cv,log)
    self.assertLess(len(packet),2_000_000);payload,expected=reply(n,k);self.assertEqual(decode_reply(h,payload).tobytes(),expected.tobytes());self.assertLess(len(payload)+16384+36,2_000_000)
 def test_follow_request_at_89_tokens_and_prepared_reply_are_exact(self):
  from mlp_codec import encode_payload,NAME as DOWN_NAME
  for n in [1,7,87,89]:
   for k in [22,24]:
    h=header(n,k);h['op']='delta_partial_mlp_prepare';payload,x=reply(n,k);remaining=32-k;cv=np.zeros(3*remaining*256,np.float32);log=np.zeros(45*(remaining//2*128+remaining*128+remaining),np.float32)
    packet=encode_continue_request(h,x,cv,log);self.assertLess(len(packet),2_000_000)
    if n==89 and k==22:self.assertGreater(len(packet),1_990_000)
    prepared=np.concatenate([np.full(n*C,-0.,np.float32),np.full(n*H,127.,np.float32),np.full(n*36,.009,np.float32),np.full(n*R,-.1234567,np.float32)])
    inner=dict(h,encoding=DOWN_NAME,op='mlp_prepare_down',dims=[n,C],tensor='model.language_model.layers.1.post_attention_layernorm.weight',aux=['model.language_model.layers.2.input_layernorm.weight'])
    wire=b'\3'+encode_payload(inner,prepared)+(cv.view('<u4')>>16).astype('<u2').tobytes();expected=np.concatenate([prepared,cv]);self.assertEqual(decode_reply(h,wire).tobytes(),expected.tobytes())
    with self.assertRaises(ValueError):decode_reply(h,payload)
 def test_partial_down_progress_is_bound_and_shape_unchanged(self):
  h=header(1,22);h['op']='delta_partial_mlp_prepare_down';h['dims'].append(1600)
  self.assertEqual(layout(h),(1,5120,4096,22,45))
  _,x=reply(1,22);cv=np.zeros(3*10*256,np.float32);log=np.zeros(45*(5*128+10*128+10),np.float32)
  self.assertLess(len(encode_continue_request(h,x,cv,log)),2_000_000)
  for rows in [0,31,2560,-32,True]:
   h['dims'][-1]=rows
   with self.assertRaises(ValueError):layout(h)
  h['dims'][-1]=1600;h['op']='delta_partial_mlp_prepare'
  with self.assertRaises(ValueError):layout(h)
 def test_incomplete_or_repeated_progress_rejected(self):
  h=header(1);cv=np.zeros(3*24*256,np.float32);log=np.zeros(45*(12*128+24*128+24),np.float32)
  for done in [4608,H]:
   with self.assertRaises(ValueError):encode_request(h,carry(1,done),cv,log)
  for dims in [[1,0,H,24,45],[1,5120,4096,23,45],[1,5120,4096,32,45],[True,5120,4096,24,45],[1,5120,4096,24,0]]:
   h['dims']=dims
   with self.assertRaises(ValueError):layout(h)
 def test_corrupt_reply_integer_scale_gate_and_finite_rejected(self):
  h=header(1);payload,_=reply(1,24)
  for offset,value in [(1+3*C,0.),(1+3*C,np.nan),(1+3*C+4*(C//256+2*R),1.1)]:
   bad=bytearray(payload);bad[offset:offset+4]=struct.pack('<f',value)
   with self.assertRaises(ValueError):decode_reply(h,bad)
  bad=bytearray(payload);bad[1+2*C]=128
  with self.assertRaises(ValueError):decode_reply(h,bad)
  for bad in [payload+b'\0',payload[:-1],b'\1'+payload[1:]]:
   with self.assertRaises(ValueError):decode_reply(h,bad)
 def test_large_prefix_and_non_bf16_history_fail_before_sending(self):
  h=header(p=132);k=24;p=132;cv=np.zeros(3*k*256,np.float32);log=np.zeros(p*(12*128+k*128+k),np.float32)
  with self.assertRaises(ValueError):encode_request(h,carry(89,5120),cv,log)
  h=header(1);cv[0]=np.float32(.1234567);log=np.zeros(45*(12*128+k*128+k),np.float32)
  with self.assertRaises(ValueError):encode_request(h,carry(1,5120),cv,log)

if __name__=='__main__':unittest.main()
