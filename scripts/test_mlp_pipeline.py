#!/usr/bin/env python3
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import encode,decode
from mlp_codec import NAME
from full_inference import TextGraph

def header(n,op='mlp_prepare_down'):
 return dict(version=2,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,op=op,tensor='model.language_model.layers.0.post_attention_layernorm.weight',dims=[n,2560],scalars=[2.,1e-6],aux=['model.language_model.layers.1.input_layernorm.weight'],encoding=NAME)
class Tests(unittest.TestCase):
 def test_partial_rows_metadata_keeps_legacy_and_rejects_bad_rows(self):
  n=1;c=n*2560;q=n*9216
  v=np.concatenate([np.full(c,-0.,dtype='<f4'),np.zeros(q,dtype='<f4'),np.ones(n*36,dtype='<f4'),np.zeros(n*64,dtype='<f4')])
  for rows in [32,256,512,2528]:
   h=header(n,'mlp_prepare_partial_down');h['dims'].append(rows);rh,got=decode(encode(h,v));self.assertEqual(rh['dims'],[n,2560,rows]);np.testing.assert_array_equal(v.view('<u4'),got.view('<u4'))
  for rows in [0,1,31,33,2560,True,2**64-1]:
   h=header(n,'mlp_prepare_partial_down');h['dims'].append(rows)
   with self.assertRaises(ValueError):encode(h,v)
  h=header(n);h['dims'].append(512)
  with self.assertRaises(ValueError):encode(h,v)

 def test_raw_partial_finish_requires_explicit_prepared_progress(self):
  n=1;c=2560;q=9216
  v=np.concatenate([np.full(c,-0.,dtype='<f4'),np.zeros(q,dtype='<f4'),np.ones(36,dtype='<f4'),np.zeros(64,dtype='<f4')])
  for rows in [32,768,1600,2528]:
   h=header(n,'mlp_down_norm_partial_prepared');h['dims'].append(rows)
   rh,got=decode(encode(h,v));self.assertEqual(rh['dims'],[n,c,rows]);self.assertEqual(v.tobytes(),got.tobytes())
   with self.assertRaisesRegex(ValueError,'prepared state'):encode(h,np.zeros(2*c,dtype='<f4'))
  with self.assertRaises(ValueError):encode(header(n,'mlp_down_norm_partial_prepared'),v)
 def test_codec_exact_segments_and_bounds(self):
  for n in [1,45,87,89]:
   c,q=n*2560,n*9216
   v=np.concatenate([np.full(c,-0.,dtype='<f4'),np.resize(np.array([-127.,127.,0.,1.,-1.],dtype='<f4'),q),np.full(n*36,0.00123,dtype='<f4'),np.full(n*64,1.0000001,dtype='<f4')])
   for op in ['mlp_prepare_down','mlp_down_norm_prepared']:
    b=encode(header(n,op),v);self.assertLessEqual(len(b),2_000_000);_,got=decode(b);np.testing.assert_array_equal(v.view('<u4'),got.view('<u4'))
   for i,bad in [(c,-128.),(c,1.1),(c,-0.),(c+q,0.),(c+q,float('inf'))]:
    broken=v.copy();broken[i]=bad
    with self.assertRaises(ValueError):encode(header(n),broken)
  for n in [0,90,2**64-1]:
   with self.assertRaises(ValueError):encode(header(n),np.empty(0,dtype='<f4'))
 def test_client_routes_state_and_restores_codec(self):
  class Fake:
   wire_codec='bf16-block256-exact-v1';max_floats=900000
   def __init__(self,fail=False):self.calls=[];self.fail=fail
   def run(self,op,v,dims,scalars,**kw):
    self.calls.append(op);assert self.wire_codec==NAME
    if self.fail:raise RuntimeError('synthetic failure')
    n=dims[0]
    if op=='mlp_prepare_down':return np.zeros(n*11876,dtype='<f4')
    return np.concatenate([np.full(n*2560,1.,dtype='<f4'),np.full(n*2560,2.,dtype='<f4')])
  for fail in [False,True]:
   t=Fake(fail);g=TextGraph(t,{'tensors':[]},arithmetic='int8',fuse_mlp=True,fuse_add_norm=True,fuse_mlp_norm=True,fuse_mlp_pipeline=True)
   x=np.zeros((1,2560),dtype='<f4')
   if fail:
    with self.assertRaises(RuntimeError):g.pipeline_mlp(x,x,'model.language_model.layers.0',0)
   else:
    a,b=g.pipeline_mlp(x,x,'model.language_model.layers.0',0);self.assertTrue(np.all(a==1));self.assertTrue(np.all(b==2));self.assertEqual(t.calls,['mlp_prepare_down','mlp_down_norm_prepared'])
   self.assertEqual(t.wire_codec,'bf16-block256-exact-v1')
if __name__=='__main__':unittest.main()
