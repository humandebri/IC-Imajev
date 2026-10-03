#!/usr/bin/env python3
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import TextGraph
from transport import encode
class FullMlpTests(unittest.TestCase):
 def test_routes_complete_mlp_and_final_norm_without_codec_change(self):
  for n,layer in [(1,0),(45,31),(80,30),(87,30),(89,30)]:
   with self.subTest(n=n,layer=layer):
    class Fake:
     wire_codec='bf16-block256-exact-v1';max_floats=900000
     def __init__(self):self.calls=[]
     def run(self,op,x,dims,scalars,tensor,aux):
      self.calls.append(op);self_outer.assertEqual(op,'mlp_full_integer');self_outer.assertEqual(dims,[n,2560]);self_outer.assertEqual(scalars,[2.,1e-6]);self_outer.assertEqual(aux,['model.language_model.norm.weight' if layer==31 else f'model.language_model.layers.{layer+1}.input_layernorm.weight']);encode(dict(op=op,dims=dims,encoding=self.wire_codec),x)
      return np.concatenate([np.full(n*2560,4.,np.float32),np.full(n*2560,8.,np.float32)])
    self_outer=self;t=Fake();g=TextGraph(t,dict(tensors=[]),arithmetic='int8',fuse_add_norm=True,fuse_mlp=True,fuse_mlp_norm=True,fuse_mlp_pipeline=True,fuse_mlp_full=True,mlp_full_token_cap=89)
    h,z=g.full_mlp(np.zeros((n,2560),np.float32),np.ones((n,2560),np.float32),f'model.language_model.layers.{layer}',layer);self.assertEqual(h.shape,(n,2560));np.testing.assert_array_equal(h,4.);np.testing.assert_array_equal(z,8.);self.assertEqual(t.calls,['mlp_full_integer']);self.assertEqual(t.wire_codec,'bf16-block256-exact-v1')
 def test_requires_pipeline_configuration(self):
  class Fake:wire_codec='bf16-block256-exact-v1'
  with self.assertRaises(ValueError):TextGraph(Fake(),dict(tensors=[]),fuse_mlp_full=True)
if __name__=='__main__':unittest.main()
