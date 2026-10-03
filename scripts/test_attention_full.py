#!/usr/bin/env python3
import pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import TextGraph
from transport import encode
class FullAttentionTests(unittest.TestCase):
 def test_prefix_head_order_terminal_and_state(self):
  for n,prefix,layer,terminal in [(1,0,3,False),(45,0,31,False),(87,45,3,True),(89,45,7,True),(87,45,31,True),(132,0,31,True)]:
   with self.subTest(n=n,prefix=prefix,layer=layer),tempfile.TemporaryDirectory() as d:
    qn=1 if terminal and layer==31 else n
    prior=np.broadcast_to(np.arange(4,dtype=np.float32)[None,:,None]+20,(prefix,4,256)).copy()
    class Fake:
     wire_codec='bf16-block256-exact-v1';max_floats=900000
     def __init__(self):self.directory=pathlib.Path(d);self.calls=[]
     def run(self,op,values,dims,tensor):
      self.calls.append((op,dims));self_outer.assertEqual(op,'attention_full_integer');self_outer.assertEqual(dims,[n,prefix,int(terminal and layer==31)])
      raw=np.asarray(values);encode(dict(op=op,dims=dims,encoding=self.wire_codec),raw)
      np.testing.assert_array_equal(raw[n*2560:n*2560+prefix*1024].reshape(4,prefix,256),prior.transpose(1,0,2))
      k=np.broadcast_to(np.arange(4,dtype=np.float32)[None,:,None],(n,4,256));return np.concatenate([np.full(qn*2560,8.,np.float32),k.ravel(),(k+10).ravel()])
    self_outer=self;t=Fake();g=TextGraph(t,dict(tensors=[]),arithmetic='int8',compact_heads=True,fuse_norm_rope=True,fuse_attention=True,fuse_attention_full=True,terminal_readout=terminal,retain_terminal_state=not terminal)
    g.position_offset=prefix;g.attention_history=lambda k,v,layer:(np.concatenate([prior,k]),np.concatenate([prior+10,v]))
    g.linear=lambda *a: self.fail('Separate output projection')
    got=g.attention(np.ones((n,2560),np.float32),'model.language_model.layers.'+str(layer)+'.self_attn',layer)
    np.testing.assert_array_equal(got,np.full((qn,2560),8.,np.float32));self.assertEqual(len(t.calls),1)
    with np.load(pathlib.Path(d)/f'states/layer-{layer:02d}.npz') as state:
     self.assertEqual(state['keys'].shape,(prefix+n,4,256));np.testing.assert_array_equal(state['keys'][:prefix],prior);np.testing.assert_array_equal(state['positions'],np.arange(prefix+n))
 def test_requires_original_fusion_and_valid_history(self):
  class Fake:wire_codec='bf16-block256-exact-v1'
  with self.assertRaises(ValueError):TextGraph(Fake(),dict(tensors=[]),fuse_attention_full=True)
if __name__=='__main__':unittest.main()
