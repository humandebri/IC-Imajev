#!/usr/bin/env python3
"""Client routing preserves full KV histories, heads and terminal Q position."""
import pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import TextGraph
from transport import encode
class AttentionFusionTests(unittest.TestCase):
 def test_routing_all_heads_history_and_terminal_readout(self):
  for n,prefix,layer,terminal in [(45,0,3,False),(87,45,3,True),(89,45,7,True),(132,0,3,True),(87,45,31,True)]:
   with self.subTest(n=n,prefix=prefix,layer=layer),tempfile.TemporaryDirectory() as d:
    class Fake:
     wire_codec='bf16-block256-exact-v1';max_floats=900000
     def __init__(self):self.directory=pathlib.Path(d);self.calls=[]
     def run(self,op,values,dims,tensor):
      encode(dict(op=op,dims=dims,encoding=self.wire_codec),values);self.calls.append((op,dims,np.asarray(values).copy()))
      if op=='attention_kv_integer':
       k=np.broadcast_to(np.arange(4,dtype=np.float32)[None,:,None],(n,4,256));return np.concatenate([k.ravel(),(k+8).ravel()])
      qn,total,offset,first,heads=dims;assert total==prefix+n and offset==total-qn
      payload=np.asarray(values);kv=total*(heads//4)*256
      keys=payload[qn*2560:qn*2560+kv].reshape(heads//4,total,256)
      for group in range(heads//4):np.testing.assert_array_equal(keys[group,prefix:],first//4+group)
      return np.broadcast_to(np.arange(first,first+heads,dtype=np.float32)[None,:,None],(qn,heads,256)).ravel()
    t=Fake();g=TextGraph(t,dict(tensors=[]),arithmetic='int8',compact_heads=True,fuse_norm_rope=True,fuse_attention=True,terminal_readout=terminal,retain_terminal_state=not terminal)
    g.position_offset=prefix;g.attention_history=lambda k,v,layer:(np.concatenate([np.zeros((prefix,4,256),np.float32),k]),np.concatenate([np.zeros((prefix,4,256),np.float32),v]))
    g.linear=lambda x,name:x
    result=g.attention(np.ones((n,2560),np.float32),'layer.self_attn',layer);qn=1 if terminal and layer==31 else n
    np.testing.assert_array_equal(result.reshape(qn,16,256),np.broadcast_to(np.arange(16,dtype=np.float32)[None,:,None],(qn,16,256)))
    self.assertEqual(t.calls[0][1],[n,prefix]);self.assertEqual(len(t.calls),2 if qn<=89 else 3)
    with np.load(pathlib.Path(d)/f'states/layer-{layer:02d}.npz') as states:self.assertEqual(states['keys'].shape,(prefix+n,4,256));np.testing.assert_array_equal(states['positions'],np.arange(prefix+n))
 def test_fusion_rejects_lossy_or_missing_norm_configuration(self):
  class Fake:wire_codec='int8-block256-v1'
  with self.assertRaises(ValueError):TextGraph(Fake(),dict(tensors=[]),arithmetic='int8',compact_heads=True,fuse_attention=True,fuse_norm_rope=True)
if __name__=='__main__':unittest.main()
