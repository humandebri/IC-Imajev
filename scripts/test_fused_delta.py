import pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import TextGraph
from transport import encode
class FusedDeltaTests(unittest.TestCase):
 def test_channel_history_state_continuation_and_terminal_omission(self):
  for retain in [False,True]:
   with tempfile.TemporaryDirectory() as directory:
    class Transport:
     wire_codec='bf16-exact';max_floats=900000
     def __init__(self):self.directory=pathlib.Path(directory);self.calls=[]
     def run(self,op,x,dims,**kw):
      n,h,first,keep=dims;self.calls.append(dims)
      encode(dict(op=op,dims=dims,encoding='bf16-exact'),x)
      channels=h*256;window=np.asarray(x[:(n+3)*channels]).reshape(n+3,channels)
      indices=np.concatenate([np.arange(first//2*128,(first+h)//2*128),np.arange(2048+first//2*128,2048+(first+h)//2*128),np.arange(4096+first*128,4096+(first+h)*128)])
      expected=np.broadcast_to(indices,(3,channels))*(1 if n==2 else 0)
      np.testing.assert_array_equal(window[:3],expected)
      np.testing.assert_array_equal(x[-h*16384:],7 if n==2 else 0)
      return np.concatenate([np.full(n*h*128,first,np.float32),np.full(h*16384,7,np.float32)]) if keep else np.full(n*h*128,first,np.float32)
    t=Transport();g=TextGraph(t,dict(tensors=[]),compact_lossless=True,compact_heads=True,fuse_delta=True,retain_terminal_state=retain,token_cap=3,delta_head_cap=8)
    g.linear=lambda x,name:x
    mixed=np.broadcast_to(np.arange(8192,dtype=np.float32),(5,8192)).copy()
    out=g.delta_stage(mixed,np.zeros((5,32,128),np.float32),np.ones((5,32),np.float32),np.ones((5,32),np.float32),'model.layers.0.linear_attn',0)
    for head in range(32):np.testing.assert_array_equal(out.reshape(5,32,128)[:,head],head//8*8)
    self.assertEqual(sum(d[3]==0 for d in t.calls),0 if retain else 4)
    with np.load(pathlib.Path(directory)/'states/layer-00.npz') as state:
     np.testing.assert_array_equal(state['conv'],mixed[-3:]);self.assertEqual('delta' in state,retain)
if __name__=='__main__':unittest.main()
