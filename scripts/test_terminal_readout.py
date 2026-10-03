import pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import TextGraph
from transport import encode
class TerminalReadoutTests(unittest.TestCase):
 def test_last_query_uses_absolute_position_and_full_kv_history(self):
  for layer,qn in [(3,4),(31,1)]:
   with tempfile.TemporaryDirectory() as directory:
    class Transport:
     wire_codec='bf16-exact';max_floats=900000
     def __init__(self):self.directory=pathlib.Path(directory);self.calls=[]
     def run(self,op,values,dims,scalars=()):
      encode(dict(op=op,dims=dims,encoding='bf16-exact'),values);self.calls.append((op,dims))
      if op=='rope_heads':return np.asarray(values).ravel()
      self.assert_shape=op=='gqa_suffix_bf16' and dims==[qn,256,16,9-qn]
      assert self.assert_shape
      return np.zeros(qn*16*256,np.float32)
    t=Transport();g=TextGraph(t,dict(tensors=[]),compact_heads=True,compact_lossless=True,retain_terminal_state=False,terminal_readout=True,attention_head_cap=8)
    g.position_offset=5;g.norm=lambda x,*args:x;g.pair=lambda op,a,b:a;calls=[]
    def linear(x,name):
     calls.append((name,len(x)))
     return np.zeros((len(x),8192 if name.endswith('q_proj') else 1024 if name.endswith(('k_proj','v_proj')) else 2560),np.float32)
    g.linear=linear
    g.attention_history=lambda k,v,layer:(np.concatenate([np.zeros((5,4,256),np.float32),k]),np.concatenate([np.zeros((5,4,256),np.float32),v]))
    out=g.attention(np.zeros((4,2560),np.float32),'model.self_attn',layer)
    self.assertEqual(out.shape,(qn,2560));self.assertEqual([n for _,n in calls],[qn,4,4,qn])
    self.assertEqual([d[3] for op,d in t.calls if op=='rope_heads'],[9-qn,5])
    with np.load(pathlib.Path(directory)/f'states/layer-{layer:02d}.npz') as state:self.assertEqual(state['keys'].shape,(9,4,256))
 def test_cache_preparation_cannot_drop_terminal_hidden(self):
  class T:wire_codec='bf16-exact'
  with self.assertRaisesRegex(ValueError,'terminal inference'):TextGraph(T(),dict(tensors=[]),compact_heads=True,terminal_readout=True)
if __name__=='__main__':unittest.main()
