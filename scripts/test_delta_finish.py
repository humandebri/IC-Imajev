import pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import TextGraph
from transport import encode
class DeltaFinishTests(unittest.TestCase):
 def test_finish_preserves_histories_states_and_cold_fallback(self):
  for n,keep,codec in [(45,True,"bf16-block256-exact-v1"),(87,False,"bf16-block256-exact-v1"),(89,False,"bf16-block256-exact-v1"),(132,False,"bf16-block256-exact-v1"),(87,False,"bf16-exact")]:
   with self.subTest(n=n,keep=keep),tempfile.TemporaryDirectory()as d:
    class Fake:
     wire_codec=codec;max_floats=900000
     def __init__(self):self.directory=pathlib.Path(d);self.calls=[];self.saved=np.zeros(n*2762,np.float32);self.saved[n*2560:n*2570]=1.
     def run(self,op,values,dims,tensor):
      encode(dict(op=op,dims=dims,tensor=tensor,encoding=self.wire_codec),values)
      count,heads,first,retain=dims;self.calls.append(op)
      conv=np.full(3*heads*256,2.,np.float32);state=np.broadcast_to(np.arange(first,first+heads,dtype=np.float32)[:,None],(heads,16384)).ravel()
      if op=='delta_project_finish':
       expected=np.broadcast_to(np.arange(16,dtype=np.float32)[None,:,None],(n,16,128)).ravel()
       np.testing.assert_array_equal(values[-len(expected):],expected)
       output=np.full(n*2560,7.,np.float32)
      else:output=np.broadcast_to(np.arange(first,first+heads,dtype=np.float32)[None,:,None],(n,heads,128)).ravel()
      return np.concatenate([output,conv,*([state]if retain else []),*([self.saved]if first==0 else [])])
    t=Fake();g=TextGraph(t,dict(tensors=[]),arithmetic='int8',fuse_delta=True,fuse_delta_projected=True,fuse_delta_finish=True,compact_heads=True,retain_terminal_state=keep)
    projected=[]
    def linear(x,name):
     projected.append(name);np.testing.assert_array_equal(x.reshape(n,32,128),np.broadcast_to(np.arange(32,dtype=np.float32)[None,:,None],(n,32,128)))
     return np.full((n,2560),7.,np.float32)
    g.linear=linear;result=g.delta(np.ones((n,2560),np.float32),'model.linear_attn',0)
    np.testing.assert_array_equal(result,7.)
    self.assertEqual(t.calls,['delta_project_capture','delta_project_finish'if n<=90 and codec=='bf16-block256-exact-v1' else'delta_project_reuse'])
    self.assertEqual(bool(projected),n>90 or codec=='bf16-exact')
    with np.load(pathlib.Path(d)/'states/layer-00.npz')as state:
     np.testing.assert_array_equal(state['conv'],2.)
     if keep:np.testing.assert_array_equal(state['delta'][:,0,0],np.arange(32))
     else:self.assertEqual(set(state.files),{'conv'})
 def test_flag_requires_projected_delta(self):
  with self.assertRaisesRegex(ValueError,'requires projected Delta'):
   TextGraph(object(),dict(tensors=[]),fuse_delta_finish=True)
if __name__=='__main__':unittest.main()
