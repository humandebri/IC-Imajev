import pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import TextGraph
from transport import encode
class DeltaProjectedTests(unittest.TestCase):
 def test_capture_reuse_preserves_groups_histories_and_optional_states(self):
  for n,keep,codec in [(45,True,'bf16-block256-exact-v1'),(87,False,'bf16-block256-exact-v1'),(132,False,'bf16-block256-exact-v1'),(132,True,'bf16-block256-exact-v1'),(87,True,'bf16-exact')]:
   with self.subTest(n=n,keep=keep,codec=codec),tempfile.TemporaryDirectory() as d:
    class Fake:
     wire_codec=codec;max_floats=900000
     def __init__(self):self.directory=pathlib.Path(d);self.calls=[];self.saved=np.zeros(n*2762,np.float32);self.saved[n*2560:n*2570]=1.
     def run(self,op,values,dims,tensor):
      encode(dict(op=op,dims=dims,tensor=tensor,encoding=codec),values);count,heads,first,retain=dims;self.calls.append((op,dims))
      if first:np.testing.assert_array_equal(values[:n*2762],self.saved)
      output=np.broadcast_to(np.arange(first,first+heads,dtype=np.float32)[None,:,None],(n,heads,128)).ravel()
      conv=np.full(3*heads*256,2.,np.float32);state=np.broadcast_to(np.arange(first,first+heads,dtype=np.float32)[:,None],(heads,16384)).ravel()
      result=np.concatenate([output,conv,*([state] if retain else []),*([self.saved] if first==0 else [])])
      encoded=encode(dict(op=op,dims=dims,tensor=tensor,encoding=codec),result);assert len(encoded)<=2000000
      return result
    t=Fake();g=TextGraph(t,dict(tensors=[]),arithmetic='int8',fuse_delta=True,fuse_delta_projected=True,compact_heads=True,retain_terminal_state=keep);g.linear=lambda x,name:x
    result=g.delta(np.ones((n,2560),np.float32),'model.linear_attn',0).reshape(n,32,128)
    np.testing.assert_array_equal(result,np.broadcast_to(np.arange(32,dtype=np.float32)[None,:,None],(n,32,128)))
    self.assertEqual(t.calls[0][0],'delta_project_capture');self.assertTrue(all(op=='delta_project_reuse' for op,dims in t.calls[1:]));self.assertEqual(sum(dims[1] for op,dims in t.calls),32)
    if n==132 and keep:self.assertEqual(t.calls[0][1][1],10)
    with np.load(pathlib.Path(d)/'states/layer-00.npz') as state:
     np.testing.assert_array_equal(state['conv'],2.)
     if keep:np.testing.assert_array_equal(state['delta'][:,0,0],np.arange(32))
     else:self.assertEqual(set(state.files),{'conv'})
if __name__=='__main__':unittest.main()
