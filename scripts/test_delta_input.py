import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import TextGraph
class InputFusionTests(unittest.TestCase):
 def test_projection_and_gate_segments_keep_their_token_order(self):
  class Transport:
   wire_codec='bf16-exact';max_floats=900000
   def run(self,op,x,dims,tensor):
    assert op=='delta_input_integer' and dims==[5]
    return np.concatenate([np.arange(5*8192,dtype=np.float32),np.arange(5*32,dtype=np.float32)+100,np.arange(5*32,dtype=np.float32)+300])
  g=TextGraph(Transport(),dict(tensors=[]),arithmetic='int8',fuse_delta=True,fuse_delta_input=True,row_cap=16384,work_cap=2500000000)
  def linear(x,name):
   self.assertTrue(name.endswith('.in_proj_z'));return np.zeros((len(x),4096),np.float32)
  g.linear=linear
  g.delta_stage=lambda mixed,z,decay,beta,name,layer:(mixed,decay,beta)
  mixed,decay,beta=g.delta(np.ones((5,2560),np.float32),'model.layers.0.linear_attn',0)
  np.testing.assert_array_equal(mixed.ravel(),np.arange(5*8192,dtype=np.float32))
  np.testing.assert_array_equal(decay.ravel(),np.arange(160,dtype=np.float32)+100)
  np.testing.assert_array_equal(beta.ravel(),np.arange(160,dtype=np.float32)+300)
if __name__=='__main__':unittest.main()
