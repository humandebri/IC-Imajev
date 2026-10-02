import pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import TextGraph
from prefix_inference import verify_module
from transport import encode,is_instruction_limit
class Frames:
    wire_codec='bf16-exact';max_floats=900000
    def __init__(self,path):self.directory=path;self.attention_heads=[]
    def run(self,op,values,dims=(),scalars=(),tensor='',aux=()):
        encode(dict(op=op,dims=list(dims),encoding=self.wire_codec),values)
        if op=='attention_heads_bf16':self.attention_heads.append(dims[2]);return np.zeros(dims[0]*dims[1]*dims[2],np.float32)
        return np.zeros(np.asarray(values).size,np.float32)
class FixTests(unittest.TestCase):
    def test_module_mismatch_rejected(self):
        class T:
            def command(self,cmd):return dict(ok=dict(module_hash='other'))
        with self.assertRaisesRegex(ValueError,'deployed'):verify_module(T(),'expected')
    def test_other_rejections_not_retried(self):
        self.assertFalse(is_instruction_limit('model mismatch'));self.assertTrue(is_instruction_limit('IC0522'))
    def test_attention_long_inputs_fit_actual_encoder(self):
        for n in [147,512]:
            with tempfile.TemporaryDirectory() as d:
                t=Frames(pathlib.Path(d));g=TextGraph(t,dict(tensors=[]),arithmetic='int8',attention_head_cap=8,compact_lossless=True)
                g.norm=lambda x,*a,**kw:x
                g.linear=lambda x,name:np.zeros((len(x),8192 if name.endswith('.q_proj') else 1024),np.float32)
                g.pair=lambda op,a,b:a
                g.attention(np.zeros((n,2560),np.float32),'model.test',3)
                self.assertEqual(sum(t.attention_heads),16);self.assertLess(max(t.attention_heads),8)
    def test_delta_frame_includes_f32_state_budget(self):
        g=TextGraph(Frames(pathlib.Path('/private/tmp')),dict(tensors=[]),compact_lossless=True)
        for n in [132,512]:
            cap=g.bounded_heads(16,n*384,n*2+16384)
            v=np.concatenate([np.zeros(cap*n*384,np.float32),np.full(cap*(n*2+16384),np.float32(.12345))])
            encode(dict(op='delta_heads_bf16',dims=[n,128,128,cap],encoding='bf16-exact'),v)
