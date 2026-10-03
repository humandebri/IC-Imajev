import pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import TextGraph
from transport import encode

class Frames:
    wire_codec='bf16-exact';max_floats=900000
    def __init__(self,path):self.directory=path;self.calls=[];self.next_query_head=0
    def run(self,op,values,dims=(),scalars=(),tensor='',aux=()):
        x=np.asarray(values).ravel();encode(dict(op=op,dims=list(dims),encoding=self.wire_codec),x)
        self.calls.append((op,list(dims)))
        if op=='gqa_heads_bf16':
            n,w,h=dims;groups=(h+3)//4;offset=self.next_query_head//4
            keys=x[h*n*w:(h+groups)*n*w].reshape(groups,n,w)
            for group in range(groups):np.testing.assert_array_equal(keys[group],offset+group+1)
            self.next_query_head+=h
            return np.zeros(n*w*h,np.float32)
        if op in ('rope_heads','rope'):return x
        if op in ('delta_heads_bf16','delta_terminal_bf16'):
            n,k,v,h=dims;tail=x[h*n*(2*k+v):].reshape(h,2*n+k*v)
            # The two-token chunk must receive the three-token chunk's state.
            np.testing.assert_array_equal(tail[:,2*n:],7 if n==2 else 0)
            y=np.zeros(h*n*v,np.float32)
            return y if op=='delta_terminal_bf16' else np.concatenate([y,np.full(h*k*v,7,np.float32)])
        if op=='gated_norm':return np.zeros(x.size//2,np.float32)
        return np.zeros(x.size,np.float32)

class CompactTests(unittest.TestCase):
    def test_gqa_group_tails_fit_real_codec(self):
        for n,expected in [(132,[16]),(147,[12,4]),(512,[2]*8)]:
            with tempfile.TemporaryDirectory() as path:
                t=Frames(pathlib.Path(path));g=TextGraph(t,dict(tensors=[]),compact_lossless=True,compact_heads=True,attention_head_cap=8)
                g.norm=lambda x,*a,**kw:x
                def linear(x,name):
                    if name.endswith('.k_proj'):return np.broadcast_to(np.repeat(np.arange(1,5,dtype=np.float32),256),(len(x),1024)).copy()
                    return np.zeros((len(x),8192 if name.endswith('.q_proj') else 1024),np.float32)
                g.linear=linear
                g.pair=lambda op,a,b:a
                g.attention(np.zeros((n,2560),np.float32),'model.test',3)
                self.assertEqual([dims[2] for op,dims in t.calls if op=='gqa_heads_bf16'],expected)
    def test_terminal_only_on_last_chunk_and_preparation_retains_state(self):
        for retain in [False,True]:
            with tempfile.TemporaryDirectory() as path:
                t=Frames(pathlib.Path(path));g=TextGraph(t,dict(tensors=[]),compact_lossless=True,compact_heads=True,retain_terminal_state=retain,delta_head_cap=8,token_cap=3)
                g.norm=lambda x,*a,**kw:x
                def linear(x,name):
                    cols=8192 if name.endswith('.in_proj_qkv') else 4096 if name.endswith('.in_proj_z') else 2560 if name.endswith('.out_proj') else 32
                    return np.zeros((len(x),cols),np.float32)
                g.linear=linear;g.convolution=lambda x,name:(x,np.zeros((3,8192),np.float32))
                g.delta(np.zeros((5,2560),np.float32),'model.test',0)
                ops=[op for op,_ in t.calls if op.startswith('delta_') and op!='delta_gates']
                self.assertEqual(ops.count('delta_terminal_bf16'),0 if retain else 4)
                with np.load(pathlib.Path(path)/'states/layer-00.npz') as state:self.assertEqual('delta' in state,retain)
if __name__=='__main__':unittest.main()
