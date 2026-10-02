"""Immutable client-held exact prefix states; no persistent query state or reference hidden inputs."""
import hashlib,json,pathlib
import numpy as np
from full_inference import TextGraph

def file_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def graph_hash():
    root=pathlib.Path(__file__).resolve().parent
    return hashlib.sha256((file_hash(root/'full_inference.py')+file_hash(root/'prefix_inference.py')).encode()).hexdigest()
def verify_module(transport,expected):
    actual=transport.command(dict(op='module_hash'))['ok']['module_hash']
    if actual!=expected:raise ValueError('deployed canister module hash mismatch')
    return actual
def load_cache(directory,manifest,wasm_hash):
    directory=pathlib.Path(directory);m=json.loads((directory/'cache.json').read_text())
    if m['version']!=1 or m['model']!=manifest['model'] or m['pack_hash']!=manifest['pack_hash'] or m['wasm_sha256']!=wasm_hash or m['graph_sha256']!=graph_hash():raise ValueError('prefix identity mismatch')
    if not 1<=len(m['token_ids'])<512:raise ValueError('prefix length')
    expected={f'layer-{i:02d}.npy' for i in range(32)}|{f'states/layer-{i:02d}.npz' for i in range(32)}
    if set(m['files'])!=expected:raise ValueError('prefix file manifest')
    states=[];hidden=[];n=len(m['token_ids'])
    for name,h in m['files'].items():
        if file_hash(directory/name)!=h:raise ValueError('prefix file hash')
    for i in range(32):
        a=np.load(directory/f'layer-{i:02d}.npy',allow_pickle=False)
        if a.shape!=(n,2560) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('prefix hidden shape')
        hidden.append(a)
        with np.load(directory/f'states/layer-{i:02d}.npz',allow_pickle=False) as f:state={k:f[k].copy() for k in f.files}
        shapes=({'keys':(n,4,256),'values':(n,4,256),'positions':(n,)} if (i+1)%4==0 else {'conv':(3,8192),'delta':(32,128,128)})
        if set(state)!=set(shapes):raise ValueError('prefix state fields')
        for k,shape in shapes.items():
            dtype=np.int32 if k=='positions' else np.float32
            if state[k].shape!=shape or state[k].dtype!=dtype or not np.isfinite(state[k]).all():raise ValueError('prefix state shape')
        if 'positions' in state and not np.array_equal(state['positions'],np.arange(n,dtype=np.int32)):raise ValueError('prefix positions')
        states.append(state)
    return dict(metadata=m,states=states,hidden=hidden,identity=file_hash(directory/'cache.json'))
class PrefixTextGraph(TextGraph):
    def __init__(self,*args,cache,**kwargs):
        super().__init__(*args,**kwargs)
        if self.arithmetic!='int8' or getattr(self.t,'wire_codec','')!='bf16-exact':raise ValueError('prefix path requires tested integer/lossless mode')
        self.cache=cache;self.position_offset=len(cache['metadata']['token_ids'])
    def initial_conv(self,channel,count):return self.cache['states'][self.current_layer]['conv'][:,channel:channel+count].copy()
    def initial_delta(self,head,heads):return self.cache['states'][self.current_layer]['delta'][head:head+heads].reshape(heads,128*128).copy()
    def attention_history(self,k,v,layer):
        state=self.cache['states'][layer]
        return np.concatenate([state['keys'],k]),np.concatenate([state['values'],v])
    def recorded_hidden(self,hidden,layer):return np.concatenate([self.cache['hidden'][layer],hidden])
    def forward(self,token_ids,layers=32):
        n=self.position_offset
        if layers!=32 or not n<len(token_ids)<=512 or list(token_ids[:n])!=self.cache['metadata']['token_ids']:raise ValueError('prefix tokens/length/layers mismatch')
        return super().forward(token_ids[n:],layers)
