"""Immutable client-held exact prefix states; no persistent query state or reference hidden inputs."""
import hashlib,json,pathlib
import numpy as np
from full_inference import TextGraph,delta_full_log_enabled

def file_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def graph_hash():
    root=pathlib.Path(__file__).resolve().parent
    return hashlib.sha256((file_hash(root/'full_inference.py')+file_hash(root/'prefix_inference.py')).encode()).hexdigest()
def verify_module(transport,expected):
    actual=transport.command(dict(op='module_hash'))['ok']['module_hash']
    if actual!=expected:raise ValueError('deployed canister module hash mismatch')
    return actual

def delta_state_version(states):
    kinds=set()
    for i,state in enumerate(states):
        if (i+1)%4==0:continue
        if set(state)=={'conv','delta'}:kinds.add(1)
        elif set(state)=={'conv','delta_log'}:kinds.add(2)
        else:raise ValueError('prefix state fields')
    if len(kinds)!=1:raise ValueError('prefix Delta representation version')
    return next(iter(kinds))

def saved_delta_state_version(directory):
    states=[]
    for i in range(32):
        if (i+1)%4==0:
            states.append({})
        else:
            with np.load(pathlib.Path(directory)/f'states/layer-{i:02d}.npz',allow_pickle=False) as f:
                states.append(dict.fromkeys(f.files))
    return delta_state_version(states)

def load_cache(directory,manifest,wasm_hash):
    directory=pathlib.Path(directory);m=json.loads((directory/'cache.json').read_text())
    if m['version'] not in (1,2) or m['model']!=manifest['model'] or m['pack_hash']!=manifest['pack_hash'] or m['wasm_sha256']!=wasm_hash or m['graph_sha256']!=graph_hash():raise ValueError('prefix identity mismatch')
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
        if (i+1)%4!=0 and 'delta_log' in state:shapes={'conv':(3,8192),'delta_log':(n*6176,)}
        if set(state)!=set(shapes):raise ValueError('prefix state fields')
        for k,shape in shapes.items():
            dtype=np.int32 if k=='positions' else np.float32
            if state[k].shape!=shape or state[k].dtype!=dtype or not np.isfinite(state[k]).all():raise ValueError('prefix state shape')
        if 'positions' in state and not np.array_equal(state['positions'],np.arange(n,dtype=np.int32)):raise ValueError('prefix positions')
        states.append(state)
    if m['version']!=delta_state_version(states):raise ValueError('prefix Delta representation version')
    return dict(metadata=m,states=states,hidden=hidden,identity=file_hash(directory/'cache.json'))
class PrefixTextGraph(TextGraph):
    def __init__(self,*args,cache,fuse_prefix_start=False,**kwargs):
        super().__init__(*args,**kwargs)
        if self.arithmetic!='int8' or getattr(self.t,'wire_codec','') not in ('bf16-exact','bf16-block256-exact-v1'):raise ValueError('prefix path requires tested integer/lossless mode')
        self.cache=cache;self.position_offset=len(cache['metadata']['token_ids'])
        version=delta_state_version(cache['states'])
        if 'version' in cache['metadata'] and cache['metadata']['version']!=version:raise ValueError('prefix Delta representation version')
        # Dense cache continuation must consume its nonzero state, even when
        # the user requests the full-log optimization for a short suffix.
        self.fuse_delta_full_log=delta_full_log_enabled(self.fuse_delta_full_log,1,self.t.wire_codec,self.retain_terminal_state,version)
        self.fuse_prefix_start=fuse_prefix_start
        if fuse_prefix_start and (cache.get('hybrid_packets') is None or not self.fuse_delta_full_log or self.retain_terminal_state):
            raise ValueError('prefix start requires hybrid full Delta and discarded suffix state')
    def prefix_start(self,token_ids):
        n=len(token_ids)
        if not 1<=n<=89:raise ValueError('prefix start suffix token bound')
        self.current_layer=0
        conv=self.cache['states'][0]['conv']
        payload=np.concatenate([np.asarray(token_ids,dtype=np.float32),conv.ravel()])
        out=self.t.run('prefix_start_integer',payload,[n,self.position_offset],tensor='model.language_model.embed_tokens.weight',prefix_packet=self.cache['hybrid_packets'][0])
        if len(out)!=2*n*2560+3*8192:raise ValueError('prefix start reply shape')
        path=self.t.directory/'states';path.mkdir(exist_ok=True)
        np.savez(path/'layer-00.npz',conv=out[2*n*2560:].reshape(3,8192))
        return out[:n*2560].reshape(n,2560),out[n*2560:2*n*2560].reshape(n,2560)
    def delta(self,x,name,layer):
        if self.cache.get('hybrid_packets') is not None:
            if not self.fuse_delta_full_log or self.retain_terminal_state:raise ValueError('hybrid prefix graph mode')
            # No per-layer log copy/scan just to select the prepared path.
            return self.delta_full_log(x,name,layer)
        return super().delta(x,name,layer)
    def delta_full_log(self,x,name,layer):
        packets=self.cache.get('hybrid_packets')
        if packets is None:return super().delta_full_log(x,name,layer)
        if self.retain_terminal_state:raise ValueError('hybrid prefix requires discarded suffix state')
        n=len(x);p=self.position_offset
        payload=np.concatenate([x.ravel(),self.initial_conv(0,8192).ravel()])
        out=self.t.run('delta_full_hybrid_integer',payload,[n,32,p,0],tensor=name+'.in_proj_qkv.weight',prefix_packet=packets[layer])
        if len(out)!=n*2560+3*8192:raise ValueError('hybrid Delta reply shape')
        path=self.t.directory/'states';path.mkdir(exist_ok=True)
        np.savez(path/f'layer-{layer:02d}.npz',conv=out[n*2560:].reshape(3,8192))
        return out[:n*2560].reshape(n,2560)
    def initial_conv(self,channel,count):return self.cache['states'][self.current_layer]['conv'][:,channel:channel+count].copy()
    def initial_delta_log(self):return self.cache['states'][self.current_layer].get('delta_log',np.empty(0,dtype=np.float32)).copy()
    def initial_delta(self,head,heads):return self.cache['states'][self.current_layer]['delta'][head:head+heads].reshape(heads,128*128).copy()
    def attention_history(self,k,v,layer):
        state=self.cache['states'][layer]
        return np.concatenate([state['keys'],k]),np.concatenate([state['values'],v])
    def recorded_hidden(self,hidden,layer):return np.concatenate([self.cache['hidden'][layer],hidden])
    def forward(self,token_ids,layers=32):
        n=self.position_offset
        if layers!=32 or not n<len(token_ids)<=512 or list(token_ids[:n])!=self.cache['metadata']['token_ids']:raise ValueError('prefix tokens/length/layers mismatch')
        return super().forward(token_ids[n:],layers)
