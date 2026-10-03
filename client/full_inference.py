"""Client-held text-only Qwen3.5/LoRA graph. All numerical kernels run via query.

NumPy is used only for routing, reshape, concatenation, zero states and reporting.
An official token-ID fixture is input; no reference hidden state enters inference.
"""
import hashlib,json,pathlib,time
import numpy as np
from transport import Transport,encode,decode,atomic

PREFIX='model.language_model.'

def delta_log_prefix_fits(tokens,codec):
    """Reserve a maximum-size header and worst-case F32 innovation/decay tail."""
    if not 1<=tokens<=90 or codec not in ('bf16-exact','bf16-block256-exact-v1'):return False
    count=tokens*(2560+6176)+3*8192
    bitmap=(count+7)//8 if codec=='bf16-exact' else ((count+255)//256+7)//8
    size=16424+bitmap+2*(tokens*(2560+2048)+3*8192)+4*tokens*(4096+32)
    return count<=900000 and size<=2000000

def delta_full_log_enabled(requested,tokens,codec,retain_terminal_state,cache_version=None):
    if cache_version==2:
        if not requested:raise ValueError('prefix Delta state representation mismatch')
        if tokens>90:raise ValueError('logged prefix continuation requires at most 90 tokens')
        return True
    if cache_version==1:return False
    if cache_version is not None:raise ValueError('prefix Delta representation version')
    return bool(requested and 1<=tokens<=90 and (not retain_terminal_state or delta_log_prefix_fits(tokens,codec)))

class JournalTransport(Transport):
    def __init__(self,*args,input_hash,**kwargs):
        super().__init__(*args,**kwargs);self.input_hash=input_hash;self.replayed=0
    def projection_context(self,values,context):
        return dict(**context,model=self.model,pack_hash=self.pack_hash,input_hash=self.input_hash,encoding=getattr(self,'wire_codec',''),values_sha256=hashlib.sha256(np.asarray(values,dtype='<f4').tobytes()).hexdigest())
    def projection_width(self,values,context,width):
        policy=self.directory/f'{self.index:06d}.projection-limit.json'
        if not policy.exists():return width
        saved=json.loads(policy.read_text())
        if saved['context']!=self.projection_context(values,context):raise ValueError('projection limit checkpoint input mismatch')
        if not isinstance(saved['width'],int) or not 1<=saved['width']<=width:raise ValueError('projection limit width mismatch')
        return saved['width']
    def remember_projection_limit(self,values,context,failed_width,width):
        prefix=self.directory/f'{self.index:06d}'
        if prefix.with_suffix('.response.bin').exists() or prefix.with_suffix('.metric.json').exists():raise ValueError('refusing to replace completed projection')
        request=prefix.with_suffix('.request.bin')
        if request.exists():request.replace(self.directory/f'{self.index:06d}.failed-width-{failed_width}.request.bin')
        atomic(prefix.with_suffix('.projection-limit.json'),(json.dumps(dict(context=self.projection_context(values,context),width=width),indent=2)+'\n').encode())
    def run(self,op,values,dims=(),scalars=(),tensor='',input_hash=None,aux=(),prefix_packet=None):
        input_hash=self.input_hash if input_hash is None else input_hash
        h=dict(version=getattr(self,'frame_version',1),model=self.model,pack_hash=self.pack_hash,input_hash=input_hash,step=self.index,op=op,tensor=tensor,dims=list(dims),scalars=list(scalars))
        if aux:h['aux']=list(aux)
        if getattr(self,'wire_codec',''):h['encoding']=self.wire_codec
        request=self.directory/f'{self.index:06d}.request.bin';response=self.directory/f'{self.index:06d}.response.bin';metric=self.directory/f'{self.index:06d}.metric.json'
        if prefix_packet is None:payload=encode(h,values)
        else:
            if op=='prefix_start_integer':
                from prefix_start import NAME,encode_request
            else:
                from prefix_hybrid import NAME,encode_request
            h['encoding']=NAME
            payload=encode_request(h,values,prefix_packet)
        if request.exists() and request.read_bytes()!=payload:raise ValueError(f'checkpoint input mismatch at query {self.index}; use a fresh directory after changing the graph')
        if response.exists() and metric.exists():
            rh,v=decode(response.read_bytes())
            if rh['step']!=self.index+1 or any(rh[k]!=value for k,value in h.items() if k not in ('step','scalars')) or not np.array_equal(np.asarray(rh['scalars'],dtype=np.float32),np.asarray(h['scalars'],dtype=np.float32)):raise ValueError('checkpoint reply identity')
            saved=json.loads(metric.read_text())
            fused=op in ('terminal_attention_mlp_integer','terminal_tail_integer') and bool(getattr(self,'fuse_terminal_decision',False))
            if fused:
                if saved.get('decision_options')!=self.decision_options or 'decision' not in saved['ok']:raise ValueError('terminal decision checkpoint options/result mismatch')
                self.terminal_decision=saved['ok']['decision']
            elif 'decision_options' in saved:raise ValueError('terminal decision checkpoint mode mismatch')
            saved['replayed']=True;self.measurements.append(saved);self.index+=1;self.replayed+=1;return v
        start=time.perf_counter()
        try:values=self._run_encoded(h,payload)
        except Exception as error:
            with (self.directory/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(index=self.index,op=op,tensor=tensor,dims=list(dims),error=str(error),wall_seconds=time.perf_counter()-start))+'\n')
            raise
        atomic(metric,(json.dumps(self.measurements[-1],indent=2)+'\n').encode());return values

class TextGraph:
    def __init__(self,transport,manifest,*,row_cap=768,token_cap=132,work_cap=300_000_000,delta_head_cap=1,attention_head_cap=1,compact_lossless=False,arithmetic='f32',fuse_add_norm=False,fuse_mlp=False,compact_heads=False,retain_terminal_state=True,fuse_delta=False,wide_mlp=False,terminal_readout=False,fuse_delta_input=False,fuse_mlp_norm=False,fuse_norm_rope=False,fuse_attention=False,fuse_delta_projected=False,fuse_delta_finish=False,fuse_mlp_pipeline=False,fuse_attention_full=False,fuse_terminal_attention=False,fuse_mlp_full=False,fuse_delta_full_log=False,mlp_full_token_cap=87,fuse_terminal_tail=False):
        self.t=transport;self.weights={t['name']:t for t in manifest['tensors']};self.row_cap=row_cap;self.token_cap=token_cap;self.work_cap=work_cap
        self.attention_head_cap=attention_head_cap
        self.delta_head_cap=delta_head_cap
        if compact_lossless and getattr(transport,'wire_codec','') not in ('bf16-exact','bf16-block256-exact-v1'):
            raise ValueError('compact-lossless needs lossless BF16 wire; INT8 blocks must retain their boundaries')
        if arithmetic not in ('f32','int8'):raise ValueError('arithmetic mode')
        self.arithmetic=arithmetic
        self.fuse_delta_full_log=fuse_delta_full_log
        if fuse_delta_full_log and not fuse_delta_projected:raise ValueError("full Delta log requires projected Delta")
        self.fuse_delta_finish=fuse_delta_finish
        if fuse_delta_finish and not fuse_delta_projected:raise ValueError("Delta finish requires projected Delta")
        self.fuse_delta_projected=fuse_delta_projected
        if fuse_delta_projected and (arithmetic!='int8' or not fuse_delta or getattr(transport,'wire_codec','') not in ('bf16-exact','bf16-block256-exact-v1')):raise ValueError('projected Delta requires integer lossless fused Delta')
        self.fuse_delta=fuse_delta;self.wide_mlp=wide_mlp
        if fuse_delta and getattr(transport,"wire_codec","") not in ("bf16-exact","bf16-block256-exact-v1"):raise ValueError("fused delta requires lossless wire")
        self.fuse_delta_input=fuse_delta_input
        if fuse_delta_input and (arithmetic!="int8" or getattr(transport,"wire_codec","") not in ("bf16-exact","bf16-block256-exact-v1")):raise ValueError("fused delta input requires integer lossless path")
        self.terminal_readout=terminal_readout
        if terminal_readout and (not compact_heads or retain_terminal_state):raise ValueError("terminal readout requires compact heads and terminal inference")
        self.compact_heads=compact_heads
        self.retain_terminal_state=retain_terminal_state
        if compact_heads and getattr(transport,'wire_codec','') not in ('bf16-exact','bf16-block256-exact-v1'):raise ValueError('compact heads require lossless BF16 wire')
        if not retain_terminal_state and not compact_heads:raise ValueError('terminal state omission requires compact heads')
        self.fuse_mlp=fuse_mlp
        self.fuse_attention_full=fuse_attention_full
        self.fuse_terminal_tail=fuse_terminal_tail
        if fuse_terminal_tail and not (fuse_terminal_attention and fuse_mlp_full and fuse_add_norm and arithmetic=='int8' and terminal_readout and not retain_terminal_state):
            raise ValueError('terminal tail requires full MLP, terminal attention/readout and exact integer graph')
        self.fuse_terminal_attention=fuse_terminal_attention
        if fuse_terminal_attention and (not fuse_attention_full or not terminal_readout or retain_terminal_state):
            raise ValueError('terminal attention fusion requires full attention and terminal inference')
        if fuse_attention_full and not fuse_attention:raise ValueError("full attention requires fused integer attention")
        self.fuse_attention=fuse_attention
        if fuse_attention and (arithmetic!='int8' or not compact_heads or not fuse_norm_rope or getattr(transport,'wire_codec','') not in ('bf16-exact','bf16-block256-exact-v1')):raise ValueError('attention fusion requires integer, compact heads and lossless norm/RoPE')
        self.fuse_norm_rope=fuse_norm_rope
        if fuse_norm_rope and getattr(transport,"wire_codec","") not in ("bf16-exact","bf16-block256-exact-v1"):raise ValueError("norm/RoPE fusion requires lossless wire")
        if mlp_full_token_cap not in (87,89):raise ValueError("full MLP token cap must be 87 or 89")
        self.mlp_full_token_cap=mlp_full_token_cap
        self.fuse_mlp_full=fuse_mlp_full
        if fuse_mlp_full and not fuse_mlp_pipeline:raise ValueError("full MLP requires MLP pipeline")
        self.fuse_mlp_pipeline=fuse_mlp_pipeline
        if fuse_mlp_pipeline and (arithmetic!='int8' or not fuse_mlp_norm or not fuse_add_norm or not fuse_mlp):raise ValueError('MLP pipeline requires integer, fused MLP/residual norm')
        self.fuse_mlp_norm=fuse_mlp_norm
        if fuse_mlp_norm and (not fuse_mlp or not fuse_add_norm):raise ValueError('MLP norm fusion requires fused MLP and add/norm')
        if fuse_mlp and (arithmetic!='int8' or getattr(transport,'wire_codec','') not in ('bf16-exact','bf16-block256-exact-v1')):raise ValueError('fused MLP requires integer arithmetic and lossless wire')
        self.fuse_add_norm=fuse_add_norm
        if fuse_add_norm and getattr(transport,'wire_codec','') not in ('bf16-exact','bf16-block256-exact-v1'):raise ValueError('fused add/norm requires lossless wire')
        self.compact_lossless=compact_lossless
        self.element_cap=getattr(transport,'max_floats',450000) if compact_lossless else 450000
        self.layers=[]
    # Default empty-history hooks keep ordinary prefill unchanged.
    position_offset=0
    def initial_conv(self,channel,count):return np.zeros((3,count),dtype=np.float32)
    def initial_delta_log(self):return np.empty(0,dtype=np.float32)
    def initial_delta(self,head,heads):return np.zeros((heads,128*128),dtype=np.float32)
    def attention_history(self,k,v,layer):return k,v
    def recorded_hidden(self,hidden,layer):return hidden
    def bounded_heads(self,requested,bf16_values,f32_values=0):
        count=bf16_values+f32_values
        cap=min(requested,getattr(self.t,'max_floats',450000)//count)
        while cap>0:
            size=cap*(2*bf16_values+4*f32_values)+(cap*count+7)//8+16424
            if size<=2000000:return cap
            cap-=1
        raise ValueError('head minimum tile does not fit')
    def linear(self,x,name):
        base=name+'.weight';w=self.weights[base];rows,cols=w['rows'],w['cols']
        a=name+'.lora_A.weight';b=name+'.lora_B.weight'
        if self.arithmetic=='int8':
            rank=self.weights[a]['rows'] if a in self.weights else 0
            return self.t.project_integer(x,rows,cols,rank,base,a if rank else '',b if rank else '',2.,self.row_cap,self.token_cap,self.work_cap)
        if a in self.weights:
            assert b in self.weights
            return self.t.project(x,rows,cols,self.weights[a]['rows'],base,a,b,2.,self.row_cap,self.token_cap,self.work_cap)
        result=np.empty((len(x),rows),dtype=np.float32)
        token_cap=min(self.token_cap,getattr(self.t,"max_floats",450000)//cols)
        for token in range(0,len(x),token_cap):
            chunk=x[token:token+token_cap];n=len(chunk);width=min(self.row_cap,450000//n,120000000//(n*cols))
            for row in range(0,rows,width):
                count=min(width,rows-row);result[token:token+n,row:row+count]=self.t.run('linear_bf16',chunk,[n,count,cols,row],tensor=base).reshape(n,count)
        return result
    def norm(self,x,name=None,factor=None):
        width=x.shape[-1];flat=x.reshape(-1,width);rows=min(self.element_cap//width,262144);result=np.empty_like(flat)
        for start in range(0,len(flat),rows):
            chunk=flat[start:start+rows]
            result[start:start+len(chunk)]=self.t.run('rms_scaled' if factor is not None else 'rms_bf16',chunk,[len(chunk),width],[1e-6,factor] if factor is not None else [1e-6],tensor='' if name is None else name+'.weight').reshape(chunk.shape)
        return result.reshape(x.shape)
    def full_mlp(self,hidden,attention,prefix,layer):
        n=len(hidden);next_norm=PREFIX+f'layers.{layer+1}.input_layernorm.weight' if layer<31 else PREFIX+'norm.weight'
        out=self.t.run('mlp_full_integer',np.concatenate([hidden.ravel(),attention.ravel()]),[n,2560],[2.,1e-6],tensor=prefix+'.post_attention_layernorm.weight',aux=[next_norm])
        if len(out)!=n*5120:raise ValueError('full MLP reply shape')
        return out[:n*2560].reshape(n,2560),out[n*2560:].reshape(n,2560)
    def pipeline_mlp(self,hidden,attention,prefix,layer):
        n=len(hidden);codec=self.t.wire_codec
        try:
            self.t.wire_codec='mlp-down-state-exact-v1'
            kw=dict(tensor=prefix+'.post_attention_layernorm.weight',aux=[PREFIX+f'layers.{layer+1}.input_layernorm.weight'])
            state=self.t.run('mlp_prepare_down',np.concatenate([hidden.ravel(),attention.ravel()]),[n,2560],[2.,1e-6],**kw)
            if len(state)!=n*(2560+9216+100):raise ValueError('prepared MLP reply shape')
            out=self.t.run('mlp_down_norm_prepared',state,[n,2560],[2.,1e-6],**kw)
            if len(out)!=n*5120:raise ValueError('prepared MLP final shape')
            return out[:n*2560].reshape(n,2560),out[n*2560:].reshape(n,2560)
        finally:self.t.wire_codec=codec
    def add_norm(self,a,b,name):
        assert a.shape==b.shape and a.ndim==2
        rows,width=a.shape;cap=min(self.element_cap//(2*width),262144)
        added=np.empty_like(a);normalized=np.empty_like(a)
        for token in range(0,rows,cap):
            n=min(cap,rows-token);out=self.t.run('add_norm_bf16',np.concatenate([a[token:token+n].ravel(),b[token:token+n].ravel()]),[n,width],[1e-6],tensor=name+'.weight')
            count=n*width;added[token:token+n]=out[:count].reshape(n,width);normalized[token:token+n]=out[count:].reshape(n,width)
        return added,normalized
    def pair(self,op,a,b):
        assert a.shape==b.shape;flat=a.ravel();other=b.ravel();result=np.empty_like(flat);cap=self.element_cap//2
        for start in range(0,len(flat),cap):
            n=min(cap,len(flat)-start);result[start:start+n]=self.t.run(op,np.concatenate([flat[start:start+n],other[start:start+n]]),[n])
        return result.reshape(a.shape)
    def add_norm_chain(self,a,b,c,name):
        if a.shape!=b.shape or a.shape!=c.shape or a.ndim!=2:raise ValueError('residual chain shape')
        rows,width=a.shape;cap=min(self.element_cap//(3*width),512)
        if cap<1:raise ValueError('residual chain minimum tile')
        added=np.empty_like(a);normalized=np.empty_like(a)
        for token in range(0,rows,cap):
            n=min(cap,rows-token)
            values=np.concatenate([v[token:token+n].ravel() for v in (a,b,c)])
            out=self.t.run('add_norm_chain_bf16',values,[n,width],[1e-6],tensor=name+'.weight')
            count=n*width;added[token:token+n]=out[:count].reshape(n,width);normalized[token:token+n]=out[count:].reshape(n,width)
        return added,normalized
    def convolution(self,x,name):
        n,channels=x.shape;result=np.empty_like(x);window=np.empty((3,channels),dtype=np.float32)
        channel_cap=min(4096,self.element_cap//(min(n,self.token_cap)+3)) if self.compact_lossless else 1024
        if channel_cap<1:raise ValueError('convolution minimum tile does not fit')
        for channel in range(0,channels,channel_cap):
            count=min(channel_cap,channels-channel);state=self.initial_conv(channel,count);token_cap=min(self.token_cap,self.element_cap//count-3)
            for token in range(0,n,token_cap):
                chunk=x[token:token+token_cap,channel:channel+count];out=self.t.run('conv_state',np.concatenate([state.ravel(),chunk.ravel()]),[len(chunk),count,4,channel],tensor=name+'.weight')
                result[token:token+len(chunk),channel:channel+count]=out[:len(chunk)*count].reshape(chunk.shape);state=out[len(chunk)*count:].reshape(3,count)
            window[:,channel:channel+count]=state
        return result,window
    def delta_projected(self,x,name,layer):
        n=len(x)
        if not 1<=n<=132:raise ValueError('projected Delta token bound')
        # The first head group prepares quantized input, original QKV/Z A
        # products and both full gate arrays. Later groups reuse that state.
        heads_cap=0
        for candidate in range(16,1,-2):
            count=n*candidate*128;history=3*candidate*256;state=candidate*16384*int(self.retain_terminal_state)
            floats=count+history+state+n*2762
            bitmap=(floats+7)//8 if self.t.wire_codec=='bf16-exact' else ((floats+255)//256+7)//8
            bound=16424+bitmap+2*(count+history)+4*state+n*(2560*2+202*4)
            if floats<=900000 and bound<=2000000:heads_cap=candidate;break
        if not heads_cap:raise ValueError('projected Delta capture reply bound')
        result=np.empty((n,32,128),np.float32);conv=np.empty((3,8192),np.float32)
        states=np.empty((32,128,128),np.float32) if self.retain_terminal_state else None
        saved=None;projected=None
        finish_values=n*(2762+2048)+3*16*256+16*16384
        finish_bitmap=finish_values//8+(finish_values%8!=0) if self.t.wire_codec=='bf16-exact' else ((finish_values+255)//256+7)//8
        finish_bound=16424+finish_bitmap+2*(n*(2560+2048)+3*16*256)+4*(n*202+16*16384)
        finish_enabled=self.fuse_delta_finish and n<=90 and heads_cap==16 and finish_bound<=2000000
        for first in range(0,32,heads_cap):
            heads=min(heads_cap,32-first);channels=heads*256
            segments=[(first//2*128,heads//2*128),(2048+first//2*128,heads//2*128),(4096+first*128,heads*128)]
            indices=np.concatenate([np.arange(start,start+width) for start,width in segments])
            history=np.concatenate([self.initial_conv(start,width) for start,width in segments],axis=1)
            state=self.initial_delta(first,heads)
            payload=np.concatenate([x.ravel() if saved is None else saved,history.ravel(),state.ravel()])
            finishing=finish_enabled and first==16
            if finishing:payload=np.concatenate([payload,result[:,:16].copy().ravel()])
            out=self.t.run('delta_project_finish' if finishing else 'delta_project_capture' if saved is None else 'delta_project_reuse',payload,[n,heads,first,int(self.retain_terminal_state)],tensor=name+'.in_proj_qkv.weight')
            count=n*2560 if finishing else n*heads*128;tail=count+3*channels+heads*16384*int(self.retain_terminal_state)
            expected=tail+(n*2762 if saved is None else 0)
            if len(out)!=expected:raise ValueError('projected Delta reply shape')
            if finishing:projected=out[:count].reshape(n,2560)
            else:result[:,first:first+heads]=out[:count].reshape(n,heads,128)
            conv[:,indices]=out[count:count+3*channels].reshape(3,channels)
            if states is not None:states[first:first+heads]=out[count+3*channels:tail].reshape(heads,128,128)
            if saved is None:saved=out[tail:]
        path=self.t.directory/'states';path.mkdir(exist_ok=True)
        np.savez(path/f'layer-{layer:02d}.npz',**(dict(conv=conv,delta=states) if states is not None else dict(conv=conv)))
        return projected if projected is not None else self.linear(result.reshape(n,4096),name+'.out_proj')
    def delta_full_log(self,x,name,layer):
        n=len(x);log=self.initial_delta_log()
        if log.ndim!=1 or log.dtype!=np.float32 or log.size%6176:raise ValueError('Delta log layout')
        p=log.size//6176
        if not 1<=n<=90 or p>132 or (self.retain_terminal_state and p):raise ValueError('full Delta log token/history bounds')
        history=self.initial_conv(0,8192)
        payload=np.concatenate([x.ravel(),history.ravel(),log])
        out=self.t.run('delta_full_log_integer',payload,[n,32,p,int(self.retain_terminal_state)],tensor=name+'.in_proj_qkv.weight')
        expected=n*2560+3*8192+(n*6176 if self.retain_terminal_state else 0)
        if len(out)!=expected:raise ValueError('full Delta log reply shape')
        result=out[:n*2560].reshape(n,2560);conv=out[n*2560:n*2560+3*8192].reshape(3,8192)
        state=dict(conv=conv)
        if self.retain_terminal_state:state['delta_log']=out[n*2560+3*8192:].copy()
        path=self.t.directory/'states';path.mkdir(exist_ok=True);np.savez(path/f'layer-{layer:02d}.npz',**state)
        return result
    def delta(self,x,name,layer):
        if self.fuse_delta_full_log and delta_full_log_enabled(True,len(x),self.t.wire_codec,self.retain_terminal_state,2 if self.initial_delta_log().size else None):
            return self.delta_full_log(x,name,layer)
        if self.fuse_delta_projected:return self.delta_projected(x,name,layer)
        n=len(x)
        combine=self.fuse_delta_input and n<=min(96,self.token_cap) and self.row_cap>=8192 and self.work_cap>=n*(8192*2560+64*(8192+2560))
        if combine:
            values=self.t.run('delta_input_integer',x,[n],tensor=name+'.in_proj_qkv.weight')
            mixed=values[:n*8192].reshape(n,8192);g=values[n*8192:n*(8192+32)].reshape(n,32);beta=values[n*(8192+32):].reshape(n,32)
            z=self.linear(x,name+'.in_proj_z').reshape(n,32,128)
        else:
            n=len(x);mixed=self.linear(x,name+'.in_proj_qkv');z=self.linear(x,name+'.in_proj_z').reshape(n,32,128)
            if self.fuse_delta and self.arithmetic=='int8':
                g=np.empty((n,32),np.float32);beta=np.empty_like(g)
                cap=min(self.token_cap,256,self.element_cap//2560)
                for token in range(0,n,cap):
                    count=min(cap,n-token)
                    gates=self.t.run('delta_gates_integer',x[token:token+count],[count],tensor=name+'.in_proj_a.weight')
                    g[token:token+count]=gates[:count*32].reshape(count,32);beta[token:token+count]=gates[count*32:].reshape(count,32)
            else:
                a=self.linear(x,name+'.in_proj_a');b=self.linear(x,name+'.in_proj_b')
                gates=self.t.run('delta_gates',np.concatenate([a.ravel(),b.ravel()]),[n,32],tensor=name+'.A_log',aux=[name+'.dt_bias']);g=gates[:n*32].reshape(n,32);beta=gates[n*32:].reshape(n,32)
        if self.fuse_delta:return self.delta_stage(mixed,z,g,beta,name,layer)
        convolved,conv_state=self.convolution(mixed,name+'.conv1d')
        q=self.norm(convolved[:,:2048].reshape(n,16,128),factor=1/128);k=self.norm(convolved[:,2048:4096].reshape(n,16,128),factor=128**-0.5);v=convolved[:,4096:].reshape(n,32,128)
        result=np.empty((n,32,128),dtype=np.float32);states=np.empty((32,128,128),dtype=np.float32) if self.retain_terminal_state else None
        if self.delta_head_cap>1 or self.compact_heads:
            head_cap=self.bounded_heads(self.delta_head_cap,min(n,self.token_cap,512)*384,min(n,self.token_cap,512)*2+16384)
            for head in range(0,32,head_cap):
                heads=min(head_cap,32-head);state=self.initial_delta(head,heads)
                for token in range(0,n,min(self.token_cap,512)):
                    count=min(self.token_cap,512,n-token)
                    activ=[np.concatenate([q[token:token+count,h//2].ravel(),k[token:token+count,h//2].ravel(),v[token:token+count,h].ravel()]) for h in range(head,head+heads)]
                    tails=[np.concatenate([g[token:token+count,h],beta[token:token+count,h],state[h-head]]) for h in range(head,head+heads)]
                    terminal=self.compact_heads and not self.retain_terminal_state and token+count==n
                    output=self.t.run('delta_terminal_bf16' if terminal else 'delta_heads_bf16',np.concatenate(activ+tails),[count,128,128,heads])
                    result[token:token+count,head:head+heads]=output[:heads*count*128].reshape(heads,count,128).transpose(1,0,2)
                    if not terminal:state=output[heads*count*128:].reshape(heads,128*128)
                if self.retain_terminal_state:states[head:head+heads]=state.reshape(heads,128,128)
        else:
            for head in range(32):
                state=self.initial_delta(head,1)[0];cap=min(self.token_cap,512)
                for token in range(0,n,cap):
                    count=min(cap,n-token);inputs=np.concatenate([q[token:token+count,head//2].ravel(),k[token:token+count,head//2].ravel(),v[token:token+count,head].ravel(),g[token:token+count,head],beta[token:token+count,head],state])
                    output=self.t.run('delta_bf16',inputs,[count,128,128]);result[token:token+count,head]=output[:count*128].reshape(count,128);state=output[count*128:]
                states[head]=state.reshape(128,128)
        path=self.t.directory/'states';path.mkdir(exist_ok=True);np.savez(path/f'layer-{layer:02d}.npz',**(dict(conv=conv_state,delta=states) if self.retain_terminal_state else dict(conv=conv_state)))
        out=np.empty_like(result);cap=self.element_cap//(2*32*128)
        for token in range(0,n,cap):
            count=min(cap,n-token);out[token:token+count]=self.t.run('gated_norm',np.concatenate([result[token:token+count].ravel(),z[token:token+count].ravel()]),[count*32,128],[1e-6],tensor=name+'.norm.weight').reshape(count,32,128)
        return self.linear(out.reshape(n,4096),name+'.out_proj')
    def delta_stage(self,mixed,z,g,beta,name,layer):
        n=len(mixed);cap=min(n,self.token_cap,512)
        heads_cap=self.bounded_heads(max(2,self.delta_head_cap),cap*384+768,cap*2+16384)//2*2
        if heads_cap<2:raise ValueError('fused delta minimum tile does not fit')
        result=np.empty((n,32,128),dtype=np.float32);conv=np.empty((3,8192),dtype=np.float32)
        states=np.empty((32,128,128),dtype=np.float32) if self.retain_terminal_state else None
        for head in range(0,32,heads_cap):
            heads=min(heads_cap,32-head)
            segments=[(head//2*128,heads//2*128),(2048+head//2*128,heads//2*128),(4096+head*128,heads*128)]
            indices=np.concatenate([np.arange(start,start+width) for start,width in segments])
            history=np.concatenate([self.initial_conv(start,width) for start,width in segments],axis=1)
            state=self.initial_delta(head,heads)
            for token in range(0,n,cap):
                count=min(cap,n-token);keep=self.retain_terminal_state or token+count<n
                window=np.concatenate([history,mixed[token:token+count,indices]])
                payload=np.concatenate([window.ravel(),z[token:token+count,head:head+heads].ravel(),g[token:token+count,head:head+heads].ravel(),beta[token:token+count,head:head+heads].ravel(),state.ravel()])
                output=self.t.run('delta_stage_bf16',payload,[count,heads,head,int(keep)],tensor=name+'.conv1d.weight',aux=[name+'.norm.weight'])
                size=count*heads*128
                result[token:token+count,head:head+heads]=output[:size].reshape(count,heads,128)
                if keep:state=output[size:].reshape(heads,128*128)
                history=window[-3:].copy()
            conv[:,indices]=history
            if self.retain_terminal_state:states[head:head+heads]=state.reshape(heads,128,128)
        path=self.t.directory/'states';path.mkdir(exist_ok=True)
        np.savez(path/f'layer-{layer:02d}.npz',**(dict(conv=conv,delta=states) if self.retain_terminal_state else dict(conv=conv)))
        return self.linear(result.reshape(n,4096),name+'.out_proj')
    def terminal_tail(self,hidden,attention):
        n=len(hidden)
        if not 1<=n<=87 or hidden.shape!=(n,2560) or attention.shape!=hidden.shape:raise ValueError('terminal tail shape')
        empty=np.empty((0,4,256),np.float32);prior_k,prior_v=self.attention_history(empty,empty,31)
        if prior_k.shape!=prior_v.shape or prior_k.shape!=(self.position_offset,4,256):raise ValueError('terminal tail prefix shape')
        payload=np.concatenate([hidden.ravel(),attention.ravel(),prior_k.transpose(1,0,2).ravel(),prior_v.transpose(1,0,2).ravel()])
        out=self.t.run('terminal_tail_integer',payload,[n,self.position_offset],tensor=PREFIX+'layers.30.post_attention_layernorm.weight')
        count=n*2560;offset=count+5120
        if len(out)!=offset+n*2048:raise ValueError('terminal tail reply shape')
        k=out[offset:offset+n*1024].reshape(n,4,256);v=out[offset+n*1024:].reshape(n,4,256)
        all_k=np.concatenate([prior_k,k]);all_v=np.concatenate([prior_v,v]);path=self.t.directory/'states';path.mkdir(exist_ok=True)
        np.savez(path/'layer-31.npz',keys=all_k,values=all_v,positions=np.arange(len(all_k),dtype=np.int32))
        return out[:count].reshape(n,2560),out[count:count+2560].reshape(1,2560),out[count+2560:offset].reshape(1,2560)
    def terminal_attention_mlp(self,x,residual,name,layer):
        if layer!=31 or residual.shape!=(1,2560):raise ValueError('terminal attention residual/layer')
        n=len(x);empty=np.empty((0,4,256),np.float32)
        prior_k,prior_v=self.attention_history(empty,empty,layer)
        if prior_k.shape!=prior_v.shape or prior_k.shape!=(self.position_offset,4,256):raise ValueError('terminal attention prefix shape')
        payload=np.concatenate([x.ravel(),residual.ravel(),prior_k.transpose(1,0,2).ravel(),prior_v.transpose(1,0,2).ravel()])
        out=self.t.run('terminal_attention_mlp_integer',payload,[n,self.position_offset],tensor=name+'.q_proj.weight')
        if len(out)!=5120+n*2048:raise ValueError('terminal attention reply shape')
        k=out[5120:5120+n*1024].reshape(n,4,256);v=out[5120+n*1024:].reshape(n,4,256)
        all_k=np.concatenate([prior_k,k]);all_v=np.concatenate([prior_v,v])
        path=self.t.directory/'states';path.mkdir(exist_ok=True)
        np.savez(path/f'layer-{layer:02d}.npz',keys=all_k,values=all_v,positions=np.arange(len(all_k),dtype=np.int32))
        return out[:2560].reshape(1,2560),out[2560:5120].reshape(1,2560)
    def attention_full(self,x,name,layer):
        n=len(x);qn=1 if self.terminal_readout and layer==31 else n
        empty=np.empty((0,4,256),np.float32)
        prior_k,prior_v=self.attention_history(empty,empty,layer)
        if prior_k.shape!=prior_v.shape or prior_k.shape!=(self.position_offset,4,256):raise ValueError('full attention prefix shape')
        payload=np.concatenate([x.ravel(),prior_k.transpose(1,0,2).ravel(),prior_v.transpose(1,0,2).ravel()])
        out=self.t.run('attention_full_integer',payload,[n,self.position_offset,int(qn==1 and self.terminal_readout and layer==31)],tensor=name+'.q_proj.weight')
        count=qn*2560
        if len(out)!=count+n*2048:raise ValueError('full attention reply shape')
        k=out[count:count+n*1024].reshape(n,4,256);v=out[count+n*1024:].reshape(n,4,256)
        all_k=np.concatenate([prior_k,k]);all_v=np.concatenate([prior_v,v]);total=len(all_k)
        path=self.t.directory/'states';path.mkdir(exist_ok=True)
        np.savez(path/f'layer-{layer:02d}.npz',keys=all_k,values=all_v,positions=np.arange(total,dtype=np.int32))
        return out[:count].reshape(qn,2560)
    def attention_fused(self,x,name,layer):
        n=len(x);qn=1 if self.terminal_readout and layer==31 else n
        if not 1<=n<=132:raise ValueError('fused attention token bound')
        kv=self.t.run('attention_kv_integer',x,[n,self.position_offset],tensor=name+'.k_proj.weight')
        if len(kv)!=n*2048:raise ValueError('fused KV reply shape')
        k=kv[:n*1024].reshape(n,4,256);v=kv[n*1024:].reshape(n,4,256)
        all_k,all_v=self.attention_history(k,v,layer);total=len(all_k)
        if not qn<=total<=512:raise ValueError('fused attention history bound')
        out=np.empty((qn,16,256),np.float32)
        # Full Q/GQA is affordable through 89 tokens; longer sequences use
        # whole GQA groups. No attention context or projection rows are dropped.
        heads_cap=16 if qn<=89 else 8
        for first in range(0,16,heads_cap):
            heads=min(heads_cap,16-first);group=first//4;groups=heads//4
            payload=np.concatenate([x[-qn:].ravel(),all_k[:,group:group+groups].transpose(1,0,2).ravel(),all_v[:,group:group+groups].transpose(1,0,2).ravel()])
            output=self.t.run('attention_q_gqa_integer',payload,[qn,total,total-qn,first,heads],tensor=name+'.q_proj.weight')
            if len(output)!=qn*heads*256:raise ValueError('fused Q/GQA reply shape')
            out[:,first:first+heads]=output.reshape(qn,heads,256)
        path=self.t.directory/'states';path.mkdir(exist_ok=True)
        np.savez(path/f'layer-{layer:02d}.npz',keys=all_k,values=all_v,positions=np.arange(total,dtype=np.int32))
        return self.linear(out.reshape(qn,4096),name+'.o_proj')
    def attention(self,x,name,layer):
        if self.fuse_attention_full and (len(x)<=89 or self.terminal_readout and layer==31):return self.attention_full(x,name,layer)
        if self.fuse_attention:return self.attention_fused(x,name,layer)
        n=len(x);qn=1 if self.terminal_readout and layer==31 else n
        qp=self.linear(x[-qn:],name+'.q_proj').reshape(qn,16,512);q=qp[:,:,:256].copy();gate=qp[:,:,256:].reshape(qn,4096)
        k=self.linear(x,name+'.k_proj').reshape(n,4,256);v=self.linear(x,name+'.v_proj').reshape(n,4,256)
        if not self.fuse_norm_rope:q=self.norm(q,name+'.q_norm');k=self.norm(k,name+'.k_norm')
        for array,heads,norm_name in [(q,16,name+".q_norm"),(k,4,name+".k_norm")]:
            count_tokens=len(array);offset=self.position_offset+n-count_tokens
            if self.attention_head_cap>1 or self.fuse_norm_rope:
                cap=self.bounded_heads(heads,count_tokens*256)
                for first in range(0,heads,cap):
                    count=min(cap,heads-first)
                    array[:,first:first+count]=self.t.run('norm_rope_heads_bf16' if self.fuse_norm_rope else 'rope_heads',array[:,first:first+count].transpose(1,0,2),[count_tokens,256,64,offset,count],[1e-6,10000000.] if self.fuse_norm_rope else [10000000.],**(dict(tensor=norm_name+'.weight') if self.fuse_norm_rope else {})).reshape(count,count_tokens,256).transpose(1,0,2)
            else:
                for head in range(heads):array[:,head]=self.t.run('rope',array[:,head],[count_tokens,256,64,offset],[10000000.]).reshape(count_tokens,256)
        all_k,all_v=self.attention_history(k,v,layer);total=len(all_k)
        # Existing causal kernel keeps absolute query grouping and summation order.
        all_q=q if self.terminal_readout else (np.concatenate([np.zeros((total-n,16,256),dtype=np.float32),q]) if total!=n else q)
        out=np.empty((qn,16,256),dtype=np.float32)
        if total>512:raise ValueError('attention prefill currently bounded to 512 tokens')
        if self.compact_heads:
            # Whole GQA groups preserve the 4:1 query/KV mapping, including tails.
            cap=0
            for candidate in (16,12,8,4,2,1):
                query_tokens=qn if self.terminal_readout else total
                values=(candidate*query_tokens+2*((candidate+3)//4)*total)*256
                if values<=self.element_cap and values*2+(values+7)//8+16424<=2000000 and candidate*(total*(total+1)//2-(total-query_tokens)*(total-query_tokens+1)//2)*256<=75000000:
                    cap=candidate;break
            if not cap:raise ValueError('GQA minimum group does not fit')
            for head in range(0,16,cap):
                heads=min(cap,16-head);first=head//4;groups=(heads+3)//4
                inputs=np.concatenate([all_q[:,head:head+heads].transpose(1,0,2).ravel(),all_k[:,first:first+groups].transpose(1,0,2).ravel(),all_v[:,first:first+groups].transpose(1,0,2).ravel()])
                out[:,head:head+heads]=self.t.run('gqa_suffix_bf16' if self.terminal_readout else 'gqa_heads_bf16',inputs,[qn,256,heads,total-qn] if self.terminal_readout else [total,256,heads]).reshape(heads,qn if self.terminal_readout else total,256).transpose(1,0,2)[-qn:]
        elif self.attention_head_cap>1:
            head_cap=self.bounded_heads(self.attention_head_cap,total*768)
            for head in range(0,16,head_cap):
                heads=min(head_cap,16-head)
                inputs=np.concatenate([np.concatenate([all_q[:,h].ravel(),all_k[:,h//4].ravel(),all_v[:,h//4].ravel()]) for h in range(head,head+heads)])
                out[:,head:head+heads]=self.t.run('attention_heads_bf16',inputs,[total,256,heads]).reshape(heads,total,256).transpose(1,0,2)[-n:]
        else:
            for head in range(16):out[:,head]=self.t.run('attention_bf16',np.concatenate([all_q[:,head].ravel(),all_k[:,head//4].ravel(),all_v[:,head//4].ravel()]),[total,256]).reshape(total,256)[-n:]
        path=self.t.directory/'states';path.mkdir(exist_ok=True);np.savez(path/f'layer-{layer:02d}.npz',keys=all_k,values=all_v,positions=np.arange(total,dtype=np.int32))
        return self.linear(self.pair('attention_gate',out.reshape(qn,4096),gate),name+'.o_proj')
    def forward(self,token_ids,layers=32):
        if not 1<=len(token_ids)<=512:raise ValueError('text prefill needs 1..512 tokens')
        prefix_started=getattr(self,'fuse_prefix_start',False)
        if prefix_started:
            start_clock=time.perf_counter()
            hidden,first_attention=self.prefix_start(token_ids)
        else:
            embedding=PREFIX+'embed_tokens.weight';parts=[]
            for start in range(0,len(token_ids),128):
                ids=token_ids[start:start+128];parts.append(self.t.run('embed',ids,[len(ids),2560],tensor=embedding).reshape(len(ids),2560))
            hidden=np.concatenate(parts,axis=0)
        pre_normalized=None;tail_done=False
        for layer in range(layers):
            self.current_layer=layer
            if layer==31 and tail_done:
                np.save(self.t.directory/'layer-31.npy',hidden)
                record=dict(layer=31,kind='full_attention',queries=0,instructions=0,candid_bytes=0,wall_seconds=0.,replayed=0,included_in_layer=30)
                self.layers.append(record);atomic(self.t.directory/'layers.json',(json.dumps(self.layers,indent=2)+'\n').encode());print(json.dumps(record),flush=True)
                continue
            layer_hidden=None
            prefix=PREFIX+f'layers.{layer}';start=len(self.t.measurements);clock=time.perf_counter()
            if prefix_started and layer==0:start=0;clock=start_clock
            normalized=None if prefix_started and layer==0 else (pre_normalized if pre_normalized is not None else self.norm(hidden,prefix+'.input_layernorm'))
            terminal_fused=self.fuse_terminal_attention and self.terminal_readout and layer==31
            if prefix_started and layer==0:attention=first_attention
            elif terminal_fused:
                hidden,pre_normalized=self.terminal_attention_mlp(normalized,hidden[-1:],prefix+'.self_attn',layer)
            else:
                attention=self.attention(normalized,prefix+'.self_attn',layer) if (layer+1)%4==0 else self.delta(normalized,prefix+'.linear_attn',layer)
            if self.terminal_readout and layer==31:hidden=hidden[-1:]
            tail_active=self.fuse_terminal_tail and layers==32 and layer==30 and len(hidden)<=87
            if tail_active:
                layer_hidden,hidden,pre_normalized=self.terminal_tail(hidden,attention);tail_done=True
            elif terminal_fused:pass
            elif self.terminal_readout and layer==31 and self.arithmetic=='int8':
                values=self.t.run('terminal_mlp_integer',np.concatenate([hidden.ravel(),attention.ravel()]),[1,2560],tensor=prefix+'.post_attention_layernorm.weight')
                hidden=values[:2560].reshape(1,2560);pre_normalized=values[2560:].reshape(1,2560)
            elif self.fuse_mlp_full and layers==32 and len(hidden)<=self.mlp_full_token_cap:
                hidden,pre_normalized=self.full_mlp(hidden,attention,prefix,layer)
            elif self.fuse_mlp_pipeline and layers==32 and layer<31 and len(hidden)<=89:
                hidden,pre_normalized=self.pipeline_mlp(hidden,attention,prefix,layer)
            else:
                norm_fused=self.fuse_mlp_norm and 3*hidden.size<=self.element_cap and (layer+1<layers or layers==32)
                if norm_fused:normalized=hidden
                elif self.fuse_add_norm:hidden,normalized=self.add_norm(hidden,attention,prefix+'.post_attention_layernorm')
                else:
                    hidden=self.pair('add_bf16',hidden,attention)
                    normalized=self.norm(hidden,prefix+'.post_attention_layernorm')
                if self.fuse_mlp:
                    gate_name=prefix+'.mlp.gate_proj';up_name=prefix+'.mlp.up_proj';shape=self.weights[gate_name+'.weight'];rank=max(self.weights[gate_name+'.lora_A.weight']['rows'],self.weights[up_name+'.lora_A.weight']['rows'])
                    product=self.t.project_mlp_integer(normalized,shape['rows'],shape['cols'],rank,gate_name+'.weight',up_name+'.weight',2.,self.row_cap,self.token_cap,min(self.work_cap*2,4500000000 if self.wide_mlp and len(normalized)<=89 else 4000000000),**(dict(add_norm=(attention,prefix+'.post_attention_layernorm.weight')) if norm_fused else {}))
                else:
                    gate=self.linear(normalized,prefix+'.mlp.gate_proj');up=self.linear(normalized,prefix+'.mlp.up_proj');product=self.pair('swiglu_bf16',gate,up)
                mlp=self.linear(product,prefix+'.mlp.down_proj')
                if self.fuse_add_norm and (layer+1<layers or layers==32):
                    norm_name=PREFIX+f'layers.{layer+1}.input_layernorm' if layer+1<layers else PREFIX+'norm'
                    if norm_fused:hidden,pre_normalized=self.add_norm_chain(hidden,attention,mlp,norm_name)
                    else:hidden,pre_normalized=self.add_norm(hidden,mlp,norm_name)
                else:hidden=self.pair('add_bf16',hidden,mlp)
            recorded=layer_hidden if tail_active else hidden
            np.save(self.t.directory/f'layer-{layer:02d}.npy',recorded if self.terminal_readout and layer==31 else self.recorded_hidden(recorded,layer))
            queries=self.t.measurements[start:];record=dict(layer=layer,kind='full_attention' if (layer+1)%4==0 else 'delta',queries=len(queries),instructions=sum(q['ok']['instructions'] for q in queries),candid_bytes=sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in queries),wall_seconds=time.perf_counter()-clock,replayed=sum(bool(q.get('replayed')) for q in queries))
            self.layers.append(record);atomic(self.t.directory/'layers.json',(json.dumps(self.layers,indent=2)+'\n').encode());print(json.dumps(record),flush=True)
        return (pre_normalized if self.fuse_add_norm else self.norm(hidden,PREFIX+'norm')) if layers==32 else hidden
