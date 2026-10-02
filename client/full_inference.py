"""Client-held text-only Qwen3.5/LoRA graph. All numerical kernels run via query.

NumPy is used only for routing, reshape, concatenation, zero states and reporting.
An official token-ID fixture is input; no reference hidden state enters inference.
"""
import hashlib,json,pathlib,time
import numpy as np
from transport import Transport,encode,decode,atomic

PREFIX='model.language_model.'
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
    def run(self,op,values,dims=(),scalars=(),tensor='',input_hash=None,aux=()):
        input_hash=self.input_hash if input_hash is None else input_hash
        h=dict(version=1,model=self.model,pack_hash=self.pack_hash,input_hash=input_hash,step=self.index,op=op,tensor=tensor,dims=list(dims),scalars=list(scalars))
        if aux:h['aux']=list(aux)
        if getattr(self,'wire_codec',''):h['encoding']=self.wire_codec
        request=self.directory/f'{self.index:06d}.request.bin';response=self.directory/f'{self.index:06d}.response.bin';metric=self.directory/f'{self.index:06d}.metric.json'
        payload=encode(h,values)
        if request.exists() and request.read_bytes()!=payload:raise ValueError(f'checkpoint input mismatch at query {self.index}; use a fresh directory after changing the graph')
        if response.exists() and metric.exists():
            rh,v=decode(response.read_bytes())
            if rh['step']!=self.index+1 or any(rh[k]!=value for k,value in h.items() if k not in ('step','scalars')) or not np.array_equal(np.asarray(rh['scalars'],dtype=np.float32),np.asarray(h['scalars'],dtype=np.float32)):raise ValueError('checkpoint reply identity')
            saved=json.loads(metric.read_text());saved['replayed']=True;self.measurements.append(saved);self.index+=1;self.replayed+=1;return v
        start=time.perf_counter()
        try:values=super().run(op,values,dims,scalars,tensor,input_hash,aux)
        except Exception as error:
            with (self.directory/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(index=self.index,op=op,tensor=tensor,dims=list(dims),error=str(error),wall_seconds=time.perf_counter()-start))+'\n')
            raise
        atomic(metric,(json.dumps(self.measurements[-1],indent=2)+'\n').encode());return values

class TextGraph:
    def __init__(self,transport,manifest,*,row_cap=768,token_cap=132,work_cap=300_000_000,delta_head_cap=1,attention_head_cap=1,compact_lossless=False,arithmetic='f32',fuse_add_norm=False,fuse_mlp=False):
        self.t=transport;self.weights={t['name']:t for t in manifest['tensors']};self.row_cap=row_cap;self.token_cap=token_cap;self.work_cap=work_cap
        self.attention_head_cap=attention_head_cap
        self.delta_head_cap=delta_head_cap
        if compact_lossless and getattr(transport,'wire_codec','')!='bf16-exact':
            raise ValueError('compact-lossless needs lossless BF16 wire; INT8 blocks must retain their boundaries')
        if arithmetic not in ('f32','int8'):raise ValueError('arithmetic mode')
        self.arithmetic=arithmetic
        self.fuse_mlp=fuse_mlp
        if fuse_mlp and (arithmetic!='int8' or getattr(transport,'wire_codec','')!='bf16-exact'):raise ValueError('fused MLP requires integer arithmetic and lossless wire')
        self.fuse_add_norm=fuse_add_norm
        if fuse_add_norm and getattr(transport,'wire_codec','')!='bf16-exact':raise ValueError('fused add/norm requires lossless wire')
        self.compact_lossless=compact_lossless
        self.element_cap=getattr(transport,'max_floats',450000) if compact_lossless else 450000
        self.layers=[]
    # Default empty-history hooks keep ordinary prefill unchanged.
    position_offset=0
    def initial_conv(self,channel,count):return np.zeros((3,count),dtype=np.float32)
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
    def delta(self,x,name,layer):
        n=len(x);mixed=self.linear(x,name+'.in_proj_qkv');z=self.linear(x,name+'.in_proj_z').reshape(n,32,128)
        a=self.linear(x,name+'.in_proj_a');b=self.linear(x,name+'.in_proj_b')
        gates=self.t.run('delta_gates',np.concatenate([a.ravel(),b.ravel()]),[n,32],tensor=name+'.A_log',aux=[name+'.dt_bias']);g=gates[:n*32].reshape(n,32);beta=gates[n*32:].reshape(n,32)
        convolved,conv_state=self.convolution(mixed,name+'.conv1d')
        q=self.norm(convolved[:,:2048].reshape(n,16,128),factor=1/128);k=self.norm(convolved[:,2048:4096].reshape(n,16,128),factor=128**-0.5);v=convolved[:,4096:].reshape(n,32,128)
        result=np.empty((n,32,128),dtype=np.float32);states=np.empty((32,128,128),dtype=np.float32)
        if self.delta_head_cap>1:
            head_cap=self.bounded_heads(self.delta_head_cap,min(n,self.token_cap,512)*384,min(n,self.token_cap,512)*2+16384)
            for head in range(0,32,head_cap):
                heads=min(head_cap,32-head);state=self.initial_delta(head,heads)
                for token in range(0,n,min(self.token_cap,512)):
                    count=min(self.token_cap,512,n-token)
                    activ=[np.concatenate([q[token:token+count,h//2].ravel(),k[token:token+count,h//2].ravel(),v[token:token+count,h].ravel()]) for h in range(head,head+heads)]
                    tails=[np.concatenate([g[token:token+count,h],beta[token:token+count,h],state[h-head]]) for h in range(head,head+heads)]
                    output=self.t.run('delta_heads_bf16',np.concatenate(activ+tails),[count,128,128,heads])
                    result[token:token+count,head:head+heads]=output[:heads*count*128].reshape(heads,count,128).transpose(1,0,2)
                    state=output[heads*count*128:].reshape(heads,128*128)
                states[head:head+heads]=state.reshape(heads,128,128)
        else:
            for head in range(32):
                state=self.initial_delta(head,1)[0];cap=min(self.token_cap,512)
                for token in range(0,n,cap):
                    count=min(cap,n-token);inputs=np.concatenate([q[token:token+count,head//2].ravel(),k[token:token+count,head//2].ravel(),v[token:token+count,head].ravel(),g[token:token+count,head],beta[token:token+count,head],state])
                    output=self.t.run('delta_bf16',inputs,[count,128,128]);result[token:token+count,head]=output[:count*128].reshape(count,128);state=output[count*128:]
                states[head]=state.reshape(128,128)
        path=self.t.directory/'states';path.mkdir(exist_ok=True);np.savez(path/f'layer-{layer:02d}.npz',conv=conv_state,delta=states)
        out=np.empty_like(result);cap=self.element_cap//(2*32*128)
        for token in range(0,n,cap):
            count=min(cap,n-token);out[token:token+count]=self.t.run('gated_norm',np.concatenate([result[token:token+count].ravel(),z[token:token+count].ravel()]),[count*32,128],[1e-6],tensor=name+'.norm.weight').reshape(count,32,128)
        return self.linear(out.reshape(n,4096),name+'.out_proj')
    def attention(self,x,name,layer):
        n=len(x);qp=self.linear(x,name+'.q_proj').reshape(n,16,512);q=self.norm(qp[:,:,:256],name+'.q_norm');gate=qp[:,:,256:].reshape(n,4096)
        k=self.norm(self.linear(x,name+'.k_proj').reshape(n,4,256),name+'.k_norm');v=self.linear(x,name+'.v_proj').reshape(n,4,256)
        for array,heads in [(q,16),(k,4)]:
            if self.attention_head_cap>1:
                cap=self.bounded_heads(heads,n*256)
                for first in range(0,heads,cap):
                    count=min(cap,heads-first)
                    array[:,first:first+count]=self.t.run('rope_heads',array[:,first:first+count].transpose(1,0,2),[n,256,64,self.position_offset,count],[10000000.]).reshape(count,n,256).transpose(1,0,2)
            else:
                for head in range(heads):array[:,head]=self.t.run('rope',array[:,head],[n,256,64,self.position_offset],[10000000.]).reshape(n,256)
        all_k,all_v=self.attention_history(k,v,layer);total=len(all_k)
        # Existing causal kernel keeps absolute query grouping and summation order.
        all_q=np.concatenate([np.zeros((total-n,16,256),dtype=np.float32),q]) if total!=n else q
        out=np.empty((n,16,256),dtype=np.float32)
        if total>512:raise ValueError('attention prefill currently bounded to 512 tokens')
        if self.attention_head_cap>1:
            head_cap=self.bounded_heads(self.attention_head_cap,total*768)
            for head in range(0,16,head_cap):
                heads=min(head_cap,16-head)
                inputs=np.concatenate([np.concatenate([all_q[:,h].ravel(),all_k[:,h//4].ravel(),all_v[:,h//4].ravel()]) for h in range(head,head+heads)])
                out[:,head:head+heads]=self.t.run('attention_heads_bf16',inputs,[total,256,heads]).reshape(heads,total,256).transpose(1,0,2)[-n:]
        else:
            for head in range(16):out[:,head]=self.t.run('attention_bf16',np.concatenate([all_q[:,head].ravel(),all_k[:,head//4].ravel(),all_v[:,head//4].ravel()]),[total,256]).reshape(total,256)[-n:]
        path=self.t.directory/'states';path.mkdir(exist_ok=True);np.savez(path/f'layer-{layer:02d}.npz',keys=all_k,values=all_v,positions=np.arange(total,dtype=np.int32))
        return self.linear(self.pair('attention_gate',out.reshape(n,4096),gate),name+'.o_proj')
    def forward(self,token_ids,layers=32):
        if not 1<=len(token_ids)<=512:raise ValueError('text prefill needs 1..512 tokens')
        embedding=PREFIX+'embed_tokens.weight';parts=[]
        for start in range(0,len(token_ids),128):
            ids=token_ids[start:start+128];parts.append(self.t.run('embed',ids,[len(ids),2560],tensor=embedding).reshape(len(ids),2560))
        hidden=np.concatenate(parts,axis=0);pre_normalized=None
        for layer in range(layers):
            self.current_layer=layer
            prefix=PREFIX+f'layers.{layer}';start=len(self.t.measurements);clock=time.perf_counter()
            normalized=pre_normalized if pre_normalized is not None else self.norm(hidden,prefix+'.input_layernorm')
            attention=self.attention(normalized,prefix+'.self_attn',layer) if (layer+1)%4==0 else self.delta(normalized,prefix+'.linear_attn',layer)
            if self.fuse_add_norm:hidden,normalized=self.add_norm(hidden,attention,prefix+'.post_attention_layernorm')
            else:
                hidden=self.pair('add_bf16',hidden,attention)
                normalized=self.norm(hidden,prefix+'.post_attention_layernorm')
            if self.fuse_mlp:
                gate_name=prefix+'.mlp.gate_proj';up_name=prefix+'.mlp.up_proj';shape=self.weights[gate_name+'.weight'];rank=max(self.weights[gate_name+'.lora_A.weight']['rows'],self.weights[up_name+'.lora_A.weight']['rows'])
                product=self.t.project_mlp_integer(normalized,shape['rows'],shape['cols'],rank,gate_name+'.weight',up_name+'.weight',2.,self.row_cap,self.token_cap,min(self.work_cap*2,4000000000))
            else:
                gate=self.linear(normalized,prefix+'.mlp.gate_proj');up=self.linear(normalized,prefix+'.mlp.up_proj');product=self.pair('swiglu_bf16',gate,up)
            mlp=self.linear(product,prefix+'.mlp.down_proj')
            if self.fuse_add_norm and (layer+1<layers or layers==32):
                norm_name=PREFIX+f'layers.{layer+1}.input_layernorm' if layer+1<layers else PREFIX+'norm'
                hidden,pre_normalized=self.add_norm(hidden,mlp,norm_name)
            else:hidden=self.pair('add_bf16',hidden,mlp)
            np.save(self.t.directory/f'layer-{layer:02d}.npy',self.recorded_hidden(hidden,layer))
            queries=self.t.measurements[start:];record=dict(layer=layer,kind='full_attention' if (layer+1)%4==0 else 'delta',queries=len(queries),instructions=sum(q['ok']['instructions'] for q in queries),candid_bytes=sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in queries),wall_seconds=time.perf_counter()-clock,replayed=sum(bool(q.get('replayed')) for q in queries))
            self.layers.append(record);atomic(self.t.directory/'layers.json',(json.dumps(self.layers,indent=2)+'\n').encode());print(json.dumps(record),flush=True)
        return (pre_normalized if self.fuse_add_norm else self.norm(hidden,PREFIX+'norm')) if layers==32 else hidden
