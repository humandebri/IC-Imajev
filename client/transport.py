"""Binary client-held state; atomic checkpoints and bounded replay. Checksums are not authentication."""
from decision_validation import validate_decision
import hashlib,json,pathlib,struct,subprocess,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def is_instruction_limit(error):
    message=str(error).lower()
    return any(k in message for k in ('ic0522','instruction limit','instruction_limit','instructions exceeded','work limit')) or ('exceeded the limit of' in message and 'instructions' in message)
def int8_prefix(header,count):
    op=header.get('op','');dims=header.get('dims',[])
    if op=='embed' and len(dims)==2 and count==dims[0]:return 0
    if op=='delta_gates':return 0
    if op=='delta_heads_bf16':
        if len(dims)!=4:raise ValueError('INT8 delta heads shape')
        n,k,v,heads=dims
        if not (1<=n<=512 and 1<=k<=128 and 1<=v<=128 and 1<=heads<=16):raise ValueError('INT8 delta heads shape')
        if count==heads*(n*(2*k+v+2)+k*v):return heads*n*(2*k+v)
        if count==heads*(n*v+k*v):return heads*n*v
        raise ValueError('INT8 delta heads payload')
    if op in ('delta','delta_bf16'):
        if len(dims)!=3:raise ValueError('INT8 delta shape')
        n,k,v=dims
        if not (1<=n<=512 and 1<=k<=128 and 1<=v<=128):raise ValueError('INT8 delta shape')
        if count==n*(2*k+v+2)+k*v:return n*(2*k+v)
        if count==n*v+k*v:return n*v
        raise ValueError('INT8 delta payload')
    return count
def frame_digest(header,body):
    version=header.get('version',1)
    if type(version) is not int:raise ValueError('unsupported frame version')
    if version in (1,3):return hashlib.sha256(body).digest()
    if version==2:
        import blake3
        return blake3.blake3(body).digest()
    raise ValueError('unsupported frame version')
def encode(header,values):
    h=json.dumps(header,separators=(',',':'),allow_nan=False).encode();v=np.asarray(values,dtype='<f4').ravel()
    codec=header.get('encoding','');limit=900000 if codec in ('bf16-exact','bf16-block256-exact-v1','int8-block256-v1','projection-block256-exact-v1') else 450000
    if codec=='mlp-stream-exact-v1':
        from mlp_stream_codec import limit as stream_limit
        limit=stream_limit(header)
    if codec=='mlp-down-state-exact-v1':
        from mlp_codec import layout
        _,c,q,tail=layout(header);limit=c+q+tail
    if header.get('op')=='delta_heads_bf16' and codec=='int8-block256-v1':limit=1200000
    if v.size>limit or not np.isfinite(v).all():raise ValueError('activation bounds')
    if len(h)>16384:raise ValueError('header size')
    if codec=='mlp-stream-exact-v1':
        from mlp_stream_codec import encode_payload
        payload=encode_payload(header,v)
    elif codec=='mlp-down-state-exact-v1':
        from mlp_codec import encode_payload
        payload=encode_payload(header,v)
    elif codec=='projection-block256-exact-v1':
        from projection_codec import encode_payload
        payload=encode_payload(header,v)
    elif codec=='bf16-block256-exact-v1':
        bits=v.view('<u4');blocks=(v.size+255)//256
        padded=np.zeros(blocks*256,dtype=bool);padded[:v.size]=(bits&65535)!=0
        flags=padded.reshape(blocks,256).any(axis=1);full=np.repeat(flags,256)[:v.size]
        pos=np.arange(v.size)+np.cumsum(full,dtype=np.int64)-full
        words=np.empty(v.size+int(full.sum()),dtype='<u2');words[pos]=(bits>>16).astype('<u2')
        words[pos[full]]=(bits[full]&65535).astype('<u2');words[pos[full]+1]=(bits[full]>>16).astype('<u2')
        payload=struct.pack('<I',v.size)+np.packbits(flags,bitorder='little').tobytes()+words.tobytes()
    elif codec=='bf16-exact':
        bits=v.view('<u4');full=(bits & 0xffff)!=0
        pos=np.arange(v.size)+np.cumsum(full,dtype=np.int64)-full
        words=np.empty(v.size+int(full.sum()),dtype='<u2')
        words[pos]=(bits >> 16).astype('<u2')
        words[pos[full]]=(bits[full] & 0xffff).astype('<u2');words[pos[full]+1]=(bits[full] >> 16).astype('<u2')
        payload=struct.pack('<I',v.size)+np.packbits(full,bitorder='little').tobytes()+words.tobytes()
    elif codec=='int8-block256-v1':
        prefix=int8_prefix(header,v.size);blocks=(prefix+255)//256
        padded=np.zeros(blocks*256,dtype=np.float32);padded[:prefix]=v[:prefix];padded=padded.reshape(blocks,256)
        maxima=np.max(np.abs(padded),axis=1) if blocks else np.empty(0,dtype=np.float32)
        scales=np.where(maxima==0,np.float32(1),np.maximum(maxima/np.float32(127),np.array([1],dtype=np.uint32).view(np.float32)[0])).astype('<f4')
        quantized=np.clip(np.rint(padded/scales[:,None]),-127,127).astype(np.int8)
        packed=np.empty((blocks,260),dtype=np.uint8);packed[:,:4]=scales.view(np.uint8).reshape(blocks,4);packed[:,4:]=quantized.view(np.uint8)
        packed_bytes=packed.tobytes()
        if prefix%256:packed_bytes=packed_bytes[:len(packed_bytes)-(256-prefix%256)]
        payload=struct.pack('<II',v.size,prefix)+packed_bytes+v[prefix:].tobytes()
    elif not codec:payload=v.tobytes()
    else:raise ValueError('unsupported encoding')
    b=struct.pack('<I',len(h))+h+payload
    if len(b)+32>2000000:raise ValueError('state size')
    return b+frame_digest(header,b)
def decode(b):
    if not 36<=len(b)<=2000000:raise ValueError('state size')
    n,=struct.unpack('<I',b[:4])
    if n>16384 or n+36>len(b):raise ValueError('header size')
    h=json.loads(b[4:4+n])
    if not isinstance(h,dict):raise ValueError('header shape')
    if frame_digest(h,b[:-32])!=b[-32:]:raise ValueError('state checksum')
    payload=b[4+n:-32];codec=h.get('encoding','')
    if codec=='prefix-start-exact-v1':
        from prefix_start import decode_reply
        return h,decode_reply(h,payload)
    if codec in ('mlp-delta-log-carry-exact-v1','mlp-delta-huffman-carry-exact-v1')and h['op']=='mlp_finish_delta_log_mlp_front':
        from mlp_delta_carry import decode_front_reply
        return h,decode_front_reply(h,payload)
    if codec in ('delta-hybrid-prefix-exact-v1','mlp-delta-log-carry-exact-v1','mlp-delta-huffman-carry-exact-v1'):
        if payload[:1]!=b'\x00':raise ValueError('hybrid reply direction')
        payload=payload[1:];codec='bf16-block256-exact-v1'
    if codec=='mlp-attention-finish-exact-v1':
        from mlp_attention_finish_codec import decode_reply
        return h,decode_reply(h,payload)
    if codec=='attention-mlp-stream-exact-v1':
        from attention_mlp_stream_codec import decode_reply
        return h,decode_reply(h,payload)
    if codec=='delta-mlp-start-exact-v1':
        from delta_mlp_start_codec import decode_reply
        return h,decode_reply(h,payload)
    if codec=='mlp-delta-stream-exact-v1':
        from mlp_delta_stream_codec import decode_reply
        return h,decode_reply(h,payload)
    if codec=='mlp-stream-exact-v1':
        from mlp_stream_codec import decode_payload
        return h,decode_payload(h,payload)
    if codec=='mlp-down-state-exact-v1':
        from mlp_codec import decode_payload
        v=decode_payload(h,payload)
    elif codec=='projection-block256-exact-v1':
        from projection_codec import decode_payload
        v=decode_payload(h,payload)
    elif codec=='bf16-block256-exact-v1':
        if len(payload)<4:raise ValueError('block codec count')
        count,=struct.unpack('<I',payload[:4]);blocks=(count+255)//256;blen=(blocks+7)//8
        if count>900000 or len(payload)<4+blen:raise ValueError('block codec bounds')
        raw=np.frombuffer(payload[4:4+blen],dtype=np.uint8)
        if blocks%8 and int(raw[-1])>>(blocks%8):raise ValueError('block codec padding')
        flags=np.unpackbits(raw,bitorder='little')[:blocks].astype(bool);full=np.repeat(flags,256)[:count]
        if len(payload)!=4+blen+2*(count+int(full.sum())):raise ValueError('block codec length')
        words=np.frombuffer(payload[4+blen:],dtype='<u2');pos=np.arange(count)+np.cumsum(full,dtype=np.int64)-full
        bits=words[pos+full].astype('<u4')<<16;bits[full]|=words[pos[full]]
        padded=np.zeros(blocks*256,dtype=bool);padded[:count]=(bits&65535)!=0
        if not np.array_equal(flags,padded.reshape(blocks,256).any(axis=1)):raise ValueError('block codec noncanonical')
        v=bits.view('<f4')
    elif codec=='bf16-exact':
        if len(payload)<4:raise ValueError('codec count')
        count,=struct.unpack('<I',payload[:4]);blen=(count+7)//8
        if count>900000 or len(payload)<4+blen:raise ValueError('codec bounds')
        bitmap=np.frombuffer(payload[4:4+blen],dtype=np.uint8)
        if count%8 and int(bitmap[-1]) >> (count%8):raise ValueError('codec padding')
        full=np.unpackbits(bitmap,bitorder='little')[:count].astype(bool)
        if len(payload)!=4+blen+2*(count+int(full.sum())):raise ValueError('codec length')
        words=np.frombuffer(payload[4+blen:],dtype='<u2');pos=np.arange(count)+np.cumsum(full,dtype=np.int64)-full
        bits=words[pos+full].astype('<u4') << 16
        if np.any(words[pos[full]]==0):raise ValueError('codec noncanonical')
        bits[full]|=words[pos[full]];v=bits.view('<f4')
    elif codec=='int8-block256-v1':
        if len(payload)<8:raise ValueError('INT8 codec count')
        count,prefix=struct.unpack('<II',payload[:8]);blocks=(prefix+255)//256
        if count>(1200000 if h.get('op')=='delta_heads_bf16' else 900000) or prefix>count or prefix!=int8_prefix(h,count):raise ValueError('INT8 codec bounds')
        if len(payload)!=8+prefix+4*blocks+4*(count-prefix):raise ValueError('INT8 codec length')
        nbytes=prefix+4*blocks;packed=np.zeros(blocks*260,dtype=np.uint8);packed[:nbytes]=np.frombuffer(payload[8:8+nbytes],dtype=np.uint8);packed=packed.reshape(blocks,260)
        scales=packed[:,:4].copy().view('<f4').ravel();q=packed[:,4:].view(np.int8)
        if np.any(~np.isfinite(scales)) or np.any(scales<=0):raise ValueError('INT8 codec scale')
        if np.any(q.ravel()[:prefix]==-128):raise ValueError('INT8 codec range')
        v=np.empty(count,dtype=np.float32);v[:prefix]=(q.astype(np.float32)*scales[:,None]).ravel()[:prefix]
        v[prefix:]=np.frombuffer(payload[8+nbytes:],dtype='<f4')
    elif not codec:
        if len(payload)%4 or len(payload)//4>450000:raise ValueError('activation bounds')
        v=np.frombuffer(payload,dtype='<f4').copy()
    else:raise ValueError('unsupported encoding')
    if not np.isfinite(v).all():raise ValueError('nonfinite state')
    return h,v
def atomic(path,data):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(data);temp.replace(path)
class Transport:
    def __init__(self,model,url,canister,pem,directory,pack_hash,wire_codec="",frame_version=1,bridge_binary=None):
        if type(frame_version) is not int or frame_version not in (1,2,3):raise ValueError('unsupported frame version')
        self.frame_version=frame_version
        self.wire_codec=wire_codec;self.max_floats=900000 if wire_codec in ("bf16-exact","bf16-block256-exact-v1","int8-block256-v1","projection-block256-exact-v1") else 450000
        self.pack_hash=pack_hash;self.model=model;self.directory=pathlib.Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.index=0;self.measurements=[]
        self.process=subprocess.Popen([str(bridge_binary or ROOT/'target/release/imajev-client'),url,canister,pem],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    def command(self,cmd):
        self.process.stdin.write(json.dumps(cmd)+'\n');self.process.stdin.flush();line=self.process.stdout.readline()
        if not line:raise RuntimeError('bridge exited')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    def upload(self,manifest,pack):return self.command({'op':'upload','manifest':str(manifest),'pack':str(pack)})
    def run(self,op,values,dims=(),scalars=(),tensor='',input_hash='0'*64,aux=()):
        h={'version':getattr(self,'frame_version',1),'model':self.model,'pack_hash':self.pack_hash,'input_hash':input_hash,'step':self.index,'op':op,'tensor':tensor,'dims':list(dims),'scalars':list(scalars)}
        if aux:h['aux']=list(aux)
        if getattr(self,'wire_codec',''):h['encoding']=self.wire_codec
        return self._run_encoded(h,encode(h,values))
    def _run_encoded(self,h,payload):
        # JournalTransport already encoded and checked this exact frame.
        op,tensor=h['op'],h['tensor']
        request=self.directory/f'{self.index:06d}.request.bin';response=self.directory/f'{self.index:06d}.response.bin'
        atomic(request,payload);last=None
        fused=op in ('terminal_attention_mlp_integer','terminal_tail_integer','mlp_stream_complete_terminal') and bool(getattr(self,'fuse_terminal_decision',False))
        command={'op':'terminal_step_decision' if fused else 'step','input':str(request),'output':str(response)}
        if fused:command['options']=self.decision_options
        for attempt in range(3):
            try:result=self.command(command);break
            except RuntimeError as e:
                last=e
                if not any(k in str(e).lower() for k in ('429','502','503','504','timeout','timed out','connection')):raise
                if attempt==2:raise
                time.sleep(0.1*2**attempt)
        returned,v=decode(response.read_bytes())
        if any(returned[k]!=v for k,v in h.items() if k not in ('step','scalars')) or returned['step']!=h['step']+1 or not np.array_equal(np.asarray(returned['scalars'],dtype=np.float32),np.asarray(h['scalars'],dtype=np.float32)):raise ValueError('state identity/progress mismatch')
        if fused:
            self.terminal_decision=validate_decision(result['ok']['decision'],self.decision_options)
            result['decision_options']=list(self.decision_options)
        self.measurements.append({'index':self.index,'op':op,'tensor':tensor,**result});self.index+=1
        return v
    def close(self):
        self.process.stdin.close();self.process.wait(timeout=10)
    def project(self,values,rows,cols,rank,base,lora_a,lora_b,scale=2.,row_cap=256,token_cap=132,work_cap=120_000_000):
        """Only independent linear projection tokens may be split this way.

        Commit a tile after a successful reply. A failed instruction-bound query
        halves output width and reuses the same input/progress, without updates.
        Conservative caps are tunable, not copied from Laya's encoder tiers.
        """
        x=np.asarray(values,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=cols or min(rows,cols,rank,row_cap,token_cap,work_cap)<=0:raise ValueError('projection shape/caps')
        result=np.empty((x.shape[0],rows),dtype=np.float32)
        token=0
        while token<len(x):
            n=min(token_cap,len(x)-token,getattr(self,"max_floats",450000)//cols,getattr(self,"max_floats",450000)//rank,work_cap//(cols+rank+rank*cols))
            if n<1:raise ValueError('projection cannot fit minimum tile')
            # Each SIMD lane is an independent token; avoid scalar tails when possible.
            if getattr(self,'wire_codec','') and n>=4:n-=n%4
            width_cap=min(row_cap,getattr(self,"max_floats",450000)//n,(work_cap//n-rank*cols)//(cols+rank))
            row=0
            while row<rows:
                context=dict(dims=[n,cols,rank,row],tensor=base,aux=[lora_a,lora_b],scalars=[scale])
                if hasattr(self,'projection_width'):width_cap=self.projection_width(x[token:token+n],context,width_cap)
                width=min(width_cap,rows-row)
                if width<1:raise ValueError('projection work cap')
                try:
                    grouped=getattr(self,'group_projections',False)
                    total=min((min(getattr(self,'projection_group_cap',2),3 if n>=64 else 2))*width,rows-row) if grouped else width
                    padded=sum(((n*min(width,total-j)+255)//256)*256 for j in range(0,total,width))
                    if padded>getattr(self,'max_floats',450000):total=width
                    if grouped:
                        flat=self.run('lora_grouped',x[token:token+n],[n,width,cols,row,total],[scale],tensor=base,aux=[lora_a,lora_b]);parts=[];offset=0
                        for j in range(0,total,width):
                            count=min(width,total-j);length=n*count;parts.append(flat[offset:offset+length].reshape(n,count));offset+=((length+255)//256)*256
                        tile=np.concatenate(parts,axis=1)
                    else:tile=self.run('lora_project',x[token:token+n],[n,width,cols,row],[scale],tensor=base,aux=[lora_a,lora_b])
                except RuntimeError as error:
                    if not is_instruction_limit(error) or width==1:raise
                    width_cap=max(1,width//2)
                    if hasattr(self,"remember_projection_limit"):self.remember_projection_limit(x[token:token+n],context,width,width_cap)
                    continue
                result[token:token+n,row:row+total]=tile.reshape(n,total)
                row+=total
            token+=n
        return result
    def project_integer(self,values,rows,cols,rank,base,lora_a='',lora_b='',scale=2.,row_cap=8192,token_cap=132,work_cap=2_500_000_000):
        """Experimental I32 base dots; original F32 inputs feed separate LoRA.

        Quantization is per-token, per-256 input columns, independent of output
        tiles. Lossless wire isolates arithmetic error from transport error.
        """
        x=np.asarray(values,dtype=np.float32)
        if getattr(self,'wire_codec','') not in ('bf16-exact','bf16-block256-exact-v1'):raise ValueError('integer evaluation requires lossless BF16 wire')
        if x.ndim!=2 or x.shape[1]!=cols or cols%256 or rows%8 or min(row_cap,token_cap,work_cap)<=0:raise ValueError('integer projection shape/caps')
        out=np.empty((len(x),rows),dtype=np.float32);token=0
        while token<len(x):
            n=min(token_cap,len(x)-token,self.max_floats//cols)
            # Avoid computing padded rows when the token/blob cap splits a projection.
            if n<len(x)-token and n>=8:n=n//8*8
            if n<1:raise ValueError('integer projection token bound')
            width_cap=min(row_cap,self.max_floats//n,30_000_000//cols,(work_cap//n-rank*cols)//(cols+rank))//8*8
            state_len=n*(cols+cols//256+rank)
            capture_cap=min(width_cap,(self.max_floats-state_len)//n)//8*8
            reuse=bool(getattr(self,'reuse_projection_inputs',False) and rank and width_cap>=8 and rows>width_cap and capture_cap>=8
                and 1+(rows-capture_cap+width_cap-1)//width_cap <= (rows+width_cap-1)//width_cap)
            if reuse and (cols+cols//256+rank) in (capture_cap,width_cap,(rows-capture_cap)%width_cap or width_cap):reuse=False
            saved_state=None
            row=0
            while row<rows:
                context=dict(dims=[n,cols,rank,row],tensor=base,aux=[lora_a,lora_b] if rank else [],scalars=[scale] if rank else [],arithmetic='int8-block256-base-f32-lora-v1')
                operands=x[token:token+n] if saved_state is None else saved_state
                if reuse:context['reuse_projection_inputs']='client-held-int8-f32-a-v1'
                if hasattr(self,'projection_width'):width_cap=self.projection_width(operands,context,width_cap)//8*8
                width=min(width_cap,rows-row,capture_cap if reuse and row==0 else width_cap)
                if width<8:raise ValueError('integer projection minimum tile')
                try:
                    if reuse:
                        previous_codec=self.wire_codec
                        try:
                            self.wire_codec='projection-block256-exact-v1'
                            flat=self.run('lora_integer_capture' if row==0 else 'lora_integer_reuse',operands,[n,width,cols,row,rank],[scale],tensor=base,aux=[lora_a,lora_b])
                        finally:self.wire_codec=previous_codec
                        if row==0:
                            if len(flat)!=n*width+state_len:raise ValueError('capture reply shape')
                            tile=flat[:n*width];saved_state=flat[n*width:]
                        else:tile=flat
                    else:tile=self.run('lora_integer' if rank else 'linear_integer_bf16',operands,[n,width,cols,row],[scale] if rank else [],tensor=base,aux=[lora_a,lora_b] if rank else [])
                except RuntimeError as error:
                    if not is_instruction_limit(error) or width<=8:raise
                    width_cap=max(8,width//2//8*8)
                    if hasattr(self,'remember_projection_limit'):self.remember_projection_limit(operands,context,width,width_cap)
                    continue
                out[token:token+n,row:row+width]=tile.reshape(n,width);row+=width
            token+=n
        return out

    def project_mlp_integer(self,values,rows,cols,rank,gate,up,scale=2.,row_cap=4096,token_cap=132,work_cap=4000000000,*,add_norm=None):
        """Two BF16-rounded integer/LoRA projections, then existing BF16 SwiGLU."""
        x=np.asarray(values,dtype=np.float32)
        if getattr(self,'wire_codec','') not in ('bf16-exact','bf16-block256-exact-v1'):raise ValueError('fused MLP requires lossless wire')
        if x.ndim!=2 or x.shape[1]!=cols or cols%256 or rows%8 or min(row_cap,token_cap,work_cap)<=0:raise ValueError('fused MLP shape/caps')
        other=None if add_norm is None else np.asarray(add_norm[0],dtype=np.float32)
        if other is not None and other.shape!=x.shape:raise ValueError('MLP residual shape')
        op='mlp_gate_up_integer' if other is None else 'mlp_add_norm_integer'
        aux=[up] if other is None else [up,add_norm[1]]
        scalars=[scale] if other is None else [scale,1e-6]
        out=np.empty((len(x),rows),dtype=np.float32);token=0
        while token<len(x):
            n=min(token_cap,len(x)-token,self.max_floats//(cols*(1 if other is None else 2)))
            if n<len(x)-token and n>=8:n=n//8*8
            if n<1:raise ValueError('fused MLP token bound')
            width_cap=min(row_cap,self.max_floats//n,30_000_000//cols,(min(work_cap,4500000000 if n<=89 else 4000000000)//(2*n)-rank*cols)//(cols+rank))//8*8
            state_len=n*(cols+cols//256+2*rank)
            capture_cap=min(width_cap,(self.max_floats-state_len)//n)//8*8
            reuse=bool(getattr(self,'reuse_projection_inputs',False) and other is None and rank
                and width_cap>=8 and rows>width_cap and capture_cap>=8
                and 1+(rows-capture_cap+width_cap-1)//width_cap <= (rows+width_cap-1)//width_cap)
            if reuse and (cols+cols//256+2*rank) in (capture_cap,width_cap,(rows-capture_cap)%width_cap or width_cap):reuse=False
            saved_state=None
            row=0
            while row<rows:
                operands=(x[token:token+n] if other is None else np.concatenate([x[token:token+n].ravel(),other[token:token+n].ravel()])) if saved_state is None else saved_state
                context=dict(dims=[n,cols,rank,row],tensor=gate,aux=aux,scalars=scalars,arithmetic='int8-block256-base-f32-lora-v1',projection=op)
                if reuse:context['reuse_projection_inputs']='client-held-int8-f32-two-a-v1'
                if hasattr(self,'projection_width'):width_cap=self.projection_width(operands,context,width_cap)//8*8
                width=min(width_cap,rows-row,capture_cap if reuse and row==0 else width_cap)
                if width<8:raise ValueError('fused MLP minimum tile')
                try:
                    if reuse:
                        previous_codec=self.wire_codec
                        try:
                            self.wire_codec='projection-block256-exact-v1'
                            flat=self.run('mlp_gate_up_capture' if row==0 else 'mlp_gate_up_reuse',operands,[n,width,cols,row,rank],scalars,tensor=gate,aux=aux)
                        finally:self.wire_codec=previous_codec
                        if row==0:
                            if len(flat)!=n*width+state_len:raise ValueError('MLP capture reply shape')
                            tile=flat[:n*width];saved_state=flat[n*width:]
                        else:tile=flat
                    else:tile=self.run(op,operands,[n,width,cols,row],scalars,tensor=gate,aux=aux)
                except RuntimeError as error:
                    if not is_instruction_limit(error) or width<=8:raise
                    width_cap=max(8,width//2//8*8)
                    if hasattr(self,'remember_projection_limit'):self.remember_projection_limit(operands,context,width,width_cap)
                    continue
                out[token:token+n,row:row+width]=tile.reshape(n,width);row+=width
            token+=n
        return out
