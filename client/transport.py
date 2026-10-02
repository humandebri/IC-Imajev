"""Binary client-held state; atomic checkpoints and bounded replay. Checksums are not authentication."""
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
def encode(header,values):
    h=json.dumps(header,separators=(',',':'),allow_nan=False).encode();v=np.asarray(values,dtype='<f4').ravel()
    codec=header.get('encoding','');limit=900000 if codec in ('bf16-exact','int8-block256-v1') else 450000
    if header.get('op')=='delta_heads_bf16' and codec=='int8-block256-v1':limit=1200000
    if v.size>limit or not np.isfinite(v).all():raise ValueError('activation bounds')
    if len(h)>16384:raise ValueError('header size')
    if codec=='bf16-exact':
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
    return b+hashlib.sha256(b).digest()
def decode(b):
    if not 36<=len(b)<=2000000 or hashlib.sha256(b[:-32]).digest()!=b[-32:]:raise ValueError('state checksum/size')
    n,=struct.unpack('<I',b[:4])
    if n>16384 or n+36>len(b):raise ValueError('header size')
    h=json.loads(b[4:4+n]);payload=b[4+n:-32];codec=h.get('encoding','')
    if codec=='bf16-exact':
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
    def __init__(self,model,url,canister,pem,directory,pack_hash,wire_codec=""):
        self.wire_codec=wire_codec;self.max_floats=900000 if wire_codec in ("bf16-exact","int8-block256-v1") else 450000
        self.pack_hash=pack_hash;self.model=model;self.directory=pathlib.Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.index=0;self.measurements=[]
        self.process=subprocess.Popen([str(ROOT/'target/release/imajev-client'),url,canister,pem],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    def command(self,cmd):
        self.process.stdin.write(json.dumps(cmd)+'\n');self.process.stdin.flush();line=self.process.stdout.readline()
        if not line:raise RuntimeError('bridge exited')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    def upload(self,manifest,pack):return self.command({'op':'upload','manifest':str(manifest),'pack':str(pack)})
    def run(self,op,values,dims=(),scalars=(),tensor='',input_hash='0'*64,aux=()):
        h={'version':1,'model':self.model,'pack_hash':self.pack_hash,'input_hash':input_hash,'step':self.index,'op':op,'tensor':tensor,'dims':list(dims),'scalars':list(scalars)}
        if aux:h['aux']=list(aux)
        if getattr(self,'wire_codec',''):h['encoding']=self.wire_codec
        request=self.directory/f'{self.index:06d}.request.bin';response=self.directory/f'{self.index:06d}.response.bin'
        atomic(request,encode(h,values));last=None
        for attempt in range(3):
            try:result=self.command({'op':'step','input':str(request),'output':str(response)});break
            except RuntimeError as e:
                last=e
                if not any(k in str(e).lower() for k in ('429','502','503','504','timeout','timed out','connection')):raise
                if attempt==2:raise
                time.sleep(0.1*2**attempt)
        returned,v=decode(response.read_bytes())
        if any(returned[k]!=v for k,v in h.items() if k not in ('step','scalars')) or returned['step']!=h['step']+1 or not np.array_equal(np.asarray(returned['scalars'],dtype=np.float32),np.asarray(h['scalars'],dtype=np.float32)):raise ValueError('state identity/progress mismatch')
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
        if getattr(self,'wire_codec','')!='bf16-exact':raise ValueError('integer evaluation requires lossless BF16 wire')
        if x.ndim!=2 or x.shape[1]!=cols or cols%256 or rows%8 or min(row_cap,token_cap,work_cap)<=0:raise ValueError('integer projection shape/caps')
        out=np.empty((len(x),rows),dtype=np.float32);token=0
        while token<len(x):
            n=min(token_cap,len(x)-token,self.max_floats//cols)
            # Avoid computing padded rows when the token/blob cap splits a projection.
            if n<len(x)-token and n>=8:n=n//8*8
            if n<1:raise ValueError('integer projection token bound')
            width_cap=min(row_cap,self.max_floats//n,30_000_000//cols,(work_cap//n-rank*cols)//(cols+rank))//8*8
            row=0
            while row<rows:
                context=dict(dims=[n,cols,rank,row],tensor=base,aux=[lora_a,lora_b] if rank else [],scalars=[scale] if rank else [],arithmetic='int8-block256-base-f32-lora-v1')
                if hasattr(self,'projection_width'):width_cap=self.projection_width(x[token:token+n],context,width_cap)//8*8
                width=min(width_cap,rows-row)
                if width<8:raise ValueError('integer projection minimum tile')
                try:
                    tile=self.run('lora_integer' if rank else 'linear_integer_bf16',x[token:token+n],[n,width,cols,row],[scale] if rank else [],tensor=base,aux=[lora_a,lora_b] if rank else [])
                except RuntimeError as error:
                    if not is_instruction_limit(error) or width<=8:raise
                    width_cap=max(8,width//2//8*8)
                    if hasattr(self,'remember_projection_limit'):self.remember_projection_limit(x[token:token+n],context,width,width_cap)
                    continue
                out[token:token+n,row:row+width]=tile.reshape(n,width);row+=width
            token+=n
        return out

    def project_mlp_integer(self,values,rows,cols,rank,gate,up,scale=2.,row_cap=4096,token_cap=132,work_cap=4000000000):
        """Two BF16-rounded integer/LoRA projections, then existing BF16 SwiGLU."""
        x=np.asarray(values,dtype=np.float32)
        if getattr(self,'wire_codec','')!='bf16-exact':raise ValueError('fused MLP requires lossless wire')
        if x.ndim!=2 or x.shape[1]!=cols or cols%256 or rows%8 or min(row_cap,token_cap,work_cap)<=0:raise ValueError('fused MLP shape/caps')
        out=np.empty((len(x),rows),dtype=np.float32);token=0
        while token<len(x):
            n=min(token_cap,len(x)-token,self.max_floats//cols)
            if n<len(x)-token and n>=8:n=n//8*8
            if n<1:raise ValueError('fused MLP token bound')
            width_cap=min(row_cap,self.max_floats//n,30_000_000//cols,(min(work_cap,4000000000)//(2*n)-rank*cols)//(cols+rank))//8*8
            row=0
            while row<rows:
                context=dict(dims=[n,cols,rank,row],tensor=gate,aux=[up],scalars=[scale],arithmetic='int8-block256-base-f32-lora-v1',projection='mlp_gate_up_integer')
                if hasattr(self,'projection_width'):width_cap=self.projection_width(x[token:token+n],context,width_cap)//8*8
                width=min(width_cap,rows-row)
                if width<8:raise ValueError('fused MLP minimum tile')
                try:tile=self.run('mlp_gate_up_integer',x[token:token+n],[n,width,cols,row],[scale],tensor=gate,aux=[up])
                except RuntimeError as error:
                    if not is_instruction_limit(error) or width<=8:raise
                    width_cap=max(8,width//2//8*8)
                    if hasattr(self,'remember_projection_limit'):self.remember_projection_limit(x[token:token+n],context,width,width_cap)
                    continue
                out[token:token+n,row:row+width]=tile.reshape(n,width);row+=width
            token+=n
        return out
