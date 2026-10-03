"""Experimental exact transfer of already quantized integer projection state."""
import struct
import numpy as np
NAME='projection-block256-exact-v1'

def layout(h):
    d=h.get('dims',[])
    if len(d)!=5 or h.get('op') not in ('lora_integer_capture','lora_integer_reuse','mlp_gate_up_capture','mlp_gate_up_reuse'):raise ValueError('projection codec metadata')
    n,rows,cols,_,rank=d
    if not 1<=n<=512 or rows<=0 or rows%8 or not 0<cols<=262144 or cols%256 or not 1<=rank<=256:raise ValueError('projection codec shape')
    y=n*rows;q=n*cols;sx=n*(cols//256);tail=sx+n*rank*(2 if h['op'].startswith('mlp_gate_up_') else 1)
    if y>900000 or q+tail>900000 or (h['op'] in ('lora_integer_capture','mlp_gate_up_capture') and y+q+tail>900000) or y==q+tail:raise ValueError('projection codec bounds')
    return y,q,sx,tail

def block_pack(v):
    count=len(v);bits=v.view('<u4');blocks=(count+255)//256
    padded=np.zeros(blocks*256,dtype=bool);padded[:count]=(bits&65535)!=0
    flags=padded.reshape(blocks,256).any(axis=1);full=np.repeat(flags,256)[:count]
    pos=np.arange(count)+np.cumsum(full,dtype=np.int64)-full
    words=np.empty(count+int(full.sum()),dtype='<u2');words[pos]=(bits>>16).astype('<u2')
    words[pos[full]]=(bits[full]&65535).astype('<u2');words[pos[full]+1]=(bits[full]>>16).astype('<u2')
    return struct.pack('<I',count)+np.packbits(flags,bitorder='little').tobytes()+words.tobytes()

def block_unpack(p):
    if len(p)<4:raise ValueError('block codec count')
    count,=struct.unpack('<I',p[:4]);blocks=(count+255)//256;blen=(blocks+7)//8
    if count>900000 or len(p)<4+blen:raise ValueError('block codec bounds')
    raw=np.frombuffer(p[4:4+blen],dtype=np.uint8)
    if blocks%8 and int(raw[-1])>>(blocks%8):raise ValueError('block codec padding')
    flags=np.unpackbits(raw,bitorder='little')[:blocks].astype(bool);full=np.repeat(flags,256)[:count]
    if len(p)!=4+blen+2*(count+int(full.sum())):raise ValueError('block codec length')
    words=np.frombuffer(p[4+blen:],dtype='<u2');pos=np.arange(count)+np.cumsum(full,dtype=np.int64)-full
    bits=words[pos+full].astype('<u4')<<16;bits[full]|=words[pos[full]]
    padded=np.zeros(blocks*256,dtype=bool);padded[:count]=(bits&65535)!=0
    if not np.array_equal(flags,padded.reshape(blocks,256).any(axis=1)):raise ValueError('block codec noncanonical')
    return bits.view('<f4')

def encode_payload(h,v):
    y,q,sx,tail=layout(h);capture=h['op'] in ('lora_integer_capture','mlp_gate_up_capture')
    if (capture and len(v)==q) or (not capture and len(v)==y):return b'\0'+block_pack(v)
    prefix=y if capture else 0
    if len(v)!=prefix+q+tail or np.any(v[:prefix].view('<u4')&65535):raise ValueError('projection codec values')
    integers=v[prefix:prefix+q]
    if not np.isfinite(integers).all() or np.any(integers<-127) or np.any(integers>127) or np.any(integers!=np.trunc(integers)) or np.any(integers.view('<u4')==0x80000000):raise ValueError('projection codec integer')
    scales=v[prefix+q:prefix+q+sx]
    if not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('projection codec scales')
    return bytes([1 if capture else 2])+(v[:prefix].view('<u4')>>16).astype('<u2').tobytes()+integers.astype(np.int8).tobytes()+v[prefix+q:].astype('<f4').tobytes()

def decode_payload(h,p):
    y,q,sx,tail=layout(h);capture=h['op'] in ('lora_integer_capture','mlp_gate_up_capture')
    if not p:raise ValueError('projection codec tag')
    tag=p[0];body=p[1:]
    if tag==0:
        v=block_unpack(body)
        if len(v)!=(q if capture else y):raise ValueError('projection codec plain length')
        return v
    if tag!=(1 if capture else 2):raise ValueError('projection codec direction')
    prefix=y if capture else 0
    if len(body)!=prefix*2+q+tail*4:raise ValueError('projection codec length')
    bf=(np.frombuffer(body,dtype='<u2',count=prefix).astype('<u4')<<16).view('<f4')
    integers=np.frombuffer(body,dtype=np.int8,count=q,offset=prefix*2)
    if np.any(integers==-128):raise ValueError('projection codec integer')
    rest=np.frombuffer(body,dtype='<f4',count=tail,offset=prefix*2+q)
    if not np.isfinite(rest[:sx]).all() or np.any(rest[:sx]<=0):raise ValueError('projection codec scales')
    return np.concatenate([bf,integers.astype('<f4'),rest])
