"""Exact transport of already prepared INT8 lanes, F32 scales/A and BF16 residual."""
import numpy as np
from projection_codec import block_pack,block_unpack
NAME='mlp-down-state-exact-v1'
def layout(h):
    d=h.get('dims',[]);s=h.get('scalars',[]);a=h.get('aux',[])
    if h.get('encoding')!=NAME or h.get('op') not in ('mlp_prepare_down','mlp_down_norm_prepared','mlp_prepare_partial_down') or not (len(d)==2 or h.get("op")=="mlp_prepare_partial_down" and len(d)==3 and type(d[2])is int and 0<d[2]<2560 and d[2]%32==0) or any(type(v)is not int for v in d) or not 1<=d[0]<=89 or d[1]!=2560 or len(s)!=2 or not np.array_equal(np.asarray(s,dtype='<f4').view('<u4'),np.asarray([2.,1e-6],dtype='<f4').view('<u4')) or len(a)!=1:raise ValueError('MLP pipeline metadata')
    tensor=h.get('tensor','');prefix='model.language_model.layers.';suffix='.post_attention_layernorm.weight'
    if not tensor.startswith(prefix) or not tensor.endswith(suffix):raise ValueError('MLP pipeline tensor')
    layer=tensor[len(prefix):-len(suffix)]
    if not layer.isdigit() or str(int(layer))!=layer or not 0<=int(layer)<31 or a[0]!=f'{prefix}{int(layer)+1}.input_layernorm.weight':raise ValueError('MLP pipeline layer/norm')
    n=d[0];return n,n*2560,n*9216,n*100

def encode_payload(h,v):
    n,c,q,tail=layout(h)
    if len(v)==2*c:return b'\0'+block_pack(v)
    if len(v)!=c+q+tail or np.any(v[:c].view('<u4')&65535):raise ValueError('MLP pipeline state shape')
    integers=v[c:c+q]
    if not np.isfinite(integers).all() or np.any(integers<-127) or np.any(integers>127) or np.any(integers!=np.trunc(integers)) or np.any(integers.view('<u4')==0x80000000):raise ValueError('MLP pipeline integer')
    sx=v[c+q:c+q+n*36]
    if not np.isfinite(sx).all() or np.any(sx<=0):raise ValueError('MLP pipeline scales')
    return b'\1'+(v[:c].view('<u4')>>16).astype('<u2').tobytes()+integers.astype(np.int8).tobytes()+v[c+q:].astype('<f4').tobytes()

def decode_payload(h,p):
    n,c,q,tail=layout(h)
    if not p:raise ValueError('MLP pipeline tag')
    if p[0]==0:
        v=block_unpack(p[1:])
        if len(v)!=2*c:raise ValueError('MLP pipeline plain shape')
        return v
    if p[0]!=1 or len(p)!=1+c*2+q+tail*4:raise ValueError('MLP pipeline state length')
    residual=(np.frombuffer(p,dtype='<u2',count=c,offset=1).astype('<u4')<<16).view('<f4')
    integers=np.frombuffer(p,dtype=np.int8,count=q,offset=1+c*2)
    if np.any(integers==-128):raise ValueError('MLP pipeline integer')
    rest=np.frombuffer(p,dtype='<f4',count=tail,offset=1+c*2+q)
    if not np.isfinite(rest).all() or np.any(rest[:n*36]<=0):raise ValueError('MLP pipeline scales/A')
    return np.concatenate([residual,integers.astype('<f4'),rest])
