"""Lossless MLP carry: BF16 residual, integer lanes and unrounded F32 state."""
import re,struct
import numpy as np
from projection_codec import block_pack,block_unpack
NAME='mlp-stream-exact-v1';C=2560;H=9216;R=64

def layout(h):
    d=h.get('dims',[]);s=h.get('scalars',[]);a=h.get('aux',[]);op=h.get('op')
    if h.get('encoding')!=NAME or op not in ('mlp_stream_prepare','mlp_stream_next','mlp_stream_finish') or len(d)!=3 or any(type(v)is not int for v in d) or len(a)!=1 or len(s)!=2 or not np.array_equal(np.asarray(s,dtype='<f4').view('<u4'),np.asarray([2.,1e-6],dtype='<f4').view('<u4')):raise ValueError('MLP stream metadata')
    n,begin,count=d
    if not 1<=n<=89 or begin<0 or count<0 or begin%256 or count%256 or begin+count>H or (op=='mlp_stream_finish' and (begin!=H or count!=0)) or (op!='mlp_stream_finish' and (count==0 or (op=='mlp_stream_prepare')!=(begin==0))):raise ValueError('MLP stream bounds')
    m=re.fullmatch(r'model\.language_model\.layers\.(0|[1-9][0-9]*)\.post_attention_layernorm\.weight',h.get('tensor',''))
    if not m or not 0<=int(m[1])<31 or a[0]!=f'model.language_model.layers.{int(m[1])+1}.input_layernorm.weight':raise ValueError('MLP stream layer/norm')
    return n,begin,count

def length(n,done):return n*(2*C+done+C//256+done//256+3*R)
def limit(h):
    n,b,c=layout(h);return max(length(n,b+c),2*n*C)
def integer(v):
    if np.any(v<-127) or np.any(v>127) or np.any(v!=np.trunc(v)) or np.any(v.view('<u4')==0x80000000):raise ValueError('MLP stream integer')
    return v.astype(np.int8).tobytes()
def encode_payload(h,v):
    n,b,c=layout(h)
    if not np.isfinite(v).all():raise ValueError('MLP stream finite')
    if len(v)==2*n*C:
        if np.any(v.view('<u4')&65535):raise ValueError('MLP stream plain precision')
        return b'\0'+block_pack(v)
    done=b+c if len(v)==length(n,b+c) else b if len(v)==length(n,b) else -1
    if done<=0 or np.any(v[:n*C].view('<u4')&65535):raise ValueError('MLP stream state shape/precision')
    cursor=n*C;payload=b'\1'+struct.pack('<I',done)+(v[:cursor].view('<u4')>>16).astype('<u2').tobytes()
    payload+=integer(v[cursor:cursor+n*C]);cursor+=n*C
    tail=n*(C//256+2*R);payload+=v[cursor:cursor+tail].astype('<f4').tobytes();cursor+=tail
    payload+=integer(v[cursor:cursor+n*done]);cursor+=n*done;payload+=v[cursor:].astype('<f4').tobytes()
    # Encoding already checked finite/BF16/integer lanes and exact length.
    # Inspect scale spans directly; do not allocate a decoded duplicate carry.
    input_scales=2*n*C
    product_scales=n*(2*C+C//256+2*R+done)
    if np.any(v[input_scales:input_scales+n*C//256]<=0) or np.any(v[product_scales:product_scales+n*done//256]<=0):raise ValueError('MLP stream scale')
    return payload

def decode_payload(h,p):
    n,b,c=layout(h)
    if p[:1]==b'\0':
        v=block_unpack(p[1:])
        if len(v)!=2*n*C or not np.isfinite(v).all() or np.any(v.view('<u4')&65535):raise ValueError('MLP stream plain precision/shape')
        return v
    if p[:1]!=b'\1' or len(p)<5:raise ValueError('MLP stream state tag')
    done,=struct.unpack('<I',p[1:5])
    if not 0<done<=H or done%256 or done not in (b,b+c) or len(p)!=5+n*(2*C+C+done+4*(C//256+done//256+3*R)):raise ValueError('MLP stream state size/progress')
    cursor=5;residual=(np.frombuffer(p,dtype='<u2',count=n*C,offset=cursor).astype('<u4')<<16).view('<f4');cursor+=n*C*2
    q=np.frombuffer(p,dtype=np.int8,count=n*C,offset=cursor);cursor+=n*C
    tail=n*(C//256+2*R);a=np.frombuffer(p,dtype='<f4',count=tail,offset=cursor);cursor+=tail*4
    product=np.frombuffer(p,dtype=np.int8,count=n*done,offset=cursor);cursor+=n*done
    rest=np.frombuffer(p,dtype='<f4',offset=cursor)
    if np.any(q==-128) or np.any(product==-128) or not np.isfinite(residual).all() or not np.isfinite(a).all() or not np.isfinite(rest).all() or np.any(a[:n*C//256]<=0) or np.any(rest[:n*done//256]<=0):raise ValueError('MLP stream state lanes/scales/finite')
    return np.concatenate([residual,q.astype('<f4'),a,product.astype('<f4'),rest])
