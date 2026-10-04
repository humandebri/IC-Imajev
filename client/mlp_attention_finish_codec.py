"""Frame only canister-produced INT8 MLP carry and exact prefix KV."""
import json,re,struct
import numpy as np
from mlp_codec import NAME as MLP_NAME,encode_payload
NAME='mlp-attention-finish-exact-v1'
FRONT='mlp_finish_attention_mlp_front'
STREAM_COMPLETE='mlp_stream_complete_attention_full'
TERMINAL='mlp_stream_complete_terminal'
C,H,KV=2560,9216,2048

def layout(h):
    d=h.get('dims',[]);s=h.get('scalars',[]);a=h.get('aux',[])
    if h.get('encoding')!=NAME or h.get('op')not in ('mlp_finish_attention_full','mlp_finish_attention_full_compact',FRONT,STREAM_COMPLETE,TERMINAL) or len(d)!=(4 if h.get('op')==FRONT else 3) or any(type(x)is not int for x in d) or len(a)!=1 or len(s)!=2 or not np.array_equal(np.asarray(s,dtype='<f4').view('<u4'),np.asarray([2.,1e-6],dtype='<f4').view('<u4')):raise ValueError('MLP attention metadata')
    n,p,rows=d[:3]
    if not 1<=n<=89 or not 0<=p<=132 or not (0<rows<H and rows%128==0 if h['op']in (STREAM_COMPLETE,TERMINAL) else 0<rows<C and rows%32==0) or h['op']==FRONT and not (0<d[3]<H and d[3]%128==0):raise ValueError('MLP attention bounds')
    m=re.fullmatch(r'model\.language_model\.layers\.(0|[1-9][0-9]*)\.post_attention_layernorm\.weight',h.get('tensor',''))
    if not m or (int(m[1])!=30 if h['op']==TERMINAL else int(m[1])>=30) or int(m[1])%4!=2 or a[0]!=f'model.language_model.layers.{int(m[1])+1}.input_layernorm.weight':raise ValueError('MLP attention scope')
    return n,p,rows

def encode_request(h,carry,prefix):
    from transport import frame_digest
    n,p,rows=layout(h);v=np.asarray(carry,dtype='<f4').ravel();kv=np.asarray(prefix,dtype='<f4').ravel()
    if not np.isfinite(v).all() or kv.size!=p*KV or not np.isfinite(kv).all() or np.any(kv.view('<u4')&65535):raise ValueError('MLP attention input shape/finite/precision')
    if h['op']in (STREAM_COMPLETE,TERMINAL):
        from mlp_stream_codec import NAME as STREAM,encode_payload as encode_stream,length
        if v.size!=length(n,rows):raise ValueError('MLP attention stream progress')
        inner=dict(h,encoding=STREAM,op='mlp_stream_complete',dims=[n,rows,H-rows]);carry=encode_stream(inner,v);payload=bytes([3])+struct.pack('<I',len(carry))+carry
    else:
        if v.size!=n*(C+H+100):raise ValueError('MLP attention down shape')
        inner=dict(h,encoding=MLP_NAME,op='mlp_down_norm_partial_prepared',dims=[n,C,rows]);payload=b'\1'+encode_payload(inner,v)
    payload+=(kv.view('<u4')>>16).astype('<u2').tobytes()
    header=json.dumps(h,separators=(',',':'),allow_nan=False).encode();body=struct.pack('<I',len(header))+header+payload
    if len(header)>16384 or len(body)+32>2_000_000:raise ValueError('MLP attention frame bounds')
    return body+frame_digest(h,body)

def next_mlp(h):
    from mlp_stream_codec import NAME as STREAM
    n,_,_=layout(h)
    if h['op']!=FRONT:raise ValueError('MLP attention front direction')
    layer=int(h['tensor'].split('.')[3])+1
    return dict(h,op='mlp_stream_prepare',encoding=STREAM,tensor=f'model.language_model.layers.{layer}.post_attention_layernorm.weight',aux=[f'model.language_model.layers.{layer+1}.input_layernorm.weight'],dims=[n,0,h['dims'][3]])

def decode_reply(h,payload):
    n,_,_=layout(h);count=n*((2 if h['op']in ('mlp_finish_attention_full_compact',STREAM_COMPLETE)else 3)*C+KV)
    if h['op']==TERMINAL:count=2*C+n*KV
    if h['op']==FRONT:
        from mlp_stream_codec import decode_payload,length
        if payload[:1]!=b'\2' or len(payload)<1+2*n*(C+KV):raise ValueError('MLP attention front reply length/direction')
        end=len(payload)-2*n*KV;hidden=(np.frombuffer(payload[1:1+2*n*C],dtype='<u2').astype('<u4')<<16).view('<f4');carry=decode_payload(next_mlp(h),payload[1+2*n*C:end]);kv=(np.frombuffer(payload[end:],dtype='<u2').astype('<u4')<<16).view('<f4')
        if carry.size!=length(n,h['dims'][3]) or not np.isfinite(hidden).all() or not np.isfinite(kv).all():raise ValueError('MLP attention front reply finite/count')
        return np.concatenate([hidden,carry,kv])
    if payload[:1]!=b'\0'  or len(payload)!=1+2*count:raise ValueError('MLP attention reply length/direction')
    values=(np.frombuffer(payload[1:],dtype='<u2').astype('<u4')<<16).view('<f4')
    if not np.isfinite(values).all():raise ValueError('MLP attention reply finite')
    return values
