"""Exact framing only: client-held MLP carry and sliced Delta prefix log."""
import json
import re
import struct
import numpy as np
from mlp_stream_codec import C,H,R,NAME as MLP_NAME,encode_payload as encode_mlp
NAME='mlp-delta-stream-exact-v1'
FOLLOW=('delta_partial_mlp_prepare','delta_partial_mlp_prepare_down','delta_partial_mlp_full','delta_partial_finish')

def layout(h):
    d=h.get('dims',[]);a=h.get('aux',[]);s=h.get('scalars',[])
    if h.get('encoding')!=NAME or h.get('op') not in ('mlp_complete_delta_partial',)+FOLLOW or len(d)!=(6 if h.get('op')=='delta_partial_mlp_prepare_down' else 5) or any(type(v)is not int for v in d) or len(a)!=1 or len(s)!=2 or not np.array_equal(np.asarray(s,dtype='<f4').view('<u4'),np.asarray([2.,1e-6],dtype='<f4').view('<u4')):raise ValueError('pair metadata')
    n,b,c,k,p=d[:5]
    if len(d)==6 and not (0<d[5]<C and d[5]%32==0):raise ValueError('pair partial down rows')
    if not 1<=n<=89 or b<=0 or c<=0 or b%128 or c%128 or b+c!=H or not 0<k<32 or k%2 or not 1<=p<=132:raise ValueError('pair bounds')
    m=re.fullmatch(r'model\.language_model\.layers\.(0|[1-9][0-9]*)\.post_attention_layernorm\.weight',h.get('tensor',''))
    if not m or not 0<=int(m[1])<30 or (int(m[1])+2)%4==0 or a[0]!=f'model.language_model.layers.{int(m[1])+1}.input_layernorm.weight':raise ValueError('pair layer scope')
    return n,b,c,k,p

def reply_count(h):
    n,b,c,k,p=layout(h);return n*(2*C if h['op'] in ('delta_partial_mlp_full','delta_partial_finish')else C+H+100)+3*(32-k)*256 if h['op'] in FOLLOW else n*(3*C+C//256+3*R+64)+3*k*256

def encode_request(h,state,history,log,*,compress_residual=False,residual_raw_threshold=0.,residual_dictionary=False):
    from transport import frame_digest
    n,b,c,k,p=layout(h)
    if not 0<=residual_raw_threshold<=1 or residual_dictionary and not compress_residual:raise ValueError('pair residual raw threshold/dictionary')
    if h['op']!='mlp_complete_delta_partial':raise ValueError('pair completion direction')
    inner=dict(h,encoding=MLP_NAME,op='mlp_stream_complete',dims=[n,b,c])
    carry=encode_mlp(inner,state)
    if carry[:1]!=bytes([1 if b%256==0 else 2]) or int.from_bytes(carry[1:5],'little')!=b:raise ValueError('pair input progress')
    cv=np.asarray(history,dtype='<f4').ravel();lv=np.asarray(log,dtype='<f4').ravel();kc=p*k//2*128
    if cv.size!=3*k*256 or lv.size!=p*(k//2*128+k*128+k) or not np.isfinite(cv).all() or not np.isfinite(lv).all() or np.any(cv.view('<u4')&65535) or np.any(lv[:kc].view('<u4')&65535) or np.any(lv[kc+p*k*128:]<0) or np.any(lv[kc+p*k*128:]>1):raise ValueError('pair prefix shape/finite')
    if compress_residual:
        from mlp_delta_carry import encode_plane,encode_dictionary_plane
        raw=carry[5:5+2*n*C];planes=encode_plane(raw[0::2],residual_raw_threshold)+(encode_dictionary_plane(raw[1::2])if residual_dictionary else encode_plane(raw[1::2],residual_raw_threshold));carry=carry[:5]+struct.pack('<I',len(planes))+planes+carry[5+2*n*C:]
    payload=bytes([6 if compress_residual else 1])+struct.pack('<I',len(carry))+carry+(cv.view('<u4')>>16).astype('<u2').tobytes()+(lv[:kc].view('<u4')>>16).astype('<u2').tobytes()+lv[kc:].tobytes()
    header=json.dumps(h,separators=(',',':'),allow_nan=False).encode();body=struct.pack('<I',len(header))+header+payload
    if len(header)>16384 or len(body)+32>2_000_000:raise ValueError('pair frame bounds')
    return body+frame_digest(h,body)

def decode_reply(h,payload):
    n,b,c,k,p=layout(h)
    if h['op']in ('delta_partial_mlp_full','delta_partial_finish'):
        if payload[:1]!=bytes([8 if h['op']=='delta_partial_finish'else 5])or len(payload)!=1+2*reply_count(h):raise ValueError('pair full reply length/direction')
        v=(np.frombuffer(payload[1:],dtype='<u2').astype('<u4')<<16).view('<f4')
        if not np.isfinite(v).all():raise ValueError('pair full reply finite')
        return v
    if h['op'] in FOLLOW:
        from mlp_codec import NAME as DOWN_NAME,decode_payload
        end=2+n*(C*2+H+400);hc=3*(32-k)*256
        if payload[:1]!=b'\3' or len(payload)!=end+2*hc:raise ValueError('pair prepared reply length/direction')
        layer=int(h['tensor'].split('.')[3])+1
        inner=dict(h,op='mlp_prepare_down',encoding=DOWN_NAME,dims=[n,C],tensor=f'model.language_model.layers.{layer}.post_attention_layernorm.weight',aux=[f'model.language_model.layers.{layer+1}.input_layernorm.weight'])
        state=decode_payload(inner,payload[1:end]);history=(np.frombuffer(payload[end:],dtype='<u2').astype('<u4')<<16).view('<f4')
        if not np.isfinite(state).all() or not np.isfinite(history).all():raise ValueError('pair prepared finite')
        return np.concatenate([state,history])
    bf=n*C;tail=n*(C//256+2*R+64+C+R)
    if payload[:1]!=b'\0' or len(payload)!=1+bf*3+tail*4+3*k*256*2:raise ValueError('pair reply length/direction')
    cursor=1
    residual=(np.frombuffer(payload[cursor:cursor+bf*2],dtype='<u2').astype('<u4')<<16).view('<f4');cursor+=bf*2
    integers=np.frombuffer(payload[cursor:cursor+bf],dtype=np.int8)
    if np.any(integers==-128):raise ValueError('pair integer')
    cursor+=bf;rest=np.frombuffer(payload[cursor:cursor+tail*4],dtype='<f4');cursor+=tail*4
    history=(np.frombuffer(payload[cursor:],dtype='<u2').astype('<u4')<<16).view('<f4')
    if not np.isfinite(residual).all() or not np.isfinite(rest).all() or not np.isfinite(history).all() or np.any(rest[:n*C//256]<=0) or np.any(rest[n*(C//256+2*R):n*(C//256+2*R+64)]<0) or np.any(rest[n*(C//256+2*R):n*(C//256+2*R+64)]>1):raise ValueError('pair reply finite/scales/gates')
    return np.concatenate([residual,integers.astype('<f4'),rest,history])


def encode_continue_request(h,carry,history,log,*,compress_base=False):
    from transport import frame_digest
    n,b,c,k,p=layout(h)
    if h['op'] not in FOLLOW:raise ValueError('pair continuation direction')
    v=np.asarray(carry,dtype='<f4').ravel();hc=3*k*256;tail=n*(C//256+2*R+64+C+R);fixed=2*n*C+tail
    if v.size!=fixed+hc or not np.isfinite(v).all() or np.any(v[:n*C].view('<u4')&65535) or np.any(v[-hc:].view('<u4')&65535):raise ValueError('pair continuation carry shape/precision')
    integers=v[n*C:2*n*C];rest=v[2*n*C:fixed]
    if np.any(integers<-127) or np.any(integers>127) or np.any(integers!=np.trunc(integers)) or np.any(integers.view('<u4')==0x80000000) or np.any(rest[:n*C//256]<=0) or np.any(rest[n*(C//256+2*R):n*(C//256+2*R+64)]<0) or np.any(rest[n*(C//256+2*R):n*(C//256+2*R+64)]>1):raise ValueError('pair continuation integer/scales/gates')
    remaining=32-k;cv=np.asarray(history,dtype='<f4').ravel();lv=np.asarray(log,dtype='<f4').ravel();kc=p*remaining//2*128
    if cv.size!=3*remaining*256 or lv.size!=p*(remaining//2*128+remaining*128+remaining) or not np.isfinite(cv).all() or not np.isfinite(lv).all() or np.any(cv.view('<u4')&65535) or np.any(lv[:kc].view('<u4')&65535) or np.any(lv[kc+p*remaining*128:]<0) or np.any(lv[kc+p*remaining*128:]>1):raise ValueError('pair continuation prefix')
    payload=b'\2'+(v[:n*C].view('<u4')>>16).astype('<u2').tobytes()+integers.astype(np.int8).tobytes()+rest.tobytes()+(cv.view('<u4')>>16).astype('<u2').tobytes()+(lv[:kc].view('<u4')>>16).astype('<u2').tobytes()+lv[kc:].tobytes()
    if compress_base:
        from mlp_delta_carry import planar,encode_plane
        prep=n*(C//256+2*R+64);base=rest[prep:prep+n*C].tobytes();planes=planar(base,4);count=n*C
        packed=b''.join(encode_plane(planes[i*count:(i+1)*count])for i in range(4))
        prefix=payload[1+3*n*C+4*tail:]
        payload=b'\4'+struct.pack('<I',len(packed))+payload[1:1+3*n*C]+rest[:prep].tobytes()+rest[prep+n*C:].tobytes()+packed+prefix
    header=json.dumps(h,separators=(',',':'),allow_nan=False).encode();body=struct.pack('<I',len(header))+header+payload
    if len(header)>16384 or len(body)+32>2_000_000:raise ValueError('pair continuation frame bounds')
    return body+frame_digest(h,body)
