"""Frame client-held KV, Q heads and original F32 A; no host model arithmetic."""
import json,struct,re
import numpy as np
from mlp_stream_codec import NAME as MLP_NAME,encode_payload,decode_payload,length
NAME='attention-mlp-stream-exact-v1';C=2560;KV=2048
COMPLETE=('mlp_complete_attention_kv','mlp_complete_attention_kv_q4')
FINISH=('attention_finish_mlp_front','attention_finish_mlp_front_q4','attention_finish_mlp_front_q4_compact')
def layout(h):
 d=h.get('dims',[]);a=h.get('aux',[]);s=h.get('scalars',[]);complete=h.get('op')in COMPLETE
 if h.get('encoding')!=NAME or h.get('op')not in COMPLETE+FINISH or len(d)!=(4 if complete else 3)or any(type(v)is not int for v in d)or len(a)!=1 or len(s)!=2 or not np.array_equal(np.asarray(s,dtype='<f4').view('<u4'),np.asarray([2.,1e-6],dtype='<f4').view('<u4')):raise ValueError('attention MLP metadata')
 n,b=d[:2];p=d[-1]
 m=re.fullmatch(r'model\.language_model\.layers\.(0|[1-9][0-9]*)\.post_attention_layernorm\.weight',h.get('tensor',''))
 if not 1<=n<=89 or not 0<b<9216 or b%128 or not 0<=p<=132 or n+p>512 or complete and b+d[2]!=9216 or not m or int(m[1])>=30 or int(m[1])%4!=(2 if complete else 3)or a[0]!=f'model.language_model.layers.{int(m[1])+1}.input_layernorm.weight':raise ValueError('attention MLP bounds/scope')
 return n,b,p

def inner(h):
 n,b,p=layout(h);complete=h['op']in COMPLETE
 return dict(h,encoding=MLP_NAME,op='mlp_stream_complete'if complete else 'mlp_stream_prepare',dims=[n,b,9216-b]if complete else[n,0,b])
def frame(h,payload):
 from transport import frame_digest
 hdr=json.dumps(h,separators=(',',':'),allow_nan=False).encode();body=struct.pack('<I',len(hdr))+hdr+payload
 if len(hdr)>16384 or len(body)+32>2_000_000:raise ValueError('attention MLP frame bounds')
 return body+frame_digest(h,body)
def bf_bytes(v):
 if not np.isfinite(v).all()or np.any(v.view('<u4')&65535):raise ValueError('attention MLP BF16 precision')
 return (v.view('<u4')>>16).astype('<u2').tobytes()
def prefix_values(prefix,p):
 kv=np.asarray(prefix,dtype='<f4').ravel()
 if kv.size!=p*KV:raise ValueError('attention MLP prefix shape/precision')
 bf_bytes(kv)
 return kv
def encode_request(h,state,prefix=None):
 n,b,p=layout(h);v=np.asarray(state,dtype='<f4').ravel();q4=h['op']in ('mlp_complete_attention_kv_q4','attention_finish_mlp_front_q4','attention_finish_mlp_front_q4_compact')
 if h['op']in COMPLETE:
  payload=encode_payload(inner(h),v)
  if payload[:1]!=bytes([1 if b%256==0 else 2]) or int.from_bytes(payload[1:5],'little')!=b:raise ValueError('attention MLP progress')
  if q4:payload=b'\4'+struct.pack('<I',len(payload))+payload+bf_bytes(prefix_values(prefix,p))
  return frame(h,payload)
 bf=n*(2*C+KV);qend=bf+n*C;send=qend+n*C//256;gend=send+(n*1024 if q4 else 0);count=gend+(n*64 if q4 else 0)
 if v.size!=count or not np.isfinite(v).all():raise ValueError('attention MLP state shape/precision')
 q=v[bf:qend];sx=v[qend:send]
 if np.any(q<-127)or np.any(q>127)or np.any(q!=np.trunc(q))or np.any(q.view('<u4')==0x80000000)or np.any(sx<=0):raise ValueError('attention MLP integer/scales')
 kv=prefix_values(prefix,p);compact=h['op']=='attention_finish_mlp_front_q4_compact';base=np.concatenate([v[:n*C],v[2*n*C:bf]])if compact else v[:bf];floats=np.concatenate([base,v[send:gend],kv])
 return frame(h,bytes([6 if compact else 5 if q4 else 2])+bf_bytes(floats)+q.astype(np.int8).tobytes()+sx.tobytes()+v[gend:].tobytes())
def decode_reply(h,payload):
 n,b,p=layout(h);q4=h['op']in ('mlp_complete_attention_kv_q4','attention_finish_mlp_front_q4','attention_finish_mlp_front_q4_compact')
 if h['op']in COMPLETE:
  bf=n*(2*C+KV);gate=n*1024 if q4 else 0;a=n*64 if q4 else 0;scales=n*C//256;end=1+(bf+gate)*2
  if payload[:1]!=bytes([4 if q4 else 1])or len(payload)!=end+n*C+4*(scales+a):raise ValueError('attention MLP reply length')
  v=(np.frombuffer(payload[1:end],dtype='<u2').astype('<u4')<<16).view('<f4');q=np.frombuffer(payload[end:end+n*C],dtype=np.int8);sx=np.frombuffer(payload[end+n*C:end+n*C+4*scales],dtype='<f4');ax=np.frombuffer(payload[end+n*C+4*scales:],dtype='<f4')
  if np.any(q==-128)or not np.isfinite(v).all()or not np.isfinite(sx).all()or np.any(sx<=0)or not np.isfinite(ax).all():raise ValueError('attention MLP reply lanes')
  return np.concatenate([v[:bf],q.astype('<f4'),sx,v[bf:],ax])
 if h['op']=='attention_finish_mlp_front_q4_compact':
  if payload[:1]!=b'\7':raise ValueError('compact attention MLP reply direction')
  v=decode_payload(inner(h),payload[1:])
  if v.size!=length(n,b):raise ValueError('compact attention MLP reply count')
  return v
 if payload[:1]!=b'\3'or len(payload)<1+2*n*KV:raise ValueError('attention MLP reply direction')
 end=len(payload)-2*n*KV;v=decode_payload(inner(h),payload[1:end]);kv=(np.frombuffer(payload[end:],dtype='<u2').astype('<u4')<<16).view('<f4')
 if v.size!=length(n,b)or not np.isfinite(kv).all():raise ValueError('attention MLP reply count/finite')
 return np.concatenate([v,kv])
