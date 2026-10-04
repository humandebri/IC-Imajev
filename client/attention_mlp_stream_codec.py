"""Frame client-held KV and quantized operands; no host model arithmetic."""
import json,struct,re
import numpy as np
from mlp_stream_codec import NAME as MLP_NAME,encode_payload,decode_payload,length
NAME='attention-mlp-stream-exact-v1';C=2560;KV=2048
def layout(h):
 d=h.get('dims',[]);a=h.get('aux',[]);s=h.get('scalars',[]);complete=h.get('op')=='mlp_complete_attention_kv'
 if h.get('encoding')!=NAME or h.get('op')not in ('mlp_complete_attention_kv','attention_finish_mlp_front')or len(d)!=(4 if complete else 3)or any(type(v)is not int for v in d)or len(a)!=1 or len(s)!=2 or not np.array_equal(np.asarray(s,dtype='<f4').view('<u4'),np.asarray([2.,1e-6],dtype='<f4').view('<u4')):raise ValueError('attention MLP metadata')
 n,b=d[:2];p=d[-1]
 m=re.fullmatch(r'model\.language_model\.layers\.(0|[1-9][0-9]*)\.post_attention_layernorm\.weight',h.get('tensor',''))
 if not 1<=n<=89 or not 0<b<9216 or b%256 or not 0<=p<=132 or n+p>512 or complete and b+d[2]!=9216 or not m or int(m[1])>=30 or int(m[1])%4!=(2 if complete else 3)or a[0]!=f'model.language_model.layers.{int(m[1])+1}.input_layernorm.weight':raise ValueError('attention MLP bounds/scope')
 return n,b,p

def inner(h):
 n,b,p=layout(h);complete=h['op']=='mlp_complete_attention_kv'
 return dict(h,encoding=MLP_NAME,op='mlp_stream_complete'if complete else 'mlp_stream_prepare',dims=[n,b,9216-b]if complete else[n,0,b])
def frame(h,payload):
 from transport import frame_digest
 hdr=json.dumps(h,separators=(',',':'),allow_nan=False).encode();body=struct.pack('<I',len(hdr))+hdr+payload
 if len(hdr)>16384 or len(body)+32>2_000_000:raise ValueError('attention MLP frame bounds')
 return body+frame_digest(h,body)
def encode_request(h,state,prefix=None):
 n,b,p=layout(h);v=np.asarray(state,dtype='<f4').ravel()
 if h['op']=='mlp_complete_attention_kv':
  payload=encode_payload(inner(h),v)
  if payload[:1]!=b'\1' or int.from_bytes(payload[1:5],'little')!=b:raise ValueError('attention MLP progress')
  return frame(h,payload)
 bf=n*(2*C+KV);count=bf+n*C+n*C//256
 if v.size!=count or not np.isfinite(v).all()or np.any(v[:bf].view('<u4')&65535):raise ValueError('attention MLP state shape/precision')
 q=v[bf:bf+n*C];sx=v[bf+n*C:]
 if np.any(q<-127)or np.any(q>127)or np.any(q!=np.trunc(q))or np.any(q.view('<u4')==0x80000000)or np.any(sx<=0):raise ValueError('attention MLP integer/scales')
 kv=np.asarray(prefix,dtype='<f4').ravel()
 if kv.size!=p*KV or not np.isfinite(kv).all()or np.any(kv.view('<u4')&65535):raise ValueError('attention MLP prefix shape/precision')
 floats=np.concatenate([v[:bf],kv]);return frame(h,b'\2'+(floats.view('<u4')>>16).astype('<u2').tobytes()+q.astype(np.int8).tobytes()+sx.tobytes())
def decode_reply(h,payload):
 n,b,p=layout(h)
 if h['op']=='mlp_complete_attention_kv':
  bf=n*(2*C+KV);end=1+bf*2
  if payload[:1]!=b'\1'or len(payload)!=end+n*C+4*n*C//256:raise ValueError('attention MLP reply length')
  v=(np.frombuffer(payload[1:end],dtype='<u2').astype('<u4')<<16).view('<f4');q=np.frombuffer(payload[end:end+n*C],dtype=np.int8);sx=np.frombuffer(payload[end+n*C:],dtype='<f4')
  if np.any(q==-128)or not np.isfinite(v).all()or not np.isfinite(sx).all()or np.any(sx<=0):raise ValueError('attention MLP reply lanes')
  return np.concatenate([v,q.astype('<f4'),sx])
 if payload[:1]!=b'\3'or len(payload)<1+2*n*KV:raise ValueError('attention MLP reply direction')
 end=len(payload)-2*n*KV;v=decode_payload(inner(h),payload[1:end]);kv=(np.frombuffer(payload[end:],dtype='<u2').astype('<u4')<<16).view('<f4')
 if v.size!=length(n,b)or not np.isfinite(kv).all():raise ValueError('attention MLP reply count/finite')
 return np.concatenate([v,kv])
