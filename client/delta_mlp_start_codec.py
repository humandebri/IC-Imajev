"""Exact framing for full Delta followed by an MLP generation chunk."""
import json,re,struct
import numpy as np
from mlp_stream_codec import decode_payload,C
NAME='delta-mlp-start-exact-v1'
CONV=3*8192

def layout(h):
 d=h.get('dims',[]);s=h.get('scalars',[]);a=h.get('aux',[])
 if h.get('encoding')!=NAME or h.get('op')not in ('delta_mlp_stream_prepare','delta_mlp_stream_start_ids') or len(d)!=3 or any(type(v)is not int for v in d) or len(s)!=2 or len(a)!=1 or not np.array_equal(np.asarray(s,dtype='<f4').view('<u4'),np.asarray([2.,1e-6],dtype='<f4').view('<u4')):raise ValueError('Delta MLP start metadata')
 n,b,p=d
 if not 1<=n<=89 or not 0<b<=9216 or b%256 or not 1<=p<=132:raise ValueError('Delta MLP start bounds')
 m=re.fullmatch(r'model\.language_model\.layers\.(0|[1-9][0-9]*)\.post_attention_layernorm\.weight',h.get('tensor',''))
 if not m or (h['op']=='delta_mlp_stream_start_ids'and int(m[1])!=0) or int(m[1])>30 or (int(m[1])+1)%4==0 or a[0]!=f'model.language_model.layers.{int(m[1])+1}.input_layernorm.weight':raise ValueError('Delta MLP start scope')
 return n,b,p

def encode_request(h,hidden,norm,history,log):
 from transport import frame_digest
 n,b,p=layout(h)
 if h['op']!='delta_mlp_stream_prepare':raise ValueError('Delta MLP start hidden direction')
 v=[np.asarray(x,dtype='<f4').ravel()for x in [hidden,norm,history,log]]
 if [x.size for x in v]!=[n*C,n*C,CONV,p*6176] or not all(np.isfinite(x).all()for x in v) or any(np.any(x.view('<u4')&65535)for x in v[:3]) or np.any(v[3][:p*2048].view('<u4')&65535) or np.any(v[3][p*6144:]<0) or np.any(v[3][p*6144:]>1):raise ValueError('Delta MLP start precision/shape/gates')
 bf=np.concatenate(v[:3]+[v[3][:p*2048]]);payload=b'\1'+(bf.view('<u4')>>16).astype('<u2').tobytes()+v[3][p*2048:].tobytes()
 header=json.dumps(h,separators=(',',':'),allow_nan=False).encode();body=struct.pack('<I',len(header))+header+payload
 if len(header)>16384 or len(body)+32>2_000_000:raise ValueError('Delta MLP start frame bounds')
 return body+frame_digest(h,body)

def decode_reply(h,payload):
 from mlp_stream_codec import NAME as MLP_NAME
 n,b,p=layout(h)
 if payload[:1]!=b'\0' or len(payload)<1+2*CONV:raise ValueError('Delta MLP start reply direction')
 end=len(payload)-2*CONV;inner=dict(h,encoding=MLP_NAME,op='mlp_stream_prepare',dims=[n,0,b])
 state=decode_payload(inner,payload[1:end]);history=(np.frombuffer(payload[end:],dtype='<u2').astype('<u4')<<16).view('<f4')
 if not np.isfinite(history).all():raise ValueError('Delta MLP start history finite')
 return np.concatenate([state,history])

def encode_ids(h,ids,history,log):
 from transport import frame_digest
 n,b,p=layout(h)
 if h['op']!='delta_mlp_stream_start_ids' or len(ids)!=n or any(type(i)is not int or not 0<=i<=16_777_216 for i in ids):raise ValueError('Delta MLP ids range/length')
 cv=np.asarray(history,dtype='<f4').ravel();lv=np.asarray(log,dtype='<f4').ravel()
 if cv.size!=CONV or lv.size!=p*6176 or not np.isfinite(cv).all() or not np.isfinite(lv).all() or np.any(cv.view('<u4')&65535) or np.any(lv[:p*2048].view('<u4')&65535) or np.any(lv[p*6144:]<0) or np.any(lv[p*6144:]>1):raise ValueError('Delta MLP ids history/log')
 bf=np.concatenate([cv,lv[:p*2048]]);payload=b'\2'+np.asarray(ids,dtype='<u4').tobytes()+(bf.view('<u4')>>16).astype('<u2').tobytes()+lv[p*2048:].tobytes()
 header=json.dumps(h,separators=(',',':'),allow_nan=False).encode();body=struct.pack('<I',len(header))+header+payload
 if len(header)>16384 or len(body)+32>2_000_000:raise ValueError('Delta MLP ids frame bounds')
 return body+frame_digest(h,body)
