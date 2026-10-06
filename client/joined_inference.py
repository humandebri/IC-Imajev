"""Eight-layer ordinary-query schedule; model arithmetic stays in the canister."""
import time
import numpy as np
from full_inference import TextGraph,PREFIX
from roll_inference import RollPrefixGraph
from delta_mlp_start_codec import NAME as START,CONV,encode_ids,encode_request as start_packet
from mlp_delta_stream_codec import NAME as PAIR,encode_request as pair,encode_continue_request as follow
from mlp_delta_carry import HUFFMAN_NAME,encode_request as finish
from attention_mlp_stream_codec import NAME as ATT,encode_request as attention_packet
from mlp_attention_finish_codec import NAME as BRIDGE,encode_request as bridge_packet
C,H,KV=2560,9216,2048

class JoinedPrefixGraph(RollPrefixGraph):
 def __init__(self,*args,**kwargs):
  # The joined schedule has independently measured fixed chunk sizes. Do not
  # record user overrides in the session while silently executing other sizes.
  for name,default in [('roll_begin',4096),('roll_heads',18),('roll_down',768)]:
   if kwargs.get(name,default)!=default:raise ValueError('joined graph uses fixed chunks; roll overrides are unsupported')
  super().__init__(*args,**kwargs)
  if not self.fuse_terminal_tail or not getattr(self.t,'fuse_terminal_decision',False):raise ValueError('joined graph requires terminal tail/decision')
 def send(self,layer,op,codec,dims,encode,*values,**kwargs):
  h=self.header(op,codec,dims,layer)
  return self.t.run_encoded(h,encode(h,*values,**kwargs))
 def save_conv(self,layer,v):
  path=self.t.directory/'states';path.mkdir(exist_ok=True);np.savez(path/f'layer-{layer:02d}.npz',conv=v.reshape(3,8192))
 def group(self,layer,k):
  if type(k)is not int or not 0<k<32 or k%2:raise ValueError('joined head group')
  z=self.cache['states'][layer];p=self.position_offset;log=z['delta_log'];cv=z['conv']
  cols=lambda lo,hi:np.concatenate([np.arange(lo//2*128,hi//2*128),np.arange(2048+lo//2*128,2048+hi//2*128),np.arange(4096+lo*128,4096+hi*128)])
  logs=lambda lo,hi:np.concatenate([log[:p*2048].reshape(p,2048)[:,lo//2*128:hi//2*128].ravel(),log[p*2048:p*6144].reshape(p,4096)[:,lo*128:hi*128].ravel(),log[p*6144:].reshape(p,32)[:,lo:hi].ravel()])
  a,b=cols(0,k),cols(k,32)
  return a,b,cv[:,a],cv[:,b],logs(0,k),logs(k,32)
 def prefix_kv(self,layer):
  z=self.cache['states'][layer];return np.concatenate([z['keys'].transpose(1,0,2).ravel(),z['values'].transpose(1,0,2).ravel()])
 def save_kv(self,layer,kv,n):
  if kv.size!=n*KV:raise ValueError('joined KV shape')
  z=self.cache['states'][layer];k=kv[:n*1024].reshape(n,4,256);v=kv[n*1024:].reshape(n,4,256)
  keys=np.concatenate([z['keys'],k]);values=np.concatenate([z['values'],v]);path=self.t.directory/'states';path.mkdir(exist_ok=True)
  np.savez(path/f'layer-{layer:02d}.npz',keys=keys,values=values,positions=np.arange(len(keys),dtype=np.int32))
 def mark(self,layer,hidden,start,clock,**extra):
  self.record(layer,hidden,start,clock,extra)
  if layer%4==3:
   import json
   from full_inference import atomic
   self.layers[-1]['kind']='full_attention';atomic(self.t.directory/'layers.json',(json.dumps(self.layers,indent=2)+'\n').encode())
 def entry(self,layer,hidden=None,norm=None,ids=None,finish_only=False):
  n=len(ids)if ids is not None else len(hidden);p=self.position_offset;b=4352;k=20;rows=1280
  start=len(self.t.measurements);clock=time.perf_counter();z=self.cache['states'][layer]
  if ids is not None:x=self.send(layer,'delta_mlp_stream_start_ids',START,[n,b,p],encode_ids,ids,z['conv'],z['delta_log'])
  else:
   if norm is None:raise ValueError('joined entry needs canister norm')
   x=self.send(layer,'delta_mlp_stream_prepare',START,[n,b,p],start_packet,hidden,norm,z['conv'],z['delta_log'])
  self.save_conv(layer,x[-CONV:]);a,c,cv1,cv2,log1,log2=self.group(layer+1,k)
  x=self.send(layer,'mlp_complete_delta_partial',PAIR,[n,b,H-b,k,p],pair,x[:-CONV],cv1,log1)
  out=x[:n*C].reshape(n,C);conv=np.empty((3,8192),np.float32);conv[:,a]=x[-3*k*256:].reshape(3,k*256)
  if not finish_only:
   self.mark(layer,out,start,clock,next_delta_heads_included=k);start=len(self.t.measurements);clock=time.perf_counter()
  op='delta_partial_finish'if finish_only else'delta_partial_mlp_prepare_down';dims=[n,b,H-b,k,p]+([]if finish_only else[rows])
  x=self.send(layer,op,PAIR,dims,follow,x,cv2,log2,compress_base=True)
  conv[:,c]=x[-3*(32-k)*256:].reshape(3,(32-k)*256);self.save_conv(layer+1,conv)
  if finish_only:
   self.mark(layer,out,start,clock,next_delta_included=layer+1)
   return out,x[n*C:2*n*C].reshape(n,C)
  return x[:n*(C+H+100)],start,clock
 def eight(self,layer,hidden=None,norm=None,ids=None):
  n=len(ids)if ids is not None else len(hidden);p=self.position_offset;b=1792;front=6272;k=26
  carry,start,clock=self.entry(layer,hidden,norm,ids)
  z=self.cache['states'][layer+2]
  x=self.send(layer+1,'mlp_finish_delta_log_mlp_front',HUFFMAN_NAME,[n,p,1280,b],finish,carry,z['conv'],z['delta_log'])
  self.save_conv(layer+2,x[-CONV:]);self.mark(layer+1,x[:n*C].reshape(n,C),start,clock,next_delta_included=layer+2)
  carry=x[n*C:-CONV];start=len(self.t.measurements);clock=time.perf_counter();prefix=self.prefix_kv(layer+3)
  x=self.send(layer+2,'mlp_complete_attention_kv_q4',ATT,[n,b,H-b,p],attention_packet,carry,prefix)
  self.save_kv(layer+3,x[2*n*C:n*(2*C+KV)],n);self.mark(layer+2,x[:n*C].reshape(n,C),start,clock,next_attention_kv_included=layer+3)
  start=len(self.t.measurements);clock=time.perf_counter()
  carry=self.send(layer+3,'attention_finish_mlp_front_q4_compact',ATT,[n,front,p],attention_packet,x,prefix)
  a,c,cv1,cv2,log1,log2=self.group(layer+4,k)
  x=self.send(layer+3,'mlp_complete_delta_partial',PAIR,[n,front,H-front,k,p],pair,carry,cv1,log1,compress_residual=True,residual_raw_threshold=.1,residual_dictionary=True)
  conv=np.empty((3,8192),np.float32);conv[:,a]=x[-3*k*256:].reshape(3,k*256)
  self.mark(layer+3,x[:n*C].reshape(n,C),start,clock,next_delta_heads_included=k)
  start=len(self.t.measurements);clock=time.perf_counter()
  x=self.send(layer+3,'delta_partial_mlp_full',PAIR,[n,front,H-front,k,p],follow,x,cv2,log2,compress_base=True)
  conv[:,c]=x[2*n*C:].reshape(3,(32-k)*256);self.save_conv(layer+4,conv)
  hidden=x[:n*C].reshape(n,C);norm=x[n*C:2*n*C].reshape(n,C);self.mark(layer+4,hidden,start,clock)
  return hidden,norm
 def remainder(self,layer,hidden,norm):
  n=len(hidden);p=self.position_offset;carry,start,clock=self.entry(layer,hidden,norm)
  prefix=self.prefix_kv(layer+2)
  x=self.send(layer+1,'mlp_finish_attention_full_compact',BRIDGE,[n,p,1280],bridge_packet,carry,prefix)
  self.save_kv(layer+2,x[2*n*C:],n);hidden=x[:n*C].reshape(n,C);attention=x[n*C:2*n*C].reshape(n,C)
  self.mark(layer+1,hidden,start,clock,next_attention_included=layer+2)
  start=len(self.t.measurements);clock=time.perf_counter();hidden,norm=self.full_mlp(hidden,attention,PREFIX+f'layers.{layer+2}',layer+2)
  self.mark(layer+2,hidden,start,clock);return hidden,norm
 def forward(self,token_ids,layers=32):
  p=self.position_offset
  if layers!=32 or list(token_ids[:p])!=self.cache['metadata']['token_ids'] or not p<len(token_ids)<=512:raise ValueError('joined prefix identity')
  ids=list(token_ids[p:]);self.join_effective=len(ids)<=87
  if not self.join_effective:return super().forward(token_ids,layers)
  self.roll_effective=True;self.roll_blocks=False
  hidden,norm=None,None
  for layer in [0,8,16,24]:
   hidden,norm=self.eight(layer,hidden,norm,ids if layer==0 else None)
   if layer<24:hidden,norm=self.remainder(layer+5,hidden,norm)
  hidden,attention=self.entry(29,hidden,norm,finish_only=True)
  return TextGraph.forward(self,ids,layers,initial_hidden=hidden,start_layer=30,first_attention=attention)
