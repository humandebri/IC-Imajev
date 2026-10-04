"""Experimental 50-query routing. Numerical work and readout stay in ordinary queries."""
import json,time
import numpy as np
from full_inference import atomic
from joined_inference import JoinedPrefixGraph,C,H,KV
from delta_mlp_start_codec import CONV
from mlp_delta_stream_codec import NAME as PAIR,encode_request as pair,encode_continue_request as follow
from mlp_delta_carry import HUFFMAN_NAME,encode_request as finish
from mlp_attention_finish_codec import NAME as BRIDGE,FRONT,STREAM_COMPLETE,TERMINAL,encode_request as bridge

def validate_tail_chunks(heads28,front28,heads29,front30,down29=1024):
 for k in [heads28,heads29]:
  if type(k)is not int or not 0<k<32 or k%2:raise ValueError('tail head bounds')
 for b in [front28,front30]:
  if type(b)is not int or not 0<b<H or b%256:raise ValueError('tail front bounds')
 if type(down29)is not int or not 0<down29<C or down29%32:raise ValueError('tail down bounds')

class TailPrefixGraph(JoinedPrefixGraph):
 def __init__(self,*args,tail_heads28=6,tail_front28=5888,tail_heads29=24,tail_front30=512,tail_down29=1024,**kwargs):
  # Reject bad scheduling parameters before loading states or making a query.
  validate_tail_chunks(tail_heads28,tail_front28,tail_heads29,tail_front30,tail_down29)
  super().__init__(*args,**kwargs)
  self.tail_heads28=tail_heads28;self.tail_front28=tail_front28;self.tail_heads29=tail_heads29;self.tail_front30=tail_front30;self.tail_down29=tail_down29
 def complete_pair(self,layer,n,b,k,carry,start,clock,plain=False,compressed=False):
  a,c,cv1,cv2,log1,log2=self.group(layer+1,k)
  x=self.send(layer,'mlp_full_delta_partial'if plain else'mlp_complete_delta_partial',PAIR,[n,b,H-b,k,self.position_offset],pair,carry,cv1,log1,**(dict(compress_residual=True,residual_raw_threshold=.1,residual_dictionary=True)if compressed else{}))
  self.mark(layer,x[:n*C].reshape(n,C),start,clock,next_delta_heads_included=k)
  conv=np.empty((3,8192),np.float32);conv[:,a]=x[-3*k*256:].reshape(3,k*256)
  return x,conv,c,cv2,log2
 def follow_pair(self,layer,n,b,k,op,front,x,conv,cols,cv,log,*,compress_hidden=False,compress_prefix=True,compress_input=False):
  dims=[n,b,H-b,k,self.position_offset]+([]if front is None else[front])
  x=self.send(layer,op,PAIR,dims,follow,x,cv,log,compress_base=True,compress_prefix=compress_prefix,compress_hidden=compress_hidden,compress_input=compress_input)
  conv[:,cols]=x[-3*(32-k)*256:].reshape(3,(32-k)*256);self.save_conv(layer+1,conv)
  return x[:-3*(32-k)*256]
 def tail(self,carry,n,start,clock):
  p=self.position_offset
  # q37: finish MLP22, all Attention23, then the first 3584 MLP23 rows.
  x=self.send(22,FRONT,BRIDGE,[n,p,1280,3584],bridge,carry,self.prefix_kv(23))
  self.save_kv(23,x[-n*KV:],n);self.mark(22,x[:n*C].reshape(n,C),start,clock,next_attention_included=23)
  start=len(self.t.measurements);clock=time.perf_counter()
  x,conv,cols,cv,log=self.complete_pair(23,n,3584,18,x[n*C:-n*KV],start,clock)
  start=len(self.t.measurements);clock=time.perf_counter()
  carry=self.follow_pair(23,n,3584,18,'delta_partial_mlp_prepare_down',768,x,conv,cols,cv,log)
  z=self.cache['states'][25]
  x=self.send(24,'mlp_finish_delta_log_mlp_front',HUFFMAN_NAME,[n,p,768,1024],finish,carry,z['conv'],z['delta_log'])
  self.save_conv(25,x[-CONV:]);self.mark(24,x[:n*C].reshape(n,C),start,clock,next_delta_included=25)
  start=len(self.t.measurements);clock=time.perf_counter()
  x,conv,cols,cv,log=self.complete_pair(25,n,1024,10,x[n*C:-CONV],start,clock)
  start=len(self.t.measurements);clock=time.perf_counter()
  carry=self.follow_pair(25,n,1024,10,'delta_partial_mlp_front',6912,x,conv,cols,cv,log,compress_hidden=True)
  x=self.send(26,STREAM_COMPLETE,BRIDGE,[n,p,6912],bridge,carry,self.prefix_kv(27))
  self.save_kv(27,x[2*n*C:],n);self.mark(26,x[:n*C].reshape(n,C),start,clock,next_attention_included=27)
  start=len(self.t.measurements);clock=time.perf_counter();k=self.tail_heads28;b=self.tail_front28
  x,conv,cols,cv,log=self.complete_pair(27,n,0,k,x[:2*n*C],start,clock,plain=True)
  start=len(self.t.measurements);clock=time.perf_counter()
  carry=self.follow_pair(27,n,0,k,'delta_partial_mlp_front',b,x,conv,cols,cv,log,compress_hidden=True,compress_input=True)
  k=self.tail_heads29
  x,conv,cols,cv,log=self.complete_pair(28,n,b,k,carry,start,clock,compressed=True)
  start=len(self.t.measurements);clock=time.perf_counter()
  carry=self.follow_pair(28,n,b,k,'delta_partial_mlp_prepare_down',self.tail_down29,x,conv,cols,cv,log,compress_prefix=False)
  b=self.tail_front30;z=self.cache['states'][30]
  x=self.send(29,'mlp_finish_delta_log_mlp_front',HUFFMAN_NAME,[n,p,self.tail_down29,b],finish,carry,z['conv'],z['delta_log'])
  self.save_conv(30,x[-CONV:]);self.mark(29,x[:n*C].reshape(n,C),start,clock,next_delta_included=30)
  start=len(self.t.measurements);clock=time.perf_counter()
  x=self.send(30,TERMINAL,BRIDGE,[n,p,b],bridge,x[n*C:-CONV],self.prefix_kv(31))
  if x.size!=2*C+n*KV:raise ValueError('terminal stream shape')
  self.save_kv(31,x[2*C:],n)
  # Match the existing compact tail exports: layer30 omitted, layer31 last token.
  (self.t.directory/'layer-30.npy').unlink(missing_ok=True);np.save(self.t.directory/'layer-31.npy',x[:C].reshape(1,C))
  q=self.t.measurements[start:];r=dict(layer=30,kind='delta',queries=len(q),instructions=sum(v['ok']['instructions']for v in q),candid_bytes=sum(v['ok']['request_bytes']+v['ok']['reply_bytes']for v in q),wall_seconds=time.perf_counter()-clock,replayed=sum(bool(v.get('replayed'))for v in q),hidden_exported=False)
  self.layers.extend([r,dict(layer=31,kind='full_attention',queries=0,instructions=0,candid_bytes=0,wall_seconds=0.,replayed=0,included_in_layer=30)])
  atomic(self.t.directory/'layers.json',(json.dumps(self.layers,indent=2)+'\n').encode());print(json.dumps(r),flush=True)
  return x[C:2*C].reshape(1,C)
 def forward(self,token_ids,layers=32):
  p=self.position_offset
  if layers!=32 or list(token_ids[:p])!=self.cache['metadata']['token_ids'] or not p<len(token_ids)<=512:raise ValueError('tail prefix identity')
  ids=list(token_ids[p:]);self.tail_effective=len(ids)<=87
  if not self.tail_effective:return super().forward(token_ids,layers)
  self.join_effective=True;self.roll_effective=True;self.roll_blocks=False
  hidden,norm=None,None
  for layer in [0,8]:
   hidden,norm=self.eight(layer,hidden,norm,ids if layer==0 else None)
   hidden,norm=self.remainder(layer+5,hidden,norm)
  hidden,norm=self.eight(16,hidden,norm)
  carry,start,clock=self.entry(21,hidden,norm)
  return self.tail(carry,len(ids),start,clock)
