"""Roll consecutive Delta layers using only canister-produced client-held carries."""
import json,time
import numpy as np
from full_inference import TextGraph,atomic,PREFIX
from prefix_inference import PrefixTextGraph
from delta_mlp_start_codec import NAME as START_NAME,CONV,encode_ids
from mlp_delta_stream_codec import NAME as PAIR_NAME,encode_request,encode_continue_request
from mlp_delta_carry import HUFFMAN_NAME as FINISH_NAME,encode_request as encode_finish

class RollPrefixGraph(PrefixTextGraph):
 def __init__(self,*args,roll_begin=4096,roll_heads=18,roll_down=768,**kwargs):
  super().__init__(*args,**kwargs)
  if type(roll_begin)is not int or not 0<roll_begin<9216 or roll_begin%256 or type(roll_heads)is not int or not 0<roll_heads<32 or roll_heads%2 or type(roll_down)is not int or not 0<roll_down<2560 or roll_down%32:raise ValueError('rolled chunk bounds')
  self.roll_blocks=True
  self.roll_begin=roll_begin;self.roll_heads=roll_heads;self.roll_down=roll_down
  if self.position_offset!=45 or self.retain_terminal_state or not self.fuse_mlp_full or not self.fuse_delta_full_log or self.t.frame_version!=3:raise ValueError('rolled graph requires prepared log prefix, full MLP and host frame')
 def header(self,op,codec,dims,layer):
  return dict(version=3,model=self.t.model,pack_hash=self.t.pack_hash,input_hash=self.t.input_hash,step=self.t.index,op=op,encoding=codec,tensor=f'{PREFIX}layers.{layer}.post_attention_layernorm.weight',aux=[f'{PREFIX}layers.{layer+1}.input_layernorm.weight'],dims=dims,scalars=[2.,1e-6])
 def record(self,layer,hidden,start,clock,extra):
  np.save(self.t.directory/f'layer-{layer:02d}.npy',self.recorded_hidden(hidden,layer))
  q=self.t.measurements[start:];r=dict(layer=layer,kind='delta',queries=len(q),instructions=sum(v['ok']['instructions']for v in q),candid_bytes=sum(v['ok']['request_bytes']+v['ok']['reply_bytes']for v in q),wall_seconds=time.perf_counter()-clock,replayed=sum(bool(v.get('replayed'))for v in q),**extra)
  self.layers.append(r);atomic(self.t.directory/'layers.json',(json.dumps(self.layers,indent=2)+'\n').encode());print(json.dumps(r),flush=True)
 def forward(self,token_ids,layers=32):
  p=self.position_offset
  if layers!=32 or list(token_ids[:p])!=self.cache['metadata']['token_ids'] or not p<len(token_ids)<=512:raise ValueError('rolled prefix identity')
  ids=list(token_ids[p:])
  # The 89-token measured entry exceeds 5B even at begin=3840. Preserve
  # the independently verified standard query graph for this boundary.
  self.roll_effective=len(ids)<=87
  if not self.roll_effective:
   self.roll_blocks=False
   try:return PrefixTextGraph.forward(self,token_ids,layers)
   finally:self.roll_blocks=True
  hidden,attention=self.roll_block(0,ids=ids)
  return TextGraph.forward(self,ids,layers,initial_hidden=hidden,start_layer=2,first_attention=attention)
 def roll_block(self,layer,hidden=None,norm=None,ids=None):
  p=self.position_offset;n=len(ids)if ids is not None else len(hidden);b=self.roll_begin;k=self.roll_heads;rows=self.roll_down;C=2560
  if not 1<=n<=89:raise ValueError('rolled suffix tokens')
  state_dir=self.t.directory/'states';state_dir.mkdir(exist_ok=True)
  start=len(self.t.measurements);clock=time.perf_counter()
  prefix=self.cache['states'][layer]
  if ids is not None:
   h=self.header('delta_mlp_stream_start_ids',START_NAME,[n,b,p],layer);packet=encode_ids(h,ids,prefix['conv'],prefix['delta_log'])
  else:
   from delta_mlp_start_codec import encode_request as encode_start
   if norm is None:raise ValueError('rolled block requires canister-produced norm')
   h=self.header('delta_mlp_stream_prepare',START_NAME,[n,b,p],layer);packet=encode_start(h,hidden,norm,prefix['conv'],prefix['delta_log'])
  x=self.t.run_encoded(h,packet)
  np.savez(state_dir/f'layer-{layer:02d}.npz',conv=x[-CONV:].reshape(3,8192));carry=x[:-CONV]
  first=np.concatenate([np.arange(k//2*128),np.arange(2048,2048+k//2*128),np.arange(4096,4096+k*128)])
  rest=np.concatenate([np.arange(k//2*128,2048),np.arange(2048+k//2*128,4096),np.arange(4096+k*128,8192)])
  prefix=self.cache['states'][layer+1];log=prefix['delta_log'];conv=prefix['conv']
  def sliced(first_group):
   sl=slice(None,k//2*128)if first_group else slice(k//2*128,None);vl=slice(None,k*128)if first_group else slice(k*128,None);gl=slice(None,k)if first_group else slice(k,None)
   return np.concatenate([log[:p*2048].reshape(p,2048)[:,sl].ravel(),log[p*2048:p*6144].reshape(p,4096)[:,vl].ravel(),log[p*6144:].reshape(p,32)[:,gl].ravel()])
  h=self.header('mlp_complete_delta_partial',PAIR_NAME,[n,b,9216-b,k,p],layer);x=self.t.run_encoded(h,encode_request(h,carry,conv[:,first],sliced(True)))
  current_conv=np.empty((3,8192),np.float32);current_conv[:,first]=x[-3*k*256:].reshape(3,k*256)
  self.record(layer,x[:n*C].reshape(n,C),start,clock,dict(next_delta_heads_included=k))
  start=len(self.t.measurements);clock=time.perf_counter()
  h=self.header('delta_partial_mlp_prepare_down',PAIR_NAME,[n,b,9216-b,k,p,rows],layer);x=self.t.run_encoded(h,encode_continue_request(h,x,conv[:,rest],sliced(False),compress_base=True))
  remain=32-k;current_conv[:,rest]=x[-3*remain*256:].reshape(3,remain*256);np.savez(state_dir/f'layer-{layer+1:02d}.npz',conv=current_conv)
  prefix=self.cache['states'][layer+2];h=self.header('mlp_finish_delta_log_integer',FINISH_NAME,[n,p,rows],layer+1)
  x=self.t.run_encoded(h,encode_finish(h,x[:n*(C+9216+100)],prefix['conv'],prefix['delta_log']))
  hidden=x[:n*C].reshape(n,C);attention=x[n*C:2*n*C].reshape(n,C);np.savez(state_dir/f'layer-{layer+2:02d}.npz',conv=x[2*n*C:].reshape(3,8192))
  self.record(layer+1,hidden,start,clock,dict(next_delta_included=layer+2))
  return hidden,attention
