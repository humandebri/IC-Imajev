"""Budget-packed experimental query graph; numerical work remains on-canister."""
import hashlib,json,time,pathlib,struct
import numpy as np
from adaptive_inference import AdaptivePrefixGraph,ShortPlan
from joined_inference import C,H,KV
from delta_mlp_start_codec import NAME as START,CONV,encode_ids
from mlp_delta_carry import HUFFMAN_NAME,encode_request as finish
from mlp_attention_finish_codec import NAME as BRIDGE,FRONT,STREAM_COMPLETE,TERMINAL,encode_request as bridge
from mlp_stream_codec import NAME as STREAM
from transport import encode,decode,atomic
MODULE='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
def aligned_front(n,rows):return min(8960,(rows*67//n//256)*256)
def packed_plan(n,p,module):
 if type(n)is not int or type(p)is not int:return None
 if not 1<=n<=69 or not 1<=p<=27 or module!=MODULE:return None
 return ShortPlan(n,p,(8448 if n>=68 else aligned_front(n,8960)),20,14,down=(2304 if n>=68 else 1280),version='packed-query-v4')
class PackedPrefixGraph(AdaptivePrefixGraph):
 def chain(self,layer,n,front,next_front,carry):
  h=self.header('mlp_stream_complete',STREAM,[n,front,H-front],layer)
  index=self.t.index;d=self.t.directory;input_path=d/f'{index:06d}.request.bin';out_path=d/f'{index:06d}.response.bin';prefix_path=d/f'{index:06d}.prefix.bin';expected_path=d/f'{index:06d}.expected.bin';hidden_path=d/f'{index:06d}.previous.bf16';conv_path=d/f'{index:06d}.conv.bf16'
  request_bytes=encode(h,carry)
  if input_path.exists() and input_path.read_bytes()!=request_bytes:raise ValueError('packed checkpoint input mismatch')
  atomic(input_path,request_bytes)
  z=self.cache['states'][layer+1];cv=z['conv'].ravel();log=z['delta_log'];p=self.position_offset
  bf=np.concatenate([cv,log[:p*2048]]);prefix=(bf.view('<u4')>>16).astype('<u2').tobytes()+np.asarray(log[p*2048:],dtype='<f4').tobytes();atomic(prefix_path,prefix)
  expected=self.header('mlp_stream_prepare',STREAM,[n,0,next_front],layer+1)
  atomic(expected_path,encode(expected,np.zeros(2*n*C,dtype=np.float32)))
  cmd=dict(op='mlp_delta_front',input=str(input_path),expected=str(expected_path),prefix=str(prefix_path),output=str(out_path),hidden=str(hidden_path),conv=str(conv_path),prefix_tokens=p,front=next_front)
  metric_path=d/f'{index:06d}.metric.json'
  if out_path.exists() and metric_path.exists():
   r=json.loads(metric_path.read_text())
   if r.get('index')!=index or r.get('op')!='mlp_delta_front' or r.get('tensor')!=h['tensor']:raise ValueError('packed checkpoint metric identity')
   if r.get('outputs_sha256')!={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (out_path,hidden_path,conv_path)}:raise ValueError('packed checkpoint output hash')
   r['replayed']=True;self.t.replayed+=1
  else:
   started=time.perf_counter()
   try:r=self.t.command(cmd)
   except Exception as error:
    with (d/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(index=index,op='mlp_delta_front',tensor=h['tensor'],dims=h['dims'],error=str(error),wall_seconds=time.perf_counter()-started))+'\n')
    raise
   r['outputs_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (out_path,hidden_path,conv_path)}
  returned,v=decode(out_path.read_bytes())
  assert all(returned[k]==val for k,val in expected.items() if k not in ('step','scalars')) and returned['step']==index+1,'chain output identity'
  assert np.asarray(returned['scalars'],dtype='<f4').tobytes()==np.asarray(expected['scalars'],dtype='<f4').tobytes(),'chain scalars'
  r.update(index=index,op='mlp_delta_front',tensor=h['tensor']);self.t.measurements.append(r);self.t.index+=1
  atomic(metric_path,(json.dumps(r,indent=2)+'\n').encode())
  def bf16(path):return (np.frombuffer(path.read_bytes(),dtype='<u2').astype('<u4')<<16).view('<f4')
  previous,conv=bf16(hidden_path),bf16(conv_path)
  if len(previous)!=n*C or len(conv)!=CONV:raise ValueError('chain reply shape')
  return v,previous,conv
 def forward(self, token_ids, layers=32):
     p = self.position_offset
     if layers != 32 or list(token_ids[:p]) != self.cache['metadata']['token_ids'] or not p < len(token_ids) <= 512:
         raise ValueError('adaptive prefix identity')
     ids = list(token_ids[p:]); plan = packed_plan(len(ids), p, self.cache['metadata']['wasm_sha256'])
     self.adaptive_effective = plan is not None
     if plan is None:
         return super().forward(token_ids, layers)
     self.tail_effective = self.join_effective = self.roll_effective = True
     self.roll_blocks = False
     n, b, k, rows = plan.suffix, plan.front, plan.heads, plan.down
     path = self.t.directory / 'adaptive-plan.json'
     path.write_text(json.dumps(plan.__dict__, indent=2) + '\n')
     start, clock = len(self.t.measurements), time.perf_counter()
     z = self.cache['states'][0]
     x = self.send(0, 'delta_mlp_stream_start_ids', START, [n, b, p], encode_ids, ids, z['conv'], z['delta_log'])
     self.save_conv(0, x[-CONV:]); carry = x[:-CONV]
     layer, kind = 0, 'generation'
     while layer <= 30:
         if kind in ('generation', 'attention_pair'):
             if kind == 'generation' and layer % 4 == 2:
                 op = TERMINAL if layer == 30 else STREAM_COMPLETE
                 x = self.send(layer, op, BRIDGE, [n, p, b], bridge, carry, self.prefix_kv(layer+1))
                 if layer == 30:
                     if x.size != 2*C+n*KV:
                         raise ValueError('adaptive terminal shape')
                     self.save_kv(31, x[2*C:], n)
                     np.save(self.t.directory/'layer-31.npy', x[:C].reshape(1,C))
                     q = self.t.measurements[start:]
                     self.layers.extend([dict(layer=30,kind='delta',queries=len(q),instructions=sum(v['ok']['instructions'] for v in q),candid_bytes=sum(v['ok']['request_bytes']+v['ok']['reply_bytes'] for v in q),wall_seconds=time.perf_counter()-clock,replayed=sum(bool(v.get('replayed')) for v in q),hidden_exported=False),dict(layer=31,kind='full_attention',queries=0,instructions=0,candid_bytes=0,wall_seconds=0.,replayed=0,included_in_layer=30)])
                     (self.t.directory/'layers.json').write_text(json.dumps(self.layers,indent=2)+'\n')
                     return x[C:2*C].reshape(1,C)
                 self.mark(layer,x[:n*C].reshape(n,C),start,clock,next_attention_included=layer+1)
                 self.save_kv(layer+1,x[2*n*C:],n)
                 carry=x[:2*n*C]; layer+=1; kind='attention_pair'
                 start,clock=len(self.t.measurements),time.perf_counter()
                 continue
             if kind == 'generation':
                 next_front = (2304 if (layer+1)%4==2 else 3840) if n>=68 else aligned_front(n, 1280 if (layer+1)%4==2 else 5120)
                 carry, previous, conv = self.chain(layer, n, b, next_front, carry)
                 self.mark(layer,previous.reshape(n,C),start,clock,next_delta_included=layer+1)
                 self.save_conv(layer+1,conv)
                 layer+=1;b=next_front;kind='generation'
                 start,clock=len(self.t.measurements),time.perf_counter()
                 if n>=68 and layer==1:
                     header=self.header('mlp_stream_next',STREAM,[n,b,6656-b],layer)
                     carry=self.t.run_encoded(header,encode(header,carry));b=6656
                 continue
             plain = kind == 'attention_pair'
             pair_heads = plan.plain_heads if plain else k
             x,conv,cols,cv,log = self.complete_pair(layer,n,0 if plain else b,pair_heads,carry,start,clock,plain=plain)
             start,clock=len(self.t.measurements),time.perf_counter()
             terminal_front = layer + 1 == 30
             carry=self.follow_pair(layer,n,0 if plain else b,pair_heads,'delta_partial_mlp_front' if terminal_front else 'delta_partial_mlp_prepare_down',b if terminal_front else rows,x,conv,cols,cv,log)
             layer+=1;kind='generation' if terminal_front else 'down'
         else:
             b=6656 if n>=68 else aligned_front(n,5632)
             if layer % 4 == 2:
                 x=self.send(layer,FRONT,BRIDGE,[n,p,rows,b],bridge,carry,self.prefix_kv(layer+1))
                 self.mark(layer,x[:n*C].reshape(n,C),start,clock,next_attention_included=layer+1)
                 self.save_kv(layer+1,x[-n*KV:],n)
                 carry=x[n*C:-n*KV]
             else:
                 z=self.cache['states'][layer+1]
                 x=self.send(layer,'mlp_finish_delta_log_mlp_front',HUFFMAN_NAME,[n,p,rows,b],finish,carry,z['conv'],z['delta_log'])
                 self.mark(layer,x[:n*C].reshape(n,C),start,clock,next_delta_included=layer+1)
                 self.save_conv(layer+1,x[-CONV:]);carry=x[n*C:-CONV]
             layer+=1;kind='generation'
             start,clock=len(self.t.measurements),time.perf_counter()
     raise AssertionError('adaptive graph did not reach terminal')
