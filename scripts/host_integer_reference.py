#!/usr/bin/env python3
"""Experimental host evaluation of the same INT8 pack under two arithmetic paths.

Keeps official prompt, LoRA/readout/calibration and nonlinear graph. Host graph
is an evaluation aid, not a claim of bit parity with Wasm nonlinear kernels.
Integer block dots use exact F32 integer arithmetic (<2**24) on Metal; scaling
and summing blocks are explicitly evaluated as separate F32 operations.
"""
import argparse,hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--arithmetic',choices=['f32','int8'],required=True);ap.add_argument('--activation-scale',choices=['block256','token'],default='block256');ap.add_argument('--merge-adapter',action='store_true',help='Experimental one-time FP32 adapter merge followed by per-row INT8 weight quantization; not adopted Wasm arithmetic');args,rest=ap.parse_known_args();sys.argv=[sys.argv[0],*rest]
import mlx.core as mx
import mlx.nn as nn
import reference
manifest=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());tensors={t['name']:t for t in manifest['tensors']};pack=np.memmap(ROOT/'checkpoints/full-int8.pack',mode='r',dtype=np.uint8)
original_factory=reference.MLXDirect;original_linear=nn.Linear.__call__;original_embedding=nn.Embedding.__call__;mapped={};embeddings={}
merged_weights={};merged_modules=set();merged_digest=hashlib.sha256()
if args.merge_adapter:
 from mlx_vlm.trainer.lora_layers import LoRALinear
 original_lora=LoRALinear.__call__
 def patched_lora(self,x):
  return self.linear(x) if id(self) in merged_modules else original_lora(self,x)
def patched_linear(self,x):
 entry=mapped.get(id(self))
 if entry is None:return original_linear(self,x)
 qw,sw,rows,cols=entry;shape=x.shape;flat=x.reshape(-1,cols).astype(mx.float32)
 if args.arithmetic=='f32':
  weights=qw.astype(mx.float32)*sw[:,None];mx.eval(weights)
  y=flat@weights.T;mx.eval(y)
 else:
  mx.eval(flat);a=np.asarray(flat,dtype=np.float32);blocks=a.reshape(len(a),cols//256 if args.activation_scale=='block256' else 1,256 if args.activation_scale=='block256' else cols)
  peak=np.max(np.abs(blocks),axis=2);sx=np.where(peak==0,np.float32(1),np.maximum(peak/np.float32(127),np.nextafter(np.float32(0),np.float32(1)))).astype(np.float32)
  aq=np.clip(np.rint(blocks/sx[:,:,None]),-127,127).astype(np.int8).reshape(len(a),cols)
  qx=mx.array(aq);scale_x=mx.array(sx);y=mx.zeros((len(a),rows),dtype=mx.float32 if args.activation_scale=='block256' else mx.int32)
  if args.activation_scale=='token' and cols>131072:raise ValueError('token integer sum could overflow I32')
  for block in range(cols//256):
   start=block*256;dot=qx[:,start:start+256].astype(mx.float32)@qw[:,start:start+256].astype(mx.float32).T;mx.eval(dot)
   if args.activation_scale=='block256':
    scaled=dot*scale_x[:,block,None];mx.eval(scaled)
    scaled=scaled*sw[None,:];mx.eval(scaled)
    y=y+scaled;mx.eval(y)
   else:
    # Each 256-term F32 dot is an exact integer (<2**24). Sum
    # these blocks in I32 before applying the single token scale.
    y=y+dot.astype(mx.int32);mx.eval(y)
  if args.activation_scale=='token':
   y=y.astype(mx.float32)*scale_x[:,0,None];mx.eval(y)
   y=y*sw[None,:];mx.eval(y)
 return y.astype(x.dtype).reshape(*shape[:-1],rows)
def patched_embedding(self,x):
 entry=embeddings.get(id(self))
 if entry is None:return original_embedding(self,x)
 qw,sw,dtype=entry
 return (qw[x].astype(mx.float32)*sw[x][...,None]).astype(dtype)
def factory(*a,**kw):
 engine=original_factory(*a,**kw)
 if args.merge_adapter:
  for name,module in engine.model.named_modules():
   if not isinstance(module,LoRALinear):continue
   key='model.'+name.replace('language_model.model.','language_model.')
   t=tensors.get(key+'.weight')
   if t is None or t['dtype']!='int8':raise ValueError('Merged adapter source tensor')
   rows,cols=t['rows'],t['cols']
   if module.linear.weight.shape!=(rows,cols) or float(module.scale)!=2.:raise ValueError('Fixed adapter merge shape/scale')
   # Preparation only: remove both adapter matmuls from question inference.
   # This changes rounding and weight quantization, so evaluate decisions
   # separately; it is not claimed to preserve the original float bits.
   weights=module.linear.weight.astype(mx.float32)+float(module.scale)*(module.lora_b.astype(mx.float32).T@module.lora_a.astype(mx.float32).T)
   mx.eval(weights)
   peak=mx.max(mx.abs(weights),axis=1)
   sw=mx.where(peak==0,mx.ones_like(peak),mx.maximum(peak/127.,mx.array(np.nextafter(np.float32(0),np.float32(1)))))
   qw=mx.clip(mx.round(weights/sw[:,None]),-127,127).astype(mx.int8)
   mx.eval(qw,sw)
   if not np.isfinite(np.asarray(sw)).all():raise ValueError('Merged weight scales')
   merged_weights[key]=(qw,sw,rows,cols);merged_modules.add(id(module))
   merged_digest.update(key.encode()+b'\0');merged_digest.update(np.asarray(qw).tobytes());merged_digest.update(np.asarray(sw,dtype='<f4').tobytes())
  expected=sum(t['name'].endswith('.lora_A.weight') for t in tensors.values())
  if len(merged_modules)!=expected:raise ValueError(('Merged adapter count',len(merged_modules),expected))
  LoRALinear.__call__=patched_lora
 for name,module in engine.model.named_modules():
  key='model.'+name.replace('language_model.model.','language_model.')
  if key.endswith('.linear'):key=key[:-7]
  t=tensors.get(key+'.weight')
  if isinstance(module,nn.Embedding) and t is not None and t['name'].endswith('embed_tokens.weight'):
   rows,cols,off=t['rows'],t['cols'],t['offset'];dtype=module.weight.dtype
   qw=mx.array(np.asarray(pack[off:off+rows*cols]).view(np.int8).reshape(rows,cols));sw=mx.array(np.asarray(pack[off+rows*cols:off+rows*cols+rows*4]).view('<f4'));mx.eval(qw,sw);embeddings[id(module)]=(qw,sw,dtype);del module['weight']
   continue
  if not isinstance(module,nn.Linear) or t is None or t['dtype']!='int8' or '.layers.' not in t['name'] or t['cols']%256:continue
  rows,cols=t['rows'],t['cols'];off=t['offset']
  if key in merged_weights:qw,sw,rows,cols=merged_weights[key]
  else:qw=mx.array(np.asarray(pack[off:off+rows*cols]).view(np.int8).reshape(rows,cols));sw=mx.array(np.asarray(pack[off+rows*cols:off+rows*cols+rows*4]).view('<f4'))
  mx.eval(qw,sw);mapped[id(module)]=(qw,sw,rows,cols)
  # The hook uses pinned pack values; release replaced BF16 model weights.
  del module['weight']
 if len(mapped)!=248:raise RuntimeError(f'Expected all 248 dense base projections, got {len(mapped)}')
 if len(embeddings)!=1:raise RuntimeError(f'Expected pinned text embedding, got {len(embeddings)}')
 nn.Embedding.__call__=patched_embedding
 nn.Linear.__call__=patched_linear
 print('Host arithmetic',args.arithmetic,'dense projections',len(mapped),flush=True)
 return engine
reference.MLXDirect=factory
reference.main()
# Extend report metadata only after a successful official evaluation.
output=pathlib.Path(rest[rest.index('--output')+1]) if '--output' in rest else ROOT/'artifacts/reference.json'
if not output.is_absolute():output=ROOT/output
r=json.loads(output.read_text());r.update(arithmetic=args.arithmetic,activation_scale=args.activation_scale,weight_pack_hash=manifest['pack_hash'],embedding_pack_hash=manifest['pack_hash'],embedding_precision='Per-row INT8 pinned pack, F32 dequantization, BF16 output',dense_base_projections=len(mapped),scope='Host official nonlinear graph with pinned INT8 base and separate original F32 LoRA/readout; arithmetic A/B; host/Wasm nonlinear parity not guaranteed',implementation_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
if args.merge_adapter:
 r.update(weight_pack_hash=None,source_weight_pack_hash=manifest['pack_hash'],merged_adapter=True,merged_projection_count=len(merged_modules),merged_projection_digest=merged_digest.hexdigest(),scope='Host-only candidate: original BF16 base + original F32 adapter merged once in F32, then per-row INT8; pinned embedding/norm/readout/calibration; no adopted canister or full-pack speed claim')
output.write_text(json.dumps(r,indent=2)+'\n')
