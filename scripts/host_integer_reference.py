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
ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--arithmetic',choices=['f32','int8'],required=True);args,rest=ap.parse_known_args();sys.argv=[sys.argv[0],*rest]
import mlx.core as mx
import mlx.nn as nn
import reference
manifest=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());tensors={t['name']:t for t in manifest['tensors']};pack=np.memmap(ROOT/'checkpoints/full-int8.pack',mode='r',dtype=np.uint8)
original_factory=reference.MLXDirect;original_linear=nn.Linear.__call__;original_embedding=nn.Embedding.__call__;mapped={};embeddings={}
def patched_linear(self,x):
 entry=mapped.get(id(self))
 if entry is None:return original_linear(self,x)
 qw,sw,rows,cols=entry;shape=x.shape;flat=x.reshape(-1,cols).astype(mx.float32)
 if args.arithmetic=='f32':
  weights=qw.astype(mx.float32)*sw[:,None];mx.eval(weights)
  y=flat@weights.T;mx.eval(y)
 else:
  mx.eval(flat);a=np.asarray(flat,dtype=np.float32);blocks=a.reshape(len(a),cols//256,256)
  peak=np.max(np.abs(blocks),axis=2);sx=np.where(peak==0,np.float32(1),np.maximum(peak/np.float32(127),np.nextafter(np.float32(0),np.float32(1)))).astype(np.float32)
  aq=np.clip(np.rint(blocks/sx[:,:,None]),-127,127).astype(np.int8).reshape(len(a),cols)
  qx=mx.array(aq);scale_x=mx.array(sx);y=mx.zeros((len(a),rows),dtype=mx.float32)
  for block in range(cols//256):
   start=block*256;dot=qx[:,start:start+256].astype(mx.float32)@qw[:,start:start+256].astype(mx.float32).T;mx.eval(dot)
   scaled=dot*scale_x[:,block,None];mx.eval(scaled)
   scaled=scaled*sw[None,:];mx.eval(scaled)
   y=y+scaled;mx.eval(y)
 return y.astype(x.dtype).reshape(*shape[:-1],rows)
def patched_embedding(self,x):
 entry=embeddings.get(id(self))
 if entry is None:return original_embedding(self,x)
 qw,sw,dtype=entry
 return (qw[x].astype(mx.float32)*sw[x][...,None]).astype(dtype)
def factory(*a,**kw):
 engine=original_factory(*a,**kw)
 for name,module in engine.model.named_modules():
  key='model.'+name.replace('language_model.model.','language_model.')
  if key.endswith('.linear'):key=key[:-7]
  t=tensors.get(key+'.weight')
  if isinstance(module,nn.Embedding) and t is not None and t['name'].endswith('embed_tokens.weight'):
   rows,cols,off=t['rows'],t['cols'],t['offset'];dtype=module.weight.dtype
   qw=mx.array(np.asarray(pack[off:off+rows*cols]).view(np.int8).reshape(rows,cols));sw=mx.array(np.asarray(pack[off+rows*cols:off+rows*cols+rows*4]).view('<f4'));mx.eval(qw,sw);embeddings[id(module)]=(qw,sw,dtype);del module['weight']
   continue
  if not isinstance(module,nn.Linear) or t is None or t['dtype']!='int8' or '.layers.' not in t['name'] or t['cols']%256:continue
  rows,cols=t['rows'],t['cols'];off=t['offset'];qw=mx.array(np.asarray(pack[off:off+rows*cols]).view(np.int8).reshape(rows,cols));sw=mx.array(np.asarray(pack[off+rows*cols:off+rows*cols+rows*4]).view('<f4'))
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
r=json.loads(output.read_text());r.update(arithmetic=args.arithmetic,weight_pack_hash=manifest['pack_hash'],embedding_pack_hash=manifest['pack_hash'],embedding_precision='Per-row INT8 pinned pack, F32 dequantization, BF16 output',dense_base_projections=len(mapped),scope='Host official nonlinear graph with pinned INT8 base and separate original F32 LoRA/readout; arithmetic A/B; host/Wasm nonlinear parity not guaranteed',implementation_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest());output.write_text(json.dumps(r,indent=2)+'\n')
