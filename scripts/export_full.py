#!/usr/bin/env python3
"""Stream the locked text model only; preserve original BF16 and separate F32 LoRA."""
import argparse,hashlib,json,pathlib,struct
import numpy as np
from checkpoint import Checkpoint
ROOT=pathlib.Path(__file__).resolve().parents[1]
NORM_SUFFIXES=('.input_layernorm.weight','.post_attention_layernorm.weight','model.norm.weight','.q_norm.weight','.k_norm.weight')
def bf16(x):
 b=np.asarray(x,dtype='<f4').view(np.uint32);return ((b+0x7fff+((b>>16)&1))>>16).astype('<u2')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='checkpoints/full');ap.add_argument('--layers',type=int,default=32);ap.add_argument('--int8',action='store_true');args=ap.parse_args();prefix=ROOT/args.output
 if not 1<=args.layers<=32:raise SystemExit('layers must be 1..32')
 if prefix.with_suffix('.pack').exists() or prefix.with_suffix('.manifest.json').exists():raise SystemExit('Refusing to overwrite pack')
 prefix.parent.mkdir(parents=True,exist_ok=True)
 files=sorted((ROOT/'checkpoints/base').glob('*.safetensors'));base=[Checkpoint(p) for p in files];adapter=Checkpoint(ROOT/'checkpoints/adapter/adapter_model.safetensors');head=Checkpoint(ROOT/'checkpoints/adapter/decision_readout.safetensors')
 tensors=[];digest=hashlib.sha256();offset=0
 def append(f,name,b,rows,cols,dtype,transform=None):
  nonlocal offset
  tensors.append(dict(name=name,offset=offset,rows=rows,cols=cols,dtype=dtype,bytes=len(b)));f.write(b);digest.update(b);offset+=len(b)
 def include(name):
  if not name.startswith('model.language_model.'):return False
  if '.layers.' in name:return int(name.split('.layers.')[1].split('.')[0])<args.layers
  return name.endswith('embed_tokens.weight') or name.endswith('model.norm.weight')
 pack=prefix.with_suffix('.pack');temp=pack.with_suffix('.pack.part')
 with temp.open('wb') as out:
  for ck in base:
   for name,t in sorted(ck.header.items()):
    if name=='__metadata__' or not include(name):continue
    shape=t['shape'];rows,cols=(shape[0],int(np.prod(shape[1:]))) if len(shape)>1 else (1,shape[0])
    dtype={'BF16':'bf16','F32':'f32'}[t['dtype']]
    if args.int8 and len(shape)>1 and name.endswith('.weight'):
     # Stream 256 rows at a time; embedding is >1GB in the original checkpoint.
     count=rows*cols;scales=np.empty(rows,dtype='<f4');start_offset=offset
     item_bytes={'BF16':2,'F32':4}[t['dtype']];data_start=ck.offset+t['data_offsets'][0]
     for row in range(0,rows,256):
      length=min(256,rows-row)
      if t['dtype']=='BF16':value=(np.frombuffer(ck.map,dtype='<u2',count=length*cols,offset=data_start+row*cols*2).astype(np.uint32)<<16).view(np.float32).reshape(length,cols)
      else:value=np.frombuffer(ck.map,dtype='<f4',count=length*cols,offset=data_start+row*cols*4).reshape(length,cols)
      scale=np.maximum(np.max(abs(value),axis=1)/np.float32(127.),np.finfo(np.float32).tiny);scales[row:row+length]=scale
      b=np.clip(np.rint(value/scale[:,None]),-127,127).astype(np.int8).tobytes();out.write(b);digest.update(b);offset+=len(b)
     b=scales.tobytes();out.write(b);digest.update(b);offset+=len(b)
     tensors.append(dict(name=name,offset=start_offset,rows=rows,cols=cols,dtype='int8',bytes=count+rows*4))
     continue
    if any(name.endswith(s) for s in NORM_SUFFIXES):
     value=ck.tensor(name)+np.float32(1.);b=bf16(value).tobytes() if dtype=='bf16' else value.astype('<f4').tobytes()
    else:
     a,z=t['data_offsets'];b=ck.map[ck.offset+a:ck.offset+z]
    append(out,name,b,rows,cols,dtype)
  for name,t in sorted(adapter.header.items()):
   if name=='__metadata__':continue
   name=name.removeprefix('base_model.model.')
   if not include(name):continue
   shape=t['shape'];a,z=t['data_offsets'];append(out,name,adapter.map[adapter.offset+a:adapter.offset+z],shape[0],shape[1],{'F32':'f32','BF16':'bf16'}[t['dtype']])
  keys=[k for k in head.header if k!='__metadata__'];assert len(keys)==1
  w=head.tensor(keys[0]);append(out,'readout-f32',w.astype('<f4').tobytes(),256,2560,'f32')
 temp.replace(pack)
 m=dict(version=1,model=hashlib.sha256((ROOT/'MODEL_LOCK.json').read_bytes()).hexdigest(),pack_hash=digest.hexdigest(),bytes=offset,tensors=tensors)
 prefix.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 prefix.with_suffix('.provenance.json').write_text(json.dumps(dict(model_lock=m['model'],layers=args.layers,base_encoding='Symmetric per-output-row INT8, F32 scale=max(abs(row))/127, RNE [-127,127]; scalar/norm parameters unchanged' if args.int8 else 'Original BF16/F32, no quantization',norm_transform='BF16-rounded +1 for official Qwen3.5 sanitizer suffixes',adapter='Separate original F32 rank64 scale2',omitted='Vision/MTP and unused tied LM head; dedicated readout retained',pack_hash=m['pack_hash']),indent=2)+'\n')
 print('exported',len(tensors),'tensors',offset,'bytes',flush=True)
if __name__=='__main__':main()
