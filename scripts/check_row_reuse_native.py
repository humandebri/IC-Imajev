#!/usr/bin/env python3
"""Independent real first-layer I32/F32 projection equivalence under exact row compaction."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
from reconstruct_terminal_mlp_reference import bf,ordered,quantize
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/row-reuse-native-v1';d.mkdir(exist_ok=False)
 manifest=ROOT/'checkpoints/full-int8.manifest.json';m=json.loads(manifest.read_text());assert sha(ROOT/'MODEL_LOCK.json')==m['model']
 identity=ROOT/'artifacts/terminal-mlp-reference-v1/pack-identity.json';ident=json.loads(identity.read_text());pack=ROOT/'checkpoints/full-int8.pack';assert ident['pack_hash']==m['pack_hash'] and ident['bytes']==pack.stat().st_size
 tensors={t['name']:t for t in m['tensors']};weights={};hashes={}
 def weight(name):
  if name not in weights:
   t=tensors[name]
   with pack.open('rb')as f:f.seek(t['offset']);raw=f.read(t['bytes'])
   assert len(raw)==t['bytes'];hashes[name]=hashlib.sha256(raw).hexdigest();count=t['rows']*t['cols']
   if t['dtype']=='int8':w=np.frombuffer(raw,dtype='i1',count=count).reshape(t['rows'],t['cols']);s=np.frombuffer(raw,dtype='<f4',count=t['rows'],offset=count)
   elif t['dtype']=='bf16':w=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4').reshape(t['rows'],t['cols']);s=None
   else:assert t['dtype']=='f32';w=np.frombuffer(raw,dtype='<f4').reshape(t['rows'],t['cols']);s=None
   weights[name]=w,s
  return weights[name]
 def integer(q,sx,name):
  w,sw=weight(name);assert q.shape[1]==w.shape[1];out=np.zeros((len(q),len(w)),dtype='<f4')
  for b in range(q.shape[1]//256):
   z=q[:,b*256:(b+1)*256].astype(np.int32)@w[:,b*256:(b+1)*256].astype(np.int32).T
   out=np.add(out,np.multiply(np.multiply(z.astype('<f4'),sx[:,b,None]),sw[None,:]))
  return out
 paid_path=ROOT/'artifacts/paid-finite-max-v1/proof/report.json';paid=json.loads(paid_path.read_text());assert paid['complete']
 root='model.language_model.layers.0.linear_attn';normw,_=weight('model.language_model.layers.0.input_layernorm.weight');qa_weight,_=weight(root+'.in_proj_qkv.lora_A.weight');za_weight,_=weight(root+'.in_proj_z.lora_A.weight');embedding=tensors['model.language_model.embed_tokens.weight'];cases=[]
 for item in paid['results']:
  ids=item['request']['request']['token_ids'][item['quote']['prefix_tokens']:];embedded=[];embedding_hashes={}
  with pack.open('rb')as f:
   for token in ids:
    f.seek(embedding['offset']+token*2560);raw=f.read(2560);f.seek(embedding['offset']+embedding['rows']*2560+token*4);scale=f.read(4)
    assert len(raw)==2560 and len(scale)==4;embedding_hashes[str(token)]=hashlib.sha256(raw+scale).hexdigest();embedded.append(bf(np.multiply(np.frombuffer(raw,dtype='i1').astype('<f4'),np.frombuffer(scale,dtype='<f4')[0])))
  x=np.array(embedded,dtype='<f4');total=np.zeros(len(x),dtype='<f4')
  for k in range(2560):total=np.add(total,np.multiply(x[:,k],x[:,k]))
  scale=np.divide(np.float32(1),np.sqrt(np.add(np.divide(total,np.float32(2560)),np.float32(1e-6)),dtype=np.float32),dtype=np.float32)
  norm=bf(np.multiply(np.multiply(x,scale[:,None]),normw));q,sx=quantize(norm);qa=ordered(norm,qa_weight);za=ordered(norm,za_weight)
  keys=[q[t].tobytes()+sx[t].tobytes()+qa[t].tobytes()+za[t].tobytes()for t in range(len(x))];first=[];index=[];seen={}
  for t,key in enumerate(keys):
   if key not in seen:seen[key]=len(first);first.append(t)
   index.append(seen[key])
  assert len(first)==len(set(ids));outputs={};count=0
  for name,ax in [('in_proj_qkv',qa),('in_proj_z',za)]:
   w,_=weight(root+'.'+name+'.lora_B.weight')
   def project(q,sx,ax):return bf(np.add(bf(integer(q,sx,root+'.'+name+'.weight')),bf(np.multiply(np.float32(2),ordered(ax,w)))))
   full=project(q,sx,ax);shared=project(q[first],sx[first],ax[first])[index];assert np.array_equal(full.view('<u4'),shared.view('<u4'))
   assert np.isfinite(full).all();outputs[name]=full;count+=full.size
  output=d/f"{item['case']}.npz";np.savez(output,norm=norm,q=q,scales=sx,qa=qa,za=za,first=np.array(first),index=np.array(index),**outputs)
  row=dict(case=item['case'],tokens=len(ids),unique_rows=len(first),values=int(count),bit_equal=True,output=str(output.relative_to(ROOT)),output_sha256=sha(output),embedding_row_hashes=embedding_hashes);cases.append(row);print(json.dumps({k:v for k,v in row.items()if k not in ['embedding_row_hashes']}),flush=True)
 sources=[Path(__file__),ROOT/'scripts/reconstruct_terminal_mlp_reference.py',ROOT/'scripts/projection_row_reuse.rs',ROOT/'MODEL_LOCK.json',manifest,identity,paid_path]
 r=dict(complete=True,scope='Independent host first-layer embedding/norm/quantization/ordered LoRA and INT8 dots. Row compaction expands to original order and matches full host calculation. Does not establish current Wasm performance or full inference correctness.',cases=cases,weight_hashes=hashes,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources});(d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(d/'frozen-tests.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in sources+list(d.glob('*.npz'))+[d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
if __name__=='__main__':main()
