#!/usr/bin/env python3
"""Exact signed coefficient/codec capacity for real Q and MLP weights; no inference changes."""
from pathlib import Path
import json,hashlib,numpy as np,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 d=ROOT/'artifacts/precomputed-coefficient-capacity-v1';d.mkdir(exist_ok=False);mp=ROOT/'checkpoints/full-int8.manifest.json';m=json.loads(mp.read_text());assert sha(ROOT/'MODEL_LOCK.json')==m['model'];pack=ROOT/'checkpoints/full-int8.pack';assert pack.stat().st_size==m['bytes'];assert sha(pack)==m['pack_hash'];print('model/pack verified',flush=True)
 plans=[('rank7-winograd',2,ROOT/'artifacts/s1-k2-mlp160-v1/plan.py'),('rank49-winograd',4,ROOT/'artifacts/s2-k2-alias-kernels-v1/plan.py'),('rank343-classical',8,ROOT/'scripts/generate_s3_stream_probe.py'),('rank343-winograd',8,ROOT/'artifacts/s3-winograd-v1/frozen-generator.py')];names=['model.language_model.layers.3.self_attn.q_proj.weight','model.language_model.layers.3.mlp.gate_proj.weight','model.language_model.layers.3.mlp.down_proj.weight'];cases=[];files=[Path(__file__),mp,ROOT/'MODEL_LOCK.json']+[p for _,_,p in plans]
 for name in names:
  t=next(t for t in m['tensors']if t['name']==name);rows,cols=t['rows'],t['cols'];assert rows%32==cols%256==0
  with pack.open('rb')as f:f.seek(t['offset']);raw=f.read(rows*cols)
  weights=np.frombuffer(raw,dtype=np.int8).reshape(rows,cols)
  for label,n,path in plans:
   ns={'__name__':'capacity','__file__':str(path)};exec(compile(path.read_text(),str(path),'exec'),ns);a,b,c,leaves,roots,*_=ns['plan']();rank=len(leaves);assert max(sum(abs(v)for v in x.values())for x in b.symbols.values())*128<32768
   counts=dict(vectors=0,wide_vectors=0,values=0,wide_values=0,minimum=0,maximum=0);hist=[0]*rank
   # Each vector contains two K values for each of four independent output lanes.
   for first in range(0,rows,n*4*8):
    chunk=weights[first:first+n*4*8].astype(np.int16);g=chunk.shape[0]//(n*4);blocks=cols//256;segment=256//n
    bases=chunk.reshape(g,4,n,blocks,n,segment//2,2).transpose(4,2,0,3,5,1,6).reshape(n*n,g,blocks,segment//2,8)
    variables={f'b{i}':bases[i]for i in range(n*n)};variables['zero']=np.zeros_like(bases[0])
    for node,terms in b.nodes:
     v=np.zeros_like(bases[0])
     for parent,sign in terms.items():v+=variables[parent]*sign
     variables[node]=v
    for mi,(_,bn)in enumerate(leaves):
     values=variables[bn].reshape(-1,8);wide=np.any((values<-128)|(values>127),axis=1);narrow=values[~wide].astype(np.int8);escape=values[wide].astype('<i2');flags=np.packbits(wide,bitorder='little');decoded_wide=np.unpackbits(flags,bitorder='little')[:len(values)].astype(bool);restored=np.empty_like(values);restored[~decoded_wide]=np.frombuffer(narrow.tobytes(),np.int8).reshape(-1,8);restored[decoded_wide]=np.frombuffer(escape.tobytes(),'<i2').reshape(-1,8);assert np.array_equal(restored,values)
     counts['vectors']+=len(values);counts['wide_vectors']+=int(wide.sum());counts['values']+=values.size;counts['wide_values']+=int(((values<-128)|(values>127)).sum());counts['minimum']=min(counts['minimum'],int(values.min()));counts['maximum']=max(counts['maximum'],int(values.max()));hist[mi]+=int(wide.sum())
   raw_bytes=rows*cols;payload=counts['vectors']*8+counts['wide_vectors']*8;flag_bytes=(counts['vectors']+7)//8;item=dict(tensor=name,rows=rows,cols=cols,plan=label,rank=rank,raw_weight_bytes=raw_bytes,raw_weight_sha256=hashlib.sha256(raw).hexdigest(),counts=counts,wide_vector_fraction=counts['wide_vectors']/counts['vectors'],payload_bytes=payload,flag_bytes=flag_bytes,minimum_codec_bytes=payload+flag_bytes,codec_ratio=(payload+flag_bytes)/raw_bytes,dense_i16_bytes=counts['values']*2,per_leaf_wide_vectors=hist,all_codec_bytes_decode_exact=True,scope='Two sequential streams: signed I8 narrow vectors, little-endian I16 wide vectors, packed one-bit selector. No random-access index/allocator/metadata overhead included; no Wasm cost claim.')
   cases.append(item);print(json.dumps({k:item[k]for k in ['tensor','plan','wide_vector_fraction','codec_ratio','minimum_codec_bytes']}),flush=True)
 projections=[t for t in m['tensors']if t['dtype']=='int8'and 'embed_tokens'not in t['name']and t['cols']%256==0 and t['rows']%32==0];projection_bytes=sum(t['rows']*t['cols']for t in projections);non_embedding=sum(t['bytes']for t in m['tensors']if 'embed_tokens'not in t['name']);other=non_embedding-projection_bytes
 lower=[]
 for label,n,_ in plans:
  rank=dict((k,len(cases[[c['plan']for c in cases].index(k)]['per_leaf_wide_vectors']))for k,_,_ in plans)[label];prepared=projection_bytes*rank//(n*n);assert projection_bytes*rank%(n*n)==0;ideal=prepared+other;lower.append(dict(plan=label,rank=rank,ideal_all_I8_projection_bytes=prepared,other_resident_tensor_bytes=other,ideal_total_tensor_bytes=ideal,fits_6GiB_without_work_buffers=ideal<=6*2**30))
 r=dict(complete=True,model=m['model'],pack_hash=m['pack_hash'],selected_real_tensors=names,all_codec_roundtrips_exact=True,projection_raw_bytes=projection_bytes,all_non_embedding_tensor_bytes=non_embedding,whole_model_ideal_lower_bounds=lower,cases=cases,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Host-only exact capacity evidence. Ideal lower bounds omit selectors, escapes, runtime/working buffers. No performance gain or model/canister migration claim.')
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(d/'evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,lower_bounds=lower)),flush=True)
if __name__=='__main__':main()
