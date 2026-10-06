#!/usr/bin/env python3
"""Measure tail priming: partial MLP22, A23, MLP23 front, Delta24 heads, MLP24 front."""
import argparse,hashlib,io,json,pathlib,shutil,sys,zipfile
import numpy as np
if not __debug__:raise RuntimeError('Proof checker requires assertions')
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from proof_inputs import read_bytes,read_report,selections
from transport import Transport,decode,is_instruction_limit
from prefix_inference import verify_module
from mlp_attention_finish_codec import NAME as BRIDGE,FRONT,encode_request as bridge
from mlp_delta_stream_codec import NAME as PAIR,encode_request as pair,encode_continue_request as follow
from mlp_stream_codec import NAME as STREAM,length
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
ap.add_argument('--cases',default='617');ap.add_argument('--front',default='512,640,768');ap.add_argument('--heads',default='8,10,12');ap.add_argument('--next-front',default='4864,5120,5248,5376,5632');a=ap.parse_args()
labels=selections(a.cases,('617','insufficient','maximum'),'cases');fronts=list(map(int,selections(a.front,tuple(map(str,range(128,9216,128))),'front')));heads=list(map(int,selections(a.heads,tuple(map(str,range(2,32,2))),'heads')));nexts=list(map(int,selections(a.next_front,tuple(map(str,range(128,9216,128))),'next front')))
d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
if any(d.iterdir()):raise ValueError('Use a new evidence directory')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=list((ROOT/'client').glob('*.py'))+list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+list((ROOT/'canisters/inference/src').rglob('*.rs'))+[ROOT/p for p in ['Cargo.toml','Cargo.lock','crates/imajev-runtime/Cargo.toml','canisters/inference/Cargo.toml','MODEL_LOCK.json','checkpoints/full-int8.manifest.json','scripts/proof_inputs.py','scripts/build_full_prefix_candidate.py']]+[pathlib.Path(__file__)];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths};refs={};rows=[];m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_bytes());module=sha(ROOT/a.wasm);base=ROOT/'artifacts/prefix_codec/full-attention-q4-proof-v1';C,H,KV=2560,9216,2048
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec='mlp-down-state-exact-v1',frame_version=3)
def reference(label,layer):
 root=base/label;r=read_report(root/'report.json',ROOT,refs);q=next(q for q in r['queries']if q['op']=='mlp_full_integer'and f'.layers.{layer}.'in q['tensor']);h,x=decode(read_bytes(root/'queries'/f"{q['index']:06d}.request.bin",ROOT,refs));_,y=decode(read_bytes(root/'queries'/f"{q['index']:06d}.response.bin",ROOT,refs));return h,x,y
def state(label,layer,prefix=False):
 path=(base/'prefix'if prefix else base/label)/'queries/states'/f'layer-{layer:02d}.npz'
 with np.load(io.BytesIO(read_bytes(path,ROOT,refs)),allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def header(h,op,codec,dims):return dict(h,version=3,step=t.index,op=op,encoding=codec,dims=dims)
def run(h,packet,scope):
 index=t.index
 try:y=t._run_encoded(h,packet)
 except RuntimeError as e:
  if not is_instruction_limit(e):raise
  failed=d/f'failed-{scope}.request.bin';shutil.copyfile(d/f'{index:06d}.request.bin',failed)
  row=dict(scope=scope,success=False,stage=h['op'],error=str(e),failed_request=failed.name,failed_request_sha256=sha(failed));rows.append(row);print(json.dumps(row),flush=True);return None
 metric=t.measurements[-1];(d/f'{index:06d}.metric.json').write_text(json.dumps(metric,indent=2)+'\n');return y,metric
try:
 verify_module(t,module)
 for label in labels:
  h22,x22,b22=reference(label,22);h23,x23,b23=reference(label,23);h24,x24,b24=reference(label,24);n=h22['dims'][0];p=45
  z=state(label,23,True);prefix=np.concatenate([z['keys'].transpose(1,0,2).ravel(),z['values'].transpose(1,0,2).ravel()]);kv=state(label,23);expected_kv=np.concatenate([kv['keys'][p:].ravel(),kv['values'][p:].ravel()]);cvfinal=state(label,24)['conv'];delta=state(label,24,True);log=delta['delta_log'];cv=delta['conv']
  t.wire_codec='mlp-down-state-exact-v1';prepared=t.run('mlp_prepare_partial_down',x22,[n,C,1280],[2.,1e-6],tensor=h22['tensor'],aux=h22['aux'],input_hash=h22['input_hash']);controls23={};controls24={}
  t.wire_codec=STREAM
  for b in fronts:controls23[b]=t.run('mlp_stream_prepare',x23,[n,0,b],[2.,1e-6],tensor=h23['tensor'],aux=h23['aux'],input_hash=h23['input_hash'])
  for b in nexts:controls24[b]=t.run('mlp_stream_prepare',x24,[n,0,b],[2.,1e-6],tensor=h24['tensor'],aux=h24['aux'],input_hash=h24['input_hash'])
  for b in fronts:
   h=header(h22,FRONT,BRIDGE,[n,p,1280,b]);first=run(h,bridge(h,prepared,prefix),f'{label}-b{b}-bridge')
   if first is None:continue
   y,metric=first;end=len(y)-n*KV;assert y[:n*C].tobytes()==b22[:n*C].tobytes();assert y[n*C:end].tobytes()==controls23[b].tobytes();assert y[end:].tobytes()==expected_kv.tobytes();carry=y[n*C:end]
   rows.append(dict(scope=f'{label}-b{b}-bridge',success=True,stage=FRONT,full_bitwise_equal=True,call=metric))
   for k in heads:
    cols=lambda lo,hi:np.concatenate([np.arange(lo//2*128,hi//2*128),np.arange(2048+lo//2*128,2048+hi//2*128),np.arange(4096+lo*128,4096+hi*128)])
    logs=lambda lo,hi:np.concatenate([log[:p*2048].reshape(p,2048)[:,lo//2*128:hi//2*128].ravel(),log[p*2048:p*6144].reshape(p,4096)[:,lo*128:hi*128].ravel(),log[p*6144:].reshape(p,32)[:,lo:hi].ravel()])
    f,r=cols(0,k),cols(k,32);h=header(h23,'mlp_complete_delta_partial',PAIR,[n,b,H-b,k,p]);second=run(h,pair(h,carry,cv[:,f],logs(0,k)),f'{label}-b{b}-h{k}-pair')
    if second is None:continue
    y,metric2=second;assert y[:n*C].tobytes()==b23[:n*C].tobytes();assert y[-3*k*256:].tobytes()==cvfinal[:,f].ravel().tobytes();rows.append(dict(scope=f'{label}-b{b}-h{k}-pair',success=True,stage=h['op'],full_bitwise_equal=True,call=metric2))
    for next_b in nexts:
     h=header(h23,'delta_partial_mlp_front',PAIR,[n,b,H-b,k,p,next_b]);scope=f'{label}-b{b}-h{k}-next{next_b}'
     try:packet=follow(h,y,cv[:,r],logs(k,32),compress_base=True,compress_prefix=True)
     except ValueError as e:
      if 'frame bounds'not in str(e):raise
      saved=d/f'failed-{scope}.header.json';saved.write_text(json.dumps(h,indent=2)+'\n');row=dict(scope=scope,success=False,stage='front_frame',frame_limit=True,error=str(e),header_file=saved.name,carry_response=f"{metric2['index']:06d}.response.bin");rows.append(row);print(json.dumps(row),flush=True);continue
     third=run(h,packet,scope)
     if third is None:continue
     out,metric3=third;assert out[:length(n,next_b)].tobytes()==controls24[next_b].tobytes();assert out[length(n,next_b):].tobytes()==cvfinal[:,r].ravel().tobytes();record=dict(scope=f'{label}-b{b}-h{k}-next{next_b}',label=label,front=b,heads=k,next_front=next_b,success=True,full_bitwise_equal=True,calls=[first[1],second[1],third[1]],three_query_instructions=sum(c['ok']['instructions']for c in [first[1],second[1],third[1]]));rows.append(record);print(json.dumps(record),flush=True)
 verify_module(t,module)
 assert sha(ROOT/a.wasm)==module and hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths} and all(sha(ROOT/p)==v for p,v in refs.items())
 (d/'report.json').write_text(json.dumps(dict(settings=vars(a),module_sha256=module,source_hashes=hashes,reference_hashes=refs,cases=rows,measurements=t.measurements,whole_inference_reduction_verified=False,requested_chains=len(labels)*len(fronts)*len(heads)*len(nexts),completed_chains=sum('calls'in row for row in rows),failed_attempts=sum(not row['success']for row in rows)),indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
