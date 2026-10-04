#!/usr/bin/env python3
"""Measure MLP23 priming followed by Delta24 and partial MLP24 down."""
import argparse,hashlib,io,json,pathlib,shutil,sys,zipfile
import numpy as np
if not __debug__:raise RuntimeError('Proof checker requires assertions')
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from proof_inputs import read_bytes,read_report,selections
from transport import Transport,decode,is_instruction_limit
from prefix_inference import verify_module
from mlp_attention_finish_codec import NAME as BRIDGE,FRONT,encode_request as bridge
from mlp_delta_stream_codec import NAME as PAIR,encode_request as pair,encode_continue_request as follow
from mlp_stream_codec import NAME as STREAM
from mlp_codec import NAME as DOWN
from mlp_delta_carry import HUFFMAN_NAME,encode_request as finish
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
ap.add_argument('--cases',default='617');ap.add_argument('--front',default='4096,4352');ap.add_argument('--heads',default='18,20,22');ap.add_argument('--down-rows',default='1280,1600,1920');ap.add_argument('--delta-front');ap.add_argument('--next-heads');ap.add_argument('--next-mlp-front',type=int,default=6912);a=ap.parse_args()
labels=selections(a.cases,('617','insufficient','maximum'),'cases');fronts=list(map(int,selections(a.front,tuple(map(str,range(128,9216,128))),'front')));heads=list(map(int,selections(a.heads,tuple(map(str,range(2,32,2))),'heads')));nexts=list(map(int,selections(a.down_rows,tuple(map(str,range(32,2560,32))),'down rows')))
fourfronts=[]if a.delta_front is None else list(map(int,selections(a.delta_front,tuple(map(str,range(128,9216,128))),'Delta front')))
nextheads=[]if a.next_heads is None else list(map(int,selections(a.next_heads,tuple(map(str,range(2,32,2))),'next heads')))
if nextheads and (not fourfronts or not 0<a.next_mlp_front<9216 or a.next_mlp_front%128):raise ValueError('next heads require Delta front and valid next MLP front')
d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
if any(d.iterdir()):raise ValueError('Use a new evidence directory')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=list((ROOT/'client').glob('*.py'))+list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'canisters/inference/src').rglob('*.rs'))+[ROOT/p for p in ['Cargo.toml','Cargo.lock','crates/imajev-runtime/Cargo.toml','canisters/inference/Cargo.toml','MODEL_LOCK.json','checkpoints/full-int8.manifest.json','scripts/proof_inputs.py','scripts/build_full_prefix_candidate.py']]+[pathlib.Path(__file__)];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths};refs={};rows=[];m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_bytes());module=sha(ROOT/a.wasm);base=ROOT/'artifacts/prefix_codec/full-attention-q4-proof-v1';C,H,KV=2560,9216,2048
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
  t.wire_codec=DOWN
  for b in nexts:controls24[b]=t.run('mlp_prepare_partial_down',x24,[n,C,b],[2.,1e-6],tensor=h24['tensor'],aux=h24['aux'],input_hash=h24['input_hash'])
  controls25={}
  if fourfronts:
   h25,x25,b25=reference(label,25);prefix25=state(label,25,True);conv25=state(label,25)['conv'];t.wire_codec=STREAM
   for f25 in fourfronts:controls25[f25]=t.run('mlp_stream_prepare',x25,[n,0,f25],[2.,1e-6],tensor=h25['tensor'],aux=h25['aux'],input_hash=h25['input_hash'])
   if nextheads:
    h26,x26,b26=reference(label,26);prefix26=state(label,26,True);final26=state(label,26)['conv'];control26=t.run('mlp_stream_prepare',x26,[n,0,a.next_mlp_front],[2.,1e-6],tensor=h26['tensor'],aux=h26['aux'],input_hash=h26['input_hash'])
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
     h=header(h23,'delta_partial_mlp_prepare_down',PAIR,[n,b,H-b,k,p,next_b]);scope=f'{label}-b{b}-h{k}-next{next_b}'
     try:packet=follow(h,y,cv[:,r],logs(k,32),compress_base=True,compress_prefix=True)
     except ValueError as e:
      if 'frame bounds'not in str(e):raise
      saved=d/f'failed-{scope}.header.json';saved.write_text(json.dumps(h,indent=2)+'\n');row=dict(scope=scope,success=False,stage='front_frame',frame_limit=True,error=str(e),header_file=saved.name,carry_response=f"{metric2['index']:06d}.response.bin");rows.append(row);print(json.dumps(row),flush=True);continue
     third=run(h,packet,scope)
     if third is None:continue
     out,metric3=third;assert out[:n*(C+H+100)].tobytes()==controls24[next_b].tobytes();assert out[n*(C+H+100):].tobytes()==cvfinal[:,r].ravel().tobytes();record=dict(scope=f'{label}-b{b}-h{k}-next{next_b}',label=label,front=b,heads=k,down_rows=next_b,success=True,full_bitwise_equal=True,calls=[first[1],second[1],third[1]],three_query_instructions=sum(c['ok']['instructions']for c in [first[1],second[1],third[1]]));rows.append(record);print(json.dumps(record),flush=True)
     for f25 in fourfronts:
      h=header(h24,'mlp_finish_delta_log_mlp_front',HUFFMAN_NAME,[n,p,next_b,f25]);scope=f'{label}-b{b}-h{k}-down{next_b}-dfront{f25}'
      fourth=run(h,finish(h,out[:n*(C+H+100)],prefix25['conv'],prefix25['delta_log']),scope)
      if fourth is None:continue
      y4,metric4=fourth;assert y4[:n*C].tobytes()==b24[:n*C].tobytes();assert y4[n*C:-24576].tobytes()==controls25[f25].tobytes();assert y4[-24576:].tobytes()==conv25.ravel().tobytes()
      record4=dict(scope=scope,success=True,full_bitwise_equal=True,calls4=[first[1],second[1],third[1],metric4],down_rows=next_b,delta_front=f25);rows.append(record4);print(json.dumps(record4),flush=True)
      for next_k in nextheads:
       log26=prefix26['delta_log'];cv26=prefix26['conv']
       def log_group(lo,hi):return np.concatenate([log26[:p*2048].reshape(p,2048)[:,lo//2*128:hi//2*128].ravel(),log26[p*2048:p*6144].reshape(p,4096)[:,lo*128:hi*128].ravel(),log26[p*6144:].reshape(p,32)[:,lo:hi].ravel()])
       first_cols,rest_cols=cols(0,next_k),cols(next_k,32)
       h=header(h25,'mlp_complete_delta_partial',PAIR,[n,f25,H-f25,next_k,p]);scope5=scope+f'-heads{next_k}'
       fifth=run(h,pair(h,y4[n*C:-24576],cv26[:,first_cols],log_group(0,next_k)),scope5)
       if fifth is None:continue
       y5,metric5=fifth;assert y5[:n*C].tobytes()==b25[:n*C].tobytes();assert y5[-3*next_k*256:].tobytes()==final26[:,first_cols].ravel().tobytes()
       rec5=dict(scope=scope5,success=True,full_bitwise_equal=True,calls5=record4['calls4']+[metric5]);rows.append(rec5);print(json.dumps(rec5),flush=True)
       h=header(h25,'delta_partial_mlp_front',PAIR,[n,f25,H-f25,next_k,p,a.next_mlp_front]);scope6=scope5+'-front'
       try:packet=follow(h,y5,cv26[:,rest_cols],log_group(next_k,32),compress_base=True,compress_prefix=True)
       except ValueError as e:
        if 'frame bounds'not in str(e):raise
        saved=d/f'failed-{scope6}.header.json';saved.write_text(json.dumps(h,indent=2)+'\n');row=dict(scope=scope6,success=False,stage='front_frame',frame_limit=True,error=str(e),header_file=saved.name,carry_response=f"{metric5['index']:06d}.response.bin");rows.append(row);print(json.dumps(row),flush=True);continue
       sixth=run(h,packet,scope6)
       if sixth is None:continue
       y6,metric6=sixth;length26=control26.size;assert y6[:length26].tobytes()==control26.tobytes();assert y6[length26:].tobytes()==final26[:,rest_cols].ravel().tobytes()
       rec6=dict(scope=scope6,success=True,full_bitwise_equal=True,calls6=rec5['calls5']+[metric6],next_mlp_front=a.next_mlp_front);rows.append(rec6);print(json.dumps(rec6),flush=True)
 verify_module(t,module)
 assert sha(ROOT/a.wasm)==module and hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths} and all(sha(ROOT/p)==v for p,v in refs.items())
 (d/'report.json').write_text(json.dumps(dict(settings=vars(a),module_sha256=module,source_hashes=hashes,reference_hashes=refs,cases=rows,measurements=t.measurements,whole_inference_reduction_verified=False,requested_chains=len(labels)*len(fronts)*len(heads)*len(nexts),completed_six_query_chains=sum('calls6'in row for row in rows),completed_five_query_chains=sum('calls5'in row for row in rows),completed_four_query_chains=sum('calls4'in row for row in rows),completed_chains=sum('calls'in row for row in rows),failed_attempts=sum(not row['success']for row in rows)),indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
