#!/usr/bin/env python3
"""Real MLP streaming equivalence; diagnostic calls are not a reduced inference graph."""
import argparse,hashlib,json,pathlib,sys,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from proof_inputs import read_report
from transport import Transport,decode
from prefix_inference import verify_module
from mlp_stream_codec import NAME,C,H,R
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
ap.add_argument('--half',action='store_true',help='Exercise unquantized BF16 pending128 carry, retaining block256 scales')
ap.add_argument('--complete',action='store_true',help='Fuse last chunk and finish, compare with frozen full MLP')
ap.add_argument('--source-case',choices=['617','insufficient','maximum'],default='617')
a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();wasm=sha(ROOT/a.wasm)
references={};source=ROOT/'artifacts/prefix_codec/full-capture-prepared-proof'/a.source_case;report=read_report(source/'report.json',ROOT,references);m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_bytes())
paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+list((ROOT/'canisters/inference/src').rglob('*.rs'))+list((ROOT/'client').glob('*.py'))+[pathlib.Path(__file__),ROOT/'scripts/proof_inputs.py',ROOT/'crates/imajev-runtime/Cargo.toml',ROOT/'canisters/inference/Cargo.toml',ROOT/'MODEL_LOCK.json',ROOT/'Cargo.lock',ROOT/'checkpoints/full-int8.manifest.json'];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths};cases=[]
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec=NAME,frame_version=3)
try:
 verify_module(t,wasm)
 for layer in [0,1,3]:
  q=next(q for q in report['queries']if q['op']=='mlp_full_integer'and f'.layers.{layer}.'in q['tensor']);request=source/'queries'/f"{q['index']:06d}.request.bin";response=source/'queries'/f"{q['index']:06d}.response.bin";references[str(request.relative_to(ROOT))]=sha(request);references[str(response.relative_to(ROOT))]=sha(response);h,x=decode(request.read_bytes());_,expected=decode(response.read_bytes());n=h['dims'][0]
  t.wire_codec='mlp-down-state-exact-v1';prepared=t.run('mlp_prepare_down',x,[n,C],[2.,1e-6],tensor=h['tensor'],aux=h['aux'],input_hash=h['input_hash']);t.wire_codec=NAME
  for chunks in ([[6016,3200],[128,256,128,8704]] if a.half else [[2048,7168],[3840,5376],[4096,5120],[4608,4608],[2048,2048,5120]]):
   baseline=None
   if a.complete:
    old=x;old_begin=0;old_start=len(t.measurements)
    for old_count in chunks:
     old=t.run('mlp_stream_prepare'if old_begin==0 else'mlp_stream_next',old,[n,old_begin,old_count],[2.,1e-6],tensor=h['tensor'],aux=h['aux'],input_hash=h['input_hash']);old_begin+=old_count
    old=t.run('mlp_stream_finish',old,[n,H,0],[2.,1e-6],tensor=h['tensor'],aux=h['aux'],input_hash=h['input_hash'])
    assert old.tobytes()==expected.tobytes()
    old_calls=t.measurements[old_start:]
    baseline=dict(ordinary_queries=len(old_calls),total_instructions=sum(c['ok']['instructions']for c in old_calls),candid_bytes=sum(c['ok']['request_bytes']+c['ok']['reply_bytes']for c in old_calls),calls=old_calls)
   state=x;begin=0;start=len(t.measurements);frames=[]
   for chunk_index,count in enumerate(chunks):
    if a.complete and chunk_index==len(chunks)-1:
     state=t.run('mlp_stream_complete',state,[n,begin,count],[2.,1e-6],tensor=h['tensor'],aux=h['aux'],input_hash=h['input_hash']);begin+=count;frames.append(t.index-1);break
    state=t.run('mlp_stream_prepare'if begin==0 else'mlp_stream_next',state,[n,begin,count],[2.,1e-6],tensor=h['tensor'],aux=h['aux'],input_hash=h['input_hash']);begin+=count;frames.append(t.index-1)
   assert begin==H
   if a.complete:
    y=state
   else:
    prefix=n*(2*C+C//256+2*R);equivalent=np.concatenate([state[:n*C],state[prefix:]]);assert equivalent.tobytes()==prepared.tobytes()
    y=t.run('mlp_stream_finish',state,[n,H,0],[2.,1e-6],tensor=h['tensor'],aux=h['aux'],input_hash=h['input_hash']);frames.append(t.index-1)
   assert y.tobytes()==expected.tobytes()
   calls=t.measurements[start:];row=dict(layer=layer,tokens=n,chunks=chunks,bitwise_prepared_equal=None if a.complete else True,fused_complete=a.complete,bitwise_hidden_norm_equal=True,source_request_sha256=sha(request),source_response_sha256=sha(response),ordinary_queries=len(calls),total_instructions=sum(c['ok']['instructions']for c in calls),max_query_instructions=max(c['ok']['instructions']for c in calls),candid_bytes=sum(c['ok']['request_bytes']+c['ok']['reply_bytes']for c in calls),calls=calls,profile_diagnostics=[])
   row['same_module_split_baseline']=baseline
   if baseline:
    row['instruction_delta']=row['total_instructions']-baseline['total_instructions'];row['candid_bytes_delta']=row['candid_bytes']-baseline['candid_bytes']
   # Measure preparation/reuse separately on the exact same state frames.
   if chunks==[4608,4608]:
    for index in frames:
     req=d/f'{index:06d}.request.bin';out=d/f'{index:06d}.profile.response.bin';metric=t.command(dict(diagnostics=True,op='profile',input=str(req),output=str(out)));assert decode(out.read_bytes())[1].tobytes()==decode((d/f'{index:06d}.response.bin').read_bytes())[1].tobytes();row['profile_diagnostics'].append(metric)
   cases.append(row);print(json.dumps(row),flush=True)
 verify_module(t,wasm);assert hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths};assert all(sha(ROOT/p)==v for p,v in references.items())
 result=dict(scope=__doc__,source_case=a.source_case,wasm_sha256=wasm,source_hashes=hashes,reference_hashes=references,half_boundaries=a.half,cases=cases,regular_diagnostic_queries=len(t.measurements),profile_diagnostic_queries=sum(len(c["profile_diagnostics"])for c in cases),goal_50_verified=False)
 (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
