#!/usr/bin/env python3
"""Verify exact boundary carries and full regression evidence; primary remains62."""
import hashlib
import json
import pathlib
import sys
import zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import decode,frame_digest
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=ROOT/'artifacts/prefix_codec/full-build-mlp-delta-stream-v3';module=sha(build/'full.wasm');sources=json.loads((build/'source-hashes.json').read_text());successful=[];failures=0;ordinary=0;profiles=0;reports={}
for folder in ['pair-follow-check-617','pair-follow-check-insufficient','pair-follow-check-maximum','pair-follow-check-maximum-cuts']:
 d=ROOT/'artifacts/f32_k_continue'/folder;r=json.loads((d/'report.json').read_text());assert r['wasm_sha256']==module and r['goal_50_verified'] is False
 with zipfile.ZipFile(d/'validated-source.zip')as z:
  for name,expected in r['source_hashes'].items():
   assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
   if name in sources:assert sources[name]==expected,name
 source=ROOT/'artifacts/prefix_codec/full-capture-prepared-proof'/r['source_case'];sr=json.loads((source/'report.json').read_text());old=ROOT/f"artifacts/column16-v1-{r['source_case']}";oldr=json.loads((old/'report.json').read_text());ordinary+=r['regular_successful_diagnostic_queries']
 for c in r['cases']:
  if not c['success']:
   failures+=1;assert c['instruction_limit'] and '5000000000'in c['error'];request=d/f"failed-layer{c['layer']}-begin{c['begin']}-heads{c['heads']}.request.bin";assert sha(request)==c['failed_request_sha256'];continue
  n,k=c['tokens'],c['heads'];idx=c['call']['index'];fh=c['follow_call']['index'];C=2560;R=64;H=9216;remain=32-k
  request=d/f'{idx:06d}.request.bin';raw=request.read_bytes();count=int.from_bytes(raw[:4],'little');head=json.loads(raw[4:4+count]);assert frame_digest(head,raw[:-32])==raw[-32:];assert len(raw)==c['request_frame_bytes']<2_000_000
  follow=d/f'{fh:06d}.request.bin';raw=follow.read_bytes();count=int.from_bytes(raw[:4],'little');head=json.loads(raw[4:4+count]);assert frame_digest(head,raw[:-32])==raw[-32:];assert head['op']=='delta_partial_mlp_prepare' and len(raw)==c['follow_request_frame_bytes']<2_000_000
  _,y=decode((d/f'{idx:06d}.response.bin').read_bytes());_,fy=decode((d/f'{fh:06d}.response.bin').read_bytes())
  original=next(q for q in sr['queries']if q['op']=='mlp_full_integer'and f".layers.{c['layer']}."in q['tensor']);req=source/'queries'/f"{original['index']:06d}.request.bin";res=source/'queries'/f"{original['index']:06d}.response.bin";assert sha(req)==c['source_request_sha256'] and sha(res)==c['source_response_sha256'];_,expected=decode(res.read_bytes());assert y[:n*C].tobytes()==expected[:n*C].tobytes()
  tensor=f"model.language_model.layers.{c['next_layer']}.linear_attn.in_proj_qkv.weight";capture=next(q for q in oldr['queries']if q['tensor']==tensor and q['op']=='delta_project_capture');creq=old/'queries'/f"{capture['index']:06d}.request.bin";cres=old/'queries'/f"{capture['index']:06d}.response.bin";assert sha(creq)==c['capture_request_sha256'] and sha(cres)==c['capture_response_sha256'];_,saved=decode(cres.read_bytes());assert y[n*C:n*(C+2762)].tobytes()==saved[n*2048+3*4096:].tobytes()
  _,base=decode((d/f'{idx+1:06d}.response.bin').read_bytes());_,ax=decode((d/f'{idx+2:06d}.response.bin').read_bytes());assert y[n*(C+2762):n*(2*C+2762)].tobytes()==base.tobytes();assert y[n*(2*C+2762):n*(2*C+2762+R)].tobytes()==ax.tobytes()
  _,prepared=decode((d/f'{fh+1:06d}.response.bin').read_bytes());_,finished=decode((d/f'{fh+2:06d}.response.bin').read_bytes());assert fy[:n*(C+H+100)].tobytes()==prepared.tobytes()
  nextq=next(q for q in sr['queries']if q['op']=='mlp_full_integer'and f".layers.{c['next_layer']}."in q['tensor']);nextreq=source/'queries'/f"{nextq['index']:06d}.request.bin";nextres=source/'queries'/f"{nextq['index']:06d}.response.bin";assert sha(nextreq)==c['next_mlp_request_sha256'] and sha(nextres)==c['next_mlp_response_sha256'];_,expected=decode(nextres.read_bytes());assert finished.tobytes()==expected.tobytes()
  state=ROOT/f"artifacts/output-pairs-v2-{r['source_case']}/queries/states/layer-{c['next_layer']:02d}.npz";assert sha(state)==c['final_history_sha256']
  first=np.concatenate([np.arange(k//2*128),np.arange(2048,2048+k//2*128),np.arange(4096,4096+k*128)]);rest=np.concatenate([np.arange(k//2*128,2048),np.arange(2048+k//2*128,4096),np.arange(4096+k*128,8192)])
  with np.load(state,allow_pickle=False)as z:
   assert y[-3*k*256:].tobytes()==z['conv'][:,first].ravel().tobytes();assert fy[-3*remain*256:].tobytes()==z['conv'][:,rest].ravel().tobytes()
  for name,call_index in [('profile',idx),('follow_profile',fh)]:
   if name in c:
    profiles+=1;_,a=decode((d/f'{call_index:06d}.profile.response.bin').read_bytes());_,b=decode((d/f'{call_index:06d}.response.bin').read_bytes());assert a.tobytes()==b.tobytes()
  spans={name:(cost,count)for name,cost,count in c['follow_profile']['ok']['spans']};assert 'head_input_quantize_once'not in spans and 'lora_matmul_A'not in spans
  assert c['call']['ok']['instructions']<=5_000_000_000 and c['follow_call']['ok']['instructions']<=5_000_000_000;successful.append(c)
 reports[folder]=sha(d/'report.json')
full_dir=ROOT/'artifacts/prefix_codec/full-mlp-delta-stream-proof-v3';full=json.loads((full_dir/'report.json').read_text());assert full['wasm_sha256']==module and len(full['cases'])==6 and not full['goal_50_verified']
with zipfile.ZipFile(full_dir/'validated-source.zip')as z:
 for name,expected in full['source_hashes'].items():
  assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
  if name in sources:assert sources[name]==expected,name
for c in full['cases']:
 assert c['full_bitwise_equal'] and c['max_query_instructions']<=5_000_000_000;assert sha(full_dir/c['label']/'report.json')==c['report_sha256'];assert sha(ROOT/f"artifacts/output-pairs-v2-{c['baseline']}/report.json")==c['baseline_report_sha256']
main=next(c for c in full['cases']if c['label']=='617');assert main['query_count']==62
result=dict(scope=__doc__,module_sha256=module,report_hashes=reports,successful_boundary_cases=len(successful),instruction_limit_cases=failures,ordinary_successful_diagnostic_queries=ordinary,profile_diagnostic_queries=profiles,all_returned_carries_and_next_finished_hidden_norm_bitwise_equal=True,full_regression_report_sha256=sha(full_dir/'report.json'),actual_primary_queries=main['query_count'],goal_50_verified=False)
(ROOT/'artifacts/f32_k_continue/mlp-delta-stream-archive-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
