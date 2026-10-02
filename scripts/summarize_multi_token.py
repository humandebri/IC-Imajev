#!/usr/bin/env python3
"""Compare actual same-request F32 tile experiments; retain source/module identity."""
import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
phases=['multi-token-before','multi-token8','multi-token16','multi-token32','multi-token64']
reports={p:json.loads((ROOT/'artifacts/linear-tiling'/p/'report.json').read_text()) for p in phases}
base=reports[phases[0]];cases=[]
for j,old in enumerate(base['cases']):
 values={}
 for p,r in reports.items():
  x=r['cases'][j];assert x['request_sha256']==old['request_sha256'] and x['bitwise_equal']
  values[p]=dict(instructions=x['instructions'],reduction_vs_before=1-x['instructions']/old['instructions'])
 cases.append(dict(index=old['index'],tensor=old['tensor'],dims=old['dims'],candidates=values))
result=dict(scope='Eight actual projection requests; F32 multiply/add and column order preserved; candidate results are not full-graph measurements',model=base['model'],pack_hash=base['pack_hash'],query_cache_controlled=False,wasm_hashes={p:r['wasm_sha256'] for p,r in reports.items()},report_hashes={p:hashlib.sha256((ROOT/'artifacts/linear-tiling'/p/'report.json').read_bytes()).hexdigest() for p in phases},cases=cases)
(ROOT/'docs/multi-token-candidates.json').write_text(json.dumps(result,indent=2)+'\n')
print({p:1-sum(r['instructions'] for r in reports[p]['cases'])/sum(r['instructions'] for r in base['cases']) for p in phases})
