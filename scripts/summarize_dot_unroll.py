#!/usr/bin/env python3
"""Shape-weighted candidate instructions against the same recovered local environment."""
import collections,hashlib,json,pathlib,struct
ROOT=pathlib.Path(__file__).resolve().parents[1];baseline=json.loads((ROOT/'artifacts/bottleneck/dot-new-upgraded-baseline/report.json').read_text());old=json.loads((ROOT/'artifacts/bottleneck-fused-full-617/first-report.json').read_text());counts=collections.Counter()
for q in old['queries']:
 if 'integer' not in q['op']:continue
 with (ROOT/f"artifacts/bottleneck-fused-full-617/queries/{q['index']:06d}.request.bin").open('rb') as f:
  size,=struct.unpack('<I',f.read(4));h=json.loads(f.read(size))
 counts[(q['op'],tuple(h['dims'][:3]))]+=1
base={(c['op'],tuple(c['dims'][:3])):c for c in baseline['cases'] if 'integer' in c['op']};trials=[]
for group in [8,16,32]:
 path=ROOT/f'artifacts/bottleneck/dot-g{group}/report.json';r=json.loads(path.read_text());cases=[]
 for c in r['cases']:
  k=(c['op'],tuple(c['dims'][:3]))
  if k not in base:continue
  previous=base[k]['normal']['instructions'];after=c['normal']['instructions'];cases.append(dict(op=c['op'],dims=c['dims'],count=counts[k],before_instructions=previous,after_instructions=after,reduction=1-after/previous,bitwise_equal=c['bitwise_equal']))
 before=sum(c['count']*c['before_instructions'] for c in cases);after=sum(c['count']*c['after_instructions'] for c in cases)
 trials.append(dict(columns_per_update=group*8,wasm_sha256=r['wasm_sha256'],weighted_projection_reduction=1-after/before,weighted_before_instructions=before,weighted_after_instructions=after,queries_covered=sum(c['count'] for c in cases),cases=cases,report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
result=dict(scope='Actual same-input queries and same upgrade state; projection shape counts extrapolated from 617 full graph; full totals measured separately',canister=baseline['canister'],url=baseline['url'],baseline_wasm_sha256=baseline['wasm_sha256'],model=old['model'],pack_hash=old['pack_hash'],trials=trials,selected_columns_per_update=256)
(ROOT/'docs/dot-unroll-candidates.json').write_text(json.dumps(result,indent=2)+'\n');print([(t['columns_per_update'],t['weighted_projection_reduction']) for t in trials])
