#!/usr/bin/env python3
"""Retain full-model evidence without committing weights or large raw journals."""
import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text())
r=read('artifacts/full-int8-canister/first-report.json')
r['queries']=[dict(index=q['index'],op=q['op'],tensor=q['tensor'],**q['ok'],wall_seconds=q['wall_seconds']) for q in r['queries']]
r['raw_report_sha256']=hashlib.sha256((ROOT/'artifacts/full-int8-canister/first-report.json').read_bytes()).hexdigest()
r['layer_errors']=read('artifacts/full-int8-canister/layer-comparison.json')
r['native_wasm']=read('artifacts/full-int8-canister/native-wasm.json')
r['upload']=read('artifacts/full-int8-upload/report.json')
r['pack_bytes']=4702451200
ref=read('artifacts/reference-first.json')['records'][0]['result']['scores'];d=r['decision_query']['ok']['decision']
r['comparison']['probability_max_error']=max(abs(a-ref[k]) for a,k in zip(d['probabilities'],['unlikely','possible','likely']))
r['comparison']['unknown_probability_error']=abs(d['unknown_probability']-ref['__unknown__'])
r['limitations']=['Single 132-token question, rotations=1; not general judgment accuracy','Difference includes INT8 and unquantized Rust/MLX arithmetic differences; no full BF16 A/B','Candid bytes exclude HTTP/CBOR/signatures','Handler instructions exclude CDK decode/encode','Heap is observed end-of-handler pages, not instantaneous peak','INT8 tiles dequantized to F32; no integer dot; intermediate state is not INT8','Local single run, query cache and other workload not controlled']
(ROOT/'docs/full-results.json').write_text(json.dumps(r,indent=2)+'\n')
print({k:r[k] for k in ['query_count','total_instructions','total_candid_bytes','wall_seconds_this_run','comparison']})
