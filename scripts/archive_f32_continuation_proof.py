#!/usr/bin/env python3
"""Archive exact F32 continuation and its full-model regression separately."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=ROOT/'artifacts/prefix_codec/full-f32-k-continue-proof'
build=ROOT/'artifacts/prefix_codec/full-build-f32-k-continue-v2'
r=json.loads((d/'report.json').read_bytes())
b=json.loads((ROOT/'artifacts/prefix_codec/full-s1-wide-proof/report.json').read_bytes())
diagnostic=json.loads((ROOT/'artifacts/f32_k_continue/check/report.json').read_bytes())
assert r['wasm_sha256']==diagnostic['wasm_sha256']==sha(build/'full.wasm')
for hashes in [r['source_hashes'],diagnostic['source_hashes']]:
    assert all(sha(ROOT/p)==v for p,v in hashes.items())
assert len(diagnostic['cases'])==5
assert all(c['native']['bitwise_equal'] and all(x['bitwise_equal'] for x in c['continuations']) for c in diagnostic['cases'])
kernels=json.loads((ROOT/'artifacts/prefix_codec/full-build-s1-wide-v2/extra-kernel-hashes.json').read_bytes())
assert all(sha(ROOT/p)==v for p,v in kernels.items())
(build/'extra-kernel-hashes.json').write_text(json.dumps(kernels,indent=2)+'\n')
with zipfile.ZipFile(build/'extra-kernel-source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in kernels:z.write(ROOT/p,p)
with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in dict.fromkeys([*r['source_hashes'],*diagnostic['source_hashes'],*kernels]):z.write(ROOT/p,p)
changes=[]
for c in r['cases']:
    old=next(x for x in b['cases'] if x['label']==c['label'])
    assert c['full_bitwise_equal']
    assert c['query_count']==old['query_count']
    assert c['total_candid_bytes']==old['total_candid_bytes']
    changes.append(dict(label=c['label'],query_count=c['query_count'],before_instructions=old['total_instructions'],after_instructions=c['total_instructions'],instruction_delta=c['total_instructions']-old['total_instructions'],candid_bytes=c['total_candid_bytes'],max_query_instructions=c['max_query_instructions'],wall_seconds=c['wall_seconds'],heap_bytes=c['max_observed_heap_bytes']))
(d/'before-after.json').write_text(json.dumps(dict(cases=changes,diagnostic_queries=diagnostic['ordinary_queries'],candidate_report_sha256=sha(d/'report.json'),baseline_report_sha256=sha(ROOT/'artifacts/prefix_codec/full-s1-wide-proof/report.json'),goal_50_verified=False),indent=2)+'\n')
print(json.dumps(changes,indent=2))
