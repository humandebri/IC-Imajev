#!/usr/bin/env python3
"""Keep the measured cold-path savings separate from unchanged main query counts."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=ROOT/'artifacts/prefix_codec/full-capture-prepared-proof'
r=json.loads((d/'report.json').read_bytes())
b=json.loads((ROOT/'artifacts/prefix_codec/full-f32-k-continue-proof/report.json').read_bytes())
diagnostic=json.loads((ROOT/'artifacts/f32_k_continue/capture-check/report.json').read_bytes())
assert r['wasm_sha256']==diagnostic['wasm_sha256']==sha(ROOT/'artifacts/prefix_codec/full-build-capture-prepared-v1/full.wasm')
for h in [r['source_hashes'],diagnostic['source_hashes']]:assert all(sha(ROOT/p)==v for p,v in h.items())
assert len(diagnostic['cases'])==31 and all(c['bitwise_equal'] and c['saved_instructions']>0 for c in diagnostic['cases'])
changes=[]
for c in r['cases']:
    old=next(x for x in b['cases'] if x['label']==c['label'])
    assert c['full_bitwise_equal'] and c['query_count']==old['query_count'] and c['total_candid_bytes']==old['total_candid_bytes']
    saved=old['total_instructions']-c['total_instructions']
    assert saved>0 if c['label']=='normal' else saved==0
    changes.append(dict(label=c['label'],query_count=c['query_count'],before_instructions=old['total_instructions'],after_instructions=c['total_instructions'],saved_instructions=saved,saved_percent=100*saved/old['total_instructions'],candid_bytes=c['total_candid_bytes'],max_query_instructions=c['max_query_instructions'],wall_seconds=c['wall_seconds'],heap_bytes=c['max_observed_heap_bytes']))
with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in dict.fromkeys([*r['source_hashes'],*diagnostic['source_hashes'],'scripts/archive_mlp_capture_proof.py']):z.write(ROOT/p,p)
(d/'before-after.json').write_text(json.dumps(dict(cases=changes,diagnostic_queries=diagnostic['ordinary_diagnostic_queries'],baseline_report_sha256=sha(ROOT/'artifacts/prefix_codec/full-f32-k-continue-proof/report.json'),candidate_report_sha256=sha(d/'report.json'),goal_50_verified=False),indent=2)+'\n')
print(json.dumps(changes,indent=2))
