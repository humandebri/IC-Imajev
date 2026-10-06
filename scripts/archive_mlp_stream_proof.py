#!/usr/bin/env python3
"""Validate streamed MLP evidence and retain unchanged-graph counters honestly."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=ROOT/'artifacts/prefix_codec/full-build-mlp-stream-v2';d=ROOT/'artifacts/prefix_codec/full-mlp-stream-proof';full=json.loads((d/'report.json').read_bytes());before=json.loads((ROOT/'artifacts/prefix_codec/full-capture-prepared-proof/report.json').read_bytes());hashes=json.loads((build/'source-hashes.json').read_bytes());assert all(sha(ROOT/p)==v for p,v in hashes.items());assert full['wasm_sha256']==sha(build/'full.wasm')
diagnostics=[]
for label,n in [('617',87),('insufficient',80),('maximum',89)]:
 p=ROOT/f'artifacts/f32_k_continue/stream-check-{label}/report.json';r=json.loads(p.read_bytes());assert r['source_case']==label and r['wasm_sha256']==full['wasm_sha256'];assert all(sha(ROOT/p)==v for p,v in r['source_hashes'].items());assert len(r['cases'])==15 and r['regular_diagnostic_queries']==51 and r['profile_diagnostic_queries']==9
 for c in r['cases']:
  assert c['tokens']==n and c['bitwise_prepared_equal'] and c['bitwise_hidden_norm_equal']
  assert c['max_query_instructions']<=5_000_000_000
  if c['chunks']==[4608,4608]:
   spans=[{name:(cost,count)for name,cost,count in x['ok']['spans']}for x in c['profile_diagnostics']]
   assert spans[0]['stream_input_quantize_once'][1]==1 and spans[0]['stream_input_A_once'][1]==2
   assert all('stream_input_quantize_once'not in s and 'stream_input_A_once'not in s for s in spans[1:])
   assert all(s['stream_product_quantize_once'][1]==1 for s in spans[:2]) and 'stream_product_quantize_once'not in spans[2]
   assert all('activation_quantize'not in s for s in spans)
 diagnostics.append(dict(label=label,report_sha256=sha(p),regular_queries=r['regular_diagnostic_queries'],profile_queries=r['profile_diagnostic_queries']))
changes=[]
for c in full['cases']:
 b=next(x for x in before['cases']if x['label']==c['label']);assert c['full_bitwise_equal'] and c['query_count']==b['query_count'] and c['total_candid_bytes']==b['total_candid_bytes']
 changes.append(dict(label=c['label'],query_count=c['query_count'],before_instructions=b['total_instructions'],after_instructions=c['total_instructions'],instruction_delta=c['total_instructions']-b['total_instructions'],candid_bytes=c['total_candid_bytes'],max_query_instructions=c['max_query_instructions'],heap_bytes=c['max_observed_heap_bytes'],wall_seconds=c['wall_seconds']))
bridge=json.loads((ROOT/'artifacts/prefix_codec/full-build-host-checksum-v3/bridge.json').read_bytes());assert all(sha(ROOT/p)==v for p,v in bridge.items());(build/'bridge.json').write_text(json.dumps(bridge,indent=2)+'\n')
with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
 for p in dict.fromkeys([*full['source_hashes'],*hashes,'scripts/check_mlp_stream.py','scripts/archive_mlp_stream_proof.py']):z.write(ROOT/p,p)
(d/'before-after.json').write_text(json.dumps(dict(cases=changes,diagnostics=diagnostics,scope='New streaming API validated; existing graph does not use it and still needs fusion work.',goal_50_verified=False),indent=2)+'\n')
print(json.dumps(changes,indent=2))
