#!/usr/bin/env python3
"""Audit actual handler budgets for50 ordinary queries; sums do not prove fusion."""
import argparse,json,pathlib,hashlib,re
ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--report',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args();p=ROOT/a.report;raw=p.read_bytes();r=json.loads(raw);d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);blocks={i:[]for i in [0,8,16,24]}
for q in r['queries']:
 match=re.search(r'layers\.(\d+)',q['tensor']);layer=int(match.group(1))if match else 0;blocks[layer//8*8].append(q)
rows=[]
for layer,qs in blocks.items():
 instructions=sum(q['ok']['instructions']for q in qs);target=13 if layer<24 else 11
 rows.append(dict(first_layer=layer,last_layer=layer+7,current_queries=len(qs),handler_instructions=instructions,target_queries=target,handler_capacity=target*5_000_000_000,nominal_slack=target*5_000_000_000-instructions,query_indices=[q['index']for q in qs],actual_fusion_verified=False))
assert sum(c['current_queries']for c in rows)==r['query_count'];assert sum(c['handler_instructions']for c in rows)==r['total_instructions']
(d/'report.json').write_text(json.dumps(dict(scope='Measured counters only; boundaries, full IC instruction cost, carried state size, codec/continuation cost and bitwise correctness remain unverified. No query reduction claimed.',source_sha256=hashlib.sha256(raw).hexdigest(),wasm_sha256=r['wasm_sha256'],model=r['model'],pack_hash=r['pack_hash'],actual_queries=r['query_count'],hypothetical_target=sum(c['target_queries']for c in rows),blocks=rows,next_required_action='Implement resumable operation cuts across8 layers; measure each<=5B and each client carry<=2MB; retain all original precision and compare all states/decision. Existing512-row MLP+Delta carry alone does not reduce the query count.'),indent=2)+'\n')
