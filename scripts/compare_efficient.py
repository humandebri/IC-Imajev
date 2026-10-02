#!/usr/bin/env python3
"""Compare actual ordinary-query runs, including every saved layer's output bits."""
import argparse,hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--directory',default='artifacts/full-int8-efficient-run');ap.add_argument('--progress',action='store_true');args=ap.parse_args();directory=ROOT/args.directory;baseline=ROOT/'artifacts/full-int8-canister';rows=[]
for p in sorted((directory/'queries').glob('layer-*.npy')):
 a=np.load(p);b=np.load(baseline/'queries'/p.name);same=bool(np.array_equal(a.view(np.uint32),b.view(np.uint32)))
 rows.append(dict(layer=int(p.stem.split('-')[1]),bitwise_equal=same,max_error=float(abs(a-b).max())))
 assert same,rows[-1]
if args.progress:
 print(json.dumps(dict(layers_checked=len(rows),all_bitwise_equal=True)));raise SystemExit(0)
old=json.loads((baseline/'first-report.json').read_text());new=json.loads((directory/'report.json').read_text())
assert len(rows)==32 and new['replayed_queries']==0
assert old['input_hash']==new['input_hash'] and old['model']==new['model'] and old['pack_hash']==new['pack_hash']
assert old['decision_query']['ok']['decision']['raw_logits']==new['decision_query']['ok']['decision']['raw_logits']
assert old['decision_query']['ok']['decision']['probabilities']==new['decision_query']['ok']['decision']['probabilities']
assert np.array_equal(np.load(baseline/'final-hidden.npy').view(np.uint32),np.load(directory/'final-hidden.npy').view(np.uint32))
keys=['query_count','total_candid_bytes','total_instructions','max_query_instructions','max_observed_heap_bytes','wall_seconds_this_run','wasm_sha256','implementation_hashes','comparison']
summary=dict(scope='132-token BOOM DAO617, rotations=1, same INT8 pack; query cache not controlled',before={k:old[k] for k in keys},after={k:new[k] for k in keys},layer_comparison=rows,final_hidden_bitwise_equal=True,decision_logits_and_probabilities_equal=True,reduction_fraction={k:1-new[k]/old[k] for k in ['query_count','total_candid_bytes','total_instructions','wall_seconds_this_run']})
(ROOT/'docs/communication-improvement.json').write_text(json.dumps(summary,indent=2)+'\n')
new['queries']=[dict(index=q['index'],op=q['op'],tensor=q['tensor'],**q['ok'],wall_seconds=q['wall_seconds']) for q in new['queries']]
new['raw_report_sha256']=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()
(ROOT/'docs/efficient-results.json').write_text(json.dumps(new,indent=2)+'\n');print(json.dumps(summary['reduction_fraction']));print(summary['after'])
