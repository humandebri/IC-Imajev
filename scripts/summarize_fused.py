#!/usr/bin/env python3
"""Summarize a fresh full graph, preserving all query measurements and precision comparisons."""
import argparse,hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--directory',default='artifacts/activation-int8-fused-617');ap.add_argument('--baseline',default='artifacts/activation-int8-617');ap.add_argument('--output',default='docs/fused-results.json');args=ap.parse_args();directory=ROOT/args.directory;before=ROOT/args.baseline
r=json.loads((directory/'report.json').read_text());old=json.loads((before/'first-report.json').read_text());a=r['decision_query']['ok']['decision'];b=old['decision_query']['ok']['decision']
assert r['input_hash']==old['input_hash'] and r['pack_hash']==old['pack_hash']
summary={k:v for k,v in r.items() if k!='queries'};rows=[]
for layer in range(32):
 x=np.load(directory/f'queries/layer-{layer:02d}.npy');y=np.load(before/f'queries/layer-{layer:02d}.npy');rows.append(dict(layer=layer,bitwise_equal=bool(np.array_equal(x.view(np.uint32),y.view(np.uint32))),max_error=float(abs(x-y).max())))
summary['fusion_comparison']=dict(before_queries=old['query_count'],after_queries=r['query_count'],all_layers_bitwise_equal=all(x['bitwise_equal'] for x in rows),layer_errors=rows,raw_logits_identical=a['raw_logits']==b['raw_logits'],probabilities_identical=a['probabilities']==b['probabilities'],unknown_identical=a['unknown_probability']==b['unknown_probability'],before_instructions=old['total_instructions'],after_instructions=r['total_instructions'],before_bytes=old['total_candid_bytes'],after_bytes=r['total_candid_bytes'],before_seconds=old['wall_seconds_this_run'],after_seconds=r['wall_seconds_this_run'])
states=[]
for layer in range(32):
 x=np.load(directory/f'queries/states/layer-{layer:02d}.npz');y=np.load(before/f'queries/states/layer-{layer:02d}.npz')
 assert x.files==y.files
 equal=all(np.array_equal(x[k].view(np.uint8),y[k].view(np.uint8)) for k in x.files);states.append(dict(layer=layer,bitwise_equal=equal))
summary['fusion_comparison']['all_states_bitwise_equal']=all(x['bitwise_equal'] for x in states);summary['fusion_comparison']['state_comparisons']=states
assert summary['fusion_comparison']['all_states_bitwise_equal']
summary['query_metrics']=[dict(index=q['index'],op=q['op'],tensor=q['tensor'],**q['ok'],wall_seconds=q['wall_seconds']) for q in r['queries']]
summary['additional_source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'crates/imajev-runtime/src/delta_simd.rs',ROOT/'crates/imajev-runtime/src/int8_kernel.rs',ROOT/'crates/imajev-runtime/src/bf16_codec.rs',ROOT/'crates/imajev-runtime/src/quantize_simd.rs',ROOT/'crates/imajev-runtime/src/profile.rs']}
summary['raw_report_sha256']=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()
summary['target_32_queries_achieved']=r['query_count']<=34
assert summary['fusion_comparison']['all_layers_bitwise_equal'] and summary['fusion_comparison']['raw_logits_identical'] and summary['fusion_comparison']['probabilities_identical'] and summary['fusion_comparison']['unknown_identical']
(ROOT/args.output).write_text(json.dumps(summary,indent=2)+'\n');print(summary['fusion_comparison']);print('max instructions',r['max_query_instructions'])
