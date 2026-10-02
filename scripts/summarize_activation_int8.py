#!/usr/bin/env python3
"""Compare lossy activation INT8 transport with the same locked weight pack."""
import argparse,hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--directory',default='artifacts/activation-int8-617');ap.add_argument('--output',default='docs/activation-int8-results.json');args=ap.parse_args();directory=ROOT/args.directory
r=json.loads((directory/'report.json').read_text());baseline=json.loads((ROOT/'artifacts/full-int8-efficient-run/first-report.json').read_text());d=r['decision_query']['ok']['decision'];bd=baseline['decision_query']['ok']['decision'];same=baseline['input_hash']==r['input_hash']
summary={k:v for k,v in r.items() if k not in ('queries',)}
summary['query_metrics']=[dict(index=q['index'],op=q['op'],tensor=q['tensor'],**q['ok'],wall_seconds=q['wall_seconds']) for q in r['queries']]
summary['raw_report_sha256']=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()
summary['evaluation_scope']='One full text-only query graph per question/order, rotations=1; weights fixed; lossy transport is separate from weight INT8'
if same:
 summary['bf16_wire_comparison']=dict(value=bd['value'],int8_value=d['value'],same_choice=bd['value']==d['value'],max_logit_error=float(abs(np.array(bd['raw_logits'])-d['raw_logits']).max()),max_probability_error=float(abs(np.array(bd['probabilities'])-d['probabilities']).max()),unknown_probability_error=abs(bd['unknown_probability']-d['unknown_probability']),candid_bytes_before=baseline['total_candid_bytes'],candid_bytes_after=r['total_candid_bytes'],byte_reduction_fraction=1-r['total_candid_bytes']/baseline['total_candid_bytes'],wall_before=baseline['wall_seconds_this_run'],wall_after=r['wall_seconds_this_run'])
 rows=[]
 for i in range(32):
  a=np.load(directory/f'queries/layer-{i:02d}.npy');b=np.load(ROOT/f'artifacts/full-int8-efficient-run/queries/layer-{i:02d}.npy');error=abs(a-b)
  rows.append(dict(layer=i,max_error=float(error.max()),mean_error=float(error.mean()),last_token_max_error=float(error[-1].max()),finite=bool(np.isfinite(a).all())))
 summary['bf16_layer_errors']=rows
(ROOT/args.output).write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items() if k in ['comparison','bf16_wire_comparison','query_count','total_candid_bytes','wall_seconds_this_run']})
