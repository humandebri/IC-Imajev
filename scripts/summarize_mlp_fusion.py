#!/usr/bin/env python3
"""Record final gains, true query costs, failed attempts, and unchanged decisions separately."""
import argparse,collections,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--directory',default='artifacts/mlp-fused-full-617');ap.add_argument('--output',default='docs/mlp-fusion-costs.json');args=ap.parse_args()
r=json.loads((ROOT/args.directory/'first-report.json').read_text());before=json.loads((ROOT/'artifacts/bottleneck-fused-full-617/first-report.json').read_text());dot=json.loads((ROOT/'artifacts/dot-unroll-full-617/first-report.json').read_text())
def totals(report):
 d=collections.defaultdict(lambda:dict(queries=0,instructions=0,bytes=0))
 for q in report['queries']:
  v=d[q['op']];v['queries']+=1;v['instructions']+=q['ok']['instructions'];v['bytes']+=q['ok']['request_bytes']+q['ok']['reply_bytes']
 return dict(d)
result=dict(scope='Actual complete local ordinary query inference; checkpoint replay excluded',before={k:before[k] for k in ['query_count','total_instructions','total_candid_bytes','wall_seconds_this_run']},dot_only={k:dot[k] for k in ['query_count','total_instructions','total_candid_bytes','wall_seconds_this_run']},after={k:r[k] for k in ['query_count','total_instructions','total_candid_bytes','wall_seconds_this_run','max_query_instructions','max_observed_heap_bytes']},instruction_reduction=1-r['total_instructions']/before['total_instructions'],query_reduction=1-r['query_count']/before['query_count'],communication_reduction=1-r['total_candid_bytes']/before['total_candid_bytes'],before_op_totals=totals(before),dot_only_op_totals=totals(dot),after_op_totals=totals(r),canister=r['canister'],url=r['url'],wasm_sha256=r['wasm_sha256'],input_hash=r['input_hash'],pack_hash=r['pack_hash'],target_50_achieved=r['query_count']<=50,ideal_instruction_budget_only_queries=(r['total_instructions']+4_999_999_999)//5_000_000_000,replayed_queries=r['replayed_queries'],failed_query_requests=len(list((ROOT/args.directory/'queries').glob('*.failed-width-*.request.bin'))),raw_report_sha256=hashlib.sha256((ROOT/args.directory/'first-report.json').read_bytes()).hexdigest(),environment_recovery=json.loads((ROOT/'artifacts/dot-unroll/new-environment-upload/report.json').read_text()))
(ROOT/args.output).write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['before_op_totals','dot_only_op_totals','after_op_totals','environment_recovery']})
