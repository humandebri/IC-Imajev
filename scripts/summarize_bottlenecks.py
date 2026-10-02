#!/usr/bin/env python3
"""Measured op totals, sampled inclusive spans, tile trials, and attribution limits."""
import collections,hashlib,json,pathlib,struct
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def totals(r):
 d=collections.defaultdict(lambda:dict(queries=0,instructions=0,candid_bytes=0,seconds=0))
 for q in r['queries']:
  v=d[q['op']];v['queries']+=1;v['instructions']+=q['ok']['instructions'];v['candid_bytes']+=q['ok']['request_bytes']+q['ok']['reply_bytes'];v['seconds']+=q['wall_seconds']
 for v in d.values():v['instruction_share']=v['instructions']/r['total_instructions']
 return dict(sorted(d.items(),key=lambda item:-item[1]['instructions']))
before=read('artifacts/integer-full-617/first-report.json');after=read('artifacts/bottleneck-fused-full-617/first-report.json');b=totals(before);a=totals(after)
trials={}
for phase in ['before','simd-scale','simd-scale-profile','simd-scale-codec','tile8','tile16','tile32','simd-scale-codec-quant','final-profile']:
 p=f'artifacts/bottleneck/{phase}/report.json'
 if not (ROOT/p).exists():continue
 r=read(p);trials[phase]=dict(wasm_sha256=r['wasm_sha256'],cases=r['cases'],source_report_sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
metrics=['query_count','total_instructions','total_candid_bytes','max_query_instructions','max_observed_heap_bytes','wall_seconds_this_run']
summary=dict(scope='Full local 32-layer BOOM617 measurements; sampled spans identify suboperation costs, never summed with parents',before={k:before[k] for k in metrics},after={k:after[k] for k in metrics},wasm_sha256=after['wasm_sha256'],instruction_reduction=1-after['total_instructions']/before['total_instructions'],query_reduction=1-after['query_count']/before['query_count'],communication_reduction=1-after['total_candid_bytes']/before['total_candid_bytes'],before_op_totals=b,after_op_totals=a,op_changes={k:dict(before_instructions=b.get(k,{}).get('instructions',0),after_instructions=a.get(k,{}).get('instructions',0)) for k in b.keys()|a.keys()},trials=trials,target_50_achieved=after['query_count']<=50,ideal_instruction_budget_only_query_count=(after['total_instructions']+4_999_999_999)//5_000_000_000,profiling_limitations=['Counters exclude CDK Candid decode/encode, HTTP and client work','Inner spans are inclusive; parent and child must not be added','Initial before profiling build changed compiler decisions and normal cost up to approximately 19%; its internal spans are diagnostic only, not the baseline full-run cost','Final totals use instruction-profile disabled; report overhead versus normal method in each profiling build','Wall-time measurements are single runs with uncontrolled cache and host load'])
# Extrapolation only for integer shapes actually covered. Do not treat this as a second full-run counter.
if 'final-profile' in trials:
 cases={(c['op'],tuple(c['dims'][:3])):c for c in trials['final-profile']['cases'] if 'integer' in c['op']};stages=collections.defaultdict(int);covered=0;cost_delta=0
 for q in after['queries']:
  if 'integer' not in q['op']:continue
  with (ROOT/f"artifacts/bottleneck-fused-full-617/queries/{q['index']:06d}.request.bin").open('rb') as f:
   n,=struct.unpack('<I',f.read(4));h=json.loads(f.read(n))
  c=cases[(q['op'],tuple(h['dims'][:3]))];covered+=1;cost_delta+=c['normal']['instructions']-q['ok']['instructions']
  for name,counter,_ in c['profile']['spans']:stages[name]+=counter
 summary['integer_stage_extrapolation']=dict(scope='Representative real query shape counters multiplied by full-run shape counts; different weights/row offsets may slightly alter costs; inclusive spans',queries_covered=covered,sampled_normal_minus_full_run_instructions=cost_delta,stages=dict(stages))
(ROOT/'docs/bottleneck-profile.json').write_text(json.dumps(summary,indent=2)+'\n');print('before',summary['before'],'after',summary['after']);print('after op totals',a)
