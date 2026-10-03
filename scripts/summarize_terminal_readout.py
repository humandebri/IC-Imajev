#!/usr/bin/env python3
"""Report fresh full-graph terminal readout runs, rejecting failures and replay."""
import collections,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 summary={}
 for name in ['prefix','617','insufficient','maximum','normal']:
  directory=ROOT/f'artifacts/terminal-v3-{name}';r=json.loads((directory/'report.json').read_text());before=json.loads((ROOT/f'artifacts/fused-stage-{name}/report.json').read_text())
  c=json.loads((ROOT/f'docs/terminal-readout-{name}-results.json').read_text())
  assert c['all_retained_hidden_bitwise_equal'] and c['retained_state_arrays_bitwise_equal']==(72 if name=='prefix' else 48)
  assert r['replayed_queries']==0
  failures=directory/'queries/failures.jsonl';assert not failures.exists() or not failures.read_text().strip()
  if 'deployed_wasm_sha256' in r:assert r['wasm_sha256']==r['deployed_wasm_sha256']
  metrics=['query_count','total_instructions','total_candid_bytes','wall_seconds_this_run','max_query_instructions','max_observed_heap_bytes']
  ops=collections.defaultdict(lambda:dict(queries=0,instructions=0))
  for q in r['queries']:ops[q['op']]['queries']+=1;ops[q['op']]['instructions']+=q['ok']['instructions']
  summary[name]=dict(before={k:before[k] for k in metrics},after={k:r[k] for k in metrics},tokens=r['tokens'],wasm_sha256=r['wasm_sha256'],ops=dict(ops),zero_failures=True,zero_replay=True,comparison=r.get('comparison'),raw_report_sha256=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest())
  print(name,json.dumps({k:v for k,v in summary[name].items() if k!='ops'}))
 (ROOT/'docs/terminal-readout-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
