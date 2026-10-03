#!/usr/bin/env python3
"""Summarize successful fresh runs; fail rather than omit rejected queries."""
import collections,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 summary={}
 for name in ['prefix','617','insufficient','maximum']:
  directory=ROOT/f'artifacts/fused-stage-{name}';r=json.loads((directory/'report.json').read_text())
  failures=directory/'queries/failures.jsonl'
  assert not failures.exists() or not failures.read_text().strip(),f'failed calls: {failures}'
  assert r['replayed_queries']==0
  assert r['wasm_sha256']==r['deployed_wasm_sha256']
  ops=collections.defaultdict(lambda:dict(queries=0,instructions=0,bytes=0))
  for q in r['queries']:
   o=ops[q['op']];o['queries']+=1;o['instructions']+=q['ok']['instructions'];o['bytes']+=q['ok']['request_bytes']+q['ok']['reply_bytes']
  summary[name]={k:r[k] for k in ['tokens','processed_tokens','query_count','total_instructions','total_candid_bytes','wall_seconds_this_run','max_query_instructions','max_observed_heap_bytes','wasm_sha256']}
  summary[name].update(ops=dict(ops),zero_failed_calls=True,zero_replayed_calls=True,report_sha256=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest())
  print(name,json.dumps({k:v for k,v in summary[name].items() if k!='ops'}))
 (ROOT/'docs/fused-stage-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
