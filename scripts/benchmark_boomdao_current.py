#!/usr/bin/env python3
"""Measure the current short-prompt canister on frozen BOOM DAO summaries."""
import hashlib,json,pathlib,statistics,subprocess,sys,time
from evaluate_prompt_accuracy import ROOT,MODULE,base_flags,atomic_json,sha
D=ROOT/'artifacts/boomdao-current-v1'
def main():
 D.mkdir(parents=True,exist_ok=True)
 source=ROOT/'artifacts/text-short-v2/inputs.json';fixture=json.loads(source.read_text());records=fixture['records'][:3]
 cases=json.loads((ROOT/'benchmarks/cases.json').read_text())[:3]
 assert len(cases)==len(records)==3
 for c,r in zip(cases,records):
  assert r['id']==c['id'] and r['options']==c['options'] and r['prefix_tokens']==27
  assert r['input_sha256']==hashlib.sha256(json.dumps(r['token_ids']).encode()).hexdigest()
 paths=[source,ROOT/'benchmarks/cases.json',ROOT/'client/packed_inference.py',ROOT/'scripts/run_prefix_canister.py',ROOT/'artifacts/query-packing-v3/build/full.wasm',ROOT/'artifacts/query-packing-v3/build/imajev-client',pathlib.Path(__file__)]
 identities={str(p.relative_to(ROOT)):sha(p) for p in paths};assert sha(ROOT/'artifacts/query-packing-v3/build/full.wasm')==MODULE
 session=dict(module=MODULE,repeats=3,source_hashes=identities,scope='Frozen historical BOOM DAO structured summaries; current short prompt; actual local canister inference; prefix preparation excluded; no gold accuracy labels')
 sp=D/'session.json'
 if sp.exists():assert json.loads(sp.read_text())==session
 else:atomic_json(sp,session)
 cmd=base_flags();cmd[cmd.index('--reference')+1]=str(source);cmd[cmd.index('--cache')+1]='artifacts/query-packing-v3/prefix-v2/queries'
 cmd+=['--hybrid-cache','artifacts/query-packing-v3/packets-v2','--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start','--packed-start']
 rows=[];start=time.monotonic()
 for repeat in range(3):
  for index,(c,r) in enumerate(zip(cases,records)):
   dest=D/f'{c["proposal_id"]}-r{repeat+1}';path=dest/'report.json'
   if not path.exists():
    with (D/f'{c["proposal_id"]}-r{repeat+1}.log').open('a') as log:subprocess.run(cmd+['--record',str(index),'--directory',str(dest)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   report=json.loads(path.read_text());decision=report['decision_query']['ok']['decision']
   assert report['wasm_sha256']==MODULE and report['input_hash']==r['input_sha256'] and report['comparison']['typed_output_valid']
   assert report['replayed_queries']==0 and not report.get('fallback') and report['executed_query_count']==report['query_count']
   assert identities=={str(p.relative_to(ROOT)):sha(p) for p in paths},'sources changed during benchmark'
   row=dict(proposal_id=c['proposal_id'],repeat=repeat+1,prediction='__unknown__' if decision['abstained'] else decision['value'],probabilities=decision['probabilities'],unknown_probability=decision['unknown_probability'],tokens=report['tokens'],suffix_tokens=report['processed_tokens'],query_count=report['query_count'],instructions=report['total_instructions'],max_query_instructions=report['max_query_instructions'],candid_bytes=report['total_candid_bytes'],wall_seconds=report['end_to_end_seconds_excluding_process_startup'],replayed_queries=0,fallback=False,report=str(path.relative_to(ROOT)),report_sha256=sha(path))
   rows.append(row);atomic_json(D/'report.json',dict(session,complete=False,runs=rows,elapsed_seconds=time.monotonic()-start));print(json.dumps(row),flush=True)
 summaries=[]
 for c in cases:
  group=[r for r in rows if r['proposal_id']==c['proposal_id']];assert len(group)==3
  assert len({json.dumps(r['probabilities']) for r in group})==1 and len({r['prediction'] for r in group})==1
  summaries.append(dict(proposal_id=c['proposal_id'],question=c['question'],state=c['state'],options=c['options'],prediction=group[0]['prediction'],probabilities=group[0]['probabilities'],unknown_probability=group[0]['unknown_probability'],tokens=group[0]['tokens'],suffix_tokens=group[0]['suffix_tokens'],query_count=group[0]['query_count'],median_instructions=statistics.median(r['instructions'] for r in group),max_query_instructions=max(r['max_query_instructions'] for r in group),median_candid_bytes=statistics.median(r['candid_bytes'] for r in group),median_wall_seconds=statistics.median(r['wall_seconds'] for r in group)))
 atomic_json(D/'report.json',dict(session,complete=True,runs=rows,cases=summaries,elapsed_seconds=time.monotonic()-start));print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
