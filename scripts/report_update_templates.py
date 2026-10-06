#!/usr/bin/env python3
"""Independently verify raw update replies, model equality and snapshot restoration."""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/update-templates-v1/proof-v2'
B=ROOT/'artifacts/update-templates-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(np.asarray(v,dtype='<f4').tobytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def main():
 report=load(D/'report.json');build=load(B/'report.json')
 assert report['complete'] and report['baseline_snapshot_restored'] and report['snapshot_deleted']
 assert report['candidate']==build['wasm_sha256']==sha(B/'full.wasm')
 for obj in [report,build]:
  for key in ['source_hashes','dependency_hashes']:
   assert all(sha(ROOT/p)==h for p,h in obj.get(key,{}).items())
 previous=sha(B/'raw.wasm')
 source=load(ROOT/'artifacts/voting-template-prefix-v1/full-build/report.json')
 for key in ['source_hashes','dependency_hashes']:
  assert all(sha(ROOT/p)==h for p,h in source[key].items())
 assert build['runtime_rlib_sha256']==sha(ROOT/'artifacts/voting-template-prefix-v1/full-build/libimajev_runtime_register.rlib')
 assert build['update_scheduler_sha256']==sha(ROOT/'artifacts/update-short-v1/build/update_inference.rs')==sha(B/'update_inference.rs')
 assert build['update_stop_instructions']==34_000_000_000
 assert len({p['function_index'] for p in build['patches']})==6
 for i,p in enumerate(build['patches']):
  assert p['wasmparser_validation'] and p['original_sha256']==previous
  assert p['export']==source['patches'][i]['export'] and p['source_sha256']==source['patches'][i]['source_sha256']
  assert p['replacement_body_sha256']==source['patches'][i]['replacement_body_sha256']
  previous=p['output_sha256'];assert previous==sha(B/('full.wasm' if i==5 else f'patched{i}.wasm'))
 fixture=load(ROOT/'artifacts/text-short-v2/inputs.json');cases=[];updates=0;max_instr=0;max_heap=0
 for bank in ['voting','common']:
  bd=D/bank;measurement=load(bd/'measurement/report.json');prefix=38 if bank=='voting' else 27
  assert measurement['complete'] and measurement['prefix_tokens']==prefix and measurement['module_sha256']==report['candidate']
  assert all(sha(ROOT/p)==h for p,h in measurement['reference_hashes'].items())
  assert len(measurement['prefix_preparation'])==32 and len(measurement['prefix_validation_updates'])==3 and len(measurement['boundary_verification_updates'])==2
  assert load(bd/'preparation/report.json')['update_calls_this_run']==721
  assert load(bd/'fixed-prefix-preparation/report.json')['update_calls']==24
  for row in measurement['cases']:
   name=row['case'];old=ROOT/f'artifacts/boomdao-current-v1/{name}-r1';rd=bd/'measurement'/f'{name}-r{row["repeat"]}'
   replies=[load(p) for p in sorted(rd.glob('[0-9][0-9].json'))]
   assert len(replies)==row['update_calls'] and len(replies)>0
   stage=0;session=None;all_ops=[]
   for raw in replies:
    p=raw['ok']['progress'];assert p['stage']>stage and p['stage']<=64
    assert p['done']==(p['stage']==64)
    if session is None:session=p['id']
    assert p['id']==session and p['instructions']<40_000_000_000
    assert raw['ok']['request_bytes']<2_000_000 and raw['ok']['reply_bytes']<2_000_000
    assert p['heap_pages']*65536<2**32
    assert len(p['hidden_hashes'])==p['stage']//2 and len(p['state_hashes'])==(p['stage']+1)//2
    assert all(n>0 for _,n in p['operations']) and sum(n for _,n in p['operations'])<=p['instructions']
    if not p['done']:assert p['instructions']>=34_000_000_000
    all_ops.extend(op for op,_ in p['operations']);stage=p['stage']
    max_instr=max(max_instr,p['instructions']);max_heap=max(max_heap,p['heap_pages']*65536)
   assert stage==64 and all_ops[:2]==['embed','initial_norm']
   assert all_ops[2:]==[('attention_full_integer' if i%4==3 else 'delta_full_hybrid_integer') if j==0 else 'mlp_full_integer' for i in range(32) for j in range(2)]
   p=replies[-1]['ok']['progress'];reference=load(old/'report.json')['decision_query']['ok']['decision']
   assert {k:v for k,v in reference.items() if k!='instructions'}=={k:v for k,v in p['decision'].items() if k!='instructions'}
   assert digest(p['final_hidden'])==digest(np.load(old/'final-hidden.npy',allow_pickle=False))
   for i in range(32):
    hidden=old/f'queries/layer-{i:02d}.npy'
    if hidden.exists():
     v=np.load(hidden,allow_pickle=False);v=v[prefix:] if i<31 else v
     assert digest(v)==p['hidden_hashes'][i],(name,i,'hidden')
    else:assert i==30
    with np.load(old/f'queries/states/layer-{i:02d}.npz',allow_pickle=False) as z:
     v=np.concatenate([z['keys'][prefix:].ravel(),z['values'][prefix:].ravel()]) if i%4==3 else z['conv'].ravel()
     assert digest(v)==p['state_hashes'][i],(name,i,'state')
   actual=dict(instructions=sum(r['ok']['progress']['instructions'] for r in replies),candid_bytes=sum(r['ok']['request_bytes']+r['ok']['reply_bytes'] for r in replies),seconds=sum(r['wall_seconds'] for r in replies))
   assert actual['instructions']==row['total_handler_instructions'] and actual['candid_bytes']==row['total_candid_bytes'] and actual['seconds']==row['call_wall_seconds']
   assert fixture['records'][{'617':0,'620':1,'653':2}[name]]['token_ids'][:prefix]==load(ROOT/('artifacts/voting-template-prefix-v1/prefix/queries/cache.json' if bank=='voting' else 'artifacts/query-packing-v3/prefix-v2/queries/cache.json'))['token_ids']
   cases.append(dict(proposal=name,repeat=row['repeat'],prefix_tokens=prefix,update_calls=len(replies),**actual));updates+=len(replies)
 assert len(cases)==9 and updates==report['inference_update_calls']
 restored=load(D/'restored.json');assert restored['restored'] and restored['cache_equal'] and restored['pack_equal'] and restored['module']==report['baseline']
 operations=load(D/'operations.json');assert sum(e['operation']==['install'] for e in operations)==2
 snapshot=load(D/'snapshot.json')['snapshot_id']
 assert any(e['operation']==['snapshot','restore'] and e['args']==[snapshot] for e in operations)
 assert any(e['operation']==['snapshot','delete'] and e['args']==[snapshot] for e in operations)
 old_summary=load(ROOT/'artifacts/update-short-v1/summary.json');summary=[]
 for name in ['617','620','653']:
  group=[r for r in cases if r['proposal']==name];assert len(group)==3 and len({r['update_calls'] for r in group})==1
  old=next(r for r in old_summary['rows'] if str(r['proposal_id'])==name)
  instructions=statistics.median(r['instructions'] for r in group)
  summary.append(dict(proposal=name,prefix_tokens=group[0]['prefix_tokens'],old_updates=old['update_calls'],updates=group[0]['update_calls'],instructions_median=instructions,instruction_reduction_percent=100*(1-instructions/old['handler_instructions_median']),candid_bytes_median=statistics.median(r['candid_bytes'] for r in group),seconds_median=statistics.median(r['seconds'] for r in group),bitwise_equal_to_original=True))
 result=dict(complete=True,module=report['candidate'],runs=9,inference_update_calls=updates,max_handler_instructions=max_instr,max_heap_bytes=max_heap,rows=summary,per_run=cases,preparation_per_bank=dict(weight_updates=721,fixed_common_state_updates=24,graph_prefix_updates=32),banks_prepared=2,validation_updates=10,baseline_restored=True,snapshot_deleted=True,scope='Local actual updates. 31 exported hidden hashes, 32 conv/suffix-KV hashes, final norm and decision/probabilities/logits match original. Initial preparation excluded. Timing is not a controlled production benchmark.')
 (D/'verified.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
