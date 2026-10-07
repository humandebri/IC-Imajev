#!/usr/bin/env python3
"""Verify saved raw Candid evidence independently of the network measurement loop."""
import hashlib,json,statistics,subprocess
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];D=R/'artifacts/paid-update-v1/proof-v3';B=R/'artifacts/paid-update-v1/build-v3'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(np.asarray(v,dtype='<f4').tobytes()).hexdigest()
def main():
 p=load(D/'report.json');b=load(B/'report.json');sources=load(D/'sources.json');assert p['complete'] and p['baseline_restored'] and p['snapshot_deleted']
 assert p['candidate']==b['wasm_sha256']==sha(B/'full.wasm')
 for hashes in [sources,b['source_hashes'],b['dependency_hashes']]:assert all(sha(R/f)==h for f,h in hashes.items())
 assert len({x['function_index']for x in b['patches']})==6 and all(x['wasmparser_validation']for x in b['patches'])
 original=load(R/'artifacts/update-templates-v1/build/report.json')
 assert [(x['export'],x['source_sha256'],x['replacement_body_sha256'])for x in b['patches']]==[(x['export'],x['source_sha256'],x['replacement_body_sha256'])for x in original['patches']]
 assert 'feature="paid-update-diagnostics"' not in b['command']
 helper=R/'artifacts/paid-update-v1/tools/args'
 # Re-decode every saved response instead of relying on the measurement assertions.
 raw_rows={}
 for path in sorted([f for directory in ['calls','upgrade-background','busy-secondary'] for f in (D/directory).glob('[0-9][0-9][0-9][0-9]-*.json')]):
  if '.input.'in path.name:continue
  row=load(path)
  if 'kind' not in row:continue
  kind='forward'if row['relay']else row['kind'];value=json.loads(subprocess.check_output([str(helper),'decode',kind,str(R/row['reply_path'])],text=True))
  if row['relay']:
   assert value==row['forward'];assert 'Ok'in value['response'];inner=R/row['reply_path'].replace('.reply.hex','.inner.hex')
   assert inner.read_text().strip()==bytes(value['response']['Ok']).hex()
   value=json.loads(subprocess.check_output([str(helper),'decode',row['kind'],str(inner)],text=True))
  assert value==row['result'];raw_rows[row['reply_path']]=row
 concurrency=load(D/'concurrency.json');upgrade=load(D/'active-upgrade.json')
 assert upgrade['returncode']!=0 and 'paid inference active'in upgrade['stderr'] and upgrade['running_receipt']['state']=='Running'
 for r in [concurrency['busy'],concurrency['primary']]:assert raw_rows[r['reply_path']]==r
 assert concurrency['active_upgrade_refused'] and concurrency['busy']['result']=={'Err':'Busy'} and concurrency['busy']['forward']['refunded']==268_000_000_000
 assert 'Ok'in concurrency['primary']['result'] and concurrency['primary']['result']['Ok']['workers'][-1]['stage']==64
 assert all(m['instructions']<40_000_000_000 and m['heap_pages']*65536<2**32 for m in concurrency['primary']['result']['Ok']['workers'])
 assert len(p['results'])==3 and len(p['callers'])==2
 typed=json.loads(subprocess.check_output([str(helper),'decode','paid_config',str(D/'typed-config.reply.hex')],text=True))
 assert typed==[r['request']for r in raw_rows.values()if r['kind']=='configure_paid'][-1]
 interface=(R/'canisters/inference/paid-inference.did').read_text()
 assert all(m+' :'in interface for m in ['infer','quote','inference_step','inference_status','retry_inference_refund','configure_paid'])
 assert 'paid_fault :'not in interface and 'paid_probe_step :'not in interface
 groups=[];allworkers=[]
 for item in p['results']:
  row=raw_rows[item['row']['reply_path']];assert row==item['row'];result=row['result']['Ok'];quote=item['quote'];debug=item['debug'];old=R/f'artifacts/boomdao-current-v1/{item["case"]}-r1';ref=load(old/'report.json')['decision_query']['ok']['decision']
  assert result['paid_cycles']==quote['fee'] and result['quote_version']==2 and row['attached_cycles']==quote['fee']+12_345_678 and row['forward']['refunded']==12_345_678
  loss=row['forward']['balance_before']-row['forward']['balance_after'];assert abs(loss-quote['fee'])<quote['fee']//100
  assert {k:v for k,v in result['decision'].items()if k!='instructions'}=={k:v for k,v in ref.items()if k!='instructions'}
  assert digest(debug['final_hidden'])==digest(np.load(old/'final-hidden.npy',allow_pickle=False))
  prefix=quote['prefix_tokens'];assert prefix==({'617':38,'620':38,'653':27}[item['case']])
  for i in range(32):
   hidden=old/f'queries/layer-{i:02d}.npy'
   if hidden.exists():
    v=np.load(hidden,allow_pickle=False);v=v[prefix:]if i<31 else v;assert digest(v)==debug['hidden_hashes'][i]
   else:assert i==30
   with np.load(old/f'queries/states/layer-{i:02d}.npz',allow_pickle=False)as z:v=np.concatenate([z['keys'][prefix:].ravel(),z['values'][prefix:].ravel()])if i%4==3 else z['conv'].ravel();assert digest(v)==debug['state_hashes'][i]
  stage=0
  for m in result['workers']:
   assert stage<m['stage']<=64 and m['instructions']<40_000_000_000 and m['heap_pages']*65536<2**32;stage=m['stage'];allworkers.append(m)
  assert stage==64 and len(result['workers'])=={'617':5,'620':4,'653':5}[item['case']]
  groups.append(dict(case=item['case'],repeat=item['repeat'],workers=len(result['workers']),instructions=sum(m['instructions']for m in result['workers']),seconds=row['seconds'],fee=quote['fee'],outer_candid_bytes=row['request_bytes']+row['reply_bytes']))
 assert len(load(D/'prefix-registration.json'))==64
 assert load(D/'preparation/report.json')['update_calls_this_run']==721 and load(D/'fixed-prefix-preparation/report.json')['update_calls']==24
 checks={c['name']:c for c in p['checks']};assert set(checks)=={'duplicate','id-conflict','insufficient','quote-version','token-bounds','worker-authority','status-authority','paused','upgrade-replay'}
 # The measurement reused a mutable request dictionary for two rejected cases.
 # Saved per-call inputs and raw replies retain the actual requests.
 for c in checks.values():
  actual=raw_rows[c['row']['reply_path']]
  assert {k:v for k,v in actual.items()if k!='request'}=={k:v for k,v in c['row'].items()if k!='request'}
  c['row']=actual
 assert checks['insufficient']['row']['result']=={'Err':{'InsufficientCycles':{'required':268_000_000_000}}}
 assert checks['quote-version']['row']['result']=={'Err':{'QuoteChanged':{'current':2}}}
 assert 'Invalid'in checks['token-bounds']['row']['result']['Err']
 for name in ['duplicate','id-conflict','insufficient','quote-version','token-bounds','paused','upgrade-replay']:
  row=checks[name]['row'];assert row['forward']['refunded']==row['attached_cycles']
 assert checks['worker-authority']['row']['result']=={'Err':'self only'} and checks['status-authority']['row']['result'] is None
 restored=load(D/'restored.json');assert restored['module']==original['baseline'] and restored['cache_equal'] and restored['pack_equal'] and restored['snapshot_deleted']
 assert p['upgrade_receipts_equal'] and p['dual_bank'];summary=[]
 for name in ['617','620','653']:
  rows=[x for x in groups if x['case']==name];assert len(rows)==1
  summary.append(dict(case=name,external_infer_calls=1,workers=rows[0]['workers'],instructions_median=statistics.median(x['instructions']for x in rows),seconds_median=statistics.median(x['seconds']for x in rows),test_fee=rows[0]['fee']))
 out=dict(complete=True,module=p['candidate'],ordinary_inferences=3,concurrent_inferences=1,active_upgrade_refused=True,busy_unpaid=True,worker_updates=len(allworkers),max_worker_handler_instructions=max(m['instructions']for m in allworkers),max_heap_bytes=max(m['heap_pages']*65536 for m in allworkers),dual_bank=True,rows=summary,boundary_checks=list(checks),raw_candid_rows_verified=len(raw_rows),bitwise_equal=True,upgrade_receipts_preserved=True,baseline_restored=True,scope='Caller-canister infer invoked once per input. Quote/debug/preparation are separate. Worker handler counters exclude CDK codec and tiny post-measurement bookkeeping. All successful replicated requests passed actual IC limits. Local fees are provisional, not mainnet pricing.')
 (D/'verified.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
