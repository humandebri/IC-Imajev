#!/usr/bin/env python3
"""Independently decode and check local worker-failure/refund evidence."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'artifacts/paid-update-v1/fault-proof-v2';B=R/'artifacts/paid-update-v1/build-diagnostic-v2'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 b=load(B/'report.json');p=dict(candidate=b['wasm_sha256'],checks=load(D/'checks.json'))
 assert p['candidate']==b['wasm_sha256']==sha(B/'full.wasm')
 for hashes in [load(D/'sources.json'),b['source_hashes'],b['dependency_hashes']]:assert all(sha(R/f)==h for f,h in hashes.items())
 assert 'feature="paid-update-diagnostics"' in b['command']
 helper=R/'artifacts/paid-update-v1/tools/args';rows={}
 def decode(kind,path):return json.loads(subprocess.check_output([str(helper),'decode',kind,str(path)],text=True))
 for directory in ['calls','busy-background','busy-secondary']:
  for path in sorted((D/directory).glob('[0-9][0-9][0-9][0-9]-*.json')):
   if '.input.'in path.name:continue
   row=load(path)
   if 'kind'not in row:continue
   raw=decode('forward'if row['relay']else row['kind'],R/row['reply_path'])
   if row['relay']:
    assert raw==row['forward'] and 'Ok'in raw['response']
    inner=R/row['reply_path'].replace('.reply.hex','.inner.hex');assert bytes(raw['response']['Ok']).hex()==inner.read_text().strip()
    raw=decode(row['kind'],inner)
   assert raw==row['result'];rows[row['reply_path']]=row
 checks={c['name']:c for c in p['checks']};fee=268_000_000_000
 checks['busy-unpaid']=dict(row=next(r for r in rows.values()if r['kind']=='infer' and r['request'].get('request_id')=='busy-secondary'))
 for name,file in [('stale-internal-job','stale-probe.reply.hex'),('invalid-internal-stage','stage-probe.reply.hex')]:checks[name]=dict(result=decode('inference_step',D/file))
 checks['busy-primary-completed']=dict(row=next(r for r in rows.values()if r['kind']=='infer' and r['request'].get('request_id')=='busy-primary'))
 for c in checks.values():
  if 'row'in c:assert rows[c['row']['reply_path']]==c['row']
 for name in ['reject','trap-continuation','pending-refund']:
  row=checks[name]['row'];state='Pending'if name=='pending-refund'else'Done'
  assert row['result']['Err']['Failed']['refund']==state and row['forward']['refunded']==12345
  assert checks[name]['receipt']['result']['refund']==state
  loss=row['forward']['balance_before']-row['forward']['balance_after'];assert abs(loss-(fee if state=='Pending'else 0))<fee//100
  retry=checks[name+'-refund-retry']['row'];assert retry['result']=={'Ok':'Done'}
  if state=='Pending':assert retry['forward']['balance_after']-retry['forward']['balance_before']>fee-fee//100
  again=checks[name+'-no-double-refund']['row'];assert again['result']=={'Ok':'Done'} and abs(again['forward']['balance_after']-again['forward']['balance_before'])<fee//100
  assert checks[name+'-foreign-refund']['row']['result']=={'Err':'missing receipt'}
 busy=checks['busy-unpaid']['row'];assert busy['result']=={'Err':'Busy'} and busy['forward']['refunded']==busy['attached_cycles']==fee
 for name,file in [('stale-internal-job','stale-probe.reply.hex'),('invalid-internal-stage','stage-probe.reply.hex')]:assert decode('inference_step',D/file)==checks[name]['result']=={'Err':'job progress mismatch'}
 primary=checks['busy-primary-completed']['row']['result']['Ok'];assert len(primary['workers'])==5 and primary['workers'][-1]['stage']==64 and primary['paid_cycles']==fee
 assert all(m['instructions']<40_000_000_000 and m['heap_pages']*65536<2**32 for m in primary['workers'])
 restored=load(D/'restored.json');assert restored['cache_equal'] and restored['pack_equal'] and restored['snapshot_deleted'] and restored['module']=='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
 out=dict(complete=True,module=p['candidate'],diagnostic_only=True,checks=list(checks),raw_candid_rows_verified=len(rows),explicit_refund_done=True,pending_refund_retry=True,no_double_refund=True,busy_unpaid=True,stale_workers_refused=True,scope='Checks completed before the large-Wasm upgrade timing assertion failed; active upgrade refusal and receipt equality are not claimed by this report.',baseline_restored=True)
 (D/'verified.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
