#!/usr/bin/env python3
"""Snapshot-protected paid inference proof using an actual caller canister."""

from historical_paid_proof import historical_only
historical_only()
import hashlib,json,subprocess,sys,time,zipfile,concurrent.futures
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport
from prefix_inference import verify_module
from paid_update_transport import PaidTransport
TARGET='6eydd-o3777-77775-aaama-cai';BASELINE='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
B=ROOT/'artifacts/paid-update-v1/build-diagnostic-v2';D=ROOT/'artifacts/paid-update-v1/fault-proof-v2';OWNER='cibxp-okw3l-gfzvi-p6ltu-23ss3-7tlfz-nk65x-tvhvb-mg56y-b5hkf-eqe'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def digest(v):return hashlib.sha256(np.asarray(v,dtype='<f4').tobytes()).hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=False);manifest=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());build=json.loads((B/'report.json').read_text());candidate=sha(B/'full.wasm');assert candidate==build['wasm_sha256']
 for key in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in build[key].items())
 paths=[Path(__file__),ROOT/'scripts/paid_update_transport.py',ROOT/'artifacts/paid-update-v1/tools/report.json',ROOT/'artifacts/paid-update-v1/tools/args',ROOT/'artifacts/paid-update-v1/tools/caller.wasm',ROOT/'artifacts/text-short-v2/inputs.json',B/'report.json',B/'full.wasm']
 sources={str(p.relative_to(ROOT)):sha(p) for p in paths};write(D/'sources.json',sources)
 with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
 events=[]
 def icp(*args):
  cmd=['icp','canister',*args,'--network','local','--identity','imajev-local'];start=time.monotonic();raw=subprocess.check_output(cmd,text=True,cwd=ROOT);events.append(dict(args=list(args),output=raw,seconds=time.monotonic()-start));write(D/'operations.json',events);return raw.strip()
 def bridge(path):return Transport(manifest['model'],'http://localhost:8001/',TARGET,str(ROOT/'artifacts/imajev-local.pem'),path,manifest['pack_hash'],bridge_binary=str(ROOT/'artifacts/query-packing-v3/build/imajev-client'))
 t=bridge(D/'before');verify_module(t,BASELINE);cache=t.command(dict(op='weight_cache_status'))['ok']['cache'];pack=t.command(dict(op='pack_status'))['ok'];t.close();write(D/'before.json',dict(module=BASELINE,cache=cache,pack=pack))
 callers=[];snapshot=None;stopped=False;results=[];checks=[]
 try:
  icp('stop',TARGET);stopped=True;snapshot=icp('snapshot','create',TARGET,'--quiet');write(D/'snapshot.json',dict(id=snapshot));print('snapshot saved',flush=True)
  icp('install',TARGET,'--mode','upgrade','--wasm',str(B/'full.wasm'),'--yes');icp('start',TARGET);stopped=False
  for i in range(2):
   caller=icp('create','--detached','--cycles','8t','--quiet');callers.append(caller);write(D/'callers.json',callers)
   icp('install',caller,'--mode','install','--wasm',str(ROOT/'artifacts/paid-update-v1/tools/caller.wasm'),'--args',f'(principal "{OWNER}")','--yes')
  print(json.dumps(dict(stage='callers-ready',callers=callers)),flush=True)
  with (D/'preparation.log').open('w')as log:subprocess.run([sys.executable,str(ROOT/'scripts/prepare_weight_cache.py'),'--canister',TARGET,'--wasm',str(B/'full.wasm'),'--directory',str(D/'preparation'),'--include-f32','--require-prepared-rope','--require-prepared-activation','--require-all-output-pairs'],check=True,cwd=ROOT,stdout=log,stderr=log)
  print('weights ready',flush=True)
  with (D/'fixed-prefix-preparation.log').open('w')as log:subprocess.run([sys.executable,str(ROOT/'scripts/prepare_fixed_prefix_states.py'),'--canister',TARGET,'--wasm',str(B/'full.wasm'),'--directory',str(D/'fixed-prefix-preparation')],check=True,cwd=ROOT,stdout=log,stderr=log)
  t=bridge(D/'prefix-registration');prefix_rows=[]
  try:
   for bank in ['voting','common']:
    base=ROOT/('artifacts/voting-template-prefix-v1/prefix/queries' if bank=='voting' else 'artifacts/query-packing-v3/prefix-v2/queries');packets=ROOT/('artifacts/voting-template-prefix-v1/packets' if bank=='voting' else 'artifacts/query-packing-v3/packets-v2')
    pd=D/('prefix-'+bank);pd.mkdir()
    for layer in range(32):
     with np.load(base/'states'/f'layer-{layer:02d}.npz',allow_pickle=False)as z:v=np.concatenate([z['keys'].transpose(1,0,2).ravel(),z['values'].transpose(1,0,2).ravel()]) if layer%4==3 else z['conv'].ravel()
     path=pd/f'{layer:02d}.f32';np.asarray(v,dtype='<f4').tofile(path);cmd=dict(op='update_prefix',layer=layer,values=str(path))
     if layer%4!=3:cmd['packet']=str(packets/f'layer-{layer:02d}.npf1')
     reply=t.command(cmd);prefix_rows.append(dict(bank=bank,layer=layer,**reply))
    print(json.dumps(dict(stage='prefix-bank-ready',bank=bank)),flush=True)
  finally:t.close()
  write(D/'prefix-registration.json',prefix_rows)
  wire=PaidTransport(D/'calls',TARGET,helper=ROOT/'artifacts/paid-update-v1/tools/args');config=dict(enabled=True,version=2,base_fee=100_000_000_000,fee_per_token=3_000_000_000,reserve_cycles=2_000_000_000_000)
  assert 'Ok'in wire.call('configure_paid',config)['result']
  records=json.loads((ROOT/'artifacts/text-short-v2/inputs.json').read_text())['records'];requests=[dict(model=manifest['model'],version=1,token_ids=r['token_ids'],options=r['options'])for r in records[:3]]
  quotes=[wire.call('quote',r)['result']['Ok']for r in requests]
  fee=quotes[0]['fee']
  def request(name):return dict(request=requests[0],request_id=name,quote_version=2)
  def fault(stage,trap=False,refund_fail=False):assert wire.call('paid_fault',dict(stage=stage,trap=trap,refund_fail=refund_fail))['result'] is None
  for name,stage,trap,pending in [('reject',0,False,False),('trap-continuation',15,True,False),('pending-refund',0,False,True)]:
   fault(stage,trap,pending);row=wire.call('infer',request(name),relay=callers[0],cycles=fee+12345)
   assert 'Failed'in row['result']['Err'],row
   error=row['result']['Err']['Failed'];assert error['refund']==('Pending' if pending else 'Done')
   assert row['forward']['refunded']==12345
   loss=row['forward']['balance_before']-row['forward']['balance_after']
   if pending:assert abs(loss-fee)<fee//100
   else:assert abs(loss)<fee//100
   receipt=wire.call('inference_status',name,relay=callers[0]);assert receipt['result']['refund']==error['refund']
   checks.append(dict(name=name,row=row,receipt=receipt))
   fault(None)
   retry=wire.call('retry_inference_refund',name,relay=callers[0]);assert retry['result']=={'Ok':'Done'}
   if pending:assert retry['forward']['balance_after']-retry['forward']['balance_before']>fee-fee//100
   again=wire.call('retry_inference_refund',name,relay=callers[0]);assert again['result']=={'Ok':'Done'}
   assert abs(again['forward']['balance_after']-again['forward']['balance_before'])<fee//100
   foreign=wire.call('retry_inference_refund',name,relay=callers[1]);assert foreign['result']=={'Err':'missing receipt'}
   checks.extend([dict(name=name+'-refund-retry',row=retry),dict(name=name+'-no-double-refund',row=again),dict(name=name+'-foreign-refund',row=foreign)])
   print(json.dumps(dict(check=name,refund=error['refund'],loss=loss)),flush=True);write(D/'checks.json',checks)
  fault(None)
  background=PaidTransport(D/'busy-background',TARGET,helper=ROOT/'artifacts/paid-update-v1/tools/args')
  with concurrent.futures.ThreadPoolExecutor(max_workers=1)as pool:
   future=pool.submit(background.call,'infer',request('busy-primary'),relay=callers[0],cycles=fee)
   current=None
   for _ in range(20):
    current=wire.call('inference_status','busy-primary',relay=callers[0])['result']
    if current is not None and current['state']=='Running':break
    time.sleep(.1)
   assert current is not None and current['state']=='Running'
   # Submit competing updates together; sequential CLI waits can outlast the job.
   secondary=PaidTransport(D/'busy-secondary',TARGET,helper=ROOT/'artifacts/paid-update-v1/tools/args')
   did=D/'probe.did';did.write_text('service:{paid_probe_step:()->(blob);}')
   def probe(job_id,label):
    probe_wire=PaidTransport(D/label,TARGET,helper=ROOT/'artifacts/paid-update-v1/tools/args')
    arg=probe_wire.args('inference_step',dict(job_id=job_id,stage=63),label)
    raw=subprocess.check_output(['icp','canister','call',TARGET,'paid_probe_step','--network','local','--identity','imajev-local','--candid',str(did),'--args-file',str(arg),'--args-format','bin','--output','hex'],cwd=ROOT,text=True)
    out=D/(label+'.reply.hex');out.write_text(raw)
    return probe_wire.decode('inference_step',out)
   def try_upgrade():
    return subprocess.run(['icp','canister','install',TARGET,'--mode','upgrade','--wasm',str(B/'full.wasm'),'--yes','--network','local','--identity','imajev-local'],cwd=ROOT,text=True,capture_output=True)
   with concurrent.futures.ThreadPoolExecutor(max_workers=4)as competitors:
    busy_future=competitors.submit(secondary.call,'infer',request('busy-secondary'),relay=callers[1],cycles=fee)
    stale_future=competitors.submit(probe,current['job_id']+1,'stale-probe')
    stage_future=competitors.submit(probe,current['job_id'],'stage-probe')
    upgrade_future=competitors.submit(try_upgrade)
    row=busy_future.result(timeout=180);assert row['result']=={'Err':'Busy'} and row['forward']['refunded']==fee
    checks.append(dict(name='busy-unpaid',row=row))
    stale=stale_future.result(timeout=180);assert stale=={'Err':'job progress mismatch'}
    checks.append(dict(name='stale-internal-job',result=stale))
    stage_error=stage_future.result(timeout=180);assert stage_error=={'Err':'job progress mismatch'}
    checks.append(dict(name='invalid-internal-stage',result=stage_error))
    upgrade=upgrade_future.result(timeout=180)
    assert upgrade.returncode!=0 and 'paid inference active'in upgrade.stderr,upgrade.stderr
    checks.append(dict(name='busy-upgrade-refused',stderr=upgrade.stderr))
   primary=future.result(timeout=180);assert 'Ok'in primary['result'];checks.append(dict(name='busy-primary-completed',row=primary))
  # Failed receipts and fee configuration survive an idle upgrade.
  saved=[wire.call('inference_status',n,relay=callers[0])['result']for n in ['reject','trap-continuation','pending-refund']]
  icp('stop',TARGET);stopped=True;icp('install',TARGET,'--mode','upgrade','--wasm',str(B/'full.wasm'),'--yes');icp('start',TARGET);stopped=False
  loaded=[wire.call('inference_status',n,relay=callers[0])['result']for n in ['reject','trap-continuation','pending-refund']];assert saved==loaded
  assert wire.call('paid_config')['result']==config
  checks.append(dict(name='upgrade-failed-receipts',equal=True));write(D/'checks.json',checks)
  assert sources=={str(p.relative_to(ROOT)):sha(p)for p in paths}
  write(D/'report.json',dict(complete=False,candidate=candidate,checks=checks,callers=callers,upgrade_receipts_equal=True,dual_bank=True))

 finally:
  if snapshot:
   if not stopped:icp('stop',TARGET);stopped=True
   icp('snapshot','restore',TARGET,snapshot);icp('start',TARGET);stopped=False
   t=bridge(D/'restored')
   try:verify_module(t,BASELINE);assert t.command(dict(op='weight_cache_status'))['ok']['cache']==cache;assert t.command(dict(op='pack_status'))['ok']==pack
   finally:t.close()
   icp('snapshot','delete',TARGET,snapshot);write(D/'restored.json',dict(module=BASELINE,cache_equal=True,pack_equal=True,snapshot_deleted=True))
  for caller in callers:icp('stop',caller)
 assert len(checks)>=16;r=json.loads((D/'report.json').read_text());r.update(complete=True,baseline_restored=True,snapshot_deleted=True);write(D/'report.json',r);print('paid proof complete; original canister restored',flush=True)
if __name__=='__main__':main()
