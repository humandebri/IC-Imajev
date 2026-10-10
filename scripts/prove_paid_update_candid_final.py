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
B=ROOT/'artifacts/paid-update-v1/build-v3';D=ROOT/'artifacts/paid-update-v1/proof-v4';OWNER='cibxp-okw3l-gfzvi-p6ltu-23ss3-7tlfz-nk65x-tvhvb-mg56y-b5hkf-eqe'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def digest(v):return hashlib.sha256(np.asarray(v,dtype='<f4').tobytes()).hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=False);manifest=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());build=json.loads((B/'report.json').read_text());candidate=sha(B/'full.wasm');assert candidate==build['wasm_sha256']
 for key in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in build[key].items())
 paths=[Path(__file__),ROOT/'scripts/paid_update_transport.py',ROOT/'artifacts/paid-update-v1/tools/report.json',ROOT/'artifacts/paid-update-v1/tools/args',ROOT/'artifacts/paid-update-v1/tools/caller.wasm',ROOT/'artifacts/text-short-v2/inputs.json',B/'report.json',B/'full.wasm',ROOT/'canisters/inference/paid-inference.did',ROOT/'artifacts/paid-update-v1/tools/upgrade-probe.wat',ROOT/'artifacts/paid-update-v1/tools/upgrade-probe.wasm']
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
  typed=subprocess.check_output(['icp','canister','call',TARGET,'paid_config','--network','local','--identity','imajev-local','--candid',str(ROOT/'canisters/inference/paid-inference.did'),'()','--query','--output','hex'],cwd=ROOT,text=True)
  typed_path=D/'typed-config.reply.hex';typed_path.write_text(typed);assert wire.decode('paid_config',typed_path)==config
  records=json.loads((ROOT/'artifacts/text-short-v2/inputs.json').read_text())['records'];requests=[dict(model=manifest['model'],version=1,token_ids=r['token_ids'],options=r['options'])for r in records[:3]]
  quotes=[wire.call('quote',r)['result']['Ok']for r in requests]
  for repeat in range(1,2):
   for i,name in enumerate(['617','620','653']):
    request=dict(request=requests[i],request_id=f'boom-{name}-{repeat}',quote_version=2);fee=quotes[i]['fee'];row=wire.call('infer',request,relay=callers[0],cycles=fee+12_345_678);assert 'Ok'in row['result'],row
    result=row['result']['Ok'];assert row['forward']['refunded']==12_345_678;assert result['paid_cycles']==fee;assert len(result['workers'])==[5,4,5][i]
    assert all(m['instructions']<40_000_000_000 and m['heap_pages']*65536<2**32 for m in result['workers'])
    debug=wire.call('paid_debug')['result'];assert debug['job_id']==result['job_id'];old=ROOT/f'artifacts/boomdao-current-v1/{name}-r1';reference=json.loads((old/'report.json').read_text())['decision_query']['ok']['decision'];a=result['decision'];assert {k:v for k,v in a.items()if k!='instructions'}=={k:v for k,v in reference.items()if k!='instructions'}
    p=quotes[i]['prefix_tokens'];assert digest(debug['final_hidden'])==digest(np.load(old/'final-hidden.npy',allow_pickle=False))
    for layer in range(32):
     hidden=old/f'queries/layer-{layer:02d}.npy'
     if hidden.exists():
      v=np.load(hidden,allow_pickle=False);v=v[p:] if layer<31 else v;assert digest(v)==debug['hidden_hashes'][layer],(name,layer,'hidden')
     else:assert layer==30
     with np.load(old/f'queries/states/layer-{layer:02d}.npz',allow_pickle=False)as z:v=np.concatenate([z['keys'][p:].ravel(),z['values'][p:].ravel()])if layer%4==3 else z['conv'].ravel();assert digest(v)==debug['state_hashes'][layer],(name,layer,'state')
    results.append(dict(case=name,repeat=repeat,request=request,quote=quotes[i],row=row,debug=debug));write(D/'progress.json',results);print(json.dumps(dict(case=name,repeat=repeat,workers=len(result['workers']),instructions=sum(m['instructions']for m in result['workers']),seconds=row['seconds'],heap=max(m['heap_pages']*65536 for m in result['workers']))),flush=True)
  # Unaccepted attached cycles must return in full, including duplicate requests.
  duplicate=wire.call('infer',results[0]['request'],relay=callers[0],cycles=quotes[0]['fee']);assert duplicate['result']['Ok']==results[0]['row']['result']['Ok'] and duplicate['forward']['refunded']==quotes[0]['fee'];checks.append(dict(name='duplicate',row=duplicate))
  conflict=json.loads(json.dumps(results[0]['request']));conflict['request']['token_ids'][-1]+=1;row=wire.call('infer',conflict,relay=callers[0],cycles=quotes[0]['fee']);assert row['result']=={'Err':'IdConflict'} and row['forward']['refunded']==quotes[0]['fee'];checks.append(dict(name='id-conflict',row=row))
  req=dict(request=requests[0],request_id='insufficient',quote_version=2);row=wire.call('infer',req,relay=callers[0],cycles=1);assert 'InsufficientCycles'in row['result']['Err'] and row['forward']['refunded']==1;checks.append(dict(name='insufficient',row=row))
  req['request_id']='old-quote';req['quote_version']=1;row=wire.call('infer',req,relay=callers[0],cycles=quotes[0]['fee']);assert 'QuoteChanged'in row['result']['Err'] and row['forward']['refunded']==quotes[0]['fee'];checks.append(dict(name='quote-version',row=row))
  req['request_id']='invalid-token';req['quote_version']=2;req['request']=json.loads(json.dumps(requests[0]));req['request']['token_ids'][-1]=248320;row=wire.call('infer',req,relay=callers[0],cycles=quotes[0]['fee']);assert 'Invalid'in row['result']['Err'] and row['forward']['refunded']==quotes[0]['fee'];checks.append(dict(name='token-bounds',row=row))
  row=wire.call('inference_step',dict(job_id=1,stage=0),relay=callers[0]);assert row['result']=={'Err':'self only'};checks.append(dict(name='worker-authority',row=row))
  row=wire.call('inference_status','boom-617-1',relay=callers[1]);assert row['result'] is None;checks.append(dict(name='status-authority',row=row))
  write(D/'checks.json',checks)
  # Use a tiny replacement module so upload time cannot hide an active upgrade.
  background=PaidTransport(D/'upgrade-background',TARGET,helper=ROOT/'artifacts/paid-update-v1/tools/args')
  secondary=PaidTransport(D/'busy-secondary',TARGET,helper=ROOT/'artifacts/paid-update-v1/tools/args')
  active_request=dict(request=requests[0],request_id='upgrade-primary',quote_version=2)
  with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
   primary_future=pool.submit(background.call,'infer',active_request,relay=callers[0],cycles=quotes[0]['fee'])
   current=None
   for _ in range(20):
    current=wire.call('inference_status','upgrade-primary',relay=callers[0])['result']
    if current is not None and current['state']=='Running':break
    time.sleep(.1)
   assert current is not None and current['state']=='Running'
   def upgrade_probe():
    return subprocess.run(['icp','canister','install',TARGET,'--mode','upgrade','--wasm',str(ROOT/'artifacts/paid-update-v1/tools/upgrade-probe.wasm'),'--yes','--network','local','--identity','imajev-local'],cwd=ROOT,text=True,capture_output=True)
   upgrading=pool.submit(upgrade_probe)
   competing=pool.submit(secondary.call,'infer',dict(request=requests[0],request_id='busy-secondary',quote_version=2),relay=callers[1],cycles=quotes[0]['fee'])
   upgrade=upgrading.result(timeout=180)
   write(D/'active-upgrade.json',dict(returncode=upgrade.returncode,stdout=upgrade.stdout,stderr=upgrade.stderr,running_receipt=current))
   assert upgrade.returncode!=0 and 'paid inference active'in upgrade.stderr,upgrade.stderr
   busy=competing.result(timeout=180);assert busy['result']=={'Err':'Busy'} and busy['forward']['refunded']==quotes[0]['fee']
   primary=primary_future.result(timeout=180);assert 'Ok'in primary['result']
   assert {k:v for k,v in primary['result']['Ok']['decision'].items()if k!='instructions'}=={k:v for k,v in results[0]['row']['result']['Ok']['decision'].items()if k!='instructions'}
   write(D/'concurrency.json',dict(active_upgrade_refused=True,busy=busy,primary=primary))
  print('active upgrade refused; Busy unpaid; original inference completed',flush=True)
  # Same-version upgrade must preserve authority, config and receipts; empty cache blocks new work.
  config['enabled']=False;config['version']=3;assert 'Ok'in wire.call('configure_paid',config)['result']
  typed=subprocess.check_output(['icp','canister','call',TARGET,'paid_config','--network','local','--identity','imajev-local','--candid',str(ROOT/'canisters/inference/paid-inference.did'),'()','--query','--output','hex'],cwd=ROOT,text=True)
  typed_path=D/'typed-config.reply.hex';typed_path.write_text(typed);assert wire.decode('paid_config',typed_path)==config
  req=dict(request=requests[0],request_id='paused',quote_version=3);row=wire.call('infer',req,relay=callers[0],cycles=quotes[0]['fee']);assert row['result']=={'Err':'Paused'} and row['forward']['refunded']==quotes[0]['fee'];checks.append(dict(name='paused',row=row))
  before=wire.call('inference_status','boom-617-1',relay=callers[0])['result'];icp('stop',TARGET);stopped=True;icp('install',TARGET,'--mode','upgrade','--wasm',str(B/'full.wasm'),'--yes');icp('start',TARGET);stopped=False
  after=wire.call('inference_status','boom-617-1',relay=callers[0])['result'];assert before==after;assert wire.call('paid_config')['result']==config
  row=wire.call('infer',results[0]['request'],relay=callers[0],cycles=quotes[0]['fee']);assert 'Ok'in row['result'] and row['forward']['refunded']==quotes[0]['fee'];checks.append(dict(name='upgrade-replay',row=row));write(D/'checks.json',checks)
  assert sources=={str(p.relative_to(ROOT)):sha(p)for p in paths}
  write(D/'report.json',dict(complete=False,candidate=candidate,results=results,checks=checks,callers=callers,upgrade_receipts_equal=True,dual_bank=True))
 finally:
  if snapshot:
   if not stopped:icp('stop',TARGET);stopped=True
   icp('snapshot','restore',TARGET,snapshot);icp('start',TARGET);stopped=False
   t=bridge(D/'restored')
   try:verify_module(t,BASELINE);assert t.command(dict(op='weight_cache_status'))['ok']['cache']==cache;assert t.command(dict(op='pack_status'))['ok']==pack
   finally:t.close()
   icp('snapshot','delete',TARGET,snapshot);write(D/'restored.json',dict(module=BASELINE,cache_equal=True,pack_equal=True,snapshot_deleted=True))
  for caller in callers:icp('stop',caller)
 assert len(results)==3;r=json.loads((D/'report.json').read_text());r.update(complete=True,baseline_restored=True,snapshot_deleted=True);write(D/'report.json',r);print('paid proof complete; original canister restored',flush=True)
if __name__=='__main__':main()
