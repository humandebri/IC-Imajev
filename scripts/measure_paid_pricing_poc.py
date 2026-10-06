#!/usr/bin/env python3
"""Snapshot-protected paid inference proof using an actual caller canister."""
import hashlib,json,subprocess,sys,time,zipfile,concurrent.futures
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport
from prefix_inference import verify_module
from paid_update_transport import PaidTransport
TARGET='6eydd-o3777-77775-aaama-cai';BASELINE='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
B=ROOT/'artifacts/paid-update-v1/build-v3';D=ROOT/'artifacts/paid-pricing-poc-v1';OWNER='cibxp-okw3l-gfzvi-p6ltu-23ss3-7tlfz-nk65x-tvhvb-mg56y-b5hkf-eqe'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def digest(v):return hashlib.sha256(np.asarray(v,dtype='<f4').tobytes()).hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=False);manifest=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());build=json.loads((B/'report.json').read_text());candidate=sha(B/'full.wasm');assert candidate==build['wasm_sha256']
 for key in ['dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in build[key].items())
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
  for i in range(1):
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
  wire=PaidTransport(D/'calls',TARGET);config=dict(enabled=True,version=2,base_fee=100_000_000_000,fee_per_token=3_000_000_000,reserve_cycles=2_000_000_000_000)
  assert 'Ok'in wire.call('configure_paid',config)['result']
  typed=subprocess.check_output(['icp','canister','call',TARGET,'paid_config','--network','local','--identity','imajev-local','--candid',str(ROOT/'canisters/inference/paid-inference.did'),'()','--query','--output','hex'],cwd=ROOT,text=True)
  typed_path=D/'typed-config.reply.hex';typed_path.write_text(typed);assert wire.decode('paid_config',typed_path)==config
  records=json.loads((ROOT/'artifacts/text-short-v2/inputs.json').read_text())['records'];requests=[dict(model=manifest['model'],version=1,token_ids=r['token_ids'],options=r['options'])for r in records[:3]]
  def status(label):
   raw=icp('status',TARGET,'--json');v=json.loads(raw);write(D/(label+'.status.json'),v)
   return dict(cycles=int(v['cycles'].replace('_','')),reserved=int(v['reserved_cycles'].replace('_','')),idle_day=int(v['idle_cycles_burned_per_day'].replace('_','')),memory=int(v['memory_size'].replace('_','')),time=time.time())
  def run_case(name,request,version):
   quote=wire.call('quote',request)['result']['Ok'];assert quote['version']==version
   before=status(name+'-before');row=wire.call('infer',dict(request=request,request_id=name,quote_version=version),relay=callers[0],cycles=quote['fee']+12345678);after=status(name+'-after')
   assert 'Ok' in row['result'],row
   result=row['result']['Ok'];assert result['paid_cycles']==quote['fee'] and row['forward']['refunded']==12345678
   assert all(m['instructions']<40000000000 and m['heap_pages']*65536<2**32 for m in result['workers'])
   elapsed=after['time']-before['time'];balance_burn=before['cycles']+before['reserved']+quote['fee']-after['cycles']-after['reserved']
   idle_estimate=elapsed*(before['idle_day']+after['idle_day'])/2/86400
   v=dict(name=name,quote=quote,row=row,before=before,after=after,instructions=sum(m['instructions'] for m in result['workers']),balance_burn=balance_burn,idle_estimate=idle_estimate,variable_estimate=balance_burn-idle_estimate)
   results.append(v);write(D/'progress.json',results);print(json.dumps({k:v[k] for k in ['name','instructions','balance_burn','variable_estimate']}),flush=True)
   return v
  # Synthetic shapes measure cost, not semantic question accuracy.
  voting=requests[0]['token_ids'][:38];common=requests[2]['token_ids'][:27]
  for bank,prefix in [('common',common),('voting',voting)]:
   for n in [1,24,57]:
    request=dict(model=manifest['model'],version=1,token_ids=prefix+([846,198,56555,279,2420,5721,321]*9)[:n],options=['yes','no'])
    run_case(f'calibrate-{bank}-{n}',request,2)
  # POC: storage is an operator subsidy; base covers entry/final processing.
  base_fee=5_000_000_000;unit=100_000_000
  # Use official 13-node execution coefficient, and a floor for worker/callback overhead.
  slope=max((r['instructions']*1.2+1_000_000_000-base_fee)/r['quote']['suffix_tokens'] for r in results)
  fee_per_token=max(100_000_000,int(np.ceil(slope/unit))*unit)
  config.update(version=3,base_fee=base_fee,fee_per_token=fee_per_token)
  write(D/'poc-config.json',config);assert 'Ok' in wire.call('configure_paid',config)['result']
  for i,name in enumerate(['617','620','653']):
   r=run_case('new-fee-'+name,requests[i],3)
   old=ROOT/f'artifacts/boomdao-current-v1/{name}-r1';reference=json.loads((old/'report.json').read_text())['decision_query']['ok']['decision'];decision=r['row']['result']['Ok']['decision']
   assert {k:v for k,v in decision.items() if k!='instructions'}=={k:v for k,v in reference.items() if k!='instructions'}
  duplicate=wire.call('infer',results[-1]['row']['request'],relay=callers[0],cycles=results[-1]['quote']['fee']);assert duplicate['forward']['refunded']==results[-1]['quote']['fee'] and duplicate['result']==results[-1]['row']['result'];checks.append(dict(name='duplicate-new-fee',row=duplicate))
  for label,version,attached in [('insufficient',3,1),('old-version',2,results[-1]['quote']['fee'])]:
   req=dict(request=requests[2],request_id=label,quote_version=version);row=wire.call('infer',req,relay=callers[0],cycles=attached);assert 'Err' in row['result'] and row['forward']['refunded']==attached;checks.append(dict(name=label,row=row))
  write(D/'checks.json',checks)
  assert wire.call('paid_config')['result']==config
  write(D/'report.json',dict(complete=True,candidate=candidate,results=results,checks=checks,config=config,scope='POC: six cost shapes, three regression inputs. Storage subsidized by operator. 13-node execution coefficient; local balance burn recorded separately.'))
 finally:
  if snapshot:
   if not stopped:icp('stop',TARGET);stopped=True
   icp('snapshot','restore',TARGET,snapshot);icp('start',TARGET);stopped=False
   t=bridge(D/'restored')
   try:verify_module(t,BASELINE);assert t.command(dict(op='weight_cache_status'))['ok']['cache']==cache;assert t.command(dict(op='pack_status'))['ok']==pack
   finally:t.close()
   icp('snapshot','delete',TARGET,snapshot);write(D/'restored.json',dict(module=BASELINE,cache_equal=True,pack_equal=True,snapshot_deleted=True))
  for caller in callers:icp('stop',caller)
 print('pricing POC complete; baseline restored',flush=True)
if __name__=='__main__':main()
