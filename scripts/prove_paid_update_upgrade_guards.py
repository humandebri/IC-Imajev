#!/usr/bin/env python3
"""Exercise real pre_upgrade guards and failed-receipt persistence without weights."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'client'))
from transport import Transport
from prefix_inference import verify_module
from paid_update_transport import PaidTransport
T='6eydd-o3777-77775-aaama-cai';BASE='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
B=R/'artifacts/paid-update-v1/build-upgrade-diagnostic';D=R/'artifacts/paid-update-v1/upgrade-guards-v1'
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=False);m=json.loads((R/'checkpoints/full-int8.manifest.json').read_text());b=json.loads((B/'report.json').read_text())
 for hashes in [b['source_hashes'],b['dependency_hashes']]:assert all(sha(R/f)==h for f,h in hashes.items())
 assert sha(B/'full.wasm')==b['wasm_sha256'];events=[]
 def icp(*a,check=True):
  p=subprocess.run(['icp','canister',*a,'--network','local','--identity','imajev-local'],cwd=R,text=True,capture_output=True)
  events.append(dict(args=list(a),code=p.returncode,stdout=p.stdout,stderr=p.stderr));write(D/'operations.json',events)
  if check and p.returncode:raise RuntimeError(p.stderr)
  return p
 def bridge(path):return Transport(m['model'],'http://localhost:8001/',T,str(R/'artifacts/imajev-local.pem'),path,m['pack_hash'],bridge_binary=str(R/'artifacts/query-packing-v3/build/imajev-client'))
 t=bridge(D/'before');verify_module(t,BASE);cache=t.command(dict(op='weight_cache_status'))['ok']['cache'];pack=t.command(dict(op='pack_status'))['ok'];t.close()
 snapshot=None;checks=[]
 try:
  icp('stop',T);snapshot=icp('snapshot','create',T,'--quiet').stdout.strip();write(D/'snapshot.json',dict(id=snapshot))
  icp('install',T,'--mode','upgrade','--wasm',str(B/'full.wasm'),'--yes');icp('start',T)
  did=D/'probe.did';did.write_text('service:{paid_upgrade_probe:(bool,nat32)->();}')
  def fixture(active,state):icp('call',T,'paid_upgrade_probe',f'({str(active).lower()}, {state}:nat32)','--candid',str(did))
  wire=PaidTransport(D/'calls',T)
  for label,active,state,reason in [('active',True,0,'paid inference active'),('refund',False,2,'refund in flight')]:
   fixture(active,state)
   p=icp('install',T,'--mode','upgrade','--wasm',str(R/'artifacts/paid-update-v1/tools/upgrade-probe.wasm'),'--yes',check=False)
   assert p.returncode!=0 and reason in p.stderr,p.stderr
   icp('start',T);fixture(False,0);checks.append(dict(name=label+'-upgrade-refused',stderr=p.stderr));print(label+' upgrade refused',flush=True)
  for state,label in [(1,'pending'),(3,'done')]:
   fixture(False,state);before=wire.call('inference_status','diagnostic-upgrade')['result'];assert before['refund']==label.title()
   icp('stop',T);icp('install',T,'--mode','upgrade','--wasm',str(B/'full.wasm'),'--yes');icp('start',T)
   after=wire.call('inference_status','diagnostic-upgrade')['result'];assert after==before
   checks.append(dict(name='failed-'+label+'-receipt-preserved',before=before,after=after));print(label+' receipt preserved',flush=True)
  fixture(False,0)
 finally:
  if snapshot:
   icp('stop',T);icp('snapshot','restore',T,snapshot);icp('start',T)
   t=bridge(D/'restored');verify_module(t,BASE);assert t.command(dict(op='weight_cache_status'))['ok']['cache']==cache;assert t.command(dict(op='pack_status'))['ok']==pack;t.close();icp('snapshot','delete',T,snapshot)
   write(D/'restored.json',dict(module=BASE,cache_equal=True,pack_equal=True,snapshot_deleted=True))
 assert len(checks)==4
 write(D/'verified.json',dict(complete=True,module=b['wasm_sha256'],diagnostic_fixture=True,checks=checks,baseline_restored=True,scope='Real IC pre_upgrade and stable-memory hooks with owner-only synthetic Job/receipt fixtures. No inference execution is claimed here. Actual inference and explicit refund are verified separately.'))
 print('upgrade guards complete; baseline restored',flush=True)
if __name__=='__main__':main()
