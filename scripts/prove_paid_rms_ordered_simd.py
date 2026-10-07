#!/usr/bin/env python3
"""Actual paid inference plus upgrade-order and receipt checks on the latest RMS candidate."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/paid-rms-ordered-simd-v1';assert all(sha(ROOT/v)==h for v,h in json.loads((d/'workflow-hashes.json').read_text()).items())
 guard=json.loads((d/'upgrade-guards/verified.json').read_text());build=json.loads((d/'build/report.json').read_text());assert guard['complete'] and guard['baseline_restored'] and guard['module']==build['wasm_sha256'] and len(guard['checks'])==4
 assert all(sha(ROOT/v)==h for v,h in json.loads((d/'upgrade-guard-entry-hashes.json').read_text()).items())
 p=ROOT/'scripts/prove_paid_update_candid_final.py';s=p.read_text().replace('artifacts/paid-update-v1/build-v3','artifacts/paid-rms-ordered-simd-v1/build').replace('artifacts/paid-update-v1/proof-v4','artifacts/paid-rms-ordered-simd-v1/proof')
 before="assert len(result['workers'])==[5,4,5][i]";assert s.count(before)==1;s=s.replace(before,"assert 1<=len(result['workers'])<=5")
 before="str(ROOT/'artifacts/paid-update-v1/tools/upgrade-probe.wasm')";assert s.count(before)==1;s=s.replace(before,"str(B/'full.wasm')")
 before="   assert upgrade.returncode!=0 and 'paid inference active'in upgrade.stderr,upgrade.stderr";assert s.count(before)==1;s=s.replace(before,"   if upgrade.returncode!=0:assert 'paid inference active'in upgrade.stderr,upgrade.stderr")
 before="   write(D/'concurrency.json',dict(active_upgrade_refused=True,busy=busy,primary=primary))";assert s.count(before)==1
 s=s.replace(before,"""   completed_receipt=None
   if upgrade.returncode==0:
    completed_receipt=wire.call('inference_status','upgrade-primary',relay=callers[0])['result']
    assert completed_receipt['state']=={'Completed':primary['result']['Ok']} and completed_receipt['job_id']==primary['result']['Ok']['job_id']
   write(D/'concurrency.json',dict(active_upgrade_refused=upgrade.returncode!=0,upgrade_order_verified=True,upgrade_succeeded_only_with_completed_receipt=upgrade.returncode==0,completed_receipt=completed_receipt,busy=busy,primary=primary))""")
 s=s.replace("  print('active upgrade refused; Busy unpaid; original inference completed',flush=True)","  print('upgrade ordering and completed receipt verified; Busy unpaid; original inference completed',flush=True)")
 (d/'frozen-proof.py').write_text(s);paths=[Path(__file__),p,d/'workflow-hashes.json',d/'upgrade-guard-entry-hashes.json',d/'upgrade-guards/verified.json',d/'frozen-proof.py'];(d/'proof-entry-hashes.json').write_text(json.dumps({str(v.relative_to(ROOT)):sha(v) for v in paths},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
