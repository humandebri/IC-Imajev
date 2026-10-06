#!/usr/bin/env python3
"""Recheck guard messages, frozen sources and raw stable-receipt replies."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'artifacts/paid-update-v1/upgrade-guards-v1';B=R/'artifacts/paid-update-v1/build-upgrade-diagnostic';P=R/'artifacts/paid-update-v1/build-v3'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 v=load(D/'verified.json');b=load(B/'report.json');assert v['complete'] and v['baseline_restored'] and sha(B/'full.wasm')==v['module']==b['wasm_sha256']
 for h in [b['source_hashes'],b['dependency_hashes']]:assert all(sha(R/f)==x for f,x in h.items())
 assert (B/'paid_inference.rs').read_text().split('#[cfg(test)]')[0].rstrip()==(P/'paid_inference.rs').read_text().rstrip()
 rows=[]
 for f in sorted((D/'calls').glob('[0-9][0-9][0-9][0-9]-*.json')):
  if '.input.'in f.name:continue
  row=load(f);raw=json.loads(subprocess.check_output([str(R/'artifacts/paid-update-v1/tools/args'),'decode',row['kind'],str(R/row['reply_path'])],text=True));assert raw==row['result'];rows.append(raw)
 assert len(rows)==4 and rows[0]==rows[1] and rows[2]==rows[3] and rows[0]['refund']=='Pending' and rows[2]['refund']=='Done'
 ops=load(D/'operations.json');assert any(o['code']!=0 and 'paid inference active'in o['stderr'] for o in ops) and any(o['code']!=0 and 'refund in flight'in o['stderr'] for o in ops)
 restored=load(D/'restored.json');assert restored['cache_equal'] and restored['pack_equal'] and restored['snapshot_deleted']
 out=dict(complete=True,module=v['module'],core_source_equal_to_production=True,raw_receipt_replies_verified=4,active_upgrade_refused=True,inflight_refund_upgrade_refused=True,failed_pending_done_receipts_preserved=True,baseline_restored=True,scope=v['scope'])
 (D/'independent-verified.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
