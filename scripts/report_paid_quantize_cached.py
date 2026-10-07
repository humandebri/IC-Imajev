#!/usr/bin/env python3
"""Audit actual paid cached quantization inference and separately verified deterministic upgrade guards."""
from pathlib import Path
import hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/paid-quantize-cached-v1';p=ROOT/'scripts/report_paid_add_norm_simd.py';s=p.read_text().replace('artifacts/paid-add-norm-simd-v2','artifacts/paid-quantize-cached-v1')
 before="assert concurrency['active_upgrade_refused'] and concurrency['busy']['result']=={'Err':'Busy'} and 'Ok' in concurrency['primary']['result']";assert s.count(before)==1
 s=s.replace(before,"assert concurrency['upgrade_order_verified'] and concurrency['busy']['result']=={'Err':'Busy'} and 'Ok' in concurrency['primary']['result']\n  if not concurrency['active_upgrade_refused']:\n   assert concurrency['upgrade_succeeded_only_with_completed_receipt'] and concurrency['completed_receipt']['state']=={'Completed':concurrency['primary']['result']['Ok']}")
 s=s.replace('Concurrent active upgrade succeeded; execution-order race unresolved. Paused and same-version upgrade checks not executed.','Paid cached quantization proof incomplete; inspect frozen proof and saved Candid replies for failed test.')
 (d/'frozen-reporter.py').write_text(s);exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 guard=d/'upgrade-guards/verified.json';g=json.loads(guard.read_text());summary_name='partial-summary.json' if '--partial' in sys.argv else 'summary.json';r=json.loads((d/summary_name).read_text());assert g['complete'] and g['baseline_restored'] and g['module']==r['module'] and len(g['checks'])==4
 assert all(sha(ROOT/v)==h for v,h in json.loads((d/'upgrade-guard-entry-hashes.json').read_text()).items())
 b=json.loads((d/'build/report.json').read_text());assert b['raw_packets_discarded_after_validation']
 r['deterministic_upgrade_guards_verified']=True;r['raw_packets_discarded_after_validation']=True
 extra=[p,d/'frozen-reporter.py',guard,d/'upgrade-guard-entry-hashes.json',d/'frozen-upgrade-guards.py'];r['workflow_hashes'].update({str(v.relative_to(ROOT)):sha(v) for v in extra});(d/summary_name).write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(d/('frozen-paid-partial-proof.zip' if '--partial' in sys.argv else 'frozen-paid-proof.zip'),'a',zipfile.ZIP_DEFLATED)as z:
  for v in extra:z.write(v,str(v.relative_to(ROOT)))
  z.write(d/summary_name,str((d/summary_name).relative_to(ROOT)))
if __name__=='__main__':main()
