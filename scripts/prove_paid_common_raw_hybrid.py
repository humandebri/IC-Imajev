#!/usr/bin/env python3
"""Snapshot-protected complete paid trial; compare all32 previously audited states."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/paid-k2-pair-guard-fold-v1';d=ROOT/'artifacts/paid-common-raw-hybrid-v1'
 for p,h in json.loads((old/'proof-entry-hashes.json').read_text()).items():assert sha(ROOT/p)==h,p
 previous=json.loads((old/'proof/report.json').read_text());summary=json.loads((old/'summary.json').read_text());audit=json.loads((old/'post-report-audit.json').read_text())
 assert previous['complete']and len(previous['results'])==3 and summary['complete']and summary['all_32_hidden_verified']and audit['complete']and audit['all_32_hidden_report_and_reference_hashes_verified']
 storage=Path.home()/'.codex/goal-artifacts/01a10ffe-1dc8-7931-b3de-a3e4d35bd910/paid-common-raw-hybrid-v1';storage.mkdir(parents=True,exist_ok=False);(d/'proof-storage').symlink_to(storage,target_is_directory=True)
 p=old/'frozen-proof.py';s=p.read_text().replace('artifacts/paid-k2-pair-guard-fold-v1','artifacts/paid-common-raw-hybrid-v1')
 s=s.replace("D=ROOT/'artifacts/paid-common-raw-hybrid-v1/proof'","D=ROOT/'artifacts/paid-common-raw-hybrid-v1/proof-storage/proof'")
 before="    debug=wire.call('paid_debug')['result'];assert debug['job_id']==result['job_id'];"
 after="""    debug=wire.call('paid_debug')['result'];assert debug['job_id']==result['job_id'];
    prior=json.loads((ROOT/'artifacts/paid-k2-pair-guard-fold-v1/proof/report.json').read_text());saved=next(v for v in prior['results']if v['case']==name)
    assert debug['hidden_hashes']==saved['debug']['hidden_hashes'],(name,'all32 hidden')
    assert debug['state_hashes']==saved['debug']['state_hashes'],(name,'all32 state')
    assert digest(debug['final_hidden'])==digest(saved['debug']['final_hidden']),(name,'final hidden')
"""
 assert s.count(before)==1;s=s.replace(before,after+"    ")
 before="  # Unaccepted attached cycles must return in full"
 s=s.replace(before,"  write(D/'all32-reference-comparison.json',dict(complete=True,reference='artifacts/paid-k2-pair-guard-fold-v1/proof/report.json',reference_sha256=sha(ROOT/'artifacts/paid-k2-pair-guard-fold-v1/proof/report.json'),all3_inputs_32_hidden_and_state_hashes_equal=True))\n"+before,1)
 # Keep old direct historical comparisons and all paid API/concurrency/upgrade/recovery checks.
 before="paths=[Path(__file__),";s=s.replace(before,"paths=[ROOT/'artifacts/paid-k2-pair-guard-fold-v1/proof/report.json',ROOT/'artifacts/paid-k2-pair-guard-fold-v1/summary.json',ROOT/'artifacts/paid-k2-pair-guard-fold-v1/post-report-audit.json',Path(__file__),",1)
 frozen=d/'frozen-proof.py';frozen.write_text(s)
 files=[Path(__file__),p,old/'proof-entry-hashes.json',old/'proof/report.json',old/'summary.json',old/'post-report-audit.json',d/'workflow-hashes.json',d/'build/report.json',frozen]
 (d/'proof-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
