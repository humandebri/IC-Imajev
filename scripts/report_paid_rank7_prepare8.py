#!/usr/bin/env python3
"""Prepare or execute complete independent report after the paid proof restores."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    d=ROOT/'artifacts/paid-rank7-prepare8-v1';original=ROOT/'scripts/report_paid_common_raw_hybrid.py'
    prior=json.loads((ROOT/'artifacts/paid-common-raw-hybrid-retry-v3/summary.json').read_text())
    assert sha(original)==prior['workflow_hashes'][str(original.relative_to(ROOT))]
    code=original.read_text().replace("ROOT = Path(__file__).resolve().parents[1]",'ROOT = Path('+repr(str(ROOT))+')')
    code=code.replace("candidate = ROOT / 'artifacts/paid-common-raw-hybrid-v1'","candidate = ROOT / 'artifacts/paid-rank7-prepare8-v1'")
    code=code.replace("directory / 'entry-hashes.json'","directory / 'proof-entry-hashes.json'")
    code=code.replace("saved = read(ROOT / 'artifacts/paid-k2-pair-guard-fold-v1/proof/report.json')","saved = read(ROOT / 'artifacts/paid-common-raw-hybrid-retry-v3/proof/report.json')")
    before="cases.append(dict(case=name, total_handler_instructions=total,"
    assert code.count(before)==1
    code=code.replace(before,"cases.append(dict(case=name, prior_total_handler_instructions=sum(w['instructions'] for w in old['row']['result']['Ok']['workers']), total_handler_instructions=total,")
    code=code.replace('saved_candid_replies_verified=True,','new_dense_f32_direct_capture_verified=False, saved_candid_replies_verified=True,')
    assert 'deterministic_extra_upgrade_guards_verified=guards is not None' in code
    compile(code,str(d/'frozen-reporter.py'),'exec')
    frozen=d/'frozen-reporter.py'
    if frozen.exists():assert frozen.read_text()==code
    else:frozen.write_text(code)
    files=[Path(__file__),original,frozen,d/'source-audit-readonly.json']
    entry=d/'reporter-entry-hashes.json'
    values={str(p.relative_to(ROOT)):sha(p)for p in files}
    if entry.exists():assert json.loads(entry.read_text())==values
    else:entry.write_text(json.dumps(values,indent=2)+'\n')
    if args.run:
        r=json.loads((d/'proof/report.json').read_text());assert r['complete'] and r['baseline_restored'] and r['snapshot_deleted']
        saved_argv=sys.argv;sys.argv=[str(frozen),'--directory',str(d)]
        try:exec(compile(code,str(frozen),'exec'),dict(__file__=str(frozen),__name__='__main__'))
        finally:sys.argv=saved_argv
    else:print('Prepared reporter only; no completed proof claim or canister mutation')
if __name__=='__main__':main()
