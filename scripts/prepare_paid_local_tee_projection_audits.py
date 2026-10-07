#!/usr/bin/env python3
"""Stage audited reporting/guard programs, with restoration gates unchanged."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['sources','report','guards','final']);args=parser.parse_args()
    old=ROOT/'artifacts/paid-rank7-prepare8-v1';d=ROOT/'artifacts/paid-local-tee-projection-v1'
    sources={
      'sources':('audit_paid_rank7_prepare8_sources.py','source-audit-readonly.json','source_hashes'),
      'report':('report_paid_rank7_prepare8.py','reporter-entry-hashes.json',None),
      'guards':('prove_paid_rank7_prepare8_upgrade_guards.py',None,None),
      'final':('audit_paid_rank7_prepare8_report.py','post-report-audit.json',None)}
    hashes={}
    for phase,(name,manifest,key)in sources.items():
        original=ROOT/'scripts'/name
        if phase=='sources':assert sha(original)==json.loads((old/manifest).read_text())[key][str(original.relative_to(ROOT))]
        elif phase=='report':assert sha(original)==json.loads((old/manifest).read_text())[str(original.relative_to(ROOT))]
        elif phase=='guards':
            # Its byte-identical generic driver is pinned by the accepted proof;
            # retain both current wrapper and canonical driver's identity here.
            assert original.exists()
            generic=ROOT/'scripts/prove_paid_common_raw_hybrid_upgrade_guards.py'
            prior=json.loads((ROOT/'artifacts/paid-common-raw-hybrid-retry-v3/summary.json').read_text())
            assert sha(generic)==prior['workflow_hashes'][str(generic.relative_to(ROOT))]
            hashes[str(generic.relative_to(ROOT))]=sha(generic)
        else:assert sha(original)==json.loads((old/manifest).read_text())['audit_script_sha256']
        code=original.read_text().replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path('+repr(str(ROOT))+')',1)
        code=code.replace('artifacts/paid-rank7-prepare8-v1','artifacts/paid-local-tee-projection-v1')
        code=code.replace('artifacts/update-rank7-prepare8-v1','artifacts/update-local-tee-projection-v1')
        code=code.replace('artifacts/paid-common-raw-hybrid-retry-v3/proof/report.json','artifacts/paid-rank7-prepare8-unrolled-v1/proof/report.json')
        if phase=='sources':code=code.replace("prior=ROOT/'artifacts/paid-common-raw-hybrid-v1'","prior=ROOT/'artifacts/paid-rank7-prepare8-unrolled-v1'")
        if phase=='sources':
            before='    assert signatures[0]==signatures[1]==signatures[2]'
            assert code.count(before)==1
            code=code.replace(before,"""    assert signatures[0]==signatures[1]
    kernels=ROOT/'artifacts/local-tee-projection-kernels-v1/report.json'
    audit=ROOT/'artifacts/local-tee-projection-kernels-v1/independent-audit.json'
    kr=json.loads(kernels.read_text());ar=json.loads(audit.read_text())
    assert kr['complete'] and kr['all_memory_bytes_exact'] and kr['replacements']==524
    paths.extend([kernels,audit])
    changed={k['symbol']:k for k in kr['kernels']};assert len(changed)==12
    prior_s=dict(signatures[2]);current_s=dict(signatures[0])
    for symbol,h in current_s.items():
        if symbol in changed:
            k=changed[symbol];assert k['source_sha256']==prior_s[symbol]
            assert sha(ROOT/k['path'])==h;paths.append(ROOT/k['path'])
        else:assert h==prior_s[symbol]
    for r in [build,parent]:assert r['direct_patch_only'] and len(r['direct_patches'])==12 and not r['changed_runtime_files']
""")
            code=code.replace('all34_patch_identities_equal_normal_and_latest_best=True','all34_patch_identities_equal_normal=True, unchanged22_equal_latest_best=True, changed12_local_tee_verified=True')
        planned=d/f'planned-{phase}.py';compile(code,str(planned),'exec')
        if planned.exists():assert planned.read_text()==code
        else:planned.write_text(code)
        hashes[str(original.relative_to(ROOT))]=sha(original);hashes[str(planned.relative_to(ROOT))]=sha(planned)
    hashes[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    entry=d/'planned-audit-entry-hashes.json'
    if entry.exists():assert json.loads(entry.read_text())==hashes
    else:entry.write_text(json.dumps(hashes,indent=2)+'\n')
    if args.phase:
        planned=d/f'planned-{args.phase}.py'
        saved_argv=sys.argv;sys.argv=[str(planned)]+(['--run']if args.phase in ['report','guards']else[])
        try:exec(compile(planned.read_text(),str(planned),'exec'),dict(__file__=str(planned),__name__='__main__'))
        finally:sys.argv=saved_argv
    else:print('Prepared source/report/guard/final auditors; full-result stages remain restoration-gated')
if __name__=='__main__':main()
