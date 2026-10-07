#!/usr/bin/env python3
"""Read-only source provenance audit while snapshot-protected proof runs."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    candidate=ROOT/'artifacts/paid-rank7-prepare8-v1';normal=ROOT/'artifacts/update-rank7-prepare8-v1';prior=ROOT/'artifacts/paid-common-raw-hybrid-v1'
    build=json.loads((candidate/'build/report.json').read_text());parent=json.loads((normal/'build/report.json').read_text())
    paths=[Path(__file__),candidate/'build/report.json',normal/'build/report.json']
    for owner,r in [(candidate,build),(normal,parent)]:
        assert sha(owner/'build/full.wasm')==r['wasm_sha256'];paths.append(owner/'build/full.wasm')
        for key in ['source_hashes','dependency_hashes']:
            for p,h in r[key].items():assert sha(ROOT/p)==h,p;paths.append(ROOT/p)
    assert build['update_candidate']==parent['wasm_sha256']
    for r in [build,parent]:assert len(r['patches'])==34 and all(p['wasmparser_validation']for p in r['patches'])
    original=json.loads((prior/'build/report.json').read_text())
    signatures=[[(p['export'],p['source_sha256'])for p in r['patches']]for r in [build,parent,original]]
    assert signatures[0]==signatures[1]==signatures[2]
    for name in ['lib.rs','update_inference.rs','paid_types.rs','paid_inference.rs']:
        new,old=candidate/'build'/name,prior/'build'/name;assert new.read_bytes()==old.read_bytes(),name;paths.extend([new,old])
    paths.extend([candidate/'optimized-paid-scheduler.rs',prior/'optimized-paid-scheduler.rs'])
    assert paths[-2].read_bytes()==paths[-1].read_bytes()
    result=dict(complete=True,module=build['wasm_sha256'],normal_module=parent['wasm_sha256'],canonical_billing_and_wrapper_byte_equal=True,scheduler_byte_equal=True,all34_patch_identities_equal_normal_and_latest_best=True,source_and_dependency_hashes_verified=True,full_paid_performance_verified=False,full_hidden_state_fidelity_verified=False,goal_complete=False,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(paths)})
    (candidate/'source-audit-readonly.json').write_text(json.dumps(result,indent=2)+'\n');print('paid prepare8 provenance and all34 unchanged projection identities verified')
if __name__=='__main__':main()
