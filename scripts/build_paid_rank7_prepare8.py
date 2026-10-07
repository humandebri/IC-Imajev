#!/usr/bin/env python3
"""Keep latest paid contract/scheduler while using the prepare8 runtime."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=ROOT/'artifacts/paid-common-raw-hybrid-v1'
    normal=ROOT/'artifacts/update-rank7-prepare8-v1'
    for p,h in json.loads((old/'workflow-hashes.json').read_text()).items():assert sha(ROOT/p)==h,p
    parent=json.loads((old/'build/report.json').read_text())
    for p,h in parent['source_hashes'].items():assert sha(ROOT/p)==h,p
    base=json.loads((normal/'build/report.json').read_text());assert len(base['patches'])==34
    d=ROOT/'artifacts/paid-rank7-prepare8-v1';d.mkdir(exist_ok=False)
    (d/'optimized-paid-scheduler.rs').write_bytes((old/'optimized-paid-scheduler.rs').read_bytes())
    code=(old/'frozen-builder.py').read_text().replace('artifacts/update-common-raw-hybrid-v1','artifacts/update-rank7-prepare8-v1').replace('artifacts/paid-common-raw-hybrid-v1','artifacts/paid-rank7-prepare8-v1')
    code=code.replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path('+repr(str(ROOT))+')')
    frozen=d/'frozen-builder.py';frozen.write_text(code)
    files=[Path(__file__),old/'workflow-hashes.json',old/'build/report.json',old/'frozen-builder.py',old/'optimized-paid-scheduler.rs',normal/'build/report.json',frozen,d/'optimized-paid-scheduler.rs']
    (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
    exec(compile(code,str(frozen),'exec'),dict(__file__=str(frozen),__name__='__main__'))
    report=json.loads((d/'build/report.json').read_text())
    for name in ['lib.rs','update_inference.rs','paid_types.rs','paid_inference.rs']:
        assert (d/'build'/name).read_bytes()==(old/'build'/name).read_bytes(),name
    (d/'paid-source-equivalence.json').write_text(json.dumps(dict(complete=True,paid_contract_and_scheduler_and_lib_byteexact=True,module=report['wasm_sha256'],full_fidelity_and_instruction_proof_pending=True),indent=2)+'\n')
if __name__=='__main__':main()
