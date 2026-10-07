#!/usr/bin/env python3
"""Fresh matrix/response/ZIP audit of the hoisted preparation prototype."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    original=ROOT/'scripts/audit_rank7_prepare8_probe.py';old=ROOT/'artifacts/rank7-prepare8-probe-v1'
    r=json.loads((old/'independent-audit.json').read_text());assert sha(original)==r['source_hashes'][str(original.relative_to(ROOT))]
    d=ROOT/'artifacts/rank7-prepare8-unrolled-probe-v1'
    code=original.read_text().replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path('+repr(str(ROOT))+')').replace('artifacts/rank7-prepare8-probe-v1','artifacts/rank7-prepare8-unrolled-probe-v1')
    code=code.replace('[4,8]','[8,16]').replace("v['prepare4']-v['prepare8']","v['prepare8']-v['prepare16']")
    code=code.replace("v['saved']>0 or v['first']==v['end']","v['saved']>0 or v['first']==v['end'] or v['pairs']==0")
    code=code.replace("len(r['replies'])==14","len(r['replies'])==22").replace('all14','all22').replace('all7_fresh','all11_fresh')
    frozen=d/'frozen-auditor.py';assert not frozen.exists();frozen.write_text(code)
    (d/'auditor-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),original,frozen]},indent=2)+'\n')
    exec(compile(code,str(frozen),'exec'),dict(__file__=str(frozen),__name__='__main__'))
if __name__=='__main__':main()
