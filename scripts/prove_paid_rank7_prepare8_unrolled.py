#!/usr/bin/env python3
"""Run the full unchanged proof, additionally comparing latest validated best."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    original=ROOT/'scripts/prove_paid_rank7_prepare8.py';old=ROOT/'artifacts/paid-rank7-prepare8-v1'
    manifest=json.loads((old/'proof-entry-hashes.json').read_text());assert sha(original)==manifest[str(original.relative_to(ROOT))]
    code=original.read_text().replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path('+repr(str(ROOT))+')',1)
    code=code.replace('artifacts/paid-rank7-prepare8-v1','artifacts/paid-rank7-prepare8-unrolled-v1')
    code=code.replace('artifacts/paid-common-raw-hybrid-retry-v3/proof/report.json','artifacts/paid-rank7-prepare8-v1/proof/report.json')
    code=code.replace('/paid-rank7-prepare8-v1/proof\'','/paid-rank7-prepare8-unrolled-v1/proof\'')
    d=ROOT/'artifacts/paid-rank7-prepare8-unrolled-v1';frozen=d/'frozen-proof-driver.py';assert not frozen.exists();frozen.write_text(code)
    (d/'proof-driver-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),original,frozen,old/'proof-entry-hashes.json']},indent=2)+'\n')
    exec(compile(code,str(frozen),'exec'),dict(__file__=str(frozen),__name__='__main__'))
if __name__=='__main__':main()
