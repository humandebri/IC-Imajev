#!/usr/bin/env python3
"""Measure the three actual paid inputs with full saved-reference and boundary checks."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]


def main():
    d=ROOT/'artifacts/paid-gated-norm-finalize-v1'
    guard=json.loads((d/'upgrade-guards/verified.json').read_text())
    audit=json.loads((d/'source-audit.json').read_text())
    assert audit['complete'] and guard['complete'] and guard['baseline_restored'] and guard['module']==audit['module']
    p=ROOT/'artifacts/paid-owned-profile-off-recovery-v1/frozen-proof.py'
    source=p.read_text().replace('artifacts/paid-owned-profile-off-recovery-v1','artifacts/paid-gated-norm-finalize-v1')
    (d/'frozen-proof.py').write_text(source)
    files=[Path(__file__),p,d/'source-audit.json',d/'upgrade-guards/verified.json',d/'frozen-proof.py']
    (d/'proof-entry-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in files},indent=2)+'\n')
    exec(compile(source,str(d/'frozen-proof.py'),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
