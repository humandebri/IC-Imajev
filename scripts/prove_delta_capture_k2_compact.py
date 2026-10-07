#!/usr/bin/env python3
"""Snapshot restored latest arithmetic capture on the new owned local target."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    d=ROOT/'artifacts/delta-capture-k2-compact-v1'
    audit=json.loads((d/'source-audit.json').read_text())
    assert audit['complete'] and audit['all30_kernel_sources_equal'] and audit['runtime_changes_only_capture_hooks']
    ready=json.loads((ROOT/'artifacts/local-goal-recovery-v1/baseline-ready.json').read_text())
    assert ready['complete'] and ready['target']=='4caro-hl777-77775-aaaba-cai'
    upstream=ROOT/'scripts/prove_delta_capture.py'
    source=upstream.read_text().replace('artifacts/delta-capture-v2','artifacts/delta-capture-k2-compact-v1').replace("TARGET='6eydd-o3777-77775-aaama-cai'","TARGET='4caro-hl777-77775-aaaba-cai'")
    (d/'frozen-proof.py').write_text(source)
    paths=[Path(__file__),upstream,d/'source-audit.json',d/'frozen-proof.py',ROOT/'artifacts/local-goal-recovery-v1/baseline-ready.json']
    (d/'proof-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    exec(compile(source,str(d/'frozen-proof.py'),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':
    main()
