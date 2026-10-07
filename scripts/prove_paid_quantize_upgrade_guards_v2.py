#!/usr/bin/env python3
"""Deterministic upgrade/refund guard fixtures on the paid cached quantization candidate."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=ROOT/'scripts/prove_paid_rms_upgrade_guards.py';s=p.read_text().replace('artifacts/paid-rms-ordered-simd-v1','artifacts/paid-quantize-cached-v2');d=ROOT/'artifacts/paid-quantize-cached-v2';(d/'guard-wrapper.py').write_text(s)
 (d/'guard-wrapper-hashes.json').write_text(json.dumps({str(v.relative_to(ROOT)):sha(v)for v in [Path(__file__),p,d/'guard-wrapper.py']},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
