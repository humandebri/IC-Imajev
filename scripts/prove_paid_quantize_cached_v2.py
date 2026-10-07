#!/usr/bin/env python3
"""Run proven full paid API workflow on the cached quantization candidate."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/paid-quantize-cached-v2';assert all(sha(ROOT/p)==h for p,h in json.loads((d/'workflow-hashes.json').read_text()).items());g=json.loads((d/'upgrade-guards/verified.json').read_text());b=json.loads((d/'build/report.json').read_text());assert g['complete'] and g['baseline_restored'] and g['module']==b['wasm_sha256']
 old=ROOT/'artifacts/paid-swiglu-cached-simd-v2';r=json.loads((old/'summary.json').read_text());assert r['complete'] and r['baseline_restored'] and r['upgrade_checks_verified']
 p=old/'frozen-proof.py';s=p.read_text().replace('artifacts/paid-swiglu-cached-simd-v2','artifacts/paid-quantize-cached-v2');(d/'frozen-proof.py').write_text(s)
 files=[Path(__file__),p,old/'summary.json',d/'workflow-hashes.json',d/'guard-wrapper-hashes.json',d/'upgrade-guards/verified.json',d/'upgrade-guard-entry-hashes.json',d/'frozen-proof.py'];(d/'proof-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-proof.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
