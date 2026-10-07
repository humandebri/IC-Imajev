#!/usr/bin/env python3
"""Build the full candidate; fidelity still requires paid and Dense proofs."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'scripts/build_update_k2_odd_roots.py';d=ROOT/'artifacts/update-k2-pair-guard-fold-v1';d.mkdir(exist_ok=False)
 s=old.read_text()
 before="old=ROOT/'artifacts/update-k2-compact-v1';d=ROOT/'artifacts/update-k2-odd-roots-v1';k=ROOT/'artifacts/k2-odd-roots-kernels-v1'"
 after="old=ROOT/'artifacts/update-k2-odd-roots-v1';d=ROOT/'artifacts/update-k2-pair-guard-fold-v1';k=ROOT/'artifacts/k2-pair-guard-fold-kernels-v1'"
 assert s.count(before)==1;s=s.replace(before,after).replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
 s=s.replace(".replace('artifacts/update-k2-compact-v1','artifacts/update-k2-odd-roots-v1')",".replace('artifacts/update-k2-odd-roots-v1','artifacts/update-k2-pair-guard-fold-v1')")
 p=d/'entry-builder.py';p.write_text(s);(d/'entry-builder-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),old,p]},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
