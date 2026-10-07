#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'scripts/validate_s2_k2_kernels.py';d=ROOT/'artifacts/s2-contiguous-roots-kernels-v1'
 source=original.read_text().replace('s2-k2-kernels-v1','s2-contiguous-roots-kernels-v1');frozen=d/'frozen-validator.py';assert not frozen.exists();frozen.write_text(source)
 (d/'validator-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
