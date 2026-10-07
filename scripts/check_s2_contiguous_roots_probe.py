#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/s2-k2-leaf-outer-probe-v1/frozen-checker.py';d=ROOT/'artifacts/s2-contiguous-roots-probe-v1'
 source=original.read_text().replace('s2-k2-leaf-outer-probe-v1','s2-contiguous-roots-probe-v1').replace('rank49_leaf_outer','rank49_contiguous_roots')
 frozen=d/'frozen-checker.py';assert not frozen.exists();frozen.write_text(source)
 (d/'checker-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
