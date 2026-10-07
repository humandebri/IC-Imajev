#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/s2-common-raw-mlp-probe-v1/frozen-builder.py';d=ROOT/'artifacts/s2-cached-tail-probe-v1';d.mkdir(exist_ok=False)
 s=original.read_text().replace('s2-common-raw-mlp-probe-v1','s2-cached-tail-probe-v1').replace('s2-common-raw-kernels-v1','s2-cached-tail-kernels-v1')
 frozen=d/'frozen-builder.py';frozen.write_text(s);(d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(s,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
