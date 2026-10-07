#!/usr/bin/env python3
"""Change only eight current-control query-load WAT bodies in the same diagnostic."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'scripts/build_k2_odd_roots_probe.py';d=ROOT/'artifacts/k2-query-quad-load-probe-v1';d.mkdir(exist_ok=False)
 s=old.read_text().replace('k2-odd-roots-probe-v1','k2-query-quad-load-probe-v1').replace('k2-odd-roots-kernels-v1','k2-query-quad-load-kernels-v1').replace("old=ROOT/'artifacts/s2-k2-probe-v1'","old=ROOT/'artifacts/k2-pair-guard-fold-probe-v1'").replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
 p=d/'frozen-builder.py';p.write_text(s);(d/'entry-builder-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),old,p]},indent=2)+'\n');exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
