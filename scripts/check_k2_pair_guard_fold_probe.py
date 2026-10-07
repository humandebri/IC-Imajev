#!/usr/bin/env python3
"""Compare fresh folded method3 with saved exact latest odd-root method3."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'scripts/check_k2_odd_roots_probe.py';d=ROOT/'artifacts/k2-pair-guard-fold-probe-v1'
 s=old.read_text().replace('k2-odd-roots-probe-v1','k2-pair-guard-fold-probe-v1').replace('artifacts/s2-k2-probe-v1/check/report.json','artifacts/k2-odd-roots-probe-v1/check/report.json')
 before='old["measurements"]["s1_pair_bounds"]';assert s.count(before)==1;s=s.replace(before,'old["measurements"]["rank49_k2"]')
 p=d/'frozen-checker.py';assert not p.exists();p.write_text(s);(d/'checker-entry-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),old,p]},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
