#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'scripts/report_k2_odd_roots_probe.py';d=ROOT/'artifacts/k2-address-fold-probe-v1'
 s=old.read_text().replace('k2-odd-roots-probe-v1','k2-address-fold-probe-v1').replace('artifacts/s2-k2-probe-v1/check/report.json','artifacts/k2-pair-guard-fold-probe-v1/check/report.json').replace('Only eight rank7 odd-tail WAT bodies differ.','Only eight rank7 address arithmetic WAT bodies differ.')
 p=d/'frozen-reporter.py';p.write_text(s);(d/'entry-reporter-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),old,p]},indent=2)+'\n');exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
