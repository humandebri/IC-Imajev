#!/usr/bin/env python3
"""Same 21 native and current-control pairs, with bounded fixed preparation."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'scripts/check_s2_k2_probe.py';d=ROOT/'artifacts/s3-k2-prepared-chunked-probe-v1'
 source=original.read_text().replace('s2-k2-probe-v1','s3-k2-prepared-chunked-probe-v1').replace('rank49_k2','rank343_k2').replace('rank49/output64','rank343/output32').replace('s1_pair_bounds','current_odd_roots_k2')
 source=source.replace('seal:()->(Prep);','prepare_coefficients:()->(Prep); seal:()->(Prep);')
 before="    preparations.append(call('seal'))";assert source.count(before)==1
 source=source.replace(before,"    for _ in range(16): preparations.append(call('prepare_coefficients'))\n"+before)
 frozen=d/'frozen-checker.py';assert not frozen.exists();frozen.write_text(source)
 (d/'checker-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
