#!/usr/bin/env python3
"""Freeze 21 fresh native/current/candidate comparisons with all inference work."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'scripts/check_s2_k2_probe.py';d=ROOT/'artifacts/s2-k2-leaf-outer-probe-v1';s=old.read_text().replace('s2-k2-probe-v1','s2-k2-leaf-outer-probe-v1').replace('rank49/output64','leaf-outer rank49/output96').replace('s1_pair_bounds','current_guard_fold_k2').replace('rank49_k2','rank49_leaf_outer')
 p=d/'frozen-checker.py';assert not p.exists();p.write_text(s);(d/'entry-checker-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),old,p]},indent=2)+'\n');exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
