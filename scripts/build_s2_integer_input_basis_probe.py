#!/usr/bin/env python3
"""Change only input preparation DAG, retain audited dot-reuse kernels/control."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 original=ROOT/'artifacts/s2-cached-tail-dot-reuse-probe-v1/frozen-builder.py'
 d=ROOT/'artifacts/s2-integer-input-basis-probe-v1';d.mkdir(exist_ok=False)
 audit=ROOT/'artifacts/rank49-integer-input-basis-v1/audit.json';r=json.loads(audit.read_text())
 assert r['complete'] and r['all49_leaf_coefficients_exact']
 assert all(sha(ROOT/p)==h for p,h in r['source_hashes'].items())
 s=original.read_text().replace('s2-cached-tail-dot-reuse-probe-v1','s2-integer-input-basis-probe-v1')
 before=" ad,bd,cd,leaves,roots,_=ns['plan']()"
 after=""" old_ad,bd,cd,old_leaves,roots,_=ns['plan']()
 p=ROOT/'scripts/plan_rank49_integer_input_basis.py';alt={'__name__':'alt','__file__':str(p)};exec(compile(p.read_text(),str(p),'exec'),alt)
 ad,alt_bd,alt_cd,leaves,alt_roots,_=alt['plan']()
 assert len(ad.nodes)==41 and len(old_ad.nodes)==44
 assert all(ad.symbols[an]==old_ad.symbols[oan]and bn==obn for(an,bn),(oan,obn)in zip(leaves,old_leaves))
 assert alt_roots==roots and alt_cd.nodes==cd.nodes
"""
 assert s.count(before)==1;s=s.replace(before,after)
 before="files=[Path(__file__),kernels/'execution-report.json'"
 after="files=[Path(__file__),ROOT/'scripts/plan_rank49_integer_input_basis.py',ROOT/'scripts/plan_rank343_integer_basis.py',ROOT/'artifacts/s3-winograd-v1/frozen-generator.py',ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1/plan.py',ROOT/'artifacts/rank49-integer-input-basis-v1/audit.json',kernels/'execution-report.json'"
 assert s.count(before)==1;s=s.replace(before,after)
 frozen=d/'frozen-builder.py';frozen.write_text(s)
 (d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),original,frozen,audit]},indent=2)+'\n')
 exec(compile(s,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
