#!/usr/bin/env python3
"""Omit rank49 dots whose query leaf is identically zero in padded token tails."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 source=ROOT/'artifacts/s2-k2-alias-kernels-v1/frozen-generator.py';d=ROOT/'artifacts/s2-k2-tail-kernels-v1';d.mkdir(exist_ok=False)
 s=source.read_text().replace('s2-k2-alias-kernels-v1','s2-k2-tail-kernels-v1')
 before="    if j==0:out.append(f'(local.set $qp(i32.add(i32.load offset={m*4}(local.get $q))(local.get $qoff)))')"
 after="    minimum=min(i//4 for i in a.symbols[leaves[m][0]])\n    out.append(f'(if(i32.gt_u(i32.sub(local.get $n)(local.get $t))(i32.const {minimum}))(then')\n"+before
 assert s.count(before)==1;s=s.replace(before,after)
 before="    out.append(f'local.set $pc{m}')"
 after=before+"\n    out.append(f')(else(local.set $pc{m}(v128.const i32x4 0 0 0 0))))')"
 assert s.count(before)==1;s=s.replace(before,after)
 (d/'frozen-generator.py').write_text(s)
 (d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),source,d/'frozen-generator.py']},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-generator.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
