#!/usr/bin/env python3
"""Dispatch once per row pair to the four-product odd-tail body."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'artifacts/s1-winograd-tail-v1';d=ROOT/'artifacts/s1-winograd-tail-dispatch-v1';d.mkdir(exist_ok=False)
 p=old/'frozen-builder.py';s=p.read_text().replace('s1-winograd-tail-v1','s1-winograd-tail-dispatch-v1')
 s=s.replace(' def row(first):',' def row(first,tail=False):')
 before="    skip=not first and m in (3,4,6)\n    if skip:out.append('(if(local.get $keep)(then')"
 after="    if tail and m in (3,4,6):\n     out.append(f'(local.set $pc{m}(v128.const i32x4 0 0 0 0))');continue"
 assert s.count(before)==1;s=s.replace(before,after)
 before="    if skip:out.append(f')(else(local.set $pc{m}(v128.const i32x4 0 0 0 0))))')"
 assert s.count(before)==1;s=s.replace(before,'')
 s=s.replace(' r1=row(True);r2=row(False)',' r1=row(True);r2=row(False);rt=row(False,True)')
 before="']+r2+['(local.set $t"
 after="']+['(if(i32.lt_u(i32.add(local.get $t)(i32.const 1))(local.get $n))(then']+r2+[')(else']+rt+['))','(local.set $t"
 assert s.count(before)==1;s=s.replace(before,after)
 (d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),p,old/'source-audit.json',d/'frozen-builder.py']
 (d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
