#!/usr/bin/env python3
"""Skip zero second-row products after cache initialization; stack output stores."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'artifacts/s1-winograd-loop-hoisted-v1'
 d=ROOT/'artifacts/s1-winograd-tail-v1';d.mkdir(exist_ok=False)
 p=old/'frozen-builder.py';s=p.read_text().replace('s1-winograd-loop-hoisted-v1','s1-winograd-tail-v1')
 before="   for m in range(7):\n    if j==0:"
 assert s.count(before)==1
 s=s.replace(before,"   for m in range(7):\n    skip=not first and m in (3,4,6)\n    if skip:out.append('(if(local.get $keep)(then')\n    if j==0:")
 before="    out+=[f'local.set $pc{m}']\n   out+=rec"
 assert s.count(before)==1
 s=s.replace(before,"    out+=[f'local.set $pc{m}']\n    if skip:out.append(f')(else(local.set $pc{m}(v128.const i32x4 0 0 0 0))))')\n   out+=rec")
 before="    out+=[address,f'v128.load offset={j*16}']"
 assert s.count(before)==1;s=s.replace(before,"    out+=[address,address,f'v128.load offset={j*16}']")
 before="'f32x4.add',f'local.set $pc0',address,'local.get $pc0',f'v128.store offset={j*16}'"
 assert s.count(before)==1;s=s.replace(before,"'f32x4.add',f'v128.store offset={j*16}'")
 lo=s.index(' # pc0 may be an output root.')
 hi=s.index(" lines+=r1+",lo)
 s=s[:lo]+" lines += ['(block $done','(br_if $done(i32.eqz(local.get $n)))']\n r1=row(True);r2=row(False)\n"+s[hi:]
 (d/'frozen-builder.py').write_text(s)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 files=[Path(__file__),p,old/'source-audit.json',old/'build/report.json',d/'frozen-builder.py']
 (d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
