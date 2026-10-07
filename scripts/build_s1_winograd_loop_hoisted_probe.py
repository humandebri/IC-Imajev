#!/usr/bin/env python3
"""Hoist the two output row pointers and second-token guard out of the quartet loop."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]


def main():
    old=ROOT/'artifacts/s1-winograd-fused-v1'
    d=ROOT/'artifacts/s1-winograd-loop-hoisted-v1';d.mkdir(exist_ok=False)
    p=old/'frozen-builder.py';source=p.read_text().replace('artifacts/s1-winograd-fused-v1','artifacts/s1-winograd-loop-hoisted-v1')
    before="(local $t i32)(local $qp i32)(local $yp i32)(local $qo i32)"
    assert source.count(before)==1
    source=source.replace(before,before+'(local $yp1 i32)(local $keep i32)')
    lo=source.index('  for ti in range(2):\n   load=')
    hi=source.index('\n  for j in range(TILE//4):',lo)
    source=source[:lo]+'''  out+=['(local.set $keep(i32.lt_u(i32.add(local.get $t)(i32.const 1))(local.get $n)))',
        f'(local.set $yp(i32.add(local.get $sums)(i32.mul(local.get $t)(i32.const {TILE*4}))))',
        f'(local.set $yp1(i32.add(local.get $yp)(i32.const {TILE*4})))']
  for ti in range(2):
   load=f'(local.set $sx{ti}(v128.load32_splat(i32.add(local.get $sx)(i32.shl(i32.mul(i32.add(local.get $t)(i32.const {ti}))(local.get $stride))(i32.const 2)))))'
   assert load.count('(')==load.count(')')
   out.append(f'(if(local.get $keep)(then {load}))'if ti else load)
'''+source[hi:]
    before="    out += [f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti}))(local.get $n))(then',f'(local.set $yp(i32.add(local.get $sums)(i32.mul(i32.add(local.get $t)(i32.const {ti}))(i32.const {TILE*4}))))']"
    assert source.count(before)==1
    source=source.replace(before,"    if ti:out.append('(if(local.get $keep)(then')\n    address='local.get $yp1'if ti else 'local.get $yp'")
    before="    out+=['local.get $yp',f'v128.load offset={j*16}']";assert source.count(before)==1
    source=source.replace(before,"    out+=[address,f'v128.load offset={j*16}']")
    before="'f32x4.add',f'local.set $pc0','local.get $yp','local.get $pc0',f'v128.store offset={j*16}', '))']"
    assert source.count(before)==1
    source=source.replace(before,"'f32x4.add',f'local.set $pc0',address,'local.get $pc0',f'v128.store offset={j*16}']\n    if ti:out.append('))')")
    (d/'frozen-builder.py').write_text(source)
    files=[Path(__file__),p,old/'source-audit.json',old/'build/report.json',ROOT/'scripts/build_s1_winograd_fused_probe.py',d/'frozen-builder.py']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (d/'entry-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):sha(f)for f in files},indent=2)+'\n')
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
