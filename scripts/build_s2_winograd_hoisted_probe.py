#!/usr/bin/env python3
"""Hoist rank49 weight addresses and keep output store addresses on the stack."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'artifacts/s2-winograd-v1';d=ROOT/'artifacts/s2-winograd-hoisted-v1';d.mkdir(exist_ok=False)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 rep=json.loads((old/'build/report.json').read_text())
 for k in ['source_hashes','dependency_hashes']:
  assert all(sha(ROOT/p)==h for p,h in rep[k].items())
 (d/'frozen-plan.py').write_bytes((old/'frozen-plan.py').read_bytes())
 s=(old/'frozen-builder.py').read_text().replace('artifacts/s2-winograd-v1','artifacts/s2-winograd-hoisted-v1')
 anchor='            if first:\n                for k in range(16):';assert s.count(anchor)==1
 s=s.replace(anchor,"            if first:\n                for b in range(16):out += [f'(local.set $bp{b} (i32.add (local.get $wp{b}) (i32.mul (i32.shr_u (local.get $cols) (i32.const 1)) (i32.const {j}))))']\n                for k in range(16):")
 before='(i32.add (local.get $wp{b}) (i32.mul (i32.shr_u (local.get $cols) (i32.const 1)) (i32.const {j})))';assert s.count(before)==2
 # Keep hoisted address initialization; only loads use the hoisted local.
 before='(v128.load8x8_s offset={k*8} '+before+')';assert s.count(before)==1;s=s.replace(before,'(v128.load8x8_s offset={k*8} (local.get $bp{b}))')
 anchor='    for j in range(22):';at=s.index(anchor)
 s=s[:at]+"    lines += [f'(local $bp{b} i32)' for b in range(16)]\n"+s[at:]
 before="['local.get $yp', f'v128.load offset={j*32+half*16}'";assert s.count(before)==1;s=s.replace(before,"['local.get $yp', 'local.get $yp', f'v128.load offset={j*32+half*16}'")
 before="'f32x4.add','local.set $v','local.get $yp','local.get $v',f'v128.store offset={j*32+half*16}'";assert s.count(before)==1;s=s.replace(before,"'f32x4.add',f'v128.store offset={j*32+half*16}'")
 (d/'frozen-builder.py').write_text(s)
 (d/'upstream-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),old/'frozen-builder.py',old/'frozen-plan.py',old/'build/report.json',d/'frozen-builder.py',d/'frozen-plan.py']},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
