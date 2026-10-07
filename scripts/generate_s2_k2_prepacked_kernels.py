#!/usr/bin/env python3
"""Rank49 with immutable precomputed I16 coefficients loaded at first dot use."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'scripts/generate_s2_k2_kernels.py';d=ROOT/'artifacts/s2-k2-prepacked-kernels-v1';d.mkdir(exist_ok=False)
 s=old.read_text().replace('s2-k2-kernels-v1','s2-k2-prepacked-kernels-v1').replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
 before=" for i in range(16):lines.append(f'(local $wp{i} i32)(local $bp{i} i32)(local $b{i} v128)')\n for name,_ in b.nodes:lines.append(f'(local ${name} v128)')"
 assert s.count(before)==1;s=s.replace(before," lines.append('(local $wp0 i32)(local $bp0 i32)')")
 before=" for i in range(16):lines.append(f'(local.set $wp{i}(i32.add(i32.load offset={i*4}(local.get $w))(local.get $start)))')"
 assert s.count(before)==1;s=s.replace(before," lines.append('(local.set $wp0(i32.add(i32.load(local.get $w))(i32.mul(i32.shr_u(local.get $start)(i32.const 8))(i32.const 25088))))')")
 lo=s.index('   if first:\n');hi=s.index('   for m in range(49):',lo)
 s=s[:lo]+"   if first:out.append(f'(local.set $bp0(i32.add(local.get $wp0)(i32.mul(i32.shr_u(local.get $cols)(i32.const 8))(i32.const {j*25088}))))')\n"+s[hi:]
 before="     out+=[f'local.get $w{m}_{j}_{k}','i32x4.dot_i16x8_s']"
 after="     out.append(f'(local.tee $w{m}_{j}_{k}(v128.load offset={m*512+k*16}(local.get $bp0)))'if first else f'local.get $w{m}_{j}_{k}');out.append('i32x4.dot_i16x8_s')"
 assert s.count(before)==1;s=s.replace(before,after)
 (d/'frozen-generator.py').write_text(s);files=[Path(__file__),old,d/'frozen-generator.py'];(d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-generator.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
