#!/usr/bin/env python3
"""Add a hoisted, unrolled direct SIMD control to the same core benchmark."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    prior=ROOT/'artifacts/rank2304-modular-meter-v1'
    b=json.loads((prior/'build.json').read_text())
    original=ROOT/'scripts/build_rank2304_modular_meter.py'
    assert sha(original)==b['source_hashes'][str(original.relative_to(ROOT))]
    source=original.read_text().replace('rank2304-modular-meter-v1','rank2304-modular-meter-v2')
    dot=[]
    for k in range(32):
        product=f'(i32x4.dot_i16x8_s(v128.load offset={k*16}(local.get $qp))(v128.load offset={k*16}(local.get $wp)))'
        dot.append(f'(local.set $v {product})' if k==0 else f'(local.set $v(i32x4.add(local.get $v){product}))')
    direct='''(func $direct_unrolled(param $q i32)(param $w i32)(param $out i32)
 (local $t i32)(local $n i32)(local $qp i32)(local $wp i32)(local $v v128)
 (loop $T(local.set $qp(i32.add(local.get $q)(i32.mul(local.get $t)(i32.const 512))))
 (local.set $n(i32.const 0))(loop $N(local.set $wp(i32.add(local.get $w)(i32.mul(local.get $n)(i32.const 512))))
 '''+'\n'.join(dot)+'''
 (i32.store(i32.add(local.get $out)(i32.shl(i32.add(i32.mul(local.get $t)(i32.const 16))(local.get $n))(i32.const 2)))
 (i32.add(i32.add(i32x4.extract_lane 0(local.get $v))(i32x4.extract_lane 1(local.get $v)))(i32.add(i32x4.extract_lane 2(local.get $v))(i32x4.extract_lane 3(local.get $v)))))
 (local.set $n(i32.add(local.get $n)(i32.const 1)))(br_if $N(i32.lt_u(local.get $n)(i32.const 16))))
 (local.set $t(i32.add(local.get $t)(i32.const 1)))(br_if $T(i32.lt_u(local.get $t)(i32.const 16)))))'''
    before='body=[direct,f\'(data'
    assert source.count(before)==1
    source=source.replace(before,'body=[direct,'+repr(direct)+',f\'(data')
    source=source.replace("['direct','universal32','checked16']","['direct','direct_unrolled','universal32','checked16']")
    source=source.replace("if mode=='direct':","if mode in ['direct','direct_unrolled']:")
    source=source.replace("call=f'(call $direct(i32.const", "call=f'(call ${mode}(i32.const")
    source=source.replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path('+repr(str(ROOT))+')')
    source=source.replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
    d=ROOT/'artifacts/rank2304-modular-meter-v2';d.mkdir(exist_ok=False)
    frozen=d/'frozen-builder.py';frozen.write_text(source)
    (d/'entry.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),original,frozen]},indent=2)+'\n')
    exec(compile(source,str(frozen),'exec'),dict(__file__=str(frozen),__name__='__main__'))
if __name__=='__main__':main()
