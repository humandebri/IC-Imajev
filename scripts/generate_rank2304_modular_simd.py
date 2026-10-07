#!/usr/bin/env python3
"""Universal I32 ring kernel and checked-I16 kernel sharing author P SLP."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def post_helper(path):
    statements=[]
    names=[f'i{i}' for i in range(48)]
    def emit(node):
        if isinstance(node,ast.Name):
            assert node.id in names
            return [f'local.get ${node.id}']
        assert isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub))
        return emit(node.left)+emit(node.right)+['i32x4.add' if isinstance(node.op,ast.Add) else 'i32x4.sub']
    for line in path.read_text().splitlines():
        if ':=' not in line:
            continue
        name,expr=line.split(';')[0].split(':=')
        name=name.strip()
        body=emit(ast.parse(expr.strip(),mode='eval').body)
        assert name not in names
        names.append(name)
        statements.extend(body+[f'local.set ${name}'])
    lines=['(func $post(param $p i32)(param $read_stride i32)(param $out i32)(param $write_stride i32)']
    lines += [f'(local ${n} v128)' for n in names]
    for i in range(48):
        lines.append(f'(local.set $i{i}(v128.load(i32.add(local.get $p)(i32.mul(local.get $read_stride)(i32.const {i})))))')
    lines+=statements
    for i in range(16):
        lines.append(f'(v128.store(i32.add(local.get $out)(i32.mul(local.get $write_stride)(i32.const {i})))(local.get $o{i}))')
    return '\n'.join(lines+[')'])


def kernel(checked):
    name='checked16' if checked else 'universal32'
    lines=['(func(export "'+name+'")'+''.join(f'(param ${p} i32)' for p in ['a','b','p','inner','outer','out','rawq','raww','flag']),
           '(local $i i32)(local $j i32)(local $k i32)(local $t i32)(local $n i32)(local $dot i32)(local $safe i32)(local $v v128)',
           '(block $done']
    if checked:
        lines+=['(local.set $safe(i32.const 1))','(local.set $i(i32.const 0))','(block $unsafe(loop $scan']
        for pointer in ['a','b']:
            lines += [f'(local.set $v(v128.load(i32.add(local.get ${pointer})(local.get $i))))',
                      '(if(v128.any_true(v128.or(i32x4.lt_s(local.get $v)(v128.const i32x4 -32768 -32768 -32768 -32768))(i32x4.gt_s(local.get $v)(v128.const i32x4 32767 32767 32767 32767))))(then(local.set $safe(i32.const 0))(br $unsafe)))']
        lines+=['(local.set $i(i32.add(local.get $i)(i32.const 16)))','(br_if $scan(i32.lt_u(local.get $i)(i32.const 147456)))','))',
                '(i32.store(local.get $flag)(local.get $safe))','(if(i32.eqz(local.get $safe))(then',
                '(local.set $t(i32.const 0))(loop $tokens(local.set $n(i32.const 0))(loop $rows(local.set $dot(i32.const 0))(local.set $k(i32.const 0))(loop $K',
                '(local.set $dot(i32.add(local.get $dot)(i32.mul(i32.load16_s(i32.add(local.get $rawq)(i32.add(i32.mul(local.get $t)(i32.const 512))(i32.shl(local.get $k)(i32.const 1)))))(i32.load16_s(i32.add(local.get $raww)(i32.add(i32.mul(local.get $k)(i32.const 32))(i32.shl(local.get $n)(i32.const 1))))))))',
                '(local.set $k(i32.add(local.get $k)(i32.const 1)))(br_if $K(i32.lt_u(local.get $k)(i32.const 256))))',
                '(i32.store(i32.add(local.get $out)(i32.shl(i32.add(i32.mul(local.get $t)(i32.const 16))(local.get $n))(i32.const 2)))(local.get $dot))',
                '(local.set $n(i32.add(local.get $n)(i32.const 1)))(br_if $rows(i32.lt_u(local.get $n)(i32.const 16))))',
                '(local.set $t(i32.add(local.get $t)(i32.const 1)))(br_if $tokens(i32.lt_u(local.get $t)(i32.const 16))))','(br $done)))']
    else:
        lines.append('(i32.store(local.get $flag)(i32.const 2))')
    lines+=['(local.set $i(i32.const 0))','(loop $products','local.get $p','local.get $i','i32.add']
    def load(pointer,offset):
        return f'(v128.load offset={offset}(i32.add(local.get ${pointer})(i32.shl(local.get $i)(i32.const 2))))'
    if checked:
        for segment in range(2):
            for pointer in ['a','b']:
                lines += [load(pointer,segment*32),load(pointer,segment*32+16),'i16x8.narrow_i32x4_s']
            lines.append('i32x4.dot_i16x8_s')
            if segment:
                lines.append('i32x4.add')
    else:
        for segment in range(4):
            lines += [load('a',segment*16),load('b',segment*16),'i32x4.mul']
            if segment:
                lines.append('i32x4.add')
    lines+=['v128.store','(local.set $i(i32.add(local.get $i)(i32.const 16)))','(br_if $products(i32.lt_u(local.get $i)(i32.const 36864)))',')',
            '(local.set $i(i32.const 0))(loop $inner_loop',
            '(call $post(i32.add(local.get $p)(i32.mul(local.get $i)(i32.const 768)))(i32.const 16)(i32.add(local.get $inner)(i32.mul(local.get $i)(i32.const 256)))(i32.const 16))',
            '(local.set $i(i32.add(local.get $i)(i32.const 1)))(br_if $inner_loop(i32.lt_u(local.get $i)(i32.const 48))))',
            '(local.set $i(i32.const 0))(loop $outer_loop',
            '(call $post(i32.add(local.get $inner)(i32.shl(local.get $i)(i32.const 4)))(i32.const 256)(i32.add(local.get $outer)(i32.shl(local.get $i)(i32.const 4)))(i32.const 256))',
            '(local.set $i(i32.add(local.get $i)(i32.const 1)))(br_if $outer_loop(i32.lt_u(local.get $i)(i32.const 16))))',
            '(local.set $i(i32.const 0))(loop $final_outer(local.set $j(i32.const 0))(loop $final_inner',
            '(local.set $v(v128.load(i32.add(local.get $outer)(i32.shl(i32.add(i32.mul(local.get $i)(i32.const 16))(local.get $j))(i32.const 4)))))',
            '(local.set $dot(i32.add(i32.add(i32x4.extract_lane 0(local.get $v))(i32x4.extract_lane 1(local.get $v)))(i32.add(i32x4.extract_lane 2(local.get $v))(i32x4.extract_lane 3(local.get $v)))))',
            '(i32.store(i32.add(local.get $out)(i32.shl(i32.add(i32.mul(i32.add(i32.and(local.get $i)(i32.const 12))(i32.shr_u(local.get $j)(i32.const 2)))(i32.const 16))(i32.add(i32.shl(i32.and(local.get $i)(i32.const 3))(i32.const 2))(i32.and(local.get $j)(i32.const 3))))(i32.const 2)))(i32.shr_s(local.get $dot)(i32.const 4)))',
            '(local.set $j(i32.add(local.get $j)(i32.const 1)))(br_if $final_inner(i32.lt_u(local.get $j)(i32.const 16))))',
            '(local.set $i(i32.add(local.get $i)(i32.const 1)))(br_if $final_outer(i32.lt_u(local.get $i)(i32.const 16))))',')',')']
    return '\n'.join(lines)


def main():
    base=ROOT/'artifacts/rank2304-scaled-slp-basis-v1'
    evidence=json.loads((base/'report.json').read_text())
    for p,h in evidence['source_hashes'].items():
        assert sha(ROOT/p)==h,p
    slp=ROOT/'artifacts/rank48-latest-programs-v1/4x4x4_48_204_P.slp'
    text='(module(import "env" "memory" (memory 1))\n'+post_helper(slp)+'\n'+kernel(False)+'\n'+kernel(True)+'\n)\n'
    assert text.count('(')==text.count(')')
    d=ROOT/'artifacts/rank2304-modular-simd-v1'
    d.mkdir(exist_ok=False)
    wat=d/'kernel.wat'
    wasm=d/'kernel.wasm'
    wat.write_text(text)
    subprocess.run([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-audit'),'wat',str(wat),str(wasm)],check=True)
    files=[Path(__file__),base/'report.json',slp,wat,wasm]
    (d/'report.json').write_text(json.dumps(dict(complete=True,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},wasm_execution_verified=False,ic_performance_verified=False,scope='Universal modular I32 SIMD and checked-I16 scalar fallback prototypes, prepared operands only. Input-dependent preparation is not measured or omitted from the full goal.'),indent=2)+'\n')
    print('compiled universal32 and checked16 SIMD kernels')


if __name__=='__main__':
    main()
