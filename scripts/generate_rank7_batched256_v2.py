#!/usr/bin/env python3
"""Two-stage rank7 tile256, with four leaf products staged in caller scratch."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def kernel(plan, seed):
    _, b, c, leaves, roots, _ = plan['plan']()
    rec, regs, capacity = plan['reconstruct'](c, roots)
    assert len(leaves) == 7
    tile, groups = 256, 32
    name = '__imajev_rank7_batched256' + ('_seed' if seed else '')
    lines = ['(module(func(export "' + name + '")' + ''.join(
        f'(param ${p} i32)' for p in ['q', 'w', 'cols', 'start', 'sx', 'stride', 'sw', 'out', 'n', 'scratch'])]
    lines += ['(local $t i32)(local $qo i32)(local $qp i32)(local $sp i32)(local $yp i32)(local $keep i32)',
              '(local $input_stride i32)(local $output_bytes i32)',
              '(local $sx0 v128)(local $sx1 v128)']
    for i in range(4):
        lines.append(f'(local $bp{i} i32)(local $wp{i} i32)(local $b{i} v128)')
    for i in range(capacity):
        lines.append(f'(local $pc{i} v128)')
    for k in range(64):
        lines.append(f'(local $x{k} v128)')
        for m in [4,5,6]:lines.append(f'(local $y{m}_{k} v128)')
    # Four reusable coefficient banks: stage1 holds four leaves; stage2 holds
    # three. Every bank is overwritten before consumption in the next stage.
    for m in range(4):
        for j in range(groups):
            for k in range(64):
                lines.append(f'(local $w{m}_{j}_{k} v128)')
    lines += ['(local.set $input_stride(i32.shr_u(local.get $cols)(i32.const 8)))',
              '(local.set $output_bytes(i32.shl(local.get $stride)(i32.const 2)))',
              '(block $empty(br_if $empty(i32.eqz(local.get $n)))']
    for i in range(4):
        lines.append(f'(local.set $wp{i}(i32.add(i32.load offset={i*4}(local.get $w))(i32.shl(local.get $start)(i32.const 1))))')

    def coefficients(ms):
        out = []
        for j in range(groups):
            for i in range(4):
                out.append(f'(local.set $bp{i}(i32.add(local.get $wp{i})(i32.mul(local.get $cols)(i32.const {j*2}))))')
            for k in range(64):
                for i in range(4):
                    out.append(f'(local.set $b{i}(v128.load8x8_s offset={k*8}(local.get $bp{i})))')
                for slot, m in enumerate(ms):
                    terms = b.symbols[leaves[m][1]]
                    out += plan['expression']({f'b{i}': v for i, v in terms.items()},
                                              lambda n: f'local.get ${n}', 'i16x8')
                    out.append(f'local.set $w{slot}_{j}_{k}')
        return out

    def dot(m, slot):
        out = [f'(local.set $qp(i32.add(i32.load offset={m*4}(local.get $q))(local.get $qo)))']
        for j in range(groups):
            for k in range(64):
                out.append(f'(local.tee $x{k}(v128.load32_splat offset={k*4}(local.get $qp)))'
                           if j == 0 else f'local.get $x{k}')
                out += [f'local.get $w{slot}_{j}_{k}', 'i32x4.dot_i16x8_s']
                if k:
                    out.append('i32x4.add')
            out.append(f'local.set $pc{m}')
            offset = (m * groups + j) * 16
            out.append(f'(v128.store offset={offset}(local.get $sp)(local.get $pc{m}))')
        return out

    setup = ['(local.set $qo(i32.add(i32.mul(i32.shr_u(local.get $t)(i32.const 1))(local.get $cols))(local.get $start)))',
             '(local.set $sp(i32.add(local.get $scratch)(i32.mul(i32.shr_u(local.get $t)(i32.const 1))(i32.const 2048))))',
             '(local.set $keep(i32.lt_u(i32.add(local.get $t)(i32.const 1))(local.get $n)))']
    # Only stage1 products occupy scratch; stage2 computes and consumes
    # its products directly before canonical reconstruction.
    for stage, ms in enumerate([[0, 1, 2, 3], [4, 5, 6]]):
        lines += coefficients(ms) + ['(local.set $t(i32.const 0))', f'(loop $stage{stage}'] + setup
        for slot, m in enumerate(ms if stage==0 else []):
            if m in [3, 4, 6]:
                lines += ['(if(local.get $keep)(then'] + dot(m, slot) + ['))']
            else:
                lines += dot(m, slot)
        if stage == 1:
            lines += ['(local.set $sx0(v128.load32_splat(i32.add(local.get $sx)(i32.shl(i32.mul(local.get $t)(local.get $input_stride))(i32.const 2)))))',
                      '(if(local.get $keep)(then(local.set $sx1(v128.load32_splat(i32.add(local.get $sx)(i32.shl(i32.mul(i32.add(local.get $t)(i32.const 1))(local.get $input_stride))(i32.const 2)))))))']
            for j in range(groups):
                for slot,m in enumerate([4,5,6]):
                    if m in [4,6]:lines.append('(if(local.get $keep)(then')
                    if j==0:lines.append(f'(local.set $qp(i32.add(i32.load offset={m*4}(local.get $q))(local.get $qo)))')
                    for k in range(64):
                        lines.append(f'(local.tee $y{m}_{k}(v128.load32_splat offset={k*4}(local.get $qp)))'if j==0 else f'local.get $y{m}_{k}')
                        lines += [f'local.get $w{slot}_{j}_{k}','i32x4.dot_i16x8_s']
                        if k:lines.append('i32x4.add')
                    lines.append(f'local.set $pc{m}')
                    if m in [4,6]:lines.append(f')(else(local.set $pc{m}(v128.const i32x4 0 0 0 0))))')
                for m in range(4):
                    load=f'(v128.load offset={(m*groups+j)*16}(local.get $sp))'
                    if m==3:load='(if(result v128)(local.get $keep)(then'+load+')(else(v128.const i32x4 0 0 0 0)))'
                    lines.append(f'(local.set $pc{m}{load})')
                lines += rec
                for ti in range(2):
                    if ti:
                        lines.append('(if(local.get $keep)(then')
                    lines.append(f'(local.set $yp(i32.add(local.get $out)(i32.mul(i32.add(local.get $t)(i32.const {ti}))(local.get $output_bytes))))')
                    for part in range(2):
                        offset = j*32 + part*16
                        lines += ['local.get $yp']
                        lines += ['v128.const i32x4 0 0 0 0'] if seed else ['local.get $yp', f'v128.load offset={offset}']
                        lines += [f'local.get $pc{regs[ti][part]}', 'f32x4.convert_i32x4_s',
                                  f'local.get $sx{ti}', 'f32x4.mul',
                                  f'(v128.load offset={offset}(local.get $sw))', 'f32x4.mul',
                                  'f32x4.add', f'v128.store offset={offset}']
                    if ti:
                        lines.append('))')
        lines += ['(local.set $t(i32.add(local.get $t)(i32.const 2)))',
                  f'(br_if $stage{stage}(i32.lt_u(local.get $t)(local.get $n)))', ')']
    lines += [')', '))']
    text = '\n'.join(lines) + '\n'
    assert text.count('(') == text.count(')')
    assert text.count('(local $') < 10000
    return text, name


def main():
    p = ROOT/'artifacts/s1-winograd-v1/plan.py'
    plan = dict(__name__='plan', __file__=str(p))
    exec(compile(p.read_text(), str(p), 'exec'), plan)
    d = ROOT/'artifacts/rank7-batched256-v2'
    d.mkdir(exist_ok=False)
    files = [Path(__file__), p]
    kernels = []
    for seed in [False, True]:
        text, name = kernel(plan, seed)
        wat = d/('seed.wat' if seed else 'normal.wat')
        wat.write_text(text)
        imported = wat.with_name(wat.stem+'-imported.wat')
        imported.write_text(text.replace('(module(func', '(module(import "env" "memory" (memory 1))(func', 1))
        wasm = imported.with_suffix('.wasm')
        subprocess.run([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-audit'), 'wat', str(imported), str(wasm)], check=True)
        files += [wat, imported, wasm]
        kernels.append(dict(seed=seed, symbol=name, locals=text.count('(local $'),
                            wat=str(wat.relative_to(ROOT)), wasm=str(wasm.relative_to(ROOT))))
    report = dict(complete=True, tile=256, stages=[[0,1,2,3],[4,5,6]],
                  scratch_bytes_per_token_pair=2048, kernels=kernels,
                  source_hashes={str(x.relative_to(ROOT)):sha(x) for x in files},
                  execution_verified=False, performance_verified=False,
                  scope='Prototype only; caller-owned input-dependent scratch counts toward inference.')
    (d/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(complete=True, kernels=kernels)))


if __name__ == '__main__':
    main()
