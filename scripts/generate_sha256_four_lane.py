#!/usr/bin/env python3
"""Generate a four-independent-message SIMD SHA256 compression prototype."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def get(x):
    return f'(local.get ${x})'


def const(n):
    return '(v128.const i32x4 ' + ' '.join([str(n)] * 4) + ')'


def op(name, *args):
    return '(' + name + ' ' + ' '.join(args) + ')'


def xor(*args):
    out = args[0]
    for value in args[1:]:
        out = op('v128.xor', out, value)
    return out


def add(*args):
    out = args[0]
    for value in args[1:]:
        out = op('i32x4.add', out, value)
    return out


def rotr(value, n):
    return op('v128.or', op('i32x4.shr_u', value, f'(i32.const {n})'), op('i32x4.shl', value, f'(i32.const {32-n})'))


def small(value, a, b, c):
    return xor(rotr(value, a), rotr(value, b), op('i32x4.shr_u', value, f'(i32.const {c})'))


def big(value, a, b, c):
    return xor(rotr(value, a), rotr(value, b), rotr(value, c))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--constants', type=Path, required=True,
                        help='Path to sha2 0.10.9 src/consts.rs used by this prototype')
    constants = parser.parse_args().constants
    text = constants.read_text()
    def numbers(name, count):
        body = re.search(r'pub const ' + name + r'[^=]*=\s*\[([^]]+)\]', text).group(1)
        values = [int(n, 16) for n in re.findall(r'0x[0-9a-f]+', body)]
        assert len(values) == count
        return values
    constants = numbers('K32:', 64)
    initial = numbers('H256_256:', 8)
    d = ROOT / 'artifacts/sha256-four-lane-kernels-v1'
    d.mkdir(exist_ok=False)
    lines = ['(module (import "env" "memory" (memory 1))', '(func (export "sha256_four")']
    lines += [f'(param $p{i} i32)' for i in range(4)] + ['(param $blocks i32)(param $out i32)']
    variables = [f'h{i}' for i in range(8)] + list('abcdefgh') + [f'w{i}' for i in range(16)] + ['t1', 'load']
    lines += [f'(local ${v} v128)' for v in variables]
    lines += [f'(local.set $h{i} {const(n)})' for i, n in enumerate(initial)]
    lines += ['(block $done (loop $blocks', '(br_if $done (i32.eqz (local.get $blocks)))']
    lines += [f'(local.set ${name} {get("h"+str(i))})' for i, name in enumerate('abcdefgh')]
    for j in range(16):
        load = f'(v128.load32_zero offset={j*4} {get("p0")})'
        for lane in range(1, 4):
            load = op(f'i32x4.replace_lane {lane}', load, f'(i32.load offset={j*4} {get("p"+str(lane))})')
        lines.append(f'(local.set $load {load})')
        swap = ' '.join(str(4*lane + n) for lane in range(4) for n in [3, 2, 1, 0])
        lines.append(f'(local.set $w{j} (i8x16.shuffle {swap} {get("load")} {get("load")}))')
    names = list('abcdefgh')
    for t, k in enumerate(constants):
        j = t % 16
        if t >= 16:
            word = add(get(f'w{j}'), small(get(f'w{(j+1)%16}'), 7, 18, 3), get(f'w{(j+9)%16}'), small(get(f'w{(j+14)%16}'), 17, 19, 10))
            lines.append(f'(local.set $w{j} {word})')
        a, b, c, dv, e, f, g, h = names
        ch = xor(op('v128.and', get(e), get(f)), op('v128.and', op('v128.not', get(e)), get(g)))
        maj = op('v128.or', op('v128.and', get(a), get(b)), op('v128.and', get(c), op('v128.or', get(a), get(b))))
        t1 = add(get(h), big(get(e), 6, 11, 25), ch, const(k), get(f'w{j}'))
        t2 = add(big(get(a), 2, 13, 22), maj)
        lines += [f'(local.set $t1 {t1})', f'(local.set ${dv} {add(get(dv), get("t1"))})', f'(local.set ${h} {add(get("t1"), t2)})']
        names = [h, a, b, c, dv, e, f, g]
    assert names == list('abcdefgh')
    lines += [f'(local.set $h{i} {add(get("h"+str(i)), get(name))})' for i, name in enumerate(names)]
    lines += [f'(local.set $p{i} (i32.add {get("p"+str(i))} (i32.const 64)))' for i in range(4)]
    lines += ['(local.set $blocks (i32.sub (local.get $blocks) (i32.const 1)))', '(br $blocks)))']
    lines += [f'(v128.store offset={i*16} (local.get $out) {get("h"+str(i))})' for i in range(8)]
    lines += ['))']
    wat = d / 'sha256-four.wat'
    wat.write_text('\n'.join(lines) + '\n')
    wasm = d / 'sha256-four.wasm'
    subprocess.run([str(ROOT / 'artifacts/wasm-audit-target/release/imajev-wasm-audit'), 'wat', str(wat), str(wasm)], check=True)
    (d / 'build.json').write_text(json.dumps(dict(complete=True, wasm_sha256=sha(wasm), constants_sha256=sha(constants), source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), wat]}, constants_path=str(constants), scope='Four independent equal-length padded streams, ordinary SIMD only. Compression prototype; padding/buffer preparation and IC timing not measured.'), indent=2) + '\n')
    print('Four-lane SHA256 Wasm compiled and validated')


if __name__ == '__main__':
    main()
