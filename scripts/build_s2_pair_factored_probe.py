#!/usr/bin/env python3
"""Share the nested rank49 weight transforms and integer reconstruction."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    upstream = ROOT/'scripts/build_s2_pair_late_probe.py'
    source = upstream.read_text()
    source = source.replace('artifacts/s2-pair-late-v1/build-v3', 'artifacts/s2-pair-factored-v1/build')
    anchor = "    prep = '''use core::arch::wasm32::*;"
    definitions = '''    positions = [[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]
    outer_b = [{0:1,3:1},{0:1},{1:1,3:-1},{2:1,0:-1},{3:1},{0:1,1:1},{2:1,3:1}]
    single_c = [{0:1,3:1,4:-1,6:1},{2:1,4:1},{1:1,3:1},{0:1,1:-1,2:1,5:1}]
    def linear(e, terms):
        r = {}
        for i, sign in e.items():
            r = g['combine'](r, terms[i], sign)
        return r
    outer_terms = [[linear(e,[{positions[b][i]:1} for b in range(4)]) for i in range(4)] for e in outer_b]
    z_terms = [[{m*7+k:sign for k,sign in e.items()} for e in single_c] for m in range(7)]
    for m in range(7):
        for k in range(7):
            assert linear(outer_b[k],outer_terms[m]) == leaves[m*7+k][1]
    for ti in range(4):
        for ri in range(4):
            assert linear(single_c[(ti//2)*2+ri//2],[z_terms[m][(ti%2)*2+ri%2] for m in range(7)]) == result[ti][ri]
    for row in z_terms:
        for e in row:
            bound = sum(abs(sign)*sum(abs(v) for v in leaves[m][0].values())*127*sum(abs(v) for v in leaves[m][1].values())*128*64 for m,sign in e.items())
            assert bound < 2**31
'''
    assert source.count(anchor) == 1
    source = source.replace(anchor, definitions+anchor)
    anchor = "    for m in range(49):\n        lines += [f'(local $p{m} v128)']"
    decl = "    for m in [0,2,3,5,6]:\n        for i in range(4):\n            lines += [f'(local $bo{m}_{i} v128)']\n    for m in range(7):\n        for i in range(4):\n            lines += [f'(local $z{m}_{i} v128)']\n"
    assert source.count(anchor) == 1
    source = source.replace(anchor, decl+anchor)
    anchor = "                    for m, (_, b) in enumerate(leaves):\n                        out += combine(b, lambda b:f'b{b}', 'i16x8')+[f'local.set $w{m}_{j}_{k}']"
    new = '''                    for m in [0,2,3,5,6]:
                        for i in range(4):
                            out += combine(outer_b[m],lambda b:f'b{positions[b][i]}','i16x8')+[f'local.set $bo{m}_{i}']
                    for m in range(7):
                        for inner, b in enumerate(outer_b):
                            name = lambda i:f'b{positions[0 if m == 1 else 3][i]}' if m in [1,4] else f'bo{m}_{i}'
                            out += combine(b,name,'i16x8')+[f'local.set $w{m*7+inner}_{j}_{k}']'''
    assert source.count(anchor) == 1
    source = source.replace(anchor,new)
    anchor = "                out += [f'local.set $p{m}']\n            for ti in range(4):"
    new = '''                out += [f'local.set $p{m}']
            for m in range(7):
                for inner, e in enumerate(single_c):
                    out += combine(e,lambda k:f'p{m*7+k}','i32x4')+[f'local.set $z{m}_{inner}']
            for ti in range(4):'''
    assert source.count(anchor) == 1
    source = source.replace(anchor,new)
    anchor = "                    out += combine(result[ti][ri], lambda m:f'p{m}', 'i32x4')+[f'local.set $c{ri}']"
    new = "                    out += combine(single_c[(ti//2)*2+ri//2],lambda m:f'z{m}_{(ti%2)*2+ri%2}','i32x4')+[f'local.set $c{ri}']"
    assert source.count(anchor) == 1
    source = source.replace(anchor,new)
    # The executed source is archived alongside its upstream identity.
    d = ROOT/'artifacts/s2-pair-factored-v1'
    d.mkdir(exist_ok=False)
    (d/'frozen-builder.py').write_text(source)
    (d/'upstream.sha256').write_text(hashlib.sha256(upstream.read_bytes()).hexdigest()+'\n')
    exec(compile(source,str(upstream),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__ == '__main__':
    main()
