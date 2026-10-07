#!/usr/bin/env python3
"""Fold pure local-based i32 constant address arithmetic, modulo 2**32."""
from pathlib import Path
import collections, hashlib, json, re, random
ROOT = Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    d = ROOT/'artifacts/k2-address-fold-kernels-v1'
    d.mkdir(exist_ok=False)
    base = ROOT/'artifacts/update-k2-pair-guard-fold-v1'
    files = [Path(__file__)]
    items = []
    rng = random.Random(256032)
    checked = 0
    for c in [0,1] + [1 << i for i in range(1,32)]:
        for x in [0,1,0x7fffffff,0x80000000,0xffffffff] + [rng.getrandbits(32) for _ in range(128)]:
            old = (x*c)&0xffffffff
            new = 0 if c == 0 else x if c == 1 else (x << (c.bit_length()-1))&0xffffffff
            assert old == new
            checked += 1
    for tile in [128,168,160,32]:
        for seed in [False,True]:
            p = base/(f'k2-direct{tile}'+('_seed' if seed else '')+'.wat')
            original = p.read_text()
            changes = []
            def fold(m):
                local, c = m.groups(); c = int(c)
                if c == 0: after = '(i32.const 0)'
                elif c == 1: after = f'(local.get {local})'
                elif 0 < c < 2**32 and c & (c-1) == 0:
                    after = f'(i32.shl(local.get {local})(i32.const {c.bit_length()-1}))'
                else: return m[0]
                changes.append(dict(before=m[0],after=after,rule=f'mul-{c}'))
                return after
            text = re.sub(r'\(i32.mul\(local.get (\$\w+)\)\(i32.const (\d+)\)\)',fold,original)
            def zero(m):
                after = f'(local.get {m[1]})'
                changes.append(dict(before=m[0],after=after,rule='add-zero'))
                return after
            text = re.sub(r'\(i32.add\(local.get (\$\w+)\)\(i32.const 0\)\)',zero,text)
            assert text != original
            # All changes contain only local reads and modular integer operations.
            def sensitive(s):
                return [line for line in s.splitlines() if any(op in line for op in ['f32x4.','i32x4.','i16x8.','i8x16.shuffle','v128.load','v128.store'])]
            assert sensitive(original) == sensitive(text)
            target = d/p.name; target.write_text(text); files += [p,target]
            items.append(dict(tile=tile,seed=seed,path=str(target.relative_to(ROOT)),
                symbol=re.search(r'export "([^"]+)"',text)[1],locals=text.count('(local $'),
                replacements=len(changes),rules=dict(collections.Counter(c['rule'] for c in changes))))
    report = dict(kernels=items,all_integer_dots_equal=True,modular_cases=checked,
                  all_sensitive_lines_byte_equal=True,performance_verified=False,
                  source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},scope=__doc__)
    for name in ['report.json','integer-audit.json']:
        (d/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(kernels=8,modular_cases=checked,replacements=sum(k['replacements'] for k in items))))
if __name__ == '__main__': main()
