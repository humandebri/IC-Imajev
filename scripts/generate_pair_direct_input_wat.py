#!/usr/bin/env python3
"""Generate the original-I16-input ABI from the exact operand-reuse kernel.

No compiled dependencies needed. load64_splat makes the same duplicated vector
at first use, so the caller can pass QuantizedRows.values() without a copy.
"""
from pathlib import Path
import re
import runpy

ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'scripts/generate_pair_reuse_wat.py'),run_name='__main__')
body=(ROOT/'artifacts/prefix_codec/full-wat/reuse.wat').read_text()
lines=body.splitlines();qp_changes=0;load_changes=0
for i,line in enumerate(lines):
    if line.startswith('(local.set $qp '):
        assert line.endswith('(i32.const 2))))')
        lines[i]=line.removesuffix('(i32.const 2))))')+'(i32.const 1))))'
        qp_changes+=1
    match=re.fullmatch(r'\(local.tee \$x(\d+) \(v128.load offset=(\d+) \(local.get \$qp\)\)\)',line)
    if match:
        g,offset=map(int,match.groups());assert offset==g*16
        lines[i]=f'(local.tee $x{g} (v128.load64_splat offset={g*8} (local.get $qp)))'
        load_changes+=1
assert (qp_changes,load_changes)==(2,128)
path=ROOT/'artifacts/prefix_codec/direct-input-build/direct-input.wat'
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text('\n'.join(lines)+'\n')
