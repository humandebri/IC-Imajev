#!/usr/bin/env python3
"""Skip unused S1 products in the odd final token pair only.

For the zero-padded second token, products1/3 have zero operands and product5
is only used by the unused second output row. All weight initialization and
all complete token pairs retain the original kernel.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--directory', required=True)
ap.add_argument('--kernel', default='artifacts/single-quad/build-v2/kernel3.wat')
a = ap.parse_args()
d = ROOT / a.directory
d.mkdir(parents=True, exist_ok=True)
if any(d.iterdir()):
    raise ValueError('Use a fresh output directory')
source = (ROOT / a.kernel).read_text()
anchor = '(loop $tokens\n(br_if $done (i32.ge_u (local.get $t) (local.get $n)))\n'
assert source.count(anchor) == 1
start = source.index(anchor)+len(anchor)
end = source.index('(local.set $t (i32.add (local.get $t) (i32.const 2)))', start)
body = source[start:end]
tail = body
for m in [1, 3, 5]:
    # The first output quartet loads the operand; subsequent quartets use
    # cached input locals. Each product's region ends at its accumulator store.
    pattern = (rf'(?:\(local.set \$qp [^\n]*\)\n)?'
               rf'(?:\(local.tee \$x{m}_0 [^\n]*\)|local.get \$x{m}_0)\n'
               rf'.*?local.set \$p{m}\n')
    tail, count = re.subn(pattern, '', tail, flags=re.DOTALL)
    assert count == 32, (m, count)
    assert f'local.set $p{m}' not in tail
    # p3 contributes to the retained first output row and must be zero. p1/p5
    # are read only in the keep=false branch, but initialize them explicitly.
tail = '\n'.join(f'(local.set $p{m} (v128.const i32x4 0 0 0 0))' for m in [1,3,5])+'\n'+tail
assert 'local.set $w' not in body, 'All retained weights must already be initialized'
assert body.count('(if (local.get $keep) (then') == 33  # scale load +32 output quartets
loop_start = source.index(anchor)
suffix = source[end:]
assert suffix == '(local.set $t (i32.add (local.get $t) (i32.const 2)))\nbr $tokens\n))\n))\n'
# Complete pairs use their original loop with one precomputed even bound.
# The odd last row is handled once outside that loop. n=1 was already handled
# by the unchanged first-pair body and therefore has no later tail.
candidate = (source[:loop_start]
             +'(local.set $fulln (i32.and (local.get $n) (i32.const -2)))\n'
             +'(block $pairs_done\n(loop $tokens\n'
             +'(br_if $pairs_done (i32.ge_u (local.get $t) (local.get $fulln)))\n'
             +body+'(local.set $t (i32.add (local.get $t) (i32.const 2)))\nbr $tokens\n))\n'
             +'(if (i32.lt_u (local.get $t) (local.get $n)) (then\n'+tail+'))\n)\n))\n')
candidate = candidate.replace('(local $t i32)', '(local $t i32) (local $fulln i32)', 1)
sha = lambda b: hashlib.sha256(b).hexdigest()
(d/'kernel.wat').write_text(candidate)
(d/'generator.json').write_text(json.dumps(dict(
    generator_sha256=sha(Path(__file__).read_bytes()), original_kernel=a.kernel,
    original_kernel_sha256=sha(source.encode()), kernel_sha256=sha(candidate.encode()),
    skipped_products=[1,3,5], tail_guard='t<n after original first pair and complete-pair loop',
    scope=__doc__), indent=2)+'\n')
print('generated odd-tail candidate; full-pair body unchanged')
