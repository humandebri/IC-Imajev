#!/usr/bin/env python3
"""Fuse load64-zero + signed low extension into the same standard SIMD load.

Both forms read exactly eight bytes and sign-extend each I8 into an I16 lane.
No input/weight values, dot order, scales, layout or arithmetic precision change.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source', required=True, type=Path)
ap.add_argument('--output', required=True, type=Path)
ap.add_argument('--report', required=True, type=Path)
args = ap.parse_args()
source = args.source.read_text()
pattern = r'\(i16x8\.extend_low_i8x16_s \(v128\.load64_zero offset=(\d+) \(local\.get \$wp\)\)\)'
result, count = re.subn(pattern, lambda m: f'(v128.load8x8_s offset={m[1]} (local.get $wp))', source)
assert count == 1024, count
assert re.search(pattern, result) is None
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(result)
args.report.write_text(json.dumps(dict(replacements=count,
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    output_sha256=hashlib.sha256(result.encode()).hexdigest(),
    equivalence='Each identical eight-byte read produces the same eight signed I16 lanes.'), indent=2)+'\n')
print(f'Fused {count} exact signed weight loads')
