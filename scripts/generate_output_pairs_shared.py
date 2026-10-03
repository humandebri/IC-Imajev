#!/usr/bin/env python3
"""One exact kernel body instead of one large monomorph per token tile size."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT/'crates/imajev-runtime/src/output_pairs_simd.rs').read_text()
assert source.count('const R:usize') == 1
assert source.count('if R>$i') == 1
source = source.replace('scripts/generate_output_pairs.py', 'scripts/generate_output_pairs_shared.py')
source = source.replace('#[target_feature(enable="simd128")]', '#[inline(never)]\n#[target_feature(enable="simd128")]')
source = source.replace('accumulate<const R:usize>', 'accumulate')
source = source.replace('sums:&mut[[f32;32];R]', 'sums:&mut[[f32;32]]')
source = source.replace('const {assert!(R<=48);}', 'assert!(sums.len()<=48);')
source = source.replace('if R>$i', 'if sums.len()>$i')
assert 'const R:usize' not in source and 'if R>' not in source
path = ROOT/'crates/imajev-runtime/src/output_pairs_shared.rs'
if not path.exists() or path.read_text() != source:
    path.write_text(source)
print('Same guarded rows, dots/scales/add order, tile dispatch and weight spans; one non-generic body')
