#!/usr/bin/env python3
"""Reuse first-token-only scratch locals across S1 output quartets.

The seven retained weight planes stay unchanged. Only two temporary planes,
consumed before the next quartet, share names. No numerical operator changes.
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
declarations = re.findall(r'\(local \$(b(?:12|21))_(\d+)_(\d+) v128\)', source)
assert len(declarations) == 2 * 32 * 32
for plane, quartet, k in declarations:
    name = f'${plane}_{quartet}_{k}'
    assert source.count(f'(local {name} v128)') == 1
    assert source.count(f'local.tee {name} ') == 1
    reads = list(re.finditer(r'local.get '+re.escape(name)+r'(?=[\s)])', source))
    assert len(reads) == 1
    # The scratch values are only used in the first-token initialization body.
    assert source.index(f'local.tee {name} ') < reads[0].start()
    if int(quartet) < 31:
        assert reads[0].start() < source.index(f'local.tee ${plane}_{int(quartet)+1}_{k} ')
candidate = re.sub(r'\$b(12|21)_\d+_(\d+)\b', r'$b\1_\2', source)
seen = set()
def dedup(match):
    text = match.group()
    if text in seen:
        return ''
    seen.add(text)
    return text
candidate = re.sub(r'\(local \$b(?:12|21)_\d+ v128\)', dedup, candidate)
def operations(text):
    # Normalize local names and declarations to verify all executable tokens
    # are preserved, including their order. Alias liveness is checked above.
    text = re.sub(r'\(local \$\S+ \S+\)', '', text)
    return re.sub(r'\$\S+', '$local', text).split()
assert operations(source) == operations(candidate)
sha = lambda b: hashlib.sha256(b).hexdigest()
(d / 'kernel.wat').write_text(candidate)
(d / 'generator.json').write_text(json.dumps(dict(
    generator_sha256=sha(Path(__file__).read_bytes()),
    original_kernel=a.kernel, original_kernel_sha256=sha(source.encode()),
    kernel_sha256=sha(candidate.encode()), scratch_locals_before=len(declarations),
    scratch_locals_after=len(seen), removed_locals=len(declarations)-len(seen),
    executable_tokens_preserved=True, scope=__doc__), indent=2)+'\n')
print(f'removed {len(declarations)-len(seen)} scratch locals; operators unchanged')
