#!/usr/bin/env python3
"""Hoist the shared byte offset used by all seven S1 operands once per token pair.

Consumes the validated raw S1 body; changes no integer/F32 arithmetic or layout.
"""
import argparse, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
p=ROOT/'artifacts/wat_s1_raw/build/kernel.wat';s=p.read_text()
shared='(i32.add (i32.mul (local.get $t) (local.get $cols)) (i32.shl (local.get $start) (i32.const 1)))'
assert s.count(shared)==14
s=s.replace(shared,'(local.get $qoffset)')
local='(local $t i32)';assert s.count(local)==1;s=s.replace(local,local+' (local $qoffset i32)')
anchor='(local.set $keep (i32.lt_u (i32.add (local.get $t) (i32.const 1)) (local.get $n)))';assert s.count(anchor)==2
s=s.replace(anchor,f'(local.set $qoffset {shared})\n'+anchor)
assert s.count(shared)==2 and s.count('(local.get $qoffset)')==14
(d/'kernel.wat').write_text(s)
sha=lambda b:hashlib.sha256(b).hexdigest()
(d/'generator.json').write_text(json.dumps(dict(scope=__doc__,generator_sha256=sha(Path(__file__).read_bytes()),input_wat_sha256=sha(p.read_bytes()),kernel_sha256=sha(s.encode()),shared_expression=shared,expression_evaluations_per_pair_before=7,expression_evaluations_per_pair_after=1,operand_span_and_arithmetic_unchanged=True),indent=2)+'\n')
print('shared offset hoisted; operand/scaling/dot/reconstruction expressions unchanged')
