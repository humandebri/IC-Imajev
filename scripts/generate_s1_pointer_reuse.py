#!/usr/bin/env python3
"""Cache the seven immutable input-plane pointers once per S1 kernel call.

Input pointer tables are immutable and do not alias the output accumulator.
Uses the already validated shared-offset body; all dots and scales are unchanged.
"""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
p=ROOT/'artifacts/s1_address_reuse/build/kernel.wat';s=p.read_text()
anchor='(local.set $sw0';i=s.index(anchor)
locals='\n'.join(f'(local $qbase{m} i32)' for m in range(7))+'\n'
init='\n'.join(f'(local.set $qbase{m} (i32.load offset={m*4} (local.get $q)))' for m in range(7))+'\n'
s=s[:i]+locals+init+s[i:]
for m in range(7):
 old=f'(local.set $qp (i32.add (i32.load offset={m*4} (local.get $q)) (local.get $qoffset)))'
 new=f'(local.set $qp (i32.add (local.get $qbase{m}) (local.get $qoffset)))'
 assert s.count(old)==2;s=s.replace(old,new)
(d/'kernel.wat').write_text(s);sha=lambda b:hashlib.sha256(b).hexdigest()
(d/'generator.json').write_text(json.dumps(dict(scope=__doc__,generator_sha256=sha(Path(__file__).read_bytes()),input_wat_sha256=sha(p.read_bytes()),kernel_sha256=sha(s.encode()),pointer_table_loads_per_call_after=7,prior_pointer_table_loads_per_token_pair=7,operand_span_and_arithmetic_unchanged=True),indent=2)+'\n')
print('immutable input pointers loaded once per call')
