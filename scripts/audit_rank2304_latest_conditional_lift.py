#!/usr/bin/env python3
"""Preserve failed universal-I16 audit; prove I32 lift with explicit safe gate."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    original = ROOT/'scripts/audit_rank2304_latest_integer_lift.py'
    source = original.read_text()
    gate = 'assert apos.max()<=32767 and blo.min()>=-32768 and bhi.max()<=32767'
    assert source.count(gate)==1
    source = source.replace(gate,
        'universal_i16_safe=bool(apos.max()<=32767 and blo.min()>=-32768 and bhi.max()<=32767);assert not universal_i16_safe')
    condition = 'assert av.min()>=-32768 and av.max()<=32767 and bv.min()>=-32768 and bv.max()<=32767'
    assert source.count(condition)==1
    source = source.replace(condition,
        'operands_i16_safe=bool(av.min()>=-32768 and av.max()<=32767 and bv.min()>=-32768 and bv.max()<=32767)')
    source = source.replace('rank2304-latest-integer-lift-v1', 'rank2304-latest-conditional-lift-v1')
    source = source.replace('all256_integer_dots_exact=True,leaf_products_exceed_signed32=',
        'all256_integer_dots_exact=True,operands_i16_safe=operands_i16_safe,activation_observed_min=int(av.min()),activation_observed_max=int(av.max()),weight_observed_min=int(bv.min()),weight_observed_max=int(bv.max()),leaf_products_exceed_signed32=')
    source = source.replace('complete=True,rank=2304,',
        'complete=True,universal_i16_safe=universal_i16_safe,conditional_i16_gate_required=True,unchanged_input_and_weights=True,rank=2304,')
    source = source.replace('I16 bounds proved over fullI8 range, final division bypower2 exact.',
        'Universal I16 bound fails. Exact scalar I32/modular lift is proved; SIMD I16 requires a checked operands gate and unchanged fallback. Final division bypower2 exact.')
    entry = ROOT/'artifacts/rank2304-latest-conditional-entry-v1'
    entry.mkdir(exist_ok=False)
    frozen = entry/'frozen-auditor.py'
    source = source.replace('files=[Path(__file__),source,',
        "files=[Path(__file__),ROOT/'scripts/audit_rank2304_latest_integer_lift.py',ROOT/'artifacts/rank2304-latest-conditional-entry-v1/frozen-auditor.py',source,")
    compile(source,str(frozen),'exec')
    frozen.write_text(source)
    (entry/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)
        for p in [Path(__file__),original,frozen]},indent=2)+'\n')
    exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':
    main()
