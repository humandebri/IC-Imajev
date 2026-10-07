#!/usr/bin/env python3
"""Build or verify exact F32 carry byte append on the local paid candidate."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/report_paid_finite_max.py';s=p.read_text().replace('artifacts/paid-finite-max-v1','artifacts/paid-f32-byte-append-v1').replace('artifacts/update-finite-max-v1','artifacts/update-f32-byte-append-v1')
 old="dict(changed_runtime_files=['finite_simd.rs'],predicate_sites_equal=True,arithmetic_kernels_equal=True)"
 new="{'changed_runtime_files': ['attention_mlp_stream.rs', 'lib.rs', 'mlp_delta_stream.rs', 'mlp_pipeline.rs', 'mlp_stream.rs', 'prefix_hybrid_codec.rs', 'projection_codec.rs'], 'added_runtime_files': ['f32_byte_append.rs'], 'arithmetic_kernels_equal': True, 'scope': 'Fourteen scalar F32 byte-append loops replaced by twenty exact borrowed bulk copies. Field order, explicit owned Vec drops, validators and little endian representation preserved. Native helper keeps scalar byte conversion.'}"
 assert s.count(old)==1;s=s.replace(old,new)
 s=s.replace('Dense Delta arithmetic kernels unchanged from previous independently verified capture module.','Dense Delta kernels unchanged; F32 carry byte append validated against saved hidden/state references.')
 s=s.replace('extras=[Path(__file__),p,',"extras=[Path(__file__),p,ROOT/'scripts/report_paid_finite_max.py',d/'frozen-row-reporter.py',")
 (ROOT/'artifacts/paid-f32-byte-append-v1/frozen-row-reporter.py').write_text(s)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
