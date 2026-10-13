#!/usr/bin/env python3
"""Generalize the verified SIMD pair tile to bounded real-token groups."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'scripts/lane_pair_bench/src/kernel.rs').read_text()
s=s.replace('pub(crate) unsafe fn accumulate(', 'pub(crate) unsafe fn accumulate<const R:usize>(')
s=s.replace('sums:&mut[[f32;32];44]', 'sums:&mut[[f32;32];R]')
s=s.replace('use core::arch::wasm32::*;', 'const {assert!(R<=48);}'+'\n'+'use core::arch::wasm32::*;')
s=s.replace('=>{$({'+chr(10)+'let row=', '=>{$(if R>$i {'+chr(10)+'let row=')
s=s.replace('rows!('+','.join(map(str,range(44)))+');','rows!('+','.join(map(str,range(48)))+');')
s=s.replace('scripts/generate_lane_pair.py','scripts/generate_output_pairs.py')
# Private caller validates q/weight/scales and dispatches only complete K256
# blocks and real token tiles. All spans are bounded by the constructors;
# these indices never wrap. Remove repeated overflow branches only here.
s=s.replace('j*cols*2+start*2','j.wrapping_mul(cols).wrapping_mul(2).wrapping_add(start.wrapping_mul(2))')
s=s.replace('$i*stride','($i as usize).wrapping_mul(stride)')
s=s.replace('($i*cols+start)*2','($i as usize).wrapping_mul(cols).wrapping_add(start).wrapping_mul(2)')
s=s.replace('|j|w.add','|j:usize|w.add')
s=s.replace('//! Q and each weight pair', '//! Address spans are validated by the private caller; no valid index wraps.\n//! Q and each weight pair')
destination=ROOT/'crates/imajev-runtime/src/output_pairs_simd.rs'
# Keep the timestamp when the bytes are already identical: otherwise a
# reproducibility check triggers another expensive runtime compilation.
if not destination.exists() or destination.read_text()!=s:
    destination.write_text(s)
