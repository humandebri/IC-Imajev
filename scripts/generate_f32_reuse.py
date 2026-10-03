#!/usr/bin/env python3
"""F32 LoRA B: original precision/K order, output-lane SIMD and all-token weight reuse."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',default='artifacts/f32_reuse/build');a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
lines=['(module (func (export "__imajev_pair_accumulate") (param $q i32) (param $w i32) (param $cols i32) (param $start i32) (param $sx i32) (param $stride i32) (param $sw i32) (param $sums i32) (param $n i32)', '(local $t i32) (local $qp i32) (local $yp i32) (local $x v128)']
for r in range(8):lines.append(f'(local $s{r} v128)')
for c in range(64):
 for r in range(8):lines.append(f'(local $w{c}_{r} v128)')
def row(first):
 out=['(local.set $qp (i32.add (local.get $q) (i32.shl (i32.mul (local.get $t) (local.get $cols)) (i32.const 2))))','(local.set $yp (i32.add (local.get $sums) (i32.shl (i32.mul (local.get $t) (local.get $stride)) (i32.const 2))))']
 for r in range(8):out.append(f'(local.set $s{r} (v128.const i32x4 0 0 0 0))')
 for c in range(64):
  for r in range(8):
   x=f'(local.tee $x (v128.load32_splat offset={c*4} (local.get $qp)))'if r==0 else '(local.get $x)'
   w=f'(local.tee $w{c}_{r} (v128.load offset={c*128+r*16} (local.get $w)))'if first else f'(local.get $w{c}_{r})'
   out.append(f'(local.set $s{r} (f32x4.add (local.get $s{r}) (f32x4.mul {x} {w})))')
 for r in range(8):out.append(f'(v128.store offset={r*16} (local.get $yp) (local.get $s{r}))')
 return out
lines+=['(local.set $t (i32.const 0))','(block $done (br_if $done (i32.eqz (local.get $n)))']+row(True)+['(local.set $t (i32.const 1))','(loop $tokens (br_if $done (i32.ge_u (local.get $t) (local.get $n)))']+row(False)+['(local.set $t (i32.add (local.get $t) (i32.const 1)))','br $tokens','))','))']
s='\n'.join(lines)+'\n';(d/'kernel.wat').write_text(s);sha=lambda b:hashlib.sha256(b).hexdigest();(d/'generator.json').write_text(json.dumps(dict(scope=__doc__,generator_sha256=sha(Path(__file__).read_bytes()),kernel_sha256=sha(s.encode()),weight_vectors=512,first_weight_read='first token only',first_input_read='first output vector only',sum_order='64 original columns ascending, separate mul/add, positive zero initial'),indent=2)+'\n');print('F32 kernel generated',len(s))
