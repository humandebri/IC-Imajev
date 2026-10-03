#!/usr/bin/env python3
"""Exact S1 with late lane reduction and operands initialized on first dot use."""
import argparse,hashlib,json,pathlib,runpy
ROOT=pathlib.Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
g=runpy.run_path(str(ROOT/'scripts/generate_wat_s1.py'),run_name='__main__');old=(ROOT/'scripts/wat_s1_bench/kernel.wat').read_text();lines=old.splitlines();stop=next(i for i,line in enumerate(lines)if line.startswith('(local.set $sw0'));lines=lines[:stop]
for j in range(8):lines.append(f'(local.set $sw{j} (v128.load offset={j*16} (local.get $sw)))')
for m in range(7):lines.append(f'(local.set $wp{m} (i32.add (i32.load offset={m*4} (local.get $w)) (i32.shl (local.get $start) (i32.const 1))))')
def row(first):
 out=['(local.set $keep (i32.lt_u (i32.add (local.get $t) (i32.const 1)) (local.get $n)))','(local.set $yp (i32.add (local.get $sums) (i32.shl (local.get $t) (i32.const 7))))','(local.set $sx0 (v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (local.get $t) (local.get $stride)) (i32.const 2)))))','(if (local.get $keep) (then (local.set $sx1 (v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const 1)) (local.get $stride)) (i32.const 2)))))))']
 for j in range(8):
  for m in range(7):
   if j==0:out.append(f'(local.set $qp (i32.add (i32.load offset={m*4} (local.get $q)) (i32.add (i32.mul (local.get $t) (local.get $cols)) (i32.shl (local.get $start) (i32.const 1)))))')
   if first:out.append(f'(local.set $wp (i32.add (local.get $wp{m}) (i32.mul (local.get $cols) (i32.const {j*2}))))')
   for k in range(32):
    out.append(f'(local.tee $x{m}_{k} (v128.load offset={k*16} (local.get $qp)))'if j==0 else f'local.get $x{m}_{k}')
    out.append(f'(local.tee $w{m}_{j}_{k} (v128.load offset={k*16} (local.get $wp)))'if first else f'local.get $w{m}_{j}_{k}')
    out.append('i32x4.dot_i16x8_s')
    if k:out.append('i32x4.add')
   out.append(f'local.set $p{m}')
  for t in range(2):
   if t:out.append('(if (local.get $keep) (then')
   out+=['local.get $yp',f'v128.load offset={t*128+j*16}']+g['combine'](g['C'][t*2])+['local.set $a']+g['combine'](g['C'][t*2+1])+['local.set $value','local.get $a','local.get $value','i8x16.shuffle 0 1 2 3 16 17 18 19 8 9 10 11 24 25 26 27','local.get $a','local.get $value','i8x16.shuffle 4 5 6 7 20 21 22 23 12 13 14 15 28 29 30 31','i32x4.add','f32x4.convert_i32x4_s',f'local.get $sx{t}','f32x4.mul',f'local.get $sw{j}','f32x4.mul','f32x4.add','local.set $value','local.get $yp','local.get $value',f'v128.store offset={t*128+j*16}']
   if t:out.append('))')
 return out
lines+=['(local.set $t (i32.const 0))','(block $done','(br_if $done (i32.eqz (local.get $n)))']+row(True)+['(local.set $t (i32.const 2))','(loop $tokens','(br_if $done (i32.ge_u (local.get $t) (local.get $n)))']+row(False)+['(local.set $t (i32.add (local.get $t) (i32.const 2)))','br $tokens','))','))'];new='\n'.join(lines)+'\n';(d/'kernel.wat').write_text(new);sha=lambda b:hashlib.sha256(b).hexdigest();(d/'generator.json').write_text(json.dumps(dict(scope=__doc__,generator_sha256=sha(pathlib.Path(__file__).read_bytes()),original_kernel_sha256=sha(old.encode()),kernel_sha256=sha(new.encode()),symbolic_strassen_identity=True,weight_vectors=7*8*32,input_vectors=7*32,first_weight_read='first token pair only',first_input_read='first output quartet only',late_reduce=True),indent=2)+'\n')
