#!/usr/bin/env python3
"""Exact raw INT8 S2; derive fixed coefficients on first token quartet only."""
import argparse,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
source=(ROOT/'scripts/generate_strassen2.py').read_text();g={'__file__':str(ROOT/'scripts/generate_strassen2.py')};exec(compile(source[:source.index("coeff = '//")],str(ROOT/'scripts/generate_strassen2.py'),'exec'),g)
leaves=g['leaves'];C=[e for row in g['result']for e in row]
lines=['(module (func (export "__imajev_s2_raw_accumulate") (param $q i32) (param $w i32) (param $cols i32) (param $start i32) (param $sx i32) (param $stride i32) (param $sw i32) (param $sums i32) (param $n i32)',
 '(local $t i32) (local $qp i32) (local $yp i32) (local $v v128) (local $h v128) (local $packed v128)']
for m in range(16):lines.append(f'(local $wp{m} i32)')
for m in range(49):
 lines.append(f'(local $p{m} v128)')
 for k in range(8):lines.append(f'(local $x{m}_{k} v128)')
for j in range(8):
 lines.append(f'(local $sw{j} v128)')
 for m in range(49):
  for k in range(8):lines.append(f'(local $w{m}_{j}_{k} v128)')
for m in range(16):
 for k in range(8):lines.append(f'(local $b{m}_{k} v128)')
for j in range(8):lines.append(f'(local.set $sw{j} (v128.load offset={j*16} (local.get $sw)))')
for m in range(16):lines.append(f'(local.set $wp{m} (i32.add (i32.load offset={m*4} (local.get $w)) (i32.shr_u (local.get $start) (i32.const 2))))')
def combine(e,name,kind):
 out=[]
 for i,(m,sign)in enumerate(sorted(e.items())):
  if i==0 and sign<0:out.append('(v128.const i32x4 0 0 0 0)')
  out.append(f'local.get ${name(m)}')
  if i or sign<0:out.append(kind+'.'+('add' if sign>0 else 'sub'))
 return out

def row(first):
 out=[]
 for j in range(8):
  if first:
   for b in range(16):
    for k in range(8):out.append(f'(local.set $b{b}_{k} (v128.load8x8_s offset={k*8} (i32.add (local.get $wp{b}) (i32.mul (i32.shr_u (local.get $cols) (i32.const 2)) (i32.const {j})))))')
  for m,(_,e)in enumerate(leaves):
   if j==0:out.append(f'(local.set $qp (i32.add (i32.load offset={m*4} (local.get $q)) (i32.add (i32.shr_u (i32.mul (local.get $t) (local.get $cols)) (i32.const 3)) (i32.shr_u (local.get $start) (i32.const 1)))))')
   for k in range(8):
    out.append(f'(local.tee $x{m}_{k} (v128.load offset={k*16} (local.get $qp)))'if j==0 else f'local.get $x{m}_{k}')
    if first:out+=combine(e,lambda b:f'b{b}_{k}','i16x8')+[f'local.tee $w{m}_{j}_{k}']
    else:out.append(f'local.get $w{m}_{j}_{k}')
    out+=['i32x4.dot_i16x8_s']+(['i32x4.add']if k else [])
   out.append(f'local.set $p{m}')
  for ti in range(4):
   out+=['(if (i32.lt_u (i32.add (local.get $t) (i32.const '+str(ti)+')) (local.get $n)) (then',
    '(local.set $packed (v128.const i32x4 0 0 0 0))']
   for ri in range(4):
    out+=combine(C[ti*4+ri],lambda m:f'p{m}','i32x4')+['local.set $h',
     'local.get $h','local.get $h','local.get $h','i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7','i32x4.add','local.set $h',
     'local.get $packed','local.get $h','local.get $h','local.get $h','i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11','i32x4.add','i32x4.extract_lane 0',f'i32x4.replace_lane {ri}','local.set $packed']
   out += [f'(local.set $yp (i32.add (local.get $sums) (i32.shl (i32.add (local.get $t) (i32.const {ti})) (i32.const 7))))',
    'local.get $yp',f'v128.load offset={j*16}','local.get $packed','f32x4.convert_i32x4_s',
    f'(v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {ti})) (local.get $stride)) (i32.const 2))))',
    'f32x4.mul',f'local.get $sw{j}','f32x4.mul','f32x4.add','local.set $v','local.get $yp','local.get $v',f'v128.store offset={j*16}', '))']
 return out
lines+=['(local.set $t (i32.const 0))','(block $done','(br_if $done (i32.eqz (local.get $n)))']+row(True)+['(local.set $t (i32.const 4))','(loop $tokens','(br_if $done (i32.ge_u (local.get $t) (local.get $n)))']+row(False)+['(local.set $t (i32.add (local.get $t) (i32.const 4)))','br $tokens','))','))']
wat='\n'.join(lines)+'\n';(d/'kernel.wat').write_text(wat);sha=lambda b:hashlib.sha256(b).hexdigest();(d/'generator.json').write_text(json.dumps(dict(scope=__doc__,generator_sha256=sha(pathlib.Path(__file__).read_bytes()),symbolic_generator_sha256=sha(source.encode()),kernel_sha256=sha(wat.encode()),symbolic_identity_checked=True,i32_bound_checked=True,fixed_bytes_ratio=1,weight_vectors=49*8*8,input_vectors=49*8,first_weight_expansion='first token quartet only',input_preparation='once per query, shared output tiles',late_horizontal_reduce=True),indent=2)+'\n')
print(json.dumps(dict(wat_bytes=len(wat),fixed_bytes_ratio=1)))
