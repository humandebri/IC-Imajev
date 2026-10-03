#!/usr/bin/env python3
"""Exact raw S2; four output quartets occupy independent I32 SIMD lanes."""
import argparse,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
source=(ROOT/'scripts/generate_strassen2.py').read_text();g={'__file__':str(ROOT/'scripts/generate_strassen2.py')};exec(compile(source[:source.index("coeff = '//")],g['__file__'],'exec'),g)
leaves=g['leaves'];C=[e for row in g['result']for e in row];assert all(v in(-1,1)for e in C for v in e.values())
def combine(e,name,kind):
 out=[]
 for i,(m,sign)in enumerate(sorted(e.items())):
  if i==0 and sign<0:out.append('(v128.const i32x4 0 0 0 0)')
  out.append(f'local.get ${name(m)}')
  if i or sign<0:out.append(kind+'.'+('add'if sign>0 else'sub'))
 return out
# Coefficients are computed once per query, duplicated across four weight lanes.
prep='''// Generated complete operand writes before committing Vec length.
use core::arch::wasm32::*;
#[target_feature(enable="simd128")]
pub(super) unsafe fn inputs(q:&imajev_runtime::int8_kernel::QuantizedRows,groups:usize,out:*mut i16){
 let cols=q.cols();
 for group in 0..groups{for block in 0..cols/256{for k in(0..64).step_by(8){
'''
for j in range(16):prep+=f'let x{j}=v128_load(q.values().as_ptr().add((group*4+{j//4})*cols+block*256+{j%4*64}+k).cast());\n'
for m,(e,_)in enumerate(leaves):
 terms=sorted(e.items());v=f'x{terms[0][0]}'if terms[0][1]>0 else f'i16x8_sub(i16x8_splat(0),x{terms[0][0]})'
 for j,sign in terms[1:]:v=f'i16x8_{"add"if sign>0 else"sub"}({v},x{j})'
 prep+=f'let value={v};let dst=out.add(({m}*groups+group)*cols+block*256+(k/2)*8);\n'
 for pair in range(4):
  idx=','.join(str(pair*4+i)for _ in range(4)for i in range(4));prep+=f'v128_store(dst.add({pair*8}).cast(),i8x16_shuffle::<{idx}>(value,value));\n'
prep+='}}}\n}\n';(ROOT/'scripts/wat_s2_lane_bench/src/prepare.rs').write_text(prep)
lines=['(module (func (export "__imajev_s2_raw_accumulate") (param $q i32) (param $w i32) (param $cols i32) (param $start i32) (param $sx i32) (param $stride i32) (param $sw i32) (param $sums i32) (param $n i32)', '(local $t i32) (local $qp i32) (local $yp i32) (local $v v128)']
for m in range(16):lines.append(f'(local $wp{m} i32) (local $b{m} v128)')
for ri in range(4):lines.append(f'(local $c{ri} v128) (local $u{ri} v128)')
for j in range(8):lines.append(f'(local $sw{j} v128)')
for m in range(49):
 lines.append(f'(local $p{m} v128)')
 for k in range(32):lines.append(f'(local $x{m}_{k} v128)')
for j in range(2):
 for m in range(49):
  for k in range(32):lines.append(f'(local $w{m}_{j}_{k} v128)')
for j in range(8):lines.append(f'(local.set $sw{j} (v128.load offset={j*16} (local.get $sw)))')
for m in range(16):lines.append(f'(local.set $wp{m} (i32.add (i32.load offset={m*4} (local.get $w)) (local.get $start)))')
def shuffle(a,b,idx):return [f'local.get ${a}',f'local.get ${b}','i8x16.shuffle '+' '.join(map(str,idx))]
def row(first):
 out=[]
 for j in range(2):
  if first:
   # Only sixteen raw vector temporaries; coefficients persist for later tokens.
   for k in range(32):
    for b in range(16):out.append(f'(local.set $b{b} (v128.load8x8_s offset={k*8} (i32.add (local.get $wp{b}) (i32.mul (local.get $cols) (i32.const {j})))))')
    for m,(_,e)in enumerate(leaves):out+=combine(e,lambda b:f'b{b}','i16x8')+[f'local.set $w{m}_{j}_{k}']
  for m in range(49):
   if j==0:out.append(f'(local.set $qp (i32.add (i32.load offset={m*4} (local.get $q)) (i32.add (i32.shr_u (i32.mul (local.get $t) (local.get $cols)) (i32.const 1)) (i32.shl (local.get $start) (i32.const 1)))))')
   for k in range(32):
    out.append(f'(local.tee $x{m}_{k} (v128.load offset={k*16} (local.get $qp)))'if j==0 else f'local.get $x{m}_{k}')
    out += [f'local.get $w{m}_{j}_{k}','i32x4.dot_i16x8_s']+(['i32x4.add']if k else [])
   out.append(f'local.set $p{m}')
  for ti in range(4):
   out+=['(if (i32.lt_u (i32.add (local.get $t) (i32.const '+str(ti)+')) (local.get $n)) (then']
   for ri in range(4):out+=combine(C[ti*4+ri],lambda m:f'p{m}','i32x4')+[f'local.set $c{ri}']
   out+=shuffle('c0','c1',[*range(0,4),*range(16,20),*range(4,8),*range(20,24)])+['local.set $u0']
   out+=shuffle('c0','c1',[*range(8,12),*range(24,28),*range(12,16),*range(28,32)])+['local.set $u1']
   out+=shuffle('c2','c3',[*range(0,4),*range(16,20),*range(4,8),*range(20,24)])+['local.set $u2']
   out+=shuffle('c2','c3',[*range(8,12),*range(24,28),*range(12,16),*range(28,32)])+['local.set $u3']
   for quarter in range(4):
    out += [f'(local.set $yp (i32.add (local.get $sums) (i32.shl (i32.add (local.get $t) (i32.const {ti})) (i32.const 7))))','local.get $yp',f'v128.load offset={(j*4+quarter)*16}']
    out += shuffle('u'+str(quarter//2),'u'+str(quarter//2+2),[*range(8*(quarter%2),8*(quarter%2)+8),*range(16+8*(quarter%2),24+8*(quarter%2))])
    out+=['f32x4.convert_i32x4_s',f'(v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {ti})) (local.get $stride)) (i32.const 2))))','f32x4.mul',f'local.get $sw{j*4+quarter}','f32x4.mul','f32x4.add','local.set $v','local.get $yp','local.get $v',f'v128.store offset={(j*4+quarter)*16}']
   out.append('))')
 return out
lines+=['(local.set $t (i32.const 0))','(block $done','(br_if $done (i32.eqz (local.get $n)))']+row(True)+['(local.set $t (i32.const 4))','(loop $tokens','(br_if $done (i32.ge_u (local.get $t) (local.get $n)))']+row(False)+['(local.set $t (i32.add (local.get $t) (i32.const 4)))','br $tokens','))','))'];wat='\n'.join(lines)+'\n';(d/'kernel.wat').write_text(wat)
sha=lambda b:hashlib.sha256(b).hexdigest();(d/'generator.json').write_text(json.dumps(dict(scope=__doc__,generator_sha256=sha(pathlib.Path(__file__).read_bytes()),symbolic_generator_sha256=sha(source.encode()),kernel_sha256=sha(wat.encode()),symbolic_identity_checked=True,i32_bound_checked=True,fixed_bytes_ratio=1,weight_vectors=49*32*2,input_vectors=49*32,raw_temporaries=16,scalar_horizontal_reduce=False),indent=2)+'\n');print(json.dumps(dict(wat_bytes=len(wat),fixed_bytes_ratio=1)))
