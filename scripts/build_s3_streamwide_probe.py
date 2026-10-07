#!/usr/bin/env python3
"""Rank343 tile512: retain weights for one leaf, stream token groups, reconstruct exact integer products afterward."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s3-streamwide-v1';J=32
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def kernel(ns):
 a,b,c,leaves,result,_=ns['plan']();recombine,roots,capacity=ns['reconstruct'](c,result)
 lines=['(module (func (export "__imajev_s3_stream_accumulate") '+' '.join(f'(param ${p} i32)'for p in ['q','w','cols','start','sx','stride','sw','sums','n'])]
 lines+=['(local $t i32)(local $m i32)(local $qp i32)(local $yp i32)(local $wp i32)(local $pp i32)(local $products i32)(local $outstride i32)(local $v v128)']
 lines+=[f'(local $pc{i} v128)'for i in range(capacity)]
 lines+=[f'(local $w{j}_{k} v128)'for j in range(J)for k in range(8)]
 lines+=[f'(local $x{k} v128)'for k in range(8)]
 lines+=[f'(local $sx{i} v128)'for i in range(8)]
 lines+=[f'(local $sw{i} v128)'for i in range(J*4)]
 lines+=[f'(local $h{i} v128)'for i in range(4)]
 lines+=['(local.set $outstride (i32.load offset=1372 (local.get $q)))','(local.set $products (i32.load offset=1376 (local.get $q)))']
 lines+=[f'(local.set $sw{i} (v128.load offset={i*16} (local.get $sw)))'for i in range(J*4)]
 # Prepared weights are [output-group][rank-leaf][eight vectors].
 lines+=['(local.set $m (i32.const 0))','(loop $leaves','(local.set $wp (i32.add (local.get $w) (i32.shl (local.get $m) (i32.const 7))))']
 lines+=[f'(local.set $w{j}_{k} (v128.load offset={j*343*128+k*16} (local.get $wp)))'for j in range(J)for k in range(8)]
 lines+=['(local.set $t (i32.const 0))','(block $leafdone (loop $tokens (br_if $leafdone (i32.ge_u (local.get $t) (local.get $n)))','(local.set $qp (i32.add (i32.load (i32.add (local.get $q) (i32.shl (local.get $m) (i32.const 2)))) (i32.shl (i32.shr_u (local.get $t) (i32.const 3)) (i32.const 7))))',f'(local.set $pp (i32.add (local.get $products) (i32.shl (i32.add (i32.mul (i32.shr_u (local.get $t) (i32.const 3)) (i32.const {J*343})) (local.get $m)) (i32.const 4))))']
 lines+=[f'(local.set $x{k} (v128.load offset={k*16} (local.get $qp)))'for k in range(8)]
 for j in range(J):
  lines+=['local.get $pp']
  for k in range(8):lines += [f'local.get $x{k}',f'local.get $w{j}_{k}','i32x4.dot_i16x8_s']+(['i32x4.add']if k else [])
  lines.append(f'v128.store offset={j*343*16}')
 lines+=['(local.set $t (i32.add (local.get $t) (i32.const 8)))','br $tokens','))','(local.set $m (i32.add (local.get $m) (i32.const 1)))','(br_if $leaves (i32.lt_u (local.get $m) (i32.const 343)))',')']
 # Every scratch slot was initialized by the complete rank traversal above.
 lines+=['(local.set $t (i32.const 0))','(block $done (loop $reconstruct (br_if $done (i32.ge_u (local.get $t) (local.get $n)))',f'(local.set $pp (i32.add (local.get $products) (i32.mul (i32.shr_u (local.get $t) (i32.const 3)) (i32.const {J*343*16}))))']
 for i in range(8):lines.append(f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {i})) (local.get $n)) (then (local.set $sx{i} (v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {i})) (local.get $stride)) (i32.const 2)))))))')
 def shuffle(x,y,idx):return [f'local.get ${x}',f'local.get ${y}','i8x16.shuffle '+' '.join(map(str,idx))]
 for j in range(J):
  lines+=[f'(local.set $pc{m} (v128.load offset={(j*343+m)*16} (local.get $pp)))'for m in range(343)]
  lines+=recombine
  for ti in range(8):
   lines += [f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {ti})) (local.get $n)) (then',f'(local.set $yp (i32.add (local.get $sums) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {ti})) (local.get $outstride)) (i32.const 2))))']
   for group in range(2):
    pcs=[f'pc{roots[ti][group*4+i]}'for i in range(4)]
    for i in range(2):
     lines+=shuffle(pcs[i*2],pcs[i*2+1],[*range(0,4),*range(16,20),*range(8,12),*range(24,28)])
     lines+=shuffle(pcs[i*2],pcs[i*2+1],[*range(4,8),*range(20,24),*range(12,16),*range(28,32)])
     lines+=['i32x4.add',f'local.set $h{i}']
    for half in range(2):
     offset=j*64+group*16+half*32;scale=j*4+group+half*2
     lines+=['local.get $yp',f'v128.load offset={offset}']
     lines+=shuffle('h0','h1',list(range(half*8,half*8+8))+list(range(16+half*8,24+half*8)))
     lines+=['f32x4.convert_i32x4_s',f'local.get $sx{ti}','f32x4.mul',f'local.get $sw{scale}','f32x4.mul','f32x4.add','local.set $v','local.get $yp','local.get $v',f'v128.store offset={offset}']
   lines.append('))')
 lines+=['(local.set $t (i32.add (local.get $t) (i32.const 8)))','br $reconstruct','))','))']
 wat='\n'.join(lines)+'\n';count=len(re.findall(r'\(local \$',wat));assert count<10000,count;return wat,count
def main():
 D.mkdir(exist_ok=False);gen=ROOT/'artifacts/s3-winograd-v1/frozen-generator.py';ns=dict(__file__=str(gen),__name__='streamwide_plan');exec(compile(gen.read_text(),str(gen),'exec'),ns);wat,count=kernel(ns);(D/'kernel.wat').write_text(wat)
 old=ROOT/'artifacts/s3-winograd-v1/frozen-builder.py';s=old.read_text().replace('artifacts/s3-winograd-v1/build','artifacts/s3-streamwide-v1/build')
 s=s.replace('(rows/32)*(cols/256)*2*343*8*8','(rows/512)*(cols/256)*32*343*8*8').replace('tile in 0..rows/32','tile in 0..rows/512').replace('j in 0..2 {for k','j in 0..32 {for k').replace('tile*32+j*16','tile*512+j*16').replace('block)*2+j','block)*32+j')
 changes=[('let len=343*groups*64;','let mut products=Vec::<i32>::with_capacity(343*32*groups*4);\n let len=343*groups*64;'),('let mut ap=[0usize;344]','let mut ap=[0usize;345]'),('ap[343]=rows;','ap[343]=rows;ap[344]=products.as_mut_ptr()as usize;'),('for r in (0..rows).step_by(32)','for r in (0..rows).step_by(512)'),('((r/32)*(cols/256)+block)*2*343*8*8','((r/512)*(cols/256)+block)*32*343*8*8'),('rows%32!=0','rows%512!=0')]
 edit='\n'.join('    exact=exact.replace('+repr(a)+','+repr(b)+')'for a,b in changes)+'\n    p.write_text(exact)';s=s.replace('    p.write_text(exact)',edit)
 lo=s.index('    wat = (ROOT/');hi=s.index("    (D / 'kernel.wat').write_text(wat)",lo);s=s[:lo]+"    wat = (ROOT/'artifacts/s3-streamwide-v1/kernel.wat').read_text()\n"+s[hi:]
 (D/'frozen-builder.py').write_text(s);paths=[Path(__file__),gen,old,D/'frozen-builder.py',D/'kernel.wat',ROOT/'artifacts/s3-winograd-v1/integer-bounds.json'];(D/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in paths},indent=2)+'\n')
 (D/'layout.json').write_text(json.dumps(dict(output_tile=512,locals=count,rank=343,token_group=8,weight_cache_vectors=256,input_registers=8,product_scratch_bytes='343 * 32 * ceil(tokens/8) *16',scratch_initialized_before_reconstruction=True,f32_block_order_unchanged=True,prepared_weight_bytes_ratio=343/32,scope='Isolated Q projection only. Full model expanded-weight storage is not implemented or claimed.'),indent=2)+'\n')
 exec(compile(s,str(D/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
