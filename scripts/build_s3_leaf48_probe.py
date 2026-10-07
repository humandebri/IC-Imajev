#!/usr/bin/env python3
"""Diagnostic Winograd tile48: share eight input registers across three output groups."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/s3-leaf48-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def kernel(ns):
 a,b,c,leaves,result,_=ns['plan']();recombine,roots,capacity=ns['reconstruct'](c,result)
 old=(ROOT/'artifacts/s3-winograd-v1/build/kernel.wat').read_text()
 head=old[:old.index('(local.set $outstride')]
 head=re.sub(r'\(local \$x\d+_\d+ v128\)','',head)
 head=re.sub(r'\(local \$pc(\d+) v128\)',lambda m:'' if int(m[1])<343 else m[0],head)
 head+='\n'+'\n'.join(f'(local $x{k} v128)' for k in range(8))+'\n'
 head+='\n'.join(f'(local $p{j}_{m} v128)' for j in range(3) for m in range(343))+'\n'
 head+='\n'.join(f'(local $w2_{m}_{k} v128)' for m in range(343) for k in range(8))+'\n'
 head+='\n'.join(f'(local $sw{i} v128)' for i in range(8,12))+'\n'
 lines=[head,'(local.set $outstride (i32.load offset=1372 (local.get $q)))']
 lines += [f'(local.set $sw{i} (v128.load offset={i*16} (local.get $sw)))' for i in range(12)]
 lines += [f'(local.set $w{j}_{m}_{k} (v128.load offset={((j*343+m)*8+k)*16} (local.get $w)))' for j in range(3) for m in range(343) for k in range(8)]
 lines+=['(local.set $t (i32.const 0))','(block $done (loop $tokens (br_if $done (i32.ge_u (local.get $t) (local.get $n)))']
 for i in range(8):
  lines.append(f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {i})) (local.get $n)) (then (local.set $sx{i} (v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {i})) (local.get $stride)) (i32.const 2)))))))')
 for m in range(343):
  lines.append(f'(local.set $qp (i32.add (i32.load offset={m*4} (local.get $q)) (i32.shl (i32.shr_u (local.get $t) (i32.const 3)) (i32.const 7))))')
  lines += [f'(local.set $x{k} (v128.load offset={k*16} (local.get $qp)))' for k in range(8)]
  for j in range(3):
   for k in range(8):lines += [f'local.get $x{k}',f'local.get $w{j}_{m}_{k}','i32x4.dot_i16x8_s']+(['i32x4.add'] if k else [])
   lines.append(f'local.set $p{j}_{m}')
 def reg(i,j):return f'p{j}_{i}' if i<343 else f'pc{i}'
 def shuffle(x,y,idx):return [f'local.get ${x}',f'local.get ${y}','i8x16.shuffle '+' '.join(map(str,idx))]
 for j in range(3):
  lines += [re.sub(r'\$pc(\d+)',lambda m:'$'+reg(int(m[1]),j),s) for s in recombine]
  for ti in range(8):
   lines += [f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {ti})) (local.get $n)) (then',f'(local.set $yp (i32.add (local.get $sums) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {ti})) (local.get $outstride)) (i32.const 2))))']
   for group in range(2):
    pcs=[reg(roots[ti][group*4+i],j) for i in range(4)]
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
 lines+=['(local.set $t (i32.add (local.get $t) (i32.const 8)))','br $tokens','))','))']
 wat='\n'.join(lines)+'\n';count=len(re.findall(r'\(local \$',wat));assert count<=10000,count
 return wat,count
def main():
 D.mkdir(exist_ok=False)
 gen=ROOT/'artifacts/s3-winograd-v1/frozen-generator.py';ns=dict(__file__=str(gen),__name__='leaf48_plan');exec(compile(gen.read_text(),str(gen),'exec'),ns)
 wat,count=kernel(ns);(D/'kernel.wat').write_text(wat)
 source=ROOT/'artifacts/s3-winograd-v1/frozen-builder.py';s=source.read_text().replace('artifacts/s3-winograd-v1/build','artifacts/s3-leaf48-v1/build')
 s=s.replace('(rows/32)*(cols/256)*2*343*8*8','(rows/48)*(cols/256)*3*343*8*8').replace('tile in 0..rows/32','tile in 0..rows/48').replace('j in 0..2 {for k','j in 0..3 {for k').replace('tile*32+j*16','tile*48+j*16').replace('block)*2+j','block)*3+j')
 s=s.replace('let s3=unsafe{crate::prepare_weights::weights(w,rows,cols)};Ok(Self{data,s3,rows,cols})','let padded_rows=rows.div_ceil(48)*48;let mut padded=vec![0i8;padded_rows*cols];padded[..w.len()].copy_from_slice(w);let s3=unsafe{crate::prepare_weights::weights(&padded,padded_rows,cols)};Ok(Self{data,s3,rows,cols})')
 s=s.replace('    p.write_text(exact)','    exact=exact.replace(\'let len=343*groups*64;\',\'let width=rows.div_ceil(48)*48;let mut padded_sw=vec![1f32;width];padded_sw[..rows].copy_from_slice(sw);let mut padded_out=vec![0f32;n*width];\\n let len=343*groups*64;\')\n    exact=exact.replace(\'ap[343]=rows;\',\'ap[343]=width;\')\n    exact=exact.replace(\'for r in (0..rows).step_by(32)\',\'for r in (0..width).step_by(48)\')\n    exact=exact.replace(\'((r/32)*(cols/256)+block)*2*343*8*8\',\'((r/48)*(cols/256)+block)*3*343*8*8\')\n    exact=exact.replace(\'sw.as_ptr().add(r),out.as_mut_ptr().add(r),n);\',\'padded_sw.as_ptr().add(r),padded_out.as_mut_ptr().add(r),n);\')\n    exact=exact.replace(\' if out.iter().any(|v|!v.is_finite()){return Err("S3 nonfinite output".into());}\',\' for t in 0..n{out[t*rows..(t+1)*rows].copy_from_slice(&padded_out[t*width..t*width+rows]);}\\n if out.iter().any(|v|!v.is_finite()){return Err("S3 nonfinite output".into());}\')\n    p.write_text(exact)')
 lo=s.index('    wat = (ROOT/');hi=s.index("    (D / 'kernel.wat').write_text(wat)",lo)
 s=s[:lo]+"    wat = (ROOT/'artifacts/s3-leaf48-v1/kernel.wat').read_text()\n"+s[hi:]
 frozen=D/'frozen-builder.py';frozen.write_text(s)
 (D/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),gen,source,frozen,D/'kernel.wat',ROOT/'artifacts/s3-winograd-v1/integer-bounds.json']},indent=2)+'\n')
 exec(compile(s,str(source),'exec'),dict(__file__=str(ROOT/'scripts/build_s3_leaf48_probe.py'),__name__='__main__'))
 print(json.dumps(dict(output_tile=48,input_registers=8,locals=count,padded_output_and_weights=True)))
if __name__=='__main__':main()
