#!/usr/bin/env python3
"""K2 per lane: four independent output columns, exact rank7 and ordered F32 blocks."""
from pathlib import Path
import hashlib,json,os,subprocess,runpy,re
from build_s1_winograd_fused_probe import weight
ROOT=Path(__file__).resolve().parents[1];TILE=128
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s1-k2-four-columns-adaptive168-v1';d.mkdir(exist_ok=False);b=d/'build';src=b/'src';src.mkdir(parents=True)
 old=ROOT/'artifacts/s1-winograd-fused-v1';ob=json.loads((old/'build/report.json').read_text())
 assert all(sha(ROOT/p)==h for p,h in ob['source_hashes'].items())
 for p in (old/'build/src').glob('*.rs'):(src/p.name).write_bytes(p.read_bytes())
 (d/'plan.py').write_bytes((old/'plan.py').read_bytes());(d/'integer-bounds.json').write_bytes((old/'integer-bounds.json').read_bytes())
 p=src/'winograd.rs';s=p.read_text()
 edits=[('for quartet in 0..rows/4 {for pair in 0..2','for quartet in 0..rows/8 {for pair in 0..4'),('let row=quartet*4+pair*2','let row=quartet*8+pair*2'),('quartet*cols+block*256+(k/4)*8+pair*4+k%4','quartet*2*cols+block*512+(k/2)*8+pair*2+k%2'),('(row/2)*self.cols+block*256+(k/4)*8+(row%2)*4+k%4','(row/4)*2*self.cols+block*512+(k/2)*8+(row%4)*2+k%2'),('pair*self.cols+block*256+(k/4)*8+k%4','pair*self.cols+block*256+k'),('let mut data=Vec::with_capacity(len);','let mut data=vec![0;len];'),('pair*cols+block*256+(k/4)*8+k%4;data[i]=v;data[i+4]=v','pair*cols+block*256+k;data[i]=v'),('v128_store(out.add(','v128_store64_lane::<0>(out.add('),('block*256+k*2).cast()','block*256+k).cast()')]
 for before,after in edits:assert s.count(before)>0,before;s=s.replace(before,after)
 s,count=re.subn(r'v128_store64_lane::<0>\((out.add\([^\n]+?\).cast\(\)),(\w+)\)',r'v128_store64_lane::<0>(\2,\1)',s);assert count==7
 p.write_text(s);(d/'runtime-edits.json').write_text(json.dumps(edits,indent=2)+'\n')
 p=src/'win_kernel.rs';text=p.read_text();stub=text[text.index('#[export_name="__imajev_win7_wide_accumulate"]'):]
 for tile in (168,32):text+='\n'+stub.replace('__imajev_win7_wide_accumulate',f'__imajev_win7_{tile}_accumulate').replace('fn wide(',f'fn wide{tile}(').replace('n*W',f'n*{tile}')
 p.write_text(text)
 p=src/'winograd.rs';text=p.read_text();lo=text.index('  for r in(0..rows).step_by(crate::win_kernel::W)');hi=text.index('  }out',lo)+3
 text=text[:lo]+'  let mut r=0;while r<rows{let tile=if (rows==8192||rows==4096)&&rows-r>=168{168}else if rows-r>=128{128}else{32};\n   let wp:[*const i8;4]=core::array::from_fn(|m|self.data.as_ptr().add(m*(self.rows/4)*cols+(r/4)*cols));let mut sums=vec![0f32;q.rows()*tile];\n   for b in 0..cols/256{let f=match tile{168=>crate::win_kernel::wide168,32=>crate::win_kernel::wide32,_=>crate::win_kernel::wide};f(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,b*256,q.scales().as_ptr().add(b),cols/256,sw.as_ptr().add(r),sums.as_mut_ptr(),q.rows());}\n   for t in 0..q.rows(){out[t*rows+r..t*rows+r+tile].copy_from_slice(&sums[t*tile..(t+1)*tile]);}r+=tile;\n  }' +text[hi:];p.write_text(text)
 def make_kernel(TILE):
  ns=runpy.run_path(str(d/'plan.py'));a,bb,c,leaves,roots,_=ns['plan']();rec,regs,capacity=ns['reconstruct'](c,roots)
  lines=['(module(func(export "__imajev_win7_wide_accumulate")(param $q i32)(param $w i32)(param $cols i32)(param $start i32)(param $sx i32)(param $stride i32)(param $sw i32)(param $sums i32)(param $n i32)', '(local $t i32)(local $qp i32)(local $qo i32)(local $keep i32)(local $yp i32)(local $yp1 i32)(local $sx0 v128)(local $sx1 v128)']
  for i in range(4):lines.append(f'(local $bp{i} i32)(local $wp{i} i32)')
  for i in range(capacity):lines.append(f'(local $pc{i} v128)')
  for m in range(7):
   for k in range(64):lines.append(f'(local $x{m}_{k} v128)')
   for j in range(TILE//8):
    for k in range(64):lines.append(f'(local $w{m}_{j}_{k} v128)')
  for j in range(TILE//4):lines.append(f'(local $sw{j} v128)')
  for j in range(TILE//4):lines.append(f'(local.set $sw{j}(v128.load offset={j*16}(local.get $sw)))')
  for i in range(4):lines.append(f'(local.set $wp{i}(i32.add(i32.load offset={i*4}(local.get $w))(i32.shl(local.get $start)(i32.const 1))))')
  def row(first,tail=False):
   out=['(local.set $qo(i32.shl(i32.add(i32.mul(i32.shr_u(local.get $t)(i32.const 1))(local.get $cols))(local.get $start))(i32.const 1)))','(local.set $keep(i32.lt_u(i32.add(local.get $t)(i32.const 1))(local.get $n)))',f'(local.set $yp(i32.add(local.get $sums)(i32.mul(local.get $t)(i32.const {TILE*4}))))',f'(local.set $yp1(i32.add(local.get $yp)(i32.const {TILE*4})))','(local.set $sx0(v128.load32_splat(i32.add(local.get $sx)(i32.shl(i32.mul(local.get $t)(local.get $stride))(i32.const 2)))))','(if(local.get $keep)(then(local.set $sx1(v128.load32_splat(i32.add(local.get $sx)(i32.shl(i32.mul(i32.add(local.get $t)(i32.const 1))(local.get $stride))(i32.const 2)))))))']
   for j in range(TILE//8):
    if first:
     for plane in range(4):out.append(f'(local.set $bp{plane}(i32.add(local.get $wp{plane})(i32.mul(local.get $cols)(i32.const {j*2}))))')
    for m in range(7):
     if tail and m in (3,4,6):
      out.append(f'(local.set $pc{m}(v128.const i32x4 0 0 0 0))');continue
     if j==0:out.append(f'(local.set $qp(i32.add(i32.load offset={m*4}(local.get $q))(local.get $qo)))')
     for k in range(64):
      out.append(f'(local.tee $x{m}_{k}(v128.load32_splat offset={k*4}(local.get $qp)))'if j==0 else f'local.get $x{m}_{k}')
      out+=weight(m,j,k)if first else [f'local.get $w{m}_{j}_{k}'];out.append('i32x4.dot_i16x8_s')
      if k:out.append('i32x4.add')
     out.append(f'local.set $pc{m}')
    out+=rec
    for ti in range(2):
     if ti:out.append('(if(local.get $keep)(then')
     address='local.get $yp1'if ti else 'local.get $yp'
     for part,mask in enumerate([[0,1,2,3,16,17,18,19,4,5,6,7,20,21,22,23],[8,9,10,11,24,25,26,27,12,13,14,15,28,29,30,31]]):
      offset=j*32+part*16
      out+=[address,address,f'v128.load offset={offset}',f'local.get $pc{regs[ti][0]}',f'local.get $pc{regs[ti][1]}','i8x16.shuffle '+' '.join(map(str,mask)),'f32x4.convert_i32x4_s',f'local.get $sx{ti}','f32x4.mul',f'local.get $sw{2*j+part}','f32x4.mul','f32x4.add',f'v128.store offset={offset}']
     if ti:out.append('))')
   return out
  lines+=['(block $done','(br_if $done(i32.eqz(local.get $n)))']+row(True)+['(local.set $t(i32.const 2))','(loop $tokens','(br_if $done(i32.ge_u(local.get $t)(local.get $n)))']+['(if(i32.lt_u(i32.add(local.get $t)(i32.const 1))(local.get $n))(then']+row(False)+[')(else']+row(False,True)+['))','(local.set $t(i32.add(local.get $t)(i32.const 2)))','br $tokens','))','))']
  wat=b/f'kernel{TILE}.wat';wat.write_text('\n'.join(lines)+'\n');assert wat.read_text().count('(')==wat.read_text().count(')');count=sum(x.count('(local $')for x in lines);assert count<10000
  return wat,count
 kernels=[make_kernel(tile)for tile in (128,168,32)]
 wat,count=kernels[0]
 for tile,(path,_)in zip((128,168,32),kernels):
  if tile!=128:path.write_text(path.read_text().replace('__imajev_win7_wide_accumulate',f'__imajev_win7_{tile}_accumulate'))
 cmd=ob['command'][:];cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs');cmd[cmd.index('-o')+1]=str(b/'raw.wasm')
 with (b/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**ob['explicit_env']),stdout=log,stderr=log,check=True)
 previous=b/'raw.wasm';patches=[];control=ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
 for path,symbol,target in [(control,'__imajev_s1_wide_accumulate',b/'control.wasm')]+[(path,'__imajev_win7_wide_accumulate'if tile==128 else f'__imajev_win7_{tile}_accumulate',b/('diagnostic.wasm'if tile==32 else f'patched{tile}.wasm'))for tile,(path,_)in zip((128,168,32),kernels)]:
  patch=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(path),str(target),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=target
 files=[Path(__file__),ROOT/'scripts/build_s1_winograd_fused_probe.py',d/'plan.py',d/'integer-bounds.json',d/'runtime-edits.json',control]+[p for p,_ in kernels]+list(src.glob('*.rs'))
 r=dict(wasm_sha256=sha(previous),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes=ob['dependency_hashes'],command=cmd,explicit_env=ob['explicit_env'],patches=patches,locals=max(n for _,n in kernels),output_tile=TILE,scope=__doc__)
 (b/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(module=r['wasm_sha256'],locals=count)))
if __name__=='__main__':main()
