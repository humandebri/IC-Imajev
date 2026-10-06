#!/usr/bin/env python3
"""Compile same-module output128 control/output160 candidate, preserving original raw weight layout."""
import hashlib,json,os,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-output160-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (D/'report.json').exists(),'Completed build exists'
 (D/'src').mkdir(exist_ok=True);source=ROOT/'scripts/s1_wide_bench/src'
 lib=(source/'lib.rs').read_text().replace('assert!(method<=3)','assert!(method==3||method==4)')
 lib=lib.replace('let out=if method==3 {','let out=if method==4 {f.coeff.as_ref().unwrap().project160(&q,prepared.as_ref().unwrap(),&f.scales[..rows],rows).unwrap()}else if method==3 {')
 lib+='\n#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}\n';(D/'src/lib.rs').write_text(lib)
 extra='''pub fn project160(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Result<Vec<f32>> {
 if q.rows()==0||q.rows()>132||q.rows()!=a.rows||q.cols()!=a.cols||q.cols()!=self.cols||rows==0||rows%32!=0||rows>self.rows||q.rows()*rows>900_000||sw.len()!=rows||!sw.iter().all(|s|s.is_finite()&&*s>0.){return Err("output160 bounds".into());}
 #[cfg(target_arch="wasm32")] let out=unsafe{
 let cols=self.cols;let mut out=vec![0.;q.rows()*rows];let ap:[*const i16;7]=core::array::from_fn(|m|a.data.as_ptr().add(m*a.pairs*cols));let mut r=0;
 while r<rows {let width=(rows-r).min(160);let tile=if width==160{160}else{128};let pad;let scales;
 let wp:[*const i8;4];let swp:*const f32;
 if width==tile{wp=core::array::from_fn(|m|self.data.as_ptr().add(m*(self.rows/4)*cols+(r/4)*cols));swp=sw.as_ptr().add(r);}
 else {pad={let mut p=vec![0i8;tile*cols];for m in 0..4{let from=m*(self.rows/4)*cols+(r/4)*cols;let to=m*(tile/4)*cols;p[to..to+width/4*cols].copy_from_slice(&self.data[from..from+width/4*cols]);}p};scales={let mut s=vec![1.;tile];s[..width].copy_from_slice(&sw[r..]);s};wp=core::array::from_fn(|m|pad.as_ptr().add(m*(tile/4)*cols));swp=scales.as_ptr();}
 let mut sums=vec![0.;q.rows()*tile];for b in 0..cols/256 {
 if tile==160{crate::kernel::wide160(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,b*256,q.scales().as_ptr().add(b),cols/256,swp,sums.as_mut_ptr(),q.rows());}
 else{crate::kernel::wide(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,b*256,q.scales().as_ptr().add(b),cols/256,swp,sums.as_mut_ptr(),q.rows());}}
 for t in 0..q.rows(){out[t*rows+r..t*rows+r+width].copy_from_slice(&sums[t*tile..t*tile+width]);}r+=width;
 }out};
 #[cfg(not(target_arch="wasm32"))] let out=self.scalar(q,a,sw,rows);
 if out.iter().any(|x|!x.is_finite()){return Err("output160 finite".into());}Ok(out)
 }
'''
 exact=(source/'exact.rs').read_text().replace('impl Prepared {','impl Prepared {\n'+extra,1);(D/'src/exact.rs').write_text(exact)
 kernel=(source/'kernel.rs').read_text()+'''
#[export_name="__imajev_s1_160_accumulate"] #[inline(never)]
pub(crate) unsafe extern "C" fn wide160(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){
let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;
for i in 0..n*160{core::ptr::write_volatile(sums.add(i),f32::from_bits(marker|0x7fc00000));}}
''';(D/'src/kernel.rs').write_text(kernel)
 old=json.loads((ROOT/'artifacts/s1_wide/raw128/report.json').read_text());cmd=old['command'][:];cmd[cmd.index('--edition=2021')+1]=str(D/'src/lib.rs');cmd[cmd.index('-o')+1]=str(D/'raw.wasm')
 control=ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat';sources=list((D/'src').glob('*.rs'))+[D/'kernel.wat',D/'generator.json',control,Path(__file__),ROOT/'scripts/generate_s1_output160.py'];hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
 assert all(sha(ROOT/p)==h for p,h in old['dependency_hashes'].items())
 with (D/'compiler.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**old['explicit_env']),check=True,stdout=log,stderr=log)
 patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch';patches=[]
 for previous,wat,out,symbol in [(D/'raw.wasm',control,D/'control.wasm','__imajev_s1_wide_accumulate'),(D/'control.wasm',D/'kernel.wat',D/'diagnostic.wasm','__imajev_s1_160_accumulate')]:patches.append(json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(out),symbol],text=True)))
 assert all(sha(ROOT/p)==h for p,h in hashes.items())
 report=dict(command=cmd,explicit_env=old['explicit_env'],source_hashes=hashes,dependency_hashes=old['dependency_hashes'],patcher_sha256=sha(patcher),patches=patches,wasm_sha256=sha(D/'diagnostic.wasm'),scope=__doc__);(D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sources:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(module=report['wasm_sha256'],bytes=(D/'diagnostic.wasm').stat().st_size)))
if __name__=='__main__':main()
