#!/usr/bin/env python3
"""Compare contiguous rank7 against unchanged current guard-fold control."""
from pathlib import Path
import json,hashlib,subprocess,os
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/k2-contiguous-roots-probe-v1';d.mkdir(exist_ok=False);b=d/'build';src=b/'src';src.mkdir(parents=True)
 k=ROOT/'artifacts/k2-contiguous-roots-kernels-v1';kr=json.loads((k/'report.json').read_text());er=json.loads((k/'execution-report.json').read_text());assert er['conditions']==208
 normal=ROOT/'artifacts/update-k2-pair-guard-fold-v1';nb=json.loads((normal/'build/report.json').read_text())
 for r in [kr,er,nb]:
  for key in ['source_hashes','dependency_hashes']:
   for p,h in r.get(key,{}).items():assert sha(ROOT/p)==h,p
 old=ROOT/'artifacts/s2-contiguous-roots-probe-v1/build/src/lib.rs';lib=old.read_text().replace('mod s2;','mod candidate;').replace('mod s2_kernel;','mod candidate_kernel;').replace('s2::','candidate::');(src/'lib.rs').write_text(lib)
 runtime=ROOT/'scripts/strassen_k2_runtime.rs';s=runtime.read_text();a=s.index('unsafe fn prepare(');z=s.index('pub(crate) fn index(',a);prep=s[a:z]
 # Caller owns padded QuantizedRows exactly as current runtime does.
 prepared='''use core::arch::wasm32::*;
 use imajev_runtime::{Result,int8_kernel::QuantizedRows};
 pub struct Prepared{data:Vec<i8>,rows:usize,cols:usize}
 pub struct Operands{data:Vec<i16>,rows:usize,cols:usize,pairs:usize}
 pub fn operands(q:&QuantizedRows)->Operands{let pairs=q.rows().div_ceil(2);let cols=q.cols();let len=7*pairs*(cols/2);let mut data=Vec::<i16>::with_capacity(len);unsafe{prepare(q,pairs,data.as_mut_ptr(),0,cols/256);data.set_len(len);}Operands{data,rows:q.rows(),cols,pairs}}
 impl Prepared{
 pub fn new(w:&[i8],rows:usize,cols:usize)->Result<Self>{assert!(rows%8==0&&cols%256==0&&w.len()==rows*cols);let plane=rows*cols/4;let mut data=vec![0i8;w.len()];for r in 0..rows{for c in 0..cols{let i=((c%256)/128*2+(r%8)/4)*plane+(r/8)*cols*2+(c/256)*512+((c%128)/2)*8+(r%4)*2+c%2;data[i]=w[r*cols+c];}}Ok(Self{data,rows,cols})}
 pub fn bytes(&self)->usize{self.data.len()}
 pub fn project_wide(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Result<Vec<f32>>{
 assert!(q.rows()==a.rows&&q.cols()==a.cols&&a.cols==self.cols&&rows<=self.rows&&rows%32==0&&sw.len()==rows);let cols=self.cols;let blocks=cols/256;
 unsafe{let len=q.rows()*rows;let mut out=Vec::<core::mem::MaybeUninit<f32>>::with_capacity(len);out.set_len(len);let ap:[*const i16;7]=core::array::from_fn(|m|a.data.as_ptr().add(m*a.pairs*(cols/2)));let plane=self.rows*cols/4;let mut r=0;
 while r<rows{let tile=if matches!(rows,8192|4096)&&rows-r>=168{168}else{32};let wp:[*const i8;4]=core::array::from_fn(|m|self.data.as_ptr().add([0,2,1,3][m]*plane+(r/8)*cols*2));
 for block in 0..blocks{let f=match(tile,block==0){(168,true)=>crate::candidate_kernel::tile168_seed,(168,false)=>crate::candidate_kernel::tile168,(32,true)=>crate::candidate_kernel::tile32_seed,_=>crate::candidate_kernel::tile32};f(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),rows,sw.as_ptr().add(r),out.as_mut_ptr().cast::<f32>().add(r),q.rows());}r+=tile;}
 let mut out=core::mem::ManuallyDrop::new(out);Ok(Vec::from_raw_parts(out.as_mut_ptr().cast::<f32>(),out.len(),out.capacity()))}}
 }
 '''+prep
 (src/'candidate.rs').write_text(prepared)
 kernels=[v for v in kr['kernels']if v['tile']in [168,32]];stubs=''
 for i,v in enumerate(kernels):
  t=v['tile'];seed='_seed'if v['seed']else'';stubs+=f'#[export_name="{v["symbol"]}_contiguous"]#[inline(never)]pub(crate)unsafe extern "C" fn tile{t}{seed}(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,out:*mut f32,n:usize){{let m=core::hint::black_box(q as usize^w as usize^cols^start^sx as usize^stride^sw as usize^out as usize^n)as u32;for p in 0..n*{t}{{core::ptr::write_volatile(out.add(p),f32::from_bits(m^{i+1}));}}}}\n'
 (src/'candidate_kernel.rs').write_text(stubs);cmd=nb['command'][:];cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs');cmd[cmd.index('-o')+1]=str(b/'raw.wasm');env=json.loads((ROOT/'artifacts/s1-k2-latest-control-v1/build/report.json').read_text())['explicit_env']
 with(b/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**env),stdout=log,stderr=log,check=True)
 mapping={sha(p):p for p in list(normal.glob('*.wat'))+list((normal/'build').glob('*.wat'))+[ROOT/'artifacts/single-quad/build-v2/kernel0.wat',ROOT/'artifacts/single-quad/build-v2/kernel2.wat']}
 paths=[(mapping[v['source_sha256']],v['export'])for v in nb['patches']]
 for v in kernels:
  p=b/(Path(v['path']).name);p.write_text((ROOT/v['path']).read_text().replace(v['symbol'],v['symbol']+'_contiguous'));paths.append((p,v['symbol']+'_contiguous'))
 prior=b/'raw.wasm';patches=[]
 for i,(p,symbol)in enumerate(paths):
  target=b/('diagnostic.wasm'if i==len(paths)-1 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(prior),str(p),str(target),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);prior=target
 files=[Path(__file__),runtime,old,k/'report.json',k/'execution-report.json',normal/'build/report.json']+list(src.glob('*.rs'))+[p for p,_ in paths];deps=[Path(cmd[i+1].split('=',1)[1])for i,v in enumerate(cmd)if v=='--extern']
 r=dict(wasm_sha256=sha(prior),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes={str(p.relative_to(ROOT)):sha(p)for p in deps},command=cmd,explicit_env=env,patches=patches,current_full_k2_control=True,single_token_current_k2_fallback=True,scope=__doc__)
 (b/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(module=r['wasm_sha256'],bytes=prior.stat().st_size,patches=len(patches))))
if __name__=='__main__':main()
