#!/usr/bin/env python3
"""Isolated rank343 K2 projection diagnostic with current odd-root K2 control."""
from pathlib import Path
import hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s3-k2-prepared-probe-v1';d.mkdir(exist_ok=False);build=d/'build';src=build/'src';src.mkdir(parents=True)
 kernels=ROOT/'artifacts/s3-k2-prepared-kernels-v1';kr=json.loads((kernels/'report.json').read_text());audit=json.loads((kernels/'layout-audit.json').read_text());validation=json.loads((kernels/'wasm-validation.json').read_text());assert audit['complete']and validation['all2_wasm_validated']
 normal=ROOT/'artifacts/update-k2-odd-roots-v1';nb=json.loads((normal/'build/report.json').read_text())
 for report in [kr,nb,audit,validation]:
  for key in ['source_hashes','dependency_hashes']:
   assert all(sha(ROOT/p)==h for p,h in report.get(key,{}).items())
 ns={'__name__':'plan','__file__':str(kernels/'plan.py')};exec(compile((kernels/'plan.py').read_text(),str(kernels/'plan.py'),'exec'),ns);ad,bd,cd,leaves,roots,_=ns['plan']()
 coeff=''
 for name,dag,names in [('A',ad,[a for a,_ in leaves]),('B',bd,[b for _,b in leaves])]:coeff+=f'pub(super)const {name}:[&[(usize,i16)];343]=[\n'+''.join('&['+','.join(f'({i},{v})'for i,v in sorted(dag.symbols[n].items()))+'],\n'for n in names)+'];\n'
 (src/'coeff.rs').write_text(coeff)
 prep='use core::arch::wasm32::*;\n#[target_feature(enable="simd128")]pub(super)unsafe fn inputs(q:&imajev_runtime::int8_kernel::QuantizedRows,groups:usize,out:*mut i16){let cols=q.cols();for group in 0..groups{for block in 0..cols/256{for k in(0..32).step_by(8){\n'
 for i in range(64):prep+=f'let a{i}=v128_load(q.values().as_ptr().add((group*8+{i//8})*cols+block*256+{i%8*32}+k).cast());\n'
 for n,terms in ad.nodes:
  entries=list(terms.items());p,v=entries[0];value=p if v>0 else 'i16x8_sub(i16x8_splat(0),'+p+')'
  for p,v in entries[1:]:value=f'i16x8_{"add"if v>0 else "sub"}({value},{p})'
  prep+=f'let {n}={value};\n'
 for m,(n,_)in enumerate(leaves):prep+=f'v128_store(out.add(({m}*groups+group)*(cols/8)+block*32+k).cast(),{n});\n'
 prep+='}}}}\n';(src/'prepare.rs').write_text(prep)
 runtime='''use imajev_runtime::{int8_kernel::QuantizedRows,Result};
#[path="coeff.rs"]mod coeff;
#[cfg(target_arch="wasm32")]#[path="prepare.rs"]mod prepare;
pub struct Prepared{data:Vec<i16>,rows:usize,cols:usize}
pub struct Operands{data:Vec<i16>,rows:usize,cols:usize,groups:usize}
impl Prepared{
 pub fn new(w:&[i8],rows:usize,cols:usize)->Result<Self>{
  if rows==0||rows%32!=0||cols==0||cols%256!=0||rows.checked_mul(cols)!=Some(w.len())||w.len()>30_000_000{return Err("rank343 fixed shape".into());}
  let blocks=cols/256;let mut data=vec![0i16;(rows/32)*blocks*343*128];
  for group in 0..rows/32{for block in 0..blocks{for m in 0..343{for k in 0..32{for lane in 0..4{
   let value:i16=coeff::B[m].iter().map(|&(j,c)|c*w[(group*32+lane*8+j%8)*cols+block*256+j/8*32+k]as i16).sum();
   data[((group*blocks+block)*343+m)*128+(k/2)*8+lane*2+k%2]=value;
  }}}}}
  Ok(Self{data,rows,cols})
 }
 pub fn bytes(&self)->usize{self.data.len()*2}
 pub fn project_wide(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Result<Vec<f32>>{
  if q.rows()==0||q.rows()>132||q.rows()!=a.rows||q.cols()!=a.cols||q.cols()!=self.cols||rows==0||rows%32!=0||rows>self.rows||q.rows()*rows>900_000||sw.len()!=rows||!sw.iter().all(|v|v.is_finite()&&*v>0.){return Err("rank343 projection shape".into());}
  let cols=self.cols;let blocks=cols/256;
  #[cfg(target_arch="wasm32")]
  let out=unsafe{
   let len=q.rows()*rows;let mut out=Vec::<core::mem::MaybeUninit<f32>>::with_capacity(len);out.set_len(len);
   let ap:[*const i16;343]=core::array::from_fn(|m|a.data.as_ptr().add(m*a.groups*(cols/8)));
   for r in (0..rows).step_by(32){let wp=[self.data.as_ptr().add((r/32)*blocks*343*128)];
    for block in 0..blocks{let f=if block==0{crate::s2_kernel::tile32_seed}else{crate::s2_kernel::tile32};f(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),rows,sw.as_ptr().add(r),out.as_mut_ptr().cast::<f32>().add(r),q.rows());}
   }
   let mut out=core::mem::ManuallyDrop::new(out);Vec::from_raw_parts(out.as_mut_ptr().cast::<f32>(),out.len(),out.capacity())
  };
  #[cfg(not(target_arch="wasm32")))]let out=unreachable!();
  Ok(out)
 }
}
pub fn operands(q:&QuantizedRows)->Operands{
 let groups=q.rows().div_ceil(8);let cols=q.cols();let len=343*groups*(cols/8);
 assert!(q.values().len()>=groups*8*cols);
 #[cfg(target_arch="wasm32")]
 let data={let mut data:Vec<i16>=Vec::with_capacity(len);unsafe{prepare::inputs(q,groups,data.as_mut_ptr());data.set_len(len);}data};
 #[cfg(not(target_arch="wasm32")))]let data=unreachable!();
 Operands{data,rows:q.rows(),cols,groups}
}
'''.replace('wasm32")))]','wasm32"))]')
 (src/'s2.rs').write_text(runtime)
 lib=(ROOT/'artifacts/s1-k2-latest-control-v1/build/src/lib.rs').read_text().replace('winograd','s2').replace('mod win_kernel','mod s2_kernel').replace('#[cfg(target_arch="wasm32")]\nmod kernel;','')
 # Run the candidate for n=1 as well: no fallback can hide a broken tail.
 (src/'lib.rs').write_text(lib)
 stubs=''
 for i,k in enumerate(kr['kernels']):
  suffix='_seed'if '_seed'in k['symbol']else ''
  stubs+=f'#[export_name="{k["symbol"]}"]#[inline(never)]pub(crate)unsafe extern "C" fn tile32{suffix}(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,out:*mut f32,n:usize){{let marker=core::hint::black_box(q as usize^w as usize^cols^start^sx as usize^stride^sw as usize^out as usize^n)as u32;for p in 0..n*32{{core::ptr::write_volatile(out.add(p),f32::from_bits((marker^{i+1})|0x7fc00000));}}}}\n'
 (src/'s2_kernel.rs').write_text(stubs)
 cmd=nb['command'][:];cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs');cmd[cmd.index('-o')+1]=str(build/'raw.wasm');env=json.loads((ROOT/'artifacts/s1-k2-latest-control-v1/build/report.json').read_text())['explicit_env']
 with (build/'compiler.log').open('w')as f:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**env),stdout=f,stderr=f,check=True)
 mapping={sha(p):p for p in list(normal.glob('*.wat'))+list((normal/'build').glob('*.wat'))+[ROOT/'artifacts/single-quad/build-v2/kernel0.wat',ROOT/'artifacts/single-quad/build-v2/kernel2.wat']}
 paths=[(mapping[p['source_sha256']],p['export'])for p in nb['patches']]+[(ROOT/k['path'],k['symbol'])for k in kr['kernels']]
 previous=build/'raw.wasm';patches=[]
 for i,(path,symbol)in enumerate(paths):
  target=build/('diagnostic.wasm'if i==len(paths)-1 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(path),str(target),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=target
 files=[Path(__file__),kernels/'layout-audit.json',kernels/'wasm-validation.json',kernels/'report.json',normal/'build/report.json']+list(src.glob('*.rs'))+[p for p,_ in paths];deps=[Path(cmd[i+1].split('=',1)[1])for i,v in enumerate(cmd)if v=='--extern']
 report=dict(wasm_sha256=sha(previous),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes={str(p.relative_to(ROOT)):sha(p)for p in deps},command=cmd,explicit_env=env,patches=patches,locals=max(k['locals']for k in kr['kernels']),output_tile=32,current_odd_roots_k2_control=True,single_token_fallback=False,scope=__doc__)
 (build/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(module=report['wasm_sha256'],patches=len(patches),locals=report['locals'])))
if __name__=='__main__':main()
