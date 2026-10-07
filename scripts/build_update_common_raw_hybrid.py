#!/usr/bin/env python3
"""Integrate measured rank49 shapes with shared rank7 layout and exact K carry."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--retry-build",action="store_true");args=ap.parse_args()
 old=ROOT/'artifacts/update-common-raw-rank7-v1/build';base=json.loads((old/'report.json').read_text());component=ROOT/'artifacts/s2-integer-input-basis-probe-v1';cr=json.loads((component/'build/report.json').read_text());kernels=ROOT/'artifacts/s2-cached-tail-dot-reuse-kernels-v1';kr=json.loads((kernels/'report.json').read_text())
 for report in [base,cr,kr]:
  for key in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in report.get(key,{}).items())
 for shape in ['q','out','gate','down']:
  report=json.loads((component/('check-'+shape)/'summary.json').read_text());assert report['complete'];assert all(sha(ROOT/p)==h for p,h in report['source_hashes'].items());assert all(v['native_bits_equal'] for v in report['cases'])
 d=ROOT/'artifacts/update-common-raw-hybrid-v1';d.mkdir(exist_ok=args.retry_build);b=d/'build';b.mkdir(exist_ok=args.retry_build);assert not (b/'report.json').exists();shutil.copytree(old/'runtime',b/'runtime',dirs_exist_ok=args.retry_build)
 for p in old.glob('*.rs'):shutil.copyfile(p,b/p.name)
 prep=(component/'build/src/prepare.rs').read_text().replace('imajev_runtime::','crate::').replace('pub(super)unsafe fn inputs(q:&crate::int8_kernel::QuantizedRows,groups:usize,out:*mut i16)','pub(super)unsafe fn inputs(q:&crate::int8_kernel::QuantizedRows,groups:usize,out:*mut i16,first:usize,end:usize)').replace('for block in 0..cols/256','for block in first..end');assert 'first..end' in prep
 (b/'runtime/rank49_prepare.rs').write_text('#![cfg(target_arch="wasm32")]\n'+prep)
 (b/'runtime/rank49_kernel.rs').write_bytes((component/'build/src/s2_kernel.rs').read_bytes())
 ns={'__name__':'plan','__file__':str(ROOT/'scripts/plan_rank49_integer_input_basis.py')};exec(compile(Path(ns['__file__']).read_text(),ns['__file__'],'exec'),ns);ad,_,_,leaves,_,_=ns['plan']()
 coeff='const A:[&[(usize,i16)];49]=[\n'+''.join('&['+','.join(f'({i},{v})'for i,v in sorted(ad.symbols[n].items()))+'],\n'for n,_ in leaves)+'];\n'
 text='''use crate::int8_kernel::QuantizedRows;
use crate::output_pairs::PackedView;
#[cfg(target_arch="wasm32")]#[path="rank49_prepare.rs"]mod prepare;
#[cfg(target_arch="wasm32")]#[path="rank49_kernel.rs"]mod kernel;
#[derive(Debug)]pub(crate)struct Operands{data:Vec<i16>,groups:usize}
impl Operands{
 pub(crate)fn new(q:&QuantizedRows)->Self{
  let groups=q.rows().div_ceil(4);let len=49*groups*(q.cols()/4);let mut data=Vec::<i16>::with_capacity(len);
  unsafe{fill(q,groups,data.as_mut_ptr(),0,q.cols()/256);data.set_len(len);}Self{data,groups}
 }
}
pub(super)fn selected(q:&QuantizedRows,w:&PackedView)->bool{
 matches!(q.rows(),56|57)&&w.start/q.cols()%8==0&&matches!((w.rows(),q.cols()),(8192,2560)|(9216,2560)|(2560,4096)|(2560,9216))
}
unsafe fn fill(q:&QuantizedRows,groups:usize,out:*mut i16,first:usize,end:usize){
 #[cfg(target_arch="wasm32")]prepare::inputs(q,groups,out,first,end);
 #[cfg(not(target_arch="wasm32"))]for group in 0..groups{for block in first..end{for k in 0..64{for m in 0..49{
  let value:i16=A[m].iter().map(|&(i,c)|c*q.values()[(group*4+i/4)*q.cols()+block*256+(i%4)*64+k]).sum();
  out.add((m*groups+group)*(q.cols()/4)+block*64+k).write(value);
 }}}}
}
pub(super)fn project(q:&QuantizedRows,w:&PackedView,sw:&[f32])->Vec<f32>{
 let a=q.rank49_operands();blocks(q,w,sw,a.data.as_ptr(),a.groups,0,q.cols()/256,None)
}
#[cfg(feature="experimental-int8-k-continue")]
pub(super)fn continued(q:&QuantizedRows,w:&PackedView,sw:&[f32],first:usize,end:usize,initial:&[f32])->Vec<f32>{
 let groups=q.rows().div_ceil(4);let len=49*groups*(q.cols()/4);
 let data=crate::profile::measure("integer_k_prepare_new_columns",||{
  let mut data=Vec::<core::mem::MaybeUninit<i16>>::with_capacity(len);
  unsafe{data.set_len(len);fill(q,groups,data.as_mut_ptr().cast(),first,end);}data
 });
 blocks(q,w,sw,data.as_ptr().cast(),groups,first,end,Some(initial))
}
fn blocks(q:&QuantizedRows,w:&PackedView,sw:&[f32],data:*const i16,groups:usize,first:usize,end:usize,initial:Option<&[f32]>)->Vec<f32>{
 let cols=q.cols();let rows=w.rows();let start=w.start/cols;let fixed=&w.fixed;
 #[cfg(not(target_arch="wasm32"))]{
  let _=(data,groups);let mut out=initial.map_or_else(||vec![0.;q.rows()*rows],|v|v.to_vec());
  for t in 0..q.rows(){for r in 0..rows{for block in first..end{
   let dot:i32=(0..256).map(|k|q.values()[t*cols+block*256+k]as i32*(fixed.data[super::index(fixed.rows,cols,start+r,block*256+k)]as i8 as i32)).sum();
   out[t*rows+r]+=(dot as f32*q.scales()[t*(cols/256)+block])*sw[r];
  }}}out
 }
 #[cfg(target_arch="wasm32")]unsafe{
  let len=q.rows()*rows;let mut out=Vec::<core::mem::MaybeUninit<f32>>::with_capacity(len);out.set_len(len);let output=out.as_mut_ptr().cast::<f32>();
  if let Some(v)=initial{core::ptr::copy_nonoverlapping(v.as_ptr(),output,len);}
  let ap:[*const i16;49]=core::array::from_fn(|m|data.add(m*groups*(cols/4)));let plane=fixed.rows*cols/4;let mut r=0;
  while r<rows{let tile=if rows-r>=96{96}else{16};
   let wp:[*const i8;16]=core::array::from_fn(|m|fixed.data.as_ptr().add(((m/4/2)*2+(m%4)%2)*plane+((start+r)/8+(m%4)/2)*cols*2+((m/4)%2)*256).cast());
   for block in first..end{let f=match(tile,initial.is_none()&&block==first){(96,true)=>kernel::tile96_seed,(96,false)=>kernel::tile96,(16,true)=>kernel::tile16_seed,_=>kernel::tile16};
    f(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),rows,sw.as_ptr().add(r),output.add(r),q.rows());
   }r+=tile;
  }
  let mut out=core::mem::ManuallyDrop::new(out);Vec::from_raw_parts(out.as_mut_ptr().cast(),out.len(),out.capacity())
 }
}
'''+coeff
 (b/'runtime/rank49_raw.rs').write_text(text)
 p=b/'runtime/strassen_raw.rs';s=p.read_text();s=s.replace('use super::PackedView;','#[path="rank49_raw.rs"]pub(crate)mod rank49;\nuse super::PackedView;',1)
 before=' if q.rows()==1{return single(q,w,sw,0,q.cols()/256,vec![0.;w.rows()]);}'
 assert s.count(before)==1;s=s.replace(before,' if rank49::selected(q,w){return Ok(rank49::project(q,w,sw));}\n'+before)
 before=' let first=begin/256;let end=(begin+count)/256;';assert s.count(before)==1
 s=s.replace(before,before+'\n if rank49::selected(q,w){let out=rank49::continued(q,w,sw,first,end,initial);if !crate::all_finite(&out){return Err("rank49 continued finite".into());}return Ok(out);}')
 p.write_text(s)
 p=b/'runtime/int8_kernel.rs';s=p.read_text();needle='strassen: std::cell::OnceCell<crate::output_pairs::strassen_raw::Operands>,';assert s.count(needle)==1;s=s.replace(needle,needle+'\n    #[cfg(feature="experimental-strassen-raw")]\n    rank49:std::cell::OnceCell<crate::output_pairs::strassen_raw::rank49::Operands>,')
 needle='strassen:std::cell::OnceCell::new(),';assert s.count(needle)==4;s=s.replace(needle,needle+'#[cfg(feature="experimental-strassen-raw")]rank49:std::cell::OnceCell::new(),')
 needle='    #[cfg(feature = "experimental-projection-reuse")]\n    pub(crate) fn from_bytes';assert s.count(needle)==1;s=s.replace(needle,'    #[cfg(feature="experimental-strassen-raw")]\n    pub(crate)fn rank49_operands(&self)->&crate::output_pairs::strassen_raw::rank49::Operands{self.rank49.get_or_init(||crate::output_pairs::strassen_raw::rank49::Operands::new(self))}\n'+needle);p.write_text(s)
 runtime=base['runtime_command'][:];runtime[runtime.index('--edition=2021')+1]=str(b/'runtime/lib.rs');runtime[runtime.index('-o')+1]=str(b/'libimajev_runtime.rlib')
 cmd=base['command'][:];cmd[cmd.index('--edition=2021')+1]=str(b/'lib.rs');cmd[cmd.index('-o')+1]=str(b/'raw.wasm')
 for i,v in enumerate(cmd):
  if v.startswith('imajev_runtime='):cmd[i]='imajev_runtime='+str(b/'libimajev_runtime.rlib')
 env=dict(os.environ,CARGO_MANIFEST_DIR=str(b),CARGO_PKG_NAME='imajev-runtime',CARGO_PKG_VERSION='0.1.0')
 with(b/'runtime-compiler.log').open('w')as log:subprocess.run(runtime,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
 env.update(CARGO_PKG_NAME='imajev-inference',CARGO_PKG_VERSION_MAJOR='0',CARGO_PKG_VERSION_MINOR='1',CARGO_PKG_VERSION_PATCH='0',CARGO_PKG_VERSION_PRE='',CARGO_CRATE_NAME='imajev_inference')
 with(b/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
 paths={sha(ROOT/p):ROOT/p for p in base['source_hashes'] if p.endswith('.wat')};items=[(p['export'],paths[p['source_sha256']])for p in base['patches']]+[(p['symbol'],ROOT/p['path'])for p in kr['kernels']]
 previous=b/'raw.wasm';patches=[]
 for i,(symbol,wat)in enumerate(items):
  target=b/('full.wasm'if i==len(items)-1 else f'patched{i}.wasm');row=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(wat),str(target),symbol],text=True));assert row['wasmparser_validation'];patches.append(row);previous=target
 files=[Path(__file__),old/'report.json',component/'build/report.json',kernels/'report.json',kernels/'execution-report.json']+list(b.glob('*.rs'))+list((b/'runtime').glob('*.rs'))+[wat for _,wat in items]
 deps=[Path(command[i+1].split('=',1)[1])for command in [runtime,cmd]for i,v in enumerate(command)if v=='--extern'];files=list(dict.fromkeys(files));deps=list(dict.fromkeys(deps))
 r=dict(wasm_sha256=sha(previous),runtime_command=runtime,command=cmd,patches=patches,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes={str(p.relative_to(ROOT)):sha(p)for p in deps},rank49_tokens=[56,57],rank49_shapes=[[8192,2560],[9216,2560],[2560,4096],[2560,9216]],scope='Full shared-layout hybrid build. Cached fully initialized rank49 input operands; separate MaybeUninit storage for selected K ranges. Same ascendingK scale/add, seed only for new output, copy carry otherwise. Other shapes retain common-layout rank7/single. No whole runtime/full fidelity or paid goal claim.')
 (b/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(b/'source.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in files+[b/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(module=r['wasm_sha256'],bytes=previous.stat().st_size,validated_patches=len(patches))))
if __name__=='__main__':main()
