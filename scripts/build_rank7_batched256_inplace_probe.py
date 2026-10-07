#!/usr/bin/env python3
"""Latest common-raw hybrid versus two-stage rank7 with full scratch cost."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    normal = ROOT/'artifacts/update-common-raw-hybrid-v1/build'
    old = ROOT/'artifacts/s3-integer-basis-probe-v1/build'
    kernels = ROOT/'artifacts/rank7-batched256-v4'
    nb = json.loads((normal/'report.json').read_text())
    ob = json.loads((old/'report.json').read_text())
    er = json.loads((kernels/'execution-report.json').read_text())
    assert er['complete'] and er['conditions']==42
    for report in [nb,ob,er]:
        for key in ['source_hashes','dependency_hashes']:
            for p,h in report.get(key,{}).items():
                assert sha(ROOT/p)==h,p
    d = ROOT/'artifacts/rank7-batched256-probe-v5'
    src = d/'build/src'
    src.mkdir(parents=True,exist_ok=False)
    (src/'lib.rs').write_text((old/'src/lib.rs').read_text().replace('#[cfg(target_arch="wasm32")]mod s2_kernel;',''))
    # Copy the proven partial-block input encoder with identical SIMD order.
    runtime = (normal/'runtime/strassen_raw.rs').read_text()
    begin = runtime.index('unsafe fn prepare(')
    end = runtime.index('\npub(crate) fn index',begin)
    prepare = runtime[begin:end]
    s = '''use imajev_runtime::int8_kernel::QuantizedRows;
pub struct Operands{data:Vec<i16>,pairs:usize}
pub fn operands(q:&QuantizedRows)->Operands{
 let pairs=q.rows().div_ceil(2);let len=7*pairs*(q.cols()/2);let mut data=Vec::<i16>::with_capacity(len);
 unsafe{prepare(q,pairs,data.as_mut_ptr(),0,q.cols()/256);data.set_len(len);}Operands{data,pairs}
}
''' + prepare + '''
pub struct Prepared{data:Vec<u8>,rows:usize,cols:usize}
fn index(rows:usize,cols:usize,row:usize,c:usize)->usize{
 ((c%256)/128*2+(row%8)/4)*(rows*cols/4)+(row/8)*cols*2+(c/256)*512+((c%128)/2)*8+(row%4)*2+c%2
}
impl Prepared{
 pub fn new(w:&[i8],rows:usize,cols:usize)->Result<Self,String>{
  assert_eq!(w.len(),rows*cols);assert_eq!(rows%256,0);let mut data=vec![0;w.len()];
  for r in 0..rows{for c in 0..cols{data[index(rows,cols,r,c)]=w[r*cols+c]as u8;}}
  Ok(Self{data,rows,cols})
 }
 pub fn bytes(&self)->usize{self.data.len()}
 pub fn project_wide(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Result<Vec<f32>,String>{
  assert_eq!(q.cols(),self.cols);assert!(rows<=self.rows&&rows%256==0);assert_eq!(sw.len(),rows);
  let mut out=vec![0.;q.rows()*rows];let mut scratch=vec![0u8;a.pairs*2048];
  let ap:[*const i16;8]=core::array::from_fn(|m|if m==7{scratch.as_ptr().cast()}else{unsafe{a.data.as_ptr().add(m*a.pairs*(self.cols/2))}});
  let plane=self.rows*self.cols/4;
  for r in(0..rows).step_by(256){
   let wp:[*const u8;4]=core::array::from_fn(|m|unsafe{self.data.as_ptr().add(m*plane+(r/8)*self.cols*2)});
   for block in 0..self.cols/256{unsafe{
    let f=if block==0{kernel::seed}else{kernel::normal};
    f(ap.as_ptr().cast(),wp.as_ptr().cast(),self.cols,block*256,q.scales().as_ptr().add(block),rows,sw.as_ptr().add(r),out.as_mut_ptr().add(r),q.rows());
   }}
  }Ok(out)
 }
}
mod kernel{
'''
    for seed in [False,True]:
        name='seed' if seed else 'normal'
        symbol='__imajev_rank7_batched256'+('_seed'if seed else'')
        s += f'''#[export_name="{symbol}"]#[inline(never)]
pub unsafe extern "C" fn {name}(q:*const u8,w:*const u8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,out:*mut f32,n:usize){{
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(out as usize)^n);
 core::ptr::write_volatile(out,core::hint::black_box(f32::from_bits((marker as u32 ^ {int(seed)})|0x7fc00000)));
}}
'''
    s += '}\n'
    (src/'s2.rs').write_text(s)
    cmd=nb['command'][:]
    cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs')
    cmd[cmd.index('-o')+1]=str(d/'build/raw.wasm')
    cmd=[('imajev_runtime='+str(normal/'libimajev_runtime.rlib'))if v.startswith('imajev_runtime=')else v for v in cmd]
    with (d/'build/compiler.log').open('w')as log:
        subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**ob['explicit_env']),stdout=log,stderr=log,check=True)
    paths={sha(ROOT/p):ROOT/p for p in nb['source_hashes']if p.endswith('.wat')}
    patches=[]
    items=[(paths[k['source_sha256']],k['export'])for k in nb['patches']]
    items += [(kernels/('seed.wat'if seed else'normal.wat'),'__imajev_rank7_batched256'+('_seed'if seed else''))for seed in [False,True]]
    previous=d/'build/raw.wasm'
    for i,(wat,symbol)in enumerate(items):
        target=d/'build'/('diagnostic.wasm'if i==len(items)-1 else f'patched{i}.wasm')
        patch=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(wat),str(target),symbol],text=True))
        assert patch['wasmparser_validation']
        patches.append(patch)
        previous=target
    deps=[Path(cmd[i+1].split('=',1)[1])for i,v in enumerate(cmd)if v=='--extern']
    files=[Path(__file__),normal/'report.json',old/'report.json',kernels/'report.json',kernels/'execution-report.json']+list(src.glob('*.rs'))+[p for p,_ in items]
    result=dict(wasm_sha256=sha(previous),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes={str(p.relative_to(ROOT)):sha(p)for p in deps},command=cmd,explicit_env=ob['explicit_env'],patches=patches,current_common_raw_hybrid_control=True,scope='Isolated rank7 tile256 two-stage component; input preparation, scratch allocation and all staging reads/writes measured. No full paid result.')
    (d/'build/report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(module=result['wasm_sha256'],patches=len(patches))))


if __name__=='__main__':
    main()
