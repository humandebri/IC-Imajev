#!/usr/bin/env python3
"""Profile unchanged S1 integer values with four counters in caller-owned scratch."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/s1-inner-profile-v1/build-v6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=False);(D/'src').mkdir()
 deps=ROOT/'artifacts/update-inference/client-target/release/deps';audit=ROOT/'artifacts/wasm-audit-target/release/deps'
 helper=['rustc','--edition=2021',str(ROOT/'scripts/s1_inner_profile_tool.rs'),'-C','opt-level=2','-C','panic=abort','-C','lto=thin','-o',str(D/'tool'),'-L','dependency='+str(deps),'-L','dependency='+str(audit)]
 dependencies=[]
 for name,folder in [('candid',deps),('serde',deps),('serde_json',deps)]:
  paths=list(folder.glob('lib'+name+'-*.rlib'));assert len(paths)==1
  helper+=['--extern',name+'='+str(paths[0])];dependencies+=paths
 subprocess.run(helper,cwd=ROOT,check=True)
 importer=['rustc','--edition=2021',str(ROOT/'scripts/s1_inner_profile_tool.rs'),'--cfg','imports_only','-C','opt-level=2','-o',str(D/'imports-tool'),'-L','dependency='+str(audit)]
 for name in ['wasmparser','serde_json']:
  paths=list(audit.glob('lib'+name+'-*.rlib'));assert len(paths)==1
  importer+=['--extern',name+'='+str(paths[0])];dependencies+=paths
 subprocess.run(importer,cwd=ROOT,check=True)
 source=ROOT/'scripts/s1_wide_bench/src'
 lib=(source/'lib.rs').read_text().replace('assert!(method<=3)','assert!(method==3||method==4)').replace('let out=if method==3 {','let out=if method>=3 {').replace('project_wide(&q,prepared.as_ref().unwrap(),&f.scales[..rows],rows)','project_wide(&q,prepared.as_ref().unwrap(),&f.scales[..rows],rows,method==4)')
 lib=lib.replace('pub struct Measurement {','pub struct Measurement {pub inner_profile:Vec<u64>,').replace('Measurement {digest:','Measurement {inner_profile:crate::inner_counts(),digest:')
 lib=lib.replace('let begin=ic_cdk::api::performance_counter(0);\n let x:', 'crate::reset_inner();let begin=ic_cdk::api::performance_counter(0);\n let x:')
 lib+='''
thread_local! {static INNER:RefCell<[u64;4]>=const{RefCell::new([0;4])};}
fn reset_inner(){INNER.with(|p|*p.borrow_mut()=[0;4]);}
fn add_inner(v:&[u64]){INNER.with(|p|{let mut p=p.borrow_mut();for i in 0..4{p[i]+=v[i];}});}
fn inner_counts()->Vec<u64>{INNER.with(|p|p.borrow().to_vec())}
#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}
'''
 (D/'src/lib.rs').write_text(lib)
 exact=(source/'exact.rs').read_text().replace('rows:usize)->Result<Vec<f32>> {\n  #[cfg(target_arch="wasm32")]if rows%crate::kernel::W','rows:usize,profile:bool)->Result<Vec<f32>> {\n  #[cfg(target_arch="wasm32")]if rows%crate::kernel::W')
 exact=exact.replace('self.simd_wide(q,a,sw,rows)','self.simd_wide(q,a,sw,rows,profile)').replace('unsafe fn simd_wide(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)','unsafe fn simd_wide(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize,profile:bool)')
 exact=exact.replace('vec![[0f32;crate::kernel::W];q.rows()]','vec![0f32;q.rows()*crate::kernel::W+8]')
 exact=exact.replace('for b in 0..cols/256{crate::kernel::wide(', 'let kernel=if profile{crate::kernel::profile}else{crate::kernel::wide};for b in 0..cols/256{kernel(')
 split=exact.index('pub fn project_wide');exact=exact[:split]+exact[split:].replace('copy_from_slice(&sums[t]);','copy_from_slice(&sums[t*crate::kernel::W..(t+1)*crate::kernel::W]);')
 anchor='for t in 0..q.rows(){out[t*rows+r..t*rows+r+crate::kernel::W]'
 assert exact.count(anchor)==1
 exact=exact.replace(anchor,'if profile {crate::add_inner(core::slice::from_raw_parts(sums.as_ptr().add(q.rows()*crate::kernel::W).cast::<u64>(),4));}\n   '+anchor)
 (D/'src/exact.rs').write_text(exact)
 kernel=(source/'kernel.rs').read_text();stub=kernel[kernel.index('#[export_name="__imajev_s1_wide_accumulate"]'):].replace('__imajev_s1_wide_accumulate','__imajev_s1_profile_accumulate').replace('fn wide(', 'fn profile(').replace('marker|0x7fc00000','(marker^512)|0x7fc00000').replace('0..n*W','0..n*W+8')
 (D/'src/kernel.rs').write_text(kernel+'\n'+stub)
 old=json.loads((ROOT/'artifacts/s1_wide/raw128/report.json').read_text());cmd=old['command'][:];cmd[cmd.index('--edition=2021')+1]=str(D/'src/lib.rs');cmd[cmd.index('-o')+1]=str(D/'raw.wasm')
 assert all(sha(ROOT/p)==h for p,h in old['dependency_hashes'].items())
 with (D/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**old['explicit_env']),stdout=log,stderr=log,check=True)
 imports=json.loads(subprocess.check_output([str(D/'imports-tool'),'imports',str(D/'raw.wasm')],text=True));(D/'imports.json').write_text(json.dumps(imports,indent=2)+'\n')
 counter=next(i for i,v in enumerate(imports)if v['module']=='ic0'and v['name']=='performance_counter');assert imports[counter]['params']==['i32'] and imports[counter]['results']==['i64']
 header='(module\n'+''.join('(import '+json.dumps(v['module'])+' '+json.dumps(v['name'])+' (func'+(' (param '+' '.join(v['params'])+')'if v['params']else'')+(' (result '+' '.join(v['results'])+')'if v['results']else'')+'))\n'for v in imports)
 control=ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat';wat=control.read_text()
 def begin():return f'(local.set $clock (call {counter} (i32.const 0)))\n'
 def end(slot):return f'(i64.store offset={slot*8} (local.get $metrics) (i64.add (i64.load offset={slot*8} (local.get $metrics)) (i64.sub (call {counter} (i32.const 0)) (local.get $clock))))\n'
 first=wat.index('(local.set $row0');pairs=wat.index('(br_if $pairs_done');pairs=wat.index('\n',pairs)+1;tail=wat.index('(if (i32.lt_u (local.get $t) (local.get $n)) (then');tail=wat.index('\n',tail)+1
 regions=[(first,wat.index('(local.set $t (i32.const 2))'),0),(pairs,wat.index('(local.set $t (i32.add (local.get $t) (i32.const 2)))',pairs),1),(tail,len(wat)-len('))\n)\n))\n'),2)]
 for lo,hi,slot in reversed(regions):
  body=wat[lo:hi];points=[]
  for j in range(32):
   start=body.index(f'(local.set $row0 (i32.add (local.get $wp0) (i32.mul (local.get $cols) (i32.const {j}))))')if slot==0 else(body.index('(local.set $qp')if j==0 else body.index(f'local.get $x0_0\nlocal.get $w0_{j}_0'))
   output=body.index(f'local.get $yp\nv128.load offset={j*16}',start)
   points.append((start,output))
  for j in range(31,-1,-1):
   start,output=points[j];stop=points[j+1][0]if j<31 else len(body)
   body=body[:start]+begin()+body[start:output]+end(slot)+begin()+body[output:stop]+end(3)+body[stop:]
  wat=wat[:lo]+body+wat[hi:]
 wat=wat.replace('(module (func',header+'(func',1).replace('__imajev_s1_wide_accumulate','__imajev_s1_profile_accumulate')
 anchor='(local.set $sw0';pos=wat.index(anchor);wat=wat[:pos]+'(local $clock i64) (local $metrics i32)\n(local.set $metrics (i32.add (local.get $sums) (i32.mul (local.get $n) (i32.const 512))))\n'+wat[pos:]
 (D/'kernel.wat').write_text(wat)
 patches=[];previous=D/'raw.wasm'
 for file,symbol,name in [(control,'__imajev_s1_wide_accumulate','control.wasm'),(D/'kernel.wat','__imajev_s1_profile_accumulate','diagnostic.wasm')]:
  output=D/name;row=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(file),str(output),symbol],text=True));assert row['wasmparser_validation'];patches.append(row);previous=output
 files=list((D/'src').glob('*.rs'))+[Path(__file__),ROOT/'scripts/s1_inner_profile_tool.rs',control,D/'kernel.wat',D/'imports.json']
 report=dict(wasm_sha256=sha(previous),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes=old['dependency_hashes'],helper_dependency_hashes={str(p.relative_to(ROOT)):sha(p)for p in dependencies},command=cmd,helper_command=helper,patches=patches,scope=__doc__,slots=['first_pair_dot_and_weight_prepare','complete_pair_dot_and_input_load','odd_tail_dot_and_input_load','integer_reconstruction_and_scale_store'])
 (D/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['wasm_sha256'])
if __name__=='__main__':main()
