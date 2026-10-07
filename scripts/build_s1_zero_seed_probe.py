#!/usr/bin/env python3
"""Initialize every output lane in K256 block0, with the original positive-zero F32 add."""
from pathlib import Path
import hashlib,json,re,os,subprocess
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-zero-seed-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=False);old=ROOT/'artifacts/s1-stack-store-all-v1/build';r=json.loads((old/'report.json').read_text())
 for hashes in [r['source_hashes'],r['dependency_hashes']]:assert all(sha(ROOT/p)==h for p,h in hashes.items())
 src=D/'src';src.mkdir()
 for p in(old/'src').glob('*.rs'):(src/p.name).write_bytes(p.read_bytes())
 p=src/'exact.rs';s=p.read_text();a=s.index('pub fn project_store(');b=s.index('pub fn project160(',a);body=s[a:b].replace('project_store','project_seed')
 anchor='let mut sums=vec![0.;q.rows()*tile];';assert body.count(anchor)==1
 body=body.replace(anchor,'let count=q.rows()*tile;let mut sums:Vec<core::mem::MaybeUninit<f32>>=Vec::with_capacity(count);sums.set_len(count);')
 body=body.replace('sums.as_mut_ptr(),','sums.as_mut_ptr().cast::<f32>(),')
 # Select an initializing export only for the first block. Later blocks read fully initialized sums.
 for fn in ['wide160','wide32','wide']:
  body=body.replace('crate::store_kernel::'+fn+'(', '(if b==0{crate::seed_kernel::'+fn+'}else{crate::store_kernel::'+fn+'})(')
 anchor=' for t in 0..q.rows(){out[t*rows+r..';assert body.count(anchor)==1
 body=body.replace(anchor,' let sums=core::mem::ManuallyDrop::new(sums);let sums=Vec::from_raw_parts(sums.as_ptr()as *mut f32,count,sums.capacity());\n'+anchor)
 p.write_text(s[:a]+body+s[a:])
 (src/'seed_kernel.rs').write_text((src/'store_kernel.rs').read_text().replace('__imajev_store_','__imajev_seed_'))
 p=src/'lib.rs';s=p.read_text().replace('project_store(&q,','project_seed(&q,',1).replace('project160(&q,','project_store(&q,',1);s+='\n#[cfg(target_arch="wasm32")]mod seed_kernel;\n';p.write_text(s)
 paths=[];sites={}
 for i in range(3):
  oldwat=old/f'kernel-store-{i}.wat';text=oldwat.read_text();symbol=re.search(r'\(export "([^"]+)"',text)[1];paths.append((oldwat,symbol))
  text,n=re.subn(r'local.get \$yp\nlocal.get \$yp\nv128.load offset=\d+', 'local.get $yp\nv128.const i32x4 0 0 0 0',text);assert n==[160,200,40][i],n
  new=D/f'kernel-seed-{i}.wat';new.write_text(text.replace('__imajev_store_','__imajev_seed_'));paths.append((new,symbol.replace('__imajev_store_','__imajev_seed_')));sites[symbol]=n
 # Legacy exports are retained and patched even though the method4/5 comparison never calls them.
 base=ROOT/'artifacts/s1-adaptive160-v1/build';paths.extend([(ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat','__imajev_s1_wide_accumulate'),(base/'kernel.wat','__imajev_s1_160_accumulate'),(base/'kernel32.wat','__imajev_s1_32_accumulate')])
 cmd=r['command'][:];cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs');cmd[cmd.index('-o')+1]=str(D/'raw.wasm')
 with(D/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**r['explicit_env']),check=True,stdout=log,stderr=log)
 previous=D/'raw.wasm';patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique';patches=[]
 for i,(wat,symbol)in enumerate(paths):
  out=D/('diagnostic.wasm'if i==len(paths)-1 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(out),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=out
 files=[Path(__file__),old/'report.json']+list(src.glob('*.rs'))+[p for p,_ in paths];result=dict(wasm_sha256=sha(previous),command=cmd,explicit_env=r['explicit_env'],patches=patches,initialized_store_sites=sites,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes=r['dependency_hashes'],scope='Every output lane initialized by block0, including odd token tail and padded output rows. Positive-zero add and F32 order preserved; later blocks use original store kernels. Component only.');(D/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(module=result['wasm_sha256'],sites=sites)))
if __name__=='__main__':main()
