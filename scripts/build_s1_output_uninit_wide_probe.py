#!/usr/bin/env python3
"""Compare direct168 output to identical kernels without final output zero-fill."""
from pathlib import Path
import hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-output-uninit-wide-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/s1-direct-output-v1/build';r=json.loads((old/'report.json').read_text())
 for k in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in r[k].items())
 D.mkdir(parents=True,exist_ok=False);src=D/'src';src.mkdir()
 for p in(old/'src').glob('*.rs'):(src/p.name).write_bytes(p.read_bytes())
 p=src/'exact.rs';s=p.read_text();a=s.index('pub fn project_direct(');b=s.index('pub fn project_seed(',a);body=s[a:b].replace('project_direct','project_uninit')
 anchor='let mut out=vec![0.;q.rows()*rows];';assert body.count(anchor)==1
 body=body.replace(anchor,'let count=q.rows()*rows;let mut out:Vec<core::mem::MaybeUninit<f32>>=Vec::with_capacity(count);out.set_len(count);let out_ptr=out.as_mut_ptr().cast::<f32>();')
 body=body.replace('swp,out.as_mut_ptr().add(r),q.rows()','swp,out_ptr.add(r),q.rows()')
 anchor='}out};';assert body.count(anchor)==1
 body=body.replace(anchor,'}let out=core::mem::ManuallyDrop::new(out);Vec::from_raw_parts(out_ptr,out.len(),out.capacity())};')
 # Bounds reject empty tokens, rows and K. Every output is written before conversion.
 body=body.replace('if q.rows()==0||','if self.cols<256||self.cols%256!=0||q.rows()==0||',1)
 p.write_text(s[:a]+body+s[a:])
 p=src/'lib.rs';s=p.read_text().replace('project_direct(&q,','project_uninit(&q,',1).replace('project_seed(&q,','project_direct(&q,',1);p.write_text(s)
 paths=[]
 for patch in r['patches']:
  matches=[ROOT/p for p,h in r['source_hashes'].items()if p.endswith('.wat') and h==patch['source_sha256']]
  assert matches,patch['export'];paths.append((matches[0],patch['export']))
 cmd=r['command'][:];cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs');cmd[cmd.index('-o')+1]=str(D/'raw.wasm')
 with(D/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**r['explicit_env']),check=True,stdout=log,stderr=log)
 previous=D/'raw.wasm';patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique';patches=[]
 for i,(wat,symbol)in enumerate(paths):
  out=D/('diagnostic.wasm'if i==len(paths)-1 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(out),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=out
 assert len({p['function_index']for p in patches})==15
 files=[Path(__file__),old/'report.json']+list(src.glob('*.rs'))+[p for p,_ in paths];result=dict(wasm_sha256=sha(previous),command=cmd,explicit_env=r['explicit_env'],patches=patches,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes=r['dependency_hashes'],scope='Only final output zero-fill removed from direct160/128/32 branch. No reference to uninitialized F32; all output coverage before conversion. Component only.');(D/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(module=result['wasm_sha256'])))
if __name__=='__main__':main()
