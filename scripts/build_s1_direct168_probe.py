#!/usr/bin/env python3
"""Compare identical adaptive168 against direct first-block initialized168 output."""
from pathlib import Path
import hashlib,json,re,os,subprocess
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-direct168-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=False);old=ROOT/'artifacts/s1-stack-store-v1/build';r=json.loads((old/'report.json').read_text())
 for hashes in [r['source_hashes'],r['dependency_hashes']]:assert all(sha(ROOT/p)==h for p,h in hashes.items())
 helper=ROOT/'scripts/build_s1_direct_output_probe.py';ns=dict(__file__=str(helper),__name__='direct_helper');exec(compile(helper.read_text(),str(helper),'exec'),ns)
 base=(old/'kernel168-store.wat').read_text();paths=[];sites={}
 for seed in [False,True]:
  text=base
  if seed:
   text,n=re.subn(r'local.get \$yp\nlocal.get \$yp\nv128.load offset=\d+','local.get $yp\nv128.const i32x4 0 0 0 0',text);assert n==210
  text,count=ns['direct'](text,168);symbol='__imajev_s1_168_directseed_accumulate'if seed else'__imajev_s1_168_direct_accumulate';text=text.replace('__imajev_s1_168_store_accumulate',symbol);p=D/('direct168-seed.wat'if seed else'direct168.wat');p.write_text(text);paths.append((p,symbol));sites[symbol]=count
 src=D/'src';src.mkdir()
 for p in(old/'src').glob('*.rs'):(src/p.name).write_bytes(p.read_bytes())
 p=src/'exact.rs';s=p.read_text();a=s.index('pub fn project_store(');b=s.index('pub fn project168(',a);body=s[a:b].replace('project_store','project_direct168')
 anchor=' let mut sums=vec![0.;q.rows()*tile];';assert body.count(anchor)==1
 direct=' if tile==168 && width==tile {for b in 0..cols/256 {(if b==0{crate::kernel::wide168_direct_seed}else{crate::kernel::wide168_direct})(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,b*256,q.scales().as_ptr().add(b),rows,swp,out.as_mut_ptr().add(r),q.rows());}r+=width;continue;}\n';body=body.replace(anchor,direct+anchor);p.write_text(s[:a]+body+s[a:])
 p=src/'kernel.rs';s=p.read_text();a=s.index('#[export_name="__imajev_s1_168_store_accumulate"]');stub=s[a:];assert stub.count('fn wide168_store(')==1
 for seed in [False,True]:
  name='wide168_direct_seed'if seed else'wide168_direct';symbol='__imajev_s1_168_directseed_accumulate'if seed else'__imajev_s1_168_direct_accumulate';s+='\n'+stub.replace('__imajev_s1_168_store_accumulate',symbol).replace('fn wide168_store(','fn '+name+'(').replace('(marker^168)','(marker^'+str(16800+int(seed))+')')
 p.write_text(s)
 p=src/'lib.rs';s=p.read_text().replace('project_store(&q,','project_direct168(&q,',1).replace('project168(&q,','project_store(&q,',1);p.write_text(s)
 paths=[(ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat','__imajev_s1_wide_accumulate'),(ROOT/'artifacts/s1-adaptive160-v1/build/kernel.wat','__imajev_s1_160_accumulate'),(ROOT/'artifacts/s1-adaptive160-v1/build/kernel32.wat','__imajev_s1_32_accumulate'),(ROOT/'artifacts/s1-adaptive168-v1/build/kernel168.wat','__imajev_s1_168_accumulate'),(old/'kernel168-store.wat','__imajev_s1_168_store_accumulate')]+paths
 cmd=r['command'][:];cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs');cmd[cmd.index('-o')+1]=str(D/'raw.wasm')
 with(D/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**r['explicit_env']),check=True,stdout=log,stderr=log)
 previous=D/'raw.wasm';patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique';patches=[]
 for i,(wat,symbol)in enumerate(paths):
  out=D/('diagnostic.wasm'if i==len(paths)-1 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(out),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=out
 files=[Path(__file__),helper,old/'report.json']+list(src.glob('*.rs'))+[p for p,_ in paths];result=dict(wasm_sha256=sha(previous),command=cmd,explicit_env=r['explicit_env'],patches=patches,row1_store_sites=sites,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes=r['dependency_hashes'],scope='Only168 width uses direct strided output and first-block positive-zero add;128/32 tails retain same baseline. Component only.');(D/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(module=result['wasm_sha256'],sites=sites)))
if __name__=='__main__':main()
