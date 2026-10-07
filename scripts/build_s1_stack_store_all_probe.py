#!/usr/bin/env python3
"""Compare original adaptive160 with identical-arithmetic stack stores for160/128/32."""
from pathlib import Path
import hashlib,json,re,os,subprocess
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-stack-store-all-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def change(s):
 pattern=r'local.get \$yp\nv128.load offset=(\d+)\n((?:(?!v128.store).)*?f32x4.add)\nlocal.set \$value\nlocal.get \$yp\nlocal.get \$value\nv128.store offset=\1'
 s,n=re.subn(pattern,lambda m:'local.get $yp\nlocal.get $yp\nv128.load offset='+m[1]+'\n'+m[2]+'\nv128.store offset='+m[1],s,flags=re.S)
 assert n>0
 return s,n

def main():
 D.mkdir(parents=True,exist_ok=False);old=ROOT/'artifacts/s1-adaptive160-v1/build';r=json.loads((old/'report.json').read_text())
 for hashes in [r['source_hashes'],r['dependency_hashes']]:assert all(sha(ROOT/p)==h for p,h in hashes.items())
 src=D/'src';src.mkdir()
 for p in(old/'src').glob('*.rs'):(src/p.name).write_bytes(p.read_bytes())
 p=src/'exact.rs';s=p.read_text();a=s.index('pub fn project160(');b=s.index(' pub fn new(',a);body=s[a:b];assert body.count('project160')==1
 s=s[:a]+body.replace('project160','project_store').replace('crate::kernel::','crate::store_kernel::')+s[a:];p.write_text(s)
 (src/'store_kernel.rs').write_text((src/'kernel.rs').read_text().replace('__imajev_','__imajev_store_'))
 p=src/'lib.rs';s=p.read_text().replace('assert!(method==3||method==4)','assert!(method==4||method==5)');anchor='let out=if method==4 {';assert s.count(anchor)==1;s=s.replace(anchor,'let out=if method==5 {f.coeff.as_ref().unwrap().project_store(&q,prepared.as_ref().unwrap(),&f.scales[..rows],rows).unwrap()}else if method==4 {');s+='\n#[cfg(target_arch="wasm32")]mod store_kernel;\n';p.write_text(s)
 cmd=r['command'][:];cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs');cmd[cmd.index('-o')+1]=str(D/'raw.wasm')
 with(D/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**r['explicit_env']),check=True,stdout=log,stderr=log)
 paths=[(ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat','__imajev_s1_wide_accumulate'),(old/'kernel.wat','__imajev_s1_160_accumulate'),(old/'kernel32.wat','__imajev_s1_32_accumulate')];sites={}
 for i,(wat,symbol)in enumerate(paths[:]):
  text,count=change(wat.read_text());new=D/f'kernel-store-{i}.wat';new.write_text(text.replace('__imajev_','__imajev_store_'));paths.append((new,symbol.replace('__imajev_','__imajev_store_')));sites[symbol]=count
 previous=D/'raw.wasm';patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique';patches=[]
 for i,(wat,symbol)in enumerate(paths):
  out=D/('diagnostic.wasm'if i==5 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(out),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=out
 sources=[Path(__file__),old/'report.json']+list(src.glob('*.rs'))+[p for p,_ in paths]
 result=dict(wasm_sha256=sha(previous),command=cmd,explicit_env=r['explicit_env'],patches=patches,store_sites=sites,instructions_saved_per_site=2,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},dependency_hashes=r['dependency_hashes'],scope='Adaptive160/128/32, same raw weights and arithmetic order. Store address remains on operand stack; component only.');(D/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(module=result['wasm_sha256'],store_sites=sites)))
if __name__=='__main__':main()
