#!/usr/bin/env python3
"""Keep store address on the Wasm stack, preserving every arithmetic operation."""
from pathlib import Path
import hashlib,json,re,os,subprocess
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-stack-store-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def change(s):
 pattern=r'local.get \$yp\nv128.load offset=(\d+)\n((?:(?!v128.store).)*?f32x4.add)\nlocal.set \$value\nlocal.get \$yp\nlocal.get \$value\nv128.store offset=\1'
 s,count=re.subn(pattern,lambda m:'local.get $yp\nlocal.get $yp\nv128.load offset='+m[1]+'\n'+m[2]+'\nv128.store offset='+m[1],s,flags=re.S)
 assert count==210,count
 return s,count

def main():
 D.mkdir(parents=True,exist_ok=False);old=ROOT/'artifacts/s1-adaptive168-v1/build';r=json.loads((old/'report.json').read_text())
 for hashes in [r['source_hashes'],r['dependency_hashes']]:assert all(sha(ROOT/p)==h for p,h in hashes.items())
 wat,count=change((old/'kernel168.wat').read_text());wat=wat.replace('__imajev_s1_168_accumulate','__imajev_s1_168_store_accumulate');(D/'kernel168-store.wat').write_text(wat)
 src=D/'src';src.mkdir()
 for p in(old/'src').glob('*.rs'):(src/p.name).write_bytes(p.read_bytes())
 p=src/'exact.rs';s=p.read_text();a=s.index('pub fn project168(');b=s.index('pub fn project160(',a);body=s[a:b];assert body.count('project168')==1
 s=s[:a]+body.replace('project168','project_store').replace('crate::kernel::wide168','crate::kernel::wide168_store')+s[a:];p.write_text(s)
 p=src/'kernel.rs';s=p.read_text();a=s.index('#[export_name="__imajev_s1_168_accumulate"]');stub=s[a:].replace('__imajev_s1_168_accumulate','__imajev_s1_168_store_accumulate').replace('fn wide168(','fn wide168_store(');assert stub.count('fn wide168_store(')==1;p.write_text(s+'\n'+stub)
 p=src/'lib.rs';s=p.read_text().replace('project168(&q,','project_store(&q,',1).replace('project160(&q,','project168(&q,',1);p.write_text(s)
 cmd=r['command'][:];cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs');cmd[cmd.index('-o')+1]=str(D/'raw.wasm')
 with(D/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**r['explicit_env']),check=True,stdout=log,stderr=log)
 patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique';previous=D/'raw.wasm';patches=[]
 paths=[(ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat','__imajev_s1_wide_accumulate'),(ROOT/'artifacts/s1-adaptive160-v1/build/kernel.wat','__imajev_s1_160_accumulate'),(ROOT/'artifacts/s1-adaptive160-v1/build/kernel32.wat','__imajev_s1_32_accumulate'),(old/'kernel168.wat','__imajev_s1_168_accumulate'),(D/'kernel168-store.wat','__imajev_s1_168_store_accumulate')]
 for i,(wat,symbol)in enumerate(paths):
  out=D/('diagnostic.wasm'if i==4 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(out),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=out
 sources=[Path(__file__),old/'report.json']+list(src.glob('*.rs'))+[p for p,_ in paths]
 result=dict(wasm_sha256=sha(previous),command=cmd,explicit_env=r['explicit_env'],patches=patches,store_sites=count,instructions_saved_per_site=2,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},dependency_hashes=r['dependency_hashes'],scope='Same adaptive168 and unchanged arithmetic order, store address/value remain on operand stack. Component only.');(D/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(module=result['wasm_sha256'],store_sites=count)))
if __name__=='__main__':main()
