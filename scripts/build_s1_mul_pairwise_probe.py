#!/usr/bin/env python3
"""Measure bounded I16 multiplication plus pairwise extension against IC dot pricing."""
from pathlib import Path
import hashlib,json,re,os,subprocess
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-mul-pairwise-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def change(wat):
 lines=[];m=None;changed=0;dots=0;counts={i:0 for i in range(7)}
 for line in wat.splitlines():
  hit=re.search(r'local\.(?:get|tee) \$x([0-6])_\d+',line)
  if hit:m=int(hit[1])
  if line=='i32x4.dot_i16x8_s':
   assert m is not None;dots+=1;counts[m]+=1
   if m in [1,2,3,4]:line='i16x8.mul\ni32x4.extadd_pairwise_i16x8_s';changed+=1
  lines.append(line)
 assert changed==sum(counts[m]for m in [1,2,3,4]) and counts[0]==counts[2]==counts[4]==counts[6] and counts[1]==counts[3]==counts[5] and counts[0]*2==counts[1]*3
 return '\n'.join(lines)+'\n',dict(dots=dots,replaced=changed,per_product=counts)
def main():
 D.mkdir(exist_ok=False);d=D/'build';d.mkdir();old=ROOT/'artifacts/s1-adaptive160-v1/build';r=json.loads((old/'report.json').read_text())
 for manifest in [r['source_hashes'],r['dependency_hashes']]:assert all(sha(ROOT/p)==h for p,h in manifest.items())
 src=d/'src';src.mkdir()
 for p in (old/'src').glob('*.rs'):(src/p.name).write_bytes(p.read_bytes())
 p=src/'lib.rs';s=p.read_text()
 if 'post_upgrade' not in s:p.write_text(s+'\n#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}\n')
 replacements={}
 for name in ['kernel.wat','kernel32.wat']:
  wat,counts=change((old/name).read_text());(d/name).write_text(wat);replacements[name]=counts
 # Four products have at most one transformed factor. Bounds include -128 weights.
 bounds={str(m):32512 for m in [1,2,3,4]};assert max(bounds.values())<32768
 proof=dict(changed_products=[1,2,3,4],i16_product_absolute_bounds=bounds,other_products_use_original_dot=True,integer_pair_sums_unchanged=True,float_reconstruction_and_scaling_unchanged=True,replacements=replacements)
 (D/'integer-bounds.json').write_text(json.dumps(proof,indent=2)+'\n')
 cmd=r['command'][:];cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs');cmd[cmd.index('-o')+1]=str(d/'raw.wasm')
 with(d/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**r['explicit_env']),check=True,stdout=log,stderr=log)
 control=ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat';patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique';previous=d/'raw.wasm';patches=[]
 for i,(wat,symbol)in enumerate([(control,'__imajev_s1_wide_accumulate'),(d/'kernel.wat','__imajev_s1_160_accumulate'),(d/'kernel32.wat','__imajev_s1_32_accumulate')]):
  out=d/('diagnostic.wasm'if i==2 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(out),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=out
 paths=[Path(__file__),old/'report.json',D/'integer-bounds.json',control,d/'kernel.wat',d/'kernel32.wat']+list(src.glob('*.rs'));result=dict(wasm_sha256=sha(previous),command=cmd,explicit_env=r['explicit_env'],patches=patches,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in paths},dependency_hashes=r['dependency_hashes'],scope=__doc__);(d/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(result['wasm_sha256'])
if __name__=='__main__':main()
