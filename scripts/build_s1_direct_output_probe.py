#!/usr/bin/env python3
"""Direct strided INT8 output, preserving block0 positive-zero add and F32 order."""
from pathlib import Path
import hashlib,json,re,os,subprocess
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-direct-output-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def direct(text,tile):
 # Existing ABI stride becomes output-row stride; input scale stride derives from cols/256.
 text=text.replace('(local.get $stride)','(local.get $input_stride)')
 at=text.index('(local ');text=text[:at]+'(local $input_stride i32) (local $output_bytes i32) (local $yp1 i32)\n'+text[at:]
 at=text.index('(local.set $sw0');text=text[:at]+'(local.set $input_stride (i32.shr_u (local.get $cols) (i32.const 8)))\n(local.set $output_bytes (i32.shl (local.get $stride) (i32.const 2)))\n'+text[at:]
 old=f'(local.set $yp (i32.add (local.get $sums) (i32.mul (local.get $t) (i32.const {tile*4}))))'
 if text.count(old)!=3:
  old=f'(local.set $yp (i32.add (local.get $sums) (i32.shl (local.get $t) (i32.const {(tile*4).bit_length()-1}))))'
 assert text.count(old)==3
 new='(local.set $yp (i32.add (local.get $sums) (i32.mul (local.get $t) (local.get $output_bytes))))\n(local.set $yp1 (i32.add (local.get $yp) (local.get $output_bytes)))';text=text.replace(old,new)
 pattern=r'local.get \$yp\n(?:(?!v128.store).)*?v128.store offset=(\d+)'
 count=0
 def rewrite(m):
  nonlocal count
  offset=int(m[1]);s=m[0]
  if offset>=tile*4:
   count+=1;s=s.replace('local.get $yp','local.get $yp1')
   s=re.sub(r'(v128\.(?:load|store) offset=)(\d+)',lambda h:h[1]+str(int(h[2])-tile*4),s)
  return s
 text=re.sub(pattern,rewrite,text,flags=re.S);assert count==tile//4*2,count
 assert text.count('(')==text.count(')');return text,count

def main():
 D.mkdir(parents=True,exist_ok=False);old=ROOT/'artifacts/s1-zero-seed-v1/build';r=json.loads((old/'report.json').read_text())
 for hashes in [r['source_hashes'],r['dependency_hashes']]:assert all(sha(ROOT/p)==h for p,h in hashes.items())
 src=D/'src';src.mkdir()
 for p in(old/'src').glob('*.rs'):(src/p.name).write_bytes(p.read_bytes())
 p=src/'exact.rs';s=p.read_text();a=s.index('pub fn project_seed(');b=s.index('pub fn project_store(',a);body=s[a:b].replace('project_seed','project_direct').replace('crate::seed_kernel::','crate::direct_seed_kernel::').replace('crate::store_kernel::','crate::direct_kernel::')
 before='let count=q.rows()*tile;let mut sums:Vec<core::mem::MaybeUninit<f32>>=Vec::with_capacity(count);sums.set_len(count);';assert body.count(before)==1;body=body.replace(before,'').replace('let width=(rows-r).min(tile);','let width=(rows-r).min(tile);assert_eq!(width,tile);')
 body=body.replace('cols/256,swp,sums.as_mut_ptr().cast::<f32>()','rows,swp,out.as_mut_ptr().add(r)')
 a1=body.index(' let sums=core::mem::ManuallyDrop::new(sums)');b1=body.index('r+=width;',a1);body=body[:a1]+' '+body[b1:]
 p.write_text(s[:a]+body+s[a:])
 for original,new,prefix in [('store_kernel.rs','direct_kernel.rs','__imajev_direct_'),('seed_kernel.rs','direct_seed_kernel.rs','__imajev_directseed_')]:
  text=(src/original).read_text();text=re.sub('__imajev_(?:store|seed)_',prefix,text);(src/new).write_text(text)
 p=src/'lib.rs';s=p.read_text().replace('project_seed(&q,','project_direct(&q,',1).replace('project_store(&q,','project_seed(&q,',1);s+='\n#[cfg(target_arch="wasm32")]mod direct_kernel;\n#[cfg(target_arch="wasm32")]mod direct_seed_kernel;\n';p.write_text(s)
 paths=[]
 base=ROOT/'artifacts/s1-stack-store-all-v1/build';orig=ROOT/'artifacts/s1-adaptive160-v1/build';counts={}
 for i,tile in enumerate([128,160,32]):
  for path,prefix in [(base/f'kernel-store-{i}.wat','__imajev_direct_'),(old/f'kernel-seed-{i}.wat','__imajev_directseed_')]:
   text=path.read_text();symbol=re.search(r'\(export "([^"]+)"',text)[1];paths.append((path,symbol));text,count=direct(text,tile);new=D/f'direct-{i}-{prefix}.wat';text=re.sub('__imajev_(?:store|seed)_',prefix,text);new.write_text(text);paths.append((new,re.search(r'\(export "([^"]+)"',text)[1]));counts[str(new.relative_to(ROOT))]=count
 paths.extend([(ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat','__imajev_s1_wide_accumulate'),(orig/'kernel.wat','__imajev_s1_160_accumulate'),(orig/'kernel32.wat','__imajev_s1_32_accumulate')])
 cmd=r['command'][:];cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs');cmd[cmd.index('-o')+1]=str(D/'raw.wasm')
 with(D/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**r['explicit_env']),check=True,stdout=log,stderr=log)
 previous=D/'raw.wasm';patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique';patches=[]
 for i,(wat,symbol)in enumerate(paths):
  out=D/('diagnostic.wasm'if i==len(paths)-1 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(out),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=out
 files=[Path(__file__),old/'report.json']+list(src.glob('*.rs'))+[p for p,_ in paths];result=dict(wasm_sha256=sha(previous),command=cmd,explicit_env=r['explicit_env'],patches=patches,row1_store_sites=counts,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes=r['dependency_hashes'],scope='Direct strided output; no tile sums allocation/copy. Positive-zero F32 add and original block arithmetic/order retained. Component only.');(D/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(module=result['wasm_sha256'],sites=counts)))
if __name__=='__main__':main()
