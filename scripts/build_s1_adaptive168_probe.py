#!/usr/bin/env python3
"""Compare exact adaptive160 with adaptive168 under the10000-local platform bound."""
from pathlib import Path
import hashlib,json,re,os,subprocess
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-adaptive168-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def kernel168(source):
 s=source.replace('__imajev_s1_160_accumulate','__imajev_s1_168_accumulate')
 s=re.sub(r'(?m)^(v128\.(?:load|store) offset=)(\d+)$',lambda m:m[1]+str(int(m[2])+32 if int(m[2])>=640 else int(m[2])),s)
 assert '(i32.mul (local.get $t) (i32.const 640))'in s;s=s.replace('(i32.mul (local.get $t) (i32.const 640))','(i32.mul (local.get $t) (i32.const 672))')
 declarations=[f'(local $w{m}_{j}_{k} v128)'for m in range(7)for j in [40,41]for k in range(32)]+[f'(local $sw{j} v128)'for j in [40,41]]
 at=s.index('(local.set $sw0');s=s[:at]+'\n'.join(declarations)+'\n'+s[at:]
 at=s.index('(local.set $wp0');s=s[:at]+''.join(f'(local.set $sw{j} (v128.load offset={j*16} (local.get $sw)))\n'for j in [40,41])+s[at:]
 def copy(b):
  b=re.sub(r'\$w([0-6])_(\d+)_(\d+)\b',lambda m:f'$w{m[1]}_{int(m[2])+8}_{m[3]}',b)
  b=re.sub(r'\$sw(\d+)\b',lambda m:f'$sw{int(m[1])+8}',b)
  b=re.sub(r'(?m)^(v128\.(?:load|store) offset=)(\d+)$',lambda m:m[1]+str(int(m[2])+128),b)
  return re.sub(r'(\(i32.mul \(local.get \$cols\) \(i32.const )(\d+)(\)\))',lambda m:m[1]+str(int(m[2])+8)+m[3],b)
 row=lambda j:f'(local.set $row0 (i32.add (local.get $wp0) (i32.mul (local.get $cols) (i32.const {j}))))'
 lo=s.index(row(32));hi=s.index(row(34),lo);at=s.index('(local.set $t (i32.const 2))');s=s[:at]+copy(s[lo:hi])+s[at:]
 anchor=lambda j:f'local.get $x0_0\nlocal.get $w0_{j}_0';pair=s.index('(loop $tokens');lo=s.index(anchor(32),pair);hi=s.index(anchor(34),lo);at=s.index('(local.set $t (i32.add (local.get $t) (i32.const 2)))',pair);s=s[:at]+copy(s[lo:hi])+s[at:]
 tail=s.index('(if (i32.lt_u (local.get $t) (local.get $n)) (then');lo=s.index(anchor(32),tail);hi=s.index(anchor(34),lo);footer='\n))\n)\n))\n';assert s.endswith(footer);at=len(s)-len(footer);s=s[:at]+'\n'+copy(s[lo:hi])+s[at:]
 assert s.count('(')==s.count(')') and s.count('local.get $sw41')==5;locals_count=s.count('(local ');assert locals_count<10000;return s,locals_count
def main():
 D.mkdir(parents=True,exist_ok=False);old=ROOT/'artifacts/s1-adaptive160-v1/build';r=json.loads((old/'report.json').read_text())
 for hashes in [r['source_hashes'],r['dependency_hashes']]:assert all(sha(ROOT/p)==h for p,h in hashes.items())
 wat,count=kernel168((old/'kernel.wat').read_text());(D/'kernel168.wat').write_text(wat);src=D/'src';src.mkdir()
 for p in(old/'src').glob('*.rs'):(src/p.name).write_bytes(p.read_bytes())
 p=src/'lib.rs';s=p.read_text().replace('assert!(method==3||method==4)','assert!(method==4||method==5)');anchor='let out=if method==4 {';assert s.count(anchor)==1;s=s.replace(anchor,'let out=if method==5 {f.coeff.as_ref().unwrap().project168(&q,prepared.as_ref().unwrap(),&f.scales[..rows],rows).unwrap()}else if method==4 {');p.write_text(s)
 p=src/'exact.rs';s=p.read_text();a=s.index('pub fn project160(');b=s.index(' pub fn new(',a);body=s[a:b];assert body.count('project160')==1;body=body.replace('project160','project168').replace('output160','output168')
 body=body.replace('let tile=if rows-r>=160{160}','let tile=if rows-r>=168{168}').replace('if tile==160{crate::kernel::wide160','if tile==168{crate::kernel::wide168');anchor=' #[cfg(target_arch="wasm32")] let out=unsafe{';assert body.count(anchor)==1;body=body.replace(anchor,' if rows%160==0{return self.project160(q,a,sw,rows);}\n'+anchor);p.write_text(s.replace('impl Prepared {','impl Prepared {\n'+body,1))
 p=src/'kernel.rs';s=p.read_text();a=s.index('#[export_name="__imajev_s1_160_accumulate"]');b=s.index('#[export_name="__imajev_s1_32_accumulate"]',a);stub=s[a:b].replace('__imajev_s1_160_accumulate','__imajev_s1_168_accumulate').replace('fn wide160','fn wide168').replace('n*160','n*168');p.write_text(s+'\n'+stub)
 cmd=r['command'][:];cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs');cmd[cmd.index('-o')+1]=str(D/'raw.wasm')
 with(D/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**r['explicit_env']),check=True,stdout=log,stderr=log)
 patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique';previous=D/'raw.wasm';patches=[]
 paths=[(ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat','__imajev_s1_wide_accumulate'),(old/'kernel.wat','__imajev_s1_160_accumulate'),(old/'kernel32.wat','__imajev_s1_32_accumulate'),(D/'kernel168.wat','__imajev_s1_168_accumulate')]
 for i,(wat,symbol)in enumerate(paths):
  out=D/('diagnostic.wasm'if i==3 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(out),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=out
 sources=[Path(__file__),old/'report.json',D/'kernel168.wat']+list(src.glob('*.rs'))+[p for p,_ in paths];result=dict(wasm_sha256=sha(previous),locals=count,command=cmd,explicit_env=r['explicit_env'],patches=patches,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},dependency_hashes=r['dependency_hashes'],scope='Same raw weight capacity/layout and I32/F32 order. Adaptive168/128/32, with original adaptive160 for rows divisible by160. Component performance only.');(D/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(module=result['wasm_sha256'],locals=count)))
if __name__=='__main__':main()
