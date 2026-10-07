#!/usr/bin/env python3
"""Validate helper relocation, preservation, and actual patched SIMD execution."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def leb(data,p):
 value=0;shift=0
 while True:
  v=data[p];p+=1;value|=(v&127)<<shift
  if not v&128:return value,p
  shift+=7
def sections(data):
 result=[];p=8
 while p<len(data):
  ident=data[p];p+=1;size,p=leb(data,p);result.append((ident,data[p:p+size]));p+=size
 assert p==len(data);return result
def bodies(data):
 n,p=leb(data,0);out=[]
 for _ in range(n):size,p=leb(data,p);out.append(data[p:p+size]);p+=size
 assert p==len(data);return out
def main():
 d=ROOT/'artifacts/s3-streaming-shared-kernels-v1';r=json.loads((d/'execution-v2-report.json').read_text())
 for p,h in r['source_hashes'].items():assert sha(ROOT/p)==h,p
 target=d/'patched.wasm';assert not target.exists()
 command=[str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-shared'),str(d/'stub.wasm'),str(d/'256.wat'),str(target),'__imajev_s3_streaming_256']
 patch=json.loads(subprocess.check_output(command,text=True));assert patch['relocated_calls']==343 and patch['helper_results']==88 and patch['wasmparser_validation']
 original=sections((d/'stub.wasm').read_bytes());updated=sections(target.read_bytes());assert [s[0]for s in original]==[s[0]for s in updated]
 for (ident,before),(_,after)in zip(original,updated):
  if ident not in [1,3,10]:assert before==after,ident
  elif ident in [1,3]:
   count,p=leb(before,0);new,q=leb(after,0);assert new==count+1 and after[q:q+len(before)-p]==before[p:]
  else:
   old=bodies(before);new=bodies(after);assert len(new)==len(old)+1
   changed=[i for i in range(len(old))if old[i]!=new[i]];assert len(changed)==1
   assert hashlib.sha256(old[changed[0]]).hexdigest()==patch['original_body_sha256']
   assert hashlib.sha256(new[changed[0]]).hexdigest()==patch['replacement_body_sha256']
   assert hashlib.sha256(new[-1]).hexdigest()==patch['helper_body_sha256']
 runner=d/'patch-run.js';source=(d/'execution-v2/run.js').read_text();source=source.replace('memory.grow(Math.ceil(bin.length/65536)-1)','memory.grow(Math.max(0,Math.ceil(bin.length/65536)-memory.buffer.byteLength/65536))');runner.write_text(source)
 cases=[];files=[Path(__file__),d/'stub.rs',d/'stub.wasm',d/'256.wat',target,runner,d/'execution-v2-report.json']+[ROOT/p for p in r['source_hashes']]
 for case in r['cases']:
  old=d/'execution-v2'/f'c{case["cols"]}-n{case["tokens"]}.json';m=json.loads(old.read_text());m['wasm']=str(target);meta=d/f'patched-c{case["cols"]}-n{case["tokens"]}.json';meta.write_text(json.dumps(m));files.append(meta)
  output=json.loads(subprocess.check_output(['node',str(runner),str(meta)],text=True));assert output['all_memory_bits_equal'];cases.append(dict(cols=case['cols'],tokens=case['tokens'],**output));print(json.dumps(cases[-1]),flush=True)
 files+=list((ROOT/'scripts/wasm_patch_shared').glob('Cargo.*'))+[ROOT/'scripts/wasm_patch_shared/src/main.rs']
 report=dict(complete=True,patch=patch,command=command,all_other_original_sections_and_bodies_byte_identical=True,conditions=len(cases),cases=cases,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},ic_execution_verified=False,performance_verified=False,scope='Shared helper appended and calls relocated in Rust Wasm; Node full-memory results compared to independent scalar dot oracle. No IC or full paid claim.')
 (d/'patch-validation.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'patch-evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in dict.fromkeys(files+[d/'patch-validation.json']):z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,conditions=len(cases),module=sha(target))))
if __name__=='__main__':main()
