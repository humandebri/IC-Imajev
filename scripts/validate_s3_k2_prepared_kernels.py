#!/usr/bin/env python3
"""Compile matching ABI stubs and replace them with the two new K2 kernels."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s3-k2-prepared-kernels-v1';r=json.loads((d/'report.json').read_text());assert all(sha(ROOT/p)==v for p,v in r['source_hashes'].items())
 source='#![no_std]\n#[panic_handler]fn panic(_: &core::panic::PanicInfo)->!{loop{}}\n'
 for i,k in enumerate(r['kernels']):source+=f'#[export_name="{k["symbol"]}"]#[inline(never)]pub unsafe extern "C" fn stub{i}(q:usize,w:usize,cols:usize,start:usize,sx:usize,stride:usize,sw:usize,out:*mut f32,n:usize){{let marker=core::hint::black_box(q^w^cols^start^sx^stride^sw^out as usize^n)as u32;for p in 0..n*32{{core::ptr::write_volatile(out.add(p),f32::from_bits((marker^{i+1})|0x7fc00000));}}}}\n'
 (d/'stubs.rs').write_text(source)
 command=['rustc','--edition=2021','--crate-type','cdylib','--target','wasm32-unknown-unknown','-C','panic=abort','-C','opt-level=3',str(d/'stubs.rs'),'-o',str(d/'raw.wasm')];subprocess.run(command,check=True)
 previous=d/'raw.wasm';patches=[]
 for i,k in enumerate(r['kernels']):
  target=d/f'validated-{i}.wasm';patch=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(ROOT/k['path']),str(target),k['symbol']],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=target
 files=[Path(__file__),d/'report.json',d/'stubs.rs',previous]+[ROOT/p for p in r['source_hashes']]
 result=dict(all2_wasm_validated=True,module=sha(previous),patches=patches,command=command,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},scope='Wasm validation only; no executed output equivalence or timing claim.')
 (d/'wasm-validation.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'validation-evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in dict.fromkeys(files+[d/'wasm-validation.json']):z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(all2_wasm_validated=True,module=result['module'])))
if __name__=='__main__':main()
