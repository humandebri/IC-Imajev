#!/usr/bin/env python3
"""Wasm-validate generated rank49 K2 kernels against distinct matching ABI exports."""
from pathlib import Path
import json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-k2-tail-dispatch-kernels-v1';r=json.loads((d/'report.json').read_text());assert all(sha(ROOT/p)==h for p,h in r['source_hashes'].items())
 source='#![no_std]\n#[panic_handler]fn panic(_: &core::panic::PanicInfo)->!{loop{}}\n'
 for i,k in enumerate(r['kernels']):
  tile=int(Path(k['path']).stem.split('_')[0]);source+=f'#[export_name="{k["symbol"]}"]#[inline(never)]pub unsafe extern "C" fn stub{i}(q:usize,w:usize,cols:usize,start:usize,sx:usize,stride:usize,sw:usize,out:*mut f32,n:usize){{let marker=core::hint::black_box(q^w^cols^start^sx^stride^sw^out as usize^n)as u32;for p in 0..n*{tile}{{core::ptr::write_volatile(out.add(p),f32::from_bits((marker^{i+1})|0x7fc00000));}}}}\n'
 (d/'stubs.rs').write_text(source);command=['rustc','--edition=2021','--crate-type','cdylib','--target','wasm32-unknown-unknown','-C','panic=abort','-C','opt-level=3',str(d/'stubs.rs'),'-o',str(d/'raw.wasm')];subprocess.run(command,check=True)
 previous=d/'raw.wasm';patches=[]
 for i,k in enumerate(r['kernels']):
  target=d/f'validated-{i}.wasm';patch=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(ROOT/k['path']),str(target),k['symbol']],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=target
 files=[Path(__file__),d/'report.json',d/'stubs.rs',previous]+[ROOT/p for p in r['source_hashes']]
 result=dict(all4_wasm_validated=True,module=sha(previous),patches=patches,command=command,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Wasm validation only; no output equivalence or performance claim.')
 (d/'wasm-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(all4_wasm_validated=True,module=result['module'])))
if __name__=='__main__':main()
