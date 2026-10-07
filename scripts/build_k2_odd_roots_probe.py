#!/usr/bin/env python3
"""Patch only eight current rank7 control bodies in the completed diagnostic module."""
from pathlib import Path
import json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/k2-odd-roots-probe-v1';d.mkdir(exist_ok=False);b=d/'build';b.mkdir();old=ROOT/'artifacts/s2-k2-probe-v1';k=ROOT/'artifacts/k2-odd-roots-kernels-v1';ob=json.loads((old/'build/report.json').read_text());kr=json.loads((k/'report.json').read_text());audit=json.loads((k/'integer-audit.json').read_text());assert audit['all_integer_dots_equal'];prior=old/'build/diagnostic.wasm';assert sha(prior)==ob['wasm_sha256']
 for report in [ob,kr,audit]:
  for p,h in report['source_hashes'].items():assert sha(ROOT/p)==h
 for p,h in ob['dependency_hashes'].items():assert sha(ROOT/p)==h
 patches=[]
 for i,item in enumerate(kr['kernels']):
  target=b/('diagnostic.wasm'if i==7 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(prior),str(ROOT/item['path']),str(target),item['symbol']],text=True));assert patch['wasmparser_validation'];patches.append(patch);prior=target
 files=[Path(__file__),old/'build/report.json',old/'build/diagnostic.wasm',k/'report.json',k/'integer-audit.json']+[ROOT/p for p in ob['source_hashes']]+[ROOT/p for p in kr['source_hashes']];r=dict(wasm_sha256=sha(prior),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},dependency_hashes=ob['dependency_hashes'],patches=patches,base_module=ob['wasm_sha256'],base_diagnostic_rust_and_runtime_unchanged=True,comparison='Same method3 against previously completed method3 counter/native evidence. Method4 rank49 is unused.')
 (b/'report.json').write_text(json.dumps(r,indent=2)+'\n');(d/'source-audit.json').write_text(json.dumps(dict(complete=True,eight_unique_validated_patches=True,all_rust_and_runtime_dependencies_byte_unchanged=True,source_hashes=r['source_hashes']),indent=2)+'\n');print(json.dumps(dict(module=r['wasm_sha256'],patches=8)))
if __name__=='__main__':main()
