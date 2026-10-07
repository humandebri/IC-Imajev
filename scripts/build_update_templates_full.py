#!/usr/bin/env python3
"""Reuse the proven query32 runtime and unchanged variable-prefix update scheduler."""
import hashlib,json,os,shutil,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'artifacts/voting-template-prefix-v1/full-build'
D=ROOT/'artifacts/update-templates-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=False)
 base=json.loads((B/'report.json').read_text())
 assert sha(B/'full.wasm')==base['wasm_sha256']
 for key in ['source_hashes','dependency_hashes']:
  assert all(sha(ROOT/p)==h for p,h in base[key].items())
 old=ROOT/'artifacts/update-short-v1/build'
 old_manifest=json.loads((old/'provenance.json').read_text())
 assert sha(old/'update_inference.rs')==old_manifest['source_hashes']['update_inference.rs']
 for p in B.glob('*.rs'):shutil.copyfile(p,D/p.name)
 shutil.copyfile(old/'update_inference.rs',D/'update_inference.rs')
 command=base['canister_command'][:]
 command[command.index('--edition=2021')+1]=str(D/'lib.rs')
 command[command.index('-o')+1]=str(D/'raw.wasm')
 assert 'feature="experimental-update-inference"' in command
 runtime=B/'libimajev_runtime_register.rlib'
 assert any(s=='imajev_runtime='+str(runtime) for s in command)
 refs=[Path(__file__),B/'report.json',B/'full.wasm',runtime,old/'provenance.json']+list(D.glob('*.rs'))
 hashes={str(p.relative_to(ROOT)):sha(p) for p in refs}
 env=dict(os.environ,CARGO_MANIFEST_DIR=str(D),CARGO_PKG_NAME='imajev-inference',CARGO_PKG_VERSION='0.1.0',CARGO_PKG_VERSION_MAJOR='0',CARGO_PKG_VERSION_MINOR='1',CARGO_PKG_VERSION_PATCH='0',CARGO_PKG_VERSION_PRE='',CARGO_CRATE_NAME='imajev_inference')
 with (D/'compiler.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=env,check=True,stdout=log,stderr=log)
 previous=D/'raw.wasm';patches=[]
 for i,patch in enumerate(base['patches']):
  # Patcher reports contain their absolute WAT input path.
  wat=Path(patch['wat']) if 'wat' in patch else None
  if wat is None:
   wat=ROOT/'artifacts/single-quad/build-v2'/f'kernel{i}.wat'
   if i==3:wat=ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
   if i==4:wat=ROOT/'artifacts/delta-register-v3/build/kernel.wat'
   if i==5:wat=ROOT/'artifacts/f32-output64-v1/build-v2/wide.wat'
  out=D/('full.wasm' if i==5 else f'patched{i}.wasm')
  row=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(wat),str(out),patch['export']],text=True))
  assert row['wasmparser_validation'];patches.append(row);previous=out
 assert len({p['function_index'] for p in patches})==6
 assert hashes=={p:sha(ROOT/p) for p in hashes}
 report=dict(baseline=base['baseline'],query_candidate=base['wasm_sha256'],wasm_sha256=sha(D/'full.wasm'),source_hashes=hashes,dependency_hashes=base['dependency_hashes'],command=command,patches=patches,scope=__doc__,runtime_rlib_sha256=sha(runtime),update_scheduler_sha256=sha(D/'update_inference.rs'),update_stop_instructions=34_000_000_000)
 (D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in refs:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(module=report['wasm_sha256'],bytes=(D/'full.wasm').stat().st_size)))
if __name__=='__main__':main()
