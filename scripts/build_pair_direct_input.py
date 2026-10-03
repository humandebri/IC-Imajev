#!/usr/bin/env python3
"""Build an isolated pair diagnostic that reads the original quantized buffer.

Fixed weight layout and dot/F32 arithmetic stay identical. A load64_splat
duplicates four I16 inputs directly at first use, without a duplicate Vec.
Uses explicitly pinned compiled runtime dependencies; never installs.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
dest=ROOT/'artifacts/prefix_codec/direct-input-build'
dest.mkdir(parents=True,exist_ok=True)
source=dest/'source';(source/'src').mkdir(parents=True,exist_ok=True)
original=ROOT/'scripts/wat_pair_bench'
for name in ['lib.rs','kernel.rs']:
    (source/'src'/name).write_bytes((original/'src'/name).read_bytes())
exact=(original/'src/exact.rs').read_text()
exact=exact.replace('pub struct Operands{data:Vec<i16>,rows:usize,cols:usize}',
                    'pub struct Operands{rows:usize,cols:usize}')
exact=exact.replace('crate::kernel::accumulate(x.data.as_ptr(),',
                    'crate::kernel::accumulate(q.values().as_ptr(),')
start=exact.index('pub fn operands(');end=exact.index('#[cfg(test)]mod tests',start)
exact=exact[:start]+'''pub fn operands(q:&QuantizedRows)->Operands {
 Operands{rows:q.rows(),cols:q.cols()}
}
'''+exact[end:]
exact=exact.replace('//! Fixed pairs and a query-local duplicated input.',
                    '//! Fixed pairs and direct original quantized input.')
assert 'x.data' not in exact and 'fn duplicate' not in exact
(source/'src/exact.rs').write_text(exact)
subprocess.run([sys.executable,str(ROOT/'scripts/generate_pair_direct_input_wat.py')],cwd=ROOT,check=True)
wat=dest/'direct-input.wat'
deps=ROOT/'target/wasm32-unknown-unknown/release/deps'
runtime=deps/'libimajev_runtime-66072ef408d55d63.rlib'
paths={'imajev_runtime':runtime}
for name in ['serde','sha2','ic_cdk','candid']:
    choices=list(deps.glob(f'lib{name}-*.rlib'));assert len(choices)==1
    paths[name]=choices[0]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=list((source/'src').glob('*.rs'))+[Path(__file__),wat,ROOT/'scripts/generate_pair_direct_input_wat.py',ROOT/'scripts/generate_pair_reuse_wat.py']
hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
dependencies={str(p.relative_to(ROOT)):sha(p) for p in paths.values()}
raw=dest/'raw.wasm'
command=['rustc','--crate-name','imajev_wat_pair_bench','--edition=2021',
         str(source/'src/lib.rs'),'--crate-type','cdylib','--target','wasm32-unknown-unknown',
         '-C','opt-level=3','-C','panic=abort','-C','codegen-units=1','-C','lto=thin',
         '-C','overflow-checks=yes','--emit=link','-o',str(raw),
         '-L',f'dependency={deps}','-L',f'dependency={ROOT/"target/release/deps"}']
for name,path in paths.items():command+=['--extern',f'{name}={path}']
explicit={'CARGO_MANIFEST_DIR':str(original),'CARGO_PKG_NAME':'imajev-wat-pair-bench',
          'CARGO_PKG_VERSION':'0.1.0','CARGO_PKG_VERSION_MAJOR':'0',
          'CARGO_PKG_VERSION_MINOR':'1','CARGO_PKG_VERSION_PATCH':'0',
          'CARGO_PKG_VERSION_PRE':'','CARGO_CRATE_NAME':'imajev_wat_pair_bench'}
compiler=subprocess.check_output(['rustc','--version','--verbose'],text=True)
with (dest/'compiler.log').open('w') as log:
    subprocess.run(command,cwd=ROOT,env=dict(os.environ,**explicit),stdout=log,stderr=log,check=True)
assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in sources}
assert dependencies=={str(p.relative_to(ROOT)):sha(p) for p in paths.values()}
module=dest/'diagnostic.wasm';patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch'
patch=json.loads(subprocess.check_output([str(patcher),str(raw),str(wat),str(module)],text=True))
(dest/'patch.json').write_text(json.dumps(patch,indent=2)+'\n')
result=dict(compiler=compiler,command=command,explicit_env=explicit,source_hashes=hashes,
            dependency_hashes=dependencies,wasm_sha256=sha(raw),patched_wasm_sha256=sha(module),
            patcher_sha256=sha(patcher),patch=patch,
            scope='Isolated diagnostic; pinned previously compiled runtime and newly frozen wrapper source. No full model or canister install.')
(dest/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(wasm_sha256=sha(module))),flush=True)
