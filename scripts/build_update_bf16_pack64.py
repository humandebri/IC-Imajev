#!/usr/bin/env python3
"""Build a local inference-core overlay changing only BF16 pack/unpack SIMD bodies."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-f32-byte-decode-v1';d=ROOT/'artifacts/update-bf16-pack64-v1';proof=ROOT/'artifacts/bf16-pack64-v1/summary.json';r=json.loads(proof.read_text());assert r['complete'] and r['all_predicates_equal'] and r['saved_candid_redecoded'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items());assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 core=d/'core-src';core.mkdir();original=ROOT/'crates/inference-core/src'
 for p in original.glob('*.rs'):(core/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'scripts/bf16_pack64.rs';(core/'bf16_pack64.rs').write_bytes(helper.read_bytes())
 p=core/'lib.rs';p.write_text(p.read_text()+'\nmod bf16_pack64;\n')
 p=core/'bf16.rs';text=p.read_text()
 for name,call in [('pack_simd','crate::bf16_pack64::pack64(x,bytes);'),('unpack_simd','crate::bf16_pack64::unpack64(bytes,x);')]:
  begin=text.index('unsafe fn '+name+'(');a=text.index('{',begin);level=1;b=a+1
  while level:level+=(text[b]=='{')-(text[b]=='}');b+=1
  text=text[:a]+'{'+call+'}'+text[b:]
 p.write_text(text);output=d/'core-build';output.mkdir();rlib=output/'libinference_core.rlib'
 command=['rustc','--crate-name','inference_core','--edition=2021',str(core/'lib.rs'),'--crate-type','rlib','--target','wasm32-unknown-unknown','-C','opt-level=3','-C','panic=abort','-C','codegen-units=1','-C','embed-bitcode=yes','-C','overflow-checks=no','-C','metadata=bf16_pack64_v1','-o',str(rlib)]
 with(output/'compiler.log').open('w')as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=log,check=True)
 (d/'core-command.json').write_text(json.dumps(dict(command=command,module=sha(rlib),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in core.glob('*.rs')}),indent=2)+'\n')
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-f32-byte-decode-v1','artifacts/update-bf16-pack64-v1')
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1
 patch='''    core_rlib=D.parent/'core-build/libinference_core.rlib'
    old_core=next(p for p in base['dependency_hashes'] if Path(p).name.startswith('libinference_core-'))
    del base['dependency_hashes'][old_core]
    base['dependency_hashes'][str(core_rlib.relative_to(ROOT))]=sha(core_rlib)
    for p in (D.parent/'core-src').glob('*.rs'):base['dependency_hashes'][str(p.relative_to(ROOT))]=sha(p)
'''
 s=s.replace(anchor,patch+anchor+"\n    runtime=[('inference_core='+str(core_rlib)) if a.startswith('inference_core=') else a for a in runtime]\n    runtime+=['-L','dependency='+str(core_rlib.parent)]")
 anchor="    env = dict(os.environ, CARGO_MANIFEST_DIR=str(D)";assert s.count(anchor)==1;s=s.replace(anchor,"    wrapper+=['-L','dependency='+str(core_rlib.parent)]\n"+anchor)
 s=s.replace("sources = [D/'finite-sites.json'","sources = list((D.parent/'core-src').glob('*.rs'))+[D.parent/'core-command.json',D/'finite-sites.json'")
 (d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),proof,helper,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py',d/'core-command.json',rlib]+list(core.glob('*.rs'))+list(original.glob('*.rs'))+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 assert all(p.read_bytes()==(d/'build/runtime'/p.name).read_bytes()for p in(old/'build/runtime').glob('*.rs'))
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=[],added_runtime_files=[],changed_dependency='inference_core BF16 pack/unpack SIMD bodies and helper module',arithmetic_kernels_equal=True,scope='Runtime source bytes unchanged. Independent core overlay preserves linear/block256/native/public validation bytes; only Wasm pack/unpack body dispatches to independently verified shuffle64/expand64 helper. Original22 arithmetic WAT kernels retained.'),indent=2)+'\n')
if __name__=='__main__':main()
