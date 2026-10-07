#!/usr/bin/env python3
"""Build the latest optimized runtime with the canonical paid API and billing sources."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'artifacts/update-add-norm-simd-v1';assert all(sha(ROOT/p)==h for p,h in json.loads((upstream/'workflow-hashes.json').read_text()).items())
 d=ROOT/'artifacts/paid-add-norm-simd-v1';d.mkdir(exist_ok=False);p=ROOT/'scripts/build_paid_update_full_candid.py'
 s=p.read_text().replace('artifacts/update-templates-v1/build','artifacts/update-add-norm-simd-v1/build').replace('artifacts/paid-update-v1/build-v3','artifacts/paid-add-norm-simd-v1/build')
 lo=s.index(" runtime=B/'libimajev_runtime_register.rlib'");hi=s.index('\n refs=',lo)
 s=s[:lo]+" runtime=B/'libimajev_runtime.rlib'\n assert any(a=='imajev_runtime='+str(runtime) for a in command)"+s[hi:]
 lo=s.index('  # Patcher reports');hi=s.index("  out=D/",lo)
 paths=['artifacts/single-quad/build-v2/kernel0.wat','artifacts/update-add-norm-simd-v1/int8_32.wat','artifacts/single-quad/build-v2/kernel2.wat','artifacts/s1-pair-bounds-v1/build/full-kernel.wat','artifacts/update-add-norm-simd-v1/delta.wat','artifacts/update-add-norm-simd-v1/wide64.wat','artifacts/update-add-norm-simd-v1/wide128.wat','artifacts/update-add-norm-simd-v1/int8_160.wat','artifacts/update-add-norm-simd-v1/wide512.wat','artifacts/update-add-norm-simd-v1/columns512.wat']
 # Resolve the adaptive160 name by its authenticated source digest.
 r=json.loads((upstream/'build/report.json').read_text());assert len(r['patches'])==10
 for i in range(10):
  if not (ROOT/paths[i]).exists() or sha(ROOT/paths[i])!=r['patches'][i]['source_sha256']:
   hits=[v for v in list(upstream.glob('*.wat'))+list((upstream/'build').glob('*.wat')) if sha(v)==r['patches'][i]['source_sha256']];assert len(hits)==1,(i,hits);paths[i]=str(hits[0].relative_to(ROOT))
 s=s[:lo]+"  wat=ROOT/"+repr(paths)+"[i]\n  assert sha(wat)==patch['source_sha256']\n"+s[hi:]
 s=s.replace("'full.wasm' if i==5","'full.wasm' if i==9").replace("function_index'] for p in patches})==6","function_index'] for p in patches})==10")
 (d/'frozen-builder.py').write_text(s);files=[Path(__file__),p,upstream/'workflow-hashes.json',d/'frozen-builder.py']+[ROOT/v for v in paths]
 (d/'workflow-hashes.json').write_text(json.dumps({str(v.relative_to(ROOT)):sha(v) for v in files},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 result=json.loads((d/'build/report.json').read_text());names=['update_inference.rs','paid_types.rs','paid_inference.rs'];assert all((d/'build'/n).read_bytes()==(ROOT/'canisters/inference/src'/n).read_bytes() for n in names)
 result['canonical_paid_sources_byte_equal']=True;result['paid_execution_verified']=False;(d/'build/report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
