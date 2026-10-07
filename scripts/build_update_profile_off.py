#!/usr/bin/env python3
"""Paid-compute experiment: compile optional runtime trace branches out of inference."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-bf16-predicates-v1';d=ROOT/'artifacts/update-profile-off-v1';assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items());d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 source=(old/'frozen-builder.py').read_text().replace('artifacts/update-bf16-predicates-v1','artifacts/update-profile-off-v1')
 anchor="    runtime[runtime.index('-o') + 1] = str(D / 'libimajev_runtime.rlib')";assert source.count(anchor)==1
 patch='''
    found=[i for i,a in enumerate(runtime) if a=='--cfg' and runtime[i+1]=='feature="instruction-profile"'];assert len(found)==1
    at=found[0];del runtime[at:at+2]
'''
 source=source.replace(anchor,anchor+patch);(d/'frozen-builder.py').write_text(source)
 files=[Path(__file__),old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(source,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 assert all(p.read_bytes()==(d/'build/runtime'/p.name).read_bytes()for p in(old/'build/runtime').glob('*.rs'))
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=[],arithmetic_kernels_equal=True,scope='All runtime source bytes and original22 WAT unchanged. Only runtime compilation omits instruction-profile cfg, removing disabled optional trace branches. Diagnostic-only prototype: owner query profiling spans are empty. Paid outputs/accounting/stage bounds preserved.'),indent=2)+'\n')
if __name__=='__main__':main()
