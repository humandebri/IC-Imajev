#!/usr/bin/env python3
"""Integrate independently verified cached SIMD quantization without changing scales or integer values."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-swiglu-cached-simd-v1';proof=ROOT/'artifacts/quantize-cached-v1/summary.json';r=json.loads(proof.read_text());assert r['all_values_scales_and_errors_equal'] and r['normal_bound_verified'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items());assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 paid=ROOT/'artifacts/paid-swiglu-cached-simd-v2/summary.json';r=json.loads(paid.read_text());assert r['complete'] and r['baseline_restored'] and r['saved_candid_replies_verified']
 d=ROOT/'artifacts/update-quantize-cached-v1';d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'artifacts/quantize-cached-v1/build/candidate.rs';s=helper.read_text();assert s.count(',elide:bool')==1 and s.count('elide&&scale')==1;s=s.replace(',elide:bool','').replace('elide&&scale','scale');(d/'quantize_cached.rs').write_text(s)
 source=(old/'frozen-builder.py').read_text().replace('artifacts/update-swiglu-cached-simd-v1','artifacts/update-quantize-cached-v1');anchor="    runtime = base['runtime_command'][:]";assert source.count(anchor)==1;source=source.replace(anchor,"    (D/'runtime/quantize_simd.rs').write_text((D.parent/'quantize_cached.rs').read_text())\n"+anchor);(d/'frozen-builder.py').write_text(source)
 files=[Path(__file__),old/'workflow-hashes.json',old/'frozen-builder.py',proof,paid,helper,d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(source,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
