#!/usr/bin/env python3
"""Integrate exact owned-gate SIMD SwiGLU in both fused MLP projection paths."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'artifacts/update-rms-ordered-simd-v1';assert all(sha(ROOT/v)==h for v,h in json.loads((upstream/'workflow-hashes.json').read_text()).items())
 paid=ROOT/'artifacts/paid-rms-ordered-simd-v1/partial-summary.json';r=json.loads(paid.read_text());assert r['paid_core_api_verified'] and r['baseline_restored']
 proof=ROOT/'artifacts/swiglu-cached-simd-v1/summary.json';r=json.loads(proof.read_text());assert r['raw_replies_verified'] and r['all_bits_equal'];assert all(sha(ROOT/v)==h for v,h in r['workflow_hashes'].items())
 d=ROOT/'artifacts/update-swiglu-cached-simd-v1';d.mkdir(exist_ok=False)
 for v in list(upstream.glob('*.wat'))+list(upstream.glob('*.rs')):(d/v.name).write_bytes(v.read_bytes())
 helper=ROOT/'artifacts/swiglu-cached-simd-v1/build/swiglu.rs';(d/'swiglu.rs').write_bytes(helper.read_bytes())
 s=(upstream/'frozen-builder.py').read_text().replace('artifacts/update-rms-ordered-simd-v1','artifacts/update-swiglu-cached-simd-v1');anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1
 extra=r'''    p=D/'runtime/lib.rs';text=p.read_text()
    before='    let out = gate\n        .into_iter()\n        .zip(up)\n        .map(|(g, u)| bf(bf_silu(g) * u))\n        .collect::<Vec<_>>();'
    assert text.count(before)==1
    p.write_text(text.replace(before,'    let out = swiglu_owned(gate,up);')+(D.parent/'swiglu.rs').read_text())
    p=D/'runtime/mlp_reuse.rs';text=p.read_text()
    before='    let out:Vec<_>=gate.into_iter().zip(up).map(|(g,u)|crate::bf(crate::bf_silu(g)*u)).collect();'
    assert text.count(before)==1
    p.write_text(text.replace(before,'    let out=crate::swiglu_owned(gate,up);'))
'''
 s=s.replace(anchor,extra+anchor);(d/'frozen-builder.py').write_text(s);files=[Path(__file__),upstream/'workflow-hashes.json',upstream/'frozen-builder.py',paid,proof,helper,d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(v.relative_to(ROOT)):sha(v)for v in files},indent=2)+'\n')
 exec(compile(s,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
