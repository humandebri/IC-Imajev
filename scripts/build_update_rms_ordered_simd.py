#!/usr/bin/env python3
"""Use measured RMS vector products with original ordered F32 summation."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'artifacts/update-add-norm-simd-v1';r=json.loads((upstream/'summary.json').read_text());assert r['verified'] and r['baseline_restored'];assert all(sha(ROOT/p)==h for p,h in json.loads((upstream/'workflow-hashes.json').read_text()).items())
 proof=ROOT/'artifacts/rms-ordered-simd-v1/summary.json';r=json.loads(proof.read_text());assert r['raw_replies_verified'] and r['all_bits_equal'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 d=ROOT/'artifacts/update-rms-ordered-simd-v1';d.mkdir(exist_ok=False)
 for p in list(upstream.glob('*.wat'))+list(upstream.glob('*.rs')):(d/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'artifacts/rms-ordered-simd-v1/build/rms_ordered_simd.rs';(d/'rms_ordered_simd.rs').write_bytes(helper.read_bytes())
 s=(upstream/'frozen-builder.py').read_text().replace('artifacts/update-add-norm-simd-v1','artifacts/update-rms-ordered-simd-v1')
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1
 extra=r'''    p=D/'runtime/lib.rs'
    text=p.read_text()
    before='pub fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {'
    assert text.count(before)==1
    text=text.replace(before,before+'\n    #[cfg(target_arch="wasm32")]\n    if x.len()>=4 && x.len()==w.len(){return unsafe{rms_ordered_simd(x,w,eps)};}')
    p.write_text(text+'\n#[cfg(target_arch="wasm32")]\n'+(D.parent/'rms_ordered_simd.rs').read_text())
'''
 s=s.replace(anchor,extra+anchor);(d/'frozen-builder.py').write_text(s)
 paths=[Path(__file__),proof,helper,upstream/'summary.json',upstream/'workflow-hashes.json',upstream/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths},indent=2)+'\n')
 exec(compile(s,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
