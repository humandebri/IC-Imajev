#!/usr/bin/env python3
"""Combine exact SIMD BF16 finalization with prefixes, shape kernels and bulk hash."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'artifacts/update-prefix-hash-v2';hashes=json.loads((upstream/'workflow-hashes.json').read_text());assert all(sha(ROOT/p)==h for p,h in hashes.items())
 proof=ROOT/'artifacts/lora-finish-v1/summary.json';r=json.loads(proof.read_text());assert r['raw_replies_verified'] and r['all_bits_equal'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 d=ROOT/'artifacts/update-lora-finish-v1';d.mkdir(exist_ok=False)
 for p in list(upstream.glob('*.wat'))+[upstream/'prefix-helper.rs',upstream/'prefix-api.rs',upstream/'reset-prefix.rs']:(d/p.name).write_bytes(p.read_bytes())
 helper=(ROOT/'artifacts/lora-finish-v1/build/finish.rs').read_text();helper=helper[helper.index('#[target_feature'):].replace('unsafe fn finish(', 'unsafe fn finish_lora_simd(')
 helper='#[cfg(target_arch="wasm32")]\n'+helper
 helper += '''
fn finish_lora_bf16(mut base:Vec<f32>,z:Vec<f32>,scale:f32)->Vec<f32>{
 assert_eq!(base.len(),z.len());
 #[cfg(target_arch="wasm32")]
 if base.len()>=4 {
  // Both vectors contain initialized F32 elements; four-lane spans are in bounds.
  // Wasm SIMD is available in this runtime. Separate multiply/add and each BF16
  // round retain the original F32 operation order, including scalar tails.
  unsafe{finish_lora_simd(&mut base,&z,scale)};return base;
 }
 base.into_iter().zip(z).map(|(v,z)|bf(bf(v)+bf(scale*z))).collect()
}
'''
 (d/'finish.rs').write_text(helper)
 source=(upstream/'frozen-builder.py').read_text().replace('artifacts/update-prefix-hash-v2','artifacts/update-lora-finish-v1')
 anchor="    runtime = base['runtime_command'][:]";assert source.count(anchor)==1
 extra='''    p=D/'runtime/lib.rs'
    text=p.read_text()
    before='base.into_iter()\\n            .zip(z)\\n            .map(|(v, z)| bf(bf(v) + bf(r.scalars[0] * z)))\\n            .collect::<Vec<_>>()'
    assert text.count(before)==1
    text=text.replace(before,'finish_lora_bf16(base,z,r.scalars[0])')
    p.write_text(text+(D.parent/'finish.rs').read_text())
'''
 source=source.replace(anchor,extra+anchor);(d/'frozen-builder.py').write_text(source)
 files=[Path(__file__),upstream/'workflow-hashes.json',upstream/'frozen-builder.py',proof,d/'frozen-builder.py',d/'finish.rs',d/'prefix-helper.rs',d/'prefix-api.rs',d/'reset-prefix.rs']+list(d.glob('*.wat'))
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2)+'\n')
 exec(compile(source,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
