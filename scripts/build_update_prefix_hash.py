#!/usr/bin/env python3
"""Combine opaque fixed prefixes with bit-identical bulk hashes for large slices."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    upstream=ROOT/'artifacts/update-bound-prefix-v1'
    hashes=json.loads((upstream/'workflow-hashes.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    proof_path=ROOT/'artifacts/float-hash-v1/summary.json'
    proof=json.loads(proof_path.read_text())
    assert proof['raw_replies_verified'] and proof['all_hashes_equal']
    assert all(sha(ROOT/p)==h for p,h in proof['workflow_hashes'].items())
    d=ROOT/'artifacts/update-prefix-hash-v1'
    d.mkdir(exist_ok=False)
    for p in list(upstream.glob('*.wat'))+[upstream/'prefix-helper.rs',upstream/'prefix-api.rs']:
        (d/p.name).write_bytes(p.read_bytes())
    source=(upstream/'frozen-builder.py').read_text().replace('artifacts/update-bound-prefix-v1','artifacts/update-prefix-hash-v1')
    anchor="    runtime = base['runtime_command'][:]"
    assert source.count(anchor)==1
    addition='''    p=D/'update_inference.rs'
    text=p.read_text()
    before='let mut h=Sha256::new();for x in v {h.update(x.to_le_bytes());}'
    assert text.count(before)==1
    after='''+repr('''let mut h=Sha256::new();
    #[cfg(target_endian="little")]
    if v.len()>=4096 {
        // F32 elements occupy initialized four-byte IEEE representations.
        // Wasm is little endian; this read-only view is exactly to_le_bytes
        // concatenation. The slice remains alive throughout the SHA update.
        let length=v.len().checked_mul(4).expect("float hash span");
        let bytes=unsafe{std::slice::from_raw_parts(v.as_ptr().cast::<u8>(),length)};
        h.update(bytes);
    }else{for x in v {h.update(x.to_le_bytes());}}
    #[cfg(not(target_endian="little"))]
    for x in v {h.update(x.to_le_bytes());}''')+'''
    p.write_text(text.replace(before,after))
'''
    source=source.replace(anchor,addition+anchor)
    (d/'frozen-builder.py').write_text(source)
    files=[Path(__file__),upstream/'workflow-hashes.json',upstream/'frozen-builder.py',proof_path,d/'frozen-builder.py',d/'prefix-helper.rs',d/'prefix-api.rs']+list(d.glob('*.wat'))
    (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2)+'\n')
    exec(compile(source,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
