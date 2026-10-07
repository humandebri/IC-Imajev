#!/usr/bin/env python3
"""Decode exact F32 bytes into a new Vec with one bulk copy."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-f32-byte-append-v1';d=ROOT/'artifacts/update-f32-byte-decode-v1';proof=ROOT/'artifacts/f32-byte-decode-v1/summary.json';r=json.loads(proof.read_text());assert r['complete'] and r['all_predicates_equal'] and r['saved_candid_redecoded'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'scripts/f32_byte_decode.rs';(d/'f32_byte_decode.rs').write_bytes(helper.read_bytes())
 replacements={'mlp_stream.rs': {'p\n        .chunks_exact(4)\n        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))\n        .collect()': 'crate::decode_f32_le(&p)'}, 'projection_codec.rs': {'p[1+q..1+q+sx*4].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect()': 'crate::decode_f32_le(&p[1+q..1+q+sx*4])', 'p[1+q+sx*4..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect()': 'crate::decode_f32_le(&p[1+q+sx*4..])'}, 'mlp_delta_stream.rs': {'payload[1 + 3 * n * C..end]\n            .chunks_exact(4)\n            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))\n            .collect()': 'crate::decode_f32_le(&payload[1 + 3 * n * C..end])'}, 'mlp_pipeline.rs': {'p[cursor+q..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect()': 'crate::decode_f32_le(&p[cursor+q..])'}, 'attention_mlp_stream.rs': {'p\n        .chunks_exact(4)\n        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))\n        .collect()': 'crate::decode_f32_le(&p)'}, 'lib.rs': {'payload\n                .chunks_exact(4)\n                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))\n                .collect()': 'crate::decode_f32_le(&payload)', 'b\n            .chunks_exact(4)\n            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))\n            .collect()': 'crate::decode_f32_le(&b)', 'scale_bytes.as_ref().chunks_exact(4)\n                .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>()': 'crate::decode_f32_le(&scale_bytes.as_ref())'}}
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-f32-byte-append-v1','artifacts/update-f32-byte-decode-v1')
 patch="    replacements="+repr(replacements)+"\n"+'''    sites={}
    for name,items in replacements.items():
        p=D/'runtime'/name;text=p.read_text()
        for before,after in items.items():assert text.count(before)==1,(name,before);text=text.replace(before,after)
        sites[name]=len(items);p.write_text(text)
    assert sum(sites.values())==9
    (D/'f32-byte-decode-sites.json').write_text(json.dumps(sites,indent=2)+'\\n')
    (D/'runtime/f32_byte_decode.rs').write_bytes((D.parent/'f32_byte_decode.rs').read_bytes())
    p=D/'runtime/lib.rs';p.write_text(p.read_text()+'\\nmod f32_byte_decode;\\npub use f32_byte_decode::decode as decode_f32_le;\\n')
'''
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor);(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),proof,helper,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=sorted(p.name for p in(old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes());assert changed==sorted(replacements),changed
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,added_runtime_files=['f32_byte_decode.rs'],arithmetic_kernels_equal=True,scope='Nine scalar F32 collect decoders replaced by exact new-Vec raw copies. Existing field order, shape and finite validators, trailing-byte semantics and little endian representation preserved. Native helper keeps scalar decode.'),indent=2)+'\n')
if __name__=='__main__':main()
