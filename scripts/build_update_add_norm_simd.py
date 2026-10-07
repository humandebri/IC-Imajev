#!/usr/bin/env python3
"""Integrate measured exact add_norm SIMD into the verified full candidate."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'artifacts/update-conv4-cached-v2';r=json.loads((upstream/'summary.json').read_text());assert r['verified'] and r['baseline_restored']
 assert all(sha(ROOT/p)==h for p,h in json.loads((upstream/'workflow-hashes.json').read_text()).items())
 proof=ROOT/'artifacts/add-norm-simd-v1/summary.json';r=json.loads(proof.read_text());assert r['raw_replies_verified'] and r['all_bits_equal'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 d=ROOT/'artifacts/update-add-norm-simd-v1';d.mkdir(exist_ok=False)
 for p in list(upstream.glob('*.wat'))+list(upstream.glob('*.rs')):(d/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'artifacts/add-norm-simd-v1/build/add_norm_simd.rs';text=helper.read_text();text=text[text.index('#[target_feature'):];(d/'add_norm_simd.rs').write_text(text)
 s=(upstream/'frozen-builder.py').read_text().replace('artifacts/update-conv4-cached-v2','artifacts/update-add-norm-simd-v1')
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1
 extra=r'''    p=D/'runtime/lib.rs'
    text=p.read_text()
    lo=text.index('            let count = d[0] * d[1];',text.index('        "add_norm_bf16"'))
    hi=text.index('\n        }',lo)
    old=text[lo:hi]
    new='            #[cfg(target_arch="wasm32")]\n            {if d[1]>=4 {unsafe{add_norm_simd(x,weight,d[0],d[1],r.scalars[0])}}else{\n'+old+'\n            }}\n            #[cfg(not(target_arch="wasm32"))]\n            {\n'+old+'\n            }'
    text=text[:lo]+new+text[hi:]
    p.write_text(text+'\n#[cfg(target_arch="wasm32")]\n'+(D.parent/'add_norm_simd.rs').read_text())
'''
 s=s.replace(anchor,extra+anchor);(d/'frozen-builder.py').write_text(s)
 paths=[Path(__file__),proof,helper,upstream/'summary.json',upstream/'workflow-hashes.json',upstream/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'))
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths},indent=2)+'\n')
 exec(compile(s,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
