#!/usr/bin/env python3
"""Add exact ordered conv4 SIMD to the verified full-update candidate."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'artifacts/update-lora-finish-v1';assert all(sha(ROOT/p)==h for p,h in json.loads((upstream/'workflow-hashes.json').read_text()).items())
 r=json.loads((upstream/'summary.json').read_text());assert r['verified'] and r['baseline_restored']
 proof=ROOT/'artifacts/conv4-simd-v3/summary.json';r=json.loads(proof.read_text());assert r['raw_replies_verified'] and r['all_bits_equal'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 d=ROOT/'artifacts/update-conv4-v1';d.mkdir(exist_ok=False)
 for p in list(upstream.glob('*.wat'))+[upstream/n for n in ['prefix-helper.rs','prefix-api.rs','reset-prefix.rs','finish.rs']]:(d/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'artifacts/conv4-simd-v3/build/conv4.rs';(d/'conv4.rs').write_bytes(helper.read_bytes())
 s=(upstream/'frozen-builder.py').read_text().replace('artifacts/update-lora-finish-v1','artifacts/update-conv4-v1')
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1
 extra='''    p=D/'runtime/lib.rs'
    text=p.read_text()
    lo=text.index('            let (n, c, k) = (d[0], d[1], d[2]);',text.index('        "conv_state"'))
    hi=text.index('\\n        }',lo)
    old=text[lo:hi]
    declaration='            let (n, c, k) = (d[0], d[1], d[2]);\\n'
    assert old.startswith(declaration)
    body=old[len(declaration):]
    new=declaration+'            #[cfg(target_arch="wasm32")]\\n            {if k==4 && c%4==0 {\\n                // The checked conv_state shape above guarantees initialized,\\n                // in-bounds four-channel spans and the complete history tail.\\n                unsafe{conv4_simd::<true>(x,weight,n,c)}\\n            }else{\\n'+body+'\\n            }}\\n            #[cfg(not(target_arch="wasm32"))]\\n            {\\n'+body+'\\n            }'
    text=text[:lo]+new+text[hi:]
    p.write_text(text+(D.parent/'conv4.rs').read_text())
'''
 s=s.replace(anchor,extra+anchor);(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),upstream/'workflow-hashes.json',upstream/'frozen-builder.py',upstream/'summary.json',proof,helper,d/'frozen-builder.py',d/'conv4.rs']+[d/n for n in ['prefix-helper.rs','prefix-api.rs','reset-prefix.rs','finish.rs']]+list(d.glob('*.wat'))
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2)+'\n')
 exec(compile(s,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
