#!/usr/bin/env python3
"""Probe post-RMS gate finalization using the latest immutable runtime."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=ROOT/'scripts/build_swiglu_cached_simd_probe.py';s=p.read_text()
    s=s.replace('swiglu-cached-simd-v1','gated-norm-finalize-v1').replace('update-rms-ordered-simd-v1','update-attention-key-lanes-v1')
    lo=s.index(" helper=r'''");hi=s.index(" (d/'swiglu.rs').write_text(helper)",lo)
    s=s[:lo]+' helper='+repr((ROOT/'scripts/gated_norm_finalize.rs').read_text())+'\n'+s[hi:]
    s=s.replace('swiglu_scalar','gated_norm_scalar').replace('swiglu_owned','gated_norm_owned').replace('imajev_swiglu_probe','imajev_gated_norm_probe').replace("d/'swiglu.rs'","d/'gated_norm.rs'")
    d=ROOT/'artifacts/gated-norm-finalize-v1';d.mkdir(exist_ok=False)
    (d/'frozen-builder.py').write_text(s)
    files=[Path(__file__),p,ROOT/'scripts/gated_norm_finalize.rs',d/'frozen-builder.py']
    (d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},indent=2)+'\n')
    exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
