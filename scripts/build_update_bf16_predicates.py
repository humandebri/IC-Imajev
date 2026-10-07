#!/usr/bin/env python3
"""Override runtime BF16 pure predicates with verified equal-summary SIMD64 loops."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-positive-finite-v1';d=ROOT/'artifacts/update-bf16-predicates-v1';proof=ROOT/'artifacts/bf16-predicates-v1/summary.json';r=json.loads(proof.read_text());assert r['complete'] and r['all_predicates_equal'] and r['saved_candid_redecoded'] and r['control_bodies_identical_to_core'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'scripts/bf16_predicates.rs';(d/'bf16_predicates.rs').write_bytes(helper.read_bytes())
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-positive-finite-v1','artifacts/update-bf16-predicates-v1')
 wrapper='''
pub fn all_bf16(x:&[f32])->bool {
 #[cfg(target_arch="wasm32")]unsafe{return crate::bf16_predicates::all64(x);}
 #[cfg(not(target_arch="wasm32"))]inference_core::bf16::all_bf16(x)
}
pub fn classify_finite(x:&[f32])->crate::Result<bool> {
 #[cfg(target_arch="wasm32")]unsafe{return crate::bf16_predicates::classify64(x);}
 #[cfg(not(target_arch="wasm32"))]inference_core::bf16::classify_finite(x)
}
'''
 patch="    (D/'runtime/bf16_predicates.rs').write_bytes((D.parent/'bf16_predicates.rs').read_bytes())\n    p=D/'runtime/bf16_codec.rs';text=p.read_text();assert text.strip()=='//! Compatibility path; implementation lives in inference-core.\\npub use inference_core::bf16::*;';p.write_text(text+"+repr(wrapper)+")\n    p=D/'runtime/lib.rs';p.write_text(p.read_text()+'\\nmod bf16_predicates;\\n')\n"
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor);(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),proof,helper,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=sorted(p.name for p in(old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes());assert changed==['bf16_codec.rs','lib.rs'],changed
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,added_runtime_files=['bf16_predicates.rs'],arithmetic_kernels_equal=True,scope='Runtime BF16 all/classify predicates use verified unrolled64 loops with identical low OR and absolute unsigned max. Error precedence and message preserved. Pack/unpack and native path unchanged.'),indent=2)+'\n')
if __name__=='__main__':main()
