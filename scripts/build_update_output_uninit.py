#!/usr/bin/env python3
"""Remove final output zero-fill only for validated complete aligned projections."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-direct168-v1';d=ROOT/'artifacts/update-output-uninit-v1'
 proofs=[ROOT/f'artifacts/{name}/summary.json'for name in ['s1-output-uninit-v1','s1-output-uninit-wide-v1','s1-output-uninit-wide-mlp-v1']]
 for p in proofs:
  r=json.loads(p.read_text());assert r['native_bits_equal'] and r['native_digests_recomputed'];assert all(sha(ROOT/q)==h for q,h in r['workflow_hashes'].items())
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-direct168-v1','artifacts/update-output-uninit-v1')
 patch='''    p=D/'runtime/strassen_raw.rs';text=p.read_text();a=text.index('pub(crate) fn project(q:');b=text.index('// One real token',a);body=text[a:b]
    anchor='let mut out=vec![0.;q.rows()*rows];';assert body.count(anchor)==1
    replacement='''+repr('''#[cfg(not(target_arch="wasm32"))]let mut out=vec![0.;q.rows()*rows];
 #[cfg(target_arch="wasm32")]let mut out_storage:Vec<core::mem::MaybeUninit<f32>>=if q.rows()>0 && rows>0 && rows%32==0 && start%4==0 && cols>=256 && cols%256==0 {let count=q.rows()*rows;let mut v=Vec::with_capacity(count);unsafe{v.set_len(count);}v}else{vec![core::mem::MaybeUninit::new(0f32);q.rows()*rows]};
 #[cfg(target_arch="wasm32")]let out_ptr=out_storage.as_mut_ptr().cast::<f32>();''')+'''
    body=body.replace(anchor,replacement).replace('s.as_ptr(),out.as_mut_ptr().add(r),q.rows()','s.as_ptr(),out_ptr.add(r),q.rows()')
    anchor='out[t*rows+r..t*rows+r+width].copy_from_slice(&sums[t*tile..t*tile+width]);';assert body.count(anchor)==1
    body=body.replace(anchor,'unsafe{core::ptr::copy_nonoverlapping(sums.as_ptr().add(t*tile),out_ptr.add(t*rows+r),width);}');anchor='r+=width;}}\\n #[cfg(not(target_arch="wasm32"))]';assert body.count(anchor)==1
    body=body.replace(anchor,'r+=width;}}\\n #[cfg(target_arch="wasm32")]let out={let storage=core::mem::ManuallyDrop::new(out_storage);unsafe{Vec::from_raw_parts(out_ptr,storage.len(),storage.capacity())}};\\n #[cfg(not(target_arch="wasm32"))]')
    assert 'out.as_mut_ptr()'not in body and 'out[t*rows+r..'not in body
    p.write_text(text[:a]+body+text[b:])
'''
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor);compile(s,'frozen-builder','exec');(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),*proofs,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=sorted(p.name for p in(old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes());assert changed==['strassen_raw.rs'],changed
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,arithmetic_kernels_equal=True,scope='Final output zero-fill omitted for nonempty aligned rows32/K256 projections. First-block direct kernels and initialized fallback copies cover all outputs before F32 Vec conversion. Other views retain zero initialization; native path unchanged.'),indent=2)+'\n')
if __name__=='__main__':main()
