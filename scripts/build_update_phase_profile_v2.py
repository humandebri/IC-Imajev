#!/usr/bin/env python3
"""Diagnostic only: profile four actual paid compute phases, keeping all arithmetic kernels."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-bf16-predicates-v1';d=ROOT/'artifacts/update-phase-profile-v2';assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items());d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 source=(old/'frozen-builder.py').read_text().replace('artifacts/update-bf16-predicates-v1','artifacts/update-phase-profile-v2')
 patch='''    p=D/'runtime/profile.rs';text=p.read_text();anchor='        let clock = CLOCK.with(|c| c.get());';assert text.count(anchor)==1
    text=text.replace(anchor,'        if !matches!(name,"base_project_inclusive"|"f32_project_inclusive"|"activation_quantize"|"delta_recurrence") {return run();}\\n'+anchor);p.write_text(text)
    p=D/'runtime/delta_full_log.rs';text=p.read_text();anchor='crate::delta_from_key_major(&qh,&kh,&vh,&gh,&bh,state,128,128)';assert text.count(anchor)==1;text=text.replace(anchor,'crate::profile::measure("delta_recurrence",||'+anchor+')');p.write_text(text)
    p=D/'runtime/lib.rs';text=p.read_text();signature='pub(crate) fn matrix_loaded(x:&[f32],w:&LoadedWeight,n:usize,rows:usize,cols:usize)->Result<Vec<f32>> {';assert text.count(signature)==1
    wrapper=signature+'\\n    profile::measure("f32_project_inclusive",||matrix_loaded_unprofiled(x,w,n,rows,cols))\\n}\\n'+signature.replace('fn matrix_loaded(', 'fn matrix_loaded_unprofiled(')
    text=text.replace(signature,wrapper);p.write_text(text)
'''
 anchor="    runtime = base['runtime_command'][:]";assert source.count(anchor)==1;source=source.replace(anchor,patch+anchor);(d/'frozen-builder.py').write_text(source)
 files=[Path(__file__),old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(source,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=sorted(p.name for p in(old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes());assert changed==['delta_full_log.rs','lib.rs','profile.rs'];(d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,arithmetic_kernels_equal=True,scope='Diagnostic only. Four coarse actual compute phase spans: every matrix_loaded call, base INT8 projection, activation quantization and outer Delta recurrence. Deep kernel/loop profiling disabled. One outer Delta recurrence span added. Every original22 WAT retained. Profiled total is not used as an optimization result.'),indent=2)+'\n')
if __name__=='__main__':main()
