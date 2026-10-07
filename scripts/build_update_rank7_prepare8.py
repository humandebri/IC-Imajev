#!/usr/bin/env python3
"""Latest normal hybrid with exactly one verified input preparation revision."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parent=ROOT/'artifacts/update-common-raw-hybrid-v1/build';base=json.loads((parent/'report.json').read_text())
    probe=ROOT/'artifacts/rank7-prepare8-probe-v1';audit=json.loads((probe/'independent-audit.json').read_text())
    assert audit['all_nonempty_measured_cases_improve']
    for r in [base,audit]:
        for key in ['source_hashes','dependency_hashes']:
            for p,h in r.get(key,{}).items():assert sha(ROOT/p)==h,p
    d=ROOT/'artifacts/update-rank7-prepare8-v1';b=d/'build';b.mkdir(parents=True,exist_ok=False)
    shutil.copytree(parent/'runtime',b/'runtime')
    for p in parent.glob('*.rs'):shutil.copyfile(p,b/p.name)
    p=b/'runtime/strassen_raw.rs';old=p.read_text();s=old
    for a,z,count in [('for k in(0..128).step_by(4)','for k in(0..128).step_by(8)',1),('v128_load64_splat(p.','v128_load(p.',4),('v128_store64_lane::<0>(v,out.add(m*pairs*(cols/2)+pair*(cols/2)+block*128+k).cast());','v128_store(out.add(m*pairs*(cols/2)+pair*(cols/2)+block*128+k).cast(),v);',1)]:
        assert s.count(a)==count,(a,s.count(a));s=s.replace(a,z)
    p.write_text(s)
    runtime=base['runtime_command'][:];runtime[runtime.index('--edition=2021')+1]=str(b/'runtime/lib.rs');runtime[runtime.index('-o')+1]=str(b/'libimajev_runtime.rlib')
    cmd=base['command'][:];cmd[cmd.index('--edition=2021')+1]=str(b/'lib.rs');cmd[cmd.index('-o')+1]=str(b/'raw.wasm')
    cmd=[('imajev_runtime='+str(b/'libimajev_runtime.rlib'))if x.startswith('imajev_runtime=')else x for x in cmd]
    env=dict(os.environ,CARGO_MANIFEST_DIR=str(b),CARGO_PKG_NAME='imajev-runtime',CARGO_PKG_VERSION='0.1.0')
    with(b/'runtime-compiler.log').open('w')as log:subprocess.run(runtime,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
    env.update(CARGO_PKG_NAME='imajev-inference',CARGO_PKG_VERSION_MAJOR='0',CARGO_PKG_VERSION_MINOR='1',CARGO_PKG_VERSION_PATCH='0',CARGO_PKG_VERSION_PRE='',CARGO_CRATE_NAME='imajev_inference')
    with(b/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
    paths={h:ROOT/p for p,h in base['source_hashes'].items()if p.endswith('.wat')}
    previous=b/'raw.wasm';patches=[]
    for i,entry in enumerate(base['patches']):
        wat=paths[entry['source_sha256']];symbol=entry['export'];target=b/('full.wasm'if i==len(base['patches'])-1 else f'patched{i}.wasm')
        patch=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(wat),str(target),symbol],text=True));assert patch['wasmparser_validation']
        assert patch['source_sha256']==entry['source_sha256'];patches.append(patch);previous=target
    changed=[]
    for p in sorted((b/'runtime').glob('*.rs')):
        if p.read_bytes()!=(parent/'runtime'/p.name).read_bytes():changed.append(p.name)
    assert changed==['strassen_raw.rs']
    for p in b.glob('*.rs'):assert p.read_bytes()==(parent/p.name).read_bytes()
    files=[Path(__file__),parent/'report.json',probe/'independent-audit.json',probe/'archive-identity.json']+list(b.glob('*.rs'))+list((b/'runtime').glob('*.rs'))+list(paths.values())
    deps=[Path(command[i+1].split('=',1)[1])for command in [runtime,cmd]for i,v in enumerate(command)if v=='--extern']
    result=dict(wasm_sha256=sha(previous),runtime_command=runtime,command=cmd,patches=patches,changed_runtime_files=changed,parent=base['wasm_sha256'],source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},dependency_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(deps)},full_paid_goal_achieved=False,scope='Compiled normal candidate only. Same34 projection kernels and all other runtime sources. Full hidden/state/decision and paid instruction proof pending; not adopted.')
    (b/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(module=result['wasm_sha256'],patches=len(patches),bytes=previous.stat().st_size)))
if __name__=='__main__':main()
