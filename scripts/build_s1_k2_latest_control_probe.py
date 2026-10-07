#!/usr/bin/env python3
"""Compare K2 with the actual current full runtime and all original22 WAT bodies."""
from pathlib import Path
import hashlib,json,re,os,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s1-k2-latest-control-v1';d.mkdir(exist_ok=False);b=d/'build';src=b/'src';src.mkdir(parents=True)
 old=ROOT/'artifacts/s1-k2-four-columns-adaptive168-v1';normal=ROOT/'artifacts/update-gated-norm-finalize-v1';ob=json.loads((old/'build/report.json').read_text());nb=json.loads((normal/'build/report.json').read_text())
 for build in (ob,nb):
  for key in ('source_hashes','dependency_hashes'):assert all(sha(ROOT/p)==h for p,h in build[key].items())
 for p in (old/'build/src').glob('*.rs'):(src/p.name).write_bytes(p.read_bytes())
 for name in ('plan.py','integer-bounds.json','runtime-edits.json'):(d/name).write_bytes((old/name).read_bytes())
 p=src/'lib.rs';s=p.read_text().replace('pub mod exact;','').replace('#[cfg(target_arch="wasm32")]mod kernel;','').replace('coeff:Option<exact::Prepared>,','').replace('coeff:None,','')
 before='f.coeff=Some(exact::Prepared::new(&f.w,f.rows,f.cols).unwrap());';assert before in s;s=s.replace(before,'')
 before='let prepared=(method==3).then(||exact::operands_mode(&q,method==1));';assert before in s;s=s.replace(before,'')
 lo=s.index(' let out=if method==4');hi=s.index(';let end=',lo)
 s=s.replace('f.coeff.as_ref().unwrap().bytes()','f.win.as_ref().unwrap().bytes()')
 lo=s.index(' let out=if method==4');hi=s.index(';let end=',lo)
 s=s[:lo]+''' let out=if method==4{f.win.as_ref().unwrap().project_wide(&q,wa.as_ref().unwrap(),&f.scales[..rows],rows).unwrap()}else{let view=f.paired.as_ref().unwrap().packed_rows(0,rows).unwrap();imajev_runtime::output_pairs::project(&q,&view,&f.scales[..rows]).unwrap()}'''+s[hi:]
 assert 'exact::'not in s;p.write_text(s)
 cmd=nb['command'][:];cmd[cmd.index('--edition=2021')+1]=str(p);cmd[cmd.index('-o')+1]=str(b/'raw.wasm');runtime=normal/'build/libimajev_runtime.rlib';assert runtime.exists()
 cmd=[('imajev_runtime='+str(runtime))if v.startswith('imajev_runtime=')else v for v in cmd]
 with (b/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**ob['explicit_env']),stdout=log,stderr=log,check=True)
 candidates=list(normal.glob('*.wat'))+list((normal/'build').glob('*.wat'))+[ROOT/'artifacts/single-quad/build-v2/kernel0.wat',ROOT/'artifacts/single-quad/build-v2/kernel2.wat'];mapping={sha(p):p for p in candidates}
 paths=[(mapping[p['source_sha256']],p['export'])for p in nb['patches']]+[(old/'build'/f'kernel{tile}.wat','__imajev_win7_wide_accumulate'if tile==128 else f'__imajev_win7_{tile}_accumulate')for tile in (128,168,32)]
 previous=b/'raw.wasm';patches=[]
 for i,(path,symbol)in enumerate(paths):
  target=b/('diagnostic.wasm'if i==len(paths)-1 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(path),str(target),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=target
 deps=[Path(cmd[i+1].split('=',1)[1])for i,v in enumerate(cmd)if v=='--extern'];files=[Path(__file__),old/'build/report.json',normal/'build/report.json',d/'plan.py',d/'integer-bounds.json',d/'runtime-edits.json']+list(src.glob('*.rs'))+[p for p,_ in paths]
 result=dict(wasm_sha256=sha(previous),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes={str(p.relative_to(ROOT)):sha(p)for p in deps},command=cmd,explicit_env=ob['explicit_env'],patches=patches,locals=ob['locals'],output_tile=128,scope=__doc__,current_full_runtime_control=True)
 (b/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(module=result['wasm_sha256'],current_full_control=True)))
if __name__=='__main__':main()
