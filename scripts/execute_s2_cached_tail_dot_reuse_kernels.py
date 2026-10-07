#!/usr/bin/env python3
"""Execute actual SIMD Wasm with bijective raw permutation and ordered scalar oracle."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-cached-tail-dot-reuse-kernels-v1';r=json.loads((d/'report.json').read_text())
 for p,h in r['source_hashes'].items():assert sha(ROOT/p)==h,p
 check=d/'execution';check.mkdir(exist_ok=False);files=[Path(__file__),d/'report.json']+[ROOT/p for p in r['source_hashes']];modules={}
 for k in r['kernels']:
  text=(ROOT/k['path']).read_text();assert text.startswith('(module(func')
  p=check/(Path(k['path']).stem+'.wat');p.write_text(text.replace('(module(func','(module(import "env" "memory" (memory 1))(func',1));wasm=p.with_suffix('.wasm')
  subprocess.run([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-audit'),'wat',str(p),str(wasm)],check=True);modules[f'{k["tile"]}/{int(k["seed"])}']=str(wasm);files.extend([p,wasm])
 plan=ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1/plan.py';ns={'__name__':'plan','__file__':str(plan)};exec(compile(plan.read_text(),str(plan),'exec'),ns);a,_,_,leaves,_,_=ns['plan']();files.append(plan)
 rng=np.random.default_rng(49016096);cases=[]
 for rows in [16,96]:
  for cols in [256,512,2560,9216]:
   blocks=cols//256;weights=rng.integers(-128,128,(rows,cols),dtype=np.int16);sw=rng.uniform(.001,.1,rows).astype('<f4');rr,cc=np.indices((rows,cols));kk=cc%64;plane=rows*cols//4
   indices=((cc%256)//128*2+(rr%8)//4)*plane+(rr//8)*cols*2+(cc//256)*512+((cc%128)//2)*8+(rr%4)*2+cc%2
   assert np.array_equal(np.sort(indices.ravel()),np.arange(rows*cols));packed=np.empty(rows*cols,dtype=np.int8);packed[indices]=weights;assert np.array_equal(packed[indices],weights)
   tokens=[0,1,3,4,5,48,56,57,87,89,132]if cols<=512 else[1,57]
   for n in tokens:
    groups=(n+3)//4;q=rng.integers(-127,128,(groups*4,cols),dtype=np.int16);q[n:]=0;compact=np.empty((49,groups,cols//4),dtype='<i2')
    for group in range(groups):
     for block in range(blocks):
      av=q[group*4:group*4+4,block*256:(block+1)*256].reshape(16,64).astype(np.int64)
      for m,(an,_)in enumerate(leaves):compact[m,group,block*64:(block+1)*64]=sum(v*av[i]for i,v in a.symbols[an].items())
    sx=rng.uniform(.001,.1,(n,blocks)).astype('<f4');stride=rows+4;initial=rng.standard_normal((n,stride)).astype('<f4');initial[:,::2]=-0.0
    for seed in [False,True]:
     expected=initial.copy()
     for block in range(blocks):
      dots=q[:n,block*256:(block+1)*256].astype(np.int64)@weights[:,block*256:(block+1)*256].astype(np.int64).T
      term=np.multiply(np.multiply(dots.astype(np.float32),sx[:,block,None],dtype=np.float32),sw,dtype=np.float32)
      prior=np.zeros((n,rows),dtype=np.float32)if seed and block==0 else expected[:,:rows];expected[:,:rows]=np.add(prior,term,dtype=np.float32)
     image=bytearray(b'\xa5'*64)
     def alloc(data):
      image.extend(b'\xa5'*((-len(image))%16));p=len(image);image.extend(data);return p
     wbase=alloc(packed.tobytes());qbase=alloc(compact.tobytes());wp=alloc(np.array([wbase+((ki//2)*2+ni%2)*plane+(ni//2)*cols*2+(ki%2)*256 for ki in range(4) for ni in range(4)],dtype='<u4').tobytes());qp=alloc(np.array([qbase+m*groups*(cols//4)*2 for m in range(49)],dtype='<u4').tobytes());sxp=alloc(sx.tobytes());swp=alloc(sw.tobytes());out=alloc(initial.tobytes());image.extend(b'\xa5'*64)
     oracle=bytearray(image);oracle[out:out+expected.nbytes]=expected.tobytes();prefix=check/f'r{rows}-c{cols}-n{n}-s{int(seed)}';inp=prefix.with_suffix('.input.bin');exp=prefix.with_suffix('.expected.bin');inp.write_bytes(image);exp.write_bytes(oracle);files.extend([inp,exp])
     cases.append(dict(rows=rows,cols=cols,n=n,seed=seed,q=qp,w=wp,sx=sxp,sw=swp,out=out,stride=stride,input=str(inp),expected=str(exp)))
 runner=check/'run.js';runner.write_text('''const fs=require('fs');
 (async()=>{const config=JSON.parse(fs.readFileSync(process.argv[2]));const memory=new WebAssembly.Memory({initial:1});const instances={};
 for(const [key,path]of Object.entries(config.modules))instances[key]=(await WebAssembly.instantiate(fs.readFileSync(path),{env:{memory}})).instance;
 let count=0;for(const c of config.cases){const input=fs.readFileSync(c.input),expected=fs.readFileSync(c.expected);memory.grow(Math.max(0,Math.ceil(input.length/65536)-memory.buffer.byteLength/65536));new Uint8Array(memory.buffer).set(input);
 for(let start=0;start<c.cols;start+=256){const instance=instances[c.rows+'/'+Number(c.seed&&start===0)];instance.exports['__imajev_s2_k2_'+c.rows+(c.seed&&start===0?'_seed':'')](c.q,c.w,c.cols,start,c.sx+start/256*4,c.stride,c.sw,c.out,c.n);}
 const actual=Buffer.from(memory.buffer,0,input.length);if(!actual.equals(expected)){let i=0;while(actual[i]===expected[i])i++;throw Error(JSON.stringify({rows:c.rows,cols:c.cols,n:c.n,seed:c.seed,mismatch_byte:i,actual:actual[i],expected:expected[i]}));}count++;console.log(JSON.stringify({rows:c.rows,cols:c.cols,n:c.n,seed:c.seed,all_memory_bits_equal:true}));}
 console.log(JSON.stringify({complete:true,conditions:count}));})().catch(e=>{console.error(e);process.exit(1)});
''')
 config=check/'config.json';config.write_text(json.dumps(dict(modules=modules,cases=cases)));files.extend([runner,config]);result=subprocess.run(['node',str(runner),str(config)],capture_output=True,text=True);(check/'node.stdout').write_text(result.stdout);(check/'node.stderr').write_text(result.stderr);files.extend([check/'node.stdout',check/'node.stderr'])
 if result.returncode:raise RuntimeError(result.stderr)
 lines=[json.loads(line)for line in result.stdout.splitlines()];assert lines[-1]==dict(complete=True,conditions=len(cases)) and len(lines)==len(cases)+1
 report=dict(complete=True,conditions=len(cases),cases=lines[:-1],all_raw_index_bijections_exact=True,raw_payload_bytes_unchanged=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},ic_execution_verified=False,performance_verified=False,scope='Actual Node SIMD Wasm full-memory comparison, independent raw weight permutation and scalar ascending-K256 ordered-F32 oracle. Imported memory is test-only. No full paid claim.')
 (d/'execution-report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'execution-evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in dict.fromkeys(files+[d/'execution-report.json']):z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,conditions=len(cases),all_memory_bits_equal=True)))
if __name__=='__main__':main()
