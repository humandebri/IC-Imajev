#!/usr/bin/env python3
"""Execute generated SIMD Wasm on Node, compare full memory against scalar dots."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s3-streaming-shared-kernels-v1';r=json.loads((d/'report.json').read_text())
 for p,h in r['source_hashes'].items():assert sha(ROOT/p)==h,p
 check=d/'execution-v2';check.mkdir(exist_ok=False)
 text=(d/'256.wat').read_text();assert text.count('(memory 1)')==1
 (check/'test.wat').write_text(text.replace('(memory 1)','(memory(export "memory")1)'))
 subprocess.run([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-audit'),'wat',str(check/'test.wat'),str(check/'test.wasm')],check=True)
 runner=check/'run.js';runner.write_text('''const fs=require('fs');
 (async()=>{const m=JSON.parse(fs.readFileSync(process.argv[2]));const bin=fs.readFileSync(m.input),expected=fs.readFileSync(m.expected);
 const {instance}=await WebAssembly.instantiate(fs.readFileSync(m.wasm));const memory=instance.exports.memory;
 memory.grow(Math.ceil(bin.length/65536)-1);new Uint8Array(memory.buffer).set(bin);
 for(let start=0;start<m.cols;start+=256)instance.exports.__imajev_s3_streaming_256(m.q,m.w,m.cols,start,m.sx+start/256*4,m.stride,m.sw,m.out,m.n);
 const actual=Buffer.from(memory.buffer,0,bin.length);if(!actual.equals(expected)){let i=0;while(actual[i]===expected[i])i++;throw Error('memory mismatch at byte '+i+' actual='+actual[i]+' expected='+expected[i]);}
 console.log(JSON.stringify({all_memory_bits_equal:true,bytes:bin.length}));})().catch(e=>{console.error(e);process.exit(1)});
''')
 source=ROOT/'artifacts/rank343-mixed-scheme-screen-v1/WWC.py';ns={'__name__':'plan','__file__':str(source)};exec(compile(source.read_text(),str(source),'exec'),ns)
 a,b,_,leaves,_,_=ns['plan']();rng=np.random.default_rng(34388256);cases=[];files=[Path(__file__),source,d/'report.json',d/'256.wat',d/'256.wasm',runner,check/'test.wat',check/'test.wasm']
 for cols in [256,512]:
  blocks=cols//256;weights=rng.integers(-128,128,(256,cols),dtype=np.int16);sw=rng.uniform(.001,.1,256).astype(np.float32)
  wb=[]
  for block in range(blocks):
   raw=weights[:,block*256:(block+1)*256].reshape(8,4,8,8,32).transpose(3,2,0,1,4).reshape(64,8,4,32).astype(np.int64)
   bs=np.stack([sum(v*raw[i]for i,v in b.symbols[bn].items())for _,bn in leaves]);assert np.max(np.abs(bs))<=4096
   wb.append(bs.reshape(343,8,4,16,2).transpose(1,0,3,2,4).astype('<i2'))
  packed_w=np.stack(wb,axis=1)
  for n in [0,1,7,8,9,48,56,57,87,88,89,132]:
   groups=(n+7)//8;q=rng.integers(-127,128,(groups*8,cols),dtype=np.int16);q[n:]=0
   qb=[]
   for block in range(blocks):
    raw=q[:,block*256:(block+1)*256].reshape(groups,8,8,32).transpose(1,2,0,3).reshape(64,groups,32).astype(np.int64)
    aq=np.stack([sum(v*raw[i]for i,v in a.symbols[an].items())for an,_ in leaves]);assert aq.size==0 or np.max(np.abs(aq))<=4064;qb.append(aq.astype('<i2'))
   packed_q=np.stack(qb,axis=2);sx=rng.uniform(.001,.1,(n,blocks)).astype('<f4');stride=260
   initial=rng.standard_normal((n,stride)).astype('<f4');initial[:,::2]=-0.0;expected=initial.copy()
   for block in range(blocks):
    dot=q[:n,block*256:(block+1)*256].astype(np.int64)@weights[:,block*256:(block+1)*256].astype(np.int64).T
    term=np.multiply(np.multiply(dot.astype(np.float32),sx[:,block,None],dtype=np.float32),sw,dtype=np.float32)
    expected[:,:256]=np.add(expected[:,:256],term,dtype=np.float32)
   image=bytearray(b'\xa5'*64)
   def alloc(data):
    image.extend(b'\xa5'*((-len(image))%16));ptr=len(image);image.extend(data);return ptr
   wbase=alloc(packed_w.tobytes());qbase=alloc(packed_q.tobytes())
   wp=alloc(np.array([wbase+j*blocks*343*256 for j in range(8)],dtype='<u4').tobytes())
   qp=alloc(np.array([qbase+m*groups*blocks*64 for m in range(343)],dtype='<u4').tobytes())
   sxp=alloc(sx.tobytes());swp=alloc(sw.tobytes());out=alloc(initial.tobytes());image.extend(b'\xa5'*64)
   expectedimage=bytearray(image);expectedimage[out:out+expected.nbytes]=expected.tobytes()
   prefix=check/f'c{cols}-n{n}';inp=prefix.with_suffix('.input.bin');exp=prefix.with_suffix('.expected.bin');meta=prefix.with_suffix('.json')
   inp.write_bytes(image);exp.write_bytes(expectedimage);metadata=dict(wasm=str(check/'test.wasm'),input=str(inp),expected=str(exp),q=qp,w=wp,cols=cols,sx=sxp,stride=stride,sw=swp,out=out,n=n);meta.write_text(json.dumps(metadata))
   completed=subprocess.run(['node',str(runner),str(meta)],capture_output=True,text=True)
   if completed.returncode:raise RuntimeError(f'cols{cols}/n{n}: {completed.stderr}')
   result=json.loads(completed.stdout);assert result['all_memory_bits_equal'];cases.append(dict(cols=cols,tokens=n,**result));files.extend([inp,exp,meta]);print(json.dumps(cases[-1]),flush=True)
 report=dict(complete=True,conditions=len(cases),cases=cases,node_version=subprocess.check_output(['node','--version'],text=True).strip(),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},ic_execution_verified=False,performance_verified=False,scope='Actual generated Wasm execution on Node: full memory exact against ascending-block scalar integer dots and ordered F32 multiply/add. Test-only memory export; kernel bodies unchanged. Not IC metering or paid inference proof.')
 (d/'execution-v2-report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'execution-v2-evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'execution-v2-report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,conditions=len(cases))))
if __name__=='__main__':main()
