#!/usr/bin/env python3
"""Execute both real SIMD exports against independent matrix memory oracles."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from audit_rank2304_scaled_slp_basis import program

ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    source=ROOT/'artifacts/rank2304-modular-simd-v1'
    prior=ROOT/'artifacts/rank2304-latest-conditional-lift-v1'
    basis=ROOT/'artifacts/rank2304-scaled-slp-basis-v1'
    for report in [source/'report.json',basis/'report.json',prior/'report.json']:
        for p,h in json.loads(report.read_text())['source_hashes'].items():
            assert sha(ROOT/p)==h,p
    with np.load(basis/'coefficients.npz') as z:
        A,B=z['A'].astype(np.int64),z['B'].astype(np.int64)
    _,c=program(ROOT/'artifacts/rank48-latest-programs-v1/4x4x4_48_204_P.slp',48,1)
    d=source/'execution'
    d.mkdir(exist_ok=False)
    layout={}
    cursor=256
    for name,size in [('a',147456),('b',147456),('p',36864),('inner',12288),('outer',4096),('out',1024),('rawq',8192),('raww',8192),('flag',4)]:
        layout[name]=[cursor,size]
        cursor=(cursor+size+79)//16*16
    cases=[]
    for i in range(12):
        with np.load(prior/f'case-{i}.npz') as z:
            q,w,reference=[z[k].astype(np.int64) for k in ['q','w','expected']]
        av=A@q.reshape(256,16)
        bv=B@w.reshape(16,16,16).transpose(0,2,1).reshape(256,16)
        safe=bool(av.min()>=-32768 and av.max()<=32767 and bv.min()>=-32768 and bv.max()<=32767)
        for mode in ['universal32','checked16']:
            initial=bytearray([0xa5])*((cursor+65535)//65536*65536)
            def put(buf,name,data):
                p,n=layout[name]
                raw=data.tobytes()
                assert len(raw)==n,(name,len(raw),n)
                buf[p:p+n]=raw
            for name,data in [('a',av),('b',bv),('rawq',q),('raww',w)]:
                put(initial,name,data.astype('<i2' if name.startswith('raw') else '<i4'))
            expected=bytearray(initial)
            put(expected,'flag',np.array([2 if mode=='universal32' else int(safe)],dtype='<i4'))
            put(expected,'out',reference.astype('<i4'))
            if mode=='universal32' or safe:
                product=av*bv
                if mode=='universal32':
                    pv=product.reshape(2304,4,4).sum(axis=1)
                else:
                    pv=product.reshape(2304,2,4,2).sum(axis=(1,3))
                pv=pv.astype('<i4').astype(np.int64).reshape(48,48,4)
                inner=np.einsum('oi,mil->mol',c,pv).astype('<i4').astype(np.int64)
                outer=np.einsum('om,mil->oil',c,inner).astype('<i4')
                for name,data in [('p',pv),('inner',inner),('outer',outer)]:
                    put(expected,name,data.astype('<i4'))
            stem=f'{i}-{mode}'
            (d/f'{stem}.input.bin').write_bytes(initial)
            (d/f'{stem}.expected.bin').write_bytes(expected)
            cases.append(dict(index=i,mode=mode,safe=safe,stem=stem,pages=len(initial)//65536))
    config=dict(wasm=str(source/'kernel.wasm'),layout=layout,cases=cases,directory=str(d))
    (d/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    runner=d/'runner.cjs'
    runner.write_text('''const fs=require('fs');
const config=JSON.parse(fs.readFileSync(process.argv[2]));
const module_=new WebAssembly.Module(fs.readFileSync(config.wasm));
const results=[];
for(const c of config.cases){
 const memory=new WebAssembly.Memory({initial:c.pages});
 const inst=new WebAssembly.Instance(module_,{env:{memory}});
 const input=fs.readFileSync(`${config.directory}/${c.stem}.input.bin`);
 const expected=fs.readFileSync(`${config.directory}/${c.stem}.expected.bin`);
 new Uint8Array(memory.buffer).set(input);
 inst.exports[c.mode](...['a','b','p','inner','outer','out','rawq','raww','flag'].map(n=>config.layout[n][0]));
 const actual=Buffer.from(memory.buffer);
 fs.writeFileSync(`${config.directory}/${c.stem}.actual.bin`,actual);
 if(!actual.equals(expected)){
  let p=0;while(actual[p]===expected[p])p++;
  throw Error(`${c.stem}: first memory difference at ${p}, actual ${actual[p]}, expected ${expected[p]}`);
 }
 results.push({...c,all_memory_bytes_exact:true});
}
fs.writeFileSync(`${config.directory}/node-results.json`,JSON.stringify(results,null,2)+'\\n');
console.log(`PASS ${results.length} complete memory comparisons`);
''')
    run=subprocess.run(['node',str(runner),str(d/'config.json')],capture_output=True,text=True)
    (d/'stdout.txt').write_text(run.stdout)
    (d/'stderr.txt').write_text(run.stderr)
    run.check_returncode()
    for case in cases:
        assert (d/f"{case['stem']}.actual.bin").read_bytes()==(d/f"{case['stem']}.expected.bin").read_bytes()
    files=[Path(__file__),ROOT/'scripts/audit_rank2304_scaled_slp_basis.py',source/'report.json',basis/'report.json',prior/'report.json']+sorted(d.iterdir())
    report=dict(complete=True,wasm_execution_verified=True,ic_performance_verified=False,
                all24_complete_memory_comparisons_exact=True,unsafe_checked16_case_indices=[c['index'] for c in cases if c['mode']=='checked16' and not c['safe']],
                universal32_all12_including_i16_overflow_exact=True,
                source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
                scope='Prepared operands only. No full inference speed or paid-goal claim; input preparation must be counted.')
    (source/'execution-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(run.stdout,end='')

if __name__=='__main__': main()
