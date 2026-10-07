#!/usr/bin/env python3
"""Actual SIMD execution, including independently predicted staged scratch."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d = ROOT/'artifacts/rank7-batched256-v4'
    report = json.loads((d/'report.json').read_text())
    for p, h in report['source_hashes'].items():
        assert sha(ROOT/p) == h
    plan = ROOT/'artifacts/s1-winograd-v1/plan.py'
    ns = dict(__name__='plan', __file__=str(plan))
    exec(compile(plan.read_text(), str(plan), 'exec'), ns)
    a, b, _, leaves, _, _ = ns['plan']()
    outdir = d/'execution'
    outdir.mkdir(exist_ok=False)
    rng = np.random.default_rng(700256)
    cases = []
    files = [Path(__file__), plan, d/'report.json'] + [ROOT/p for p in report['source_hashes']]
    for cols in [256,512,2560]:
        blocks = cols//256
        rows = 256
        w = rng.integers(-128,128,(rows,cols),dtype=np.int16)
        rr, cc = np.indices(w.shape)
        plane = rows*cols//4
        idx = ((cc%256)//128*2+(rr%8)//4)*plane+(rr//8)*cols*2+(cc//256)*512+((cc%128)//2)*8+(rr%4)*2+cc%2
        assert np.array_equal(np.sort(idx.ravel()),np.arange(rows*cols))
        packed = np.empty(rows*cols,dtype=np.int8)
        packed[idx] = w
        sw = rng.uniform(.001,.1,rows).astype('<f4')
        for n in [0,1,2,3,48,56,57]:
            pairs = (n+1)//2
            q = rng.integers(-127,128,(pairs*2,cols),dtype=np.int16)
            q[n:] = 0
            compact = np.empty((7,pairs,cols//2),dtype='<i2')
            for pair in range(pairs):
                for block in range(blocks):
                    av = q[pair*2:pair*2+2,block*256:(block+1)*256].reshape(4,128).astype(np.int64)
                    for m,(an,_) in enumerate(leaves):
                        compact[m,pair,block*128:(block+1)*128] = sum(v*av[i] for i,v in a.symbols[an].items())
            sx = rng.uniform(.001,.1,(n,blocks)).astype('<f4')
            stride = rows+4
            initial = rng.standard_normal((n,stride)).astype('<f4')
            initial[:,::2] = -0.0
            for seed in [False,True]:
                image = bytearray(b'\xa5'*64)
                def alloc(data):
                    image.extend(b'\xa5'*((-len(image))%16))
                    address = len(image)
                    image.extend(data)
                    return address
                wb = alloc(packed.tobytes())
                qb = alloc(compact.tobytes())
                wp = alloc(np.array([wb+i*plane for i in [0,1,2,3]],dtype='<u4').tobytes())
                qp = alloc(np.array([qb+m*pairs*cols for m in range(7)],dtype='<u4').tobytes())
                sp = alloc(sx.tobytes())
                swp = alloc(sw.tobytes())
                op = alloc(initial.tobytes())
                scratch = alloc(b'\xa5'*(pairs*2048))
                image[qp+28:qp+32]=np.array([scratch],dtype='<u4').tobytes()
                image.extend(b'\xa5'*64)
                expected = initial.copy()
                for block in range(blocks):
                    dots = q[:n,block*256:(block+1)*256].astype(np.int64) @ w[:,block*256:(block+1)*256].astype(np.int64).T
                    term = np.multiply(np.multiply(dots.astype(np.float32),sx[:,block,None],dtype=np.float32),sw,dtype=np.float32)
                    prior = np.zeros((n,rows),dtype=np.float32) if seed and block==0 else expected[:,:rows]
                    expected[:,:rows] = np.add(prior,term,dtype=np.float32)
                oracle = bytearray(image)
                oracle[op:op+expected.nbytes] = expected.tobytes()
                # Predict exactly the last block's surviving four leaf products.
                block = blocks-1
                for pair in range(pairs):
                    for m,(_,bn) in enumerate(leaves[:4]):
                        if m==3 and pair*2+1>=n:
                            continue
                        av = compact[m,pair,block*128:(block+1)*128].astype(np.int64)
                        for j in range(32):
                            bv = []
                            for ni in range(2):
                                for ki in range(2):
                                    bv.append(w[j*8+ni*4:j*8+ni*4+4,block*256+ki*128:block*256+ki*128+128].astype(np.int64))
                            # B indices are K-major,N-minor.
                            bv = [bv[0],bv[2],bv[1],bv[3]]
                            leaf = sum(v*bv[i] for i,v in b.symbols[bn].items())
                            dot = (leaf @ av).astype('<i4').tobytes()
                            address = scratch+pair*2048+(m*32+j)*16
                            oracle[address:address+16] = dot
                base = outdir/f'c{cols}-n{n}-s{int(seed)}'
                ip = base.with_suffix('.input.bin')
                ep = base.with_suffix('.expected.bin')
                ip.write_bytes(image)
                ep.write_bytes(oracle)
                files += [ip,ep]
                cases.append(dict(cols=cols,n=n,seed=seed,q=qp,w=wp,sx=sp,sw=swp,out=op,stride=stride,scratch=scratch,input=str(ip),expected=str(ep)))
    runner = outdir/'run.js'
    runner.write_text("""const fs=require('fs');
(async()=>{const c=JSON.parse(fs.readFileSync(process.argv[2]));const memory=new WebAssembly.Memory({initial:1});const modules={};for(const [k,p]of Object.entries(c.modules))modules[k]=(await WebAssembly.instantiate(fs.readFileSync(p),{env:{memory}})).instance;for(const x of c.cases){const input=fs.readFileSync(x.input),expected=fs.readFileSync(x.expected);memory.grow(Math.max(0,Math.ceil(input.length/65536)-memory.buffer.byteLength/65536));new Uint8Array(memory.buffer).set(input);for(let k=0;k<x.cols;k+=256){const seed=x.seed&&k===0;modules[Number(seed)].exports['__imajev_rank7_batched256'+(seed?'_seed':'')](x.q,x.w,x.cols,k,x.sx+k/256*4,x.stride,x.sw,x.out,x.n,x.scratch);}const actual=Buffer.from(memory.buffer,0,input.length);if(!actual.equals(expected)){let i=0;while(actual[i]===expected[i])i++;throw Error(JSON.stringify({cols:x.cols,n:x.n,seed:x.seed,byte:i,actual:actual[i],expected:expected[i]}));}}console.log(JSON.stringify({complete:true,conditions:c.cases.length,all_memory_bits_equal:true}));})().catch(e=>{console.error(e);process.exit(1)});
""")
    config = outdir/'config.json'
    config.write_text(json.dumps(dict(modules={int(k['seed']):str(ROOT/k['wasm']) for k in report['kernels']},cases=cases)))
    result = subprocess.run(['node',str(runner),str(config)],capture_output=True,text=True)
    (outdir/'stdout.txt').write_text(result.stdout)
    (outdir/'stderr.txt').write_text(result.stderr)
    assert result.returncode==0, result.stderr
    assert json.loads(result.stdout)==dict(complete=True,conditions=len(cases),all_memory_bits_equal=True)
    files += [runner,config,outdir/'stdout.txt',outdir/'stderr.txt']
    verified = dict(complete=True,conditions=len(cases),all_memory_bits_equal=True,
                    scratch_allocation_and_stores_included_in_oracle=True,
                    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(files)},
                    ic_performance_verified=False,scope='Node SIMD only; no full inference or paid goal result')
    (d/'execution-report.json').write_text(json.dumps(verified,indent=2)+'\n')
    print(result.stdout)


if __name__=='__main__':
    main()
