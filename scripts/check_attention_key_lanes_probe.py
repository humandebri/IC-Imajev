#!/usr/bin/env python3
"""Compare saved Wasm scores with an independent ascending-K F32 host oracle."""
from pathlib import Path
import hashlib,json,struct,subprocess,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/attention-key-lanes-v1/check';B=D.parent/'build'
TARGET='4xhad-gd777-77775-aaacq-cai'
HELPER=ROOT/'artifacts/f32-block-native/release/f32_args'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def oracle(q,k,prefix):
    n,width=q.shape;divisor=np.sqrt(np.float32(width),dtype=np.float32);rows=[]
    for t in range(n):
        keys=k[:prefix+t+1];s=np.full(len(keys),np.float32(-0.0),dtype='<f4')
        for j in range(width):s=np.add(s,np.multiply(q[t,j],keys[:,j],dtype=np.float32),dtype=np.float32)
        values=np.divide(s,divisor,dtype=np.float32);bits=values.view('<u4')
        rounded=(bits+np.uint32(0x7fff)+((bits>>np.uint32(16))&np.uint32(1)))&np.uint32(0xffff0000)
        rows.append(rounded.astype('<u4').tobytes())
    return b''.join(rows)


def main():
    D.mkdir(exist_ok=False);build=json.loads((B/'report.json').read_text())
    cmd=['icp','canister','status',TARGET,'--network','local','--identity','imajev-local','--json']
    status=json.loads(subprocess.check_output(cmd,text=True));assert status['status']=='Running' and status['module_hash'].removeprefix('0x')==build['module']
    did=D/'probe.did';did.write_text('service:{project:(vec nat8,nat8)->(record{digest:vec nat8;quantize_instructions:nat64;input_prepare_instructions:nat64;project_instructions:nat64;total_instructions:nat64;output_values:nat64;heap_pages:nat64}) query;}')
    hashes={**build['source_hashes'],**build['dependency_hashes'],str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str(HELPER.relative_to(ROOT)):sha(HELPER),str(did.relative_to(ROOT)):sha(did)}
    cases=[]
    shapes=[(1,1,0),(3,3,1),(7,7,3),(4,4,0),(5,8,2),(8,16,3),
            (1,256,131),(45,256,0),(48,256,27),(56,256,38),(57,256,27),
            (67,256,27),(87,256,45),(132,256,0),(9,255,4)]
    for seed,(n,width,prefix) in enumerate(shapes):
        rng=np.random.default_rng(5770+seed);x=rng.normal(0,.5,(2*n+prefix)*width).astype('<f4')
        x=(x.view('<u4')&np.uint32(0xffff0000)).view('<f4');x[::127]=np.float32(-0.)
        cases.append((f'shape-{n}-{width}-{prefix}',x[:n*width].reshape(n,width),x[n*width:].reshape(n+prefix,width),prefix,'synthetic BF16 operands'))
    for sign in [0,0x80000000]:
        x=np.full((2*7+3)*128,sign,dtype='<u4').view('<f4')
        cases.append((f'zero-{sign:08x}',x[:7*128].reshape(7,128),x[7*128:].reshape(10,128),3,'signed-zero sum identity'))
    x=np.resize(np.array([1,0x80000001,0x007fffff,0x807fffff,0x3f800000,0xbf800000],dtype='<u4'),18*128).view('<f4')
    cases.append(('subnormal-mixed',x[:9*128].reshape(9,128),x[9*128:].reshape(9,128),0,'finite F32 products and underflow'))
    captures=json.loads((ROOT/'artifacts/delta-capture-owned-profile-off-v1/proof/report.json').read_text())
    for name in ['617','620','653']:
        row=next(v for v in captures['captures'] if v['case']==name and v['layer']==0)
        p=Path(row['path']);assert sha(p)==row['sha256'];hashes[str(p.relative_to(ROOT))]=sha(p)
        raw=p.read_bytes();n=struct.unpack_from('<I',raw,4)[0];a=np.frombuffer(raw,dtype='<f4',offset=24+2*32*16384*4)
        q=a[:32*n*128].reshape(32,n,128)[0].copy();k=a[32*n*128:2*32*n*128].reshape(32,n,128)[0].copy()
        cases.append((f'delta-operand-{name}',q,k,0,'saved actual Delta Q/K F32 operands used only as score arithmetic fixtures; not an attention inference proof'))
    rows=[];paths=[]
    for label,q,k,prefix,scope in cases:
        n,width=q.shape;p=D/f'{label}.input.bin';p.write_bytes(struct.pack('<III',n,width,prefix)+q.astype('<f4').tobytes()+k.astype('<f4').tobytes())
        expected=oracle(q,k,prefix);ep=D/f'{label}.expected.bin';ep.write_bytes(expected);wanted=list(hashlib.sha256(expected).digest());measurements=[]
        for method in [0,1]:
            arg=D/f'{label}-{method}.args.bin';reply=D/f'{label}-{method}.hex'
            subprocess.run([str(HELPER),'query',str(method),str(p),str(arg)],check=True)
            raw=subprocess.check_output(['icp','canister','call',TARGET,'project','--args-file',str(arg),'--args-format','bin','--query','--network','local','--identity','imajev-local','--candid',str(did),'--output','hex'],text=True);reply.write_text(raw)
            m=json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement'],text=True))
            assert m['digest']==wanted,(label,method);assert m['output_values']==len(expected)//4
            assert m['total_instructions']==m['input_prepare_instructions']+m['project_instructions']
            measurements.append(dict(method=method,measurement=m,reply_sha256=sha(reply)));paths.extend([arg,reply])
        body=[v['measurement']['project_instructions']for v in measurements]
        row=dict(label=label,n=n,width=width,prefix=prefix,scope=scope,input_sha256=sha(p),oracle_sha256=sha(ep),measurements=measurements,body_reduction_percent=100*(1-body[1]/body[0]),bits_equal=True);rows.append(row);paths.extend([p,ep]);print(json.dumps(dict(label=label,body=body,reduction=row['body_reduction_percent'])),flush=True)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    for row in rows:
        for m in row['measurements']:
            reply=D/f"{row['label']}-{m['method']}.hex";assert sha(reply)==m['reply_sha256'];assert json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement'],text=True))==m['measurement']
    result=dict(complete=True,module=build['module'],cases=rows,queries=len(rows)*2,all_bits_equal=True,transpose_in_body=True,full_inference_verified=False,source_hashes=hashes,evidence_hashes={str(p.relative_to(ROOT)):sha(p)for p in paths},scope=__doc__)
    (D/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(D/'frozen-check.zip','w',zipfile.ZIP_DEFLATED)as z:
        for p in [ROOT/name for name in hashes]+paths+[D/'report.json']:
            z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps(dict(complete=True,cases=len(rows),queries=result['queries'],all_bits_equal=True)))


if __name__=='__main__':main()
