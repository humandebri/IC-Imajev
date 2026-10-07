#!/usr/bin/env python3
"""Compare current prepare8 to an address-hoisted, unrolled equivalent."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=ROOT/'artifacts/rank7-prepare8-probe-v1';b=json.loads((old/'build.json').read_text())
    for p,h in b['source_hashes'].items():assert sha(ROOT/p)==h,p
    parent=ROOT/'artifacts/update-rank7-prepare8-v1/build'
    pb=json.loads((parent/'report.json').read_text());source=parent/'runtime/strassen_raw.rs'
    assert sha(source)==pb['source_hashes'][str(source.relative_to(ROOT))]
    text=source.read_text();control=text[text.index('unsafe fn prepare('):text.index('\npub(crate) fn index')].replace('&QuantizedRows','&Q').replace('fn prepare(','fn fill8(')
    rust=(old/'probe.rs').read_text();a=rust.index('unsafe fn fill8(');z=rust.index('#[no_mangle]',a)
    assert rust[a:z].strip()==control.strip(),'control is not current best preparer'
    d=ROOT/'artifacts/rank7-prepare8-unrolled-probe-v1';d.mkdir(exist_ok=False)
    rust='\n'.join(line for line in rust.splitlines()if not line.startswith('#[export_name="canister_update prepare4_'))+'\n'
    rust=rust.replace('static mut OUT:[i16;259840]=[0;259840];','static mut OUT:[i16;1039360]=[0;1039360];')
    rust=rust.replace('if width==4{prepare4(q.as_ptr().cast(),cols,pairs,out,first,end);}else{prepare8(q.as_ptr().cast(),cols,pairs,out,first,end);}', 'if width==8{prepare8(q.as_ptr().cast(),cols,pairs,out,first,end);}else{prepare16(q.as_ptr().cast(),cols,pairs,out,first,end);}')
    for c in b['cases']:
        for name in ['input','expected']:(d/Path(c[name]).name).write_bytes((ROOT/c[name]).read_bytes())
    # A kernel block covers exactly128 elements. Four source pointers and seven
    # output bases stay fixed throughout sixteen independent eight-lane groups.
    unrolled='''#[no_mangle]pub unsafe extern "C" fn prepare16(q:*const i16,cols:usize,pairs:usize,out:*mut i16,first:usize,end:usize){
 if first>=end||pairs==0{return;}
 let stride=pairs*(cols/2);
 for pair in 0..pairs{for block in first..end{
 let qa=q.add(pair*2*cols+block*256);let qb=qa.add(128);let qc=qa.add(cols);let qd=qc.add(128);
 let po=pair*(cols/2)+block*128;
'''
    for m in range(7):unrolled+=f'let o{m}=out.add({m}*stride+po);\n'
    unrolled+='''macro_rules! chunk{($k:expr)=>{{
 let a=v128_load(qa.add($k).cast());let b=v128_load(qb.add($k).cast());let c=v128_load(qc.add($k).cast());let d=v128_load(qd.add($k).cast());
 let cd=i16x8_add(c,d);let cda=i16x8_sub(cd,a);
 v128_store(o0.add($k).cast(),a);v128_store(o1.add($k).cast(),b);v128_store(o2.add($k).cast(),i16x8_sub(b,cda));
 v128_store(o3.add($k).cast(),d);v128_store(o4.add($k).cast(),cd);v128_store(o5.add($k).cast(),cda);v128_store(o6.add($k).cast(),i16x8_sub(a,c));
}}}
'''+''.join(f'chunk!({k});'for k in range(0,128,8))+'\n}}}\n'
    rust+=unrolled
    cases=[]
    for c in b['cases']:
        c=dict(c)
        for name in ['input','expected']:c[name]=str((d/Path(c[name]).name).relative_to(ROOT))
        cases.append(c)
    for n,cols,first,end in [(0,256,0,1),(57,4096,0,16),(57,9216,0,36),(48,10240,8,19)]:
        i=len(cases);pairs=(n+1)//2
        q=np.random.default_rng(923+i).integers(-127,128,size=(pairs*2,cols),dtype=np.int16);q[n:]=0
        expected=np.full((7,pairs,cols//2),0x5a5a,dtype='<i2')
        for pair in range(pairs):
            for block in range(first,end):
                a,bv,c,dv=[q[pair*2+t,block*256+k*128:block*256+(k+1)*128].astype(np.int32)for t,k in [(0,0),(0,1),(1,0),(1,1)]]
                for m,v in enumerate([a,bv,a+bv-c-dv,dv,c+dv,-a+c+dv,a-c]):expected[m,pair,block*128:(block+1)*128]=v
        input_=d/f'input-{i}.bin';input_.write_bytes(q.astype('<i2').tobytes());ep=d/f'expected-{i}.bin';ep.write_bytes(expected.tobytes())
        rust+=f'#[repr(align(16))]struct Aligned{i}([u8;{input_.stat().st_size}]);\nstatic INPUT{i}:Aligned{i}=Aligned{i}(*include_bytes!("input-{i}.bin"));\n'
        rust+=f'#[export_name="canister_update prepare8_{i}"]pub unsafe extern "C" fn test8_{i}(){{measure(8,&INPUT{i}.0,{cols},{pairs},{first},{end});}}\n'
        cases.append(dict(index=i,n=n,cols=cols,pairs=pairs,first=first,end=end,input=str(input_.relative_to(ROOT)),expected=str(ep.relative_to(ROOT))))
    for c in cases:
        i=c['index'];rust+=f'#[export_name="canister_update prepare16_{i}"]pub unsafe extern "C" fn test16_{i}(){{measure(16,&INPUT{i}.0,{c["cols"]},{c["pairs"]},{c["first"]},{c["end"]});}}\n'
    (d/'probe.rs').write_text(rust)
    cmd=b['command'][:];cmd[cmd.index('--edition=2021')+1]=str(d/'probe.rs');cmd[cmd.index('-o')+1]=str(d/'probe.wasm')
    run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);(d/'compiler.log').write_text(run.stdout+run.stderr);run.check_returncode()
    (d/'probe.did').write_text('service:{\n'+''.join(f'prepare{width}_{c["index"]}:()->(nat64,blob);\n'for c in cases for width in [8,16])+'}\n')
    files=[Path(__file__),old/'build.json',parent/'report.json',source,d/'probe.rs',d/'probe.wasm',d/'probe.did',d/'compiler.log']+sorted(d.glob('*.bin'))
    (d/'build.json').write_text(json.dumps(dict(complete=True,cases=cases,command=cmd,current_best_prepare8_control_byteexact=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Rank7 preparation only; prepare16 is a name for16unrolled eight-lane groups, not sixteen-lane SIMD. No full inference or paid gain claim.'),indent=2)+'\n');print('compiled current prepare8 versus hoisted unrolled candidate')
if __name__=='__main__':main()
