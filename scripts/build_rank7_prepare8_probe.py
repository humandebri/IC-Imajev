#!/usr/bin/env python3
"""Compile byte-preserved four-lane encoder and eight-lane revision together."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    base=ROOT/'artifacts/update-common-raw-hybrid-v1/build'
    b=json.loads((base/'report.json').read_text())
    source=base/'runtime/strassen_raw.rs'
    assert sha(source)==b['source_hashes'][str(source.relative_to(ROOT))]
    text=source.read_text();begin=text.index('unsafe fn prepare(');end=text.index('\npub(crate) fn index',begin)
    original=text[begin:end].replace('&QuantizedRows','&Q')
    revised=original.replace('step_by(4)','step_by(8)').replace('v128_load64_splat','v128_load').replace('v128_store64_lane::<0>','v128_store')
    revised=revised.replace('v128_store(v,out.add(m*pairs*(cols/2)+pair*(cols/2)+block*128+k).cast());','v128_store(out.add(m*pairs*(cols/2)+pair*(cols/2)+block*128+k).cast(),v);')
    assert revised!=original and revised.count('v128_load(')==4 and revised.count('v128_store(')==1
    d=ROOT/'artifacts/rank7-prepare8-probe-v1';d.mkdir(exist_ok=False)
    rust='''#![no_std]
use core::arch::wasm32::*;
#[panic_handler]fn panic(_: &core::panic::PanicInfo)->!{core::arch::wasm32::unreachable()}
struct Q<'a>{cols:usize,values:&'a[i16]}
impl Q<'_>{fn cols(&self)->usize{self.cols}fn values(&self)->&[i16]{self.values}}
'''+original.replace('fn prepare(','fn fill4(')+revised.replace('fn prepare(','fn fill8(')
    for width in [4,8]:
        rust+=f'''#[no_mangle]pub unsafe extern "C" fn prepare{width}(q:*const i16,cols:usize,pairs:usize,out:*mut i16,first:usize,end:usize){{
 let q=Q{{cols,values:core::slice::from_raw_parts(q,pairs*2*cols)}};fill{width}(&q,pairs,out,first,end);
}}
'''
    rust+='''#[link(wasm_import_module="ic0")]extern "C"{fn performance_counter(i:i32)->u64;fn msg_reply_data_append(p:i32,n:i32);fn msg_reply();}
static mut OUT:[i16;259840]=[0;259840];
unsafe fn measure(width:usize,q:&[u8],cols:usize,pairs:usize,first:usize,end:usize){
 let len=7*pairs*(cols/2);let out=core::ptr::addr_of_mut!(OUT).cast::<i16>();
 for i in 0..len{out.add(i).write(0x5a5a);}
 let start=performance_counter(0);
 if width==4{prepare4(q.as_ptr().cast(),cols,pairs,out,first,end);}else{prepare8(q.as_ptr().cast(),cols,pairs,out,first,end);}
 let used=performance_counter(0)-start;
 let mut h=[0u8;24];h[..10].copy_from_slice(&[68,73,68,76,1,109,123,2,120,0]);h[10..18].copy_from_slice(&used.to_le_bytes());
 let mut x=len*2;let mut at=18;loop{let mut v=(x&127)as u8;x>>=7;if x!=0{v|=128;}h[at]=v;at+=1;if x==0{break;}}
 msg_reply_data_append(h.as_ptr()as i32,at as i32);msg_reply_data_append(out as i32,(len*2)as i32);msg_reply();
}
'''
    cases=[]
    configurations=[(1,256,0,1),(3,512,1,2),(48,2560,0,10),(56,2560,0,10),(57,2560,0,10),(57,2560,3,7),(57,2560,10,10)]
    for i,(n,cols,first,end) in enumerate(configurations):
        pairs=(n+1)//2
        rng=np.random.default_rng(718+i)
        q=rng.integers(-127,128,size=(pairs*2,cols),dtype=np.int16)
        q[n:]=0
        q[0,:8]=[-127,127,0,-1,1,126,-126,100]
        if n>1:q[1,:8]=[127,-127,1,0,-1,-126,126,-100]
        input_=d/f'input-{i}.bin';input_.write_bytes(q.astype('<i2').tobytes())
        expected=np.full((7,pairs,cols//2),0x5a5a,dtype='<i2')
        for pair in range(pairs):
            for block in range(first,end):
                a=q[pair*2,block*256:block*256+128].astype(np.int32)
                bb=q[pair*2,block*256+128:(block+1)*256].astype(np.int32)
                c=q[pair*2+1,block*256:block*256+128].astype(np.int32)
                dd=q[pair*2+1,block*256+128:(block+1)*256].astype(np.int32)
                for m,v in enumerate([a,bb,a+bb-c-dd,dd,c+dd,-a+c+dd,a-c]):expected[m,pair,block*128:(block+1)*128]=v
        path=d/f'expected-{i}.bin';path.write_bytes(expected.tobytes())
        rust+=f'#[repr(align(16))]struct Aligned{i}([u8;{input_.stat().st_size}]);\nstatic INPUT{i}:Aligned{i}=Aligned{i}(*include_bytes!("input-{i}.bin"));\n'
        for width in [4,8]:
            rust+=f'#[export_name="canister_update prepare{width}_{i}"]pub unsafe extern "C" fn test{width}_{i}(){{measure({width},&INPUT{i}.0,{cols},{pairs},{first},{end});}}\n'
        cases.append(dict(index=i,n=n,cols=cols,pairs=pairs,first=first,end=end,input=str(input_.relative_to(ROOT)),expected=str(path.relative_to(ROOT))))
    (d/'probe.rs').write_text(rust)
    cmd=['rustc','--crate-name','prepare8_probe','--edition=2021',str(d/'probe.rs'),'--crate-type','cdylib','--target','wasm32-unknown-unknown','-C','opt-level=3','-C','target-feature=+simd128','-C','panic=abort','-o',str(d/'probe.wasm')]
    run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);(d/'compiler.log').write_text(run.stdout+run.stderr);run.check_returncode()
    (d/'probe.did').write_text('service:{\n'+''.join(f'prepare{width}_{c["index"]}:()->(nat64,blob);\n' for c in cases for width in [4,8])+'}\n')
    files=[Path(__file__),source,base/'report.json',d/'probe.rs',d/'probe.wasm',d/'probe.did',d/'compiler.log']+sorted(d.glob('*.bin'))
    (d/'build.json').write_text(json.dumps(dict(complete=True,cases=cases,command=cmd,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},scope='Exact rank7 operand preparation only. Allocation/quantization/projection/orchestration excluded; no full inference or paid gain claim.'),indent=2)+'\n')
    print('compiled original prepare4 and revised prepare8')
if __name__=='__main__':main()
