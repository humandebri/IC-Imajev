#!/usr/bin/env python3
"""Compile exact rank1127 tables without enormous expanded Rust prep loops.

The original expanded compilation remains running in its own directory.
Query kernel WAT, source model weights, F32 order and integer DAG are unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old = ROOT/'artifacts/rank1127-prepared-v1/build'
    evidence = ROOT/'artifacts/rank23-symbolic-v1'
    audit = json.loads((evidence/'rank1127-simd-layout-audit.json').read_text())
    assert audit['complete']
    assert all(sha(ROOT/p)==h for p,h in audit['hashes'].items())
    plan = json.loads((evidence/'mixed-2-3-2.json').read_text())
    d = ROOT/'artifacts/rank1127-generic-prep-v1/build'
    d.mkdir(parents=True,exist_ok=False)
    src = d/'src'
    shutil.copytree(old/'src',src)
    shutil.copyfile(old/'kernel.wat',d/'kernel.wat')
    prep = '''use core::arch::wasm32::*;
#[target_feature(enable="simd128")]
unsafe fn transform(values:&mut[v128],nodes:&[(usize,i8,usize,i8)]){
for(i,&(a,sa,b,sb))in nodes.iter().enumerate(){
let mut x=*values.get_unchecked(a);if sa<0{x=i16x8_sub(i16x8_splat(0),x);}
if sb!=0{let y=*values.get_unchecked(b);x=if sb>0{i16x8_add(x,y)}else{i16x8_sub(x,y)};}
*values.get_unchecked_mut(145+i)=x;}}
'''
    programs = {}
    maps = {}
    for side in ('a','b'):
        mapping = {f'{side}{i}':i+1 for i in range(144)}
        program = []
        for name,terms in plan['nodes'][side]:
            assert 1<=len(terms)<=2 and all(v in (-1,1) for v in terms.values())
            entries = list(terms.items())
            first,sign = entries[0]
            second,sign2 = entries[1] if len(entries)>1 else (None,0)
            program.append((mapping[first],sign,mapping[second] if second else 0,sign2))
            mapping[name] = 145+len(program)-1
        programs[side] = program
        maps[side] = mapping
        prep += f'const {side.upper()}_NODES:&[(usize,i8,usize,i8)]=&'+str(program).replace('[','[',1).replace(']',';',1) if False else ''
        prep += f'const {side.upper()}_NODES:&[(usize,i8,usize,i8)]=&['+','.join(f'({a},{sa},{b},{sb})' for a,sa,b,sb in program)+'];\n'
        column = 0 if side=='a' else 1
        prep += f'const {side.upper()}_LEAVES:&[usize]=&['+','.join(str(mapping[p[column]]) for p in plan['leaves'])+'];\n'
    prep += '''#[target_feature(enable="simd128")]
pub(super) unsafe fn input(q:&imajev_runtime::int8_kernel::QuantizedRows,block:usize)->Vec<i16>{
let groups=q.rows().div_ceil(12);let len=groups*1127*24;let mut out=Vec::<i16>::with_capacity(len);
let mut values=vec![i16x8_splat(0);145+A_NODES.len()];
for group in 0..groups{for k in 0..3{
for row in 0..12{for s in 0..12{let kk=s*22+k*8;let pos=(group*12+row)*q.cols()+block*256+kk;
let value=if group*12+row>=q.rows(){i16x8_splat(0)}else if k*8+7<22&&kk+7<256{v128_load(q.values().as_ptr().add(pos).cast())}else{
let tail:[i16;8]=core::array::from_fn(|x|if k*8+x<22&&kk+x<256{q.values()[pos+x]}else{0});v128_load(tail.as_ptr().cast())};values[1+row*12+s]=value;}}
transform(&mut values,A_NODES);
for(m,&index)in A_LEAVES.iter().enumerate(){v128_store(out.as_mut_ptr().add((group*1127+m)*24+k*8).cast(),*values.get_unchecked(index));}
}}out.set_len(len);out}
#[target_feature(enable="simd128")]
pub(super) unsafe fn weights(w:&[i8],rows:usize,cols:usize)->Vec<i16>{
let tiles=rows.div_ceil(12);let blocks=cols/256;let len=tiles*blocks*1127*24;let mut out=Vec::<i16>::with_capacity(len);
let mut values=vec![i16x8_splat(0);145+B_NODES.len()];
for tile in 0..tiles{for block in 0..blocks{for k in 0..3{
for s in 0..12{for col in 0..12{let row=tile*12+col;let kk=s*22+k*8;
let bytes:[i16;8]=core::array::from_fn(|x|if row<rows&&k*8+x<22&&kk+x<256{w[row*cols+block*256+kk+x]as i16}else{0});values[1+s*12+col]=v128_load(bytes.as_ptr().cast());}}
transform(&mut values,B_NODES);
for(m,&index)in B_LEAVES.iter().enumerate(){v128_store(out.as_mut_ptr().add(((tile*blocks+block)*1127+m)*24+k*8).cast(),*values.get_unchecked(index));}
}}}out.set_len(len);out}
'''
    # Independent coefficient expansion proves every table node and leaf equals the input DAG.
    for side in ('a','b'):
        vectors = [{}]+[{i:1} for i in range(144)]
        symbolic = {f'{side}{i}':{i:1} for i in range(144)}
        for (name,terms),(a,sa,b,sb) in zip(plan['nodes'][side],programs[side]):
            result = {}
            for index,sign in ((a,sa),(b,sb)):
                for key,value in vectors[index].items():
                    result[key] = result.get(key,0)+sign*value
            result = {k:v for k,v in result.items() if v}
            expected = {}
            for parent,sign in terms.items():
                for key,value in symbolic[parent].items():
                    expected[key] = expected.get(key,0)+sign*value
            expected = {k:v for k,v in expected.items() if v}
            assert result==expected
            symbolic[name] = expected
            vectors.append(result)
            assert maps[side][name]==len(vectors)-1
        assert all(vectors[maps[side][leaf[0 if side=='a' else 1]]]==symbolic[leaf[0 if side=='a' else 1]] for leaf in plan['leaves'])
    (src/'prepare_s3.rs').write_text(prep)
    prior = json.loads((ROOT/'artifacts/rank161-prepared-v1/build/report.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in prior['dependency_hashes'].items())
    command = prior['command'][:]
    command[command.index('--edition=2021')+1] = str(src/'lib.rs')
    command[command.index('-o')+1] = str(d/'raw.wasm')
    with (d/'compiler.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,**prior['explicit_env']),stdout=log,stderr=log,check=True)
    control = ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
    patcher = ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'
    previous = d/'raw.wasm'
    patches = []
    for file,symbol,output in ((control,'__imajev_s1_wide_accumulate',d/'control.wasm'),(d/'kernel.wat','__imajev_s3_stream_accumulate',d/'diagnostic.wasm')):
        patch = json.loads(subprocess.check_output([str(patcher),str(previous),str(file),str(output),symbol],text=True))
        assert patch['wasmparser_validation']
        patches.append(patch)
        previous = output
    paths = [Path(__file__),ROOT/'scripts/build_rank1127_prepared_probe.py',old/'kernel.wat',d/'kernel.wat',control,evidence/'rank1127-simd-layout-audit.json',evidence/'mixed-2-3-2.json',evidence/'upstream/LICENSE']+list(src.glob('*.rs'))
    result = dict(wasm_sha256=sha(previous),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},dependency_hashes=prior['dependency_hashes'],command=command,explicit_env=prior['explicit_env'],patches=patches,rank=1127,token_group=12,output_tile=12,locals=8685,emitted_integer_stack_identity_verified=True,query_kernel_byte_equal_to_expanded=True,table_nodes_and_leaves_symbolic_identity=True,diagnostic_only=True,scope=__doc__)
    (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(d/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps({'module':result['wasm_sha256'],'locals':8685}))


if __name__=='__main__':
    main()
