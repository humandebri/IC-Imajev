#!/usr/bin/env python3
"""Generate exact rank343 block256 SIMD with streamed query-local operands.

Fixed weights retain the adopted S1 raw layout/capacity. Eight tokens and
32 output rows share a block. F32 scale and block-add order remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


class DAG:
    def __init__(self, prefix, count):
        self.prefix = prefix
        self.symbols = {f'{prefix}{i}':{i:1} for i in range(count)}
        self.symbols['zero'] = {}
        self.names = {tuple(v.items()):k for k,v in self.symbols.items()}
        self.nodes = []

    def add(self, a, b, sign=1):
        value = dict(self.symbols[a])
        for k,v in self.symbols[b].items():
            value[k] = value.get(k,0)+sign*v
        value = {k:v for k,v in value.items() if v}
        key = tuple(sorted(value.items()))
        if key in self.names: return self.names[key]
        name = f'{self.prefix}d{len(self.nodes)}'
        terms = Counter({a:1}); terms[b] += sign
        terms = {k:v for k,v in terms.items() if v and k!='zero'}
        assert all(v in [-1,1] for v in terms.values())
        self.nodes.append((name,terms)); self.symbols[name] = value; self.names[key] = name
        return name

    def matrix_add(self,a,b,sign=1):
        return [[self.add(x,y,sign) for x,y in zip(ar,br)] for ar,br in zip(a,b)]


def plan():
    a,b,c = DAG('a',64),DAG('b',64),DAG('p',343)
    leaves = []
    def multiply(x,y):
        n=len(x)
        if n==1:
            index=len(leaves); leaves.append((x[0][0],y[0][0])); return [[f'p{index}']]
        h=n//2
        def split(z): return [[row[j:j+h] for row in z[i:i+h]] for i,j in [(0,0),(0,h),(h,0),(h,h)]]
        a11,a12,a21,a22 = split(x); b11,b12,b21,b22 = split(y)
        p0=multiply(a.matrix_add(a11,a22),b.matrix_add(b11,b22))
        p1=multiply(a.matrix_add(a21,a22),b11)
        p2=multiply(a11,b.matrix_add(b12,b22,-1))
        p3=multiply(a22,b.matrix_add(b21,b11,-1))
        p4=multiply(a.matrix_add(a11,a12),b22)
        p5=multiply(a.matrix_add(a21,a11,-1),b.matrix_add(b11,b12))
        p6=multiply(a.matrix_add(a12,a22,-1),b.matrix_add(b21,b22))
        c11=c.matrix_add(c.matrix_add(c.matrix_add(p0,p3),p4,-1),p6)
        c12=c.matrix_add(p2,p4); c21=c.matrix_add(p1,p3)
        c22=c.matrix_add(c.matrix_add(c.matrix_add(p0,p1,-1),p2),p5)
        return [xx+yy for xx,yy in zip(c11,c12)]+[xx+yy for xx,yy in zip(c21,c22)]
    result=multiply([[f'a{i*8+j}' for j in range(8)] for i in range(8)],[[f'b{i*8+j}' for j in range(8)] for i in range(8)])
    assert len(leaves)==343
    assert max(sum(abs(v) for v in e.values()) for e in a.symbols.values())*127 < 32768
    assert max(sum(abs(v) for v in e.values()) for e in b.symbols.values())*128 < 32768
    max_bound=0
    for expression in c.symbols.values():
        bound=sum(abs(sign)*sum(abs(v) for v in a.symbols[leaves[m][0]].values())*127*
                  sum(abs(v) for v in b.symbols[leaves[m][1]].values())*128*32 for m,sign in expression.items())
        max_bound=max(max_bound,bound); assert bound<2**31
    for i in range(8):
        for j in range(8):
            expression={}
            for m,sign in c.symbols[result[i][j]].items():
                for ai,av in a.symbols[leaves[m][0]].items():
                    for bi,bv in b.symbols[leaves[m][1]].items():
                        expression[ai,bi]=expression.get((ai,bi),0)+sign*av*bv
            assert {k:v for k,v in expression.items() if v} == {(i*8+k,k*8+j):1 for k in range(8)}
    return a,b,c,leaves,result,max_bound


def expression(terms, get, kind):
    out=[]
    for i,(name,sign) in enumerate(terms.items()):
        if i==0 and sign<0: out.append('(v128.const i32x4 0 0 0 0)')
        out.append(get(name))
        if i or sign<0: out.append(f'{kind}.{"add" if sign>0 else "sub"}')
    return out or ['(v128.const i32x4 0 0 0 0)']


def reconstruct(c,result):
    roots=[v for row in result for v in row]
    uses=Counter(roots)
    for _,terms in c.nodes: uses.update(terms.keys())
    mapping={f'p{i}':i for i in range(343)}
    free=[i for i in range(343) if not uses[f'p{i}']]
    capacity=343; code=[]
    for name,terms in c.nodes:
        # Read both operands before recycling the register of a last use.
        code += expression(terms,lambda n:f'local.get $pc{mapping[n]}','i32x4')
        for parent in terms:
            uses[parent]-=1
            if uses[parent]==0: free.append(mapping[parent])
        if free: slot=free.pop()
        else: slot=capacity; capacity+=1
        mapping[name]=slot
        code.append(f'local.set $pc{slot}')
    assert len({mapping[n] for n in set(roots)}) == len(set(roots))
    return code,[[mapping[n] for n in row] for row in result],capacity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',default='artifacts/s3-stream-v1/build')
    d=ROOT/parser.parse_args().directory; d.mkdir(parents=True,exist_ok=False)
    a,b,c,leaves,result,max_bound=plan()
    recombine,roots,pc_count=reconstruct(c,result)
    prepare='''// Generated rank343, complete writes into one block-local allocation.
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub(crate) unsafe fn prepare(q:&imajev_runtime::int8_kernel::QuantizedRows,groups:usize,block:usize,out:*mut i16){
use core::arch::wasm32::*; let cols=q.cols();
for group in 0..groups {for k in (0..32).step_by(4) {
'''
    for i in range(64):
        prepare+=f'let a{i}=v128_load64_splat(q.values().as_ptr().add((group*8+{i//8})*cols+block*256+{i%8*32}+k).cast());\n'
    for name,terms in a.nodes:
        entries=list(terms.items()); x,sign=entries[0]; value=x if sign>0 else f'i16x8_sub(i16x8_splat(0),{x})'
        for x,sign in entries[1:]: value=f'i16x8_{"add" if sign>0 else "sub"}({value},{x})'
        prepare+=f'let {name}={value};\n'
    for m,(name,_) in enumerate(leaves):
        prepare+=f'v128_store(out.add(({m}*groups+group)*64+k*2).cast(),{name});\n'
    prepare+='}}}\n'; (d/'prepare_s3.rs').write_text(prepare)
    lines=['(module (func (export "__imajev_s3_stream_accumulate") '+
           ' '.join(f'(param ${p} i32)' for p in ['q','w','cols','start','sx','stride','sw','sums','n']),
           '(local $t i32) (local $qp i32) (local $yp i32) (local $outstride i32) (local $v v128)',
           '(local $zero v128)']
    for i in range(pc_count): lines.append(f'(local $pc{i} v128)')
    for i in range(64): lines.append(f'(local $bp{i} i32) (local $b{i} v128)')
    for name,_ in b.nodes: lines.append(f'(local ${name} v128)')
    for m in range(343):
        for k in range(8): lines.append(f'(local $x{m}_{k} v128)')
    for j in range(2):
        for m in range(343):
            for k in range(8): lines.append(f'(local $w{j}_{m}_{k} v128)')
    for i in range(8): lines.append(f'(local $sx{i} v128) (local $sw{i} v128)')
    for i in range(4): lines.append(f'(local $h{i} v128)')
    lines.append('(local.set $outstride (i32.load offset=1372 (local.get $q)))')
    for i in range(8): lines.append(f'(local.set $sw{i} (v128.load offset={i*16} (local.get $sw)))')
    # Derive weights once per invocation, sharing transforms across rank343.
    for j in range(2):
        for ci in range(8):
            for ri in range(8):
                plane=(ri%2)*2+ci//4; offset=(ci%4)*64+(ri%4)//2*4
                lines.append(f'(local.set $bp{ci*8+ri} (i32.add (i32.load offset={plane*4} (local.get $w)) (i32.add (local.get $start) (i32.add (i32.mul (local.get $cols) (i32.const {ri//4+j*4})) (i32.const {offset})))))')
        for k in range(8):
            for i in range(64):
                lines.append(f'(local.set $b{i} (i16x8.extend_low_i8x16_s (i8x16.shuffle 0 1 2 3 16 17 18 19 8 9 10 11 24 25 26 27 (v128.load32_zero offset={k*8} (local.get $bp{i})) (v128.load32_zero offset={k*8} (i32.add (local.get $bp{i}) (i32.shl (local.get $cols) (i32.const 1)))))))')
            for name,terms in b.nodes:
                lines+=expression(terms,lambda name:f'local.get ${name}','i16x8')+[f'local.set ${name}']
            for m,(_,name) in enumerate(leaves): lines.append(f'(local.set $w{j}_{m}_{k} (local.get ${name}))')
    lines+=['(local.set $t (i32.const 0))','(block $done (loop $tokens (br_if $done (i32.ge_u (local.get $t) (local.get $n)))']
    for i in range(8):
        lines.append(f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {i})) (local.get $n)) (then (local.set $sx{i} (v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {i})) (local.get $stride)) (i32.const 2)))))))')
    def shuffle(x,y,idx): return [f'local.get ${x}',f'local.get ${y}','i8x16.shuffle '+' '.join(map(str,idx))]
    for j in range(2):
        for m in range(343):
            if j==0: lines.append(f'(local.set $qp (i32.add (i32.load offset={m*4} (local.get $q)) (i32.shl (i32.shr_u (local.get $t) (i32.const 3)) (i32.const 7))))')
            for k in range(8):
                lines.append(f'(local.tee $x{m}_{k} (v128.load offset={k*16} (local.get $qp)))' if j==0 else f'local.get $x{m}_{k}')
                lines += [f'local.get $w{j}_{m}_{k}','i32x4.dot_i16x8_s']+(['i32x4.add'] if k else [])
            lines.append(f'local.set $pc{m}')
        lines+=recombine
        for ti in range(8):
            lines.append(f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {ti})) (local.get $n)) (then')
            lines.append(f'(local.set $yp (i32.add (local.get $sums) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {ti})) (local.get $outstride)) (i32.const 2))))')
            for group in range(2):
                pcs=[f'pc{roots[ti][group*4+i]}' for i in range(4)]
                for i in range(2):
                    lines+=shuffle(pcs[i*2],pcs[i*2+1],[*range(0,4),*range(16,20),*range(8,12),*range(24,28)])
                    lines+=shuffle(pcs[i*2],pcs[i*2+1],[*range(4,8),*range(20,24),*range(12,16),*range(28,32)])
                    lines+=['i32x4.add',f'local.set $h{i}']
                for half in range(2):
                    offset=j*64+group*16+half*32; scale=j*4+group+half*2
                    lines+=['local.get $yp',f'v128.load offset={offset}']
                    lines+=shuffle('h0','h1',list(range(half*8,half*8+8))+list(range(16+half*8,24+half*8)))
                    lines+=['f32x4.convert_i32x4_s',f'local.get $sx{ti}','f32x4.mul',f'local.get $sw{scale}','f32x4.mul','f32x4.add','local.set $v','local.get $yp','local.get $v',f'v128.store offset={offset}']
            lines.append('))')
    lines+=['(local.set $t (i32.add (local.get $t) (i32.const 8)))','br $tokens','))','))']
    wat='\n'.join(lines)+'\n'; (d/'kernel.wat').write_text(wat)
    project='''
pub fn project_s3(&self,q:&QuantizedRows,sw:&[f32],rows:usize)->Result<Vec<f32>> {
 if q.rows()==0||q.rows()>132||q.cols()!=self.cols||rows==0||rows%32!=0||rows>self.rows||q.rows()*rows>900_000||sw.len()!=rows||!sw.iter().all(|v|v.is_finite()&&*v>0.){return Err("S3 projection bounds".into());}
 #[cfg(not(target_arch="wasm32"))]return Err("Diagnostic S3 is Wasm only".into());
 #[cfg(target_arch="wasm32")]unsafe {
 let n=q.rows();let cols=q.cols();let groups=n.div_ceil(8);let mut out=vec![0f32;n*rows];
 let len=343*groups*64;let mut operands=Vec::<i16>::with_capacity(len);
 let mut ap=[0usize;344];for m in 0..343{ap[m]=operands.as_ptr().add(m*groups*64)as usize;}ap[343]=rows;
 for block in 0..cols/256 {
  crate::prepare_s3::prepare(q,groups,block,operands.as_mut_ptr());
  operands.set_len(len);
  for r in (0..rows).step_by(32) {
   let wp:[*const i8;4]=core::array::from_fn(|m|self.data.as_ptr().add(m*(self.rows/4)*cols+(r/4)*cols));
   crate::s3_kernel::accumulate(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),cols/256,sw.as_ptr().add(r),out.as_mut_ptr().add(r),n);
  }
 }
 if out.iter().any(|v|!v.is_finite()){return Err("S3 nonfinite output".into());}Ok(out)
 }
}
'''
    (d/'project.rs').write_text(project)
    stub='''// Unpatched body fails closed and has the exact strided write effect.
#[export_name="__imajev_s3_stream_accumulate"]
#[inline(never)]
pub(crate) unsafe extern "C" fn accumulate(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,out:*mut f32,n:usize){
 let rows=core::hint::black_box(q.cast::<usize>().add(343).read());
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(out as usize)^n)as u32;
 for t in 0..n{for i in 0..32{core::ptr::write_volatile(out.add(t*rows+i),f32::from_bits(marker|0x7fc00000));}}
}
'''
    (d/'s3_kernel.rs').write_text(stub)
    locals_count=len(re.findall(r'\(local \$',wat)); assert locals_count<=10000,locals_count
    identity=dict(rank=343,token_group=8,output_tile=32,leaf_k=32,locals=locals_count,
                  input_nodes=len(a.nodes),weight_nodes=len(b.nodes),reconstruction_nodes=len(c.nodes),reconstruction_registers=pc_count,
                  max_integer_reconstruction_bound=max_bound,symbolic_identity=True,fixed_bytes_ratio=1,
                  query_operand_bytes_per_block='343 * ceil(tokens/8) * 64 * 2',scope=__doc__)
    identity['generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity['generated_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}
    (d/'generator.json').write_text(json.dumps(identity,indent=2)+'\n'); print(json.dumps(identity))


if __name__=='__main__': main()
