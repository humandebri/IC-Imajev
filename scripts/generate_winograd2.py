#!/usr/bin/env python3
"""Generate an exact two-level integer factorization and explicit Wasm leaf."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'scripts/winograd2_bench/src'
leaves = []

def combine(a, b, sign=1):
    r = dict(a)
    for k, v in b.items():
        r[k] = r.get(k, 0) + sign * v
    return {k: v for k, v in r.items() if v}

def matrix_add(a, b, sign=1):
    return [[combine(x, y, sign) for x, y in zip(ar, br)] for ar, br in zip(a, b)]

def multiply(a, b):
    n = len(a)
    if n == 1:
        m = len(leaves)
        leaves.append((a[0][0], b[0][0]))
        return [[{m: 1}]]
    h = n // 2
    split = lambda x: [
        [row[j:j+h] for row in x[i:i+h]]
        for i, j in [(0, 0), (0, h), (h, 0), (h, h)]
    ]
    a11, a12, a21, a22 = split(a)
    b11, b12, b21, b22 = split(b)
    s1=matrix_add(a21,a22);s2=matrix_add(s1,a11,-1);s3=matrix_add(a11,a21,-1);s4=matrix_add(a12,s2,-1)
    t1=matrix_add(b12,b11,-1);t2=matrix_add(b22,t1,-1);t3=matrix_add(b22,b12,-1);t4=matrix_add(t2,b21,-1)
    p0=multiply(a11,b11);p1=multiply(a12,b21);p2=multiply(s4,b22);p3=multiply(a22,t4)
    p4=multiply(s1,t1);p5=multiply(s2,t2);p6=multiply(s3,t3)
    u2=matrix_add(p0,p5);u3=matrix_add(u2,p6);u4=matrix_add(u2,p4)
    c11=matrix_add(p0,p1);c12=matrix_add(u4,p2);c21=matrix_add(u3,p3,-1);c22=matrix_add(u3,p4)
    return [x+y for x,y in zip(c11,c12)] + [x+y for x,y in zip(c21,c22)]

result = multiply([[{i*4+j:1} for j in range(4)] for i in range(4)],
                  [[{i*4+j:1} for j in range(4)] for i in range(4)])
assert len(leaves) == 49
# Prove every reconstructed coefficient equals the original 4x4 product.
for i in range(4):
    for j in range(4):
        terms = {}
        for m, c in result[i][j].items():
            for ai, av in leaves[m][0].items():
                for bi, bv in leaves[m][1].items():
                    key = (ai, bi)
                    terms[key] = terms.get(key, 0) + c*av*bv
        terms = {k:v for k,v in terms.items() if v}
        assert terms == {(i*4+k, k*4+j):1 for k in range(4)}
assert all(sum(abs(v) for v in x.values()) <= 16 for p in leaves for x in p)
# Bound the actual symbolic polynomial after each flattened reconstruction
# addition, including cancellation. Fixed/input transforms fit I16; each leaf
# and every explicitly shared U sum also have an exact integer identity.
polys=[]
for aa,bb in leaves:
    poly={}
    for ai,av in aa.items():
        for bi,bv in bb.items():poly[(ai,bi)]=poly.get((ai,bi),0)+av*bv
    polys.append(poly)
max_bound=0
for row in result:
    for terms in row:
        partial={}
        for m,c in sorted(terms.items()):
            partial=combine(partial,polys[m],c)
            bound=sum(abs(v) for v in partial.values())*127*128*64
            max_bound=max(max_bound,bound);assert bound<2**31
# Validate shared reconstruction intermediates recursively in the same order
# as the Wasm U nodes; these are bounds on original variable polynomials.
def check_shared(products):
    assert len(products)==7
    u2=combine(products[0],products[5]);u3=combine(u2,products[6]);u4=combine(u2,products[4])
    values=[combine(products[0],products[1]),combine(u4,products[2]),combine(u3,products[3],-1),combine(u3,products[4])]
    for p in products+[u2,u3,u4]+values:assert sum(abs(v) for v in p.values())*127*128*64<2**31
    return values
inner=[check_shared(polys[m*7:(m+1)*7]) for m in range(7)]
for j in range(4):check_shared([inner[m][j] for m in range(7)])

coeff = '// Generated exact symbolic factorization.\n'
for name, expressions in [('A',[p[0] for p in leaves]),('B',[p[1] for p in leaves])]:
    coeff += f'pub(super) const {name}: [&[(usize,i16)];49] = [\n'
    coeff += ''.join(' &['+','.join(f'({k},{v})' for k,v in sorted(e.items()))+'],\n' for e in expressions)
    coeff += '];\n'
coeff += 'pub(super) const C: [&[(usize,i32)];16] = [\n'
coeff += ''.join(' &['+','.join(f'({k},{v})' for k,v in sorted(e.items()))+'],\n' for row in result for e in row)
coeff += '];\n'
(DEST/'coeff.rs').write_text(coeff)

def expression(e):
    total = 'i16x8_splat(0)'
    for k,v in sorted(e.items()):
        assert v in [-1,1]
        total = f'i16x8_{"add" if v>0 else "sub"}({total},x{k})'
    return total

prep = '''// Generated complete writes, before Vec length is committed.
use core::arch::wasm32::*;
#[target_feature(enable="simd128")]
pub(super) unsafe fn weights(w:&[i8],rows:usize,cols:usize,out:*mut i16) {
 let stride=cols/4;let groups=rows/4;
 for group in 0..groups {for block in 0..cols/256 {
  let dst:[*mut i16;49]=core::array::from_fn(|m|out.add(m*groups*stride+group*stride+block*64));
  for k in (0..64).step_by(8) {
'''
for j in range(16):
    prep += f'   let x{j}=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+{j%4})*cols+block*256+{j//4*64}+k).cast()));\n'
for m,(_,b) in enumerate(leaves):
    prep += f'   v128_store(dst[{m}].add(k).cast(),{expression(b)});\n'
prep += '  }\n }}\n}\n'
prep += '''#[target_feature(enable="simd128")]
pub(super) unsafe fn inputs(q:&imajev_runtime::int8_kernel::QuantizedRows,groups:usize,out:*mut i16) {
 let cols=q.cols();let stride=cols/4;
 for group in 0..groups {for block in 0..cols/256 {
  let dst:[*mut i16;49]=core::array::from_fn(|m|out.add(m*groups*stride+group*stride+block*64));
  for k in (0..64).step_by(8) {
'''
for j in range(16):
    prep += f'   let x{j}=v128_load(q.values().as_ptr().add((group*4+{j//4})*cols+block*256+{j%4*64}+k).cast());\n'
for m,(a,_) in enumerate(leaves):
    prep += f'   v128_store(dst[{m}].add(k).cast(),{expression(a)});\n'
prep += '  }\n }}\n}\n'
(DEST/'prepare.rs').write_text(prep)

kernel = '''// Generated K64 leaf: 32 row quartets share each input vector.
#[target_feature(enable="simd128")]
pub(crate) unsafe fn dot_tile<const R:usize>(q:*const i16,w:*const i16,stride:usize)->[[i32;32];R] {
 use core::arch::wasm32::*;const{assert!(R<=32);}
 let qp:[*const i16;R]=core::array::from_fn(|i|q.add(i.wrapping_mul(stride)));
 let wp:[*const i16;32]=core::array::from_fn(|i|w.add(i.wrapping_mul(stride)));
 let weights=[
'''
for j in range(32):
    kernel += ' ['+','.join(f'v128_load(wp[{j}].add({k*8}).cast())' for k in range(8))+'],\n'
kernel += ' ];\n let mut out=[[0i32;32];R];\n'
dot = [f'i32x4_dot_i16x8($input[{k}],weights[$j][{k}])' for k in range(8)]
while len(dot)>1:
    dot = [f'i32x4_add({a},{b})' for a,b in zip(dot[::2],dot[1::2])]
kernel += ' macro_rules! dot {($input:ident,$j:literal)=>{'+dot[0]+'};}\n'
kernel += ' macro_rules! row {($i:literal)=>{if R>$i {\n let input=['+','.join(f'v128_load(qp[$i].add({k*8}).cast())' for k in range(8))+'];\n'
for g in range(8):
    j=g*4
    kernel += f''' {{let a0=dot!(input,{j});let a1=dot!(input,{j+1});let a2=dot!(input,{j+2});let a3=dot!(input,{j+3});
 let a=i32x4_add(i32x4_shuffle::<0,1,4,5>(a0,a1),i32x4_shuffle::<2,3,6,7>(a0,a1));
 let b=i32x4_add(i32x4_shuffle::<0,1,4,5>(a2,a3),i32x4_shuffle::<2,3,6,7>(a2,a3));
 let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
 v128_store(out[$i].as_mut_ptr().add({j}).cast(),total);}}
'''
kernel += ' }}}\n'+''.join(f' row!({i});\n' for i in range(32))+' out\n}\n'
(DEST/'kernel.rs').write_text(kernel)

tile = '''// Generated reconstruction. Validated complete spans never wrap.
use super::{Prepared,Operands};
use imajev_runtime::int8_kernel::QuantizedRows;
#[target_feature(enable="simd128")]
pub(super) unsafe fn evaluate<const R:usize>(w:&Prepared,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize,t:usize,r:usize,out:&mut[f32]) {
 use core::arch::wasm32::*;const{assert!(R<=32);}
 let stride=w.cols/4;let blocks=w.cols/256;let zero=f32x4_splat(0.);
 let mut sums=[[[[zero;8];4];4];R];
 let ap:[*const i16;49]=core::array::from_fn(|m:usize|a.data.as_ptr().add(m.wrapping_mul(a.groups).wrapping_mul(stride).wrapping_add((t/4).wrapping_mul(stride))));
 let wp:[*const i16;49]=core::array::from_fn(|m:usize|w.data.as_ptr().add(m.wrapping_mul(w.rows/4).wrapping_mul(stride).wrapping_add((r/4).wrapping_mul(stride))));
 let sp=q.scales().as_ptr().add(t.wrapping_mul(blocks));
'''
for j in range(32):
    tile += f' let sw{j}=v128_load(sw.as_ptr().add(r.wrapping_add({j*4})).cast());\n'
for g in range(8):
    for ri in range(4):
        tile += f' let sw{g}_{ri}={{let ab=i32x4_shuffle::<{ri},{ri+4},{ri},{ri+4}>(sw{g*4},sw{g*4+1});let cd=i32x4_shuffle::<{ri},{ri+4},{ri},{ri+4}>(sw{g*4+2},sw{g*4+3});i32x4_shuffle::<0,1,4,5>(ab,cd)}};\n'
tile += ' for block in 0..blocks {let start=block.wrapping_mul(64);\n let products=[\n'
for m in range(49):
    tile += f'  crate::kernel::dot_tile::<R>(ap[{m}].add(start),wp[{m}].add(start),stride),\n'
tile += ' ];\n macro_rules! row {($i:literal)=>{if R>$i {\n'
for ti in range(4):
    tile += f' let sx{ti}=f32x4_splat(*sp.add(($i*4usize+{ti}).wrapping_mul(blocks).wrapping_add(block)));\n'
for g in range(8):
    tile += ' {\n'
    for m in range(49):
        tile += f' let p{m}=v128_load(products[{m}][$i].as_ptr().add({g*4}).cast());\n'
    # Reconstruct the seven inner 2x2 products once. Reusing them removes
    # the repeated partial sums in a fully flattened 49-product expression.
    def reconstruct(v,prefix):
        global tile
        tile += f' let {prefix}_u2=i32x4_add({v[0]},{v[5]});\n'
        tile += f' let {prefix}_u3=i32x4_add({prefix}_u2,{v[6]});\n'
        tile += f' let {prefix}_u4=i32x4_add({prefix}_u2,{v[4]});\n'
        return [f'i32x4_add({v[0]},{v[1]})',f'i32x4_add({prefix}_u4,{v[2]})',f'i32x4_sub({prefix}_u3,{v[3]})',f'i32x4_add({prefix}_u3,{v[4]})']
    for m in range(7):
        expressions=reconstruct([f'p{m*7+k}' for k in range(7)],f's{m}')
        for j,d in enumerate(expressions):
            tile += f' let s{m}_{j}={d};\n'
    outer=[reconstruct([f's{m}_{j}' for m in range(7)],f'o{j}') for j in range(4)]
    for ti in range(4):
        for ri in range(4):
            j=(ti%2)*2+ri%2
            d=outer[j][(ti//2)*2+ri//2]
            dst=f'sums[$i][{ti}][{ri}][{g}]'
            tile += f' {dst}=f32x4_add({dst},f32x4_mul(f32x4_mul(f32x4_convert_i32x4({d}),sx{ti}),sw{g}_{ri}));\n'
    tile += ' }\n'
tile += ' }}}\n'+''.join(f' row!({i});\n' for i in range(32))+' }\n'
tile += ' macro_rules! store {($i:literal)=>{if R>$i {\n'
for ti in range(4):
    tile += f' {{let token=t.wrapping_add($i*4usize+{ti});if token<q.rows() {{let dst=out.as_mut_ptr().add(token.wrapping_mul(rows).wrapping_add(r));\n'
    for g in range(8):
        tile += f''' {{let a=sums[$i][{ti}][0][{g}];let b=sums[$i][{ti}][1][{g}];let c=sums[$i][{ti}][2][{g}];let d=sums[$i][{ti}][3][{g}];
 let ab0=i32x4_shuffle::<0,4,1,5>(a,b);let ab1=i32x4_shuffle::<2,6,3,7>(a,b);
 let cd0=i32x4_shuffle::<0,4,1,5>(c,d);let cd1=i32x4_shuffle::<2,6,3,7>(c,d);
 v128_store(dst.add({g*16}).cast(),i32x4_shuffle::<0,1,4,5>(ab0,cd0));
 v128_store(dst.add({g*16+4}).cast(),i32x4_shuffle::<2,3,6,7>(ab0,cd0));
 v128_store(dst.add({g*16+8}).cast(),i32x4_shuffle::<0,1,4,5>(ab1,cd1));
 v128_store(dst.add({g*16+12}).cast(),i32x4_shuffle::<2,3,6,7>(ab1,cd1));}}
'''
    tile += ' }}\n'
tile += ' }}}\n'+''.join(f' store!({i});\n' for i in range(32))+'}\n'
(DEST/'tile.rs').write_text(tile)
print(json.dumps({'leaf_products':49,'original_products':64,
 'max_reconstruction_terms':max(len(e) for row in result for e in row),
 'symbolic_identity_verified':True,'max_flat_intermediate_bound':max_bound,'max_input_abs':max(sum(abs(v) for v in a.values())*127 for a,b in leaves),'max_weight_abs':max(sum(abs(v) for v in b.values())*128 for a,b in leaves)}))
