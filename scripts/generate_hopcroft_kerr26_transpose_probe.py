#!/usr/bin/env python3
"""Exact transposed rank26: four tokens, four K parts, two output parts."""
import hashlib,json,re
from pathlib import Path
from collections import Counter
from generate_s3_stream_probe import DAG,expression
from generate_hopcroft_kerr26_probe import plan as original_plan
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/hopcroft-kerr26-transpose-v1/build'

def plan():
 oa,ob,oc,ol,orr,_=original_plan();a,b,c=DAG('a',16),DAG('b',8),DAG('p',26)
 def transfer(src,target,name,permutation):
  current='zero'
  for i,sign in sorted(src.symbols[name].items()):
   assert abs(sign)==1
   current=target.add(current,f'{target.prefix}{permutation(i)}',sign)
  return current
 leaves=[(transfer(ob,a,y,lambda i:(i%4)*4+i//4),transfer(oa,b,x,lambda i:(i%4)*2+i//4)) for x,y in ol]
 roots=[[transfer(oc,c,orr[j][i],lambda p:p) for j in range(2)] for i in range(4)]
 bound=0
 for dag,scale in [(a,127),(b,128)]:assert max(sum(abs(v) for v in e.values()) for e in dag.symbols.values())*scale<32768
 for e in c.symbols.values():
  estimate=sum(abs(s)*sum(abs(v) for v in a.symbols[leaves[m][0]].values())*127*sum(abs(v) for v in b.symbols[leaves[m][1]].values())*128*64 for m,s in e.items());bound=max(bound,estimate);assert estimate<2**31
 for i in range(4):
  for j in range(2):
   e={}
   for m,s in c.symbols[roots[i][j]].items():
    for ai,av in a.symbols[leaves[m][0]].items():
     for bi,bv in b.symbols[leaves[m][1]].items():e[ai,bi]=e.get((ai,bi),0)+s*av*bv
   assert {k:v for k,v in e.items() if v}=={(i*4+k,k*2+j):1 for k in range(4)}
 return a,b,c,leaves,roots,bound

def recombine(c,roots):
 uses=Counter(n for row in roots for n in row)
 for _,e in c.nodes:uses.update(e.keys())
 slots={f'p{i}':i for i in range(26)};free=[i for i in range(26) if not uses[f'p{i}']];count=26;code=[]
 for name,e in c.nodes:
  code+=expression(e,lambda n:f'local.get $p{slots[n]}','i32x4')
  for n in e:
   uses[n]-=1
   if not uses[n]:free.append(slots[n])
  if free:slot=free.pop()
  else:slot=count;count+=1
  slots[name]=slot;code+=[f'local.set $p{slot}']
 assert len({slots[n] for row in roots for n in row})==len({n for row in roots for n in row})
 return code,[[slots[n] for n in row] for row in roots],count

def main():
 D.mkdir(parents=True,exist_ok=False);a,b,c,leaves,roots,bound=plan();reconstruct,slots,count=recombine(c,roots)
 prepare='''#[cfg(target_arch="wasm32")] #[target_feature(enable="simd128")]
pub(crate) unsafe fn prepare(q:&imajev_runtime::int8_kernel::QuantizedRows,groups:usize,block:usize,out:*mut i16){use core::arch::wasm32::*;let cols=q.cols();for group in 0..groups {for k in(0..64).step_by(4){\n'''
 for i in range(16):prepare+=f'let a{i}=v128_load64_splat(q.values().as_ptr().add((group*4+{i//4})*cols+block*256+{(i%4)*64}+k).cast());\n'
 for name,e in a.nodes:
  values=list(e.items());first,sign=values[0];value=first if sign>0 else f'i16x8_sub(i16x8_splat(0),{first})'
  for name2,sign in values[1:]:value=f'i16x8_{"add" if sign>0 else "sub"}({value},{name2})'
  prepare+=f'let {name}={value};\n'
 for m,(name,_) in enumerate(leaves):prepare+=f'v128_store(out.add(({m}*groups+group)*128+k*2).cast(),{name});\n'
 prepare+='}}}\n';(D/'prepare_s2.rs').write_text(prepare)
 code=['(module (func (export "__imajev_s2_wide_stream_accumulate") '+' '.join(f'(param ${p} i32)' for p in ['q','w','cols','start','sx','stride','sw','sums','n']), '(local $t i32) (local $qp i32) (local $yp i32) (local $outstride i32) (local $v v128)']
 code += [f'(local $p{i} v128)' for i in range(count)]+[f'(local $bp{i} i32) (local $b{i} v128)' for i in range(8)]+[f'(local ${name} v128)' for name,_ in b.nodes]+[f'(local $x{m}_{k} v128)' for m in range(26) for k in range(16)]+[f'(local $w{j}_{m}_{k} v128)' for j in range(16) for m in range(26) for k in range(16)]+[f'(local $sx{i} v128)' for i in range(4)]+[f'(local $sw{i} v128)' for i in range(16)]
 code+=['(local.set $outstride (i32.load offset=104 (local.get $q)))']
 for i in range(16):code += [f'(local.set $sw{i} (v128.load offset={i*16} (local.get $sw)))']
 for j in range(16):
  for ki in range(4):
   for oi in range(2):
    i=ki*2+oi;plane=oi*2+ki//2;offset=(ki%2)*128
    code += [f'(local.set $bp{i} (i32.add (i32.load offset={plane*4} (local.get $w)) (i32.add (local.get $start) (i32.add (i32.mul (local.get $cols) (i32.const {j})) (i32.const {offset})))))']
  for k in range(16):
   for i in range(8):code += [f'(local.set $b{i} (i16x8.extend_low_i8x16_s (i8x16.shuffle 0 1 2 3 16 17 18 19 8 9 10 11 24 25 26 27 (v128.load32_zero offset={k*8} (local.get $bp{i})) (v128.load32_zero offset={k*8+4} (local.get $bp{i})))))']
   for name,e in b.nodes:code+=expression(e,lambda n:f'local.get ${n}','i16x8')+[f'local.set ${name}']
   for m,(_,name) in enumerate(leaves):code += [f'(local.set $w{j}_{m}_{k} (local.get ${name}))']
 code += ['(block $done (loop $tokens (br_if $done (i32.ge_u (local.get $t) (local.get $n)))']
 for i in range(4):code += [f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {i})) (local.get $n)) (then (local.set $sx{i} (v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {i})) (local.get $stride)) (i32.const 2)))))))']
 for j in range(16):
  for m in range(26):
   if j==0:code += [f'(local.set $qp (i32.add (i32.load offset={m*4} (local.get $q)) (i32.shl (i32.shr_u (local.get $t) (i32.const 2)) (i32.const 8))))']
   for k in range(16):code += [f'(local.tee $x{m}_{k} (v128.load offset={k*16} (local.get $qp)))' if j==0 else f'local.get $x{m}_{k}',f'local.get $w{j}_{m}_{k}','i32x4.dot_i16x8_s']+(['i32x4.add'] if k else [])
   code += [f'local.set $p{m}']
  code+=reconstruct
  for i in range(4):
   code += [f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {i})) (local.get $n)) (then',f'(local.set $yp (i32.add (local.get $sums) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {i})) (local.get $outstride)) (i32.const 2))))','local.get $yp',f'v128.load offset={j*16}']
   for indices in [[*range(0,4),*range(16,20),*range(8,12),*range(24,28)],[*range(4,8),*range(20,24),*range(12,16),*range(28,32)]]:code += [f'local.get $p{slots[i][0]}',f'local.get $p{slots[i][1]}','i8x16.shuffle '+' '.join(map(str,indices))]
   code += ['i32x4.add','f32x4.convert_i32x4_s',f'local.get $sx{i}','f32x4.mul',f'local.get $sw{j}','f32x4.mul','f32x4.add','local.set $v','local.get $yp','local.get $v',f'v128.store offset={j*16}', '))']
 code+=['(local.set $t (i32.add (local.get $t) (i32.const 4)))','br $tokens','))','))'];wat='\n'.join(code)+'\n';assert wat.count('(')==wat.count(')');locals=len(re.findall(r'\(local \$',wat));assert locals<10000;(D/'kernel.wat').write_text(wat)
 project=(ROOT/'artifacts/hopcroft-kerr26-v1/build/project.rs').read_text().replace('div_ceil(2)','div_ceil(4)').replace('rows%128','rows%64').replace('step_by(128)','step_by(64)');(D/'project.rs').write_text(project)
 stub=(ROOT/'artifacts/hopcroft-kerr26-v1/build/s2_kernel.rs').read_text().replace('0..128','0..64');(D/'s2_kernel.rs').write_text(stub)
 (D/'generator.json').write_text(json.dumps(dict(rank=26,shape=[4,4,2],output_tile=64,token_group=4,leaf_k=64,locals=locals,input_nodes=len(a.nodes),weight_nodes=len(b.nodes),reconstruction_nodes=len(c.nodes),max_integer_bound=bound,symbolic_identity=True,fixed_bytes_ratio=1,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n');print(locals)
if __name__=='__main__':main()
