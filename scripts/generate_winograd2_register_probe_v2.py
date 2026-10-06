#!/usr/bin/env python3
"""Generate rank49 Winograd with shared transforms and register-cached raw S1 weights."""
import hashlib,json,re
from pathlib import Path
from collections import Counter
from generate_s3_stream_probe import DAG,expression
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/winograd2-register-v2/build'
def plan():
 a,b,c=DAG('a',16),DAG('b',16),DAG('p',49);leaves=[]
 def mul(x,y):
  n=len(x)
  if n==1:
   m=len(leaves);leaves.append((x[0][0],y[0][0]));return [[f'p{m}']]
  h=n//2
  def split(z):return [[row[j:j+h] for row in z[i:i+h]] for i,j in [(0,0),(0,h),(h,0),(h,h)]]
  a11,a12,a21,a22=split(x);b11,b12,b21,b22=split(y)
  s1=a.matrix_add(a21,a22);s2=a.matrix_add(s1,a11,-1);s3=a.matrix_add(a11,a21,-1);s4=a.matrix_add(a12,s2,-1)
  t1=b.matrix_add(b12,b11,-1);t2=b.matrix_add(b22,t1,-1);t3=b.matrix_add(b22,b12,-1);t4=b.matrix_add(t2,b21,-1)
  p0=mul(a11,b11);p1=mul(a12,b21);p2=mul(s4,b22);p3=mul(a22,t4);p4=mul(s1,t1);p5=mul(s2,t2);p6=mul(s3,t3)
  u2=c.matrix_add(p0,p5);u3=c.matrix_add(u2,p6);u4=c.matrix_add(u2,p4)
  c11=c.matrix_add(p0,p1);c12=c.matrix_add(u4,p2);c21=c.matrix_add(u3,p3,-1);c22=c.matrix_add(u3,p4)
  return [x+y for x,y in zip(c11,c12)]+[x+y for x,y in zip(c21,c22)]
 roots=mul([[f'a{i*4+j}' for j in range(4)] for i in range(4)],[[f'b{i*4+j}' for j in range(4)] for i in range(4)])
 assert len(leaves)==49
 bound=0
 for e in c.symbols.values():
  estimate=sum(abs(sign)*sum(abs(v) for v in a.symbols[leaves[m][0]].values())*127*sum(abs(v) for v in b.symbols[leaves[m][1]].values())*128*64 for m,sign in e.items());bound=max(bound,estimate);assert estimate<2**31
 for dag,scale in [(a,127),(b,128)]:assert max(sum(abs(v) for v in e.values()) for e in dag.symbols.values())*scale<32768
 for i in range(4):
  for j in range(4):
   e={}
   for m,sign in c.symbols[roots[i][j]].items():
    for ai,av in a.symbols[leaves[m][0]].items():
     for bi,bv in b.symbols[leaves[m][1]].items():e[ai,bi]=e.get((ai,bi),0)+sign*av*bv
   assert {k:v for k,v in e.items() if v}=={(i*4+k,k*4+j):1 for k in range(4)}
 return a,b,c,leaves,roots,bound
def reconstruct(c,result):
 uses=Counter(v for row in result for v in row)
 for _,terms in c.nodes:uses.update(terms.keys())
 mapping={f'p{i}':i for i in range(49)};free=[i for i in range(49) if not uses[f'p{i}']];capacity=49;code=[]
 for name,terms in c.nodes:
  code+=expression(terms,lambda n:f'local.get $pc{mapping[n]}','i32x4')
  for parent in terms:
   uses[parent]-=1
   if not uses[parent]:free.append(mapping[parent])
  if free:slot=free.pop()
  else:slot=capacity;capacity+=1
  mapping[name]=slot;code+=[f'local.set $pc{slot}']
 assert len({mapping[n] for row in result for n in row})==len({n for row in result for n in row})
 return code,[[mapping[n] for n in row] for row in result],capacity
def main():
 D.mkdir(parents=True,exist_ok=False);a,b,c,leaves,result,bound=plan();recombine,roots,count=reconstruct(c,result)
 prepare='''#[cfg(target_arch="wasm32")] #[target_feature(enable="simd128")]
pub(crate) unsafe fn prepare(q:&imajev_runtime::int8_kernel::QuantizedRows,groups:usize,block:usize,out:*mut i16){use core::arch::wasm32::*;let cols=q.cols();for group in 0..groups {for k in (0..64).step_by(4){
'''
 for i in range(16):prepare+=f'let a{i}=v128_load64_splat(q.values().as_ptr().add((group*4+{i//4})*cols+block*256+{i%4*64}+k).cast());\n'
 for name,terms in a.nodes:
  entries=list(terms.items());x,sign=entries[0];value=x if sign>0 else f'i16x8_sub(i16x8_splat(0),{x})'
  for x,sign in entries[1:]:value=f'i16x8_{"add" if sign>0 else "sub"}({value},{x})'
  prepare+=f'let {name}={value};\n'
 for m,(name,_) in enumerate(leaves):prepare+=f'v128_store(out.add(({m}*groups+group)*128+k*2).cast(),{name});\n'
 prepare+='}}}\n';(D/'prepare_s2.rs').write_text(prepare)
 lines=['(module (func (export "__imajev_s2_wide_stream_accumulate") '+' '.join(f'(param ${p} i32)' for p in ['q','w','cols','start','sx','stride','sw','sums','n']), '(local $t i32) (local $qp i32) (local $yp i32) (local $outstride i32) (local $v v128)']
 lines += [f'(local $pc{i} v128)' for i in range(count)]+[f'(local $bp{i} i32) (local $b{i} v128)' for i in range(16)]+[f'(local ${name} v128)' for name,_ in b.nodes]
 lines += [f'(local $x{m}_{k} v128)' for m in range(49) for k in range(16)]+[f'(local $w{j}_{m}_{k} v128)' for j in range(8) for m in range(49) for k in range(16)]
 lines += [f'(local $sx{i} v128)' for i in range(4)]+[f'(local $sw{i} v128)' for i in range(16)]+['(local $h0 v128) (local $h1 v128)']
 lines+=['(local.set $outstride (i32.load offset=196 (local.get $q)))']
 for i in range(16):lines += [f'(local.set $sw{i} (v128.load offset={i*16} (local.get $sw)))']
 for j in range(8):
  for ci in range(4):
   for ri in range(4):
    plane=(ri%2)*2+ci//2;offset=(ci%2)*128+(ri%4)//2*4
    lines += [f'(local.set $bp{ci*4+ri} (i32.add (i32.load offset={plane*4} (local.get $w)) (i32.add (local.get $start) (i32.add (i32.mul (local.get $cols) (i32.const {j*2})) (i32.const {offset})))))']
  for k in range(16):
   for i in range(16):lines += [f'(local.set $b{i} (i16x8.extend_low_i8x16_s (i8x16.shuffle 0 1 2 3 16 17 18 19 8 9 10 11 24 25 26 27 (v128.load32_zero offset={k*8} (local.get $bp{i})) (v128.load32_zero offset={k*8} (i32.add (local.get $bp{i}) (local.get $cols))))))']
   for name,terms in b.nodes:lines+=expression(terms,lambda name:f'local.get ${name}','i16x8')+[f'local.set ${name}']
   for m,(_,name) in enumerate(leaves):lines += [f'(local.set $w{j}_{m}_{k} (local.get ${name}))']
 lines+=['(block $done (loop $tokens (br_if $done (i32.ge_u (local.get $t) (local.get $n)))']
 for i in range(4):lines += [f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {i})) (local.get $n)) (then (local.set $sx{i} (v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {i})) (local.get $stride)) (i32.const 2)))))))']
 def shuffle(x,y,idx):return [f'local.get ${x}',f'local.get ${y}','i8x16.shuffle '+' '.join(map(str,idx))]
 for j in range(8):
  for m in range(49):
   if j==0:lines += [f'(local.set $qp (i32.add (i32.load offset={m*4} (local.get $q)) (i32.shl (i32.shr_u (local.get $t) (i32.const 2)) (i32.const 8))))']
   for k in range(16):
    lines += [f'(local.tee $x{m}_{k} (v128.load offset={k*16} (local.get $qp)))' if j==0 else f'local.get $x{m}_{k}',f'local.get $w{j}_{m}_{k}','i32x4.dot_i16x8_s']+(['i32x4.add'] if k else [])
   lines += [f'local.set $pc{m}']
  lines+=recombine
  for ti in range(4):
   lines += [f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {ti})) (local.get $n)) (then',f'(local.set $yp (i32.add (local.get $sums) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {ti})) (local.get $outstride)) (i32.const 2))))']
   pcs=[f'pc{roots[ti][i]}' for i in range(4)]
   for i in range(2):
    lines+=shuffle(pcs[i*2],pcs[i*2+1],[*range(0,4),*range(16,20),*range(8,12),*range(24,28)])+shuffle(pcs[i*2],pcs[i*2+1],[*range(4,8),*range(20,24),*range(12,16),*range(28,32)])+['i32x4.add',f'local.set $h{i}']
   for half in range(2):
    offset=j*32+half*16;scale=j*2+half
    lines+=['local.get $yp',f'v128.load offset={offset}']+shuffle('h0','h1',list(range(half*8,half*8+8))+list(range(16+half*8,24+half*8)))+['f32x4.convert_i32x4_s',f'local.get $sx{ti}','f32x4.mul',f'local.get $sw{scale}','f32x4.mul','f32x4.add','local.set $v','local.get $yp','local.get $v',f'v128.store offset={offset}']
   lines+=['))']
 lines+=['(local.set $t (i32.add (local.get $t) (i32.const 4)))','br $tokens','))','))'];wat='\n'.join(lines)+'\n';(D/'kernel.wat').write_text(wat)
 original=(ROOT/'artifacts/s3-stream-v1/build/project.rs').read_text()
 project=original.replace('project_s3','project_s2').replace('S3','S2').replace('prepare_s3','prepare_s2').replace('s3_kernel','s2_kernel').replace('343','49').replace('344','50').replace('div_ceil(8)','div_ceil(4)').replace('groups*64','groups*128').replace('rows%32','rows%64').replace('step_by(32)','step_by(64)');(D/'project.rs').write_text(project)
 stub=(ROOT/'artifacts/s3-stream-v1/build/s3_kernel.rs').read_text().replace('__imajev_s3_stream_accumulate','__imajev_s2_wide_stream_accumulate').replace('add(343)','add(49)').replace('0..32','0..64');(D/'s2_kernel.rs').write_text(stub)
 assert wat.count('(')==wat.count(')'), 'WAT parenthesis balance'
 count=len(re.findall(r'\(local \$',wat));assert count<10000
 (D/'generator.json').write_text(json.dumps(dict(rank=49,output_tile=64,token_group=4,leaf_k=64,locals=count,input_nodes=len(a.nodes),weight_nodes=len(b.nodes),reconstruction_nodes=len(c.nodes),max_integer_bound=bound,symbolic_identity=True,fixed_bytes_ratio=1,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n');print(count)
if __name__=='__main__':main()
