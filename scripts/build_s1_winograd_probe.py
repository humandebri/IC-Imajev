#!/usr/bin/env python3
"""Rank7 Winograd using unchanged S1 raw weights and ordered F32 block accumulation."""
from pathlib import Path
import hashlib,json,os,re,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-winograd-v1';TILE=128
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(exist_ok=False);B=D/'build';S=B/'src';S.mkdir(parents=True)
 upstream=ROOT/'artifacts/s2-winograd-v1/frozen-plan.py'
 text=upstream.read_text().replace("DAG('a',16),DAG('b',16),DAG('p',49)","DAG('a',4),DAG('b',4),DAG('p',7)")
 text=text.replace('range(4)','range(2)').replace('*4+','*2+').replace('==49','==7').replace('range(49)','range(7)').replace('capacity=49','capacity=7').replace('49+i','7+i').replace('*128*64','*128*128')
 (D/'plan.py').write_text(text);ns={'__name__':'rank7','__file__':str(D/'plan.py')};exec(compile(text,str(D/'plan.py'),'exec'),ns)
 a,b,c,leaves,roots,_=ns['plan']();rec,regs,capacity=ns['reconstruct'](c,roots)
 bounds=[]
 for name,e in c.symbols.items():
  terms={}
  for m,sign in e.items():
   for ai,av in a.symbols[leaves[m][0]].items():
    for bi,bv in b.symbols[leaves[m][1]].items():terms[ai,bi]=terms.get((ai,bi),0)+sign*av*bv
  bound=sum(abs(v)for v in terms.values())*127*128*128;assert bound<2**31;bounds.append({'name':name,'bound':bound})
 proof=dict(rank=7,input_i16_bound=max(sum(abs(v)for v in e.values())for e in a.symbols.values())*127,weight_i16_bound=max(sum(abs(v)for v in e.values())for e in b.symbols.values())*128,bounds=bounds,roots_are_original_dots=True)
 assert proof['input_i16_bound']<32768 and proof['weight_i16_bound']<32768
 (D/'integer-bounds.json').write_text(json.dumps(proof,indent=2)+'\n')
 def rustexpr(e,names):
  return ''.join((''if i==0 and sign>0 else '+'if sign>0 else '-')+names[k]for i,(k,sign)in enumerate(sorted(e.items())))
 original=ROOT/'scripts/s1_wide_bench/src';code=(original/'exact.rs').read_text().replace('crate::kernel::','crate::win_kernel::')
 anchor='[b11+b22,b11,b12-b22,b21-b11,b22,b11+b12,b21+b22][m]'
 assert code.count(anchor)==1;code=code.replace(anchor,'['+','.join(rustexpr(b.symbols[y],['b11','b12','b21','b22'])for _,y in leaves)+'][m]')
 anchor='let d=[[p[0]+p[3]-p[4]+p[6],p[2]+p[4]],[p[1]+p[3],p[0]-p[1]+p[2]+p[5]]];'
 assert code.count(anchor)==1;code=code.replace(anchor,'let d=['+','.join('['+','.join(rustexpr(c.symbols[x],[f'p[{i}]'for i in range(7)])for x in row)+']'for row in roots)+'];')
 lo=code.index('pub fn operands(q:');code=code[:lo]
 code+='''pub fn operands(q:&QuantizedRows)->Operands{let pairs=q.rows().div_ceil(2);let cols=q.cols();let len=7*pairs*cols;
 #[cfg(target_arch="wasm32")]let data={let mut data=Vec::with_capacity(len);unsafe{inputs(q,pairs,data.as_mut_ptr());data.set_len(len);}data};
 #[cfg(not(target_arch="wasm32"))]let data={let mut data=vec![0;len];for pair in 0..pairs{for block in 0..cols/256{for k in 0..128{let p=pair*2*cols+block*256+k;let(a11,a12,a21,a22)=(q.values()[p],q.values()[p+128],q.values()[p+cols],q.values()[p+cols+128]);
 '''
 expressions=[rustexpr(a.symbols[x],['a11','a12','a21','a22'])for x,_ in leaves]
 code+='for(m,v)in ['+','.join(expressions)+'].into_iter().enumerate(){let i=m*pairs*cols+pair*cols+block*256+(k/4)*8+k%4;data[i]=v;data[i+4]=v;}}}}data};Operands{data,rows:q.rows(),cols,pairs}}\n'
 code+='''#[cfg(target_arch="wasm32")]#[target_feature(enable="simd128")]
 unsafe fn inputs(q:&QuantizedRows,pairs:usize,out:*mut i16){use core::arch::wasm32::*;let cols=q.cols();for pair in 0..pairs{for block in 0..cols/256{let p=q.values().as_ptr().add(pair*2*cols+block*256);for k in(0..128).step_by(4){
 let a0=v128_load64_splat(p.add(k).cast());let a1=v128_load64_splat(p.add(128+k).cast());let a2=v128_load64_splat(p.add(cols+k).cast());let a3=v128_load64_splat(p.add(cols+128+k).cast());
 '''
 for name,terms in a.nodes:
  entries=list(terms.items());value=entries[0][0];assert entries[0][1]==1
  for parent,sign in entries[1:]:value=f'i16x8_{"add"if sign>0 else "sub"}({value},{parent})'
  code+=f'let {name}={value};\n'
 for m,(name,_)in enumerate(leaves):code+=f'v128_store(out.add({m}*pairs*cols+pair*cols+block*256+k*2).cast(),{name});\n'
 code+='}}}}\n';(S/'winograd.rs').write_text(code)
 kernel=(original/'kernel.rs').read_text().replace('__imajev_s1_','__imajev_win7_').replace('__imajev_pair_','__imajev_win7_pair_')
 (S/'win_kernel.rs').write_text(kernel)
 lib=(original/'lib.rs').read_text().replace('pub mod exact;','pub mod exact;\nmod winograd;\n#[cfg(target_arch="wasm32")]mod win_kernel;')
 lib=lib.replace('sealed:bool}','win:Option<winograd::Prepared>,sealed:bool}').replace('sealed:false}','win:None,sealed:false}')
 lib=lib.replace('f.sealed=true;','f.win=Some(winograd::Prepared::new(&f.w,f.rows,f.cols).unwrap());f.sealed=true;')
 lib=lib.replace('assert!(method<=3)','assert!(method==3||method==4)').replace('(method>0).then(','(method==3).then(')
 lib=lib.replace('let prep=ic_cdk::api::performance_counter(0);','let wa=(method==4).then(||winograd::operands(&q));let prep=ic_cdk::api::performance_counter(0);')
 lib=lib.replace('let out=if method==3 {','let out=if method==4 {f.win.as_ref().unwrap().project_wide(&q,wa.as_ref().unwrap(),&f.scales[..rows],rows).unwrap()}else if method==3 {')
 lib+='\n#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}\n';(S/'lib.rs').write_text(lib)
 for name in ['kernel.rs','exact.rs']:(S/name).write_bytes((original/name).read_bytes())
 lines=['(module (func (export "__imajev_win7_wide_accumulate") (param $q i32)(param $w i32)(param $cols i32)(param $start i32)(param $sx i32)(param $stride i32)(param $sw i32)(param $sums i32)(param $n i32)', '(local $t i32)(local $qp i32)(local $yp i32)(local $qo i32)(local $sx0 v128)(local $sx1 v128)']
 for i in range(4):lines.append(f'(local $wp{i} i32)(local $b{i} v128)')
 for name,_ in b.nodes:lines.append(f'(local ${name} v128)')
 for i in range(capacity):lines.append(f'(local $pc{i} v128)')
 for m in range(7):
  for k in range(32):lines.append(f'(local $x{m}_{k} v128)')
  for j in range(TILE//4):
   for k in range(32):lines.append(f'(local $w{m}_{j}_{k} v128)')
 for j in range(TILE//4):lines.append(f'(local $sw{j} v128)')
 lines.append('(local $value v128)')
 for j in range(TILE//4):lines.append(f'(local.set $sw{j}(v128.load offset={j*16}(local.get $sw)))')
 for i in range(4):lines.append(f'(local.set $wp{i}(i32.add(i32.load offset={i*4}(local.get $w))(local.get $start)))')
 def row(first):
  # Input offsets count I16 elements; prepared weight offsets count bytes.
  out=['(local.set $qo(i32.shl(i32.add(i32.mul(i32.shr_u(local.get $t)(i32.const 1))(local.get $cols))(local.get $start))(i32.const 1)))']
  for ti in range(2):
   load=f'(local.set $sx{ti}(v128.load32_splat(i32.add(local.get $sx)(i32.shl(i32.mul(i32.add(local.get $t)(i32.const {ti}))(local.get $stride))(i32.const 2)))))'
   assert load.count('(')==load.count(')')
   out.append(f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti}))(local.get $n))(then {load}))')
  for j in range(TILE//4):
   if first:
    for k in range(32):
     for i,plane in enumerate([0,2,1,3]):out.append(f'(local.set $b{i}(v128.load8x8_s offset={k*8}(i32.add(local.get $wp{plane})(i32.mul(local.get $cols)(i32.const {j})))))')
     for name,terms in b.nodes:out+=ns['expression'](terms,lambda n:f'local.get ${n}','i16x8')+[f'local.set ${name}']
     for m,(_,name)in enumerate(leaves):out+=[f'local.get ${name}',f'local.set $w{m}_{j}_{k}']
   for m in range(7):
    if j==0:out.append(f'(local.set $qp(i32.add(i32.load offset={m*4}(local.get $q))(local.get $qo)))')
    for k in range(32):
     out+=[f'(local.tee $x{m}_{k}(v128.load offset={k*16}(local.get $qp)))'if j==0 else f'local.get $x{m}_{k}',f'local.get $w{m}_{j}_{k}','i32x4.dot_i16x8_s']
     if k:out+=['i32x4.add']
    out+=[f'local.set $pc{m}']
   out+=rec
   for ti in range(2):
    out += [f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti}))(local.get $n))(then',f'(local.set $yp(i32.add(local.get $sums)(i32.mul(i32.add(local.get $t)(i32.const {ti}))(i32.const {TILE*4}))))']
    out+=['local.get $yp',f'v128.load offset={j*16}']
    for mask in [[0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27],[4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31]]:
     out+=[f'local.get $pc{regs[ti][0]}',f'local.get $pc{regs[ti][1]}','i8x16.shuffle '+' '.join(map(str,mask))]
    out+=['i32x4.add','f32x4.convert_i32x4_s',f'local.get $sx{ti}','f32x4.mul',f'local.get $sw{j}','f32x4.mul','f32x4.add',f'local.set $pc0','local.get $yp','local.get $pc0',f'v128.store offset={j*16}', '))']
  return out
 # pc0 may be an output root. Use a separate float temporary, preserving all roots.
 lines+=['(block $done','(br_if $done(i32.eqz(local.get $n)))']
 r1=row(True);r2=row(False)
 for out in [r1,r2]:
  for i,x in enumerate(out):
   if x=='f32x4.add':assert out[i+1]=='local.set $pc0';out[i+1]='local.set $value';assert out[i+3]=='local.get $pc0';out[i+3]='local.get $value'
 lines+=r1+['(local.set $t(i32.const 2))','(loop $tokens','(br_if $done(i32.ge_u(local.get $t)(local.get $n)))']+r2+['(local.set $t(i32.add(local.get $t)(i32.const 2)))','br $tokens','))','))']
 wat=B/'kernel.wat';wat.write_text('\n'.join(lines)+'\n');assert wat.read_text().count('(')==wat.read_text().count(')');locals_count=sum(x.count('(local $')for x in lines);assert locals_count<10000
 old=json.loads((ROOT/'artifacts/s1_wide/raw128/report.json').read_text());cmd=old['command'][:];cmd[cmd.index('--edition=2021')+1]=str(S/'lib.rs');cmd[cmd.index('-o')+1]=str(B/'raw.wasm')
 assert all(sha(ROOT/p)==h for p,h in old['dependency_hashes'].items())
 with(B/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**old['explicit_env']),stdout=log,stderr=log,check=True)
 previous=B/'raw.wasm';patches=[];control=ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
 for file,symbol,out in [(control,'__imajev_s1_wide_accumulate',B/'control.wasm'),(wat,'__imajev_win7_wide_accumulate',B/'diagnostic.wasm')]:
  patch=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(file),str(out),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=out
 sources=[Path(__file__),upstream,D/'plan.py',D/'integer-bounds.json',wat,control]+list(S.glob('*.rs'))
 report=dict(wasm_sha256=sha(previous),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},dependency_hashes=old['dependency_hashes'],command=cmd,explicit_env=old['explicit_env'],patches=patches,locals=locals_count,output_tile=TILE,scope=__doc__)
 (B/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'module':report['wasm_sha256'],'locals':locals_count}))
if __name__=='__main__':main()
