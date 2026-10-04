#!/usr/bin/env python3
"""Exact raw S1 widened output tile: input loads and token address work shared across more outputs."""
import argparse,hashlib,json,pathlib,ast,math
ROOT=pathlib.Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',required=True);ap.add_argument('--tile',type=int,choices=[64,128,256],required=True);a=ap.parse_args();tile=a.tile;quartets=tile//4;d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
base_source=(ROOT/'scripts/generate_wat_s1.py').read_text();tree=ast.parse(base_source)
# Load only the algebra constants and pure combine function; do not execute the
# original generator's filesystem side effects.
nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)and t.id in ('A','B','C')for t in n.targets) or isinstance(n,ast.FunctionDef)and n.name=='combine']
g={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/'scripts/generate_wat_s1.py'),'exec'),g)
for out,c in enumerate(g['C']):
 terms={}
 for m,v in c.items():
  for ai,av in g['A'][m].items():
   for bi,bv in g['B'][m].items():terms[ai,bi]=terms.get((ai,bi),0)+v*av*bv
 assert {k:v for k,v in terms.items()if v}=={(out//2*2+k,k*2+out%2):1 for k in range(2)}
 assert sum(abs(v)*sum(abs(x)for x in g['A'][m].values())*127*sum(abs(x)for x in g['B'][m].values())*128*128 for m,v in c.items())<2**31
old=(ROOT/'artifacts/s1_address_reuse/build/kernel.wat').read_text()
lines=old.splitlines()[:2]
for m in range(7):
 lines.append(f'(local $wp{m} i32) (local $p{m} v128)')
 for k in range(32):lines.append(f'(local $x{m}_{k} v128)')
 for j in range(quartets):
  for k in range(32):lines.append(f'(local $w{m}_{j}_{k} v128)')
for j in range(quartets):lines.append(f'(local $sw{j} v128)')
for j in range(quartets):
 for k in range(32):lines.append(f'(local $b12_{j}_{k} v128) (local $b21_{j}_{k} v128)')
for m in range(4):lines.append(f'(local $row{m} i32)')
for j in range(quartets):lines.append(f'(local.set $sw{j} (v128.load offset={j*16} (local.get $sw)))')
for m in range(4):lines.append(f'(local.set $wp{m} (i32.add (i32.load offset={m*4} (local.get $w)) (local.get $start)))')
def row(first):
 out=['(local.set $keep (i32.lt_u (i32.add (local.get $t) (i32.const 1)) (local.get $n)))',f'(local.set $yp (i32.add (local.get $sums) (i32.shl (local.get $t) (i32.const {int(math.log2(tile*4))}))))','(local.set $sx0 (v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (local.get $t) (local.get $stride)) (i32.const 2)))))','(if (local.get $keep) (then (local.set $sx1 (v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const 1)) (local.get $stride)) (i32.const 2)))))))']
 for j in range(quartets):
  if first:
   for b in range(4):out.append(f'(local.set $row{b} (i32.add (local.get $wp{b}) (i32.mul (local.get $cols) (i32.const {j}))))')
  for m in range(7):
   if j==0:out.append(f'(local.set $qp (i32.add (i32.load offset={m*4} (local.get $q)) (i32.add (i32.mul (local.get $t) (local.get $cols)) (i32.shl (local.get $start) (i32.const 1)))))')
   for k in range(32):
    out.append(f'(local.tee $x{m}_{k} (v128.load offset={k*16} (local.get $qp)))'if j==0 else f'local.get $x{m}_{k}')
    if first:
     def load(quarter):return f'(v128.load8x8_s offset={k*8} (local.get $row{quarter}))'
     if m==0:out += [f'(local.tee $w1_{j}_{k} {load(0)})',f'(local.tee $w4_{j}_{k} {load(3)})','i16x8.add',f'local.tee $w0_{j}_{k}']
     elif m==1:out.append(f'local.get $w1_{j}_{k}')
     elif m==2:out += [f'(local.tee $b12_{j}_{k} {load(2)})',f'local.get $w4_{j}_{k}','i16x8.sub',f'local.tee $w2_{j}_{k}']
     elif m==3:out += [f'(local.tee $b21_{j}_{k} {load(1)})',f'local.get $w1_{j}_{k}','i16x8.sub',f'local.tee $w3_{j}_{k}']
     elif m==4:out.append(f'local.get $w4_{j}_{k}')
     elif m==5:out += [f'local.get $w1_{j}_{k}',f'local.get $b12_{j}_{k}','i16x8.add',f'local.tee $w5_{j}_{k}']
     else:out += [f'local.get $b21_{j}_{k}',f'local.get $w4_{j}_{k}','i16x8.add',f'local.tee $w6_{j}_{k}']
    else:out.append(f'local.get $w{m}_{j}_{k}')
    out.append('i32x4.dot_i16x8_s')
    if k:out.append('i32x4.add')
   out.append(f'local.set $p{m}')
  for t in range(2):
   if t:out.append('(if (local.get $keep) (then')
   out+=['local.get $yp',f'v128.load offset={t*tile*4+j*16}']+g['combine'](g['C'][t*2])+['local.set $a']+g['combine'](g['C'][t*2+1])+['local.set $value','local.get $a','local.get $value','i8x16.shuffle 0 1 2 3 16 17 18 19 8 9 10 11 24 25 26 27','local.get $a','local.get $value','i8x16.shuffle 4 5 6 7 20 21 22 23 12 13 14 15 28 29 30 31','i32x4.add','f32x4.convert_i32x4_s',f'local.get $sx{t}','f32x4.mul',f'local.get $sw{j}','f32x4.mul','f32x4.add','local.set $value','local.get $yp','local.get $value',f'v128.store offset={t*tile*4+j*16}']
   if t:out.append('))')
 return out
lines+=['(local.set $t (i32.const 0))','(block $done','(br_if $done (i32.eqz (local.get $n)))']+row(True)+['(local.set $t (i32.const 2))','(loop $tokens','(br_if $done (i32.ge_u (local.get $t) (local.get $n)))']+row(False)+['(local.set $t (i32.add (local.get $t) (i32.const 2)))','br $tokens','))','))'];new='\n'.join(lines)+'\n';shared='(i32.add (i32.mul (local.get $t) (local.get $cols)) (i32.shl (local.get $start) (i32.const 1)))';assert new.count(shared)==14;new=new.replace(shared,'(local.get $qoffset)');anchor='(local.set $keep (i32.lt_u (i32.add (local.get $t) (i32.const 1)) (local.get $n)))';assert new.count(anchor)==2;new=new.replace(anchor,'(local.set $qoffset '+shared+')\n'+anchor);new=new.replace('__imajev_pair_accumulate','__imajev_s1_wide_accumulate');(d/'kernel.wat').write_text(new);sha=lambda b:hashlib.sha256(b).hexdigest();(d/'generator.json').write_text(json.dumps(dict(scope=__doc__,generator_sha256=sha(pathlib.Path(__file__).read_bytes()),original_kernel_sha256=sha(old.encode()),kernel_sha256=sha(new.encode()),symbolic_strassen_identity=True,tile=tile,fixed_bytes_ratio=1,derived_weight_vectors=7*quartets*32,raw_weight_vectors=4*quartets*32,input_vectors=7*32,first_weight_read='first token pair only',first_input_read='first output quartet only',late_reduce=True),indent=2)+'\n')
