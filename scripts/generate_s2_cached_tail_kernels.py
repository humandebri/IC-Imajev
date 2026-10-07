#!/usr/bin/env python3
"""One remaining token uses original basis reconstructed from cached rank49 B leaves."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-cached-tail-kernels-v1';d.mkdir(exist_ok=False);original=ROOT/'artifacts/s2-contiguous-roots-kernels-v1/frozen-generator.py';bp=ROOT/'artifacts/s2-cached-leaf-basis-v1/report.json';br=json.loads(bp.read_text());assert br['all32_basis_identities_exact'];assert all(sha(ROOT/p)==h for p,h in br['source_hashes'].items())
 s=original.read_text();before='  return out\n lines+=';assert s.count(before)==1
 replacement='''  direct=['(local.set $qoff(i32.add(i32.shr_u(i32.mul(local.get $t)(local.get $cols))(i32.const 3))(i32.shr_u(local.get $start)(i32.const 1))))', '(local.set $sx0(v128.load32_splat(i32.add(local.get $sx)(i32.shl(i32.mul(local.get $t)(local.get $input_stride))(i32.const 2)))))', '(local.set $yp(i32.add(local.get $sums)(i32.mul(local.get $t)(local.get $output_bytes))))']
  for ki in range(4):
   am,sign=BASIS['A'][ki]['terms'][0];assert len(BASIS['A'][ki]['terms'])==1 and sign==1
   direct.append(f'(local.set $qp(i32.add(i32.load offset={am*4}(local.get $q))(local.get $qoff)))')
   for j in range(groups):
    for ni in range(4):
     if ki:direct.append(f'local.get $p{j}_{ni}')
     for k in range(32):
      direct.append(f'(local.tee $x{k}(v128.load32_splat offset={k*4}(local.get $qp)))'if j==0 and ni==0 else f'local.get $x{k}')
      terms=BASIS['B'][ki*4+ni]['terms']
      for ii,(bm,sgn)in enumerate(terms):
       if ii==0 and sgn<0:direct.append('v128.const i32x4 0 0 0 0')
       direct.append(f'local.get $w{bm}_{j}_{k}')
       if ii or sgn<0:direct.append('i16x8.add'if sgn>0 else'i16x8.sub')
      direct.append('i32x4.dot_i16x8_s')
      if ki or k:direct.append('i32x4.add')
     direct.append(f'local.set $p{j}_{ni}')
  for j in range(groups):
   for ni in range(4):
    offset=j*64+ni*16
    direct+=['local.get $yp']+(['v128.const i32x4 0 0 0 0']if seed else['local.get $yp',f'v128.load offset={offset}'])+[f'local.get $p{j}_{ni}','f32x4.convert_i32x4_s','local.get $sx0','f32x4.mul',f'local.get $sw{j*4+ni}','f32x4.mul','f32x4.add',f'v128.store offset={offset}']
  return ['(if(i32.eq(i32.sub(local.get $n)(local.get $t))(i32.const 1))(then']+direct+[')(else']+out+['))']
 lines+='''
 s=s.replace(before,replacement);frozen=d/'frozen-generator.py';frozen.write_text(s);gen=dict(__name__='generator',__file__=str(original),BASIS=br['bases']);exec(compile(s,str(frozen),'exec'),gen)
 plan=ROOT/'artifacts/s2-common-raw-kernels-v1/plan.py';(d/'plan.py').write_bytes(plan.read_bytes());ns=dict(__name__='plan',__file__=str(plan));exec(compile(plan.read_text(),str(plan),'exec'),ns);files=[Path(__file__),original,bp,frozen,plan,d/'plan.py'];kernels=[]
 for tile in [96,16]:
  for seed in [False,True]:
   text,count,symbol=gen['kernel'](ns,tile,seed)
   for m in range(16):
    before=f'(local.set $wp{m}(i32.add(i32.load offset={m*4}(local.get $w))(local.get $start)))';assert text.count(before)==1;text=text.replace(before,before.replace('(local.get $start)','(i32.shl(local.get $start)(i32.const 1))'))
   pattern=r'(\(local.set \$bp\d+\(i32.add\(local.get \$wp\d+\)\(i32.mul\(local.get \$cols\)\(i32.const )(\d+)(\)\)\)\))';text,n=re.subn(pattern,lambda m:m[1]+str(int(m[2])*4)+m[3],text);assert n==tile//16*16 and 'i8x16.shuffle'not in text
   p=d/(str(tile)+('_seed'if seed else'')+'.wat');p.write_text(text);files.append(p);kernels.append(dict(path=str(p.relative_to(ROOT)),tile=tile,seed=seed,symbol=symbol,locals=count))
 r=dict(complete=True,kernels=kernels,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Shared four-plane raw bank. One-token final quartet uses16 direct K/N products with cached-B basis reconstruction; all other quartets unchanged. Integer products/order changed in tail; ordered F32 block scale/add preserved. Performance and execution unverified.');(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(kernels=4,locals=[k['locals']for k in kernels])))
if __name__=='__main__':main()
