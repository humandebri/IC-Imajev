#!/usr/bin/env python3
"""Replace rank7 odd-row four-leaf reconstruction with four original matrix products."""
from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parents[1]
def transform(text,tile):
 anchor=')(else\n(local.set $qo';lo=text.index(anchor)+len(')(else\n');end=text.index('(local.set $t(i32.add(local.get $t)(i32.const 2)))',lo)
 original=text[lo:end];prefix=original[:original.index('(local.set $qp(')];out=[prefix.rstrip()]
 for j in range(tile//8):
  # C00=a0*b0+a1*b1; C01=a0*(b0+b4)+a1*b2, with cached b4=B01-B00.
  for product,am in enumerate([0,1,0,1]):
   if j==0:out.append(f'(local.set $qp(i32.add(i32.load offset={am*4}(local.get $q))(local.get $qo)))')
   for k in range(64):
    if j==0 and product<2:out.append(f'(local.tee $x{am}_{k}(v128.load32_splat offset={k*4}(local.get $qp)))')
    else:out.append(f'local.get $x{am}_{k}')
    if product==2:out.extend([f'local.get $w0_{j}_{k}',f'local.get $w4_{j}_{k}','i16x8.add'])
    else:out.append(f'local.get $w{[0,1,0,2][product]}_{j}_{k}')
    out.append('i32x4.dot_i16x8_s')
    if k:out.append('i32x4.add')
   out.append(f'local.set $pc{product}')
  out.extend(['local.get $pc0','local.get $pc1','i32x4.add','local.set $pc7','local.get $pc2','local.get $pc3','i32x4.add','local.set $pc10'])
  # Preserve the exact original ascending-K F32 conversion/scales/add/store body.
  pattern=rf'local.get \$yp\n(?:.|\n)*?v128.store offset={j*32+16}\n'
  pos=original.index(f'v128.store offset={j*32}\n');start=original.rfind('local.get $yp\n',0,pos)
  # rfind finds the output load's second yp; step back one line for store address.
  if original[:start].endswith('local.get $yp\n'):start-=len('local.get $yp\n')
  finish=original.index(f'v128.store offset={j*32+16}\n',pos)+len(f'v128.store offset={j*32+16}\n')
  body=original[start:finish];assert body.count('f32x4.add')==2 and body.count('local.get $pc7')==2 and body.count('local.get $pc10')==2
  out.append(body.rstrip())
 tail='\n'.join(out)+'\n))\n';result=text[:lo]+tail+text[end:];assert result.count('(')==result.count(')');return result,original,tail

def main():
 d=ROOT/'artifacts/k2-odd-four-kernels-v1';d.mkdir(exist_ok=False);old=ROOT/'artifacts/update-k2-compact-v1';items=[];files=[Path(__file__)]
 for tile in [128,168,160,32]:
  for seed in [False,True]:
   name=f'k2-direct{tile}'+('_seed'if seed else '')+'.wat';p=old/name;text,original,tail=transform(p.read_text(),tile);target=d/name;target.write_text(text)
   assert tail.count('i32x4.dot_i16x8_s')==4*64*(tile//8)
   assert original.count('i32x4.dot_i16x8_s')==4*64*(tile//8)
   items.append(dict(tile=tile,seed=seed,path=str(target.relative_to(ROOT)),symbol=re.search(r'export "([^"]+)"',text)[1],original_tail_dots=4*64*(tile//8),candidate_tail_dots=4*64*(tile//8),locals=text.count('(local $')));files.extend([p,target])
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r=dict(kernels=items,first_and_full_pair_paths_byte_equal=True,integer_reduction='C00=A00*B00+A01*B10; C01=A00*(B00+(B01-B00))+A01*B11',source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},performance_verified=False)
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(items))
if __name__=='__main__':main()
