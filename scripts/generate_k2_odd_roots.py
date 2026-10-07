#!/usr/bin/env python3
"""Keep four cached Winograd dots, reconstruct only the valid odd-row roots."""
from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/k2-odd-roots-kernels-v1';d.mkdir(exist_ok=False);old=ROOT/'artifacts/update-k2-compact-v1';items=[];files=[Path(__file__)]
 canonical='''local.get $pc0
local.get $pc1
i32x4.add
local.set $pc7
local.get $pc0
local.get $pc5
i32x4.add
local.set $pc8
local.get $pc8
local.get $pc6
i32x4.add
local.set $pc9
local.get $pc8
local.get $pc4
i32x4.add
local.get $pc2
i32x4.add
local.set $pc10
local.get $pc9
local.get $pc3
i32x4.sub
local.set $pc11
local.get $pc9
local.get $pc4
i32x4.add
local.set $pc12
'''
 reduced='''local.get $pc0
local.get $pc1
i32x4.add
local.set $pc7
local.get $pc0
local.get $pc5
i32x4.add
local.get $pc2
i32x4.add
local.set $pc10
'''
 for tile in [128,168,160,32]:
  for seed in [False,True]:
   name=f'k2-direct{tile}'+('_seed'if seed else '')+'.wat';p=old/name;s=p.read_text();lo=s.index(')(else\n(local.set $qo')+len(')(else\n');end=s.index('(local.set $t(i32.add(local.get $t)(i32.const 2)))',lo);tail=s[lo:end];assert tail.count(canonical)==tile//8
   candidate=tail.replace(canonical,reduced)
   for m in [3,4,6]:
    zero=f'(local.set $pc{m}(v128.const i32x4 0 0 0 0))\n';assert candidate.count(zero)==tile//8;candidate=candidate.replace(zero,'')
   assert candidate.count('i32x4.dot_i16x8_s')==tail.count('i32x4.dot_i16x8_s')==4*64*(tile//8)
   text=s[:lo]+candidate+s[end:];assert text.count('(')==text.count(')');target=d/name;target.write_text(text);files.extend([p,target]);items.append(dict(tile=tile,seed=seed,path=str(target.relative_to(ROOT)),symbol=re.search(r'export "([^"]+)"',s)[1],locals=text.count('(local $')))
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r=dict(kernels=items,first_and_full_pair_paths_byte_equal=True,original_dots_byte_equal=True,original_float_stores_byte_equal=True,integer_roots='pc7=p0+p1; pc10=p0+p5+p2; p3/p4/p6 zero in odd tail',source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},performance_verified=False)
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(kernels=8,retained_products=[0,1,2,5])))
if __name__=='__main__':main()
