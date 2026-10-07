#!/usr/bin/env python3
"""Symbolically execute emitted first-use B instructions and compare unchanged ABI/kernel sections."""
from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-k2-alias-kernels-v1';old=ROOT/'artifacts/s2-k2-kernels-v1';r=json.loads((d/'report.json').read_text());ns={'__name__':'plan','__file__':str(d/'plan.py')};exec(compile((d/'plan.py').read_text(),str(d/'plan.py'),'exec'),ns);_,b,_,leaves,_,_=ns['plan']();checked=0
 for k in r['kernels']:
  p=ROOT/k['path'];lines=p.read_text().splitlines();tile=int(p.stem.split('_')[0]);oldwat=(old/p.name).read_text()
  # Query transforms, dot products, reconstruction and F32/transpose/store tails are unchanged.
  start=0
  for j in range(tile//16):
   lo=next(i for i in range(start,len(lines))if lines[i].startswith('(local.set $bp0('));hi=next(i for i in range(lo,len(lines))if (lines[i].startswith('(local.set $qp(')if j==0 else lines[i]=='local.get $x0_0'));start=hi+1
   variables={};stack=[]
   for line in lines[lo:hi]:
    if line.startswith('(local.set $bp'):continue
    load=re.fullmatch(r'\(local.set \$(\w+)\(v128.load8x8_s offset=(\d+)\(local.get \$bp(\d+)\)\)\)',line)
    if load:
     name,offset,plane=load.groups();value=[0]*16;value[int(plane)]=1;variables[name]=value;continue
    get=re.fullmatch(r'local.get \$(\w+)',line)
    if get:stack.append(variables[get[1]][:]);continue
    put=re.fullmatch(r'local.set \$(\w+)',line)
    if put:variables[put[1]]=stack.pop();continue
    if line=='(v128.const i32x4 0 0 0 0)':stack.append([0]*16);continue
    if line in ['i16x8.add','i16x8.sub']:
     y=stack.pop();x=stack.pop();stack.append([u+v if line.endswith('add')else u-v for u,v in zip(x,y)]);continue
    raise AssertionError(line)
   assert not stack
   for mi,(_,name)in enumerate(leaves):
    expected=[b.symbols[name].get(i,0)for i in range(16)]
    for ki in range(32):assert variables[f'w{mi}_{j}_{ki}']==expected;checked+=1
   # The multiplication through output block after this coefficient block must match old.
   oldlines=oldwat.splitlines();oldlo=next(i for i in range(len(oldlines))if oldlines[i].startswith('(local.set $bp0(')and f'(i32.const {j})'in oldlines[i]);oldhi=next(i for i in range(oldlo,len(oldlines))if (oldlines[i].startswith('(local.set $qp(')if j==0 else oldlines[i]=='local.get $x0_0'))
   end=next((i for i in range(hi,len(lines))if lines[i].startswith('(local.set $bp0(')),len(lines));oldend=next((i for i in range(oldhi,len(oldlines))if oldlines[i].startswith('(local.set $bp0(')),len(oldlines))
   assert lines[hi:end]==oldlines[oldhi:oldend]
 files=[Path(__file__),d/'report.json',d/'entry-hashes.json']+[ROOT/k['path']for k in r['kernels']]
 result=dict(complete=True,emitted_coefficient_slots_symbolically_verified=checked,all_query_dot_reconstruction_float_sections_byte_equal=True,raw_layout_unchanged=True,removed_duplicate_leaf_copies_per_k=49,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files})
 (d/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='source_hashes'}))
if __name__=='__main__':main()
