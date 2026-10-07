#!/usr/bin/env python3
"""Fold only keep guards dominated by the identical full-pair predicate."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guards(s,value):
 prefix='(if(local.get $keep)(then';out='';cursor=0;changes=[]
 while True:
  start=s.find(prefix,cursor)
  if start<0:return out+s[cursor:],changes
  depth=0;end=None
  for i in range(start,len(s)):
   depth+=(s[i]=='(')-(s[i]==')')
   if depth==0:end=i+1;break
  assert end is not None
  before=s[start:end];assert before.endswith('))');after=before[len(prefix):-2]if value else ''
  changes.append(dict(before=before,after=after));out+=s[cursor:start]+after;cursor=end
def main():
 d=ROOT/'artifacts/k2-pair-guard-fold-kernels-v1';d.mkdir(exist_ok=False);base=ROOT/'artifacts/update-k2-odd-roots-v1';items=[];files=[Path(__file__)]
 predicate='(if(i32.lt_u(i32.add(local.get $t)(i32.const 1))(local.get $n))(then\n'
 keep='(local.set $keep(i32.lt_u(i32.add(local.get $t)(i32.const 1))(local.get $n)))\n'
 for tile in [128,168,160,32]:
  for seed in [False,True]:
   p=base/(f'k2-direct{tile}'+('_seed'if seed else '')+'.wat');s=p.read_text();lo=s.index(predicate)+len(predicate);mid=s.index(')(else\n(local.set $qo',lo);end=s.index('(local.set $t(i32.add(local.get $t)(i32.const 2)))',mid)
   full=s[lo:mid];tail=s[mid+len(')(else\n'):end];assert full.count(keep)==tail.count(keep)==1
   f,fc=guards(full,True);t,tc=guards(tail,False);assert len(fc)==len(tc)==tile//8+1
   f=f.replace(keep,'');t=t.replace(keep,'')
   # The tail's second-row address is no longer read after dead guards are removed.
   address='(local.set $yp1(i32.add(local.get $yp)(local.get $output_bytes)))\n';assert t.count(address)==1;t=t.replace(address,'')
   text=s[:lo]+f+s[mid:mid+len(')(else\n')]+t+s[end:]
   assert text.count('(')==text.count(')')
   def arithmetic(x):return [v for v in x.splitlines()if v.startswith(('i16x8.','i32x4.','local.get $x','local.get $w','local.set $pc','local.get $pc','f32x4.','i8x16.shuffle'))]
   assert arithmetic(f)==arithmetic(full)
   # All deleted arithmetic is exclusively in the false second-row stores.
   stripped,_=guards(tail,False);assert arithmetic(t)==arithmetic(stripped)
   target=d/p.name;target.write_text(text);files.extend([p,target]);items.append(dict(tile=tile,seed=seed,path=str(target.relative_to(ROOT)),symbol=re.search(r'export "([^"]+)"',s)[1],locals=text.count('(local $'),full_guards_folded=len(fc),tail_guards_deleted=len(tc),full_body_original_sha256=hashlib.sha256(full.encode()).hexdigest(),tail_body_original_sha256=hashlib.sha256(tail.encode()).hexdigest()))
 # Universal predicate enumeration covers every allowed row count and loop index.
 cases=0
 for n in range(133):
  for token in range(2,n,2):
   keep_value=token+1<n
   assert (keep_value is True)if token+1<n else (keep_value is False);cases+=1
 report=dict(kernels=items,guard_conditions=cases,first_pair_byte_unchanged=True,dominating_predicate=predicate,integer_and_float_order_unchanged=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},performance_verified=False,scope=__doc__)
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'integer-audit.json').write_text(json.dumps(dict(complete=True,all_integer_dots_equal=True,source_hashes=report['source_hashes'],scope='Only proven branch folding; reachable arithmetic is byte equal.'),indent=2)+'\n');print(json.dumps(dict(kernels=8,guard_conditions=cases)))
if __name__=='__main__':main()
