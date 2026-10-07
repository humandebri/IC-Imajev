#!/usr/bin/env python3
"""Inline single-use rank49 C nodes and remove root-copy locals in raw layout."""
from pathlib import Path
from collections import Counter
import hashlib,json,re,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reconstruction(c,roots):
 uses=Counter(n for row in roots for n in row)
 for _,terms in c.nodes:uses.update(terms.keys())
 nodes=dict(c.nodes);keep={n for n,_ in c.nodes if uses[n]>1 or n in {n for row in roots for n in row}}
 mapping={f'p{i}':i for i in range(49)}
 mapping.update({n:49+i for i,n in enumerate(n for n,_ in c.nodes if n in keep)})
 def emit(n):
  if n in mapping:return[f'local.get $pc{mapping[n]}']
  ops=[]
  for i,(parent,sign)in enumerate(nodes[n].items()):
   if i==0 and sign<0:ops.append('(v128.const i32x4 0 0 0 0)')
   ops+=emit(parent)
   if i or sign<0:ops.append('i32x4.'+('add'if sign>0 else'sub'))
  return ops
 lines=[];state={i:{i:1}for i in range(49)}
 for n,_ in c.nodes:
  if n not in keep:continue
  body=[]
  for i,(parent,sign)in enumerate(nodes[n].items()):
   if i==0 and sign<0:body.append('(v128.const i32x4 0 0 0 0)')
   body+=emit(parent)
   if i or sign<0:body.append('i32x4.'+('add'if sign>0 else'sub'))
  stack=[]
  for line in body:
   if line.startswith('local.get'):stack.append(dict(state[int(line.split('$pc')[1])]))
   elif line.startswith('(v128.const'):stack.append({})
   else:
    right=stack.pop();left=stack.pop();value=dict(left)
    for m,v in right.items():value[m]=value.get(m,0)+(v if line.endswith('add')else-v)
    stack.append({m:v for m,v in value.items()if v})
  assert len(stack)==1 and stack[0]==dict(c.symbols[n]);state[mapping[n]]=stack.pop();lines+=body+[f'local.set $pc{mapping[n]}']
 for row in roots:
  for n in row:assert state[mapping[n]]==dict(c.symbols[n])
 return lines,[[mapping[n]for n in row]for row in roots],len(mapping)
def remove_copies(text):
 pattern=re.compile(r'(?:local.get \$\w+\nlocal.set \$c[0-3]\n){4}')
 matches=list(pattern.finditer(text));parts=[];cursor=0
 for i,match in enumerate(matches):
  parts.append(text[cursor:match.start()]);mapping={c:p for p,c in re.findall(r'local.get (\$\w+)\nlocal.set (\$c[0-3])',match[0])};assert set(mapping)=={'$c0','$c1','$c2','$c3'}
  end=matches[i+1].start()if i+1<len(matches)else len(text);body=text[match.end():end]
  body=re.sub(r'\$c[0-3]\b',lambda m:mapping[m[0]],body);parts.append(body);cursor=end
 assert matches and cursor==len(text);return ''.join(parts),len(matches)
def main():
 d=ROOT/'artifacts/s2-inline-roots-kernels-v1';d.mkdir(exist_ok=False)
 original=ROOT/'scripts/generate_s2_k2_leaf_outer_kernels.py';gen={'__name__':'generator','__file__':str(original)};exec(compile(original.read_text(),str(original),'exec'),gen)
 plan=ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1/plan.py';ns={'__name__':'plan','__file__':str(plan)};exec(compile(plan.read_text(),str(plan),'exec'),ns)
 a,b,c,leaves,roots,_=ns['plan']();old_rec,_,old_cap=ns['reconstruct'](c,roots);rec,_,cap=reconstruction(c,roots);ns['reconstruct']=reconstruction
 files=[Path(__file__),original,plan];items=[]
 for tile in [96,16]:
  for seed in [False,True]:
   before,_,symbol=gen['kernel'](ns,tile,seed);text,copies=remove_copies(before)
   assert text.count('(')==text.count(')')
   def arithmetic(s):return[line for line in s.splitlines()if line.startswith(('i32x4.dot','f32x4.','v128.store','v128.load offset'))]
   assert arithmetic(before)==arithmetic(text)
   assert not re.search(r'local\.(get|set) \$c[0-3]\b',text)
   p=d/(str(tile)+('_seed'if seed else '')+'.wat');p.write_text(text);files.append(p);items.append(dict(path=str(p.relative_to(ROOT)),symbol=symbol,locals=text.count('(local $'),root_copy_groups_removed=copies,root_copy_instructions_removed=copies*8))
 report=dict(rank=49,kernels=items,previous_C_instruction_lines=len(old_rec),C_instruction_lines=len(rec),previous_register_capacity=old_cap,register_capacity=cap,all16_C_coefficient_identities_exact=True,per_node_signed_operand_order_preserved=True,weight_and_query_cache_layout_unchanged=True,raw_capacity_unchanged=True,root_copy_removal_F32_and_dot_ops_unchanged=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},wasm_execution_verified=False,performance_verified=False,scope=__doc__)
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps({k:report[k]for k in ['previous_C_instruction_lines','C_instruction_lines','register_capacity','kernels']}))
if __name__=='__main__':main()
