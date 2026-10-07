#!/usr/bin/env python3
"""Stream leaves and inline single-use C expressions preserving operator order."""
from pathlib import Path
from collections import Counter
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/rank343-streaming-inline-reconstruction-v1';d.mkdir(exist_ok=False)
 source=ROOT/'artifacts/rank343-mixed-scheme-screen-v1/WWC.py';ns={'__name__':'plan','__file__':str(source)};exec(compile(source.read_text(),str(source),'exec'),ns)
 _,_,c,_,roots,_=ns['plan']();root_names=[n for row in roots for n in row];uses=Counter(root_names)
 for _,terms in c.nodes:uses.update(terms.keys())
 nodes=dict(c.nodes);keep={name for name,_ in c.nodes if uses[name]>1 or name in root_names}
 def expression(name,expand=False):
  if name not in nodes or name in keep and not expand:return [('get',name)]
  ops=[]
  for k,(parent,sign)in enumerate(nodes[name].items()):
   if k==0 and sign<0:ops.append(('zero',None))
   ops+=expression(parent)
   if k or sign<0:ops.append(('add'if sign>0 else'sub',None))
  return ops
 recipes={name:expression(name,True)for name,_ in c.nodes if name in keep}
 refs={name:Counter(arg for op,arg in recipe if op=='get')for name,recipe in recipes.items()}
 uses=Counter(root_names)
 for count in refs.values():uses.update(count)
 mapping={};state={};free=[];capacity=0;peak=0;events=[];pending=list(recipes)
 def allocate(name,value):
  nonlocal capacity,peak
  if free:slot=free.pop()
  else:slot=capacity;capacity+=1
  assert slot not in mapping.values();mapping[name]=slot;state[slot]=value;peak=max(peak,len(mapping));return slot
 def evaluate(ops):
  stack=[]
  for op,arg in ops:
   if op=='get':stack.append(dict(state[arg]))
   elif op=='zero':stack.append({})
   else:
    right=stack.pop();left=stack.pop();result=dict(left)
    for k,v in right.items():result[k]=result.get(k,0)+(v if op=='add'else-v)
    stack.append({k:v for k,v in result.items()if v})
  assert len(stack)==1;return stack[0]
 def drain():
  while True:
   ready=next((n for n in pending if all(p in mapping for p in refs[n])),None)
   if ready is None:return
   pending.remove(ready);ops=[(op,mapping[arg]if op=='get'else arg)for op,arg in recipes[ready]];value=evaluate(ops);assert value==dict(c.symbols[ready])
   for parent,count in refs[ready].items():
    uses[parent]-=count
    if uses[parent]==0:free.append(mapping.pop(parent))
   slot=allocate(ready,value);events.append(dict(kind='reconstruct',name=ready,instructions=ops,destination=slot))
 for m in range(343):
  name=f'p{m}';slot=allocate(name,{m:1});events.append(dict(kind='leaf',index=m,destination=slot));drain()
 assert not pending and set(mapping)==set(root_names)
 # Independent replay catches overwritten operands or stale slot reuse.
 state={}
 for e in events:
  if e['kind']=='leaf':state[e['destination']]={e['index']:1}
  else:
   value=evaluate(e['instructions']);assert value==dict(c.symbols[e['name']]);state[e['destination']]=value
 for row in roots:
  for name in row:assert state[mapping[name]]==dict(c.symbols[name])
 old,_,old_capacity=ns['reconstruct'](c,roots);count=sum(len(e['instructions'])+1 for e in events if e['kind']=='reconstruct');assert count==len(old)
 # The emitted expression operator/read order equals the original inline recipe.
 original_ops=[line for line in old if not line.startswith('local.set')];canonical=[]
 for name in recipes:
  for op,arg in recipes[name]:canonical.append(op)
 translated=['get'if line.startswith('local.get')else'zero'if line.startswith('(v128.const')else'add'if line.endswith('add')else'sub'for line in original_ops];assert canonical==translated
 report=dict(complete=True,scheme='WWC',register_capacity=capacity,peak_live_values=peak,all_leaves_first_capacity=old_capacity,reconstruction_instruction_lines=count,previous_streaming_lines=2800,events=events,root_registers=[[mapping[n]for n in row]for row in roots],all64_symbolic_outputs_exact=True,all_emitted_symbolic_intermediates_exact=True,inline_operator_order_unchanged=True,leaf_dot_order_unchanged=True,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),source]},wasm_execution_verified=False,performance_verified=False,scope=__doc__)
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in [Path(__file__),source,d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps({k:report[k]for k in ['complete','register_capacity','peak_live_values','reconstruction_instruction_lines','all64_symbolic_outputs_exact']}))
if __name__=='__main__':main()
