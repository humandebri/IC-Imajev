#!/usr/bin/env python3
"""Interleave each exact leaf dot with ready integer DAG nodes and recycle slots.

No Wasm performance claim. Leaf order and each C node's signed operand order
are preserved; only integer reconstruction moves earlier, modulo2**32.
"""
from pathlib import Path
from collections import Counter
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/rank343-streaming-reconstruction-v1';d.mkdir(exist_ok=True);assert not (d/'report.json').exists()
 source=ROOT/'artifacts/rank343-mixed-scheme-screen-v1/WWC.py'
 ns={'__name__':'plan','__file__':str(source)};exec(compile(source.read_text(),str(source),'exec'),ns)
 a,b,c,leaves,roots,_=ns['plan']();root_names=[n for row in roots for n in row]
 uses=Counter(root_names)
 for _,terms in c.nodes:uses.update(terms.keys())
 pending=list(c.nodes);mapping={};registers={};free=[];capacity=0;events=[];live_peak=0
 def allocate(name,value):
  nonlocal capacity,live_peak
  if free:slot=free.pop()
  else:slot=capacity;capacity+=1
  assert slot not in mapping.values()
  mapping[name]=slot;registers[slot]=value;live_peak=max(live_peak,len(mapping));return slot
 def drain():
  while True:
   found=None
   for i,(name,terms)in enumerate(pending):
    if all(parent in mapping for parent in terms):found=i;break
   if found is None:return
   name,terms=pending.pop(found);operands=[dict(name=p,slot=mapping[p],sign=sign)for p,sign in terms.items()]
   value={}
   for p,sign in terms.items():
    for m,coefficient in registers[mapping[p]].items():value[m]=value.get(m,0)+sign*coefficient
   value={m:v for m,v in value.items()if v};assert value==dict(c.symbols[name])
   for p in terms:
    uses[p]-=1
    if uses[p]==0:free.append(mapping.pop(p))
   slot=allocate(name,value);events.append(dict(kind='reconstruct',name=name,operands=operands,destination=slot))
 for m in range(343):
  name=f'p{m}';slot=allocate(name,{m:1});events.append(dict(kind='leaf',index=m,destination=slot));drain()
 assert not pending and set(mapping)==set(root_names)
 assert len(set(mapping.values()))==len(set(root_names))
 for row in roots:
  for name in row:assert registers[mapping[name]]==dict(c.symbols[name])
 # Re-run emitted event stream independently, checking recycled read/write slots.
 state={}
 for event in events:
  if event['kind']=='leaf':state[event['destination']]={event['index']:1};continue
  result={}
  for operand in event['operands']:
   for m,v in state[operand['slot']].items():result[m]=result.get(m,0)+operand['sign']*v
  result={m:v for m,v in result.items()if v};assert result==dict(c.symbols[event['name']]);state[event['destination']]=result
 for row in roots:
  for name in row:assert state[mapping[name]]==dict(c.symbols[name])
 old_rec,_,old_capacity=ns['reconstruct'](c,roots)
 rec_lines=sum(len(e['operands'])+len(e['operands'])-1+1+int(e['operands'][0]['sign']<0)*2 for e in events if e['kind']=='reconstruct')
 candidates=[]
 for groups in [4,6,8,12]:
  for tokens in [7,8,11,14,17]:
   products=capacity*groups*tokens
   overhead=16*groups+16+tokens*8+groups*8+groups+6
   candidates.append(dict(output_tile=groups*32,token_capacity=tokens*8,integer_slots=products,
      conservative_extra_locals=overhead,total_locals=products+overhead,fits10000=products+overhead<10000))
 report=dict(complete=True,scheme='WWC',events=events,root_registers=[[mapping[n]for n in row]for row in roots],
  all64_coefficient_identities_exact=True,all_emitted_event_coefficients_exact=True,
  leaf_dot_order_unchanged=True,reconstruction_operand_orders_unchanged=True,
  register_capacity=capacity,peak_live_values=live_peak,all_leaves_first_register_capacity=old_capacity,
  reconstruction_instruction_lines=rec_lines,previous_inline_instruction_lines=len(old_rec),
  capacity_candidates=candidates,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),source]},
  wasm_execution_verified=False,performance_verified=False,
  scope='Host symbolic streaming scheduler and register reuse proof. All integer arithmetic stays modulo2**32; final signed dot remains4161536-bound. Capacity counts exclude future code-size/ABI/helper/global/runtime constraints.')
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k]for k in ['register_capacity','peak_live_values','all_leaves_first_register_capacity','reconstruction_instruction_lines','previous_inline_instruction_lines']}))
 print(json.dumps([x for x in candidates if x['fits10000']and x['token_capacity']>=56]))
if __name__=='__main__':main()
