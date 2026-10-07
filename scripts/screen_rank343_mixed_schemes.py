#!/usr/bin/env python3
"""Screen eight exact recursive rank343 schemes before IC performance claims."""
from pathlib import Path
import hashlib,itertools,json,collections,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/rank343-mixed-scheme-screen-v1';d.mkdir(exist_ok=False)
 win=ROOT/'artifacts/s3-k2-prepared-kernels-v1/plan.py'
 classic=ROOT/'scripts/generate_s3_stream_probe.py'
 ws=win.read_text();cs=classic.read_text()
 start='        p0=multiply(';end='    result=multiply('
 cb=cs[cs.index(start):cs.index(end)]
 start='        s1=a.matrix_add('
 wb=ws[ws.index(start):ws.index(end)]
 items=[];files=[Path(__file__),win,classic]
 for choices in itertools.product('WC',repeat=3):
  label=''.join(choices)
  body='        if SCHEME[(8//n).bit_length()-1] == "C":\n'+''.join('    '+line+'\n' for line in cb.splitlines())
  body+='        else:\n'+''.join('    '+line+'\n' for line in wb.splitlines())
  source=ws.replace(wb,body,1).replace('def plan():',f'SCHEME="{label}"\n\ndef plan():',1)
  assert source!=ws
  p=d/(label+'.py');p.write_text(source);files.append(p)
  ns={'__name__':'screen','__file__':str(p)};exec(compile(source,str(p),'exec'),ns)
  a,b,c,leaves,roots,bound=ns['plan']() # Includes independent coefficient identity for all64 outputs.
  rec,regs,capacity=ns['reconstruct'](c,roots)
  operations=collections.Counter(line.split()[0] for line in rec)
  uses=collections.Counter(r for row in roots for r in row)
  for _,terms in c.nodes:uses.update(terms)
  # Validate the emitted reconstruction register program against symbolic DAG.
  products=[{i:1} for i in range(343)]
  registers={i:v for i,v in enumerate(products)};stack=[]
  def combine(x,y,sign):
   out=dict(x)
   for k,v in y.items():out[k]=out.get(k,0)+sign*v
   return {k:v for k,v in out.items() if v}
  for line in rec:
   if line.startswith('local.get'):stack.append(dict(registers[int(line.split('$pc')[1])]))
   elif line.startswith('local.set'):registers[int(line.split('$pc')[1])]=stack.pop()
   elif line.startswith('(v128.const'):stack.append({})
   elif line in ['i32x4.add','i32x4.sub']:
    right=stack.pop();left=stack.pop();stack.append(combine(left,right,1 if line.endswith('add') else -1))
   else:raise AssertionError(line)
  assert not stack
  for ti in range(8):
   for oi in range(8):assert registers[regs[ti][oi]]==dict(c.symbols[roots[ti][oi]])
  aops=sum(len(terms)-1+int(next(iter(terms.values()))<0) for _,terms in a.nodes)
  bops=sum(len(terms)-1+int(next(iter(terms.values()))<0) for _,terms in b.nodes)
  items.append(dict(scheme=label,plan=str(p.relative_to(ROOT)),rank=len(leaves),
    input_nodes=len(a.nodes),input_vector_add_sub=aops,weight_nodes=len(b.nodes),
    weight_vector_add_sub=bops,reconstruction_nodes=len(c.nodes),
    reconstruction_operations=dict(operations),reconstruction_instruction_lines=len(rec),
    reconstruction_locals=capacity,input_i16_bound=max(sum(abs(v)for v in e.values())for e in a.symbols.values())*127,
    weight_i16_bound=max(sum(abs(v)for v in e.values())for e in b.symbols.values())*128,
    unreduced_integer_bound=bound,final_dot_bound=256*127*128,
    all_64_coefficient_identities_exact=True,emitted_reconstruction_symbolically_exact=True))
 baseline=next(x for x in items if x['scheme']=='WWW')
 for item in items:
  item['input_add_delta']=item['input_vector_add_sub']-baseline['input_vector_add_sub']
  item['reconstruction_line_delta']=item['reconstruction_instruction_lines']-baseline['reconstruction_instruction_lines']
 report=dict(complete=True,schemes=items,baseline_scheme='WWW',
   performance_verified=False,wasm_execution_verified=False,
   scope='Structural instruction counts only. All input-dependent preparation must remain in measured handlers. Immutable preparation and memory capacity are not established by this screen.',
   source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files})
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(items,indent=2))
if __name__=='__main__':main()
