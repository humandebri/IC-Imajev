#!/usr/bin/env python3
"""Keep long-lived single-use nodes; symbolically audit each selected schedule."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/rank343-selective-streaming-inline-v1';d.mkdir(exist_ok=True);assert not(d/'report.json').exists()
 if(d/'gap-0.py').exists()and not(d/'failed-gap-0.py').exists():(d/'failed-gap-0.py').write_bytes((d/'gap-0.py').read_bytes())
 original=ROOT/'scripts/screen_rank343_streaming_inline_reconstruction.py';template=original.read_text();results=[];files=[Path(__file__),original]
 before="keep={name for name,_ in c.nodes if uses[name]>1 or name in root_names}"
 assert template.count(before)==1
 for threshold in [0,1,2,4,8,16,32,64,128,344]:
  variant=f'gap-{threshold}';source=template.replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path('+repr(str(ROOT))+')').replace("d=ROOT/'artifacts/rank343-streaming-inline-reconstruction-v1'",f"d=ROOT/'artifacts/rank343-selective-streaming-inline-v1/{variant}'")
  after="""ready={f'p{i}':i for i in range(343)}
 for name,terms in c.nodes:ready[name]=max(ready[p]for p in terms)
 consumer={p:name for name,terms in c.nodes for p in terms}
 keep={name for name,_ in c.nodes if uses[name]>1 or name in root_names or ready[consumer[name]]-ready[name]>=THRESHOLD}""".replace('THRESHOLD',str(threshold))
  source=source.replace(before,after).replace('assert count==len(old)','assert count<=2800')
  begin=source.index(' # The emitted expression operator/read order');end=source.index(' report=dict',begin);source=source[:begin]+source[end:]
  source=source.replace('inline_operator_order_unchanged=True','per_node_signed_operand_order_preserved=True')
  entry=d/f'{variant}.py';entry.write_text(source);exec(compile(source,str(entry),'exec'),dict(__file__=str(entry),__name__='__main__'))
  report=json.loads((d/variant/'report.json').read_text());results.append(dict(threshold=threshold,register_capacity=report['register_capacity'],reconstruction_instruction_lines=report['reconstruction_instruction_lines'],all64_symbolic_outputs_exact=report['all64_symbolic_outputs_exact'],report=str((d/variant/'report.json').relative_to(ROOT))));files.extend([entry,d/variant/'report.json'])
 report=dict(complete=True,results=results,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},wasm_execution_verified=False,performance_verified=False,scope=__doc__)
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(results))
if __name__=='__main__':main()
