#!/usr/bin/env python3
"""Generate exact rank26 2x4x4 INT8 SIMD from Lille's Hopcroft/Kerr scheme."""
import hashlib,json,re
from pathlib import Path
from generate_s3_stream_probe import DAG
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/hopcroft-kerr26-v1/build'
REFERENCE=ROOT/'artifacts/hopcroft-kerr26-v1/reference/2x4x4_raw.mpl'

def plan():
 text=REFERENCE.read_text();a,b,c=DAG('a',8),DAG('b',16),DAG('p',26);leaves=[]
 def linear(dag,terms):
  current='zero'
  for sign,name in terms:current=dag.add(current,name,-1 if sign=='-' else 1)
  return current
 def terms(s,kind,width):
  return [(sign,f'{kind.lower()}{(int(i)-1)*width+int(j)-1}') for sign,i,j in re.findall(r'([+-]?)'+kind+r'_(\d+)_(\d+)',s)]
 for m,left,right in re.findall(r'm_(\d+)=\(([^)]+)\)\*\(([^)]+)\)',text):
  assert int(m)==len(leaves)+1
  leaves.append((linear(a,terms(left,'A',4)),linear(b,terms(right,'B',4))))
 roots=[[None]*4 for _ in range(2)]
 for i,j,e in re.findall(r'C_(\d+)_(\d+)=([^,\n]+)',text):
  roots[int(i)-1][int(j)-1]=linear(c,[(sign,f'p{int(m)-1}') for sign,m in re.findall(r'([+-]?)m_(\d+)',e)])
 assert len(leaves)==26 and all(all(row) for row in roots)
 for dag,scale in [(a,127),(b,128)]:assert max(sum(abs(v) for v in e.values()) for e in dag.symbols.values())*scale<32768
 bound=0
 for e in c.symbols.values():
  estimate=sum(abs(sign)*sum(abs(v) for v in a.symbols[leaves[m][0]].values())*127*sum(abs(v) for v in b.symbols[leaves[m][1]].values())*128*64 for m,sign in e.items());bound=max(bound,estimate);assert estimate<2**31
 for i in range(2):
  for j in range(4):
   terms={}
   for m,s in c.symbols[roots[i][j]].items():
    for ai,av in a.symbols[leaves[m][0]].items():
     for bi,bv in b.symbols[leaves[m][1]].items():terms[ai,bi]=terms.get((ai,bi),0)+s*av*bv
   assert {k:v for k,v in terms.items() if v}=={(i*4+k,k*4+j):1 for k in range(4)}
 return a,b,c,leaves,roots,bound

def main():
 source=(ROOT/'scripts/generate_winograd2_register_probe_v2.py').read_text()
 # Reuse the measured raw-weight SIMD packing and output scaling order, with
 # the rank26 rectangular plan, two tokens, and 128 output rows.
 source=source[source.index('def reconstruct'):source.index("if __name__")]
 source=source.replace('range(49)','range(26)').replace('capacity=49','capacity=26')
 source=source.replace('range(16):prepare','range(8):prepare').replace('group*4+','group*2+')
 source=source.replace('for j in range(8):','for j in range(16):')
 source=source.replace("range(4)]+[f'(local $sw", "range(2)]+[f'(local $sw")
 source=source.replace('for i in range(4):lines +=', 'for i in range(2):lines +=')
 source=source.replace('for ti in range(4):','for ti in range(2):')
 source=source.replace('(i32.shr_u (local.get $t) (i32.const 2))','(i32.shr_u (local.get $t) (i32.const 1))')
 source=source.replace('(i32.const 4)))\',\'br $tokens', '(i32.const 2)))\',\'br $tokens')
 source=source.replace("for j in range(8) for m", "for j in range(16) for m")
 source=source.replace(".replace('343','49').replace('344','50').replace('div_ceil(8)','div_ceil(4)')", ".replace('343','26').replace('344','27').replace('div_ceil(8)','div_ceil(2)')")
 source=source.replace(".replace('rows%32','rows%64').replace('step_by(32)','step_by(64)')", ".replace('rows%32','rows%128').replace('step_by(32)','step_by(128)')")
 source=source.replace(".replace('add(343)','add(49)').replace('0..32','0..64')", ".replace('add(343)','add(26)').replace('0..32','0..128')")
 source=source.replace('offset=196','offset=104').replace("for i in range(16)]+['(local $h0","for i in range(32)]+['(local $h0").replace("for i in range(16):lines += [f'(local.set $sw","for i in range(32):lines += [f'(local.set $sw").replace('rank=49,output_tile=64,token_group=4','rank=26,output_tile=128,token_group=2')
 source=source.replace('rank49','rank26').replace('tokens=4,output_rows=64','tokens=2,output_rows=128')
 namespace=dict(Path=Path,ROOT=ROOT,D=D,plan=plan,__file__=__file__,hashlib=hashlib,json=json,re=re)
 from collections import Counter
 from generate_s3_stream_probe import expression
 namespace.update(Counter=Counter,expression=expression)
 exec(source,namespace)
 namespace['main']()
 report=json.loads((D/'generator.json').read_text());report['reference_url']='https://fmm.univ-lille.fr/2x4x4.html';report['reference_sha256']=hashlib.sha256(REFERENCE.read_bytes()).hexdigest();report['tokens']=2;report['output_rows']=128
 (D/'generator.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
