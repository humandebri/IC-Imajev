#!/usr/bin/env python3
"""Move owned graph vectors through actual paid scheduler instead of copying them."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_paid_bf16_predicates.py'
 outer=p.read_text().replace('artifacts/paid-bf16-predicates-v1','artifacts/paid-owned-graph-v1')
 # Apply the existing exact kernel replacement builder, with a new frozen scheduler.
 outer=outer.replace(" exec(compile(s,str(p),'exec')",''' s=s.replace("(d/scheduler.name).write_bytes(scheduler.read_bytes())", """(d/scheduler.name).write_bytes(scheduler.read_bytes())
 scheduler=d/scheduler.name
 text=scheduler.read_text()
 edits={
  'let mut x=s.norm.clone();':'let mut x=std::mem::take(&mut s.norm);',
  'let(y,b)=evaluate(&r,&x).unwrap_or_else(|e|ic_cdk::trap(&e));reads+=b;':'let(mut y,b)=evaluate(&r,&x).unwrap_or_else(|e|ic_cdk::trap(&e));reads+=b;',
  's.attention=y[..count].to_vec();':'y.truncate(count);s.attention=y;',
  'let y=y.into_values().unwrap_or_else(|e|ic_cdk::trap(&e));':'let mut y=y.into_values().unwrap_or_else(|e|ic_cdk::trap(&e));',
  's.attention=y[..s.n*C].to_vec();':'y.truncate(s.n*C);s.attention=y;',
  's.hidden=y[..s.n*C].to_vec();s.norm=y[s.n*C..].to_vec();':'s.norm=y.split_off(s.n*C);s.hidden=y;',
 }
 counts={}
 for before,after in edits.items():
  count=text.count(before);assert count==(2 if before in [list(edits)[0],list(edits)[1]]else 1),(before,count)
  text=text.replace(before,after);counts[before]=count
 scheduler.write_text(text)
 (d/'scheduler-comparison.json').write_text(json.dumps(dict(changes=counts,scope='Norm taken into current input at both attention branches. Output attention Vec truncated after state hash. MLP output split once, hidden retains original capacity for next input. No tensor values, order, hashes, paid accounting, stage or stop bounds changed.'),indent=2)+'\\\\n')""")
 exec(compile(s,str(p),'exec')''')
 assert 'edits={'in outer
 exec(compile(outer,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
