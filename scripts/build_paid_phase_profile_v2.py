#!/usr/bin/env python3
"""Diagnostic paid graph with four coarse phase counters, not an optimization candidate."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_paid_bf16_predicates.py';outer=p.read_text().replace('artifacts/paid-bf16-predicates-v1','artifacts/paid-phase-profile-v2').replace('artifacts/update-bf16-predicates-v1','artifacts/update-phase-profile-v2')
 outer=outer.replace(" exec(compile(s,str(p),'exec')",''' s=s.replace("(d/scheduler.name).write_bytes(scheduler.read_bytes())", """(d/scheduler.name).write_bytes(scheduler.read_bytes())
 scheduler=d/scheduler.name
 text=scheduler.read_text()
 anchor='fn run(mut s:Session,start:u64,mut reads:u64,mut ops:Vec<(String,u64)>) -> UpdateProgress {';assert text.count(anchor)==1
 text=text.replace(anchor,anchor+'\\\\n    if s.stage==0 {PHASE_ROWS.with(|p|p.borrow_mut().clear());}\\\\n    imajev_runtime::profile::start(||ic_cdk::api::performance_counter(0));')
 anchor='GRAPH.with(|g|g.borrow_mut().session=Some(s));reply';assert text.count(anchor)==1
 text=text.replace(anchor,'PHASE_ROWS.with(|p|p.borrow_mut().extend(imajev_runtime::profile::finish()));'+anchor)
 text+='\\\\nthread_local! {static PHASE_ROWS:RefCell<Vec<(String,u64,u64)>>=RefCell::new(Vec::new());}\\\\n#[ic_cdk::query]fn paid_graph_profile()->String{owner();PHASE_ROWS.with(|p|serde_json::to_string(&*p.borrow()).unwrap())}\\\\n'
 scheduler.write_text(text)""")
 exec(compile(s,str(p),'exec')''')
 exec(compile(outer,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
