#!/usr/bin/env python3
"""Preserve optimized digest and bound prefixes in the canonical two-bank paid scheduler."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'artifacts/paid-add-norm-simd-v1';assert all(sha(ROOT/p)==h for p,h in json.loads((upstream/'workflow-hashes.json').read_text()).items())
 d=ROOT/'artifacts/paid-add-norm-simd-v2';d.mkdir(exist_ok=False)
 scheduler=(ROOT/'canisters/inference/src/update_inference.rs').read_text();optimized=ROOT/'artifacts/update-add-norm-simd-v1/build/update_inference.rs';o=optimized.read_text()
 assert scheduler.count('struct Prefix { values: Vec<f32>, packet: Vec<u8> }')==1
 scheduler=scheduler.replace('struct Prefix { values: Vec<f32>, packet: Vec<u8> }','struct Prefix { values: Vec<f32>, packet: Vec<u8>, prepared: Option<imajev_runtime::ServerDeltaPrefix> }')
 lo=scheduler.index('fn digest(');hi=scheduler.index('\n#[ic_cdk::update]',lo);olo=o.index('fn digest(');ohi=o.index('\n#[ic_cdk::update]',olo);scheduler=scheduler[:lo]+o[olo:ohi]+scheduler[hi:]
 before='    if i%4==3 {\n        if values.len()!=p*2048';assert scheduler.count(before)==1;scheduler=scheduler.replace(before,'    let mut prepared=None;\n'+before)
 before='        imajev_runtime::server_delta_hybrid_input(&r,&input,&packet)?;';assert scheduler.count(before)==1;scheduler=scheduler.replace(before,'        if GRAPH.with(|g|g.borrow().banks.is_empty()) {imajev_runtime::prefix_state_cache::clear();}\n        prepared=Some(imajev_runtime::prepare_server_delta_prefix(&r,&input,&packet)?);')
 before='Prefix{values,packet:packet.into_vec()}';assert scheduler.count(before)==1;scheduler=scheduler.replace(before,'Prefix{values,packet:packet.into_vec(),prepared}')
 before='imajev_runtime::server_delta_hybrid_input(&r,&x,&p.packet)';assert scheduler.count(before)==1;scheduler=scheduler.replace(before,'imajev_runtime::server_delta_bound_input(&r,&x,p.prepared.as_ref().expect("prepared fixed Delta prefix"))')
 (d/'optimized-paid-scheduler.rs').write_text(scheduler)
 s=(upstream/'frozen-builder.py').read_text().replace('artifacts/paid-add-norm-simd-v1','artifacts/paid-add-norm-simd-v2');anchor=" p=D/'lib.rs';text=p.read_text()";assert s.count(anchor)==1;s=s.replace(anchor," (D/'update_inference.rs').write_bytes((D.parent/'optimized-paid-scheduler.rs').read_bytes())\n"+anchor)
 (d/'frozen-builder.py').write_text(s);paths=[Path(__file__),upstream/'workflow-hashes.json',upstream/'frozen-builder.py',optimized,ROOT/'canisters/inference/src/update_inference.rs',d/'optimized-paid-scheduler.rs',d/'frozen-builder.py']
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths},indent=2)+'\n')
 exec(compile(s,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 b=d/'build';r=json.loads((b/'report.json').read_text());assert all((b/n).read_bytes()==(ROOT/'canisters/inference/src'/n).read_bytes() for n in ['paid_types.rs','paid_inference.rs']);assert (b/'update_inference.rs').read_text()==scheduler
 r.update(canonical_billing_sources_byte_equal=True,bound_prefix_and_bulk_digest=True,paid_execution_verified=False);(b/'report.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
