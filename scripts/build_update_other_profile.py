#!/usr/bin/env python3
"""Diagnose remaining Rust stage spans around unchanged validated SIMD kernels."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    upstream=ROOT/'artifacts/update-bound-prefix-v1'
    hashes=json.loads((upstream/'workflow-hashes.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    d=ROOT/'artifacts/update-other-profile-v1'
    d.mkdir(exist_ok=False)
    for p in list(upstream.glob('*.wat'))+[upstream/'prefix-helper.rs',upstream/'prefix-api.rs']:
        (d/p.name).write_bytes(p.read_bytes())
    source=(upstream/'frozen-builder.py').read_text().replace('artifacts/update-bound-prefix-v1','artifacts/update-other-profile-v1')
    anchor="    runtime = base['runtime_command'][:]"
    assert source.count(anchor)==1
    addition='''    p=D/'runtime/lib.rs'
    text=p.read_text()
    before='pub fn execute(r: &Request, x: &[f32], weight: &[f32]) -> Result<Vec<f32>> {'
    assert text.count(before)==1
    text=text.replace(before,before.replace('fn execute(', 'fn execute_unprofiled('))
    text+='\\npub fn execute(r:&Request,x:&[f32],weight:&[f32])->Result<Vec<f32>>{let name=match r.op.as_str(){"conv_state"=>"execute_conv_state","rms_bf16"=>"execute_rms_bf16","rms_scaled"=>"execute_rms_scaled","gated_norm"=>"execute_gated_norm","add_bf16"=>"execute_add_bf16","delta_gates"=>"execute_delta_gates",_=>"execute_other_inclusive"};profile::measure(name,||execute_unprofiled(r,x,weight))}\\n'
    before='base.into_iter()\\n            .zip(z)\\n            .map(|(v, z)| bf(bf(v) + bf(r.scalars[0] * z)))\\n            .collect::<Vec<_>>()'
    assert text.count(before)==1
    text=text.replace(before,'profile::measure("lora_finalize_bf16",||'+before+')')
    before='let out = gate\\n        .into_iter()\\n        .zip(up)\\n        .map(|(g, u)| bf(bf_silu(g) * u))\\n        .collect::<Vec<_>>();'
    assert text.count(before)==1
    text=text.replace(before,before.replace('let out = gate','let out = profile::measure("mlp_activate_bf16",||gate').replace('.collect::<Vec<_>>();','.collect::<Vec<_>>());'))
    before='Ok(DecodedQueryInput::DeltaHybrid(prefix.input(r,input)?))'
    assert text.count(before)==1
    text=text.replace(before,'Ok(DecodedQueryInput::DeltaHybrid(profile::measure("bound_prefix_input",||prefix.input(r,input))?))')
    p.write_text(text)
    p=D/'runtime/delta_full_log.rs'
    text=p.read_text()
    before='crate::delta_from_key_major(&qh,&kh,&vh,&gh,&bh,state,128,128)?'
    assert text.count(before)==1
    p.write_text(text.replace(before,'crate::profile::measure("delta_head_inclusive",||crate::delta_from_key_major(&qh,&kh,&vh,&gh,&bh,state,128,128))?'))
    p=D/'update_inference.rs'
    text=p.read_text()
    text+='\\nthread_local! {static LAST_PROFILE: RefCell<Vec<(String,u64,u64)>> = RefCell::new(Vec::new());}\\n#[ic_cdk::query]fn last_instruction_profile()->String{owner();LAST_PROFILE.with(|p|serde_json::to_string(&*p.borrow()).unwrap())}\\n'
    for signature in ['fn update_infer_start(ids: Vec<u32>, options: Vec<String>) -> Result<UpdateProgress,String> {','fn update_infer_continue(id:u64, stage:u64) -> Result<UpdateProgress,String> {']:
        assert text.count(signature)==1
        text=text.replace(signature,signature+'\\n imajev_runtime::profile::start(||ic_cdk::api::performance_counter(0));')
    before='GRAPH.with(|g|g.borrow_mut().session=Some(s));reply'
    assert text.count(before)==1
    text=text.replace(before,'LAST_PROFILE.with(|p|*p.borrow_mut()=imajev_runtime::profile::finish());\\n'+before)
    before='fn digest(v: &[f32]) -> String {'
    assert text.count(before)==1
    text=text.replace(before,before.replace('fn digest(', 'fn digest_unprofiled('))
    text+='\\nfn digest(v:&[f32])->String{imajev_runtime::profile::measure("exported_hash",||digest_unprofiled(v))}\\n'
    p.write_text(text)
'''
    source=source.replace(anchor,addition+anchor+"\n    runtime += ['--cfg', 'feature=\"instruction-profile\"']")
    (d/'frozen-builder.py').write_text(source)
    paths=[Path(__file__),upstream/'workflow-hashes.json',upstream/'frozen-builder.py',d/'frozen-builder.py',d/'prefix-helper.rs',d/'prefix-api.rs']+list(d.glob('*.wat'))
    (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths},indent=2)+'\n')
    exec(compile(source,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
