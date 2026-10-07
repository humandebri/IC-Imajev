#!/usr/bin/env python3
"""Diagnose the remaining paid compute phases with unchanged arithmetic kernels."""
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
PHASES = ['base_project_inclusive', 'f32_project_inclusive',
          'activation_quantize', 'delta_recurrence', 'execute_norm',
          'execute_conv', 'execute_add', 'lora_finalize', 'mlp_activate',
          'bound_prefix_input', 'exported_hash']


def modify_runtime(text):
    edits = []
    wrappers = [
        ('pub fn execute(r: &Request, x: &[f32], weight: &[f32]) -> Result<Vec<f32>> {',
         'execute', 'r,x,weight',
         'match r.op.as_str(){"rms_bf16"|"rms_scaled"|"gated_norm"=>Some("execute_norm"),"conv_state"=>Some("execute_conv"),"add_bf16"=>Some("execute_add"),_=>None}'),
        ('fn finish_lora_bf16(mut base:Vec<f32>,z:Vec<f32>,scale:f32)->Vec<f32>{',
         'finish_lora_bf16', 'base,z,scale', 'Some("lora_finalize")'),
        ('pub fn swiglu_owned(mut gate:Vec<f32>,up:Vec<f32>)->Vec<f32>{',
         'swiglu_owned', 'gate,up', 'Some("mlp_activate")'),
        ('pub fn server_delta_bound_input(r:&Request,input:&[f32],prefix:&ServerDeltaPrefix)->Result<DecodedQueryInput> {',
         'server_delta_bound_input', 'r,input,prefix', 'Some("bound_prefix_input")'),
    ]
    for signature, name, args, label in wrappers:
        assert text.count(signature) == 1, name
        renamed = signature.replace('fn '+name+'(', 'fn '+name+'_phase_unprofiled(')
        wrapper = signature.replace('mut base:', 'base:').replace('mut gate:', 'gate:')
        wrapper += ('\n let label:Option<&\'static str>='+label+';'
                    '\n if let Some(label)=label {profile::measure(label,||'+name+
                    '_phase_unprofiled('+args+'))}else{'+name+
                    '_phase_unprofiled('+args+')}\n}\n')
        replacement = wrapper + renamed
        text = text.replace(signature, replacement)
        edits.append(dict(before=signature, after=replacement))
    return text, edits


def main():
    p = ROOT/'scripts/build_update_phase_profile_v2.py'
    source = p.read_text().replace('artifacts/update-phase-profile-v2',
                                 'artifacts/update-phase-profile-v3')
    old = '"base_project_inclusive"|"f32_project_inclusive"|"activation_quantize"|"delta_recurrence"'
    assert source.count(old) == 1
    source = source.replace(old, '|'.join(json.dumps(x) for x in PHASES))
    anchor = 'text=text.replace(signature,wrapper);p.write_text(text)'
    assert source.count(anchor) == 1
    source = source.replace(anchor,
        "text=text.replace(signature,wrapper);text,edits=__import__('build_update_phase_profile_v3').modify_runtime(text);"
        "(D.parent/'runtime-phase-edits.json').write_text(json.dumps(edits,indent=2));p.write_text(text)")
    source = source.replace('Four coarse actual compute phase spans:',
                           'Eleven named coarse compute phases:')
    exec(compile(source,str(p),'exec'), dict(__file__=__file__, __name__='__main__',
                                           modify_runtime=modify_runtime))
    d = ROOT/'artifacts/update-phase-profile-v3'
    files = [Path(__file__),p,d/'runtime-phase-edits.json']
    (d/'phase-builder-entry-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):
        hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')


if __name__ == '__main__':
    main()
