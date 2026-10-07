#!/usr/bin/env python3
"""Capture latest paid arithmetic with owned scheduler, one stage per update.

Correctness-only counterpart: payment wrapper and performance stage grouping
are covered separately by the completed paid proof. No performance claim here.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old = ROOT/'artifacts/update-attention-key-lanes-v1'
    paid = ROOT/'artifacts/paid-attention-key-lanes-v1'
    proof = json.loads((paid/'summary.json').read_text())
    assert proof['complete'] and proof['all_32_hidden_verified']
    for hashes in (json.loads((old/'workflow-hashes.json').read_text()),proof['workflow_hashes']):
        assert all(sha(ROOT/p)==h for p,h in hashes.items())
    d = ROOT/'artifacts/delta-capture-attention-key-lanes-v1'
    d.mkdir(exist_ok=False)
    for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):
        (d/p.name).write_bytes(p.read_bytes())
    (d/'delta_capture.rs').write_bytes((ROOT/'scripts/delta_capture.rs').read_bytes())
    source = (old/'frozen-builder.py').read_text().replace('artifacts/update-attention-key-lanes-v1','artifacts/delta-capture-attention-key-lanes-v1')
    anchor = "    runtime = base['runtime_command'][:]"
    injection = '''    (D/'runtime/delta_capture.rs').write_bytes((D.parent/'delta_capture.rs').read_bytes())
    p=D/'runtime/lib.rs';p.write_text(p.read_text()+'\\npub mod delta_capture;\\n')
    p=D/'runtime/delta_full_log.rs';text=p.read_text()
    edits={
      ' let original=&x[..n*COLS];':' crate::delta_capture::begin(&states,n,key_major,r.step);\\n let original=&x[..n*COLS];',
      '  for v in &mut values {*v=crate::bf(*v);}':'  crate::delta_capture::head(head,&qh,&kh,&vh,&gh,&bh,state,&values);\\n  for v in &mut values {*v=crate::bf(*v);}',
    }
    for before,after in edits.items():assert text.count(before)==1;text=text.replace(before,after)
    p.write_text(text)
    p=D/'update_inference.rs';text=p.read_text()
    edits={
      'let mut x=s.norm.clone();':'let mut x=std::mem::take(&mut s.norm);',
      'let(y,b)=evaluate(&r,&x).unwrap_or_else(|e|ic_cdk::trap(&e));reads+=b;':'let(mut y,b)=evaluate(&r,&x).unwrap_or_else(|e|ic_cdk::trap(&e));reads+=b;',
      's.attention=y[..count].to_vec();':'y.truncate(count);s.attention=y;',
      'let y=y.into_values().unwrap_or_else(|e|ic_cdk::trap(&e));':'let mut y=y.into_values().unwrap_or_else(|e|ic_cdk::trap(&e));',
      's.attention=y[..s.n*C].to_vec();':'y.truncate(s.n*C);s.attention=y;',
      's.hidden=y[..s.n*C].to_vec();s.norm=y[s.n*C..].to_vec();':'s.norm=y.split_off(s.n*C);s.hidden=y;',
      'Prefix{values,packet:packet.into_vec(),prepared}':'Prefix{values,packet:Vec::new(),prepared}',
      '    while s.stage<64 && ic_cdk::api::performance_counter(0)-start<STOP {':'    let first_stage=s.stage;\\n    while s.stage<64 && s.stage==first_stage {',
    }
    counts={}
    for before,after in edits.items():
      count=text.count(before);assert count==(2 if before in list(edits)[:2] else 1),(before,count)
      text=text.replace(before,after);counts[before]=count
    text+='\\n#[ic_cdk::query]\\nfn delta_capture_chunk(offset:u32,length:u32)->Result<Vec<u8>,String>{owner();imajev_runtime::delta_capture::chunk(offset,length)}\\n'
    p.write_text(text)
    (D.parent/'capture-scheduler-changes.json').write_text(json.dumps(counts,indent=2)+'\\n')
'''
    assert source.count(anchor)==1
    source = source.replace(anchor,injection+anchor)
    (d/'frozen-builder.py').write_text(source)
    files=[Path(__file__),ROOT/'scripts/delta_capture.rs',old/'workflow-hashes.json',old/'frozen-builder.py',paid/'summary.json',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'))
    (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2)+'\n')
    exec(compile(source,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':
    main()
