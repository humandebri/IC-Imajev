#!/usr/bin/env python3
"""Build a separate one-stage correctness diagnostic for exact dense Delta states."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-quantize-cached-v1'
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 d=ROOT/'artifacts/delta-capture-v2';d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 (d/'delta_capture.rs').write_bytes((ROOT/'scripts/delta_capture.rs').read_bytes())
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-quantize-cached-v1','artifacts/delta-capture-v2')
 anchor="    runtime = base['runtime_command'][:]"
 patch='''    (D/'runtime/delta_capture.rs').write_bytes((D.parent/'delta_capture.rs').read_bytes())
    p=D/'runtime/lib.rs';p.write_text(p.read_text()+'\\npub mod delta_capture;\\n')
    p=D/'runtime/delta_full_log.rs';text=p.read_text()
    anchor=' let original=&x[..n*COLS];'
    assert text.count(anchor)==1
    text=text.replace(anchor,' crate::delta_capture::begin(&states,n,key_major,r.step);\\n'+anchor)
    anchor='  for v in &mut values {*v=crate::bf(*v);}'
    assert text.count(anchor)==1
    text=text.replace(anchor,'  crate::delta_capture::head(head,&qh,&kh,&vh,&gh,&bh,state,&values);\\n'+anchor)
    p.write_text(text)
    p=D/'update_inference.rs';text=p.read_text()
    anchor='Prefix{values,packet:packet.into_vec(),prepared}'
    assert text.count(anchor)==1
    text=text.replace(anchor,'Prefix{values,packet:Vec::new(),prepared}')
    anchor='    while s.stage<64 && ic_cdk::api::performance_counter(0)-start<STOP {'
    assert text.count(anchor)==1
    text=text.replace(anchor,'    let first_stage=s.stage;\\n    while s.stage<64 && s.stage==first_stage {')
    text+='\\n#[ic_cdk::query]\\nfn delta_capture_chunk(offset:u32,length:u32)->Result<Vec<u8>,String>{owner();imajev_runtime::delta_capture::chunk(offset,length)}\\n'
    p.write_text(text)
'''
 assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 (d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),ROOT/'scripts/delta_capture.rs',old/'workflow-hashes.json',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'))
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
