#!/usr/bin/env python3
"""Allow owner reset of completed diagnostic graphs while retaining fixed weights."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    upstream=ROOT/'artifacts/update-other-profile-v1'
    hashes=json.loads((upstream/'workflow-hashes.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    d=ROOT/'artifacts/update-other-profile-v3'
    d.mkdir(exist_ok=False)
    for p in list(upstream.glob('*.wat'))+[upstream/'prefix-helper.rs',upstream/'prefix-api.rs']:
        (d/p.name).write_bytes(p.read_bytes())
    source=(upstream/'frozen-builder.py').read_text().replace('artifacts/update-other-profile-v1','artifacts/update-other-profile-v3')
    anchor="    runtime = base['runtime_command'][:]"
    assert source.count(anchor)==1
    extra='''    p=D/'update_inference.rs'
    p.write_text(p.read_text()+''' + repr('''
#[ic_cdk::update]
fn reset_update_prefix()->Result<(),String>{
 owner();
 GRAPH.with(|g|->Result<(),String>{let mut g=g.borrow_mut();
  if g.session.as_ref().is_some_and(|s|s.stage<64){return Err("inference is active".into());}
  *g=GraphStore::default();Ok(())
 })?;
 imajev_runtime::prefix_state_cache::clear();
 Ok(())
}
''')+''')
'''
    source=source.replace(anchor,extra+anchor)
    (d/'frozen-builder.py').write_text(source)
    paths=[Path(__file__),upstream/'workflow-hashes.json',upstream/'frozen-builder.py',d/'frozen-builder.py',d/'prefix-helper.rs',d/'prefix-api.rs']+list(d.glob('*.wat'))
    (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths},indent=2)+'\n')
    exec(compile(source,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
