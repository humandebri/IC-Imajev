#!/usr/bin/env python3
"""Owned paid scheduler with coarse phase diagnostics; no performance claim."""
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]


def add_profile(text):
    before = text
    anchor = 'fn run(mut s:Session,start:u64,mut reads:u64,mut ops:Vec<(String,u64)>) -> UpdateProgress {'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor+'\n if s.stage==0 {PHASE_ROWS.with(|p|p.borrow_mut().clear());}\n imajev_runtime::profile::start(||ic_cdk::api::performance_counter(0));')
    anchor = 'GRAPH.with(|g|g.borrow_mut().session=Some(s));reply'
    assert text.count(anchor) == 1
    text = text.replace(anchor, 'PHASE_ROWS.with(|p|p.borrow_mut().extend(imajev_runtime::profile::finish()));'+anchor)
    anchor = 'fn digest(v: &[f32]) -> String {'
    assert text.count(anchor) == 1
    text = text.replace(anchor,anchor.replace('fn digest(', 'fn digest_phase_unprofiled('))
    text += '\nfn digest(v:&[f32])->String{imajev_runtime::profile::measure("exported_hash",||digest_phase_unprofiled(v))}\n'
    text += '\nthread_local! {static PHASE_ROWS:RefCell<Vec<(String,u64,u64)>>=RefCell::new(Vec::new());}\n#[ic_cdk::query]fn paid_graph_profile()->String{owner();PHASE_ROWS.with(|p|serde_json::to_string(&*p.borrow()).unwrap())}\n'
    d = ROOT/'artifacts/paid-phase-profile-v3'
    (d/'scheduler-before-profile.rs').write_text(before)
    return text


def main():
    p = ROOT/'scripts/build_paid_owned_profile_off.py'
    source = p.read_text().replace('artifacts/paid-owned-profile-off-v1','artifacts/paid-phase-profile-v3').replace('artifacts/update-profile-off-v1','artifacts/update-phase-profile-v3')
    anchor = ' scheduler.write_text(text)'
    assert source.count(anchor) == 1
    source = source.replace(anchor, " text=__import__('build_paid_phase_profile_v3').add_profile(text)\n"+anchor)
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__',add_profile=add_profile))
    d = ROOT/'artifacts/paid-phase-profile-v3'
    files = [Path(__file__),p,d/'scheduler-before-profile.rs',ROOT/'artifacts/update-phase-profile-v3/phase-builder-entry-hashes.json']
    (d/'phase-builder-entry-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):
        hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')


if __name__ == '__main__':
    main()
