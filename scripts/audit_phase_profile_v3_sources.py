#!/usr/bin/env python3
"""Reverse only profiling wrappers to compare independently frozen arithmetic sources."""
from pathlib import Path
import hashlib, json, zipfile
from build_update_phase_profile_v3 import PHASES

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d = ROOT/'artifacts/update-phase-profile-v3'
    old = ROOT/'artifacts/update-phase-profile-v2'
    edits = json.loads((d/'runtime-phase-edits.json').read_text())
    text = (d/'build/runtime/lib.rs').read_text()
    for edit in reversed(edits):
        assert text.count(edit['after']) == 1
        text = text.replace(edit['after'],edit['before'])
    assert text == (old/'build/runtime/lib.rs').read_text()
    text = (d/'build/runtime/profile.rs').read_text()
    labels = '|'.join(json.dumps(x) for x in PHASES)
    assert text.count(labels) == 1
    assert text.replace(labels,'|'.join(json.dumps(x) for x in PHASES[:4])) == (old/'build/runtime/profile.rs').read_text()
    for p in (old/'build/runtime').glob('*.rs'):
        if p.name not in ['lib.rs','profile.rs']:
            assert p.read_bytes() == (d/'build/runtime'/p.name).read_bytes(),p.name
    build = json.loads((d/'build/report.json').read_text())
    for key in ['source_hashes','dependency_hashes']:
        assert all(sha(ROOT/p)==h for p,h in build[key].items())
    assert len(build['patches']) == 22 and all(p['wasmparser_validation'] for p in build['patches'])
    paid = ROOT/'artifacts/paid-phase-profile-v3'
    predecessor = ROOT/'artifacts/paid-owned-profile-off-v1/build/update_inference.rs'
    assert (paid/'scheduler-before-profile.rs').read_bytes() == predecessor.read_bytes()
    from build_paid_phase_profile_v3 import add_profile
    expected = add_profile(predecessor.read_text())
    assert (paid/'build/update_inference.rs').read_text() == expected
    pb = json.loads((paid/'build/report.json').read_text())
    for key in ['source_hashes','dependency_hashes']:
        assert all(sha(ROOT/p)==h for p,h in pb[key].items())
    assert len(pb['patches']) == 22 and all(p['wasmparser_validation'] for p in pb['patches'])
    for a,b in zip(build['patches'],pb['patches']):
        assert a['source_sha256'] == b['source_sha256']
    for name in ['paid_types.rs','paid_inference.rs']:
        assert (paid/'build'/name).read_bytes() == (ROOT/'canisters/inference/src'/name).read_bytes()
    paths = [Path(__file__),ROOT/'scripts/build_update_phase_profile_v3.py',ROOT/'scripts/build_paid_phase_profile_v3.py',d/'build/report.json',paid/'build/report.json',d/'phase-builder-entry-hashes.json',paid/'phase-builder-entry-hashes.json',d/'runtime-phase-edits.json',paid/'scheduler-before-profile.rs',predecessor]
    result = dict(complete=True,diagnostic_only=True,arithmetic_kernels_unchanged=True,
                  owned_scheduler_equal_before_profiling=True, canonical_billing_equal=True,
                  normal_module=sha(d/'build/full.wasm'), paid_module=sha(paid/'build/full.wasm'),
                  phases=PHASES, hashes={str(p.relative_to(ROOT)):sha(p) for p in paths})
    (paid/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(paid/'source-audit.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths+[paid/'source-audit.json']:
            z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps(dict(complete=True,normal_module=result['normal_module'],paid_module=result['paid_module'])))


if __name__ == '__main__':
    main()
