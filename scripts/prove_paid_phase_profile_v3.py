#!/usr/bin/env python3
"""Measure named coarse phases on all actual paid inputs with saved-bit validation."""
from pathlib import Path
import hashlib, json
from build_update_phase_profile_v3 import PHASES
from prove_paid_phase_profile_v2 import patch_profile_proof

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = ROOT/'artifacts/paid-phase-profile-v3'
    guard = json.loads((d/'upgrade-guards/verified.json').read_text())
    audit = json.loads((d/'source-audit.json').read_text())
    assert audit['complete'] and guard['complete'] and guard['baseline_restored']
    assert guard['module'] == audit['paid_module']
    p = ROOT/'artifacts/paid-owned-profile-off-recovery-v1/frozen-proof.py'
    source = p.read_text().replace('artifacts/paid-owned-profile-off-recovery-v1','artifacts/paid-phase-profile-v3')
    source = patch_profile_proof(source)
    anchor = "allowed={'base_project_inclusive','f32_project_inclusive','activation_quantize','delta_recurrence'}"
    assert source.count(anchor) == 1
    source = source.replace(anchor,'allowed='+repr(set(PHASES)))
    (d/'frozen-proof.py').write_text(source)
    files = [Path(__file__),p,ROOT/'scripts/prove_paid_phase_profile_v2.py',ROOT/'scripts/build_update_phase_profile_v3.py',d/'source-audit.json',d/'upgrade-guards/verified.json',d/'frozen-proof.py']
    (d/'proof-entry-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):
        hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
    exec(compile(source,str(d/'frozen-proof.py'),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__ == '__main__':
    main()
