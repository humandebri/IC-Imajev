#!/usr/bin/env python3
"""Replay saved-bit and Candid audits for the eleven-phase diagnostic."""
from pathlib import Path
import json, re
from build_update_phase_profile_v3 import PHASES

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT/'scripts/report_paid_phase_profile_v2.py'
    source = p.read_text().replace('artifacts/paid-phase-profile-v2','artifacts/paid-phase-profile-v3').replace('artifacts/update-phase-profile-v2','artifacts/update-phase-profile-v3')
    comparison = json.loads((ROOT/'artifacts/update-phase-profile-v3/runtime-comparison.json').read_text())
    source, count = re.subn(r'^ new=.*$', ' new='+repr(repr(comparison)),source,flags=re.M)
    assert count == 1
    allowed = "allowed={'base_project_inclusive','f32_project_inclusive','activation_quantize','delta_recurrence'}"
    assert source.count(allowed) == 1
    source = source.replace(allowed, 'allowed='+repr(set(PHASES)))
    source = source.replace("d/'scheduler-source-audit.json'", "d/'source-audit.json'")
    source = source.replace('"extras=[Path(__file__),p,',"\"extras=[Path(__file__),p,ROOT/'scripts/report_paid_phase_profile_v2.py',ROOT/'scripts/build_update_phase_profile_v3.py',ROOT/'scripts/build_paid_phase_profile_v3.py',")
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__ == '__main__':
    main()
