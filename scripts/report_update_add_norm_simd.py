#!/usr/bin/env python3
"""Verify bit references, baseline restoration and the unchanged reused cache."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    d=ROOT/'artifacts/update-add-norm-simd-v1'
    subprocess.run([sys.executable,str(ROOT/'scripts/report_update_f32_candidate.py'),'--directory',str(d.relative_to(ROOT))],check=True)
    proof=json.loads((d/'proof/report.json').read_text())
    common=proof['banks'][1]['preparation']
    assert common['update_calls_this_run']==0 and common['initial']==common['final']
    r=json.loads((d/'summary.json').read_text())
    r.update(fixed_weights_reused=True,common_weight_preparation_updates=0)
    (d/'summary.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
