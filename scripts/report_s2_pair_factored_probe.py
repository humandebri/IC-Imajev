#!/usr/bin/env python3
"""Decode shared-transform query evidence and compare with the previous exact S2."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT/'scripts/report_s2_pair_late_probe.py'
    source = p.read_text().replace('artifacts/s2-pair-late-v1','artifacts/s2-pair-factored-v1').replace('check-v2/report.json','check/report.json').replace('s2_pair_late','s2_pair_factored')
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
    d = ROOT/'artifacts/s2-pair-factored-v1'
    current = json.loads((d/'summary.json').read_text())
    old = ROOT/'artifacts/s2-pair-late-v1/summary.json'
    previous = json.loads(old.read_text())
    by_name = {r['label']:r for r in previous['cases']}
    for row in current['cases']:
        baseline = by_name[row['label']]
        assert baseline['before'] == row['before']
        row['previous_s2_instructions'] = baseline['after']
        row['previous_s2_reduction_percent'] = 100*(1-row['after']/baseline['after'])
    current['previous_summary_sha256'] = hashlib.sha256(old.read_bytes()).hexdigest()
    (d/'summary.json').write_text(json.dumps(current,indent=2)+'\n')


if __name__ == '__main__':
    main()
