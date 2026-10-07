#!/usr/bin/env python3
"""Reconcile the quad-load substitution cost with all saved component counters."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'artifacts/k2-query-quad-load-probe-v1'

def main():
    summary_path = DIR / 'summary.json'
    summary = json.loads(summary_path.read_text())
    for relative, expected in summary['workflow_hashes'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    for entry in DIR.glob('entry-*-hashes.json'):
        for relative, expected in json.loads(entry.read_text()).items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    with zipfile.ZipFile(DIR / 'frozen-workflow.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for relative, expected in summary['workflow_hashes'].items():
            assert hashlib.sha256(archive.read(relative)).hexdigest() == expected, relative
    cases = []
    for case in summary['cases']:
        n = case['tokens']
        rows = 4096 if case['label'] == 'normal' else 8192
        tiles = 26 if rows == 4096 else 49
        leaves = 7 * (n // 2) + 4 * (n % 2)
        packets = 0 if n == 1 else tiles * 10 * leaves * 16
        predicted = packets * 14
        observed = case['after'] - case['before']
        assert predicted == observed, case
        cases.append(dict(label=case['label'], tokens=n, rows=rows,
                          row_tiles=tiles, packets=packets,
                          predicted_extra=predicted, observed_extra=observed))
    report = dict(complete=True, adopted=False, all_21_deltas_exact=True,
                  old_cost_per_packet=12, new_cost_per_packet=26,
                  instruction_cost_source='https://github.com/dfinity/ic/blob/master/rs/embedders/src/wasm_utils/instrumentation.rs',
                  limitation='Upstream master is not pinned to the local runtime. The 14 per packet difference is validated by these measurements only; this is not a general cost interpreter.',
                  summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  frozen_workflow_sha256=hashlib.sha256((DIR / 'frozen-workflow.zip').read_bytes()).hexdigest(),
                  cases=cases)
    (DIR / 'cost-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(complete=True, cases=len(cases), all_deltas_exact=True)))

if __name__ == '__main__':
    main()
