#!/usr/bin/env python3
"""Independently decode archived diagnostic measurements; reject regressions."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/s2-pair-late-v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = D/'check-v2/report.json'
    r = json.loads(p.read_text())
    assert all(sha(ROOT/p) == h for p,h in r['source_hashes'].items())
    helper = ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'
    assert len(r['cases']) == 19
    cases = []
    for c in r['cases']:
        for key in ['s1_pair_bounds','s2_pair_late']:
            m = c['measurements'][key]
            reply = ROOT/m['reply']
            assert sha(reply) == m['reply_sha256']
            raw = json.loads(subprocess.check_output([str(helper),'decode',str(reply),'measurement'],text=True))
            assert all(m[k] == v for k,v in raw.items())
            assert raw['digest'] == c['native']['digest']
            assert raw['total_instructions'] == sum(raw[k] for k in ['quantize_instructions','input_prepare_instructions','project_instructions'])
        before = c['measurements']['s1_pair_bounds']['total_instructions']
        after = c['measurements']['s2_pair_late']['total_instructions']
        assert abs(c['reduction_percent']-100*(1-after/before)) < 1e-10
        cases.append(dict(label=c['label'],tokens=c['tokens'],before=before,after=after,reduction_percent=c['reduction_percent']))
    output = dict(complete=True,check_report_sha256=sha(p),module=r['wasm_sha256'],ordinary_queries=38,native_bits_equal=True,adopted=False,cases=cases,scope='Q projection only; a slower candidate is never connected to full inference.')
    (D/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
