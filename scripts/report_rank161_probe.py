#!/usr/bin/env python3
"""Audit frozen rank161 component counters, Candid and native digests."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory',required=True)
    ap.add_argument('--measurement-name',required=True)
    a = ap.parse_args()
    d = ROOT/a.directory
    report = json.loads((d/'check/report.json').read_text())
    build = json.loads((d/'build/report.json').read_text())
    manifest = json.loads((d/'workflow-hashes.json').read_text())
    for hashes in (report['source_hashes'],build['source_hashes'],build['dependency_hashes'],manifest):
        for p,h in hashes.items():
            assert sha(ROOT/p)==h,p
    assert sha(d/'build/diagnostic.wasm') == report['wasm_sha256'] == build['wasm_sha256']
    helper = ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'
    calls = report['preparations']+[v for case in report['cases'] for v in case['measurements'].values()]
    for call in calls:
        path = ROOT/call['reply']
        assert sha(path) == call['reply_sha256']
        kind = 'measurement' if 'digest' in call else 'preparation'
        decoded = json.loads(subprocess.check_output([str(helper),'decode',str(path),kind],text=True))
        assert all(call[k]==v for k,v in decoded.items())
    comparisons = []
    for case in report['cases']:
        path = d/'check'/f"{case['label']}.input.bin"
        assert sha(path) == case['input_sha256']
        assert case['bitwise_equal']
        assert all(v['digest']==case['native']['digest'] for v in case['measurements'].values())
        before = case['measurements']['s1_pair_bounds']['total_instructions']
        after = case['measurements'][a.measurement_name]['total_instructions']
        assert case['reduction_percent'] == 100*(1-after/before)
        comparisons.append(dict(label=case['label'],before=before,after=after,reduction_percent=case['reduction_percent']))
    assert len(report['cases']) == 19 and report['ordinary_queries']==38
    for name,hashes in (('check/source.zip',report['source_hashes']),('build/source.zip',build['source_hashes'])):
        with zipfile.ZipFile(d/name) as z:
            assert len(z.namelist()) == len(set(z.namelist()))
            for p,h in hashes.items():
                assert hashlib.sha256(z.read(p)).hexdigest()==h
    result = dict(complete=True,module=report['wasm_sha256'],cases=19,ordinary_queries=38,
                  raw_candid_redecoded=len(calls),all_native_bits_equal=True,
                  workflow_source_dependency_input_hashes_verified=True,adopted=False,
                  diagnostic_only=True,whole_inference_connected=False,
                  all_cases_slower_than_control=all(c['after']>c['before'] for c in comparisons),
                  comparison=comparisons,audit_script_sha256=sha(Path(__file__)))
    (d/'post-report-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    files=[Path(__file__),d/'post-report-audit.json',d/'check/report.json',d/'build/report.json',
           d/'workflow-hashes.json',d/'build/source.zip',d/'check/source.zip',d/'frozen-check.py']
    files += list((d/'check').glob('*.hex'))
    with zipfile.ZipFile(d/'frozen-audit.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(d/'frozen-audit.zip') as z:
        assert len(z.namelist())==len(set(z.namelist()))
        for p in files:
            assert z.read(str(p.relative_to(ROOT))) == p.read_bytes()
    print(json.dumps({'complete':True,'all_native_bits_equal':True,'all_cases_slower':result['all_cases_slower_than_control']}))


if __name__ == '__main__':
    main()
