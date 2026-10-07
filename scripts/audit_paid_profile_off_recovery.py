#!/usr/bin/env python3
"""Post-report independent file, raw-Candid, totals and restoration audit."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/paid-owned-profile-off-recovery-v1'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    summary = read(D/'summary.json')
    proof = read(D/'proof/report.json')
    for key in ('complete','baseline_restored','snapshot_deleted','dual_bank','upgrade_receipts_equal'):
        assert proof[key]
    for key in ('all_32_hidden_verified','paid_core_api_verified','payment_boundary_checks_verified',
                'deterministic_upgrade_guards_verified','saved_candid_replies_verified'):
        assert summary[key]
    for name in ('workflow_hashes','reference_hashes'):
        for p, h in summary[name].items():
            assert sha(ROOT/p) == h, p
    assert sha(D/'build/full.wasm') == proof['candidate'] == summary['module']
    baseline = read(D/'proof/before.json')
    restored = read(D/'proof/restored.json')
    assert baseline['module'] == restored['module']
    assert all(restored[k] for k in ('cache_equal','pack_equal','snapshot_deleted'))
    guards = read(D/'upgrade-guards/verified.json')
    assert guards['complete'] and guards['baseline_restored'] and len(guards['checks']) == 4
    helper = ROOT/'artifacts/paid-update-v1/tools/args'
    def decode(kind, path):
        return json.loads(subprocess.check_output([str(helper),'decode',kind,str(path)],text=True))
    calls = 0
    for path in (D/'proof').rglob('*.json'):
        call = read(path)
        if not isinstance(call, dict) or not {'kind','reply_path','args_path','result'} <= call.keys():
            continue
        reply = ROOT/call['reply_path']
        assert (ROOT/call['args_path']).stat().st_size == call['request_bytes']
        assert len(bytes.fromhex(reply.read_text().strip().removeprefix('0x'))) == call['reply_bytes']
        if call['relay']:
            outer = decode('forward', reply)
            assert outer == call['forward']
            if 'Ok' in outer['response']:
                inner = reply.with_name(reply.name.replace('.reply.hex','.inner.hex'))
                assert bytes.fromhex(inner.read_text().strip().removeprefix('0x')) == bytes(outer['response']['Ok'])
                actual = decode(call['kind'], inner)
            else:
                actual = {'transport_error':outer['response']['Err']}
        else:
            actual = decode(call['kind'], reply)
        assert actual == call['result']
        calls += 1
    assert calls > 30
    totals = {}
    prior = read(ROOT/'artifacts/paid-owned-graph-v1/summary.json')
    for case in summary['cases']:
        item = next(row for row in proof['results'] if row['case'] == case['case'])
        workers = item['row']['result']['Ok']['workers']
        total = sum(w['instructions'] for w in workers)
        assert total == case['total_handler_instructions']
        old = next(row for row in prior['cases'] if row['case'] == case['case'])
        assert old['total_handler_instructions'] - total == 12010
        totals[case['case']] = total
    assert not summary['all_targets_met']
    with zipfile.ZipFile(D/'frozen-paid-proof.zip') as z:
        assert len(z.namelist()) == len(set(z.namelist()))
        assert z.read(str((D/'summary.json').relative_to(ROOT))) == (D/'summary.json').read_bytes()
        for p, h in summary['workflow_hashes'].items():
            assert hashlib.sha256(z.read(p)).hexdigest() == h, p
    result = {'complete': True, 'raw_candid_calls_redecoded':calls,
              'worker_totals':totals,'all_32_hidden_report_and_reference_hashes_verified':True,
              'workflow_hashes_verified':True,'zip_unique_latest_summary_verified':True,
              'baseline_restored':True,'temporary_snapshot_deleted':True,
              'gain_each_case_from_owned_graph':12010,'all_targets_met':False,
              'latest_dense_direct_capture_complete':False,
              'audit_script_sha256':sha(Path(__file__))}
    (D/'post-report-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
