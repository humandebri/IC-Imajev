#!/usr/bin/env python3
"""Audit candidate provenance without changing any live proof's build report."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    candidate = ROOT / 'artifacts/paid-common-raw-hybrid-v1'
    normal = ROOT / 'artifacts/update-common-raw-hybrid-v1'
    prior = ROOT / 'artifacts/paid-k2-pair-guard-fold-v1'
    build = json.loads((candidate / 'build/report.json').read_text())
    parent = json.loads((normal / 'build/report.json').read_text())
    paths = [Path(__file__), candidate / 'build/report.json', normal / 'build/report.json']
    for owner, report in [(candidate, build), (normal, parent)]:
        assert sha(owner / 'build/full.wasm') == report['wasm_sha256']
        for key in ['source_hashes', 'dependency_hashes']:
            for path, digest in report[key].items():
                assert sha(ROOT / path) == digest, path
                paths.append(ROOT / path)
        paths.append(owner / 'build/full.wasm')
    assert build['update_candidate'] == parent['wasm_sha256']
    assert len(build['patches']) == len(parent['patches']) == 34
    assert all(row['wasmparser_validation'] for row in build['patches'])
    assert [(row['export'], row['source_sha256']) for row in build['patches']] == [
        (row['export'], row['source_sha256']) for row in parent['patches']]
    for name in ['lib.rs', 'update_inference.rs', 'paid_types.rs', 'paid_inference.rs']:
        new, old = candidate / 'build' / name, prior / 'build' / name
        assert new.read_bytes() == old.read_bytes(), name
        paths.extend([new, old])
    assert (candidate / 'optimized-paid-scheduler.rs').read_bytes() == (prior / 'optimized-paid-scheduler.rs').read_bytes()
    paths.extend([candidate / 'optimized-paid-scheduler.rs', prior / 'optimized-paid-scheduler.rs'])
    result = dict(complete=True, module=build['wasm_sha256'], normal_module=parent['wasm_sha256'],
                  canonical_billing_and_wrapper_byte_equal=True, scheduler_byte_equal=True,
                  all34_patch_identities_equal_normal=True, source_and_dependency_hashes_verified=True,
                  full_paid_performance_verified=False, full_hidden_state_fidelity_verified=False,
                  goal_complete=False, source_hashes={str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)})
    (candidate / 'source-audit-readonly.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'source_hashes'}))


if __name__ == '__main__':
    main()
