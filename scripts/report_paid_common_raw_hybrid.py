#!/usr/bin/env python3
"""Independently decode completed paid trials and verify exact references."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype='<f4').tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    assert directory.is_relative_to(ROOT / 'artifacts')
    proof = directory / 'proof'
    candidate = ROOT / 'artifacts/paid-common-raw-hybrid-v1'
    report = read(proof / 'report.json')
    assert all(report[key] for key in ['complete', 'baseline_restored', 'snapshot_deleted', 'dual_bank', 'upgrade_receipts_equal'])
    build = read(candidate / 'build/report.json')
    assert sha(candidate / 'build/full.wasm') == report['candidate'] == build['wasm_sha256']
    provenance = read(candidate / 'source-audit-readonly.json')
    assert provenance['complete'] and provenance['canonical_billing_and_wrapper_byte_equal']
    manifests = [directory / 'entry-hashes.json', candidate / 'workflow-hashes.json', proof / 'sources.json']
    workflow = {}
    for values in [read(path) for path in manifests] + [build['source_hashes'], build['dependency_hashes'], provenance['source_hashes']]:
        for path, expected in values.items():
            file = ROOT / path
            assert sha(file) == expected, path
            if not file.is_relative_to(ROOT):
                alias = ROOT / 'artifacts/paid-common-raw-hybrid-v1/proof-storage/proof/recovery/verification/verified.json'
                assert alias.resolve() == file.resolve(), 'unexpected external provenance file'
                file = alias
            workflow[str(file.relative_to(ROOT))] = expected
    before, restored = read(proof / 'before.json'), read(proof / 'restored.json')
    assert before['module'] == restored['module'] == build['baseline']
    assert all(restored[key] for key in ['cache_equal', 'pack_equal', 'snapshot_deleted'])
    decoder = ROOT / 'artifacts/paid-update-v1/tools/args'
    def decode(kind, path):
        return json.loads(subprocess.check_output([str(decoder), 'decode', kind, str(path)], text=True))
    calls, decoded_results, decoded_debugs = [], [], []
    call_files = list(proof.rglob('*.json'))
    guard_file = directory / 'upgrade-guards/verified.json'
    guards = None
    if guard_file.exists():
        guards = read(guard_file)
        assert guards['complete'] and guards['baseline_restored'] and guards['module'] == report['candidate']
        assert {item['name'] for item in guards['checks']} == {'active-upgrade-refused', 'refund-upgrade-refused', 'failed-pending-receipt-preserved', 'failed-done-receipt-preserved'}
        for path, expected in read(directory / 'upgrade-guard-entry-hashes.json').items():
            assert sha(ROOT / path) == expected, path
        call_files.extend((directory / 'upgrade-guards').rglob('*.json'))
    for path in sorted(call_files):
        call = read(path)
        if not isinstance(call, dict) or not {'kind', 'args_path', 'reply_path', 'result'} <= call.keys():
            continue
        argument, reply = ROOT / call['args_path'], ROOT / call['reply_path']
        assert argument.stat().st_size == call['request_bytes']
        raw = bytes.fromhex(reply.read_text().strip().removeprefix('0x'))
        assert len(raw) == call['reply_bytes']
        if call['relay']:
            forward = decode('forward', reply)
            assert forward == call['forward']
            if 'Ok' in forward['response']:
                inner = reply.with_name(reply.name.replace('.reply.hex', '.inner.hex'))
                assert bytes.fromhex(inner.read_text().strip().removeprefix('0x')) == bytes(forward['response']['Ok'])
                actual = decode(call['kind'], inner)
            else:
                actual = {'transport_error': forward['response']['Err']}
        else:
            actual = decode(call['kind'], reply)
        assert actual == call['result'], path
        if call['kind'] == 'infer' and 'Ok' in actual:
            decoded_results.append(actual['Ok'])
        if call['kind'] == 'paid_debug':
            decoded_debugs.append(actual)
        calls.extend([path, argument, reply])
    assert len(calls) // 3 > 30
    saved = read(ROOT / 'artifacts/paid-k2-pair-guard-fold-v1/proof/report.json')
    reconstruction = ROOT / 'artifacts/terminal-mlp-reference-v1'
    independent = read(reconstruction / 'summary.json')
    assert independent['complete'] and independent['all_32_hidden_verified']
    references = {}
    def reference(path):
        references[str(path.relative_to(ROOT))] = sha(path)
    for field in ['workflow_hashes', 'reference_hashes']:
        for path, expected in independent.get(field, {}).items():
            assert sha(ROOT / path) == expected, path
            reference(ROOT / path)
    records = read(ROOT / 'artifacts/text-short-v2/inputs.json')['records'][:3]
    cases = []
    assert len(report['results']) == 3
    for item in report['results']:
        name = item['case']
        index = ['617', '620', '653'].index(name)
        request = item['request']['request']
        assert request['token_ids'] == records[index]['token_ids'] and request['options'] == records[index]['options']
        result, debug = item['row']['result']['Ok'], item['debug']
        assert result in decoded_results and debug in decoded_debugs and result['job_id'] == debug['job_id']
        assert result['paid_cycles'] == item['quote']['fee'] and item['row']['forward']['refunded'] == 12_345_678
        old = next(value for value in saved['results'] if value['case'] == name)
        assert len(debug['hidden_hashes']) == len(debug['state_hashes']) == 32
        assert debug['hidden_hashes'] == old['debug']['hidden_hashes'] and debug['state_hashes'] == old['debug']['state_hashes']
        historical = ROOT / f'artifacts/boomdao-current-v1/{name}-r1'
        decision = read(historical / 'report.json')['decision_query']['ok']['decision']
        assert {key: value for key, value in result['decision'].items() if key != 'instructions'} == {key: value for key, value in decision.items() if key != 'instructions'}
        for key in ['raw_logits', 'probabilities', 'unknown_probability']:
            assert digest(result['decision'][key]) == digest(decision[key])
        assert digest(debug['final_hidden']) == digest(np.load(historical / 'final-hidden.npy', allow_pickle=False))
        reference(historical / 'report.json'); reference(historical / 'final-hidden.npy')
        prefix = item['quote']['prefix_tokens']
        assert prefix == (27 if name == '653' else 38)
        for layer in range(32):
            hidden = historical / f'queries/layer-{layer:02d}.npy'
            if hidden.exists():
                values = np.load(hidden, allow_pickle=False)
                assert digest(values[prefix:] if layer < 31 else values) == debug['hidden_hashes'][layer]
            else:
                assert layer == 30
                row = next(value for value in read(reconstruction / 'reconstruction/report.json')['results'] if value['case'] == name and value['layer'] == 30)
                hidden = ROOT / row['output']
                assert sha(hidden) == row['output_sha256']
                assert digest(np.load(hidden, allow_pickle=False)[prefix - 27:]) == debug['hidden_hashes'][30]
            reference(hidden)
            state = historical / f'queries/states/layer-{layer:02d}.npz'
            with np.load(state, allow_pickle=False) as values:
                value = np.concatenate([values['keys'][prefix:].ravel(), values['values'][prefix:].ravel()]) if layer % 4 == 3 else values['conv'].ravel()
                assert digest(value) == debug['state_hashes'][layer]
            reference(state)
        workers = result['workers']
        assert workers and all(value['instructions'] < 40_000_000_000 and value['heap_pages'] * 65536 < 2**32 for value in workers)
        total = sum(value['instructions'] for value in workers)
        cases.append(dict(case=name, total_handler_instructions=total, workers=len(workers), max_heap_bytes=max(value['heap_pages'] * 65536 for value in workers), target_met=total <= 100_000_000_000))
    assert {value['case'] for value in cases} == {'617', '620', '653'}
    concurrency = read(proof / 'concurrency.json')
    assert concurrency['upgrade_order_verified'] and concurrency['busy']['result'] == {'Err': 'Busy'} and 'Ok' in concurrency['primary']['result']
    if not concurrency['active_upgrade_refused']:
        assert concurrency['upgrade_succeeded_only_with_completed_receipt'] and concurrency['completed_receipt']['state'] == {'Completed': concurrency['primary']['result']['Ok']}
    assert {'duplicate', 'id-conflict', 'insufficient', 'quote-version', 'token-bounds', 'worker-authority', 'status-authority', 'paused', 'upgrade-replay'} <= {value['name'] for value in report['checks']}
    extras = [Path(__file__), *manifests, candidate / 'source-audit-readonly.json', proof / 'report.json', proof / 'before.json', proof / 'restored.json', proof / 'concurrency.json', reconstruction / 'summary.json', reconstruction / 'reconstruction/report.json', *calls]
    if guards is not None:
        extras.extend([directory / 'upgrade-guard-entry-hashes.json', directory / 'frozen-upgrade-guards.py', guard_file,
                       directory / 'upgrade-guards/restored.json', directory / 'upgrade-guards/operations.json'])
        extras.extend(ROOT / path for path in read(directory / 'upgrade-guard-entry-hashes.json'))
    for path in extras:
        workflow[str(path.relative_to(ROOT))] = sha(path)
    reference(ROOT / 'artifacts/paid-k2-pair-guard-fold-v1/proof/report.json')
    summary = dict(complete=True, module=report['candidate'], cases=cases, all_32_hidden_verified=True, all_32_state_verified=True,
                   saved_candid_replies_verified=True, raw_candid_calls=len(calls) // 3, paid_core_api_verified=True,
                   baseline_restored=True, all_targets_met=all(value['target_met'] for value in cases),
                   deterministic_extra_upgrade_guards_verified=guards is not None, goal_complete=False,
                   workflow_hashes=workflow, reference_hashes=references)
    output = directory / 'summary.json'
    output.write_text(json.dumps(summary, indent=2) + '\n')
    archive = directory / 'frozen-paid-proof.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in [ROOT / name for name in workflow] + [output]:
            bundle.write(path, str(path.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as bundle:
        assert len(bundle.namelist()) == len(set(bundle.namelist()))
        assert bundle.read(str(output.relative_to(ROOT))) == output.read_bytes()
        for name, expected in workflow.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == expected
    print(json.dumps(dict(cases=cases, all_targets_met=summary['all_targets_met'], all_32_hidden_verified=True)))


if __name__ == '__main__':
    main()
