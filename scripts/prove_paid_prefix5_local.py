#!/usr/bin/env python3
"""Verify current paid inference on an explicitly selected, prepared LOCAL canister.

Does not install a WASM or alter the network lifecycle. Use a dedicated test
canister and an owner-managed relay; configure/cache preparation is separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from check_canister_api_exports import check_exports
from paid_prefix import COMMON_PREFIX, MAX_TOKENS, length_fixture
from paid_update_transport import PaidTransport


def verify_cases(wire, relay, record, model, lengths, fault_recovery=False, observe=lambda row: None):
    run_id = uuid.uuid4().hex[:12]
    rows = []
    for total in lengths:
        ids = length_fixture(record, total)
        request = dict(model=model, version=1, token_ids=ids, options=record['options'])
        request_id = f'{run_id}-length-{total}'
        payload = dict(request=request, request_id=request_id)
        quote = wire.call('quote', request)['result']
        if total > MAX_TOKENS:
            assert 'Invalid' in quote['Err'], quote
            denied = wire.call('infer', payload, relay=relay, cycles=1)
            assert 'Invalid' in denied['result']['Err'], denied
            assert denied['forward']['refunded'] == 1, denied
            row = dict(tokens=total, rejected_before_payment=True, call=denied)
        else:
            quote = quote['Ok']
            assert quote['prefix_tokens'] == len(COMMON_PREFIX)
            assert quote['suffix_tokens'] == total - len(COMMON_PREFIX)
            fee = quote['fee']
            result = wire.call('infer', payload, relay=relay, cycles=fee + 12345)
            assert result['forward']['refunded'] == 12345, result
            value = result['result']['Ok']  # Accepted lengths must finish, not merely refund.
            assert value['paid_cycles'] == fee
            assert value['prefix_tokens'] == 5 and value['suffix_tokens'] == total - 5
            assert value['workers'] and all(w['instructions'] < 40_000_000_000 for w in value['workers'])
            assert all(w['heap_pages'] * 65536 <= 4 * 1024**3 for w in value['workers'])
            duplicate = wire.call('infer', payload, relay=relay, cycles=fee)
            assert duplicate['result'] == result['result'] and duplicate['forward']['refunded'] == fee
            row = dict(tokens=total, suffix_tokens=total - 5, success=True,
                       duplicate_no_charge=True, quote=quote, call=result,
                       workers=len(value['workers']), seconds=result['seconds'],
                       total_instructions=sum(w['instructions'] for w in value['workers']),
                       max_worker_instructions=max(w['instructions'] for w in value['workers']),
                       max_heap_bytes=max(w['heap_pages'] for w in value['workers']) * 65536)
        rows.append(row)
        observe(row)
    if fault_recovery:
        successful = next(row for row in reversed(rows) if row.get('success'))
        payload = dict(successful['call']['request'], request_id=f'{run_id}-fault')
        fee = successful['quote']['fee']
        stage = successful['call']['result']['Ok']['workers'][0]['stage']
        wire.call('paid_fault', dict(stage=stage, trap=True, refund_fail=True))
        try:
            failed = wire.call('infer', payload, relay=relay, cycles=fee)
            assert failed['forward']['refunded'] == 0
            assert failed['result']['Err']['Failed']['refund'] == 'Pending', failed
        finally:
            wire.call('paid_fault', dict(stage=None, trap=False, refund_fail=False))
        retried = wire.call('retry_inference_refund', payload['request_id'], relay=relay)
        assert retried['result'] == {'Ok': 'Done'}, retried
        again = wire.call('retry_inference_refund', payload['request_id'], relay=relay)
        assert again['result'] == {'Ok': 'Done'}, again
        assert again['forward']['balance_after'] <= again['forward']['balance_before'], again
        replay = wire.call('infer', payload, relay=relay, cycles=fee)
        assert replay['result']['Err']['Failed']['refund'] == 'Done'
        assert replay['forward']['refunded'] == fee
        recovery = wire.call('infer', dict(payload, request_id=f'{run_id}-recovery'), relay=relay, cycles=fee)
        expected = successful['call']['result']['Ok']['decision']
        actual = recovery['result']['Ok']['decision']
        assert {k:v for k,v in actual.items() if k != 'instructions'} == {k:v for k,v in expected.items() if k != 'instructions'}
        row = dict(fault_refund_retry_no_double_refund=True, recovery_equal=True,
                   failed=failed, retried=retried, again=again, replay=replay, recovery=recovery)
        rows.append(row)
        observe(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canister', required=True, help='Prepared dedicated local model canister')
    parser.add_argument('--relay', required=True, help='Owner-managed local relay with cycles')
    parser.add_argument('--wasm', type=Path, required=True, help='Exact installed uncompressed WASM')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, help='Current TextPreparer records JSON; defaults to BOOM #653')
    parser.add_argument('--lengths', default='94,95,512,513,768,1024,1025')
    parser.add_argument('--fault-recovery', action='store_true', help='Requires the diagnostic build')
    parser.add_argument('--diagnostics', action='store_true', help='Verify a diagnostic build without requiring fault injection')
    parser.add_argument('--helper', type=Path, help='Candid helper pinned to the installed module')
    parser.add_argument('--call-helper', type=Path, help='Pinned local call helper with extended polling')
    args = parser.parse_args()
    # Reject old exports and old fixture layouts before any network request.
    check_exports(args.wasm, diagnostics=args.fault_recovery or args.diagnostics)
    if args.inputs:
        record = json.loads(args.inputs.read_text())['records'][0]
    else:
        from prepare_text import TextPreparer
        sample = json.loads((ROOT / 'frontend/src/boom-examples.generated.json').read_text())[1]
        record = TextPreparer().prepare(dict(sample['input'], id=sample['id']))
    lengths = list(map(int, args.lengths.split(',')))
    for total in lengths:
        length_fixture(record, total)
    module = hashlib.sha256(args.wasm.read_bytes()).hexdigest()
    status = json.loads(subprocess.check_output(['icp', 'canister', 'status', args.canister,
        '--network', 'local', '--identity', 'imajev-local', '--json'], text=True, cwd=ROOT))
    if status['module_hash'].removeprefix('0x') != module or status['status'] != 'Running':
        raise ValueError('local canister module/status mismatch')
    args.directory.mkdir(parents=True, exist_ok=False)
    wire = PaidTransport(args.directory / 'calls', args.canister, helper=args.helper, call_helper=args.call_helper)
    config = wire.call('paid_config')['result']
    if config['base_fee'] == 0:
        raise ValueError('Configure the tariff and prepare the dedicated canister before running this proof')
    report = dict(complete=False, network='local', canister=args.canister, relay=args.relay,
                  module=module, prefix_tokens=5, config=config, cases=[])
    def observe(row):
        report['cases'].append(row)
        (args.directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key:value for key,value in row.items() if key not in ('call','quote','failed','retried','again','replay','recovery')}), flush=True)
    try:
        verify_cases(wire, args.relay, record, json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())['model'],
                     lengths, args.fault_recovery, observe)
        report['complete'] = True
    finally:
        (args.directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
