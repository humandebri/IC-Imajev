#!/usr/bin/env python3
"""Compare paid update inference on a dedicated uploaded local target; delete it and its relay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from paid_update_transport import PaidTransport

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'client'))
from transport import Transport

OWNER = 'cibxp-okw3l-gfzvi-p6ltu-23ss3-7tlfz-nk65x-tvhvb-mg56y-b5hkf-eqe'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits(values):
    return hashlib.sha256(np.asarray(values, dtype='<f4').tobytes()).hexdigest()


def semantic(decision):
    return {k: v for k, v in decision.items() if k != 'instructions'}


def reference(debug, decision, path):
    hidden_layers = []
    for layer in range(32):
        hidden = path/'queries'/f'layer-{layer:02d}.npy'
        if hidden.exists():
            values = np.load(hidden, allow_pickle=False)
            assert bits(values[27:] if layer < 31 else values) == debug['hidden_hashes'][layer], (layer, 'query hidden')
            hidden_layers.append(layer)
        elif layer != 30:
            raise ValueError(f'missing query hidden: {layer}')
        with np.load(path/'queries/states'/f'layer-{layer:02d}.npz', allow_pickle=False) as state:
            values = np.concatenate([state['keys'][27:].ravel(), state['values'][27:].ravel()]) if layer % 4 == 3 else state['conv'].ravel()
            assert bits(values) == debug['state_hashes'][layer], (layer, 'query state')
    assert bits(np.load(path/'final-hidden.npy', allow_pickle=False)) == bits(debug['final_hidden'])
    old = json.loads((path/'report.json').read_text())['decision_query']['ok']['decision']
    assert semantic(decision) == semantic(old)
    for field in ('raw_logits', 'probabilities', 'unknown_probability'):
        assert bits(decision[field]) == bits(old[field]), field
    return dict(hidden_layers=hidden_layers, state_layers=list(range(32)), fused_layer30_hidden_not_exported=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    directory, build = args.directory.resolve(), args.build.resolve()
    if not directory.is_relative_to(ROOT) or not build.is_relative_to(ROOT):
        raise ValueError('evidence/build must stay in repository')
    receipt = json.loads((directory/'target.json').read_text())
    target = receipt['canister']
    network = json.loads(subprocess.check_output(['icp', 'network', 'status', '--json'], cwd=ROOT, text=True))
    if receipt['created_for'] != 'independent-runtime-paid-update-comparison' or receipt['network'] != 'http://localhost:8001/' or network['api_url'] != receipt['network']:
        raise ValueError('requires dedicated creation receipt on selected localhost network')
    if target in ('4caro-hl777-77775-aaaba-cai', 'xis3j-paaaa-aaaai-axumq-cai'):
        raise ValueError('shared/production target forbidden')
    if (directory/'comparison.json').exists():
        raise ValueError('evidence already exists')
    modules = dict(before=build/'baseline/full.wasm', after=build/'full.wasm')
    hashes = {key: sha(path) for key, path in modules.items()}
    provenance = json.loads((build/'report.json').read_text())
    assert hashes == dict(before=provenance['baseline_wasm_sha256'], after=provenance['wasm_sha256'])
    tools = ROOT/'artifacts/paid-update-v1/tools'
    tool_report = json.loads((tools/'report.json').read_text())
    assert all(sha(ROOT/name) == digest for name, digest in tool_report['source_hashes'].items())
    sources = [Path(__file__), ROOT/'scripts/paid_update_transport.py', ROOT/'scripts/prepare_weight_cache.py', ROOT/'scripts/register_common_update_prefix.py', ROOT/'artifacts/text-short-v2/inputs.json', tools/'args', tools/'caller.wasm', ROOT/'artifacts/query-packing-v3/build/imajev-client']
    sources += sorted((ROOT/'client').glob('*.py'))
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    report = dict(complete=False, local_only=True, target=target, modules=hashes, source_hashes=source_hashes,
                  build_report_sha256=sha(build/'report.json'), cases=[], operations=[], cleanup={},
                  instruction_scope='Paid worker counter-zero checkpoints; recording/reply tail excluded. Replicated execution success separately checked.')
    relay = None

    def save():
        (directory/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')

    def icp(*command):
        result = subprocess.run(['icp', 'canister', *command, '--network', 'local', '--identity', 'imajev-local'], cwd=ROOT, text=True, capture_output=True)
        report['operations'].append(dict(command=list(command), returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        save()
        result.check_returncode()
        return result.stdout.strip()

    records = json.loads((ROOT/'artifacts/text-short-v2/inputs.json').read_text())['records']
    manifest = json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
    fixtures = []
    for n in (28, 84, 86, 116, 117, 256, 257, 512):
        original = records[1] if n == 86 else records[2]
        if n < 84:
            ids = original['token_ids'][:27]+original['token_ids'][-(n-27):]
        else:
            extra = ([198, 220, 16]*((n-len(original['token_ids'])+2)//3))[:n-len(original['token_ids'])]
            ids = original['token_ids'][:-4]+extra+original['token_ids'][-4:]
        assert len(ids) == n
        fixtures.append((f'tokens-{n}', dict(model=manifest['model'], version=1, token_ids=ids, options=original['options']), n if n in (84, 86) else None))
    labels = ['no', 'yes', 'hold', 'approve', 'reject', 'insufficient', 'unknown']
    for count in range(2, 8):
        request = dict(fixtures[0][1], options=labels[:count])
        fixtures.append((f'options-{count}', request, None))
    before = {}
    try:
        status = json.loads(icp('status', target, '--json'))
        assert status['module_hash'].removeprefix('0x') == hashes['before']
        assert int(status['settings']['wasm_memory_limit'].replace('_', '')) == 4*2**30
        relay = icp('create', '--detached', '--cycles', '30t', '--quiet')
        report['relay'] = relay
        save()
        icp('install', relay, '--mode', 'install', '--wasm', str(tools/'caller.wasm'), '--args', f'(principal "{OWNER}")', '--yes')
        replay_payload = replay_result = replay_fee = None
        for variant, module in modules.items():
            stage = directory/variant
            stage.mkdir(exist_ok=False)
            wire = PaidTransport(stage/'calls', target)
            if variant == 'after':
                icp('install', target, '--mode', 'upgrade', '--wasm', str(module), '--yes')
                replay = wire.call('infer', replay_payload, relay=relay, cycles=replay_fee)
                assert replay['result'] == replay_result and replay['forward']['refunded'] == replay_fee
                report['receipt_replay_after_upgrade_without_rewarm'] = True
                save()
            for script, name, flags in (
                ('prepare_weight_cache.py', 'weights', ['--include-f32', '--require-prepared-rope', '--require-prepared-activation', '--require-all-output-pairs']),
                ('register_common_update_prefix.py', 'prefix', [])):
                with (stage/f'{name}.log').open('w') as stream:
                    subprocess.run([sys.executable, '-B', str(ROOT/'scripts'/script), '--canister', target, '--wasm', str(module), '--directory', str(stage/name), *flags], cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
                print(variant, name, 'ready', flush=True)
            version = 2 if variant == 'before' else 3
            config = dict(enabled=True, version=version, base_fee=100_000_000_000, fee_per_token=3_000_000_000, reserve_cycles=2_000_000_000_000)
            assert 'Ok' in wire.call('configure_paid', config)['result']
            assert wire.call('inference_step', dict(job_id=1, stage=0))['result'] == {'Err': 'self only'}
            for label, request, query_tokens in fixtures:
                n = len(request['token_ids'])
                quote = wire.call('quote', request)['result']['Ok']
                assert quote['fee'] == config['base_fee']+(n-27)*config['fee_per_token']
                payload = dict(request=request, request_id=f'{variant}-{label}', quote_version=version)
                row = wire.call('infer', payload, relay=relay, cycles=quote['fee']+12345)
                assert 'Ok' in row['result'], (variant, label, row['result'])
                assert row['forward']['refunded'] == 12345
                result = row['result']['Ok']
                assert result['paid_cycles'] == quote['fee'] and result['quote_version'] == version
                assert result['prefix_tokens'] == 27 and result['suffix_tokens'] == n-27
                workers = result['workers']
                assert 0 < len(workers) <= quote['max_steps']
                assert all(0 < w['instructions'] < 40_000_000_000 and w['heap_pages']*65536 <= 2**32 for w in workers)
                debug = wire.call('paid_debug')['result']
                assert debug['job_id'] == result['job_id']
                assert len(debug['hidden_hashes']) == len(debug['state_hashes']) == 32
                (stage/f'{label}-debug.json').write_text(json.dumps(debug, indent=2)+'\n')
                duplicate = wire.call('infer', payload, relay=relay, cycles=quote['fee'])
                assert duplicate['result'] == row['result'] and duplicate['forward']['refunded'] == quote['fee']
                metrics = dict(workers=len(workers), instructions=sum(w['instructions'] for w in workers), max_worker_instructions=max(w['instructions'] for w in workers), max_heap_bytes=max(w['heap_pages']*65536 for w in workers), seconds=row['seconds'], returned_unused_cycles=12345, duplicate_no_charge=True)
                query_coverage = None
                if query_tokens:
                    record = 1 if query_tokens == 86 else 2
                    qpath = ROOT/'artifacts/laya-removal-20261008/pruned-full-model-proof'/variant/f'record-{record}'
                    query_coverage = reference(debug, result['decision'], qpath)
                item = dict(label=label, tokens=n, options=len(request['options']), variant=variant, metrics=metrics, debug_sha256=sha(stage/f'{label}-debug.json'), query_reference_bitwise_equal=bool(query_tokens), query_reference_coverage=query_coverage)
                if variant == 'before':
                    before[label] = (debug, result, metrics, request)
                    if n == 512:
                        replay_payload, replay_result, replay_fee = payload, row['result'], quote['fee']
                else:
                    old_debug, old_result, old_metrics, old_request = before[label]
                    assert old_request == request
                    assert debug['hidden_hashes'] == old_debug['hidden_hashes'] and debug['state_hashes'] == old_debug['state_hashes']
                    assert bits(debug['final_hidden']) == bits(old_debug['final_hidden'])
                    assert semantic(result['decision']) == semantic(old_result['decision'])
                    for field in ('raw_logits', 'probabilities', 'unknown_probability'):
                        assert bits(result['decision'][field]) == bits(old_result['decision'][field])
                    item.update(bitwise_equal=True, before=old_metrics, instruction_ratio=metrics['instructions']/old_metrics['instructions'])
                report['cases'].append(item)
                save()
                print(json.dumps(item), flush=True)
            # Validate errors before funds are accepted, including request-ID conflicts.
            valid = fixtures[0][1]
            quote = wire.call('quote', valid)['result']['Ok']
            checks = []
            for label, request, request_id, qversion, funds, expected in (
                ('empty-suffix', dict(valid, token_ids=valid['token_ids'][:27]), f'{variant}-empty', version, quote['fee'], 'Invalid'),
                ('prefix', dict(valid, token_ids=[0]+valid['token_ids'][1:]), f'{variant}-bad-prefix', version, quote['fee'], 'Invalid'),
                ('limit513', dict(valid, token_ids=valid['token_ids']+([16]*485)), f'{variant}-overflow', version, quote['fee'], 'Invalid'),
                ('quote', valid, f'{variant}-old-quote', version-1, quote['fee'], 'QuoteChanged'),
                ('funds', valid, f'{variant}-low-funds', version, quote['fee']-1, 'InsufficientCycles'),
                ('conflict', dict(valid, options=labels[:2]), f'{variant}-tokens-28', version, quote['fee'], 'IdConflict')):
                response = wire.call('infer', dict(request=request, request_id=request_id, quote_version=qversion), relay=relay, cycles=funds)
                assert expected in response['result'].get('Err', {}), response['result']
                assert response['forward']['refunded'] == funds
                checks.append(dict(label=label, expected=expected, rejected_before_payment=True, all_attached_cycles_returned=True))
            report[variant+'_validation'] = checks
            save()
            # The owner entrypoints share the graph with paid infer, but have
            # their own start/continuation wrappers and response framing.
            direct = Transport(manifest['model'], receipt['network'], target,
                str(ROOT/'artifacts/imajev-local.pem'), stage/'owner-calls', manifest['pack_hash'],
                bridge_binary=str(ROOT/'artifacts/query-packing-v3/build/imajev-client'))
            calls = []
            try:
                request = next(r for label, r, _ in fixtures if label == 'tokens-84')
                command = dict(op='update_infer_start', ids=request['token_ids'][27:], options=request['options'])
                for _ in range(8):
                    reply = direct.command(command)
                    calls.append(reply)
                    progress = reply['ok']['progress']
                    assert progress['instructions'] < 40_000_000_000
                    if progress['done']:
                        break
                    command = dict(op='update_infer_continue', id=progress['id'], stage=progress['stage'])
                assert progress['done'], 'owner continuation bound reached'
                qpath = ROOT/'artifacts/laya-removal-20261008/pruned-full-model-proof'/variant/'record-2'
                coverage = reference(progress, progress['decision'], qpath)
                old_debug = before['tokens-84'][0]
                assert progress['hidden_hashes'] == old_debug['hidden_hashes']
                assert progress['state_hashes'] == old_debug['state_hashes']
                assert bits(progress['final_hidden']) == bits(old_debug['final_hidden'])
                errors = []
                for command, expected in (
                    (dict(op='update_infer_continue', id=progress['id'], stage=progress['stage']), 'session progress mismatch'),
                    (dict(op='update_infer_start', ids=[], options=request['options']), 'suffix token bounds')):
                    try:
                        direct.command(command)
                    except RuntimeError as error:
                        assert expected in str(error), str(error)
                        errors.append(expected)
                    else:
                        raise AssertionError('invalid owner request accepted')
                path = stage/'owner-update.json'
                path.write_text(json.dumps(dict(calls=calls, rejected=errors), indent=2)+'\n')
                report[variant+'_owner_update'] = dict(tokens=84, calls=len(calls), query_reference_bitwise_equal=True,
                    all32_hidden_and_state_equal_to_paid=True, query_reference_coverage=coverage, report_sha256=sha(path), rejected=errors,
                    instructions=sum(r['ok']['progress']['instructions'] for r in calls))
                save()
                print(variant, 'owner update verified', flush=True)
            finally:
                direct.close()
        assert hashes == {key: sha(path) for key, path in modules.items()}
        assert source_hashes == {str(p.relative_to(ROOT)): sha(p) for p in sources}
        assert all(sha(ROOT/name) == digest for name, digest in provenance['source_hashes'].items())
        report['complete'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        for name, canister in (('target', target), ('relay', relay)):
            if canister is None:
                continue
            report['cleanup'][name] = {}
            for action in ('stop', 'delete'):
                result = subprocess.run(['icp', 'canister', action, canister, '--network', 'local', '--identity', 'imajev-local'], cwd=ROOT, text=True, capture_output=True)
                report['cleanup'][name][action] = dict(returncode=result.returncode, stderr=result.stderr)
                if result.returncode:
                    report['complete'] = False
        save()
    if not report['complete']:
        raise RuntimeError('update verification or dedicated-canister cleanup incomplete')


if __name__ == '__main__':
    main()
