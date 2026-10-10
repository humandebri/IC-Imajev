#!/usr/bin/env python3
"""Measure paid token boundaries on a snapshot-protected LOCAL canister."""

from historical_paid_proof import historical_only
historical_only()
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport, encode, atomic
from prefix_inference import verify_module, load_cache, PrefixTextGraph
from prefix_hybrid import load_packets
from full_inference import JournalTransport
from paid_update_transport import PaidTransport

BASELINE = '6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
OWNER = 'cibxp-okw3l-gfzvi-p6ltu-23ss3-7tlfz-nk65x-tvhvb-mg56y-b5hkf-eqe'
TARGET = '4caro-hl777-77775-aaaba-cai'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bits(v):
    return hashlib.sha256(np.asarray(v, dtype='<f4').tobytes()).hexdigest()


def write(p, v):
    p.write_text(json.dumps(v, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--lengths', default='84,85,95,116,117,132,500')
    ap.add_argument('--expected-limit', type=int, default=500)
    ap.add_argument('--fault-recovery', action='store_true')
    ap.add_argument('--reference-lengths', help='CSV subset; defaults to every successful length')
    ap.add_argument('--saved-reference-directories', type=Path, nargs='+', default=[],
                    help='Reuse complete independent-query references for identical input fixtures')
    ap.add_argument('--require-success', action='store_true', help='Fail if any accepted length fails inference')
    a = ap.parse_args()
    source = a.source_root.resolve()
    network_storage = (source / '.icp/cache/networks/local').resolve()
    free_bytes = shutil.disk_usage(network_storage).free
    if free_bytes < 30 * 2**30:
        ap.error(f'local snapshot-protected proof needs at least 30 GiB free in {network_storage}; found {free_bytes / 2**30:.2f} GiB')
    # Reuse immutable model assets and existing helper binaries in a worktree.
    # Measurement outputs always go into its own, real artifacts directory.
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    for name in ['target', 'checkpoints', 'artifacts/paid-update-v1',
                 'artifacts/query-packing-v3', 'artifacts/prefix-state-cache-v1',
                 'artifacts/imajev-local.pem']:
        destination = ROOT / name
        if not destination.exists():
            destination.symlink_to(source / name)
    d = a.directory.resolve()
    d.mkdir(parents=True, exist_ok=False)
    build = a.build.resolve()
    module = sha(build / 'full.wasm')
    build_report = json.loads((build / 'report.json').read_text())
    assert module == build_report.get('module', build_report.get('wasm_sha256'))
    # All CLI operations explicitly target the already inspected local network.
    real_icp = shutil.which('icp')
    bindir = d / 'bin'
    bindir.mkdir()
    shim = bindir / 'icp'
    shim.write_text('#!/bin/sh\nexec ' + shlex.quote(real_icp) + ' "$@" --project-root-override ' + shlex.quote(str(source)) + '\n')
    shim.chmod(0o755)
    os.environ['PATH'] = str(bindir) + os.pathsep + os.environ['PATH']
    operations = []

    def icp(*args):
        result = subprocess.run(['icp', 'canister', *args, '--network', 'local',
                                 '--identity', 'imajev-local'], text=True, cwd=ROOT, capture_output=True)
        operations.append(dict(args=list(args), output=result.stdout, stderr=result.stderr, returncode=result.returncode))
        write(d / 'operations.json', operations)
        result.check_returncode()
        return result.stdout.strip()

    manifest = json.loads((source / 'checkpoints/full-int8.manifest.json').read_text())
    bridge_binary = source / 'artifacts/query-packing-v3/build/imajev-client'

    def bridge(path):
        return Transport(manifest['model'], 'http://localhost:8001/', TARGET,
                         str(source / 'artifacts/imajev-local.pem'), path,
                         manifest['pack_hash'], bridge_binary=str(bridge_binary))

    t = bridge(d / 'before')
    try:
        verify_module(t, BASELINE)
        before = dict(module=BASELINE, cache=t.command(dict(op='weight_cache_status'))['ok']['cache'],
                      pack=t.command(dict(op='pack_status'))['ok'])
    finally:
        t.close()
    assert before == json.loads((source / 'artifacts/local-goal-recovery-v1/baseline-state.json').read_text())
    status = json.loads(icp('status', TARGET, '--json'))
    assert status['status'] == 'Running' and status['settings']['controllers'] == [OWNER]
    write(d / 'before.json', before)
    snapshot = None
    caller = None
    restored = False
    stop_attempted = False
    report = dict(complete=False, local_only=True, target=TARGET, endpoint='http://localhost:8001/',
                  module=module, cases=[], query_comparisons=[],
                  network_storage=str(network_storage), free_bytes_before=free_bytes,
                  instruction_scope='Counter zero checkpoint; small recording/reply tail excluded. Actual replicated message success separately checked.')
    try:
        # A failed CLI response may still have stopped the canister.
        stop_attempted = True
        icp('stop', TARGET)
        snapshot = icp('snapshot', 'create', TARGET, '--quiet')
        write(d / 'snapshot.json', dict(id=snapshot, target=TARGET, baseline=BASELINE))
        print('baseline snapshot saved', flush=True)
        icp('install', TARGET, '--mode', 'upgrade', '--wasm', str(build / 'full.wasm'), '--yes')
        icp('start', TARGET)
        caller = icp('create', '--detached', '--cycles', '10t', '--quiet')
        write(d / 'caller.json', dict(id=caller))
        icp('install', caller, '--mode', 'install', '--wasm',
            str(source / 'artifacts/paid-update-v1/tools/caller.wasm'),
            '--args', f'(principal "{OWNER}")', '--yes')
        for script, name, flags in [
            ('prepare_weight_cache.py', 'weights', ['--include-f32', '--require-prepared-rope',
             '--require-prepared-activation', '--require-all-output-pairs']),
            ('prepare_fixed_prefix_states.py', 'fixed-prefix', []),
        ]:
            with (d / f'{name}.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT / 'scripts' / script), '--canister', TARGET,
                                '--wasm', str(build / 'full.wasm'), '--directory', str(d / name), *flags],
                               cwd=ROOT, check=True, stdout=log, stderr=log)
            print(name + ' prepared', flush=True)
        cache_path = source / 'artifacts/query-packing-v3/prefix-v2/queries'
        packets_path = source / 'artifacts/query-packing-v3/packets-v2'
        cache = load_cache(cache_path, manifest, BASELINE)
        cache['hybrid_packets'] = load_packets(packets_path, cache)
        t = bridge(d / 'prefix-registration')
        prefix_rows = []
        try:
            for layer, state in enumerate(cache['states']):
                v = np.concatenate([state['keys'].transpose(1, 0, 2).ravel(),
                                    state['values'].transpose(1, 0, 2).ravel()]) if layer % 4 == 3 else state['conv'].ravel()
                path = d / f'prefix-{layer:02d}.f32'
                np.asarray(v, dtype='<f4').tofile(path)
                cmd = dict(op='update_prefix', layer=layer, values=str(path))
                if layer % 4 != 3:
                    cmd['packet'] = str(packets_path / f'layer-{layer:02d}.npf1')
                prefix_rows.append(t.command(cmd))
        finally:
            t.close()
        write(d / 'prefix-registration.json', prefix_rows)
        wire = PaidTransport(d / 'calls', TARGET)
        caller_wire = PaidTransport(d / 'caller-balance', caller)
        config = dict(base_fee=100_000_000_000,
                      fee_per_token=3_000_000_000, reserve_cycles=2_000_000_000_000)
        assert 'Ok' in wire.call('configure_paid', config)['result']
        denied = wire.call('inference_step', dict(job_id=1, stage=0))['result']
        assert denied == {'Err': 'self only'}, denied
        report['outside_worker_rejected'] = True
        records = json.loads((source / 'artifacts/text-short-v2/inputs.json').read_text())['records']
        original = records[2]
        assert len(original['token_ids']) == 84
        debug_by_length = {}
        requests = {}
        for n in map(int, a.lengths.split(',')):
            # Synthetic length stress fixtures: preserve the original token IDs
            # and final four chat tokens; insert deterministic ordinary tokens.
            assert 32 <= n <= 513, n
            if n < 84:
                ids = original['token_ids'][:n-4] + original['token_ids'][-4:]
            else:
                extra = ([198, 220, 16] * ((n - 84 + 2) // 3))[:n - 84]
                ids = original['token_ids'][:-4] + extra + original['token_ids'][-4:]
            assert len(ids) == n and ids[:27] == cache['metadata']['token_ids']
            req = dict(model=manifest['model'], version=1, token_ids=ids, options=original['options'])
            requests[n] = req
            quoted = wire.call('quote', req)['result']
            if n > a.expected_limit:
                assert 'Invalid' in quoted.get('Err', {}), quoted
                rejected = wire.call('infer', dict(request=req, request_id=f'length-{n}'), relay=caller, cycles=100_000_000_000)
                assert 'Invalid' in rejected['result'].get('Err', {}), rejected
                assert rejected['forward']['refunded'] == 100_000_000_000
                report['cases'].append(dict(tokens=n, success=False, rejected_before_payment=True, full_cycles_returned=True, call=rejected))
                write(d / 'report.json', report)
                print(f'{n} rejected before payment', flush=True)
                continue
            quote = quoted['Ok']
            payload = dict(request=req, request_id=f'length-{n}')
            balance_before = caller_wire.call('balance')['result']
            inference_before = int(json.loads(icp('status', TARGET, '--json'))['cycles'].replace('_', ''))
            row = wire.call('infer', payload, relay=caller, cycles=quote['fee'] + 12345)
            balance_after = caller_wire.call('balance')['result']
            inference_after = int(json.loads(icp('status', TARGET, '--json'))['cycles'].replace('_', ''))
            item = dict(tokens=n, suffix_tokens=n - 27, quote=quote, call=row,
                        input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        balance_before=balance_before, balance_after=balance_after,
                        inference_cycles_before=inference_before, inference_cycles_after=inference_after,
                        execution_cycles=inference_before + quote['fee'] - inference_after)
            assert row['forward']['refunded'] == 12345
            if 'Ok' in row['result']:
                result = row['result']['Ok']
                workers = result['workers']
                item.update(success=True, workers=len(workers), seconds=row['seconds'],
                            instructions=sum(w['instructions'] for w in workers),
                            max_worker_instructions=max(w['instructions'] for w in workers),
                            max_heap_bytes=max(w['heap_pages'] * 65536 for w in workers))
                assert item['max_worker_instructions'] < 40_000_000_000
                assert item['max_heap_bytes'] <= 2**32
                debug = wire.call('paid_debug')['result']
                assert debug['job_id'] == result['job_id']
                assert len(debug['hidden_hashes']) == len(debug['state_hashes']) == 32
                debug_by_length[n] = debug
                write(d / f'debug-{n}.json', debug)
                duplicate = wire.call('infer', payload, relay=caller, cycles=quote['fee'])
                assert duplicate['result'] == row['result'] and duplicate['forward']['refunded'] == quote['fee']
                item['duplicate_no_charge'] = True
            else:
                failed = row['result']['Err']['Failed']
                assert failed['refund'] == 'Done', failed
                # Verify the fee actually reached the caller balance; sender
                # execution/communication charges remain separate.
                assert balance_after > balance_before - quote['fee'] // 2
                item.update(success=False, reason=failed['reason'], refund=failed['refund'], seconds=row['seconds'])
            report['cases'].append(item)
            write(d / 'report.json', report)
            print(json.dumps({k: v for k, v in item.items() if k not in ['call', 'quote']}), flush=True)
            if a.require_success:
                assert item['success'], (n, item.get('reason'))
        if a.fault_recovery:
            n=max(debug_by_length)
            successful=next(c for c in report['cases'] if c['tokens']==n)
            stage=successful['call']['result']['Ok']['workers'][0]['stage']
            wire.call('paid_fault', dict(stage=stage, trap=True, refund_fail=False))
            payload=dict(request=requests[n], request_id='carry-fault')
            fee=successful['quote']['fee']
            fault_balance_before=caller_wire.call('balance')['result']
            failed=wire.call('infer',payload,relay=caller,cycles=fee)
            fault_balance_after=caller_wire.call('balance')['result']
            assert fault_balance_after > fault_balance_before-fee//2
            # Only relay/query execution charges remain after returning the fee.
            assert 0 <= fault_balance_before-fault_balance_after < 100_000_000
            assert failed['result']['Err']['Failed']['refund']=='Done', failed
            wire.call('paid_fault',dict(stage=None, trap=False, refund_fail=False))
            repeated=wire.call('infer',payload,relay=caller,cycles=fee)
            assert repeated['result']==failed['result'] and repeated['forward']['refunded']==fee
            recovery=wire.call('infer',dict(request=requests[84],request_id='after-carry-fault'),relay=caller,
                               cycles=next(c for c in report['cases'] if c['tokens']==84)['quote']['fee'])
            got=recovery['result']['Ok']['decision']
            want=next(c for c in report['cases'] if c['tokens']==84)['call']['result']['Ok']['decision']
            assert {k:v for k,v in got.items() if k!='instructions'}=={k:v for k,v in want.items() if k!='instructions'}
            report['fault_recovery']=dict(stage=stage,failed=failed,duplicate=repeated,recovery=recovery,
                                         fee_refunded=True,caller_balance_before=fault_balance_before,caller_balance_after=fault_balance_after,
                                         failed_duplicate_no_charge=True,fresh_inference_equal=True)
            write(d/'report.json',report)
            print('mid-stream trap refund and fresh-job recovery verified',flush=True)
        # Independently orchestrate ordinary queries for each newly successful
        # length, preserving exact arithmetic and all layer/state comparisons.
        for n in debug_by_length:
            if a.reference_lengths and n not in set(map(int,a.reference_lengths.split(','))):
                continue
            if n not in debug_by_length:
                continue
            req = requests[n]
            qd = d / f'query-reference-{n}'
            saved = None
            for candidate in a.saved_reference_directories:
                old = json.loads((candidate / 'report.json').read_text())
                assert old['complete'] and old['baseline_restored']
                if n not in {v['tokens'] for v in old['query_comparisons']}:
                    continue
                previous = next(v for v in old['cases'] if v['tokens']==n)
                current = next(v for v in report['cases'] if v['tokens']==n)
                assert previous['input_sha256']==current['input_sha256']
                # Hash includes the entire public request: model, version, IDs and options.
                old_inputs = list((candidate / 'calls').glob('*.input.json'))
                assert any(json.loads(p.read_text())==dict(request=req,request_id=f'length-{n}') for p in old_inputs)
                debug=debug_by_length[n]
                origin=candidate / f'query-reference-{n}'
                for layer in range(32):
                    values=np.load(origin / f'layer-{layer:02d}.npy',allow_pickle=False)
                    assert bits(values[27:] if layer<31 else values)==debug['hidden_hashes'][layer],(n,layer,'saved hidden')
                    with np.load(origin / 'states' / f'layer-{layer:02d}.npz',allow_pickle=False) as z:
                        state=np.concatenate([z['keys'][27:].ravel(),z['values'][27:].ravel()]) if layer%4==3 else z['conv'].ravel()
                        assert bits(state)==debug['state_hashes'][layer],(n,layer,'saved state')
                assert bits(np.load(origin / 'final-hidden.npy',allow_pickle=False))==bits(debug['final_hidden'])
                decision=json.loads((origin / 'decision.json').read_text())
                paid=current['call']['result']['Ok']['decision']
                assert {k:v for k,v in decision.items() if k!='instructions'}=={k:v for k,v in paid.items() if k!='instructions'}
                qd.symlink_to(origin.resolve(),target_is_directory=True)
                saved=dict(tokens=n,all32_hidden_and_state_equal=True,final_hidden_equal=True,
                           decision_probabilities_logits_equal=True,saved_reference_directory=str(origin.resolve()),
                           saved_report_sha256=sha(candidate / 'report.json'),
                           ordinary_queries=next(v['ordinary_queries'] for v in old['query_comparisons'] if v['tokens']==n),
                           fresh_reference_queries=False)
                break
            if saved:
                report['query_comparisons'].append(saved)
                write(d / 'report.json',report)
                print(json.dumps(saved),flush=True)
                continue
            t = JournalTransport(manifest['model'], 'http://localhost:8001/', TARGET,
                                 str(source / 'artifacts/imajev-local.pem'), qd, manifest['pack_hash'],
                                 wire_codec='bf16-block256-exact-v1',
                                 input_hash=hashlib.sha256(json.dumps(req['token_ids']).encode()).hexdigest(),
                                 bridge_binary=str(bridge_binary))
            try:
                verify_module(t, module)
                graph = PrefixTextGraph(t, manifest, cache=cache, arithmetic='int8',
                                        compact_heads=True, compact_lossless=True, retain_terminal_state=False, terminal_readout=True,
                                        fuse_add_norm=True, fuse_mlp=True, fuse_mlp_norm=True,
                                        fuse_mlp_pipeline=True, fuse_mlp_full=True, mlp_full_token_cap=89,
                                        fuse_delta=True, fuse_delta_projected=True, fuse_delta_full_log=True,
                                        fuse_norm_rope=True, fuse_attention=True, fuse_attention_full=True,
                                        fuse_terminal_attention=True)
                if n > 116:
                    from full_inference import TextGraph
                    graph = TextGraph(t, manifest, arithmetic='int8', row_cap=8192, token_cap=64, work_cap=2_000_000_000,
                                      delta_head_cap=16, attention_head_cap=4, compact_heads=True, compact_lossless=True,
                                      retain_terminal_state=False, terminal_readout=True, fuse_add_norm=True,
                                      fuse_mlp=True, fuse_mlp_norm=True, fuse_norm_rope=True, fuse_delta=True, fuse_attention=False)
                # Production decisions are returned by the final query itself.
                t.fuse_terminal_decision = n <= 116
                t.decision_options = req['options']
                hidden = graph.forward(req['token_ids'])
                debug = debug_by_length[n]
                assert bits(hidden[-1]) == bits(debug['final_hidden']), (n, 'final hidden')
                for layer in range(32):
                    values = np.load(qd / f'layer-{layer:02d}.npy', allow_pickle=False)
                    assert bits(values[27:] if layer < 31 else values) == debug['hidden_hashes'][layer], (n, layer, 'hidden')
                    with np.load(qd / 'states' / f'layer-{layer:02d}.npz', allow_pickle=False) as z:
                        state = np.concatenate([z['keys'][27:].ravel(), z['values'][27:].ravel()]) if layer % 4 == 3 else z['conv'].ravel()
                        assert bits(state) == debug['state_hashes'][layer], (n, layer, 'state')
                np.save(qd / 'final-hidden.npy', hidden[-1])
                result = getattr(t, 'terminal_decision', None)
                paid = next(c for c in report['cases'] if c['tokens'] == n)['call']['result']['Ok']['decision']
                if result is not None:
                    write(qd / 'decision.json', result)
                    assert {k: v for k, v in result.items() if k != 'instructions'} == {k: v for k, v in paid.items() if k != 'instructions'}
                comparison = dict(tokens=n, all32_hidden_and_state_equal=True, final_hidden_equal=True,
                                  decision_probabilities_logits_equal=result is not None, ordinary_queries=len(t.measurements))
                if result is None:
                    comparison['decision_comparison_unavailable'] = 'Unfused long reference graph has no final decision query; use an independent saved reference for probability comparison.'
                report['query_comparisons'].append(comparison)
                write(qd / 'comparison.json', comparison)
                write(d / 'report.json', report)
                print(json.dumps(comparison), flush=True)
            finally:
                t.close()
        report['complete'] = True
    except Exception as error:
        report['error'] = str(error)
        write(d / 'failure.json', dict(error=str(error)))
        raise
    finally:
        restoration_error = None
        if snapshot or stop_attempted:
            print('restoring baseline snapshot' if snapshot else 'restarting unchanged baseline', flush=True)
            try:
                if snapshot:
                    icp('stop', TARGET)
                    icp('snapshot', 'restore', TARGET, snapshot)
                icp('start', TARGET)
                t = bridge(d / 'restored')
                try:
                    verify_module(t, BASELINE)
                    assert t.command(dict(op='weight_cache_status'))['ok']['cache'] == before['cache']
                    assert t.command(dict(op='pack_status'))['ok'] == before['pack']
                    restored = True
                finally:
                    t.close()
                if restored and snapshot:
                    icp('snapshot', 'delete', TARGET, snapshot)
                    report['snapshot_deleted'] = True
            except Exception as error:
                restoration_error = error
                report['restoration_error'] = str(error)
        stopped = False
        if caller:
            try:
                icp('stop', caller)
                stopped = True
            except Exception as error:
                report['caller_stop_error'] = str(error)
        report['baseline_restored'] = restored
        report['test_caller_stopped'] = stopped
        write(d / 'report.json', report)
        print(json.dumps(dict(complete=report['complete'], baseline_restored=restored)), flush=True)
        if restoration_error:
            raise RuntimeError('baseline restoration failed; available snapshot and failure details retained') from restoration_error



if __name__ == '__main__':
    main()
