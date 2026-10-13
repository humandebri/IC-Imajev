#!/usr/bin/env python3
"""Compare paid inference on explicit loopback targets; never approves migration."""
import argparse
import json
import hashlib
from contextlib import nullcontext
import shutil
from pathlib import Path
import struct
import subprocess
from uuid import uuid4
from urllib.parse import urlparse
from paid_update_transport import PaidTransport
from paid_instruction_accounting import CheckpointMeter, CheckpointTrace, account, MESSAGE_INSTRUCTION_LIMIT
from paid_state_evidence import capture, validate_layout, layout_files


def bits(values):
    return b''.join(struct.pack('<f', value) for value in values).hex()


def numerical(result):
    d = result['decision']
    return dict(value=d['value'], abstained=d['abstained'], calibration=d['calibration_version'],
                logits=bits(d['raw_logits']), probabilities=bits(d['probabilities']), unknown=bits([d['unknown_probability']]))


def request_id_for_run(run_id, total):
    return f'source-migration-{run_id}-{total}'


def checkpoint_comparison(baseline, actual):
    """Diagnostic checkpoints exclude epilogues, admission and callback messages."""
    reference = sum(m['instructions'] for m in baseline['workers'])
    candidate = sum(m['instructions'] for m in actual['workers'])
    return dict(reference_worker_checkpoint_instructions=reference,
                candidate_worker_checkpoint_instructions=candidate,
                worker_checkpoint_within_105_percent=candidate * 100 <= reference * 105)


def verification_summary(reference, candidate):
    return dict(complete=False, paid_numerical_gate_passed=False,
                paid_instruction_gate_passed=False, whole_message_accounting_complete=False,
                release_eligible=False, run_id=uuid4().hex, inputs=[], failures=[],
                reference_module=reference, candidate_module=candidate,
                remaining=['Whole-message instruction accounting and budgets',
                           'Intermediate paid state comparison', 'Live fault/refund and upgrade verification'])


def main():
    if not __debug__:
        raise SystemExit('Paid verification requires Python assertions enabled')
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['reference-canister', 'candidate-canister', 'reference-module', 'candidate-module', 'relay', 'identity-pem', 'url']:
        p.add_argument('--' + name, required=True)
    for name in ['directory', 'helper', 'call-helper', 'fixture', 'bridge']:
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--lengths', help='Subset for smoke checks; full gate requires all six lengths')
    p.add_argument('--checkpoint-state-dir', type=Path, help='Explicit dedicated local subnet state directory for whole-message counters')
    p.add_argument('--checkpoint-timeout', type=int, default=180, help='Maximum seconds to wait for a fresh immutable local checkpoint')
    p.add_argument('--capture-paid-state', action='store_true', help='Retain local model-bearing heap evidence and compare all paid intermediate states')
    p.add_argument('--state-layout-evidence', type=Path, help='Typed diagnostic getter/heap conformance report')
    p.add_argument('--candidate-build-report', type=Path, help='Production source build report matching the state layout diagnostic sources')
    p.add_argument('--historical-reference-request', action='store_true', help='Explicitly encode the pinned reference module\'s former model/version request fields')
    a = p.parse_args()
    if a.checkpoint_timeout <= 0:
        p.error('--checkpoint-timeout must be positive')
    if a.capture_paid_state and not a.checkpoint_state_dir:
        p.error('--capture-paid-state requires --checkpoint-state-dir')
    if a.capture_paid_state and (not a.state_layout_evidence or not a.candidate_build_report):
        p.error('--capture-paid-state requires layout conformance and candidate build reports')
    url = urlparse(a.url)
    if url.scheme != 'http' or url.hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise SystemExit('Explicit loopback targets only')
    if a.reference_canister == a.candidate_canister:
        raise SystemExit('Distinct reference and candidate required')
    def check_modules():
        for target, expected in [(a.reference_canister, a.reference_module), (a.candidate_canister, a.candidate_module)]:
            raw = subprocess.check_output([str(a.bridge.resolve()), a.url, target, a.identity_pem], input='{"op":"module_hash"}\n', text=True)
            assert json.loads(raw)['ok']['module_hash'] == expected, 'Validation target module mismatch'
    check_modules()
    cases = json.loads(a.fixture.read_text())['cases']
    if {c['total'] for c in cases} != {9, 96, 128, 256, 512, 1024}:
        raise SystemExit('Required short and boundary fixtures missing')
    if a.lengths:
        wanted = set(map(int, a.lengths.split(',')))
        cases = [c for c in cases if c['total'] in wanted]
        if len(cases) != len(wanted):
            raise SystemExit('Unsupported length')
    a.directory.mkdir(parents=True, exist_ok=False)
    if a.capture_paid_state:
        layout_build = validate_layout(a.state_layout_evidence)
        candidate_build = json.loads(a.candidate_build_report.read_text())
        assert candidate_build['module'] == a.candidate_module and candidate_build['verification_only'] is False
        assert candidate_build['rustc'] == layout_build['rustc']
        assert all(candidate_build['source_hashes'][name] == layout_build['source_hashes'][name] for name in
                   ('canisters/inference/src/paid_types.rs', 'canisters/inference/src/paid_inference.rs'))
        for name in layout_files(a.state_layout_evidence):
            source = a.state_layout_evidence.resolve().parent/name
            destination = a.directory/'state-layout'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            # The conformance files are immutable, self-contained local evidence.
            shutil.copyfile(source, destination)
        shutil.copyfile(a.candidate_build_report, a.directory/'candidate-build-report.json')
    wires = [PaidTransport(a.directory / label, target, a.helper.resolve(), a.call_helper.resolve(), url=a.url, identity_pem=a.identity_pem)
             for label, target in [('reference', a.reference_canister), ('candidate', a.candidate_canister)]]
    if a.historical_reference_request:
        if a.reference_module != '8fd78b39b4ab626b4623826c162275737d875b7df25341d1aa2852d6e50a546c':
            raise ValueError('Historical request encoding is restricted to the known reference module')
        release = json.loads((Path(__file__).resolve().parents[1] / 'frontend/src/inference-release.json').read_text())
        wires[0].historical_request_model = release['model']
    summary = verification_summary(a.reference_module, a.candidate_module)
    if a.capture_paid_state:
        summary['state_layout_evidence'] = 'state-layout/'+a.state_layout_evidence.name
        summary['candidate_build_evidence'] = dict(file='candidate-build-report.json', sha256=hashlib.sha256(a.candidate_build_report.read_bytes()).hexdigest())
    meters = [CheckpointMeter(a.checkpoint_state_dir, target, module, timeout=a.checkpoint_timeout) for target, module in
              [(a.reference_canister, a.reference_module), (a.candidate_canister, a.candidate_module)]] if a.checkpoint_state_dir else [None, None]
    try:
        config = dict(base_fee=10_000_000_000, fee_per_token=3_000_000_000, reserve_cycles=2_000_000_000_000)
        for wire in wires:
            assert wire.call('configure_paid', config)['result'] == {'Ok': None}
            assert wire.call('paid_config')['result'] == config
            # Relay is not the inference owner and must not alter its configuration.
            assert 'transport_error' in wire.call('configure_paid', config, relay=a.relay)['result']
            assert wire.call('paid_config')['result'] == config
            worker = wire.call('inference_step', dict(job_id=0, stage=0), relay=a.relay)['result']
            assert 'Err' in worker or 'transport_error' in worker
        for case in cases:
            check_modules()
            n, request = case['total'], case['request']
            assert len(request['token_ids']) == n
            results, accounting = [], []
            for label, wire, meter in zip(['reference', 'candidate'], wires, meters):
                print(json.dumps(dict(total=n, phase=label)), flush=True)
                request_id = request_id_for_run(summary['run_id'], n)
                assert wire.call('inference_status', request_id, relay=a.relay)['result'] is None, 'Fresh request ID already has a receipt'
                quote = wire.call('quote', request)['result']['Ok']
                payload = dict(request=request, request_id=request_id)
                insufficient = wire.call('infer', payload, relay=a.relay, cycles=quote['fee'] - 1)
                assert 'InsufficientCycles' in insufficient['result']['Err']
                assert insufficient['forward']['refunded'] == quote['fee'] - 1
                before = meter.fresh(a.directory / f'n{n}-{label}-before.pbuf') if meter else None
                with (CheckpointTrace(meter, before, a.directory / f'n{n}-{label}-retries', a.directory) if meter else nullcontext()) as trace:
                    row = wire.call('infer', payload, relay=a.relay, cycles=quote['fee'] + 1_000_000)
                    result = row['result']['Ok']
                    after = meter.fresh(a.directory / f'n{n}-{label}-after.pbuf', after=before, worker_count=len(result['workers'])) if meter else None
                if meter:
                    measured = account(before, after, result['workers'], job_id=result['job_id'], retries=trace.records)
                    original = wire.d / f'{wire.counter:04d}-infer.json'
                    measured['result_evidence'] = dict(file=str(original.relative_to(a.directory.resolve())), sha256=hashlib.sha256(original.read_bytes()).hexdigest())
                    if not measured['message_budget_passed']:
                        summary['failed_measurement'] = dict(total=n, target=label, accounting=measured)
                    assert measured['message_budget_passed'], 'Whole-message paid budget exceeded'
                    if a.capture_paid_state:
                        measured['state_evidence'] = capture(meter, after, result, a.directory / f'n{n}-{label}-state', a.directory)
                    accounting.append(measured)
                assert result['paid_cycles'] == quote['fee']
                assert row['forward']['refunded'] == 1_000_000
                assert result['prefix_tokens'] == 5 and result['suffix_tokens'] == n - 5
                assert len(result['workers']) <= quote['max_steps']
                assert max(m['instructions'] for m in result['workers']) <= MESSAGE_INSTRUCTION_LIMIT
                assert row['request_bytes'] < 1_990_000 and row['reply_bytes'] < 1_990_000
                replay = wire.call('infer', payload, relay=a.relay, cycles=quote['fee'])
                assert replay['result']['Ok'] == result and replay['forward']['refunded'] == quote['fee']
                receipt = wire.call('inference_status', request_id, relay=a.relay)['result']
                assert receipt['state']['Completed'] == result and receipt['refund'] == 'None'
                assert wire.call('inference_status', request_id)['result'] is None, 'Receipt leaked across caller scope'
                changed = dict(token_ids=request['token_ids'][:], options=request['options'])
                changed['token_ids'][-1] = (changed['token_ids'][-1] + 1) % 248320
                conflict = wire.call('infer', dict(request=changed, request_id=request_id), relay=a.relay, cycles=quote['fee'])
                assert conflict['result']['Err'] == 'IdConflict' and conflict['forward']['refunded'] == quote['fee']
                results.append(result)
            baseline, actual = results
            check_modules()
            assert numerical(actual) == numerical(baseline), 'Paid numerical bits differ'
            # These counters cannot establish either whole-message budgets or
            # total instruction regression; retain them only as diagnostics.
            summary['inputs'].append(dict(total=n, request_id=request_id,
                                          **checkpoint_comparison(baseline, actual)))
            if accounting:
                ref, cand = accounting
                summary['inputs'][-1]['whole_message_accounting'] = dict(reference=ref, candidate=cand)
                assert cand['total_instructions'] * 100 <= ref['total_instructions'] * 105, 'More than 5% whole-message paid instruction regression'
                if a.capture_paid_state:
                    fields = ('hidden_hashes', 'state_hashes', 'final_hidden_bits')
                    assert all(ref['state_evidence']['state'][f] == cand['state_evidence']['state'][f] for f in fields), 'Paid intermediate state bits differ'
            (a.directory / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
        summary['complete'] = len(cases) == 6
        summary['paid_numerical_gate_passed'] = summary['complete']
        summary['whole_message_accounting_complete'] = summary['complete'] and all('whole_message_accounting' in row for row in summary['inputs'])
        summary['paid_instruction_gate_passed'] = summary['whole_message_accounting_complete']
        if summary['whole_message_accounting_complete']:
            summary['remaining'].remove('Whole-message instruction accounting and budgets')
        summary['intermediate_state_gate_passed'] = summary['complete'] and a.capture_paid_state
        if summary['intermediate_state_gate_passed']:
            summary['remaining'].remove('Intermediate paid state comparison')
    except Exception as error:
        summary['failures'].append(str(error))
        raise
    finally:
        (a.directory / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
