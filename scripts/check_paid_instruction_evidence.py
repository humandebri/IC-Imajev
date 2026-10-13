"""Revalidate original replica counters and paid results before migration approval."""
import argparse
import hashlib
import json
from pathlib import Path
from paid_instruction_accounting import account, decode_snapshot
from verify_optimized_paid import numerical, request_id_for_run
from paid_state_evidence import validate as validate_state, validate_layout


def checked(directory, name, expected):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe paid evidence path')
    current = directory
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink in paid evidence path')
    raw = current.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Paid evidence checksum mismatch')
    return raw


def validate(report_path):
    directory = report_path.resolve().parent
    summary = json.loads(report_path.read_text())
    if not all(summary.get(k) is True for k in ('complete', 'paid_numerical_gate_passed', 'paid_instruction_gate_passed', 'whole_message_accounting_complete')) or summary.get('failures'):
        raise ValueError('Incomplete paid numerical/instruction evidence')
    if sorted(row['total'] for row in summary['inputs']) != [9, 96, 128, 256, 512, 1024]:
        raise ValueError('Paid length coverage mismatch')
    if summary.get('intermediate_state_gate_passed') is True:
        layout = Path(summary['state_layout_evidence'])
        if layout.is_absolute() or '..' in layout.parts:
            raise ValueError('Unsafe layout conformance path')
        layout_build = validate_layout(directory/layout)
        build_evidence = summary['candidate_build_evidence']
        candidate = json.loads(checked(directory,build_evidence['file'],build_evidence['sha256']))
        if candidate['module'] != summary['candidate_module'] or candidate['verification_only'] is not False or candidate['rustc'] != layout_build['rustc']:
            raise ValueError('State conformance does not match production module/compiler')
        if any(candidate['source_hashes'][name] != layout_build['source_hashes'][name] for name in
               ('canisters/inference/src/paid_types.rs','canisters/inference/src/paid_inference.rs')):
            raise ValueError('State conformance source mismatch')
    for row in summary['inputs']:
        total = row['total']
        if row['request_id'] != request_id_for_run(summary['run_id'], total):
            raise ValueError('Paid request identity mismatch')
        results, measurements, states = [], [], []
        for label in ('reference', 'candidate'):
            module = summary[label+'_module']
            recorded = row['whole_message_accounting'][label]
            snapshots = []
            for point in ('before', 'after'):
                original = recorded[point]
                raw = checked(directory, f'n{total}-{label}-{point}.pbuf', original['sha256'])
                snapshots.append(decode_snapshot(raw, original['height'], original['canister'], module))
            evidence = recorded['result_evidence']
            wire = json.loads(checked(directory, evidence['file'], evidence['sha256']))
            if wire['kind'] != 'infer' or wire['request']['request_id'] != row['request_id'] or len(wire['request']['request']['token_ids']) != total:
                raise ValueError('Paid original result identity mismatch')
            for field in ('request_bytes', 'reply_bytes'):
                size = wire.get(field)
                if type(size) is not int or not 0 < size < 1_990_000:
                    raise ValueError('Paid communication budget exceeded or unavailable')
            result = wire['result']['Ok']
            retries = []
            for original in recorded.get('retry_evidence', []):
                raw = checked(directory, original['evidence_file'], original['sha256'])
                retry = decode_snapshot(raw, original['height'], original['canister'], module)
                retry['evidence_file'] = original['evidence_file']
                if retry != original:
                    raise ValueError('Retry report differs from original checkpoint')
                retries.append(retry)
            measured = account(*snapshots, result['workers'], job_id=result.get('job_id'), retries=retries)
            measured['result_evidence'] = evidence
            if 'state_evidence' in recorded:
                states.append(validate_state(directory, recorded['state_evidence'], snapshots[1], result['job_id']))
                measured['state_evidence'] = recorded['state_evidence']
            if measured != recorded or not measured['message_budget_passed']:
                raise ValueError('Paid accounting report differs from original evidence')
            results.append((wire['request']['request'], result))
            measurements.append(measured['total_instructions'])
        if results[0][0] != results[1][0] or numerical(results[0][1]) != numerical(results[1][1]):
            raise ValueError('Paid original input/numerical mismatch')
        if measurements[1] * 100 > measurements[0] * 105:
            raise ValueError('Paid whole-message instruction regression')
        if summary.get('intermediate_state_gate_passed') is True:
            if len(states) != 2 or any(states[0][f] != states[1][f] for f in ('hidden_hashes','state_hashes','final_hidden_bits')):
                raise ValueError('Incomplete or mismatched paid intermediate states')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    validate(args.report)
    print('Verified original paid whole-message counters, budgets and numerical comparisons.')
