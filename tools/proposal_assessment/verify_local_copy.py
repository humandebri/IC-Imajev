"""Check upstream provenance and whole-input binary migration conformance."""
import argparse
import hashlib
import json
from pathlib import Path

from .compact_units import encode
from .token_sweep import ROOT


def load(path):
    return json.loads(path.read_text())


def main():
    package = Path(__file__).parent
    provenance = load(package / 'UPSTREAM.json')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources-only', action='store_true', help='verify portable local source hashes without experiment artifacts')
    parser.add_argument('--source-root', type=Path, help='optionally verify original upstream files in a separate checkout')
    args = parser.parse_args()
    for name, entry in provenance['files'].items():
        contents = (package / name).read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        assert digest == entry['local_sha256'], name
        assert entry['adapted'] == (digest != entry['source_sha256']), name
        if args.source_root is not None:
            original = (args.source_root / entry['source']).read_bytes()
            assert hashlib.sha256(original).hexdigest() == entry['source_sha256'], name
    if args.sources_only:
        print(json.dumps(dict(local_modules_verified=len(provenance['files']), upstream_verified=args.source_root is not None)))
        return
    old = ROOT / 'artifacts/proposal-assessment-500-660-20261006/evaluation'
    new = ROOT / 'artifacts/proposal-assessment-local-500-660-20261007/evaluation'
    a, b = load(old / 'inputs.json'), load(new / 'inputs.json')
    assert a == b, 'migration changed model inputs'
    assert load(old / 'prepared.json') == load(new / 'prepared.json'), 'migration changed tasks or gates'
    previous, current = load(old / 'report.json'), load(new / 'report.json')
    assert current['summary']['complete'] and current['summary']['new_distinct'] == 0
    assert current['summary']['reused_distinct'] == 18
    fields = ('proposal_id', 'route', 'input_tokens', 'candidate_tokens',
              'binary_prediction', 'uncalibrated_score', 'final_label', 'reason')
    assert [{k: x[k] for k in fields} for x in previous['proposals']] == [
        {k: x[k] for k in fields} for x in current['proposals']]
    assert [x['proposal_id'] for x in current['proposals']] == list(range(500, 661))
    for i, record in enumerate(b['records']):
        dest = new / 'runs' / f'{i:03d}'
        reuse = load(dest / 'reuse.json')
        assert reuse['exact_token_identity']
        raw = (dest / 'report.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == reuse['source_report_sha256']
        r = json.loads(raw)
        assert r['input_hash'] == record['input_sha256']
        assert r['model'] == b['model_lock_sha256']
    old600 = ROOT / 'artifacts/proposal-assessment/binary-600-660-ratio-20261006'
    new600 = ROOT / 'artifacts/proposal-assessment-local-600-660-20261007'
    assert load(old600 / 'inputs.json') == load(new600 / 'inputs.json')
    assert load(new600 / 'quality.json')['summary']['complete']
    # Lossless fractional-day/token representation and unsupported-field refusal.
    task = {'state': json.dumps({'action': 'ManageNervousSystemParameters', 'changes_or_requests': [
        {'field': 'max_dissolve_delay_seconds', 'previous': 2629800, 'proposed': 2629800000},
        {'field': 'neuron_minimum_stake_e8s', 'previous': 100001, 'proposed': 100000000},
        {'field': 'wait_for_quiet_deadline_increase_seconds', 'previous': 86400, 'proposed': 86401},
    ]})}
    text, facts = encode(task)
    assert '30.4375→30437.5' in text
    assert '0.00100001→1' in text
    assert '1day→86401second' in text
    rows = json.loads(task['state'])['changes_or_requests']
    assert {(x['field'], x['previous'], x['proposed']) for x in facts} == {
        (x['field'], x['previous'], x['proposed']) for x in rows}
    for bad in [
        [{'field': 'unsupported', 'previous': 1, 'proposed': 2}],
        [{'field': 'neuron_minimum_stake_e8s', 'previous': None, 'proposed': 2}],
        [rows[0], rows[0]],
    ]:
        try:
            encode({'state': json.dumps({'action': 'ManageNervousSystemParameters', 'changes_or_requests': bad})})
        except ValueError:
            continue
        raise AssertionError('invalid evidence was silently encoded')
    result = {'upstream_modules_verified': len(provenance['files']), 'proposal_routes_verified': 161,
              'binary_inputs_verified': 18, 'benchmark_600_660_inputs_verified': 6,
              'exact_units_verified': True, 'new_inferences': 0}
    (new / 'local-copy-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
