#!/usr/bin/env python3
"""Verify source coverage, exact tokenizer output and bounded query32 inputs."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/proposal-full-161-20261007'
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_text import TextPreparer

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    for path, digest in json.loads((D / 'preparation-identities.json').read_text()).items():
        assert sha(Path(path)) == digest
    fixture = json.loads((D / 'inputs.json').read_text())
    prepared = json.loads((D / 'prepared.json').read_text())
    p = TextPreparer()
    records = fixture['records']
    assert len({tuple(r['token_ids']) for r in records}) == len(records)
    common = records[0]['token_ids'][:42]
    for r in records:
        assert 42 < len(r['token_ids']) <= 128
        assert r['token_ids'][:42] == common
        assert len(r['token_ids'][42:]) <= 80
        assert p.tokenizer.encode(p.render(r['prompt']), add_special_tokens=False) == r['token_ids']
        assert hashlib.sha256(json.dumps(r['token_ids']).encode()).hexdigest() == r['input_sha256']
        assert r['gold'] is None
    assert sorted(x['proposal_id'] for x in prepared['proposals']) == list(range(500, 661))
    fragments_checked = 0
    changed_fields = 0
    for proposal in prepared['proposals']:
        assert sha(Path(proposal['snapshot'])) == proposal['snapshot_sha256']
        atoms = {a['source']: a['text'] for a in proposal['atoms']}
        parts = {key: [] for key in atoms}
        for window in proposal['windows']:
            r = records[window['record']]
            text = '\n'.join(f['text'] for f in window['fragments'])
            assert text in r['prompt']
            for f in window['fragments']:
                assert atoms[f['source']][f['start']:f['end']] == f['text']
                parts[f['source']].append(f)
                fragments_checked += 1
        for name, fragments in parts.items():
            pos = 0
            for f in fragments:
                assert f['start'] == pos
                pos = f['end']
            assert pos == len(atoms[name])
            assert ''.join(f['text'] for f in fragments) == atoms[name]
        for n, item in enumerate(proposal['facts']['items']):
            if item.get('direction') == 'unchanged':
                continue
            changed_fields += 1
            projected = json.loads(atoms[f'items/{n}'])
            for key in ('previous', 'proposed', 'unit', 'recipient', 'amount_e8s',
                        'function_id', 'unknowns', 'source_status', 'declared_target',
                        'payload_sha256', 'rendered_payload_sha256'):
                if key in item:
                    assert projected[key] == item[key]
    report = {'verified': True, 'proposals': 161, 'distinct_inputs': len(records),
              'prefix_tokens': 42, 'suffix_max_tokens': 80, 'total_max_tokens': 128,
              'actual_max_tokens': max(len(x['token_ids']) for x in records),
              'fragments_checked': fragments_checked, 'changed_items_preserved': changed_fields,
              'source_coverage': 'Full atoms reconstructed, exact changed values retained',
              'raw_snapshot_lossless': False, 'binaries_and_code_verified': False,
              'whole_proposal_equivalence_proven': False,
              'verified_files': {str(x): sha(x) for x in (D / 'inputs.json', D / 'prepared.json', Path(__file__))}}
    (D / 'input-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
