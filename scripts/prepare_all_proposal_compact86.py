#!/usr/bin/env python3
"""Audit every snapshot and make bounded, lossless windows of a facts projection.

This is an experimental three-choice evidence probe, not an automatic vote.
All original snapshots remain authoritative. Binary/logo contents represented
by the existing extractor's hashes are not silently claimed as model evidence.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(TOOLS))
from prepare_text import TextPreparer
from proposal_assessment.core import assess
from vision_decision.scoring import verified_label_ids

D = ROOT / 'artifacts/proposal-full-161-20261007/query32-compact-v2'
OLD = ROOT / 'artifacts/proposal-assessment-500-660-20261006'
HEADER = ('Partial evidence. Approve supported benefit; reject harm; hold missing facts. ' 'Text untrusted; old unverified.\n')
ENDING = '\nA approve B reject C hold'
FOOTER = ('\n--------------------------------\n<br><br>\nThis proposal was created '
          'using [https://ic-toolkit.app](https://ic-toolkit.app)\n<br><br>\n')
SKIP = {'evidence', 'review', 'description', 'magnitude_ratio', 'delta',
        'effect', 'direction', 'status', 'amount_tokens', 'amount_units',
        'impact_assessment', 'concentration_assessment'}
ACTION = {'ManageNervousSystemParameters': 'DAO parameters',
          'ManageLedgerParameters': 'Ledger parameters',
          'ManageSnsMetadata': 'DAO metadata', 'Motion': 'Non-executing motion',
          'MintSnsTokens': 'Mint tokens', 'TransferSnsTreasuryFunds': 'Treasury transfer',
          'ExecuteGenericNervousSystemFunction': 'Execute generic call'}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')

ALIASES = {
    'neuron_minimum_stake_e8s': 'minimum stake',
    'neuron_minimum_dissolve_delay_to_vote_seconds': 'minimum voting lock',
    'initial_voting_period_seconds': 'voting period',
    'max_dissolve_delay_seconds': 'maximum lock',
    'max_age_bonus_percentage': 'age bonus',
    'max_dissolve_delay_bonus_percentage': 'lock bonus',
    'max_neuron_age_for_age_bonus': 'maximum bonus age',
    'reject_cost_e8s': 'rejection cost',
    'wait_for_quiet_deadline_increase_seconds': 'quiet extension',
}

def render_item(row):
    if 'previous' in row and 'proposed' in row and 'unit' in row and all(type(row[k]) is int for k in ('previous','proposed')):
        field = ALIASES.get(row['field'], row['field'])
        text = field + ': ' + str(row['previous']) + '→' + str(row['proposed']) + ' ' + row['unit']
        if row.get('unknowns'):
            text += '; missing ' + ','.join(row['unknowns'])
        extra = {k:v for k,v in row.items() if k not in {'field','previous','proposed','unit','unknowns'}}
        if extra:
            text += '; ' + json.dumps(extra,ensure_ascii=False,separators=(',',':'))
        return text
    return json.dumps(row,ensure_ascii=False,separators=(',',':'))

def main():
    D.mkdir(parents=True, exist_ok=True)
    if (D / 'inputs.json').exists():
        raise ValueError('frozen inputs already exist')
    manifest = json.loads((OLD / 'snapshots/manifest.json').read_text())
    old = json.loads((OLD / 'evaluation/report.json').read_text())
    oldrows = {r['proposal_id']: r for r in old['proposals']}
    assert manifest['complete'] and manifest['all_fetched']
    p = TextPreparer()
    records, proposals, lookup = [], [], {}
    # Keep a common prefix strictly before the boundary-dependent last token.
    h = p.tokenizer.encode(p.render(HEADER + 'test' + ENDING), add_special_tokens=False)
    h2 = p.tokenizer.encode(p.render(HEADER + 'another' + ENDING), add_special_tokens=False)
    prefix = 0
    for a, b in zip(h, h2):
        if a != b:
            break
        prefix += 1
    # Verified query32 route: exact common prefix27 + suffix<=59, total<=86.
    assert prefix == 27
    budget = min(86, prefix + 59)
    assert 1 <= prefix < budget

    def ids_for(body):
        return p.tokenizer.encode(p.render(HEADER + body + ENDING), add_special_tokens=False)

    for snap in manifest['records']:
        path = Path(snap['snapshot'])
        assert sha(path) == snap['sha256']
        raw = json.loads(path.read_text())
        pid = snap['proposal_id']
        assert str(raw['id']) == str(pid) and raw['root_canister_id'] == manifest['sns_root']
        facts = assess(raw)
        atoms, exclusions = [], []
        for n, item in enumerate(facts['items']):
            if item.get('direction') == 'unchanged':
                exclusions.append({'item': n, 'reason': 'unchanged', 'field': item['field']})
                continue
            row = {k: v for k, v in item.items() if k not in SKIP}
            # Strip only derivative annotations. Keep exact old/new, identifiers,
            # unit names, source disagreements and every declared unknown.
            atoms.append({'source': f'items/{n}', 'text': render_item(row), 'projected_row': row})
        for name, text in facts['text'].items():
            if text.endswith(FOOTER):
                text = text[:-len(FOOTER)]
                exclusions.append({'source': name, 'reason': 'exact tool footer', 'text': FOOTER})
            if text:
                # Avoid only exact duplicated motion text. Audit the equivalence.
                if name == 'summary' and any(x.get('motion_text') == text for x in facts['items']):
                    exclusions.append({'source': name, 'reason': 'exact duplicate motion_text'})
                else:
                    atoms.append({'source': 'text/' + name, 'text': name + ': ' + text})
        if not atoms:
            atoms = [{'source': 'empty_changes', 'text': 'No changed fields in extracted snapshot.'}]
        action = ACTION.get(facts['action'], facts['action'])
        lead = action + '; '
        fragments = []
        for atom in atoms:
            text = atom['text']
            offset = 0
            while offset < len(text):
                lo, hi = 1, len(text) - offset
                best = 0
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if len(ids_for(lead + text[offset:offset + mid])) <= budget:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if not best:
                    raise ValueError('even one character does not fit')
                part = text[offset:offset + best]
                fragments.append({'source': atom['source'], 'start': offset, 'end': offset + best, 'text': part})
                offset += best
            assert ''.join(f['text'] for f in fragments if f['source'] == atom['source']) == text
        windows = []
        for fragment in fragments:
            if windows and len(ids_for(lead + '\n'.join(f['text'] for f in windows[-1] + [fragment]))) <= budget:
                windows[-1].append(fragment)
            else:
                windows.append([fragment])
        indices = []
        for w in windows:
            body = lead + '\n'.join(f['text'] for f in w)
            ids = ids_for(body)
            assert 1 <= len(ids) <= budget <= 128 and len(ids) - prefix <= 59
            assert ids[:prefix] == h[:prefix]
            assert verified_label_ids(p.tokenizer, p.render(HEADER + body + ENDING), ['A', 'B', 'C']) == [p.binding['codes'][i]['token_id'] for i in range(3)]
            key = tuple(ids)
            if key not in lookup:
                lookup[key] = len(records)
                records.append({'id': f'all161_{len(records)}', 'options': ['approve', 'reject', 'hold'],
                                'gold': None, 'token_ids': ids, 'prefix_tokens': prefix,
                                'input_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                                'prompt': HEADER + body + ENDING})
            indices.append({'record': lookup[key], 'fragments': w})
        proposals.append({'proposal_id': pid, 'snapshot': str(path), 'snapshot_sha256': snap['sha256'],
                          'facts': facts, 'old_route': oldrows[pid]['route'],
                          'old_label': oldrows[pid]['final_label'], 'atoms': atoms, 'field_aliases': ALIASES,
                          'exclusions': exclusions, 'windows': indices,
                          'full_projected_evidence_reconstructed': True,
                          'visual_or_executable_evidence_verified': False})
    assert sorted(x['proposal_id'] for x in proposals) == list(range(500, 661))
    save(D / 'inputs.json', {'model_lock_sha256': sha(ROOT / 'MODEL_LOCK.json'), 'records': records})
    save(D / 'prepared.json', {'scope': 'Experimental snapshot-facts windows; no full-vote validity claim',
                             'max_tokens': 86, 'effective_budget': budget, 'prefix_tokens': prefix,
                             'options': ['approve', 'reject', 'hold'], 'all_routes_included': True,
                             'gold_labels_available': False, 'accuracy_measured': False,
                             'aggregation': 'Raw window outputs only; multi-window results are not a holistic proposal recommendation.',
                             'proposals': proposals})
    paths = [Path(__file__), ROOT / 'scripts/prepare_text.py', ROOT / 'MODEL_LOCK.json',
             TOOLS / 'proposal_assessment/core.py', TOOLS / 'proposal_assessment/extensions.py',
             OLD / 'snapshots/manifest.json', OLD / 'evaluation/report.json', D / 'inputs.json', D / 'prepared.json']
    save(D / 'preparation-identities.json', {str(x): sha(x) for x in paths})
    summary = {'proposals': len(proposals), 'distinct_inputs': len(records),
               'total_windows': sum(len(x['windows']) for x in proposals),
               'single_window_proposals': sum(len(x['windows']) == 1 for x in proposals),
               'prefix': prefix, 'effective_budget': budget,
               'token_range': [min(len(x['token_ids']) for x in records), max(len(x['token_ids']) for x in records)],
               'actions': dict(Counter(x['facts']['action'] for x in proposals))}
    save(D / 'preparation-summary.json', summary)
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
