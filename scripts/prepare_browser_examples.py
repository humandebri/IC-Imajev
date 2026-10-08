#!/usr/bin/env python3
"""Generate compact built-in examples from their retained original excerpts."""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from tools.proposal_assessment.compact_units import encode
from tools.proposal_assessment.million_notation import format_million, exact_value

SOURCE = ROOT / 'frontend/data/boom-examples.json'
OUTPUT = ROOT / 'frontend/src/boom-examples.generated.json'
PARAMETER = re.compile(r'\s*([a-z_]+):\s*Some\(\s*(\d+),\s*\),')


def parameter_rows(text):
    heading = 'SNS parameters adjustment\n\n# Proposal to change nervous system parameters:\n## Current nervous system parameters:'
    if not text.startswith(heading):
        raise ValueError('unsupported parameter excerpt')
    sections = text[len(heading):].split('\n## New nervous system parameters:')
    if len(sections) != 2:
        raise ValueError('one old/new section required')
    parsed = []
    for section in sections:
        values, end = {}, 0
        for match in PARAMETER.finditer(section):
            if section[end:match.start()].strip() or match[1] in values:
                raise ValueError('unparsed or duplicate parameter')
            value = int(match[2])
            if not 0 <= value < 2**64:
                raise ValueError('parameter outside uint64')
            values[match[1]], end = value, match.end()
        if section[end:].strip() or not values:
            raise ValueError('unparsed parameter text')
        parsed.append(values)
    old, new = parsed
    if old.keys() != new.keys():
        raise ValueError('old/new fields differ')
    return [dict(field=field, previous=old[field], proposed=new[field]) for field in old]


def compact_parameters(text):
    rows = parameter_rows(text)
    state, facts = encode({'state': json.dumps(dict(action='ManageNervousSystemParameters', changes_or_requests=rows))})
    # Verify exact reconstruction, including fractional days, before publishing.
    for fact in facts:
        if fact['unit_group'] not in ('days', 'seconds'):
            raise ValueError('unexpected built-in parameter unit')
        scale = 86400 if fact['unit_group'] == 'days' else 1
        for key, rendered in zip(('previous', 'proposed'), fact['rendered_values']):
            if Fraction(rendered) * scale != fact[key]:
                raise ValueError('unit conversion lost precision')
    return state, dict(kind='parameters', facts=facts, identifiers=[])


def compact_mint(text):
    pattern = (r'SNS Adjustment\n\n# Proposal to mint SNS Tokens:\n'
               r'## Amount: ([0-9]+(?:\.[0-9]+)?) SNS Tokens\n'
               r'## Amount \(e8s\): ([0-9]+)\n'
               r'## Target principal: ([^\n]+)\n## Target account: ([^\n]+)\n## Memo: ([0-9]+)')
    match = re.fullmatch(pattern, text)
    if not match:
        raise ValueError('unsupported mint excerpt; no text may be dropped')
    amount, e8s, principal, account, memo = match.groups()
    if Fraction(amount) * 100000000 != int(e8s) or not 0 < int(e8s) < 2**64 or not 0 <= int(memo) < 2**64:
        raise ValueError('inconsistent or invalid mint values')
    tokens = format_million(amount)
    if exact_value(tokens) != Fraction(amount):
        raise ValueError('amount formatting lost precision')
    identifiers = []
    aliases = {}
    for value in (principal, account):
        if value not in aliases:
            alias = f'P{len(aliases) + 1}'
            aliases[value] = alias
            identifiers.append(dict(alias=alias, value=value))
    target = f'principal/account {aliases[principal]}' if principal == account else f'principal {aliases[principal]}; account {aliases[account]}'
    state = f'Mint {tokens} SNS tokens; {target}; memo {int(memo)}. Supply and prior holdings unknown.'
    return state, dict(kind='mint', amount_tokens=amount, amount_e8s=e8s,
                      principal=principal, account=account, memo=int(memo), identifiers=identifiers,
                      missing_from_excerpt=['total_supply', 'recipient_prior_holdings'])


def generate():
    from prepare_text import TextPreparer
    preparer = TextPreparer()
    result = []
    samples = json.loads(SOURCE.read_text())
    if [sample['id'] for sample in samples] != ['boom-620', 'boom-653', 'boom-617']:
        raise ValueError('only the three approved built-in examples are supported')
    for sample in samples:
        original = sample['input']
        compact, audit = compact_mint(original['state']) if sample['id'] == 'boom-653' else compact_parameters(original['state'])
        def prepare(state):
            return preparer.prepare(dict(original, id=sample['id'], state=state))
        record = prepare(compact)
        if len(record['token_ids']) > 96 and audit['kind'] == 'parameters':
            # Only generated labels/unit punctuation; never rewrite free text.
            compact = compact.replace(': ', ':')
            for fact in audit['facts']:
                compact = compact.replace(' ' + fact['rendered_values'][0] + '→', fact['rendered_values'][0] + '→')
            record = prepare(compact)
        if not 28 <= len(record['token_ids']) <= 96 or record['prefix_tokens'] != 27:
            raise ValueError(f'{sample["id"]} does not fit the public query budget')
        source_hash = hashlib.sha256(original['state'].encode()).hexdigest()
        result.append(dict(sample, input=dict(original, state=compact), originalState=original['state'],
                           sourceStateSha256=source_hash, tokens=len(record['token_ids']), audit=audit,
                           compactionVersion=1))
    return (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    encoded = generate()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_bytes() != encoded:
            raise SystemExit('Generated BOOM examples are stale; run prepare_browser_examples.py')
    else:
        OUTPUT.write_bytes(encoded)
    print('Verified compact BOOM examples: ' + ', '.join(f'{s["label"]}={s["tokens"]} tokens' for s in json.loads(encoded)))


if __name__ == '__main__':
    main()
