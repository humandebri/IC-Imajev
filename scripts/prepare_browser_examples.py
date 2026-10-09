#!/usr/bin/env python3
"""Generate compact built-in examples from their retained original excerpts."""
import argparse
import hashlib
import json
from fractions import Fraction
from decimal import Decimal
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
    _, facts = encode({'state': json.dumps(dict(action='ManageNervousSystemParameters', changes_or_requests=rows))})
    by_field = {fact['field']: fact for fact in facts}
    supported = {'neuron_minimum_dissolve_delay_to_vote_seconds', 'max_dissolve_delay_seconds'}
    if 'neuron_minimum_dissolve_delay_to_vote_seconds' not in by_field or not by_field.keys() <= supported:
        raise ValueError('unsupported parameter combination')
    minimum = by_field['neuron_minimum_dissolve_delay_to_vote_seconds']
    old, new = minimum['rendered_values']
    unit = minimum['unit_group']
    if len(facts) == 1:
        previous_unit = unit[:-1] if old == '1' else unit
        state = f'The minimum voting lock duration changes from {old} {previous_unit} to {new} {unit}.'
    else:
        maximum = by_field['max_dissolve_delay_seconds']
        # Hours retain both exact maximum values with fewer tokenizer tokens.
        maximum['unit_group'] = 'hours'
        maximum['rendered_values'] = [format(Decimal(maximum[key]) / 3600, 'f') for key in ('previous', 'proposed')]
        before, after = maximum['rendered_values']
        state = f'Minimum voting lock changes from {old} to {new} {unit}; maximum from {before} to {after} hours.'
    # Verify exact reconstruction, including fractional days, before publishing.
    for fact in facts:
        if fact['unit_group'] not in ('days', 'hours', 'seconds'):
            raise ValueError('unexpected built-in parameter unit')
        scale = {'days': 86400, 'hours': 3600, 'seconds': 1}[fact['unit_group']]
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
    target = f'{aliases[principal]} (principal/account)' if principal == account else f'principal {aliases[principal]}, account {aliases[account]}'
    state = f'Mint {tokens} SNS tokens to {target}, memo {int(memo)}. Supply and prior holdings unknown.'
    return state, dict(kind='mint', amount_tokens=amount, amount_e8s=e8s,
                      principal=principal, account=account, memo=int(memo), identifiers=identifiers,
                      missing_from_excerpt=['total_supply', 'recipient_prior_holdings'])


def compact_treasury_transfer(text):
    path = ROOT / 'frontend/data/boom-584-proposal.json'
    raw = path.read_bytes()
    proposal = json.loads(raw)
    payload = proposal['proposal_action_payload']
    if (str(proposal['id']) != '584' or proposal['root_canister_id'] != 'xjngq-yaaaa-aaaaq-aabha-cai'
            or proposal['proposal_action_type'] != 'TransferSnsTreasuryFunds'
            or proposal['status'] != 'REJECTED'
            or text != proposal['proposal_title'] + '\n\n' + proposal['payload_text_rendering']
            or proposal['proposal_title'] != 'SNS Metadata Adjustment'
            or payload['from_treasury'] != 2 or payload['amount_e8s'] != 2000000000000000
            or payload['to_subaccount'] is not None or payload['memo'] is not None):
        raise ValueError('treasury transfer source identity or terms mismatch')
    principal = payload['to_principal']
    if f'## Target account: {principal}\n' not in text or '## Amount (e8s): 2000000000000000\n' not in text:
        raise ValueError('treasury rendering differs from payload')
    return ('Title: SNS Metadata Adjustment. Payload: send 20M BOOM from DAO treasury to P1.'), dict(
        kind='treasury_transfer', action=proposal['proposal_action_type'], status=proposal['status'],
        amount_e8s=str(payload['amount_e8s']), from_treasury=payload['from_treasury'],
        proposal_sha256=hashlib.sha256(raw).hexdigest(),
        identifiers=[dict(alias='P1', value=principal)],
        semantics='Rejected proposal: evaluate the requested execution, not a completed transfer. Title/action mismatch is observable; malicious intent is not established by the payload alone.')


def attach_ledger(state, audit):
    snapshot = json.loads((ROOT / 'frontend/data/boom-653-ledger.json').read_text())
    if (snapshot['rootCanisterId'] != 'xjngq-yaaaa-aaaaq-aabha-cai'
            or snapshot['ledgerCanisterId'] != 'vtrom-gqaaa-aaaaq-aabia-cai'
            or snapshot['recipient'] != audit['principal'] or audit['principal'] != audit['account']
            or snapshot['subaccount'] is not None or snapshot['decimals'] != 8):
        raise ValueError('ledger snapshot identity mismatch')
    supply, balance = (int(snapshot[key]) for key in ('totalSupplyE8s', 'recipientBalanceE8s'))
    if not 0 <= balance <= supply or supply <= 0:
        raise ValueError('invalid ledger supply or balance')
    def bounds(value, scale):
        lower = value // scale
        upper = (value + scale - 1) // scale
        return str(lower) if lower == upper else f'{lower}-{upper}'
    # Enclosing ranges retain enough evidence within the 96-token budget.
    # Exact query values stay in the audit and are shown alongside the input.
    mint = int(audit['amount_e8s'])
    post_supply, post_balance = supply + mint, balance + mint
    # An observable financial metric, computed from raw data, not an answer label.
    percent_low = 100 * post_balance // post_supply
    percent_high = (100 * post_balance + post_supply - 1) // post_supply
    state = (f'Supply {bounds(supply, 10**14)}M tokens; account {bounds(balance, 10**8)}. '
             f'Mint {format_million(audit["amount_tokens"])} to account. '
             f'New share {percent_low}-{percent_high}%.')
    audit['derived'] = dict(post_supply_e8s=str(post_supply), post_balance_e8s=str(post_balance),
                           share_percent_bounds=[percent_low, percent_high],
                           formula='100 * (recipientBalanceE8s + amount_e8s) / (totalSupplyE8s + amount_e8s)')
    audit['input_bounds'] = dict(supply_scale_e8s=10**14, balance_scale_e8s=10**8,
                                supply=[supply // 10**14, (supply + 10**14 - 1) // 10**14],
                                balance=[balance // 10**8, (balance + 10**8 - 1) // 10**8])
    audit['ledger_snapshot'] = snapshot
    audit['missing_from_excerpt'] = []
    return state, audit


def analyze_neurons(snapshot, previous, proposed):
    if snapshot['rootCanisterId'] != 'xjngq-yaaaa-aaaaq-aabha-cai':
        raise ValueError('neuron snapshot identity mismatch')
    neurons = snapshot['neurons']
    if len(neurons) != snapshot['totalNeurons'] or len({n['id'] for n in neurons}) != len(neurons):
        raise ValueError('incomplete or duplicate neuron snapshot')
    reference = snapshot['referenceTimestampSeconds']
    before, after = [], []
    for neuron in neurons:
        power = int(neuron['voting_power'])
        if power < 0:
            raise ValueError('negative voting power')
        state = neuron['dissolve_state']
        if state is None:
            delay = 0
        elif set(state) == {'DissolveDelaySeconds'}:
            delay = state['DissolveDelaySeconds']
        elif set(state) == {'WhenDissolvedTimestampSeconds'}:
            delay = max(0, state['WhenDissolvedTimestampSeconds'] - reference)
        else:
            raise ValueError('unknown dissolve state')
        if type(delay) is not int or delay < 0:
            raise ValueError('invalid dissolve delay')
        if power > 0 and delay >= previous:
            before.append(neuron)
        if power > 0 and delay >= proposed:
            after.append(neuron)
    def group(rows):
        total = sum(int(n['voting_power']) for n in rows)
        largest = max((int(n['voting_power']) for n in rows), default=0)
        return dict(count=len(rows), indexed_voting_power=str(total), largest_neuron_voting_power=str(largest),
                    largest_share_bps_floor=(largest * 10000 // total) if total else None)
    lost = len({n['id'] for n in before} - {n['id'] for n in after})
    return dict(before=group(before), after=group(after), excluded_neurons=lost,
                excluded_percent_bps_floor=(lost * 10000 // len(before)) if before else None)


def attach_neurons(state, audit):
    path = ROOT / 'frontend/data/boom-617-neurons.json'
    raw = path.read_bytes()
    snapshot = json.loads(raw)
    minimum = next(fact for fact in audit['facts'] if fact['field'] == 'neuron_minimum_dissolve_delay_to_vote_seconds')
    analysis = analyze_neurons(snapshot, minimum['previous'], minimum['proposed'])
    before, after = analysis['before'], analysis['after']
    old, new = minimum['rendered_values']
    share = after['largest_share_bps_floor']
    concentration = f'largest remaining indexed voting share >={share // 100}%.' if share is not None else 'No voting power remains.'
    state = (f'Current-neuron simulation, not historical: voting lock {old}->{new} days; '
             f'eligible neurons {before["count"]}->{after["count"]}; ' + concentration + ' No relocking.')
    audit['participation_snapshot'] = dict(
        **{key: snapshot[key] for key in ('sourceUrl', 'sourceKind', 'startedAt', 'completedAt', 'referenceTimestampSeconds', 'totalNeurons', 'scenario')},
        sha256=hashlib.sha256(raw).hexdigest(), analysis=analysis,
        semantics='Neurons, not unique people. No relocking. Positive indexed voting power and delay >= threshold. Concentration holds indexed weights fixed; bonuses are not recalculated.')
    return state, audit


def generate():
    from prepare_text import TextPreparer
    preparer = TextPreparer()
    result = []
    samples = json.loads(SOURCE.read_text())
    if [sample['id'] for sample in samples] != ['boom-584', 'boom-653', 'boom-617']:
        raise ValueError('only the three approved built-in examples are supported')
    for sample in samples:
        original = sample['input']
        if sample['id'] == 'boom-653':
            compact, audit = attach_ledger(*compact_mint(original['state']))
        elif sample['id'] == 'boom-584':
            compact, audit = compact_treasury_transfer(original['state'])
        else:
            compact, audit = attach_neurons(*compact_parameters(original['state']))
        question = original['question']
        def prepare(state):
            return preparer.prepare(dict(original, id=sample['id'], state=state, question=question))
        record = prepare(compact)
        release = json.loads((ROOT / 'frontend/src/inference-release.json').read_text())
        max_suffix = release['max_tokens'] - release['prefix_tokens']
        if not 1 <= len(record['token_ids']) - record['prefix_tokens'] <= max_suffix or record['prefix_tokens'] != 5:
            raise ValueError(f'{sample["id"]} does not fit the public query budget')
        source_hash = hashlib.sha256(original['state'].encode()).hexdigest()
        result.append(dict(sample, input=dict(original, state=compact, question=question), originalState=original['state'], originalQuestion=original['question'],
                           sourceStateSha256=source_hash, tokens=len(record['token_ids']), audit=audit,
                           compactionVersion=4))
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
