"""Known SNS action shapes and conservative Dashboard-rendering reconciliation."""
import copy
import hashlib
import re

from .core import uint


def evidence(path, **details):
    return {'path': path, **details}


def opaque_text(value):
    if not isinstance(value, str):
        return copy.deepcopy(value)
    return {'characters': len(value), 'sha256': hashlib.sha256(value.encode()).hexdigest(),
            'value_omitted': True}


def rendering_updates(proposal, action):
    text = proposal.get('payload_text_rendering', '')
    if not isinstance(text, str):
        return {}, set()
    if action == 'ManageLedgerParameters':
        heading = '# Proposal to change ledger parameters:'
        specs = {'token_name': ('# Set token name: ', r'(.*)\.\s*'),
                 'token_symbol': ('# Set token symbol: ', r'(.*)\.\s*'),
                 'token_logo': ('# Set token logo: ', r'(.*)\.\s*'),
                 'transfer_fee': ('# Set token transfer fee: ', r'(\d+) token-quantums\.\s*')}
    else:
        heading = '# Proposal to upgrade sns metadata:'
        specs = {field: ('# New ' + field + ': ', r'(.*)') for field in ('url', 'name', 'description')}
        specs['logo'] = ('# New logo (base64 encoding):', None)
    if not text.startswith(heading + '\n'):
        return {}, set()
    parsed, ambiguous = {}, set()
    lines = text.splitlines()
    for field, (prefix, pattern) in specs.items():
        starts = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        if not starts:
            continue
        if len(starts) != 1:
            ambiguous.add(field)
            continue
        i = starts[0]
        raw = lines[i][len(prefix):]
        if pattern is None:
            # Official metadata renderer places the logo last. Reject new headings.
            tail = lines[i+1:]
            if raw.strip() or any(line.startswith('# ') for line in tail):
                ambiguous.add(field)
                continue
            value = '\n'.join(tail).strip()
        else:
            match = re.fullmatch(pattern, raw)
            if not match:
                ambiguous.add(field)
                continue
            value = match[1].rstrip() if action == 'ManageSnsMetadata' else match[1]
            if field == 'description':
                continuation = []
                for line in lines[i+1:]:
                    if line.startswith('# '):
                        break
                    continuation.append(line)
                if continuation:
                    value += '\n' + '\n'.join(continuation)
            if field == 'transfer_fee':
                value = int(value)
                if not uint(value):
                    ambiguous.add(field)
                    continue
        parsed[field] = value
    return parsed, ambiguous


def changes(action, proposal, payload):
    known = ({'transfer_fee', 'token_name', 'token_symbol', 'token_logo'} if action == 'ManageLedgerParameters'
             else {'url', 'name', 'description', 'logo'})
    rendered, ambiguous = rendering_updates(proposal, action)
    rows = []
    for field in sorted(set(payload) | set(rendered) | ambiguous):
        value = payload.get(field)
        if value is None and field not in rendered and field not in ambiguous:
            continue
        structured = value is not None
        item = {'field': field, 'status': 'insufficient_data', 'review': 'review',
                'previous': None, 'unknowns': ['previous_value'], 'evidence': []}
        if field not in known:
            item.update(status='unsupported', proposed=copy.deepcopy(value), unknowns=['field_semantics'])
            item['evidence'].append(evidence('/proposal_action_payload/' + field.replace('~', '~0').replace('/', '~1')))
            rows.append(item)
            continue
        if structured:
            item['evidence'].append(evidence('/proposal_action_payload/' + field))
        if field in rendered:
            item['evidence'].append(evidence('/payload_text_rendering', method='known_renderer_parse',
                                             chain_state_verified=False, field=field))
        if field in ambiguous:
            item['source_status'] = 'ambiguous_rendering'
            item['unknowns'].append('unambiguous_rendering')
        elif structured and field in rendered and value != rendered[field]:
            item['source_status'] = 'conflicting_sources'
            item['unknowns'].append('consistent_proposed_value')
        elif not structured:
            value = rendered.get(field)
            item['source_status'] = 'rendering_only'
            item['structured_field_missing_or_null'] = True
            item['unknowns'].append('structured_proposed_value')
        else:
            item['source_status'] = 'consistent' if field in rendered else 'structured_only'
        valid = uint(value) if field == 'transfer_fee' else isinstance(value, str) and bool(value)
        if not valid:
            item['unknowns'].append('valid_proposed_value')
        item['proposed'] = opaque_text(value) if field in ('logo', 'token_logo') else copy.deepcopy(value)
        if item['source_status'] == 'conflicting_sources':
            item['rendered_proposed'] = (opaque_text(rendered[field]) if field in ('logo', 'token_logo')
                                         else rendered[field])
        item['effect'] = 'requests_' + field + '_update'
        rows.append(item)
    return rows


def transfer(payload):
    treasury = payload.get('from_treasury')
    amount = payload.get('amount_e8s')
    unknowns = ['treasury_balance_before', 'recipient_control', 'recipient_relationship']
    valid = type(treasury) is int and treasury in (1, 2)
    if not valid:
        unknowns.append('known_treasury_type')
    amount_valid = uint(amount) and amount > 0
    if not amount_valid:
        unknowns.append('valid_positive_amount')
    recipient = payload.get('to_principal')
    recipient_valid = isinstance(recipient, str) and bool(recipient)
    if not recipient_valid:
        unknowns.append('recipient')
    memo = payload.get('memo')
    memo_valid = memo is None or uint(memo)
    if not memo_valid:
        unknowns.append('valid_optional_memo')
    subaccount = payload.get('to_subaccount')
    sub_valid = (subaccount is None or
        isinstance(subaccount, str) and re.fullmatch(r'[0-9a-fA-F]{64}', subaccount) is not None or
        isinstance(subaccount, list) and len(subaccount) == 32 and
        all(type(v) is int and 0 <= v <= 255 for v in subaccount))
    if not sub_valid:
        unknowns.append('valid_optional_32_byte_subaccount')
    item = {'field': 'treasury_transfer', 'status': 'assessed' if all((valid, amount_valid, recipient_valid, memo_valid, sub_valid)) else 'insufficient_data',
            'review': 'review', 'effect': 'treasury_transfer_requested',
            'from_treasury': treasury, 'asset': {1: 'ICP', 2: 'SNS_TOKEN'}.get(treasury) if type(treasury) is int else None,
            'amount_e8s': amount, 'recipient': recipient, 'memo': memo,
            'subaccount': copy.deepcopy(subaccount), 'unknowns': unknowns,
            'impact_assessment': 'insufficient_data', 'evidence': [evidence('/proposal_action_payload')]}
    if amount_valid:
        # Avoid decimal context and float rounding, including the full uint64 range.
        item['amount_units'] = f'{amount // 100000000}.{amount % 100000000:08d}'
    return [item]


def generic(proposal, payload):
    fid, raw = payload.get('function_id'), payload.get('payload')
    item = {'field': 'generic_call', 'status': 'insufficient_data', 'review': 'review',
            'function_id': fid, 'effect': 'generic_call_requested',
            'unknowns': ['execution_semantics', 'target_code'],
            'evidence': [evidence('/proposal_action_payload')]}
    definition = proposal.get('nervous_system_function')
    if isinstance(definition, dict) and uint(fid) and fid >= 1000 and str(definition.get('id')) == str(fid):
        types = definition.get('function_type', {})
        target = types.get('GenericNervousSystemFunction') if isinstance(types, dict) else None
        if isinstance(target, dict):
            item['declared_target'] = copy.deepcopy(target)
            item['evidence'].append(evidence('/nervous_system_function', chain_state_verified=False))
        else:
            item['unknowns'].append('generic_function_definition')
    else:
        item['unknowns'].append('matching_function_definition')
    if isinstance(raw, list) and all(type(v) is int and 0 <= v <= 255 for v in raw):
        blob = bytes(raw)
        item.update(payload_bytes=len(blob), payload_sha256=hashlib.sha256(blob).hexdigest())
    else:
        item['unknowns'].append('raw_payload_bytes')
    text = proposal.get('payload_text_rendering', '')
    if isinstance(text, str):
        hashes = re.findall(r'^## Payload sha256:\s*\n\s*([0-9a-f]{64})\s*$', text, re.MULTILINE)
        if len(hashes) == 1:
            item['rendered_payload_sha256'] = hashes[0]
            if item.get('payload_sha256') != hashes[0]:
                item['unknowns'].append('payload_hash_matches_rendering')
                item['source_status'] = 'conflicting_sources'
        elif len(hashes) > 1:
            item['unknowns'].append('unambiguous_rendered_payload_hash')
    return [item]


def assess_action(action, proposal, payload):
    if action in ('ManageLedgerParameters', 'ManageSnsMetadata'):
        return changes(action, proposal, payload)
    if action == 'TransferSnsTreasuryFunds':
        rows = transfer(payload)
        known = {'from_treasury', 'amount_e8s', 'memo', 'to_principal', 'to_subaccount'}
    elif action == 'ExecuteGenericNervousSystemFunction':
        rows = generic(proposal, payload)
        known = {'function_id', 'payload'}
    else:
        text = payload.get('motion_text')
        rows = [{'field': 'motion', 'status': 'assessed' if isinstance(text, str) and text else 'insufficient_data',
                 'review': 'review', 'effect': 'non_executing_motion', 'motion_text': copy.deepcopy(text),
                 'unknowns': ['future_followup_actions'], 'evidence': [evidence('/proposal_action_payload/motion_text')]}]
        known = {'motion_text'}
    for field, value in sorted(payload.items()):
        if field not in known and value is not None:
            rows.append({'field': field, 'status': 'unsupported', 'review': 'review',
                         'proposed': copy.deepcopy(value), 'unknowns': ['field_semantics'],
                         'evidence': [evidence('/proposal_action_payload/' + field.replace('~', '~0').replace('/', '~1'))]})
    return rows
