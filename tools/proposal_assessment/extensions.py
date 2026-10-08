"""Typed facts for ledger, metadata, treasury, motion and generic-call snapshots."""
from copy import deepcopy
import hashlib
import re
from .core import uint, _unsupported

_LEDGER = {'token_name':('# Set token name: ',r'(.*)\.\s*'),
           'token_symbol':('# Set token symbol: ',r'(.*)\.\s*'),
           'token_logo':('# Set token logo: ',r'(.*)\.\s*'),
           'transfer_fee':('# Set token transfer fee: ',r'(\d+) token-quantums\.\s*')}
_METADATA = {name:('# New '+name+': ',r'(.*)') for name in ('url','name','description')}
_METADATA['logo'] = ('# New logo (base64 encoding):',None)

def opaque_text(value):
    if isinstance(value,str):
        return dict(characters=len(value),sha256=hashlib.sha256(value.encode()).hexdigest(),value_omitted=True)
    return deepcopy(value)

def rendering_updates(proposal, action):
    text = proposal.get('payload_text_rendering','')
    ledger = action=='ManageLedgerParameters'
    heading = '# Proposal to change ledger parameters:' if ledger else '# Proposal to upgrade sns metadata:'
    if not isinstance(text,str) or not text.startswith(heading+'\n'):
        return {},set()
    lines,parsed,ambiguous = text.splitlines(),{},set()
    for field,(prefix,pattern) in (_LEDGER if ledger else _METADATA).items():
        indices = [index for index,line in enumerate(lines) if line.startswith(prefix)]
        if not indices:
            continue
        if len(indices)!=1:
            ambiguous.add(field)
            continue
        index = indices[0]
        tail,raw = lines[index+1:],lines[index][len(prefix):]
        if pattern is None:
            if raw.strip() or any(line.startswith('# ') for line in tail):
                ambiguous.add(field)
                continue
            value = '\n'.join(tail).strip()
        else:
            match = re.fullmatch(pattern,raw)
            if match is None:
                ambiguous.add(field)
                continue
            value = match[1] if ledger else match[1].rstrip()
            if field=='description':
                boundary = next((i for i,line in enumerate(tail) if line.startswith('# ')),len(tail))
                if boundary:
                    value += '\n'+'\n'.join(tail[:boundary])
            if field=='transfer_fee':
                value = int(value)
                if not uint(value):
                    ambiguous.add(field)
                    continue
        parsed[field] = value
    return parsed,ambiguous

def changes(action, proposal, payload):
    allowed = set(_LEDGER if action=='ManageLedgerParameters' else _METADATA)
    rendered,ambiguous = rendering_updates(proposal,action)
    facts = []
    for field in sorted(set(payload)|set(rendered)|ambiguous):
        value = payload.get(field)
        structured = value is not None
        if not structured and field not in rendered and field not in ambiguous:
            continue
        if field not in allowed:
            row = dict(field=field,status='unsupported',review='review',previous=None,unknowns=['field_semantics'],
                       evidence=[dict(path='/proposal_action_payload/'+field.replace('~','~0').replace('/','~1'))],proposed=deepcopy(value))
            facts.append(row)
            continue
        row = dict(field=field,status='insufficient_data',review='review',previous=None,unknowns=['previous_value'],evidence=[])
        if structured:
            row['evidence'].append(dict(path='/proposal_action_payload/'+field))
        if field in rendered:
            row['evidence'].append(dict(path='/payload_text_rendering',method='known_renderer_parse',chain_state_verified=False,field=field))
        if field in ambiguous:
            status,unknown = 'ambiguous_rendering','unambiguous_rendering'
        elif structured and field in rendered and value!=rendered[field]:
            status,unknown = 'conflicting_sources','consistent_proposed_value'
        elif not structured:
            status,unknown = 'rendering_only','structured_proposed_value'
            value = rendered.get(field)
        else:
            status,unknown = ('consistent' if field in rendered else 'structured_only'),None
        row['source_status'] = status
        if status=='rendering_only':
            row['structured_field_missing_or_null'] = True
        if unknown:
            row['unknowns'].append(unknown)
        if not (uint(value) if field=='transfer_fee' else isinstance(value,str) and bool(value)):
            row['unknowns'].append('valid_proposed_value')
        encode = opaque_text if field in ('logo','token_logo') else deepcopy
        row['proposed'] = encode(value)
        if status=='conflicting_sources':
            row['rendered_proposed'] = encode(rendered[field])
        row['effect'] = 'requests_'+field+'_update'
        facts.append(row)
    return facts

def transfer(payload):
    treasury,amount = payload.get('from_treasury'),payload.get('amount_e8s')
    recipient,memo,account = payload.get('to_principal'),payload.get('memo'),payload.get('to_subaccount')
    known = type(treasury) is int and treasury in (1,2)
    positive = uint(amount) and amount>0
    identified = isinstance(recipient,str) and bool(recipient)
    memo_ok = memo is None or uint(memo)
    account_ok = account is None or (isinstance(account,str) and re.fullmatch('[0-9a-fA-F]{64}',account) is not None) or (isinstance(account,list) and len(account)==32 and all(type(v) is int and v in range(256) for v in account))
    unknowns = ['treasury_balance_before','recipient_control','recipient_relationship']
    unknowns.extend(name for valid,name in ((known,'known_treasury_type'),(positive,'valid_positive_amount'),
                                          (identified,'recipient'),(memo_ok,'valid_optional_memo'),(account_ok,'valid_optional_32_byte_subaccount')) if not valid)
    row = dict(field='treasury_transfer',status='assessed' if all((known,positive,identified,memo_ok,account_ok)) else 'insufficient_data',
               review='review',effect='treasury_transfer_requested',from_treasury=treasury,
               asset={1:'ICP',2:'SNS_TOKEN'}.get(treasury) if type(treasury) is int else None,
               amount_e8s=amount,recipient=recipient,memo=memo,subaccount=deepcopy(account),unknowns=unknowns,
               impact_assessment='insufficient_data',evidence=[dict(path='/proposal_action_payload')])
    if positive:
        row['amount_units'] = '{}.{:08d}'.format(*divmod(amount,100000000))
    return [row]

def generic(proposal, payload):
    function_id,raw = payload.get('function_id'),payload.get('payload')
    row = dict(field='generic_call',status='insufficient_data',review='review',function_id=function_id,effect='generic_call_requested',
               unknowns=['execution_semantics','target_code'],evidence=[dict(path='/proposal_action_payload')])
    definition = proposal.get('nervous_system_function')
    matches = isinstance(definition,dict) and uint(function_id) and function_id>=1000 and str(definition.get('id'))==str(function_id)
    if matches:
        types = definition.get('function_type',{})
        target = types.get('GenericNervousSystemFunction') if isinstance(types,dict) else None
        if isinstance(target,dict):
            row['declared_target'] = deepcopy(target)
            row['evidence'].append(dict(path='/nervous_system_function',chain_state_verified=False))
        else:
            row['unknowns'].append('generic_function_definition')
    else:
        row['unknowns'].append('matching_function_definition')
    if isinstance(raw,list) and all(type(byte) is int and byte in range(256) for byte in raw):
        row.update(payload_bytes=len(raw),payload_sha256=hashlib.sha256(bytes(raw)).hexdigest())
    else:
        row['unknowns'].append('raw_payload_bytes')
    rendering = proposal.get('payload_text_rendering','')
    if isinstance(rendering,str):
        hashes = re.findall(r'^## Payload sha256:\s*\n\s*([0-9a-f]{64})\s*$',rendering,re.M)
        if len(hashes)==1:
            row['rendered_payload_sha256'] = hashes[0]
            if row.get('payload_sha256')!=hashes[0]:
                row['unknowns'].append('payload_hash_matches_rendering')
                row['source_status'] = 'conflicting_sources'
        elif len(hashes)>1:
            row['unknowns'].append('unambiguous_rendered_payload_hash')
    return [row]

def assess_action(action, proposal, payload):
    if action in ('ManageLedgerParameters','ManageSnsMetadata'):
        return changes(action,proposal,payload)
    if action=='TransferSnsTreasuryFunds':
        rows,allowed = transfer(payload),{'from_treasury','amount_e8s','memo','to_principal','to_subaccount'}
    elif action=='ExecuteGenericNervousSystemFunction':
        rows,allowed = generic(proposal,payload),{'function_id','payload'}
    else:
        text = payload.get('motion_text')
        rows = [dict(field='motion',status='assessed' if isinstance(text,str) and text else 'insufficient_data',review='review',
                     effect='non_executing_motion',motion_text=deepcopy(text),unknowns=['future_followup_actions'],
                     evidence=[dict(path='/proposal_action_payload/motion_text')])]
        allowed = {'motion_text'}
    return rows+[_unsupported(name,value) for name,value in sorted(payload.items()) if name not in allowed and value is not None]
