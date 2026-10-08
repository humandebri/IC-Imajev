"""Snapshot facts built from typed SNS fields and explicit review thresholds.

Historical prompt strings and field metadata live in prompt_contract.json; they
are retained data, not executable classifier or runtime code.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from decimal import Decimal

CONTRACT = json.loads((Path(__file__).parent/'prompt_contract.json').read_text())
FIELDS = CONTRACT['fields']
BOOL_FIELDS = set(CONTRACT['boolean_fields'])
LEVELS = {'no_escalation':0,'review':1,'critical_review':2}

def uint(value):
    return type(value) is int and value in range(2**64)

def old_parameter(proposal, field, *, boolean=False):
    rendering = proposal.get('payload_text_rendering')
    heading = '## Current nervous system parameters:'
    if not isinstance(rendering,str) or rendering.count(heading)!=1:
        return None,None
    section = rendering.partition(heading)[2].partition('\n## ')[0]
    starts = list(re.finditer(r'^\s*'+re.escape(field)+r'\s*:',section,re.M))
    if len(starts)!=1:
        return None,None
    expression = r'\s*Some\(\s*'+('(true|false)' if boolean else r'(\d+)')+r'\s*,?\s*\)'
    match = re.match(expression,section[starts[0].end():])
    if match is None:
        return None,None
    value = match[1]=='true' if boolean else int(match[1])
    if not boolean and not uint(value):
        return None,None
    quote = section[starts[0].start():starts[0].end()+match.end()].strip()
    return value,dict(path='/payload_text_rendering',quote=quote,method='historical_rendering_parse',chain_state_verified=False)

def _parameter(proposal, field, proposed, review_factor, critical_factor):
    previous,source = old_parameter(proposal,field,boolean=field in BOOL_FIELDS)
    row = dict(field=field,proposed=deepcopy(proposed),previous=previous,
               evidence=[dict(path='/proposal_action_payload/'+field.replace('~','~0').replace('/','~1'))],
               unknowns=[],review='review')
    if source:
        row['evidence'].append(source)
    if field in BOOL_FIELDS:
        row.update(unit='boolean',description=field)
        if type(proposed) is not bool or type(previous) is not bool:
            row.update(status='insufficient_data',unknowns=['previous_value' if type(previous) is not bool else 'valid_proposed_boolean'])
        elif proposed==previous:
            row.update(status='assessed',direction='unchanged',effect='unchanged',review='no_escalation')
        else:
            row.update(status='assessed',direction='toggle',effect='sets_'+field+'_'+str(proposed).lower(),unknowns=['actual_system_impact'])
    elif field not in FIELDS:
        if uint(proposed) and previous is not None and proposed==previous:
            row.update(status='assessed',direction='unchanged',delta=0,effect='unchanged',review='no_escalation')
        else:
            row.update(status='unsupported',unknowns=['field_semantics'])
    else:
        unit,description,increase,decrease,unknown = FIELDS[field]
        row.update(unit=unit,description=description)
        if not uint(proposed) or previous is None:
            row.update(status='insufficient_data',unknowns=['previous_value' if previous is None else 'valid_proposed_uint64'])
        else:
            delta = proposed-previous
            direction = 'unchanged' if delta==0 else 'increase' if delta>0 else 'decrease'
            row.update(status='assessed',direction=direction,delta=delta,effect='unchanged' if delta==0 else increase if delta>0 else decrease)
            if not delta:
                row['review'] = 'no_escalation'
            else:
                row['unknowns'] = [unknown]
                low,high = sorted((previous,proposed))
                row['magnitude_ratio'] = dict(numerator=high,denominator=low) if low else None
                if low:
                    row['review'] = next(level for factor,level in ((critical_factor,'critical_review'),(review_factor,'review'),(1,'no_escalation')) if high>=low*factor)
    return row

def _mint(payload):
    amount,recipient = payload.get('amount_e8s'),payload.get('to_principal')
    row = dict(field='token_mint',status='insufficient_data',review='review',amount_e8s=amount,recipient=deepcopy(recipient),
               memo=deepcopy(payload.get('memo')),subaccount=deepcopy(payload.get('to_subaccount')),
               unknowns=['supply_before','recipient_holdings_before','recipient_control'],effect='mint_requested',
               concentration_assessment='insufficient_data',evidence=[dict(path='/proposal_action_payload')])
    if uint(amount) and amount>0:
        row['amount_tokens'] = format(Decimal(amount)/100000000,'f')
    else:
        row['unknowns'].append('valid_positive_mint_amount')
    if not isinstance(recipient,str) or not recipient:
        row['unknowns'].append('recipient')
    return [row]+[_unsupported(field,value) for field,value in sorted(payload.items())
                  if field not in {'amount_e8s','to_principal','to_subaccount','memo'} and value is not None]

def _unsupported(field,value):
    return dict(field=field,status='unsupported',review='review',proposed=deepcopy(value),unknowns=['field_semantics'],
                evidence=[dict(path='/proposal_action_payload/'+field.replace('~','~0').replace('/','~1'))])

def assess(proposal, *, review_factor=10, critical_factor=1000):
    if not isinstance(proposal,dict):
        raise ValueError('proposal must be an object')
    if type(review_factor) is not int or type(critical_factor) is not int or not 2<=review_factor<=critical_factor:
        raise ValueError('require 2 <= review_factor <= critical_factor (integers)')
    canonical = json.dumps(proposal,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    action,payload = proposal.get('proposal_action_type'),proposal.get('proposal_action_payload')
    unspecified = []
    supported = {'ManageNervousSystemParameters','MintSnsTokens','ManageLedgerParameters','ManageSnsMetadata',
                 'TransferSnsTreasuryFunds','Motion','ExecuteGenericNervousSystemFunction'}
    if not isinstance(payload,dict):
        rows = [dict(field='payload',status='insufficient_data',proposed=deepcopy(payload),review='review',unknowns=['structured_payload'],evidence=[])]
    elif action=='ManageNervousSystemParameters':
        unspecified = sorted(name for name,value in payload.items() if value is None)
        rows = [_parameter(proposal,name,value,review_factor,critical_factor) for name,value in sorted(payload.items()) if value is not None]
    elif action=='MintSnsTokens':
        rows = _mint(payload)
    elif action in supported:
        from .extensions import assess_action
        rows = assess_action(action,proposal,payload)
    else:
        rows = [dict(field='action',status='unsupported',review='review',proposed=deepcopy(payload),unknowns=['action_semantics'],
                     evidence=[dict(path='/proposal_action_type'),dict(path='/proposal_action_payload')])]
    if not rows:
        rows = [dict(field='payload',status='insufficient_data',review='review',unknowns=['proposed_changes'],evidence=[])]
    return dict(schema_version=1,proposal_id=proposal.get('id'),action=action,rules_revision=2,
                canonical_proposal_sha256=hashlib.sha256(canonical).hexdigest(),scope='proposal_snapshot_only',execution_authorization=False,
                policy=dict(review_factor=review_factor,critical_factor=critical_factor,meaning='explicit review policy; no_escalation is not a safety judgment'),
                items=rows,unspecified_fields=unspecified,text={name:proposal[name] if isinstance(proposal.get(name),str) else '' for name in ('proposal_title','summary')},
                review_priority=max((r['review'] for r in rows),key=LEVELS.get),action_supported=action in supported,
                all_items_assessed=all(r['status']=='assessed' for r in rows),advisory=[])

def make_tasks(report):
    sources = [dict(path='/'+name,text=text) for name,text in report['text'].items() if text.strip()]
    tasks = deepcopy(CONTRACT['text_tasks'])
    for task in tasks:
        task['sources'] = sources
        task['state'] = '\n'.join(s['path']+': '+s['text'] for s in sources)
    return tasks

def validate_prediction(response, options):
    if not isinstance(response,dict) or response.get('label') not in options:
        raise ValueError('invalid model response label')
    logits = response.get('logits')
    if logits is not None and (not isinstance(logits,list) or len(logits)!=len(options) or any(type(v) not in (int,float) or not math.isfinite(v) for v in logits)):
        raise ValueError('invalid logits')
    return response

def run_advisory(report, adapter):
    result = deepcopy(report)
    result['model'] = adapter.metadata()
    for task in make_tasks(report):
        row = {key:deepcopy(task[key]) for key in ('task_id','question','options','option_descriptions')}
        if not task['sources']:
            row.update(status='insufficient_data',reason='proposal title/summary are unavailable')
        else:
            try:
                response = validate_prediction(adapter.predict(deepcopy(task)),task['options'])
                evidence = response.get('evidence',[])
                if not isinstance(evidence,list):
                    raise ValueError('evidence must be a list')
                texts = {source['path']:source['text'] for source in task['sources']}
                verified = []
                for span in evidence:
                    if not isinstance(span,dict):
                        raise ValueError('invalid evidence span')
                    text,start,end = texts.get(span.get('source')),span.get('start'),span.get('end')
                    if text is None or type(start) is not int or type(end) is not int or not 0<=start<end<=len(text):
                        raise ValueError('invalid evidence offsets')
                    verified.append(dict(source=span['source'],start=start,end=end,quote=text[start:end]))
                row.update(status='model_prediction',label=response['label'],evidence=verified,evidence_offsets_validated=bool(verified),correctness_verified=False)
                for source,target in (('logits','raw_logits'),('input_tokens','input_tokens')):
                    if response.get(source) is not None:
                        row[target] = response[source]
            except Exception as error:
                row.update(status='unavailable',reason=str(error)[:500])
        result['advisory'].append(row)
    return result
