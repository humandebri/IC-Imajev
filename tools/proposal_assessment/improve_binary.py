"""Exact input encoding and approval evidence checks for Imajev assessments."""
from fractions import Fraction
import json
from .million_notation import format_million
from .token_sweep import numeric_change, save, sha
from .binary_benchmark import binary_decision

VARIANTS = {
    'ratio': 'Which vote protects participation? Approve lower barriers; reject prohibitive stake or lock requirements. Consider all changes together.',
}
ELIGIBILITY = ('neuron_minimum_stake_e8s','neuron_minimum_dissolve_delay_to_vote_seconds')
LABELS = {'min stake':'Minimum stake', 'min voting lock':'Minimum voting lock',
          'max lock':'Maximum lock', 'max lock bonus':'Maximum lock bonus'}

def terminating_decimal(value):
    remainder = value.denominator
    for prime in (2,5):
        while remainder % prime == 0:
            remainder //= prime
    if remainder != 1:
        return None
    sign = '-' if value < 0 else ''
    integer, remainder = divmod(abs(value.numerator),value.denominator)
    digits = []
    while remainder:
        digit,remainder = divmod(remainder*10,value.denominator)
        digits.append(str(digit))
    return sign + str(integer) + ('.'+''.join(digits) if digits else '')

def quantity(value):
    return format_million(value).replace('M',' million')

def _lines(task, ratios):
    state = json.loads(task['state'])
    if state['action'] != 'ManageNervousSystemParameters':
        raise ValueError('parameter changes only')
    result = []
    for row in sorted(state['changes_or_requests'],key=lambda r:r['field']):
        pair = row.get('previous'),row.get('proposed')
        if any(type(v) is not int or v < 0 for v in pair):
            raise ValueError('known nonnegative integer old/new required')
        title,values = numeric_change(row).split(': ',1)
        title,unit = title.rsplit(' ',1)
        title = LABELS.get(title,title)
        displayed = values.split(' -> ')
        if unit == 'seconds':
            days = [terminating_decimal(Fraction(v,86400)) for v in pair]
            if None not in days:
                displayed,unit = days,'days'
        result.append((row['field'],False,f'{title}: {quantity(displayed[0])} -> {quantity(displayed[1])} {unit}'))
        if not ratios or row['field'] not in ELIGIBILITY or pair[0] == pair[1]:
            continue
        if min(pair) == 0:
            detail = 'increases from zero' if pair[0] == 0 else 'decreases to zero'
        else:
            factor = Fraction(max(pair),min(pair))
            decimal = terminating_decimal(factor)
            multiplier = quantity(decimal) if decimal is not None else f'{factor.numerator}/{factor.denominator}'
            detail = ('increases' if pair[1]>pair[0] else 'decreases') + ' ' + multiplier + '-fold'
        result.append((row['field'],True,title + ' requirement ' + detail))
    return result

def encode_state(task, ratios=True):
    return '; '.join(text for _,_,text in _lines(task,ratios))

def compact_exact_state(task):
    rows = json.loads(task['state'])['changes_or_requests']
    eligible = [r for r in rows if r['field'] in ELIGIBILITY and r['previous']!=r['proposed'] and min(r['previous'],r['proposed'])>0]
    preferred = [r for r in eligible if r['proposed'] > r['previous']] or eligible
    selected = max(preferred,key=lambda r:Fraction(max(r['previous'],r['proposed']),min(r['previous'],r['proposed'])))['field'] if preferred else None
    lines = [text for field,ratio,text in _lines(task,True) if not ratio or selected is None or field==selected]
    substitutions = {'Minimum stake':'stake','Minimum voting lock':'min voting lock','Maximum lock bonus':'lock bonus',
                     'Maximum lock':'max lock','max age bonus':'age bonus','max bonus age':'bonus age',
                     ': ':' ',' -> ':'→',' percent':'%'}
    text = '; '.join(lines)
    for original,replacement in substitutions.items():
        text = text.replace(original,replacement)
    return text

def approval_evidence_gate(task, recommendation):
    if recommendation != 'approve':
        return recommendation,None
    rows = [r for r in json.loads(task['state'])['changes_or_requests'] if r['field'] in ELIGIBILITY]
    if not rows or any(type(r.get(k)) is not int for r in rows for k in ('previous','proposed')):
        return 'hold','Direct eligibility improvement is not established.'
    if any(r['proposed']>r['previous'] for r in rows):
        return 'hold','An eligibility requirement becomes stricter; net benefit needs further evidence.'
    if any(r['proposed']<r['previous'] for r in rows):
        return recommendation,None
    return 'hold','No directly demonstrated eligibility improvement.'

def results(out):
    prepared,inputs,identity = [json.loads((out/name).read_text()) for name in ('prepared.json','inputs.json','identity.json')]
    for name in ('prepared','inputs'):
        if sha(out/f'{name}.json') != identity[f'{name}_sha256']:
            raise ValueError('frozen input hash mismatch')
    rows = []
    for entry in prepared['entries']:
        row = dict(entry,status='pending')
        path = out/'runs'/f"{entry['record_index']:03d}"/'report.json'
        if path.exists():
            report = json.loads(path.read_text())
            record = inputs['records'][entry['record_index']]
            expected_module = '6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
            if not (report['wasm_sha256']==report['deployed_wasm_sha256']==expected_module and
                    report['input_hash']==record['input_sha256'] and report['model']==inputs['model_lock_sha256'] and
                    [layer['layer'] for layer in report['layers']]==list(range(32)) and report['tokens']==entry['tokens'] and
                    not report['replayed_queries'] and not report.get('fallback') and
                    report['executed_query_count']==report['query_count'] and not (path.parent/'active-staging.json').exists()):
                raise ValueError('historical inference identity or execution mismatch')
            logits = report['decision_query']['ok']['decision']['raw_logits']
            if len(logits)!=3 or set(record['options'])!={'approve','reject'}:
                raise ValueError('binary readout contract mismatch')
            scores = dict(zip(record['options'],logits[:2]))
            canonical = [scores['approve'],scores['reject']]
            prediction,score = binary_decision(canonical,.5)
            row.update(status='evaluated',prediction=prediction,score=score,canonical_logits=canonical,
                       expectation_match=prediction==entry['expected'],report_sha256=sha(path),
                       decisions={str(t):binary_decision(canonical,t)[0] for t in prepared['thresholds']})
        elif (path.parent/'error.json').exists():
            row['status'] = 'execution_error'
        rows.append(row)
    save(out/'results.json',dict(complete=all(r['status']=='evaluated' for r in rows),rows=rows,
                               accuracy_measured=False,scope=prepared['scope']))
    return rows
