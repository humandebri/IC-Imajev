"""Prompt/feature experiments with exact arithmetic and explicit order checks.

Expectations never enter prompts; all outputs remain advisory. This is tuning,
not independently established accuracy. No external LLM preprocessing.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys
from .binary_benchmark import IMAJEV, binary_decision, run
from .million_notation import format_million
from .token_sweep import ROOT, numeric_change, save, sha

VARIANTS = {
    'format': 'Should this proposal be approved? Reject changes that make voting inaccessible, even if another barrier is lowered.',
    'ratio': 'Which vote protects participation? Approve lower barriers; reject prohibitive stake or lock requirements. Consider all changes together.',
    'ratio_swapped': 'Which vote protects participation? Approve lower barriers; reject prohibitive stake or lock requirements. Consider all changes together.',
}
THRESHOLDS = [.5, .6, .7, .8, .9, .95]


def quantity(value):
    return format_million(value).replace('M', ' million')


def terminating_decimal(value):
    """Exact rational rendering; return None for nonterminating decimals."""
    denominator=value.denominator;twos=fives=0
    while denominator%2==0:denominator//=2;twos+=1
    while denominator%5==0:denominator//=5;fives+=1
    if denominator!=1:return None
    places=max(twos,fives)
    scaled=value.numerator*(10**places//value.denominator)
    digits=str(scaled).zfill(places+1)
    return (digits[:-places]+'.'+digits[-places:]).rstrip('0').rstrip('.') if places else digits


def encode_state(task, ratios=True):
    state = json.loads(task['state'])
    if state['action'] != 'ManageNervousSystemParameters':
        raise ValueError('participation policy only supports parameter changes')
    lines = []
    labels = {'min stake': 'Minimum stake', 'min voting lock': 'Minimum voting lock',
              'max lock': 'Maximum lock', 'max lock bonus': 'Maximum lock bonus'}
    for row in sorted(state['changes_or_requests'],key=lambda row:row['field']):
        field = row['field'];old,new=row.get('previous'),row.get('proposed')
        if type(old) is not int or type(new) is not int or old<0 or new<0:
            raise ValueError('known nonnegative integer old/new required')
        text = numeric_change(row)
        name,values = text.split(': ',1);before,after=values.split(' -> ',1)
        units = name.split()[-1];name=name.rsplit(' ',1)[0]
        if units=='seconds':
            days=[terminating_decimal(Fraction(x,86400)) for x in (old,new)]
            if all(x is not None for x in days):before,after=days;units='days'
        name=labels.get(name,name)
        lines.append(f'{name}: {quantity(before)} -> {quantity(after)} {units}')
        if ratios and field in ('neuron_minimum_stake_e8s','neuron_minimum_dissolve_delay_to_vote_seconds') and old!=new:
            if min(old,new)>0:
                ratio=Fraction(max(old,new),min(old,new))
                decimal=terminating_decimal(ratio)
                factor=quantity(decimal) if decimal is not None else f'{ratio.numerator}/{ratio.denominator}'
                lines.append(f'{name} requirement {"increases" if new>old else "decreases"} {factor}-fold')
            else:
                lines.append(f'{name} requirement {"increases from zero" if old==0 else "decreases to zero"}')
    return '; '.join(lines)


def compact_exact_state(task):
    """Keep every old/new number; compact labels and one redundant ratio fact."""
    rows=json.loads(task['state'])['changes_or_requests']
    lines=encode_state(task).split('; ')
    # Retain the strongest calculated increase (otherwise strongest decrease).
    # Other ratios can always be reconstructed from the retained raw values.
    eligible=[r for r in rows if r['field'] in ('neuron_minimum_stake_e8s',
              'neuron_minimum_dissolve_delay_to_vote_seconds') and r['previous']!=r['proposed']
              and min(r['previous'],r['proposed'])>0]
    if eligible:
        increases=[r for r in eligible if r['proposed']>r['previous']]
        best=max(increases or eligible,key=lambda r:Fraction(max(r['previous'],r['proposed']),min(r['previous'],r['proposed'])))
        prefix='Minimum stake' if best['field']=='neuron_minimum_stake_e8s' else 'Minimum voting lock'
        lines=[line for line in lines if ' requirement ' not in line or line.startswith(prefix+' requirement ')]
    text='; '.join(lines)
    for old,new in [('Minimum stake','stake'),('Minimum voting lock','min voting lock'),
                    ('Maximum lock bonus','lock bonus'),('Maximum lock','max lock'),
                    ('max age bonus','age bonus'),('max bonus age','bonus age'),
                    (': ',' '),(' -> ','→'),(' percent','%')]:text=text.replace(old,new)
    return text


def approval_evidence_gate(task, recommendation):
    """An approval requires directly demonstrated eligibility improvement.

    No hard rejection thresholds. Conflicting/unknown directions are reviewed;
    this gate never converts hold/approve into reject and reads no labels/IDs.
    """
    if recommendation!='approve':return recommendation,None
    rows=json.loads(task['state'])['changes_or_requests']
    eligibility=[r for r in rows if r['field'] in ('neuron_minimum_stake_e8s',
                  'neuron_minimum_dissolve_delay_to_vote_seconds')]
    if not eligibility or any(type(r.get('previous')) is not int or type(r.get('proposed')) is not int for r in eligibility):
        return 'hold','Direct eligibility improvement is not established.'
    if any(r['proposed']>r['previous'] for r in eligibility):
        return 'hold','An eligibility requirement becomes stricter; net benefit needs further evidence.'
    if not any(r['proposed']<r['previous'] for r in eligibility):
        return 'hold','No directly demonstrated eligibility improvement.'
    return recommendation,None


def prepare(out):
    if out.exists():raise ValueError('fresh directory required')
    oldroot=ROOT/'artifacts/proposal-assessment/heldout-128-20261006'
    source=json.loads((oldroot/'prepared.json').read_text())['entries']
    # All three were previously evaluated: explicitly a tuning screen.
    selected=[source[i] for i in (0,3,4)]
    sys.path.insert(0,str(IMAJEV/'scripts'));from prepare_text import TextPreparer
    p=TextPreparer();records=[];entries=[]
    for variant,question in VARIANTS.items():
        for e in selected:
            state=encode_state(e['task'],ratios=variant!='format')
            options=['reject','approve'] if variant.endswith('swapped') else ['approve','reject']
            prompt=f'State: {state}\nQuestion: {question}\n'+'\n'.join(f'{code}: {label}' for code,label in zip('AB',options))
            ids=p.tokenizer.encode(p.render(prompt),add_special_tokens=False)
            assert len(ids)<=128
            records.append(dict(id=f'improve_{len(records)}',options=options,gold=None,token_ids=ids,
                                input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),prompt=prompt))
            entries.append(dict(record_index=len(records)-1,proposal_ids=e['proposal_ids'],variant=variant,
                                tokens=len(ids),expected=e['reference_prediction'],synthetic=e.get('synthetic',False),
                                task=e['task'],state=state))
    out.mkdir(parents=True)
    save(out/'inputs.json',dict(model_lock_sha256=sha(IMAJEV/'MODEL_LOCK.json'),records=records))
    save(out/'prepared.json',dict(entries=entries,thresholds=THRESHOLDS,scope='tuning screen; not heldout accuracy',
                                 variants=VARIANTS,expectations_in_model_input=False))
    origin=ROOT/'artifacts/proposal-assessment/binary-44-20261006'
    shutil.copyfile(origin/'runner.py',out/'runner.py')
    shutil.copyfile(__file__,out/'prepare_source.py')
    shutil.copyfile(Path(__file__).parent/'million_notation.py',out/'numeric_source.py')
    save(out/'identity.json',dict(inputs_sha256=sha(out/'inputs.json'),prepared_sha256=sha(out/'prepared.json'),
         runner_sha256=sha(out/'runner.py'),bridge_sha256=sha(IMAJEV/'target/release/imajev-client')))
    print(json.dumps([(e['variant'],e['proposal_ids'],e['tokens']) for e in entries]))


def results(out):
    prepared=json.loads((out/'prepared.json').read_text());inputs=json.loads((out/'inputs.json').read_text())
    identity=json.loads((out/'identity.json').read_text())
    assert sha(out/'prepared.json')==identity['prepared_sha256'] and sha(out/'inputs.json')==identity['inputs_sha256']
    rows=[]
    for e in prepared['entries']:
        row=dict(e,status='pending');i=e['record_index'];path=out/'runs'/f'{i:03d}'/'report.json'
        if path.exists():
            report=json.loads(path.read_text());record=inputs['records'][i]
            assert report['wasm_sha256']==report['deployed_wasm_sha256']=='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
            assert report['input_hash']==record['input_sha256'] and report['model']==inputs['model_lock_sha256']
            assert [x['layer'] for x in report['layers']]==list(range(32)) and report['tokens']==e['tokens']
            assert not report['replayed_queries'] and not report.get('fallback') and report['executed_query_count']==report['query_count']
            assert not (path.parent/'active-staging.json').exists()
            logits=report['decision_query']['ok']['decision']['raw_logits'];assert len(logits)==3
            # Convert presentation order to canonical approve/reject scores.
            scores=dict(zip(record['options'],logits[:2]));canonical=[scores['approve'],scores['reject']]
            label,score=binary_decision(canonical,.5)
            row.update(status='evaluated',prediction=label,score=score,canonical_logits=canonical,
                       expectation_match=label==e['expected'],report_sha256=sha(path),
                       decisions={str(t):binary_decision(canonical,t)[0] for t in prepared['thresholds']})
        elif (path.parent/'error.json').exists():row['status']='execution_error'
        rows.append(row)
    save(out/'results.json',dict(complete=all(r['status']=='evaluated' for r in rows),rows=rows,
                               accuracy_measured=False,scope=prepared['scope']))
    print(json.dumps([(r['variant'],r['proposal_ids'],r.get('prediction',r['status']),r.get('score')) for r in rows]))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','report']);p.add_argument('--output',type=Path,required=True);p.add_argument('--shards',type=int,default=1);p.add_argument('--shard',type=int,default=0);a=p.parse_args();out=a.output.resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='run':run(out,a.shard,a.shards)
    else:results(out)


if __name__=='__main__':main()
