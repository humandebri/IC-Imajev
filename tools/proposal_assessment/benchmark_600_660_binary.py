"""Prepare the retained 61-snapshot archive using Imajev's current evidence tools."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from .binary_benchmark import IMAJEV, run
from .budget_policy import evidence_gate
from .core import assess
from .vote import make_vote_task
from .routing import select_task
from .improve_binary import VARIANTS, encode_state, compact_exact_state, results, approval_evidence_gate
from .compact_units import encode, QUESTION
from .validate_binary_improvement import THRESHOLD, COMPACT_QUESTION
from .token_sweep import ROOT, sha, save
sys.path.insert(0,str(ROOT/'scripts'))
from proposal_snapshots import read_snapshot
from repository_paths import existing_directory

def prepare(out, compact_overflow=False, compact_ratio=False, snapshot_archive=None):
    if out.exists():
        raise ValueError('fresh output directory required')
    archive = Path(snapshot_archive) if snapshot_archive is not None else ROOT/'artifacts/proposal-assessment/boomdao-600-660/v1'
    manifest = json.loads((archive/'manifest.json').read_text())
    reference_path = ROOT/'tools/proposal_assessment/gpt-6.1-sol-medium-20261005-1509/vote-results.json'
    reference = json.loads(reference_path.read_text())
    if not manifest['complete'] or not reference['complete']:
        raise ValueError('incomplete source archive')
    records = {row['proposal_id']:row for row in manifest['records']}
    predictions = {row['proposal_id']:row for row in reference['proposals']}
    if len(manifest['records'])!=61 or len(reference['proposals'])!=61 or set(records)!=set(range(600,661)) or set(predictions)!=set(records):
        raise ValueError('complete unique 600-660 range required')
    verified = []
    # All selected snapshots are verified before loading a tokenizer or writing output.
    for proposal_id in sorted(records):
        record = records[proposal_id]
        _,proposal = read_snapshot(archive,record,manifest['sns_root'])
        if record['sha256']!=predictions[proposal_id]['snapshot_sha256']:
            raise ValueError('reference snapshot identity mismatch')
        verified.append((proposal_id,proposal))
    sys.path.insert(0,str(IMAJEV/'scripts'))
    from prepare_text import TextPreparer
    from vision_decision.scoring import verified_label_ids
    preparer = TextPreparer()
    grouped = {}
    for proposal_id,proposal in verified:
        task = make_vote_task(assess(proposal))
        key = json.dumps(task,sort_keys=True)
        entry = grouped.setdefault(key,dict(task=task,proposal_ids=[],expected=predictions[proposal_id]['prediction']['label'],
                                           split='previously_seen_real_600_660',synthetic=False))
        if entry['expected']!=predictions[proposal_id]['prediction']['label']:
            raise ValueError('conflicting references for identical tasks')
        entry['proposal_ids'].append(proposal_id)
    inputs,entries,gates = [],[],[]
    for entry in grouped.values():
        task = entry['task']
        selection = select_task(task)
        if not selection['requires_model']:
            gates.append(dict(entry,label=None,tokens=0,route='skipped',reason=selection['reason']))
            continue
        gate = evidence_gate(task)
        if gate['route']!='model':
            gates.append(dict(entry,**gate,tokens=0))
            continue
        state = json.loads(task['state'])
        if state['action']!='ManageNervousSystemParameters' or not any(r.get('field') in ('neuron_minimum_stake_e8s','neuron_minimum_dissolve_delay_to_vote_seconds') for r in state['changes_or_requests']):
            gates.append(dict(entry,label='hold',tokens=0,route='outside_direct_eligibility_scope'))
            continue
        try:
            rendered = encode_state(task)
            question,layout = VARIANTS['ratio'],'ratio'
            prompt = f'State: {rendered}\nQuestion: {question}\nA: approve\nB: reject'
            ids = preparer.tokenizer.encode(preparer.render(prompt),add_special_tokens=False)
            if len(ids)>128:
                rendered,question,layout = compact_exact_state(task),COMPACT_QUESTION,'compact_ratio'
                prompt = f'State: {rendered}\nQuestion: {question}\nA: approve\nB: reject'
                ids = preparer.tokenizer.encode(preparer.render(prompt),add_special_tokens=False)
            if len(ids)>128 and compact_overflow:
                try:
                    rendered,preserved = encode(task,ratio_words=compact_ratio)
                    entry['preserved_facts'] = preserved
                    layout = 'unit_grouped_ratio_words' if compact_ratio else 'unit_grouped_overflow'
                    options = 'A approve B reject' if compact_ratio else 'A: approve\nB: reject'
                    prompt = f'{rendered}\n{QUESTION}\n{options}'
                    ids = preparer.tokenizer.encode(preparer.render(prompt),add_special_tokens=False)
                except ValueError:
                    pass
        except (ValueError,KeyError) as error:
            gates.append(dict(entry,label='hold',tokens=0,route='unsupported_encoding',reason=str(error)))
            continue
        if len(ids)>128:
            gates.append(dict(entry,label='hold',tokens=0,full_input_tokens=len(ids),route='budget_overflow',candidate_prompt=prompt))
            continue
        if verified_label_ids(preparer.tokenizer,preparer.render(prompt),['A','B'])!=[preparer.binding['codes'][i]['token_id'] for i in range(2)]:
            raise ValueError('tokenizer label binding mismatch')
        entries.append(dict(entry,record_index=len(inputs),variant=layout,state=rendered,tokens=len(ids)))
        inputs.append(dict(id=f'boom600_{len(inputs)}',options=['approve','reject'],gold=None,token_ids=ids,
                           input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),prompt=prompt))
    out.mkdir(parents=True)
    package = Path(__file__).parent
    sources = list(package.glob('*.py'))+[package/'prompt_contract.json',package/'PROVENANCE.json']
    save(out/'source-hashes.json',{str(path):sha(path) for path in sources})
    save(out/'inputs.json',dict(model_lock_sha256=sha(IMAJEV/'MODEL_LOCK.json'),records=inputs))
    save(out/'prepared.json',dict(entries=entries,gates=gates,thresholds=[THRESHOLD],scope='Frozen 600-660 numerical participation; no human gold',
                                  approval_gate_enabled=True,local_rejection_guard_enabled=False,accuracy_measured=False,
                                  compact_overflow_enabled=compact_overflow,compact_ratio_enabled=compact_ratio,reference_is_gold=False,
                                  source_reference_sha256=sha(reference_path),source_manifest_sha256=sha(archive/'manifest.json')))
    runner = ROOT/'artifacts/proposal-assessment/binary-improved-final-20261006/runner.py'
    shutil.copyfile(runner,out/'runner.py')
    save(out/'identity.json',dict(inputs_sha256=sha(out/'inputs.json'),prepared_sha256=sha(out/'prepared.json'),
                                  runner_sha256=sha(out/'runner.py'),bridge_sha256=sha(IMAJEV/'target/release/imajev-client')))

def report(out):
    prepared = json.loads((out/'prepared.json').read_text())
    rows = results(out)
    for row in rows:
        if row['status']=='evaluated':
            row['final_label'],row['approval_gate_reason'] = approval_evidence_gate(row['task'],row['decisions'][str(THRESHOLD)])
    save(out/'quality.json',dict(complete=all(r['status']=='evaluated' for r in rows),model_rows=rows,gates=prepared['gates'],accuracy_measured=False))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compact-overflow',action='store_true')
    parser.add_argument('--compact-ratio',action='store_true')
    parser.add_argument('--snapshot-archive',type=existing_directory)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.mode=='prepare':
        prepare(out,args.compact_overflow or args.compact_ratio,args.compact_ratio,snapshot_archive=args.snapshot_archive)
    elif args.mode=='run':
        run(out,0,1)
    else:
        report(out)

if __name__=='__main__':
    main()
