#!/usr/bin/env python3
"""Run frozen proposal-assessment tasks on the existing local Imajev canister.

Exact task duplicates share a prediction. No input truncation or model updates.
Reserved model unknown is unavailable, never silently mapped to unclear/hold.
"""
import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT
sys.path.insert(0, str(ROOT / 'tools'))
from proposal_assessment.core import assess, make_tasks, run_advisory
from proposal_assessment.vote import make_vote_task, predict_vote
from proposal_assessment.evaluation import evaluate
from prepare_text import TextPreparer
from evaluate_prompt_accuracy import base_flags, MODULE
sys.path.insert(0, str(ROOT / 'client'))
from decision_validation import validate_decision
from proposal_snapshots import read_snapshot

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def task_key(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def write(path, data):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    manifest_path = SOURCE / 'artifacts/proposal-assessment/boomdao-600-660/v1/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['complete'] and manifest['all_fetched']
    assert sorted(r['proposal_id'] for r in manifest['records']) == list(range(600, 661))
    preparer, tasks, proposals, records = TextPreparer(), {}, [], []
    for rec in sorted(manifest['records'], key=lambda r: r['proposal_id']):
        path, proposal = read_snapshot(manifest_path.parent, rec, manifest['sns_root'])
        facts = assess(proposal)
        text_tasks, vote = make_tasks(facts), make_vote_task(facts)
        proposals.append(dict(proposal_id=rec['proposal_id'], snapshot_path=str(path), snapshot_sha256=rec['sha256'],
                              facts=facts, model_tasks=text_tasks, vote_task=vote))
        for task in text_tasks + [vote]:
            key = task_key(task)
            if key in tasks:
                tasks[key]['proposal_ids'].append(rec['proposal_id'])
                continue
            case = dict(id='pa_' + key[:16],
                        question=task['question'] + (' ' + task['instruction'] if task.get('instruction') else ''),
                        state=json.loads(task['state']) if task['task_id'] == 'approval_recommendation' else {'text': task['state']},
                        options=[dict(value=o, description=task['option_descriptions'][o]) for o in task['options']])
            entry = dict(task=task, case=case, proposal_ids=[rec['proposal_id']])
            try:
                record = preparer.prepare(case)
                assert record['prefix_tokens'] == 26
                record.update(task_key=key, task_id=task['task_id'])
                entry.update(record_index=len(records), tokens=len(record['token_ids']), status='pending')
                records.append(record)
            except ValueError as error:
                if 'text prefill requires 1..512 tokens' not in str(error):
                    raise
                entry.update(status='unavailable', reason=str(error) + '; no truncation')
            tasks[key] = entry
    paths = [ROOT / 'scripts/proposal_snapshots.py', ROOT / 'MODEL_LOCK.json', ROOT / 'scripts/prepare_text.py', pathlib.Path(__file__),
             ROOT / 'scripts/run_prefix_canister.py', ROOT / 'scripts/evaluate_prompt_accuracy.py',
             ROOT / 'artifacts/query-packing-v3/build/full.wasm', ROOT / 'artifacts/query-packing-v3/build/imajev-client',
             ROOT / 'artifacts/decision-index-v1/dense-prefix/queries/cache.json',
             ROOT / 'artifacts/decision-index-v1/prefix/queries/cache.json', ROOT / 'artifacts/decision-index-v1/packets/cache.json']
    paths += sorted((ROOT / 'client').glob('*.py'))
    paths += [ROOT / 'tools/proposal_assessment' / n for n in ('core.py','extensions.py','adapters.py','evaluation.py','vote.py')]
    assert sha(ROOT / 'artifacts/query-packing-v3/build/full.wasm') == MODULE
    fixture = dict(model_lock_sha256=sha(ROOT / 'MODEL_LOCK.json'), records=records)
    for name in ('prefix', 'dense-prefix'):
        cache = json.loads((ROOT / 'artifacts/decision-index-v1' / name / 'queries/cache.json').read_text())
        assert all(r['token_ids'][:26] == cache['token_ids'] for r in records)
    write(directory / 'inputs.json', fixture)
    write(directory / 'prepared.json', dict(proposals=proposals, tasks=tasks))
    write(directory / 'session.json', dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          canister='6eydd-o3777-77775-aaama-cai', url='http://localhost:8001/', wasm_sha256=MODULE,
          model_lock_sha256=fixture['model_lock_sha256'], manifest_sha256=sha(manifest_path),
          planned_unique_tasks=len(tasks), runnable_unique_tasks=len(records), expanded_tasks=244,
          input_sha256=sha(directory / 'inputs.json'), source_hashes={str(p): sha(p) for p in paths},
          adaptation='Original question, instruction, options and descriptions retained. Text state is wrapped in a text object; vote JSON state decoded to an object. Standard Imajev short text prompt, reserved unknown candidate, 26-token prefix reused. No truncation.',
          artifact_retention='Reports, logs, metrics and hashes retained; successful-run intermediate binary states removed after hashing.'))
    print(json.dumps(dict(prepared=len(tasks), runnable=len(records), unavailable=len(tasks)-len(records),
                         min_tokens=min(len(r['token_ids']) for r in records), max_tokens=max(len(r['token_ids']) for r in records))), flush=True)

def command(directory, index, record):
    n = len(record['token_ids']) - 26
    cmd = base_flags()
    cmd[cmd.index('--reference') + 1] = str(directory / 'inputs.json')
    cmd[cmd.index('--cache') + 1] = str(ROOT / 'artifacts/decision-index-v1' / ('prefix/queries' if n <= 90 else 'dense-prefix/queries'))
    if n <= 90:
        cmd += ['--hybrid-cache', 'artifacts/decision-index-v1/packets', '--terminal-readout', '--fuse-terminal-attention',
                '--fuse-terminal-decision', '--fuse-terminal-tail', '--fuse-prefix-start', '--tail-start', '--join-start', '--roll-start', '--packed-start']
    else:
        cmd.remove('--fuse-delta-full-log')
        if n <= 132:
            cmd += ['--terminal-readout', '--fuse-terminal-attention', '--fuse-terminal-decision']
        else:
            for flag in ('--fuse-delta-projected','--fuse-delta-finish','--fuse-attention','--fuse-attention-full'):
                cmd.remove(flag)
            cmd += ['--terminal-readout']
    cmd += ['--record', str(index), '--directory', str(directory / 'runs' / f'{index:03d}')]
    return cmd

def compact_artifacts(run):
    # Only generated files inside this run; reference prefix/model files are untouched.
    hashes = {}
    for path in run.rglob('*'):
        if path.is_file() and path.suffix in ('.bin', '.npy', '.npz', '.bf16'):
            hashes[str(path.relative_to(run))] = dict(sha256=sha(path), bytes=path.stat().st_size)
    write(run / 'intermediate-artifact-hashes.json', hashes)
    for name in hashes:
        (run / name).unlink()

def verified_report(target, record, session):
    report = json.loads((target / 'report.json').read_text())
    if (report.get('wasm_sha256') != session['wasm_sha256']
            or report.get('deployed_wasm_sha256') != session['wasm_sha256']
            or report.get('input_hash') != record['input_sha256']
            or not report.get('comparison', {}).get('typed_output_valid')
            or report.get('replayed_queries') != 0 or report.get('fallback')
            or type(report.get('query_count')) is not int or report['query_count'] <= 0
            or report.get('executed_query_count') != report['query_count']):
        raise ValueError(f'invalid canister run report: {target}')
    validate_decision(report['decision_query']['ok']['decision'], record['options'])
    return report

def archive_attempt(directory, target):
    # Preserve failed/interrupted evidence, then start with an empty journal.
    archived = directory / 'aborted'
    archived.mkdir(exist_ok=True)
    attempt = 1
    while (archived / f'{target.name}-attempt-{attempt:03d}').exists():
        attempt += 1
    target.rename(archived / f'{target.name}-attempt-{attempt:03d}')

def run(directory, retry_errors=False):
    fixture = json.loads((directory / 'inputs.json').read_text())
    session = json.loads((directory / 'session.json').read_text())
    assert sha(directory / 'inputs.json') == session['input_sha256']
    for index, record in enumerate(fixture['records']):
        target = directory / 'runs' / f'{index:03d}'
        if (target / 'report.json').exists():
            try:
                verified_report(target, record, session)
            except (ValueError, KeyError, TypeError):
                if not retry_errors:
                    raise
            else:
                continue
        if (target / 'error.json').exists() and not retry_errors:
            continue
        assert all(sha(p) == h for p, h in session['source_hashes'].items()), 'source changed during run'
        if target.exists() and any(target.iterdir()):
            archive_attempt(directory, target)
        target.mkdir(parents=True, exist_ok=True)
        cmd = command(directory, index, record)
        write(target / 'command.json', cmd)
        print(json.dumps(dict(starting=index, total=len(fixture['records']), task_id=record['task_id'], tokens=len(record['token_ids']))), flush=True)
        start = time.monotonic()
        with (target / 'run.log').open('a') as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(target / 'error.json', dict(returncode=result.returncode, elapsed_seconds=time.monotonic()-start,
                                            log_tail=(target / 'run.log').read_text()[-4000:]))
            print(json.dumps(dict(failed=index, returncode=result.returncode)), flush=True)
        else:
            report = verified_report(target, record, session)
            assert all(sha(p) == h for p, h in session['source_hashes'].items()), 'source changed during run'
            compact_artifacts(target)
            print(json.dumps(dict(completed=index+1, prediction=report['comparison']['value'], queries=report['query_count'],
                                  instructions=report['total_instructions'], elapsed_seconds=time.monotonic()-start)), flush=True)
        score(directory)
    score(directory)

def score(directory):
    prepared = json.loads((directory / 'prepared.json').read_text())
    session = json.loads((directory / 'session.json').read_text())
    fixture = json.loads((directory / 'inputs.json').read_text())
    if sha(directory / 'inputs.json') != session['input_sha256']:
        raise ValueError('benchmark input hash mismatch')
    answers, runs, pending = {}, [], []
    for key, entry in prepared['tasks'].items():
        if entry['status'] == 'unavailable':
            answers[key] = dict(status='unavailable', reason=entry['reason'])
            continue
        target = directory / 'runs' / f'{entry["record_index"]:03d}'
        if not (target / 'report.json').exists():
            if (target / 'error.json').exists():
                answers[key] = dict(status='unavailable', reason=json.loads((target / 'error.json').read_text())['log_tail'])
            else:
                answers[key] = dict(status='pending', reason='not yet run')
                pending.append(key)
            continue
        report = verified_report(target, fixture['records'][entry['record_index']], session)
        decision = report['decision_query']['ok']['decision']
        answers[key] = dict(status='model_prediction' if not decision['abstained'] else 'unavailable',
                            label=decision['value'], reason='reserved model unknown/abstention' if decision['abstained'] else None,
                            logits=decision['raw_logits'][:len(entry['task']['options'])], evidence=[],
                            input_tokens=entry['tokens'], decision=decision)
        runs.append(dict(task_key=key, proposal_ids=entry['proposal_ids'], task_id=entry['task']['task_id'],
                         tokens=entry['tokens'], prediction=decision['value'], abstained=decision['abstained'],
                         queries=report['query_count'], instructions=report['total_instructions'],
                         candid_bytes=report['total_candid_bytes'], seconds=report['end_to_end_seconds_excluding_process_startup'],
                         replayed_queries=report['replayed_queries'], report_sha256=sha(target / 'report.json')))
    metadata = dict(name='imajev-canister-int8', adapter='actual local canister ordinary queries', **session)
    class Adapter:
        def metadata(self): return metadata
        def predict(self, task):
            a = answers[task_key(task)]
            if a['status'] != 'model_prediction':
                raise RuntimeError(a['reason'])
            return {k: a[k] for k in ('label','logits','evidence','input_tokens')}
    adapter = Adapter()
    comparison = dict(schema_version=1, complete=not pending, proposals=[])
    votes = []
    for e in prepared['proposals']:
        trial = run_advisory(e['facts'], adapter)
        comparison['proposals'].append({k: e[k] for k in ('snapshot_path','snapshot_sha256','facts','model_tasks')} |
                                      {'model_assessments': [{'model': trial['model'], 'advisory': trial['advisory']}]})
        votes.append(dict(proposal_id=e['proposal_id'], snapshot_sha256=e['snapshot_sha256'], task=e['vote_task'],
                          prediction=predict_vote(adapter, e['vote_task'])))
    labels_path = SOURCE / 'artifacts/proposal-assessment/boomdao-600-660/v1/reviewed-text-labels.json'
    labels = json.loads(labels_path.read_text())
    comparison['evaluation'] = evaluate(comparison, labels)
    comparison['labels_sha256'] = sha(labels_path)
    representatives, seen = [], set()
    for e in comparison['proposals']:
        text = json.dumps(e['facts']['text'], ensure_ascii=False, sort_keys=True)
        if text not in seen:
            seen.add(text)
            representatives.append(e)
    unique_labels = dict(schema_version=1, snapshots={e['snapshot_sha256']: labels['snapshots'][e['snapshot_sha256']] for e in representatives})
    comparison['unique_text_evaluation'] = evaluate(dict(proposals=representatives), unique_labels)
    report = dict(complete=not pending, model=metadata, unique_task_statuses=dict(collections.Counter(a['status'] for a in answers.values())),
                  completed_canister_inferences=len(runs), runnable_unique_tasks=session['runnable_unique_tasks'],
                  pending=len(pending), text_evaluation=comparison['evaluation'], unique_text_evaluation=comparison['unique_text_evaluation'],
                  vote_counts=dict(collections.Counter(e['prediction'].get('label','unavailable') for e in votes)),
                  vote_accuracy_measured=False, text_labels='assistant provisional; not human validated',
                  total_executed_queries=sum(r['queries'] for r in runs), total_instructions=sum(r['instructions'] for r in runs),
                  total_candid_bytes=sum(r['candid_bytes'] for r in runs), runs=runs, votes=votes, task_results=answers)
    write(directory / 'text-comparison.json', comparison)
    write(directory / 'report.json', report)

def main():
    global SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='artifacts/proposal-assessment-canister-20261005')
    parser.add_argument('--mode', choices=('prepare','run','score'), required=True)
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--source-root', type=pathlib.Path, default=ROOT, help='root containing artifacts/proposal-assessment/boomdao-600-660/v1')
    args = parser.parse_args()
    SOURCE = args.source_root.expanduser().resolve()
    if not (SOURCE / 'artifacts/proposal-assessment/boomdao-600-660/v1').is_dir():
        parser.error('snapshot archive missing; specify --source-root for the retained benchmark archive')
    directory = ROOT / args.directory
    if args.mode == 'prepare': prepare(directory)
    elif args.mode == 'run': run(directory, args.retry_errors)
    else: score(directory)

if __name__ == '__main__':
    main()
