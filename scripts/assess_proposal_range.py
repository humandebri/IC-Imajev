#!/usr/bin/env python3
"""Frozen proposal snapshots -> existing binary participation tool -> Imajev queries."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT / 'scripts'))
from proposal_assessment.core import assess
from proposal_assessment.vote import make_vote_task
from proposal_assessment.routing import select_task
from proposal_assessment.budget_policy import evidence_gate
from proposal_assessment.improve_binary import encode_state, compact_exact_state, VARIANTS, results, approval_evidence_gate
from proposal_assessment.compact_units import encode, QUESTION
from proposal_assessment.validate_binary_improvement import THRESHOLD, COMPACT_QUESTION
from proposal_assessment.binary_benchmark import run
from proposal_assessment.token_sweep import save, sha
from prepare_text import TextPreparer
from vision_decision.scoring import verified_label_ids
from evaluation_run_lock import run_lock
from proposal_snapshots import read_snapshot
from repository_paths import existing_directory

ORIGIN = TOOLS.parent / 'artifacts/proposal-assessment'

def prepare(directory, reuse=True, snapshot_archive=None):
    out = directory / 'evaluation'
    if out.exists():
        raise ValueError('evaluation directory exists')
    manifest_path = directory / 'snapshots/manifest.json'
    snapshot_archive = Path(snapshot_archive) if snapshot_archive is not None else manifest_path.parent
    if sha(snapshot_archive / 'manifest.json') != sha(manifest_path):
        raise ValueError('selected snapshot archive manifest mismatch')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['complete'] and manifest['all_fetched']
    assert sorted(r['proposal_id'] for r in manifest['records']) == list(range(manifest['first'], manifest['last'] + 1))
    preparer = TextPreparer()
    records, entries, gates, facts_rows = [], [], [], []
    indices = {}
    for snapshot in manifest['records']:
        path, proposal = read_snapshot(snapshot_archive, snapshot, manifest['sns_root'])
        facts = assess(proposal)
        facts_rows.append(dict(proposal_id=snapshot['proposal_id'], snapshot_sha256=snapshot['sha256'], facts=facts))
        task = make_vote_task(facts)
        entry = dict(task=task, proposal_ids=[snapshot['proposal_id']], expected=None,
                     synthetic=False, split='range_500_660_no_gold', snapshot_sha256=snapshot['sha256'])
        selected = select_task(task)
        if not selected['requires_model']:
            gates.append(dict(entry, route='skipped', label=None, tokens=0, reason=selected['reason']))
            continue
        gate = evidence_gate(task)
        if gate['route'] != 'model':
            gates.append(dict(entry, **gate, tokens=0))
            continue
        state = json.loads(task['state'])
        eligible = any(r.get('field') in ('neuron_minimum_stake_e8s', 'neuron_minimum_dissolve_delay_to_vote_seconds') for r in state['changes_or_requests'])
        if state['action'] != 'ManageNervousSystemParameters' or not eligible:
            gates.append(dict(entry, label='hold', tokens=0, route='outside_direct_eligibility_scope', reason='Outside numerical stake/voting-lock participation scope.'))
            continue
        try:
            rendered, question, layout = encode_state(task), VARIANTS['ratio'], 'ratio'
            prompt = f'State: {rendered}\nQuestion: {question}\nA: approve\nB: reject'
            ids = preparer.tokenizer.encode(preparer.render(prompt), add_special_tokens=False)
            if len(ids) > 128:
                rendered, question, layout = compact_exact_state(task), COMPACT_QUESTION, 'compact_ratio'
                prompt = f'State: {rendered}\nQuestion: {question}\nA: approve\nB: reject'
                ids = preparer.tokenizer.encode(preparer.render(prompt), add_special_tokens=False)
            if len(ids) > 128:
                rendered, preserved = encode(task, ratio_words=True)
                layout = 'unit_grouped_ratio_words'
                entry['preserved_facts'] = preserved
                prompt = f'{rendered}\n{QUESTION}\nA approve B reject'
                ids = preparer.tokenizer.encode(preparer.render(prompt), add_special_tokens=False)
        except (ValueError, KeyError) as error:
            gates.append(dict(entry, label='hold', tokens=0, route='unsupported_encoding', reason=str(error)))
            continue
        if len(ids) > 128:
            gates.append(dict(entry, label='hold', tokens=0, full_input_tokens=len(ids), route='budget_overflow', reason='All changed values preserved; input exceeds 128 tokens.', candidate_prompt=prompt))
            continue
        assert verified_label_ids(preparer.tokenizer, preparer.render(prompt), ['A', 'B']) == [preparer.binding['codes'][i]['token_id'] for i in range(2)]
        key = json.dumps(ids)
        if key in indices:
            entries[indices[key]]['proposal_ids'].append(snapshot['proposal_id'])
            continue
        index = len(records)
        indices[key] = index
        records.append(dict(id=f'range_{index}', options=['approve','reject'], gold=None, token_ids=ids,
                            input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), prompt=prompt))
        entries.append(dict(entry, record_index=index, variant=layout, state=rendered, tokens=len(ids)))
    out.mkdir()
    save(out/'inputs.json', dict(model_lock_sha256=sha(ROOT/'MODEL_LOCK.json'), records=records))
    save(out/'prepared.json', dict(entries=entries, gates=gates, thresholds=[THRESHOLD], scope='Frozen 500-660; numerical participation only; no human gold',
                                  first=manifest['first'], last=manifest['last'], approval_gate_enabled=True, local_rejection_guard_enabled=False,
                                  accuracy_measured=False, reference_is_gold=False, source_manifest_sha256=sha(manifest_path)))
    save(out/'facts.json', facts_rows)
    shutil.copyfile(ORIGIN/'binary-improved-final-20261006/runner.py', out/'runner.py')
    source_paths = [Path(__file__), ROOT/'scripts/proposal_snapshots.py', ROOT/'MODEL_LOCK.json'] + list((TOOLS/'proposal_assessment').glob('*.py'))
    save(out/'source-hashes.json', {str(p):sha(p) for p in source_paths})
    save(out/'identity.json',dict(inputs_sha256=sha(out/'inputs.json'), prepared_sha256=sha(out/'prepared.json'), runner_sha256=sha(out/'runner.py'), bridge_sha256=sha(ROOT/'target/release/imajev-client')))
    # Exact token/model identity permits reuse, separately counted from new execution.
    previous_range = ROOT/'artifacts/proposal-assessment-500-660-20261006/evaluation'
    for previous in ([previous_range, ORIGIN/'binary-600-660-ratio-20261006', ORIGIN/'binary-improved-final-20261006'] if reuse else []):
        previous_rows = results(previous)
        old = json.loads((previous/'inputs.json').read_text())
        if old['model_lock_sha256'] != sha(ROOT/'MODEL_LOCK.json'):
            continue
        for i, record in enumerate(records):
            destination = out/'runs'/f'{i:03d}'
            if destination.exists():
                continue
            for row in previous_rows:
                j = row['record_index']
                candidate = old['records'][j]
                if row['status']=='evaluated' and candidate['token_ids']==record['token_ids'] and candidate['options']==record['options']:
                    shutil.copytree(previous/'runs'/f'{j:03d}', destination)
                    save(destination/'reuse.json', dict(source=str(previous/'runs'/f'{j:03d}'), exact_token_identity=True, source_report_sha256=sha(previous/'runs'/f'{j:03d}'/'report.json')))
                    break
    print(json.dumps(dict(proposals=len(facts_rows), model_distinct=len(records), gates=len(gates), reused=sum((out/'runs'/f'{i:03d}'/'reuse.json').exists() for i in range(len(records))), token_range=[min((e['tokens'] for e in entries),default=0),max((e['tokens'] for e in entries),default=0)])),flush=True)


def report(directory, results_fn=results):
    out = directory/'evaluation'
    prepared = json.loads((out/'prepared.json').read_text())
    rows = results_fn(out)
    for row in rows:
        if row['status']=='evaluated':
            row['final_label'], row['reason'] = approval_evidence_gate(row['task'], row['decisions'][str(THRESHOLD)])
        else:
            row['final_label'], row['reason'] = None, row['status']
        row['route'] = 'participation_model'
        row['reused'] = (out/'runs'/f"{row['record_index']:03d}"/'reuse.json').exists()
    expanded = []
    for row in rows+prepared['gates']:
        for pid in row['proposal_ids']:
            expanded.append(dict(proposal_id=pid, route=row['route'], input_tokens=row['tokens'], candidate_tokens=row.get('full_input_tokens'),
                                 binary_prediction=row.get('prediction'), uncalibrated_score=row.get('score'), final_label=row.get('final_label',row.get('label')),
                                 reason=row.get('reason'), reused=row.get('reused',False)))
    expanded.sort(key=lambda r:r['proposal_id'])
    assert [r['proposal_id'] for r in expanded]==list(range(prepared['first'],prepared['last']+1))
    considered=[r for r in expanded if r['route']!='skipped']
    summary=dict(complete=all(r['status']=='evaluated' for r in rows),total_proposals=len(expanded),selected=len(considered),skipped=len(expanded)-len(considered),
                 counts=dict(Counter(r['final_label'] for r in considered)), routes=dict(Counter(r['route'] for r in expanded)),model_distinct=len(rows),
                 evaluated_distinct=sum(r['status']=='evaluated' for r in rows), reused_distinct=sum(r['status']=='evaluated' and r['reused'] for r in rows),
                 new_distinct=sum(r['status']=='evaluated' and not r['reused'] for r in rows),accuracy_measured=False)
    save(out/'report.json',dict(summary=summary, proposals=expanded, model_rows=rows))
    with (out/'proposals.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(expanded[0]));writer.writeheader();writer.writerows(expanded)
    lines=['# BOOM DAO proposal 500〜660：Imajev検査','', '2026-10-06。公開Dashboardから161件を取得し、snapshot hashを照合。既存proposal_assessmentの重要項目選別・数値抽出・二択参加方針・128-token上限・未校正スコア閾値0.6・承認証拠検査を適用。資金等の証拠不足・対応範囲外はTool hold、Motion/表示変更等は除外。投票は実行していない。','', f'集計: {json.dumps(summary,ensure_ascii=False)}','', '同一token列・選択肢・モデルの検証済み過去推論は再利用し、新規推論と区別。全161件が独立した新規モデル推論という意味ではない。過去GPTラベルや採決結果は入力しない。正解ラベルがないため正答率は測定しない。','', '旧値はproposal rendering由来でchain検証なし。判定対象は数値上の投票参加条件であり、proposalの安全性全般の承認ではない。128 tokens以内でも32 query保証はない。','', '|ID|処理|tokens|二択|スコア|最終判定|理由|','|---|---|---:|---|---:|---|---|']
    for r in expanded:
        score=f"{r['uncalibrated_score']:.4f}" if r['uncalibrated_score'] is not None else '—'
        label=r['final_label'] or ('除外' if r['route']=='skipped' else '未完了')
        lines.append(f"|{r['proposal_id']}|{r['route']}|{r['input_tokens']}|{r['binary_prediction'] or '—'}|{score}|{label}|{(r['reason'] or '').replace('|','/')}|")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','report']);p.add_argument('--directory',type=Path,required=True);p.add_argument('--no-reuse',action='store_true');p.add_argument('--snapshot-archive',type=existing_directory,help='archive containing manifest.json and snapshots/; defaults to DIRECTORY/snapshots');a=p.parse_args();d=a.directory.resolve()
    if a.mode=='prepare':prepare(d,reuse=not a.no_reuse,snapshot_archive=a.snapshot_archive)
    elif a.mode=='report':report(d)
    else:
        with run_lock(d/'evaluation'):
            for path,digest in json.loads((d/'evaluation/source-hashes.json').read_text()).items():
                assert sha(Path(path))==digest, f'source changed: {path}'
            run(d/'evaluation',0,1)
            report(d)

if __name__=='__main__':main()
