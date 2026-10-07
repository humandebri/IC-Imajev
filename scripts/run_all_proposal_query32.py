#!/usr/bin/env python3
"""Execute every distinct bounded snapshot window with pinned ordinary queries."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/proposal-full-161-20261007/query32-v1'
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluation_run_lock import run_lock

MODULE = '6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4'
CODEC = 'e9644266f9bea8efdff5c1d5017b7c82beb67c3030279e6f27248b90fa51ef6e'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    pending = p.with_suffix(p.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    pending.replace(p)

def check_frozen():
    for name in ('preparation-identities.json', 'execution-identities.json'):
        for path, digest in json.loads((D / name).read_text()).items():
            assert sha(Path(path)) == digest, f'source changed: {path}'

def setup():
    paths = [Path(__file__), ROOT / 'scripts/run_merged_query32.py',
             ROOT / 'scripts/prepare_prefix_reuse.py', D / 'base-command.json',
             ROOT / 'artifacts/merged-query32-v1/build/full.wasm',
             ROOT / 'artifacts/text-short-v2/codec-build/diagnostic.wasm',
             ROOT / 'artifacts/query32-v1/client-build/imajev-client']
    paths += list((ROOT / 'client').glob('*.py'))
    identity = {str(p): sha(p) for p in paths}
    path = D / 'execution-identities.json'
    if path.exists():
        assert json.loads(path.read_text()) == identity
    else:
        save(path, identity)
    check_frozen()

def call(command, directory):
    directory.mkdir(parents=True, exist_ok=True)
    save(directory / 'command.json', command)
    with (directory / 'run.log').open('a') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, check=True)
    check_frozen()

def validate(index):
    fixture = json.loads((D / 'inputs.json').read_text())
    r = fixture['records'][index]
    dest = D / 'runs' / f'{index:03d}'
    report = json.loads((dest / 'report.json').read_text())
    manifest = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    assert 1 <= len(r['token_ids']) <= 86
    assert report['tokens'] == len(r['token_ids'])
    assert r['prefix_tokens'] == 27 and len(r['token_ids']) - 27 <= 59
    assert report['query32_effective'] and report['query_count'] == 32
    assert report['input_hash'] == r['input_sha256']
    assert report['model'] == manifest['model'] == fixture['model_lock_sha256']
    assert report['pack_hash'] == manifest['pack_hash']
    assert report['wasm_sha256'] == report['deployed_wasm_sha256'] == MODULE
    assert [x['layer'] for x in report['layers']] == list(range(32))
    assert report['query_count'] == report['executed_query_count']
    assert not report['replayed_queries'] and not report.get('fallback')
    assert report['comparison']['typed_output_valid']
    assert report['max_query_instructions'] < 5_000_000_000
    assert report['max_observed_heap_bytes'] < 2 ** 32
    assert all(q['ok']['request_bytes'] < 2_000_000 and q['ok']['reply_bytes'] < 2_000_000 for q in report['queries'])
    d = report['decision_query']['ok']['decision']
    assert len(d['raw_logits']) == 4 and len(d['probabilities']) == 3
    assert d['value'] in (None, 'approve', 'reject', 'hold')
    row = {'record': index, 'tokens': report['tokens'], 'prediction': d['value'] or '__unknown__',
           'abstained': d['abstained'], 'probabilities': d['probabilities'],
           'unknown_probability': d['unknown_probability'], 'raw_logits': d['raw_logits'],
           'queries': report['query_count'], 'instructions': report['total_instructions'],
           'candid_bytes': report['total_candid_bytes'], 'seconds': report['wall_seconds_this_run'],
           'report_sha256': sha(dest / 'report.json')}
    save(dest / 'verified.json', row)
    # Keep numerical output hashes and all metrics. Only remove this workflow's
    # transient request/reply frames after validated completion, to bound disk.
    hashes = {}
    for file in (dest / 'queries').rglob('*'):
        if file.is_file() and file.suffix in ('.bin', '.npy', '.npz'):
            hashes[str(file.relative_to(dest))] = {'bytes': file.stat().st_size, 'sha256': sha(file)}
    save(dest / 'intermediate-artifact-hashes.json', hashes)
    for name in hashes:
        (dest / name).unlink()
    return row

def summarize():
    prepared = json.loads((D / 'prepared.json').read_text())
    fixture = json.loads((D / 'inputs.json').read_text())
    rows = []
    for i in range(len(fixture['records'])):
        path = D / 'runs' / f'{i:03d}' / 'verified.json'
        if path.exists():
            row = json.loads(path.read_text())
            assert row['report_sha256'] == sha(path.parent / 'report.json')
            rows.append(row)
    by_index = {r['record']: r for r in rows}
    expanded = []
    for proposal in prepared['proposals']:
        output = [by_index.get(w['record']) for w in proposal['windows']]
        complete = all(x is not None for x in output)
        labels = [x['prediction'] for x in output if x]
        full_single = complete and len(output) == 1
        expanded.append({'proposal_id': proposal['proposal_id'], 'old_route': proposal['old_route'],
                         'old_label': proposal['old_label'], 'windows': len(output),
                         'completed_windows': len(labels), 'model_complete': complete,
                         'single_window_prediction': labels[0] if full_single else None,
                         'window_predictions': labels,
                         'holistic_model_recommendation': None,
                         'interpretation': 'Full facts projection in one frame; binary/links/code unverified' if full_single else 'Partial evidence windows; no holistic recommendation',
                         'distinct_records': sorted(set(w['record'] for w in proposal['windows']))})
    summary = {'complete': len(rows) == len(fixture['records']),
               'proposals': 161, 'completed_proposals': sum(x['model_complete'] for x in expanded),
               'distinct_inputs': len(fixture['records']), 'completed_inputs': len(rows),
               'total_windows': sum(x['windows'] for x in expanded),
               'completed_windows': sum(x['completed_windows'] for x in expanded),
               'single_window_proposals': sum(x['windows'] == 1 for x in expanded),
               'predictions_distinct_inputs': dict(Counter(x['prediction'] for x in rows)),
               'inference_queries': sum(x['queries'] for x in rows),
               'inference_instructions': sum(x['instructions'] for x in rows),
               'accuracy_measured': False, 'human_gold_labels': False,
               'max_tokens': 86, 'actual_max_tokens': max(len(x['token_ids']) for x in fixture['records']),
               'previous_routes': dict(Counter(x['old_route'] for x in expanded))}
    save(D / 'report.json', {'summary': summary, 'inputs': rows, 'proposals': expanded})
    with (D / 'proposals.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(expanded[0]))
        writer.writeheader()
        writer.writerows(expanded)
    lines = ['# 全161 proposalの86-token・32-query入力検証', '',
             '2026-10-07。500〜660の凍結snapshotを使用。事前除外なし。変更値・識別子・抽出済み説明文・未知項目を保持し、長いfacts投影を部分入力に分割。チャット付加tokenを含め各入力86以下。共通prefix27＋後半59以下。', '',
             f"完了 {summary['completed_proposals']}/161件、異なる入力 {len(rows)}/{summary['distinct_inputs']}、部分入力 {summary['completed_windows']}/{summary['total_windows']}。最大{summary['actual_max_tokens']} tokens。", '',
             '部分入力の出力をproposal全体の賛否に集約していない。人手正解ラベルなし、正答率未測定。画像・実行コード・外部リンクは検証していない。従来の二択参加条件と今回の三択内容評価は異なる質問であり、直接の精度比較には使えない。', '',
             f"異なる入力の生出力: {summary['predictions_distinct_inputs']}。推論query {summary['inference_queries']:,}（前半準備・圧縮は別計上）。", '',
             '全変更値が入力のどこに現れるかはprepared.jsonのatoms/windows/source/start/endで追跡できる。exact tool footer、同一motion本文の重複、変更されないfield、factsの派生注釈のみ除外。任意の文字切り捨てはない。ロゴ等は既存抽出器のhash表現であり、元バイナリ内容をモデルに与えたとは主張しない。', '',
             '|proposal|従来の処理|部分数|完了|単一入力の出力|部分入力の出力|', '|---|---|---:|---|---|---|']
    for row in expanded:
        lines.append(f"|{row['proposal_id']}|{row['old_route']}|{row['windows']}|{row['model_complete']}|{row['single_window_prediction'] or '—'}|{','.join(row['window_predictions']) or '未実行'}|")
    (D / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return summary

def execute(limit=None):
    setup()
    fixture = json.loads((D / 'inputs.json').read_text())
    prepared = json.loads((D / 'prepared.json').read_text())
    base = json.loads((D / 'base-command.json').read_text())
    prefix = D / 'prefix'
    packets = D / 'packets'
    if not (prefix / 'report.json').exists():
        print('preparing common prefix', flush=True)
        call(base + ['--record', '0', '--prefix-tokens', str(prepared['prefix_tokens']),
                     '--prepare-prefix', '--directory', str(prefix)], prefix)
    if not (packets / 'cache.json').exists():
        print('preparing prefix packets', flush=True)
        call([sys.executable, '-B', str(ROOT / 'scripts/prepare_prefix_reuse.py'),
              '--prefix-directory', str(prefix), '--directory', str(packets),
              '--canister', '7vs54-wt777-77775-aaajq-cai', '--codec-module', CODEC,
              '--run-report', str(packets / 'report.json')], packets)
    flags = ['--hybrid-cache', str(packets), '--terminal-readout', '--fuse-terminal-attention',
             '--fuse-terminal-decision', '--fuse-terminal-tail', '--fuse-prefix-start',
             '--tail-start', '--join-start', '--roll-start', '--query32-start']
    completed = 0
    for i, record in enumerate(fixture['records']):
        dest = D / 'runs' / f'{i:03d}'
        if (dest / 'verified.json').exists():
            continue
        if limit is not None and completed >= limit:
            break
        assert len(record['token_ids']) <= 86
        print(json.dumps({'stage': 'start', 'record': i, 'tokens': len(record['token_ids'])}), flush=True)
        call(base + flags + ['--record', str(i), '--step-offset', str(9000000 + i * 10000),
                             '--directory', str(dest)], dest)
        row = validate(i)
        completed += 1
        print(json.dumps({'stage': 'verified', **row}), flush=True)
        summarize()
    print(json.dumps(summarize()), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'report'])
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    with run_lock(D):
        if args.mode == 'run':
            execute(args.limit)
        else:
            print(json.dumps(summarize()), flush=True)
