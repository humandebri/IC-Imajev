#!/usr/bin/env python3
"""Rerun local binary assessment with fresh prefix and model queries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import assess_proposal_range as assessment
from evaluation_run_lock import run_lock
from repository_paths import existing_directory
from tools.proposal_assessment.binary_benchmark import binary_decision
from tools.proposal_assessment.validate_binary_improvement import THRESHOLD

D = ROOT / 'artifacts/proposal-assessment-fresh-500-660-20261007'
E = D / 'evaluation'
O = ROOT / 'artifacts/proposal-query-optimization-v1'
MODULE = '6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4'
CODEC = 'e9644266f9bea8efdff5c1d5017b7c82beb67c3030279e6f27248b90fa51ef6e'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(p)


def frozen():
    for name in ('source-hashes.json', 'fresh-source-hashes.json'):
        for path, digest in read(E / name).items():
            assert sha(Path(path)) == digest, f'source changed: {path}'


def prepare(snapshot_archive=None):
    snapshot_archive = Path(snapshot_archive) if snapshot_archive is not None else ROOT / 'artifacts/proposal-assessment-500-660-20261006/snapshots'
    D.mkdir(exist_ok=False)
    (D / 'snapshots').mkdir()
    (D / 'snapshots/manifest.json').write_bytes((snapshot_archive / 'manifest.json').read_bytes())
    assessment.prepare(D, reuse=False, snapshot_archive=snapshot_archive)
    fixture = read(E / 'inputs.json')
    assert fixture == read(O / 'inputs.json'), 'different inputs require a new query schedule'
    save(D / 'plan.json', read(O / 'plan.json'))
    paths = [Path(__file__), O / 'final-runner.py', O / 'proposal_query32_graph.py',
             O / 'proposal_query32_balanced_v3_graph.py', ROOT / 'scripts/prepare_prefix_reuse.py',
             ROOT / 'checkpoints/full-int8.manifest.json', ROOT / 'artifacts/merged-query32-v1/build/full.wasm',
             ROOT / 'artifacts/query32-v1/client-build/imajev-client', E / 'inputs.json',
             E / 'prepared.json', D / 'plan.json'] + list((ROOT / 'client').glob('*.py'))
    save(E / 'fresh-source-hashes.json', {str(p): sha(p) for p in paths})
    save(D / 'execution-policy.json', dict(model_module=MODULE, inference_canister='7st3i-3l777-77775-aaaja-cai',
         codec_canister='7vs54-wt777-77775-aaajq-cai', threshold=THRESHOLD,
         prediction_reuse=False, prior_prefix_reuse=False, snapshot_refresh=False,
         scope='Whole numerical participation tasks; original routing, exact units and approval evidence gate.'))


def archive_attempt(dest):
    """Preserve an unfinished attempt before starting with an empty journal."""
    if not dest.exists() or not any(dest.iterdir()):
        return
    relative = dest.relative_to(D)
    archived = D / 'aborted' / relative.parent
    archived.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (archived / f'{dest.name}-attempt-{attempt:03d}').exists():
        attempt += 1
    dest.rename(archived / f'{dest.name}-attempt-{attempt:03d}')


def call(cmd, dest):
    archive_attempt(dest)
    dest.mkdir(parents=True, exist_ok=True)
    save(dest / 'command.json', cmd)
    with (dest / 'run.log').open('a') as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=log, check=True)
    frozen()


def command(bank):
    cmd = read(ROOT / 'artifacts/proposal-full-161-20261007/base-command.json')
    cmd[1] = str(O / 'final-runner.py')
    cmd[cmd.index('--reference') + 1] = str(E / 'inputs.json')
    cmd[cmd.index('--cache') + 1] = str(D / 'prefixes' / f'{bank:02d}' / 'queries')
    return cmd


def validate(i):
    fixture = read(E / 'inputs.json'); x = fixture['records'][i]
    dest = E / 'runs' / f'{i:03d}'; r = read(dest / 'report.json')
    manifest = read(ROOT / 'checkpoints/full-int8.manifest.json')
    assert r['model'] == fixture['model_lock_sha256'] == manifest['model']
    assert r['pack_hash'] == manifest['pack_hash'] and r['input_hash'] == x['input_sha256']
    assert r['wasm_sha256'] == r['deployed_wasm_sha256'] == MODULE
    assert r['tokens'] == len(x['token_ids']) <= 128
    assert [z['layer'] for z in r['layers']] == list(range(32))
    assert r['executed_query_count'] == r['query_count']
    assert not r['replayed_queries'] and not r.get('fallback')
    assert r['comparison']['typed_output_valid']
    assert not (dest / 'reuse.json').exists()
    assert r['max_query_instructions'] < 5_000_000_000 and r['max_observed_heap_bytes'] < 2**32
    assert all(q['ok']['request_bytes'] < 2_000_000 and q['ok']['reply_bytes'] < 2_000_000 for q in r['queries'])
    decision = r['decision_query']['ok']['decision']; logits = decision['raw_logits']
    assert len(logits) == 3 and x['options'] == ['approve', 'reject']
    label, score = binary_decision(logits[:2], .5)
    old = read(ROOT / 'artifacts/proposal-assessment-500-660-20261006/evaluation/runs' / f'{i:03d}' / 'report.json')
    old_logits = old['decision_query']['ok']['decision']['raw_logits']
    row = dict(record=i, input_sha256=x['input_sha256'], report_sha256=sha(dest / 'report.json'),
               raw_logits=logits, prediction=label, score=score, queries=r['query_count'],
               fresh_inference=True, prior_logits_bit_equal=logits == old_logits,
               seconds=r['wall_seconds_this_run'])
    save(dest / 'verified.json', row)
    hashes = {str(p.relative_to(dest)): dict(bytes=p.stat().st_size, sha256=sha(p))
              for p in (dest / 'queries').rglob('*') if p.is_file() and p.suffix in ('.bin', '.npz', '.npy', '.bf16')}
    save(dest / 'intermediate-artifact-hashes.json', hashes)
    for path in hashes:
        (dest / path).unlink()
    return row


def fresh_results(out):
    prepared = read(out / 'prepared.json'); rows = []
    for entry in prepared['entries']:
        row = dict(entry, status='pending'); i = entry['record_index']
        path = out / 'runs' / f'{i:03d}' / 'verified.json'
        if path.exists():
            v = read(path)
            assert sha(path.parent / 'report.json') == v['report_sha256']
            assert v['fresh_inference']
            row.update(status='evaluated', prediction=v['prediction'], score=v['score'],
                       canonical_logits=v['raw_logits'][:2], report_sha256=v['report_sha256'],
                       expectation_match=None, decisions={str(t): binary_decision(v['raw_logits'][:2], t)[0]
                                                         for t in prepared['thresholds']})
        rows.append(row)
    save(out / 'results.json', dict(complete=all(x['status'] == 'evaluated' for x in rows), rows=rows,
                                  scope=prepared['scope'], accuracy_measured=False))
    return rows


def summarize():
    assessment.report(D, results_fn=fresh_results)
    r = read(E / 'report.json'); rows = [read(p) for p in sorted((E / 'runs').glob('*/verified.json'))]
    prep = []
    for bank in read(D / 'plan.json')['banks']:
        n = read(D / 'plan.json')['banks'].index(bank)
        p = D / 'prefixes' / f'{n:02d}' / 'report.json'
        c = D / 'packets' / f'{n:02d}' / 'report.json'
        if p.exists() and c.exists():
            a, b = read(p), read(c)
            assert a['deployed_wasm_sha256'] == MODULE and a['replayed_queries'] == 0
            prep.append(dict(bank=n, prefix_queries=a['executed_query_count'], codec_queries=b['preparation_queries']))
    old = read(ROOT / 'artifacts/proposal-assessment-500-660-20261006/evaluation/report.json')
    previous = {p['proposal_id']: p for p in old['proposals']}
    changes = [dict(proposal_id=p['proposal_id'], previous=previous[p['proposal_id']]['final_label'], current=p['final_label'])
               for p in r['proposals'] if p['route'] != 'participation_model' or p['final_label'] is not None
               if previous[p['proposal_id']]['final_label'] != p['final_label']]
    extra = dict(fresh_inferences=len(rows), reused_predictions=0, completed_prefix_banks=len(prep),
                 inference_queries=sum(x['queries'] for x in rows),
                 preparation_queries=sum(x['prefix_queries'] + x['codec_queries'] for x in prep),
                 prior_logits_bit_equal=sum(x['prior_logits_bit_equal'] for x in rows), changes=changes)
    save(D / 'fresh-summary.json', dict(summary=r['summary'], execution=extra))
    text = (E / 'REPORT.md').read_text()
    text = text.replace('2026-10-06。公開Dashboardから161件を取得し、snapshot hashを照合。',
                        '2026-10-07新規推論。2026-10-06取得済みの161件のsnapshotをhash照合し、ローカルに取り込んだ二択ハーネスで全件の処理経路を再計算。')
    text = text.replace('同一token列・選択肢・モデルの検証済み過去推論は再利用し、新規推論と区別。',
                        '既存の予測結果とprefix状態は再利用していない。モデル対象の同一token列のみ今回の実行内でまとめて新規推論した。')
    text += '\n\n新規実行の集計: ' + json.dumps(extra, ensure_ascii=False) + '\n'
    (E / 'REPORT.md').write_text(text)
    return dict(summary=r['summary'], execution=extra)


def execute():
    frozen(); plan = read(D / 'plan.json')
    flags = ['--terminal-readout', '--fuse-terminal-attention', '--fuse-terminal-decision',
             '--fuse-terminal-tail', '--fuse-prefix-start', '--tail-start', '--join-start', '--roll-start']
    for p in plan['records']:
        i, b = p['record'], p['bank']; dest = E / 'runs' / f'{i:03d}'
        if (dest / 'verified.json').exists():
            continue
        prefix, packets = D / 'prefixes' / f'{b:02d}', D / 'packets' / f'{b:02d}'
        if not (prefix / 'report.json').exists():
            # Codec packets belong to this prefix attempt, even if their cache
            # marker survived an interrupted preparation.
            archive_attempt(packets)
            print(json.dumps(dict(stage='fresh-prefix', bank=b, tokens=p['prefix'])), flush=True)
            call(command(b) + ['--record', str(plan['banks'][b]['record']), '--prefix-tokens', str(p['prefix']),
                              '--prepare-prefix', '--directory', str(prefix)], prefix)
        if not (packets / 'cache.json').exists() or not (packets / 'report.json').exists():
            call([sys.executable, '-B', str(ROOT / 'scripts/prepare_prefix_reuse.py'), '--prefix-directory', str(prefix),
                  '--directory', str(packets), '--canister', '7vs54-wt777-77775-aaajq-cai', '--codec-module', CODEC,
                  '--run-report', str(packets / 'report.json')], packets)
        route = ['--query32-start'] if p['route'] == 'query32' else []
        print(json.dumps(dict(stage='fresh-inference', record=i, prefix=p['prefix'], suffix=p['suffix'], route=p['route'])), flush=True)
        call(command(b) + flags + route + ['--hybrid-cache', str(packets), '--record', str(i),
             '--step-offset', str(18000000 + i * 10000), '--directory', str(dest)], dest)
        row = validate(i); summarize(); print(json.dumps(dict(stage='verified', **row)), flush=True)
    print(json.dumps(summarize()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare', 'run', 'report'])
    parser.add_argument('--snapshot-archive', type=existing_directory, help='prepare source archive containing manifest.json and snapshots/')
    parser.add_argument('--directory', type=Path, default=D, help='new run directory or retained run to inspect')
    args = parser.parse_args()
    D = args.directory.expanduser().resolve(); E = D / 'evaluation'
    if args.mode == 'prepare':
        prepare(args.snapshot_archive)
    else:
        with run_lock(E):
            if args.mode == 'run':
                execute()
            else:
                print(json.dumps(summarize()))
