"""Apply the frozen binary participation pipeline to all 61 archived proposals."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

from .binary_benchmark import IMAJEV, run
from .budget_policy import evidence_gate
from .improve_binary import VARIANTS, encode_state, compact_exact_state, results, approval_evidence_gate
from .routing import select_task
from .token_sweep import ROOT, save, sha
from .validate_binary_improvement import THRESHOLD, COMPACT_QUESTION

sys.path.insert(0, str(ROOT / 'scripts'))
from proposal_snapshots import read_snapshot
from repository_paths import existing_directory


def prepare(out, compact_overflow=False, compact_ratio=False, snapshot_archive=None):
    if out.exists():
        raise ValueError('fresh output directory required')
    reference = ROOT/'tools/proposal_assessment/gpt-6.1-sol-medium-20261005-1509/vote-results.json'
    archive = (Path(snapshot_archive) if snapshot_archive is not None else
               ROOT/'artifacts/proposal-assessment/boomdao-600-660/v1')
    manifest_path = archive / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = json.loads(reference.read_text())
    assert manifest['complete'] and source['complete']
    snapshots = {r['proposal_id']: r for r in manifest['records']}
    assert set(snapshots) == set(range(600, 661))
    assert {r['proposal_id'] for r in source['proposals']} == set(snapshots)
    grouped = {}
    for proposal in source['proposals']:
        snapshot = snapshots[proposal['proposal_id']]
        read_snapshot(archive, snapshot, manifest['sns_root'])
        assert snapshot['sha256'] == proposal['snapshot_sha256']
        key = json.dumps(proposal['task'], sort_keys=True)
        if key not in grouped:
            grouped[key] = dict(task=proposal['task'], proposal_ids=[], expected=proposal['prediction']['label'])
        entry = grouped[key]
        assert entry['expected'] == proposal['prediction']['label']
        entry['proposal_ids'].append(proposal['proposal_id'])
    sys.path.insert(0, str(IMAJEV/'scripts'))
    from prepare_text import TextPreparer
    from vision_decision.scoring import verified_label_ids
    preparer = TextPreparer()
    records, entries, gates = [], [], []
    for entry in grouped.values():
        entry.update(split='previously_seen_real_600_660', synthetic=False)
        selected = select_task(entry['task'])
        if not selected['requires_model']:
            gates.append({**entry, 'label': None, 'tokens': 0, 'route': 'skipped', 'reason': selected['reason']})
            continue
        gate = evidence_gate(entry['task'])  # Experimental rejection guard stays disabled.
        if gate['route'] != 'model':
            gates.append({**entry, **gate, 'tokens': 0})
            continue
        state = json.loads(entry['task']['state'])
        eligible = any(r.get('field') in ('neuron_minimum_stake_e8s', 'neuron_minimum_dissolve_delay_to_vote_seconds')
                       for r in state['changes_or_requests'])
        if state['action'] != 'ManageNervousSystemParameters' or not eligible:
            gates.append({**entry, 'label': 'hold', 'tokens': 0, 'route': 'outside_direct_eligibility_scope'})
            continue
        try:
            rendered_state = encode_state(entry['task'])
            question, layout = VARIANTS['ratio'], 'ratio'
            prompt = f'State: {rendered_state}\nQuestion: {question}\nA: approve\nB: reject'
            token_ids = preparer.tokenizer.encode(preparer.render(prompt), add_special_tokens=False)
            if len(token_ids) > 128:
                rendered_state = compact_exact_state(entry['task'])
                question, layout = COMPACT_QUESTION, 'compact_ratio'
                prompt = f'State: {rendered_state}\nQuestion: {question}\nA: approve\nB: reject'
                token_ids = preparer.tokenizer.encode(preparer.render(prompt), add_special_tokens=False)
        except (ValueError, KeyError) as error:
            gates.append({**entry, 'label': 'hold', 'tokens': 0, 'route': 'unsupported_encoding', 'reason': str(error)})
            continue
        if len(token_ids) > 128:
            if compact_overflow:
                from .compact_units import encode, QUESTION
                try:
                    rendered_state, preserved_facts = encode(entry['task'],ratio_words=compact_ratio)
                    layout = 'unit_grouped_ratio_words' if compact_ratio else 'unit_grouped_overflow'
                    options_text = 'A approve B reject' if compact_ratio else 'A: approve\nB: reject'
                    prompt = f'{rendered_state}\n{QUESTION}\n{options_text}'
                    token_ids = preparer.tokenizer.encode(preparer.render(prompt), add_special_tokens=False)
                    entry['preserved_facts'] = preserved_facts
                except ValueError:
                    pass
        if len(token_ids) > 128:
            gates.append({**entry, 'label': 'hold', 'tokens': 0, 'full_input_tokens': len(token_ids),
                          'route': 'budget_overflow', 'candidate_prompt': prompt})
            continue
        assert verified_label_ids(preparer.tokenizer, preparer.render(prompt), ['A', 'B']) == [preparer.binding['codes'][i]['token_id'] for i in range(2)]
        records.append(dict(id=f'boom600_{len(records)}', options=['approve', 'reject'], gold=None,
                            token_ids=token_ids, input_sha256=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(), prompt=prompt))
        entries.append({**entry, 'record_index': len(records)-1, 'variant': layout,
                        'state': rendered_state, 'tokens': len(token_ids)})
    out.mkdir(parents=True)
    save(out/'inputs.json', dict(model_lock_sha256=sha(IMAJEV/'MODEL_LOCK.json'), records=records))
    save(out/'prepared.json', dict(entries=entries, gates=gates, thresholds=[THRESHOLD],
         scope='fixed <=128-token conditional numerical participation pipeline; previously seen 600-660',
         approval_gate_enabled=True, local_rejection_guard_enabled=False, accuracy_measured=False,
         compact_overflow_enabled=compact_overflow,
         compact_ratio_enabled=compact_ratio,
         reference_is_gold=False, source_reference_sha256=sha(reference), source_manifest_sha256=sha(manifest_path)))
    origin = ROOT/'artifacts/proposal-assessment/binary-improved-final-20261006'
    shutil.copyfile(origin/'runner.py', out/'runner.py')
    save(out/'identity.json', dict(inputs_sha256=sha(out/'inputs.json'), prepared_sha256=sha(out/'prepared.json'),
         runner_sha256=sha(out/'runner.py'), bridge_sha256=sha(IMAJEV/'target/release/imajev-client')))
    for name in ('benchmark_600_660_binary.py', 'binary_benchmark.py', 'improve_binary.py', 'routing.py', 'budget_policy.py', 'million_notation.py', 'validate_binary_improvement.py', 'compact_units.py'):
        shutil.copyfile(Path(__file__).parent/name, out/name.replace('.py', '_source.py'))
    if compact_overflow:
        previous = ROOT/'artifacts/proposal-assessment/binary-600-660-20261006'
        old_records = json.loads((previous/'inputs.json').read_text())['records']
        for i, record in enumerate(records):
            j = next((j for j,r in enumerate(old_records) if r['token_ids']==record['token_ids'] and r['options']==record['options']), None)
            if j is not None:
                shutil.copytree(previous/'runs'/f'{j:03d}', out/'runs'/f'{i:03d}')
                save(out/'runs'/f'{i:03d}'/'reuse.json', dict(source=str(previous/'runs'/f'{j:03d}'), exact_token_identity=True))
    print(json.dumps(dict(model_inputs=[(e['proposal_ids'], e['tokens']) for e in entries],
                         gates=[(e['proposal_ids'], e['route'], e.get('full_input_tokens')) for e in gates])))


def report(out):
    rows = results(out)
    prepared = json.loads((out/'prepared.json').read_text())
    for row in rows:
        if row['status'] == 'evaluated':
            row['final_label'], row['approval_gate_reason'] = approval_evidence_gate(row['task'], row['decisions'][str(THRESHOLD)])
    all_rows = [{**r, 'route': 'participation_model'} for r in rows] + prepared['gates']
    expanded = []
    for row in all_rows:
        for proposal_id in row['proposal_ids']:
            label = row.get('final_label', row.get('label'))
            expanded.append(dict(proposal_id=proposal_id, route=row['route'], tokens=row['tokens'],
                candidate_tokens=row.get('full_input_tokens'), raw_prediction=row.get('prediction'), score=row.get('score'),
                final_label=label, reference=row['expected'], reference_match=None if label is None else label == row['expected']))
    expanded.sort(key=lambda r: r['proposal_id'])
    assert [r['proposal_id'] for r in expanded] == list(range(600, 661))
    considered = [r for r in expanded if r['route'] != 'skipped']
    modeled = [r for r in considered if r['route'] == 'participation_model']
    summary = dict(complete=all(r['status']=='evaluated' for r in rows), total_proposals=61,
        selected_proposals=len(considered), skipped_proposals=61-len(considered),
        model_distinct=len(rows), model_proposals=len(modeled), tool_only_hold_proposals=len(considered)-len(modeled),
        counts=dict(Counter(r['final_label'] for r in considered)),
        reference_matches=sum(r['reference_match'] is True for r in considered),
        model_reference_matches=sum(r['reference_match'] is True for r in modeled),
        tokens_distinct=sum(r['tokens'] for r in rows), tokens_without_dedup=sum(r['tokens'] for r in modeled),
        max_input_tokens=max((r['tokens'] for r in rows), default=0), accuracy_measured=False,
        human_golds=False, independent_validation=False)
    save(out/'quality.json', dict(summary=summary, model_rows=rows, gates=prepared['gates'], proposals=expanded))
    with (out/'proposal-tokens.csv').open('w') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(expanded[0]));writer.writeheader();writer.writerows(expanded)
    lines=['# BOOM DAO #600〜660：固定二択方式の再評価', '',
        '以前取得した61件のsnapshotをhash照合し、重要項目選別・入力上限128・閾値0.6・承認証拠検査を変更せず適用。モデルはQwen3.5-4B＋Imajev、INT8。追加LLMによる前処理なし。同一taskをまとめて1回推論し、対応する各proposalへ結果を展開。', '',
        f'全61件のうち重要{len(considered)}件、除外{61-len(considered)}件。モデル実測{len(rows)}種類・{len(modeled)}件、Toolのみのhold {len(considered)-len(modeled)}件。除外はapprove/reject/holdではない。', '',
        f'判定数：{summary["counts"]}。過去GPT参照との一致は重要対象{summary["reference_matches"]}/{len(considered)}件、モデル対象{summary["model_reference_matches"]}/{len(modeled)}件。既知の評価集合で、人手goldに対する正答率や独立検証ではない。', '',
        '|proposal|処理|入力tokens／件|二択|未校正スコア|最終判定|過去GPT参照|','|---|---|---:|---|---:|---|---|']
    for row in sorted(all_rows, key=lambda r:r['proposal_ids'][0]):
        label=row.get('final_label',row.get('label'))
        lines.append(f'|{",".join(map(str,row["proposal_ids"]))}|{row["route"]}|{row["tokens"]}|{row.get("prediction","—")}|{row.get("score",0):.3f}|{label if label is not None else "除外"}|{row["expected"]}|')
    lines += ['', f'入力合計は重複を除いて{summary["tokens_distinct"]} tokens、各proposalを個別に実行する場合{summary["tokens_without_dedup"]} tokens。最大{summary["max_input_tokens"]}。質問・選択肢・chat特殊記号込み。Toolのみ・除外はモデル入力0。', '',
        '128を超える入力は旧新の数値をすべて保つ圧縮を試し、まだ超える場合はhold。拒否の追加数値閾値は無効。資金移動・mint・generic callの証拠不足はToolでhold。対応範囲は数値上の参加条件で、安全性全般の採否を保証しない。', '',
        '閾値は未校正のA/Bスコアで正解確率ではない。入力整形・token化・進行制御・hold判定はPython。モデル推論は既存ローカルcanisterをqueryし、canister更新なし。32層・Wasm/モデル/入力hash・query数・replay/fallbackなしを照合。', '']
    if prepared.get('compact_overflow_enabled'):
        lines += ['今回の追加実験は予算超過入力だけ単位をまとめる形式に変更。全変更項目の旧新数値は保持し、quiet延長の整数日をexact表記。同じ参加方針を短く表記。IDや参照ラベルはencoderに渡さない。4種類は前回とtoken列・選択肢が完全一致したため検証済み実測を再利用し、600・601を新規推論。入力表現が変わった調整実験であり、新規独立検証ではない。', '']
    if prepared.get('compact_ratio_enabled'):
        lines += ['ratio_words版では最低投票lockの倍率（その項目がなければstake倍率）をコードで計算して明示。丸い百・千の数値を英語で正確に表記し、選択肢はA approve B rejectへ変更。意味上の二択・閾値・承認証拠検査は同一。ID別の拒否規則はない。複数の入力形式を同じ既知proposalで試した調整実験で、prompt/倍率説明/選択肢表記の個別寄与は未分離。', '']
    elif prepared.get('compact_overflow_enabled'):
        lines += ['unit_grouped_overflow版は旧新値から再計算できる倍率説明を省略する。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    shutil.copyfile(__file__,out/'report_source.py')
    save(out/'files-sha256.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='files-sha256.json'})
    print(json.dumps(summary))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','report']);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compact-overflow',action='store_true')
    parser.add_argument('--compact-ratio',action='store_true')
    parser.add_argument('--snapshot-archive',type=existing_directory,
                        help='prepare source archive containing manifest.json and snapshots/')
    args=parser.parse_args();out=args.output.resolve()
    if args.mode=='prepare':prepare(out,args.compact_overflow or args.compact_ratio,args.compact_ratio,
                                   snapshot_archive=args.snapshot_archive)
    elif args.mode=='run':run(out,0,1)
    else:report(out)


if __name__=='__main__':main()
