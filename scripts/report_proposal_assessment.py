#!/usr/bin/env python3
"""Produce a compact report from the completed real-canister benchmark."""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/proposal-assessment-canister-20261005'
r = json.loads((D / 'report.json').read_text())
assert r['complete'] and r['completed_canister_inferences'] == r['runnable_unique_tasks']
text = json.loads((D / 'text-comparison.json').read_text())
prepared = json.loads((D / 'prepared.json').read_text())
aborted = []
for path in (D / 'aborted').glob('*/queries/*.metric.json'):
    metric = json.loads(path.read_text())
    if not metric.get('replayed'):
        aborted.append(metric)
stats = r['text_evaluation']['models']['imajev-canister-int8']
unique = r['unique_text_evaluation']['models']['imajev-canister-int8']
words = {'approve': '承認', 'reject': '否決', 'hold': '保留', 'unavailable': '実行不可/棄権'}
counts = r['vote_counts']
sys.path.insert(0, '/Volumes/KINGSTON/ICP/IC-Laya-Standalone/tools')
from proposal_assessment.evaluation import evaluate
source = Path('/Volumes/KINGSTON/ICP/IC-Laya-Standalone/artifacts/proposal-assessment/boomdao-600-660/v1')
reference_labels = json.loads((source / 'reviewed-text-labels.json').read_text())
eligible_labels = {'schema_version': 1, 'snapshots': {
    e['snapshot_sha256']: reference_labels['snapshots'][e['snapshot_sha256']]
    for e in prepared['proposals'] if int(e['proposal_id']) != 660}}
prior = json.loads((source / 'comparison.json').read_text())
prior_eligible = evaluate(prior, eligible_labels)['models']
current_eligible = evaluate(text, eligible_labels)['models']['imajev-canister-int8']
lines = [
    '# Canister上のImajev：proposal_assessmentベンチ', '',
    '2026-10-05（日本時間）。IC-Laya-Standalone/tools/proposal_assessment のBOOM DAO 600〜660全61件を、稼働中のローカルImajev canisterで評価した。GPT APIの予測は使用していない。', '',
    f'接続先は `{r["model"]["url"]}`、canisterは `{r["model"]["canister"]}`。Wasm hashは `{r["model"]["wasm_sha256"]}`。基盤は固定Qwen3.5-4B＋Imajev adapter、INT8 base・F32 LoRA/readout・BF16/F32中間演算。', '',
    f'本文説明3問×61件と採否1問×61件の計244問。task JSONが完全一致するものをまとめると44種類で、うち30種類を通常queryによる全32層推論で実行した。14種類は512-token入力上限により拒否した。要約・切り詰め・正解ラベルによる入力修正はしていない。', '',
    '## 本文説明', '',
    '| 項目 | 全183問 | 文面重複を除く30問 |', '|---|---:|---:|',
    f'| 選択肢を回答できた数 | {stats["predicted"]}/{stats["expected"]} | {unique["predicted"]}/{unique["expected"]} |',
    f'| 暫定ラベルとの一致 | {stats["correct"]}/{stats["expected"]}（{stats["correct_over_all_labeled_tasks"]:.2%}） | {unique["correct"]}/{unique["expected"]}（{unique["correct_over_all_labeled_tasks"]:.2%}） |',
    f'| 回答できた問の一致率 | {stats["accuracy_on_predictions"]:.2%} | {unique["accuracy_on_predictions"]:.2%} |', '',
    '参照ラベルは既存のassistantによる暫定ラベル。独立した人間の正解検証を受けていない。元のLaya実測（183問）はtyped-decisions 9/183、english 20/183、multilingual 13/183。常にabsentの基準は173/183。今回の入力形式と上限・unknown候補はLayaと異なるため、同条件の一般性能順位ではない。', '',
    '今回の入力上限に収まる180問（660を除く）に参照ラベルを揃えた比較：', '',
    '| モデル・基準 | 一致/180問 |', '|---|---:|',
    f'| Imajev canister | {current_eligible["correct"]}/180 |',
    *[f'| Laya {name}（既存実測） | {s["correct"]}/180 |' for name, s in prior_eligible.items()],
    '| 常にabsent | 173/180 |', '',
    '## 承認・否決・保留', '', '| 選択 | 件数 |', '|---|---:|',
]
for label in ('approve', 'reject', 'hold', 'unavailable'):
    lines.append(f'| {words[label]} | {counts.get(label, 0)} |')
lines += ['', '512 tokensを超える採否入力は、根拠や判定方針を削らず実行不可とした。Imajevのreserved unknownによる棄権も、holdやunclearへ変換せず別記する。採否の正解ラベルがないため正答率は測定していない。', '',
          '602と同じ入力の16件は、最低投票ロック期間を1日から2日に引き上げる提案で、本文は「SNS parameters adjustment」の定型文。影響するneuron数は不明である。モデルはこの入力へapproveと回答したが、理由を生成しないため、どの材料を採否の根拠にしたかは分からない。', '',
          '## 実行量', '',
          f'- 完了した推論：{r["completed_canister_inferences"]}回、その通常query {r["total_executed_queries"]:,}回。',
          f'- Handler命令：{r["total_instructions"]:,}。',
          f'- Candid request＋reply：{r["total_candid_bytes"]:,} bytes。HTTP・署名等は含まない。',
          f'- 各実行のend-to-end時間の合計：{sum(x["seconds"] for x in r["runs"]):.1f}秒。prefix準備・モデル重み準備・後処理は含まない。',
          '- 既存の26-token共通prefix状態を再利用。推論の中間状態はclient-held。各推論の入力依存の数値計算はcanister上で実行した。',
          '- 各実行でmodule hash・token hash・typed response・全queryの実行数を検査した。実行済み応答のjournal replayは0。',
          f'- 並行実行を打ち切った補助実行は採点と上記合計に含まない。その完了query metricは{len(aborted)}件をaborted/に保存し、同じtaskを後で最初から実行した。',
          '- 同じcanister上の別ベンチと一時的に競合したため、所要時間にはその影響がある。一般的な推論速度やLayaとの速度比較には使用しない。',
          '- 重みのupload・prepare・canisterのupgrade・投票は行っていない。', '',
          '## 全61件', '', '| ID | action | 理由説明 | 影響利用者 | 移行・緩和 | 採否 |', '|---|---|---|---|---|---|']
votes = {x['proposal_id']: x for x in r['votes']}
for e in text['proposals']:
    pid = int(e['facts']['proposal_id'])
    rows = {x['task_id']: x for x in e['model_assessments'][0]['advisory']}
    labels = [rows[t].get('label', 'unavailable') for t in ('rationale','affected_users','mitigation')]
    vote = votes[pid]['prediction'].get('label','unavailable')
    lines.append(f'| {pid} | {e["facts"]["action"]} | {labels[0]} | {labels[1]} | {labels[2]} | {words[vote]} |')
lines += ['', '## 入力形式と保存', '',
          r['model']['adaptation'], '',
          '質問・instructionと選択肢の説明を保持した。Imajevの固定short promptと最後のunknown候補で符号化するため、Layaのclassifier用promptとtoken数は同一ではない。採点ラベル・過去モデルの予測・採決結果は推論へ渡さない。', '',
          '同一taskは1回だけ推論して該当proposalへ展開した。244回の独立反復ではなく、反復分散や選択肢順序の感度も測っていない。', '',
          '生の確率・logits・unknown・token数・全queryの計測は[report.json](report.json)、本文の暫定ラベル比較は[text-comparison.json](text-comparison.json)。個別実行のcommandとrun.log、report、query metricをruns/に保存した。成功した実行の中間binary stateはhash・byte数を記録して削除し、ディスク使用量を抑えた。', '',
          '再現コマンド：', '', '```sh',
          '.venv/bin/python -B scripts/benchmark_proposal_assessment.py --mode prepare --directory artifacts/proposal-assessment-new-run',
          '.venv/bin/python -B scripts/benchmark_proposal_assessment.py --mode run --directory artifacts/proposal-assessment-new-run',
          '```', '',
          '## 上限・棄権・失敗の一覧', '']
for key, item in prepared['tasks'].items():
    ans = r['task_results'][key]
    if ans['status'] != 'model_prediction':
        lines.append(f'- proposal {", ".join(map(str,item["proposal_ids"]))} / {item["task"]["task_id"]}：{ans["reason"]}')
lines += ['']
(D / 'REPORT.md').write_text('\n'.join(lines))
print(D / 'REPORT.md')
