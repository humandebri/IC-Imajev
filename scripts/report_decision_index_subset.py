#!/usr/bin/env python3
"""Finalize the partial benchmark only after every frozen input has a report."""
import collections,json,statistics
import evaluate_decision_index_subset as run
import evaluate_prompt_accuracy as e
R=e.ROOT;D=R/'artifacts/decision-index-v1'

def main():
    result=run.summarize()
    assert result['completed']==100 and not result['errors'],'100 valid reports are required'
    assert json.loads((D/'verification.json').read_text())['verified']
    for family,metrics in result['by_dataset'].items():
        rows=[r for r in result['cases'] if r['dataset']==family]
        metrics.update(median_queries=statistics.median(r['query_count'] for r in rows),
                       median_candid_bytes=statistics.median(r['total_candid_bytes'] for r in rows),
                       min_tokens=min(r['tokens'] for r in rows),max_tokens=max(r['tokens'] for r in rows))
    routes=collections.defaultdict(list)
    diagnostic_metrics=[]
    for row in result['cases']:
        n=row['tokens']-26
        routes['logged_1_90' if n<=90 else 'dense_fused_91_132' if n<=132 else 'dense_chunked_133_512'].append(row)
        directory=D/'runs'/f'{row["index"]:03d}'
        if (directory/'large/report.json').exists():
            # Earlier route discovery stopped at a client bound; these completed
            # initial queries do not belong to the final forward pass.
            for path in sorted((directory/'queries').glob('*.metric.json')):
                metric=json.loads(path.read_text())
                diagnostic_metrics.append(dict(case=row['index'],path=str(path.relative_to(R)),instructions=metric['ok']['instructions'],candid_bytes=metric['ok']['request_bytes']+metric['ok']['reply_bytes']))
    result['routes']={k:dict(total=len(v),median_queries=statistics.median(r['query_count'] for r in v),
                            median_candid_bytes=statistics.median(r['total_candid_bytes'] for r in v)) for k,v in routes.items()}
    result['route_discovery_overhead']=dict(completed_queries=len(diagnostic_metrics),measured_instructions=sum(r['instructions'] for r in diagnostic_metrics),
                                         measured_candid_bytes=sum(r['candid_bytes'] for r in diagnostic_metrics),queries=diagnostic_metrics,
                                         note='Excludes module-hash checks and any in-flight call interrupted when concurrency was changed')
    result['limitations']=[
        'Fixed 20 questions from each of five datasets; no full Decision Index score or Clef head-to-head claim.',
        'VAST is reported as subset accuracy, not the official full-dataset macro-F1.',
        'One presentation order; current locked INT8 canister and short prompt; source questions and all criteria are preserved.',
        'Inference query and byte counts exclude prefix preparation and module-hash verification; Candid bytes exclude HTTP/CBOR/signatures.',
        'Some exact checkpoints were resumed after adjusting concurrency; query_count includes saved computations from this same input.',
        'Wall time is affected by local memory pressure, parallel execution and checkpoint resumes; not a production latency comparison.',
    ]
    e.atomic_json(D/'report.json',result)
    lines=['# Decision Index: current canister, fixed 100-question subset','',
           f"正解 {result['correct']}/100（{result['accuracy']:.1%}）。unknown {result['unknown']}件。全100問のcanister推論が完了し、入力・モデル・出力・正解対応を検証済み。",'',
           '| データセット | 正解 | 部分評価の正解率 | query数の中央値 | Candid通信量の中央値 |','|---|---:|---:|---:|---:|']
    for family,m in result['by_dataset'].items():
        lines.append(f"| {family} | {m['correct']}/20 | {m['accuracy']:.0%} | {m['median_queries']:g} | {m['median_candid_bytes']/1e6:.1f} MB |")
    lines+=['','元データは公開の[Decision Index再現キット](https://github.com/apolinario/decision-index)の固定版取得・正規化・抽出処理で準備。seed=20261005、各データセットでgroup_idのSHA256順に先頭20適格問を推論前に固定した。CLadderは公式5,000件の抽出後に選択。入力81〜329トークン、全選択肢と元のstateを保持。今回の100問抽出では長さや選択肢による除外は発生していない。','',
            '## 解釈','',
            'これは5データセットの部分評価。公式Decision Indexは多数のデータセットと独自の重み・チャンス補正を使うため、その公式スコアとは比較しない。VASTの表はaccuracyであり、公式全件macro-F1とは異なる。Clef Flashには同じ100問を実行していないため、両モデルの精度差は確定できない。','',
            '## 実行条件','',
            f"モデルはMODEL_LOCK.jsonの固定Imajev 4B。INT8ベース、F32 LoRA/readout、BF16/F32計算境界、短縮prompt、1提示順。canister module SHA256: `{e.MODULE}`。",'',
            'stateが空オブジェクトなので共有prefixは26トークン。ログ形式と通常の密なstate形式のprefixを別に用意し、全32層のhiddenがビット一致することを確認。追加トークン数に応じて、20問はログ形式の短文経路、19問は通常形式の結合経路、61問は通常形式のトークン分割経路を使った。モデルの重み・質問・選択肢は同一条件。','',
            '最初の経路検査で90/132トークンのクライアント上限を検出し、別の既存経路へ切り替えた。診断時の追加計算はreport.jsonのroute_discovery_overheadに別記。ローカルのメモリ負荷に応じて並列数を変更し、入力ハッシュが一致する途中結果から再開した。報告query数にはその入力で実行済みの計算を含む。','',
            '通信量はCandidのrequest+replyの合計（decimal MB）。HTTP/CBOR/署名、共有prefixの初回準備、module_hash等の検証callは含まない。命令数はhandler内部の計測で、Candid decode/encodeは含まない。ローカルの時間値はメモリ負荷・並列実行・再開の影響を受けるため、本番の遅延やClef GPU実行時間との比較には使わない。','',
            '## 保存先','',
            '- `artifacts/decision-index-v1/inputs.json`: 固定した入力と抽出条件。',
            '- `artifacts/decision-index-v1/selected-source-rows.jsonl`: 元の100問と正解・出典。',
            '- `artifacts/decision-index-v1/report.json`: 集計と各問の予測・query数・通信量。',
            '- `artifacts/decision-index-v1/verification.json`: 入力とモデルのハッシュ、型、確率、正解対応、命令上限の検証。',
            '- `artifacts/decision-index-v1/runs/`: 実測ログ、query metrics、hidden/state、完了後に削除した中間frameのハッシュ。','',
            '取得データはローカルの評価用artifactsに保存し、外部へアップロードしていない。','']
    (R/'docs/DECISION_INDEX_SUBSET.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','route_discovery_overhead')},ensure_ascii=False))

if __name__=='__main__':main()
