#!/usr/bin/env python3
"""Write the verified short-input partial evaluation report."""
import json,pathlib
import evaluate_prompt_accuracy as e
R=e.ROOT;D=R/'artifacts/decision-index-short-v1'

def main():
    result=json.loads((D/'report.json').read_text());fixture=json.loads((D/'inputs.json').read_text())
    assert result['complete'] and result['completed']==100 and result['verification']['verified']
    names={'WinoGrande':'WinoGrande（文の穴埋め）','HellaSwag':'HellaSwag（文章の続き）','Humicroedit':'Humicroedit（見出しのユーモア比較）',
           'iSarcasmEval-A-En':'iSarcasmEval A-En（皮肉の有無）','iSarcasmEval-C-En':'iSarcasmEval C-En（皮肉の文章を選ぶ）'}
    lines=['# Decision Index由来の短文100問: 現在のImajev canister','',
           f"全100問のcanister推論が完了。正解 {result['correct']}/100（{result['accuracy']:.1%}）、unknown {result['unknown']}件。入力・モデル・確率・正解対応を検証済み。",'',
           '| タスク | 正解 | 正解率 | 多数派を常に選ぶ基準 | query数の中央値 | Candid通信量の中央値 |','|---|---:|---:|---:|---:|---:|']
    for family,m in result['by_dataset'].items():
        lines.append(f"| {names[family]} | {m['correct']}/20 | {m['accuracy']:.0%} | {m['majority_accuracy']:.0%} | {m['median_queries']:g} | {m['median_candid_bytes']/1e6:.1f} MB |")
    sarcastic=result['by_dataset']['iSarcasmEval-A-En']
    lines+=['','今回の抽出では、穴埋め・文章の続き・2文から皮肉を選ぶ課題は多数派基準を上回った。ユーモア比較と単独文の皮肉判定は多数派基準を下回った。高い結果が出たタスクも各20問に限られる。unknownは誤答として正解率に含めた。','', '## 評価範囲と偏り','',
            'これは[Decision Index再現キット](https://github.com/apolinario/decision-index)の公開ソースから作った部分評価。4ベンチマークの5タスクを各20問として扱い、iSarcasmEvalの2タスクで40%を占める。公式Decision Index全体の重み・チャンス補正・全件の公式スコアは再現していない。','',
            '共通prefixを除いた追加トークンが87以下になる問題だけを候補にした。質問・state・選択肢は切り詰めずに保持した。短い言語判断、皮肉、ユーモアに偏る。長文の推論、検索、ツール選択、画像理解の性能は評価していない。','',
            '特にHellaSwagでは10,042問のうち60問（約0.6%）だけが短文条件に合い、その中から20問を選んだ。この正解率をHellaSwag全体の性能として解釈してはいけない。','',
            f"皮肉の有無は正解ラベルがno=17、yes=3。常にnoと答えるだけで85%正解になる。現在モデルの皮肉クラスF1は {sarcastic['sarcastic_class_f1']:.3f}、正解率は {sarcastic['accuracy']:.1%}。各タスクのクラス分布・macro-F1・多数派基準はreport.jsonにも保存した。",'',
            'Clef Flashには同じ100問を実行していない。Cloudflare掲載の全ベンチスコアと、この短文抽出の数字を直接比較して両モデルの性能差を断定しない。','',
            '## 入力の固定','',
            f"seed=20261005。各タスクのgroup_idをSHA256(seed:family:group_id)で並べ、トークン条件を満たす先頭20問を推論前に固定した。予測結果や正解ラベルで選んでいない。元の問題・選択肢・正解・出典はselected-source-rows.jsonlに保存。再現キットcommitは `{fixture['official_kit_commit']}`。",'',
            'Humicroeditの同点問題は公式の正規化段階で除外される。元2,960問のうち非同点2,628問を短文選択の候補とした。その他の選択・除外件数はinputs.jsonのinventoryに記録している。','',
            '## 実行条件','',
            f"MODEL_LOCK.jsonで固定したImajev 4B。INT8ベース、F32 LoRA/readout、BF16/F32計算境界、text-only-short-v2、1提示順。canisterは6eydd-o3777-77775-aaama-cai、module SHA256は `{e.MODULE}`。",'',
            '入力69〜114トークン。stateの形式に応じて26または27トークンの共有prefixを使う。追加69トークン以下では動的なpacked経路、70〜87では既存の短文経路を使う。全問を新たに実行し、保存済みの別問の推論結果で代用していない。', '',
            f"合計query数 {result['execution']['total_queries']:,}、中央値 {result['execution']['median_queries']:g}、最小 {result['execution']['min_queries']}、最大 {result['execution']['max_queries']}。通信量の中央値 {result['execution']['median_candid_bytes']/1e6:.1f} MB。fallback {result['execution']['fallbacks']}、checkpointの再利用query {result['execution']['replayed_queries']}。",'',
            'query数・通信量は推論本体のもの。共有prefixの初回準備、module_hashなどの検証callは含まない。通信量はCandidのrequest+reply、decimal MB。HTTP/CBOR/署名は含まない。命令数はhandler内部で計測し、Candidのdecode/encodeは含まない。ローカルの時間値はメモリ負荷や並列実行の影響があり、本番の遅延やClefのGPU実行時間と比較しない。','',
            '元の長文を含む100問の計画は、ユーザーの指示で途中停止した。48問の途中結果はartifacts/decision-index-v1/partial-report.jsonに残した。その数字を今回の短文100問の集計に混ぜていない。元の評価と入力が一致する問題では、今回の新規推論の確率が前回とビット一致することも確認した。','',
            '## 保存先と再実行','',
            '- `artifacts/decision-index-short-v1/inputs.json`: 固定入力・抽出条件・ハッシュ。',
            '- `artifacts/decision-index-short-v1/selected-source-rows.jsonl`: 元の100問と出典・正解。',
            '- `artifacts/decision-index-short-v1/report.json`: 集計、各問の予測・確率・計測値。',
            '- `artifacts/decision-index-short-v1/verification.json`: 100問の実測と型・ハッシュ・命令上限の検証。',
            '- `artifacts/decision-index-short-v1/runs/`: 実測ログとquery journal。','',
            '準備済みの同じ入力を続行・再集計するコマンド:','',
            '```sh','.venv/bin/python scripts/benchmark_decision_index_short.py --workers 2',
            '.venv/bin/python scripts/verify_decision_index_short.py',
            '.venv/bin/python scripts/report_decision_index_short.py','```','',
            '既にある同じ実測reportはそのまま使い、新規の独立実行を測り直したい場合は新しい出力ディレクトリのセッションを用意する。取得データはローカルartifacts内に留め、外部へアップロードしていない。','']
    (R/'docs/DECISION_INDEX_SHORT_SUBSET.md').write_text('\n'.join(lines))
    print(json.dumps(dict(correct=result['correct'],accuracy=result['accuracy'],unknown=result['unknown'],by_dataset=result['by_dataset'],execution=result['execution']),ensure_ascii=False))

if __name__=='__main__':main()
