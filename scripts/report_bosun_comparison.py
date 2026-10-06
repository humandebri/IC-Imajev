#!/usr/bin/env python3
"""Verify and report the completed native Bosun comparison."""
import json
import math
import pathlib

from evaluate_bosun_local import DIRECTORY, ROOT, atomic, digest, read


def main():
    report = read(DIRECTORY / 'report.json')
    inputs = read(DIRECTORY / 'inputs.json')
    session = read(DIRECTORY / 'evaluation-session.json')
    assert report['complete'] and report['completed'] == 132
    assert report['verification']['verified']
    assert all(digest(ROOT / p) == h for p, h in inputs['source_hashes'].items())
    assert all(digest(ROOT / p) == h for p, h in session['source_hashes'].items())
    for row, frozen in zip(report['cases'], inputs['records'], strict=True):
        assert row['index'] == frozen['index']
        assert digest(ROOT / frozen['imajev_report']) == frozen['imajev_report_sha256']
        path = DIRECTORY / 'runs' / f"{row['index']:03d}.json"
        assert digest(path) == row['report_sha256']
        result = read(path)['output']
        assert result['prompt'] == frozen['prompt']
        count = len(frozen['candidates'])
        assert result['candidate_to_slot'] == {c['id']: i for i, c in enumerate(frozen['candidates'])}
        logits = result['slot_logits']
        exps = [math.exp(v - max(logits)) for v in logits]
        normalized = [v / sum(exps) for v in exps]
        assert all(abs(a-b) < 1e-6 for a,b in zip(normalized, row['probabilities'], strict=True))
        assert len(row['probabilities']) == count
        prediction = frozen['candidates'][max(range(count), key=row['probabilities'].__getitem__)]['id']
        assert prediction == row['bosun_prediction']
        assert row['bosun_correct'] == (prediction == frozen['gold'])
        assert row['imajev_correct'] == (row['imajev_prediction'] == frozen['gold'])
    cpu = read(DIRECTORY / 'cpu-check/report.json')
    closed = read(DIRECTORY / 'closed-set-check/report.json')
    assert cpu['native_inferences'] == cpu['same_predictions'] == 5
    assert closed['native_inferences'] == 100
    assert cpu['source_inputs_sha256'] == closed['source_inputs_sha256'] == digest(DIRECTORY / 'inputs.json')
    report['diagnostics'] = dict(
        cpu_float32=dict(samples=5, same_predictions=5, max_probability_difference=cpu['max_probability_difference']),
        no_added_unknown=dict(native_inferences=100, correct=closed['correct'], by_dataset=closed['by_dataset']))
    report['verification'].update(independent_probability_readout_verified=True,
                                  imajev_individual_report_hashes_verified=True,
                                  cpu_float32_checks=5, closed_set_native_inferences=100,
                                  total_native_inferences_including_diagnostics=237)
    atomic(DIRECTORY / 'report.json', report)
    atomic(DIRECTORY / 'verification.json', report['verification'])
    names = {'WinoGrande':'文の穴埋め', 'HellaSwag':'文章の続き', 'Humicroedit':'ユーモア比較',
             'iSarcasmEval-A-En':'皮肉の有無', 'iSarcasmEval-C-En':'2文から皮肉を選ぶ'}
    short = report['short100']; extra = report['curated24']
    lines = ['# Bosun 3.1 1.7Bと現在のImajev: 同じ短文100問と追加評価', '',
             f"固定した同じ短文100問でBosun {short['bosun_correct']}/100、Imajev {short['imajev_correct']}/100。今回はBosunへの置き換えによる精度改善を確認できなかった。", '',
             '| 課題（各20問） | Bosun | 現在のImajev |', '|---|---:|---:|']
    for family, m in report['by_dataset'].items():
        lines.append(f"| {names[family]} | {m['bosun_correct']}/20 | {m['imajev_correct']}/20 |")
    lines += ['', f"Imajevの誤答をBosunが直した問題は{short['imajev_errors_fixed']}問、Imajevの正答をBosunが間違えた問題は{short['imajev_correct_regressed']}問。両方正解{short['both_correct']}問、両方誤答{short['both_wrong']}問。unknownは誤答として扱った。対応ありのMcNemar exact p={short['paired_mcnemar_exact_p']:.6f}はこの固定100問に対する探索的な値。広い課題全体の優劣は示さない。", '',
              '## 追加の実用途評価', '',
              f"既存の正解付き24問でBosun {extra['bosun_correct']}/24、Imajev {extra['imajev_correct']}/24。24問の中にunknownが正解の8問を含み、別の8問として重複加算しない。", '',
              f"Bosunは回答可能な16問中{extra['known_correct']}問正解、unknownが正解の8問中{extra['unknown_correct']}問正解。Imajevはそれぞれ14/16、8/8。Bosunの回答可能な問題でのunknown選択は{extra['false_abstentions']}件。", '',
              f"さらに8問の選択肢順入れ替えを実行。追加32通り全体ではBosun {report['curated32_orders']['bosun_correct']}/32、Imajev {report['curated32_orders']['imajev_correct']}/32。順入れ替えでBosunの回答が変わったのは{len(report['order_changes'])}件。独立した32問の精度とは扱わない。", '',
              '## 実行と確認', '',
              f"Bosun revision `{session['model_revision']}`、Qwen/Qwen3-1.7B base revision `{session['base_revision']}`。MacのMetal GPUでBF16ベース＋公開PEFT adapter、公開BosunForDecision.predictを使用。1提示順、追加学習・生成・入力切り詰めなし。重み・tokenizer・コード・入力・Imajevの各実測reportのSHA256を検証した。", '',
              'state、質問、元の選択肢の値・説明を保持した。Imajevと同じくbare unknownを明示的な選択肢として追加し、unknownの予測IDを__unknown__に対応させた。Bosunのnative predictは提示された通常slotの候補だけをsoftmaxするため、予約されたnull slotを独自に有効化していない。これはunknownを含めた比較用の入力調整であり、公開DecisionBenchそのままの設定ではない。', '',
              'Bosun標準rendererの選択肢shuffleには、元の順番を保持する最初のseedを各入力で選んだ。seedの選択は正解・予測を参照しない。入力を全132通り固定してから推論し、candidate_to_slotとsoftmaxを独立に検証した。', '',
              f"各課題の先頭1問、計5問をCPU FP32で再実行し、5問ともBF16 GPUと同じ回答。確率の最大絶対差は{cpu['max_probability_difference']:.4f}。これは全100問の数値精度一致を証明するものではない。", '',
              f"unknown追加の影響を調べるため、元の選択肢だけの100問もBosunで新規実行した。正解{closed['correct']}/100。主比較の入力を変更・差し替えた数字ではなく、別条件の診断値。元の選択肢だけでも今回の低めの結果は変わらなかった。", '',
              f"主比較132通り＋CPU確認5問＋元の選択肢のみ100問、合計237回のnative推論。Bosunのcanisterへの移植・query回数・cycles・通信量は未測定。", '',
              '## 入力長と解釈の限界', '',
              f"同じ短文100問の入力トークン中央値: Bosun {short['bosun_median_tokens']:g}、Imajev {short['imajev_median_tokens']:g}。Bosunは{short['bosun_min_tokens']}〜{short['bosun_max_tokens']}トークン。tokenizerと標準promptが違うため、トークン数の差をそのまま計算量比として解釈しない。", '',
              'Bosunは未量子化BF16のローカル推論、Imajevは短いpromptを使うINT8 canister推論。これは現在のシステムと公開Bosun推論の比較であり、同じ数値精度・promptでbackboneだけを比較したものではない。', '',
              '短い入力への選択偏り、各課題20問、皮肉の2課題が全体40%を占める構成を維持した。HellaSwagは全10,042問中60問だけが短文条件に合い、その中の20問。24問の実用途評価は既存の開発用の小さな固定セットで、独立した広域ベンチではない。総合Decision Indexや全データセットの性能に一般化しない。詳細は[短文100問の選択条件](DECISION_INDEX_SHORT_SUBSET.md)。', '',
              'Bosunの公開DecisionBench 84.9%は別の課題集合の成績。同ベンチに含まれる課題系統を学習していると作者が明記しており、今回の結果との不一致は公開値の誤りを示すものではない。[Bosunモデルカード](https://huggingface.co/Hanno-Labs/bosun-v3.1-1.7b)。', '',
              '## 保存先', '',
              '- `artifacts/bosun-short-v1/inputs.json`: 固定132通り、元問題の対応、正解、baseline、native promptとtoken IDs。',
              '- `artifacts/bosun-short-v1/evaluation-session.json`: revision、数値精度、パッケージ版、全モデルファイルのhash。',
              '- `artifacts/bosun-short-v1/report.json`: 集計、問題ごとの予測、確率、改善・悪化の対応。',
              '- `artifacts/bosun-short-v1/verification.json`: 237回の推論と検証。',
              '- `artifacts/bosun-short-v1/runs/`: native推論のprompt、確率、各実測記録。',
              '- `artifacts/bosun-short-v1/cpu-check/`、`closed-set-check/`: 別条件の確認記録。', '',
              '再集計・検証・文書生成: `.venv/bin/python scripts/report_bosun_comparison.py`。推論は別環境 `artifacts/bosun-short-v1/env`、固定済みモデルをオフラインで使用。']
    (ROOT / 'docs/BOSUN_COMPARISON.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(short100=short, curated24=extra, diagnostics=report['diagnostics']), ensure_ascii=False))


if __name__ == '__main__':
    main()
