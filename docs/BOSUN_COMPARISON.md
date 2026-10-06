# Bosun 3.1 1.7Bと現在のImajev: 同じ短文100問と追加評価

固定した同じ短文100問でBosun 46/100、Imajev 70/100。今回はBosunへの置き換えによる精度改善を確認できなかった。

| 課題（各20問） | Bosun | 現在のImajev |
|---|---:|---:|
| 文の穴埋め | 12/20 | 14/20 |
| 文章の続き | 10/20 | 15/20 |
| ユーモア比較 | 8/20 | 9/20 |
| 皮肉の有無 | 11/20 | 13/20 |
| 2文から皮肉を選ぶ | 5/20 | 19/20 |

Imajevの誤答をBosunが直した問題は10問、Imajevの正答をBosunが間違えた問題は34問。両方正解36問、両方誤答20問。unknownは誤答として扱った。対応ありのMcNemar exact p=0.000388はこの固定100問に対する探索的な値。広い課題全体の優劣は示さない。

## 追加の実用途評価

既存の正解付き24問でBosun 17/24、Imajev 22/24。24問の中にunknownが正解の8問を含み、別の8問として重複加算しない。

Bosunは回答可能な16問中11問正解、unknownが正解の8問中6問正解。Imajevはそれぞれ14/16、8/8。Bosunの回答可能な問題でのunknown選択は0件。

さらに8問の選択肢順入れ替えを実行。追加32通り全体ではBosun 21/32、Imajev 28/32。順入れ替えでBosunの回答が変わったのは1件。独立した32問の精度とは扱わない。

## 実行と確認

Bosun revision `1d8dc82a20e4a32ed60927a47272d6efff48eed2`、Qwen/Qwen3-1.7B base revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。MacのMetal GPUでBF16ベース＋公開PEFT adapter、公開BosunForDecision.predictを使用。1提示順、追加学習・生成・入力切り詰めなし。重み・tokenizer・コード・入力・Imajevの各実測reportのSHA256を検証した。

state、質問、元の選択肢の値・説明を保持した。Imajevと同じくbare unknownを明示的な選択肢として追加し、unknownの予測IDを__unknown__に対応させた。Bosunのnative predictは提示された通常slotの候補だけをsoftmaxするため、予約されたnull slotを独自に有効化していない。これはunknownを含めた比較用の入力調整であり、公開DecisionBenchそのままの設定ではない。

Bosun標準rendererの選択肢shuffleには、元の順番を保持する最初のseedを各入力で選んだ。seedの選択は正解・予測を参照しない。入力を全132通り固定してから推論し、candidate_to_slotとsoftmaxを独立に検証した。

各課題の先頭1問、計5問をCPU FP32で再実行し、5問ともBF16 GPUと同じ回答。確率の最大絶対差は0.0094。これは全100問の数値精度一致を証明するものではない。

unknown追加の影響を調べるため、元の選択肢だけの100問もBosunで新規実行した。正解50/100。主比較の入力を変更・差し替えた数字ではなく、別条件の診断値。元の選択肢だけでも今回の低めの結果は変わらなかった。

主比較132通り＋CPU確認5問＋元の選択肢のみ100問、合計237回のnative推論。Bosunのcanisterへの移植・query回数・cycles・通信量は未測定。

## 入力長と解釈の限界

同じ短文100問の入力トークン中央値: Bosun 136、Imajev 90。Bosunは114〜171トークン。tokenizerと標準promptが違うため、トークン数の差をそのまま計算量比として解釈しない。

Bosunは未量子化BF16のローカル推論、Imajevは短いpromptを使うINT8 canister推論。これは現在のシステムと公開Bosun推論の比較であり、同じ数値精度・promptでbackboneだけを比較したものではない。

短い入力への選択偏り、各課題20問、皮肉の2課題が全体40%を占める構成を維持した。HellaSwagは全10,042問中60問だけが短文条件に合い、その中の20問。24問の実用途評価は既存の開発用の小さな固定セットで、独立した広域ベンチではない。総合Decision Indexや全データセットの性能に一般化しない。詳細は[短文100問の選択条件](DECISION_INDEX_SHORT_SUBSET.md)。

Bosunの公開DecisionBench 84.9%は別の課題集合の成績。同ベンチに含まれる課題系統を学習していると作者が明記しており、今回の結果との不一致は公開値の誤りを示すものではない。[Bosunモデルカード](https://huggingface.co/Hanno-Labs/bosun-v3.1-1.7b)。

## 保存先

- `artifacts/bosun-short-v1/inputs.json`: 固定132通り、元問題の対応、正解、baseline、native promptとtoken IDs。
- `artifacts/bosun-short-v1/evaluation-session.json`: revision、数値精度、パッケージ版、全モデルファイルのhash。
- `artifacts/bosun-short-v1/report.json`: 集計、問題ごとの予測、確率、改善・悪化の対応。
- `artifacts/bosun-short-v1/verification.json`: 237回の推論と検証。
- `artifacts/bosun-short-v1/runs/`: native推論のprompt、確率、各実測記録。
- `artifacts/bosun-short-v1/cpu-check/`、`closed-set-check/`: 別条件の確認記録。

再集計・検証・文書生成: `.venv/bin/python scripts/report_bosun_comparison.py`。推論は別環境 `artifacts/bosun-short-v1/env`、固定済みモデルをオフラインで使用。
