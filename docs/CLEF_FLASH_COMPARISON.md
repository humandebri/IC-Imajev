# Clef Flash公開スコアと現在のImajevの短文部分評価

同じ100問での比較は未実施。以下は課題の種類を対応させた参考表であり、問題集合・件数・数値精度・推論方式が異なる。差を両モデルの性能差として解釈しない。

| 課題 | 今回のImajev（各20問） | Clef Flash公開スコア |
|---|---:|---:|
| WinoGrande | 70.0% | 97.5% |
| HellaSwag | 75.0% | 98.6% |
| Humicroedit | 45.0% | 75.1% |
| iSarcasmEval A-En（皮肉の有無） | 正解率65%、皮肉クラスF1 0.364 | 未掲載 |
| iSarcasmEval C-En（2文から皮肉を選ぶ） | 正解率95% | 未掲載 |

Imajev総合70%は今回の短文100問の生の正解率。Clef FlashのDecision Index 57.07は異なる課題群をチャンス補正・重み付けして集計した指標なので、70対57.07で優劣を判断できない。

Imajevは入力長で候補を絞っており、HellaSwagでは10,042問中60問（約0.6%）から20問を選んだ。その他の偏り・多数派基準は[今回の報告](DECISION_INDEX_SHORT_SUBSET.md)を参照。

Clef側の元データはCloudflareの公開leaderboard。generated_utc=2026-10-01T15:33:42+00:00、source=self-reported（提供元の報告値）。iSarcasmEval（id40）はmissing一覧にあり、0点とは扱わない。公開スコアは小数3桁の値を百分率で表示した。

出典: [Cloudflare公開leaderboard](https://clef-evals.workers-ai-mle.workers.dev/)・[公開JSON](https://clef-evals.workers-ai-mle.workers.dev/data/leaderboard.json)・[Clef Flashモデルカード](https://huggingface.co/Cloudflare/clef-flash)。取得したJSONをartifacts/decision-index-short-v1/comparison/clef-public-leaderboard.jsonに保存。

公平な直接比較には、今回固定したselected-source-rows.jsonlの同じ100問をClef Flashにも入力し、同じ選択肢・正解対応・unknownの採点規則で集計する必要がある。Clef固有の入力形式に変換しても、state・質問・選択肢の意味と情報は保つ。
