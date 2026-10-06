# Prompt短縮の追加精度評価

2026-10-05。元の公式standard promptと、現在の `text-only-short-v2` を実際のローカルcanisterで比較する。重み、INT8演算、F32 LoRA、readout、校正設定を固定する。現在の短縮版は画像/stateの非命令性の一文と情報不足時unknownの一文を削除し、unknownの説明を裸の `unknown` に変更している。モデル演算の結合だけの影響は別途bitwise parityで検証済みであり、この評価は入力prompt変更による回答の正誤を調べる。

## 評価設計

正解を明示できる24問：既存control7問、新規17問。一次集計は24問それぞれの最初の選択肢順序だけを数える。8問は別の選択肢順序でも評価し、合計32問・順序の組合せを両promptで実行する。推論回数は64で、それぞれrotations=1。複数順序の結果を投票・平均して1回答にするserving評価ではない。歴史的なリスク質問には一意の正解がないため一次正答率から外す。

- 正解あり16問、unknownが正解8問。
- yes6、no8、decreases1、treasury1、unknown8。
- 数値の増減、同じ値、最大値と最小値の区別、分単位/秒単位の一致、否定・burn/transferとmintの区別。
- 新旧いずれかの値の欠落、受取先の欠落、誤った前提、選択肢に正解がない、矛盾する値。
- state中の引用された回答指示2問も含める。goldは引用指示ではなく、stateに明示されたパラメータと質問から決める。

goldと理由は `benchmarks/prompt_accuracy_extra.json` と既存 `benchmarks/cases.json` に記録し、推論前に全入力とgoldを `artifacts/prompt-accuracy-v2/inputs.json` へ固定した。推論後に都合のよい問題を選び直さない。これは小さい手作業の診断セットで、実利用質問を無作為抽出した母集団精度の推定ではない。

## 実行・集計

実験用canister `6eydd-o3777-77775-aaama-cai`、module `6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931`。署名・module hashを各推論の前後で照合する。元の45-token prefixは同moduleで再準備し、以前の全32層のprefix stateとbitwise一致を確認。短縮版は既存27-token prefixを使用。prefixは質問を含まない固定部分で、保存した質問のhiddenや正解はモデルへの入力にしない。

実行は `scripts/evaluate_prompt_accuracy.py`。独立した後半のread-only queryを `evaluate_prompt_accuracy_worker.py` でも実行し、journalの担当範囲が重なる前にworkerを停止する。各推論は入力hash、module hash、出力の選択肢数・型を検査する。fallbackがあれば記録し、技術的な失敗と回答の誤りを区別する。最終結果には全64個の実際の推論reportへのpathとhashを保存する。途中の集計は `report_prompt_accuracy_progress.py` で、片方しか終わっていない組は回答変化に数えない。

一次正答率に加え、正解あり問題の正答数、unknown問題の棄権成功数、不要な棄権、情報不足なのに回答した件数、回答したときの正答数、正解候補の平均確率を集計する。選択肢の順序による回答変化は別項目で示し、24問の一次集計に順序違いを混ぜない。短縮によって正解→誤りとなったregressionと、誤り→正解のimprovementを全件列挙する。

集計コードの4検証では、不要な棄権と情報不足への断定を別の誤りとして数えること、選択肢順序違いを一次精度の分母へ追加しないこと、片方が未完了のpairを変化と判定しないこと、正解候補の確率を並べ替え後のoption valueから引くことを確認した。

## 結果

64個の実推論を完了。全reportのhash、入力identity、署名検証済みmodule identity、typed decision、source hashを照合した。各journalの完了ログは1回のみで、重複実行・replay・fallback・未集計の失敗はない。後半workerはserial側の担当範囲と重なる前に停止し、残りをserial側が完了した。

| 一次評価（24問、順序違いを除く） | 元のprompt | 現在の短縮版 |
|---|---:|---:|
| 全体正答率 | 22/24（91.7%） | 22/24（91.7%） |
| 正解あり問題 | 14/16（87.5%） | 14/16（87.5%） |
| unknownが正解の問題 | 8/8（100%） | 8/8（100%） |
| 正解ありなのに棄権 | 0 | 0 |
| unknownなのに断定 | 0 | 0 |

32個の質問・順序の組合せすべてで、元と短縮版の最終回答が一致。短縮による正解→誤回答0、誤回答→正解0、回答変更0。8問の選択肢入れ替えでも、両promptとも回答変更0（16検査）。順序違いを含めた集計は両方28/32（87.5%）だが、誤回答の2問も各2回数えるため、一次の24問正答率と分母を区別する。

共通する誤回答は次の2問。

1. `maximum`：最大lock期間2629800秒→2629800000秒という増加に対して、両方ともno。正解はyes。既存の誤回答を維持。
2. `delay-equal-units`：最小投票delay60分→3600秒という同じ長さに対して、両方ともyes。正解はno。追加評価で見つかった単位換算の誤り。

新規17問では両方16/17。情報不足、前提の誤り、選択肢に正解がない、矛盾する証拠、state中の引用指示といった追加問題ではunknown/必要な回答を維持した。モデルの既存の数値/単位に関する誤回答はprompt短縮によって改善していない。

確率は同一ではない。一次24問の正解候補の平均確率は90.282%→89.623%。負の対数尤度（小さい方が良い）は0.168143→0.169750、multiclass Brier score（小さい方が良い）は0.095642→0.092185で、指標により変化の向きも異なる。確率/校正性能が厳密に維持されたとは主張しない。少数の手作業セットなので、これらを母集団の精度や校正の改善/劣化と断定しない。

この診断セットでは、現在の短縮版による回答精度の低下は検出されなかった。24問に限った結果であり、未知の質問での同等精度の保証ではない。今回の結果だけを根拠にdefault promptをさらに変更することはしていない。

証跡：`artifacts/prompt-accuracy-v2/report.json` に全64結果・個別reportのpath/hash・一次/順序別/カテゴリー別集計・全pair比較を保存。`verification.json` は64件の照合結果、`probability-scores.json` は確率指標、`original-prefix-parity.json` は全32層の元prefix一致、`worker-report.json` は独立query担当範囲の実行記録。`scripts/verify_prompt_accuracy.py` で再監査できる。

