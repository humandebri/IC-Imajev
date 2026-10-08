# Prefix27のtoken数に応じたquery実行

対象は `xis3j-paaaa-aaaai-axumq-cai`、module hashは `3291a064976fb13df548582871ae96a3347f052569eceb7b9807dc23ec5b2495`。canisterのupgrade、重みやprefixの再準備、推論updateは行わず、既存の匿名queryだけを使う。

## 実行計画

`frontend/src/query-profiles.json` が許可する連続したsuffix範囲、層ごとのMLP生成位置、追加queryの位置を持つ。token数は固定27-token prefixを除いた値。module hashが一致し、当該token数の測定済み計画が存在するときだけ実行する。入力上限、token表示、Workerの入力検証はこの一つの設定を参照する。

`step` の `mlp_stream_next` によりMLP生成を256単位で分割する。入力のquantizationと途中のF32状態はcanisterが生成したcarryに保持し、クライアントは演算せずに次のqueryへ送る。初期query・MLP生成・橋渡し・terminalで、wireのstep番号は実際に成功したquery数から決める。モデルの層番号は独立して保持する。特にterminalの返信stepを32へ固定しない。

進捗は計画を選んだ後に `0 / total` から始まり、返信を検証してから増やす。2回のreadiness queryとmodule hash証明の取得は推論のquery数に含めない。各推論はcarry、step、AbortSignalを独立して持つ。途中の失敗を自動再試行せず、キャンセルや5分の期限を尊重する。

## 測定と数値検証

明示的な測定コマンドは `frontend/scripts/profile-query-plans.mjs`。匿名queryのみを呼び、前後で署名付きmodule hash証明を確認する。測定はデフォルトのUIテストには含めない。証跡は `artifacts/adaptive-query-20261008/` に保存する。

```sh
cd frontend
node --experimental-strip-types scripts/profile-query-plans.mjs 1 2 3
node --experimental-strip-types scripts/profile-query-plans.mjs --optimize 56 57
node --experimental-strip-types scripts/profile-query-plans.mjs --real --optimize 69
node --experimental-strip-types scripts/prepare-query-profiles.mjs
```

範囲の測定では、共通prefixと有効なtoken IDを用いて正確なsuffix長を作る。短いものは完全なユーザーpromptとは限らないため、token長のプロトコル検証と扱う。既存32queryを変更しない範囲ではその同じ経路を測定し、全入力を独立した数値計算で再検証したとは扱わない。変更した範囲では別の分割位置による新規query実行と比較し、terminalの全payload（最終hidden・norm・KV）、raw logits、判定、確率のbit一致を必須にする。84tokenでは既存32query、拡大範囲では256から8960までMLP生成を別queryで済ませる63query経路を参照にする。

最適化では、同じ入力に対する二つの実測経路の差からMLP生成位置の候補を算出する。この推定値をブラウザの判断には使わない。候補をmainnetで実行し、数値結果が一致し、全handler counterが4B以下、要求・返信のCandidサイズがそれぞれ1,990,000bytes未満、heapが4GiB未満である場合だけ設定へ採用する。成功した通常queryはCDKのdecode/encodeを含む実行上限内で完了している。handler counterそのものはCDKのdecode/encodeを含まない。

測定済みという意味は記録した入力と実行環境で成功したことであり、すべてのtoken内容や混雑時の応答時間を保証するものではない。大きい入力の回数を減らしても、carryの通信量増加で遅くなる場合がある。96tokenの61query候補は、同じ実入力の63query参照より通信・応答時間が増えたため公開の計画には採用しない。

## 採用した測定結果

全69種類のsuffix長を測定し、変更した14種類は別のquery経路とbit一致した。

| 合計token数 | 推論query数 | 測定した最大handler命令数 |
| --- | ---: | ---: |
| 28〜82 | 32 | 3,955,895,133 |
| 83〜83 | 34 | 3,898,751,905 |
| 84〜84 | 36 | 3,879,775,523 |
| 85〜96 | 63 | 3,816,766,616 |

2回のreadiness queryは上表の回数に追加される。全採用計画でCandid要求の最大値は 1,622,366bytes、返信は 1,472,621bytes。

## 32query未満の検討

現在の一般query経路を調べた範囲では、32query未満の実行経路はない。`delta_mlp_stream_start_ids` はDeltaとMLP生成を開始するが、down projectionとresidual/normの完了は別queryが必要。`mlp_delta_front` と `attention_mlp_front` は現在のMLPを一つ完了し、次のDelta/AttentionとMLP生成を準備する。既存の部分Delta・Attention融合も次層のMLPまで二つ同時に完了する処理ではない。例外のterminal融合はlayer30と31に固定され、現在の32query経路ですでに使用している。

したがって、開始1回、layer0〜29のMLP完了30回、terminal1回の32回を維持する。単にfrontを大きくするだけではこの境界を省略できない。32回未満には、複数層のMLP完了をまとめる新しいcanister処理が必要であり、今回のフロント変更には含めない。

## 公開と復旧

Candidの6メソッド、匿名署名検証、固定module/model/pack/manifestを維持する。公開対象はCloudflareの `ic-imajev` と既存の `imajev.kinic.xyz`。公開直前のWorker versionを保存し、不具合時はそのversionへ戻す。canisterの復旧操作は必要ない。
