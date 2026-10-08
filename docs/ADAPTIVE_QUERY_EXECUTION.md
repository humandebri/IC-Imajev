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
node --experimental-strip-types scripts/profile-query-plans.mjs --balanced 58 59 60 61 62 63 64 65 66 67 68 69
node --experimental-strip-types scripts/profile-query-plans.mjs --real --balanced 58 64 69
node --experimental-strip-types scripts/profile-query-plans.mjs --real --published 58 64 69
node --experimental-strip-types scripts/prepare-query-profiles.mjs --balanced --through=68
node --experimental-strip-types scripts/prepare-query-profiles.mjs --balanced --through=68 --check
node --experimental-strip-types scripts/check-query-calibration-policy.mjs
```

範囲の測定では、共通prefixと有効なtoken IDを用いて正確なsuffix長を作る。短いものは完全なユーザーpromptとは限らないため、token長のプロトコル検証と扱う。既存32queryを変更しない範囲ではその同じ経路を測定し、全入力を独立した数値計算で再検証したとは扱わない。変更した範囲では別の分割位置による新規query実行と比較し、terminalの全payload（最終hidden・norm・KV）、raw logits、判定、確率のbit一致を必須にする。84tokenでは既存32query、拡大範囲では256から8960までMLP生成を別queryで済ませる63query経路を参照にする。

最適化では、同じ入力に対する二つの実測経路の差からMLP生成位置の候補を算出する。この推定値をブラウザの判断には使わない。候補をmainnetで実行し、数値結果が一致し、全handler counterが4B以下、要求・返信のCandidサイズがそれぞれ1,990,000bytes未満、heapが4GiB未満である場合だけ設定へ採用する。成功した通常queryはCDKのdecode/encodeを含む実行上限内で完了している。handler counterそのものはCDKのdecode/encodeを含まない。

`--balanced` は層ごとのfrontとcompletionを256〜8960の256刻みで探索する。各層のfrontを状態として、予測したhandler命令数が3.85B以下の経路からquery数が最小のものを動的計画法で選び、同じquery数ならcarryを含むCandid通信量の予測値で選ぶ。初期query、追加のMLP生成query、橋渡し、terminalをすべて計上する。予測は候補の生成だけに使い、公開設定には実際のqueryのcounterとサイズを記録する。推定式の誤差もあるため、予測上の3.85Bと採用条件の4Bは区別する。

新しい測定は毎回 `artifacts/adaptive-query-20261008/measurements/<時刻>-<UUID>/` に保存する。開始時に保存先を出力し、同じ入力でもcanisterへのqueryと前後のmodule確認を新しく実行する。候補と参照の両方をこの保存先で測定するため、過去の所要時間を速度比較へ流用しない。候補生成に使う以前の層別の測定表は、元のarchiveから読み取る。

`--resume` を明示した場合だけ、元のarchiveの数値検証を再開できる。キャッシュの再利用時はその旨を出力し、module hash・plan・入力hash・選択肢・完了状態が一致した証跡だけを使う。これは新しい速度測定ではない。`--published --resume` の併用は測定開始前に拒否する。`--published` はその時点のフロント設定を使う比較測定なので、設定を切り替える前に実行する。

公開設定には測定を自動反映しない。`frontend/scripts/query-plan-decisions.json` にmodule hashと全採用証跡のSHA256、却下済み証跡と理由を記録する。設定生成は、flagの有無にかかわらず未レビュー・却下済みの証跡を拒否する。ビルド前の設定チェックも採用記録との一致を要求する。新しい候補を採用する場合はbit一致・予算・通信量・応答時間をレビューし、対象のverified JSONをarchiveへ保存したうえで、採用記録と設定を同じ変更に含める。新しい測定のhashが異なれば、以前と同じplanでも自動では採用しない。

`prepare-query-profiles.mjs --balanced --through=68` は全69種類のsuffix長の証跡を必須にし、suffix68（合計95token）までの採用済み最適化だけを組み込む。最適化した範囲では以前の63query経路と比べて実測query数とCandid総通信量の両方が減ったことも要求し、残りの長さでは既存の採用済み計画を維持する。`--balanced` だけの実行は、却下済みの96token候補を含むため拒否され、設定ファイルを変更しない。

測定済みという意味は記録した入力と実行環境で成功したことであり、すべてのtoken内容や混雑時の応答時間を保証するものではない。大きい入力の回数を減らしても、carryの通信量増加で遅くなる場合がある。最初の貪欲な探索による96tokenの61query候補は、同じ実入力の63query参照より通信・応答時間が増えたため採用しなかった。今回の動的計画法による候補は、その候補と分割位置が異なる。回数が同じでも通信量と応答時間を改めて検証する。

## 採用した測定結果

全69種類のsuffix長を測定し、変更した14種類は別のquery経路とbit一致した。

| 合計token数 | 推論query数 | 測定した最大handler命令数 |
| --- | ---: | ---: |
| 28〜82 | 32 | 3,955,895,133 |
| 83〜83 | 34 | 3,898,751,905 |
| 84〜84 | 36 | 3,879,775,523 |
| 85 | 37 | 3,835,656,824 |
| 86 | 39 | 3,855,075,794 |
| 87 | 41 | 3,857,261,283 |
| 88〜89 | 43 | 3,843,010,843 |
| 90〜92 | 46 | 3,852,620,574 |
| 93〜95 | 54 | 3,831,713,717 |
| 96 | 63 | 3,816,766,616 |

2回のreadiness queryは上表の回数に追加される。全採用計画でCandid要求の最大値は 1,802,117bytes、返信は 1,472,621bytes。

拡大範囲の全12種類を再測定し、同じ入力の参照経路とbit一致した。以前は全て63queryだったが、token数ごとの計画へ切り替える。全12種類で実測したCandid総通信量と総命令数が減少した。測定中に異なるtoken長を並行実行したものがあるため、これらの時間を負荷条件の揃った速度比較とは扱わず、実入力で現行計画と新計画を順に比較する。

採用範囲は85〜95token。96tokenの動的計画法による61query候補は通信量が128,361,199→123,074,009bytesへ減ったが、同じ実入力を順に実行すると101.409→113.926秒になった。最終payload・判定・logits・確率はbit一致し、最大handler命令数も3,835,623,734で条件内だったが、速度改善を確認できなかったため採用しない。96tokenは63queryを維持する。この比較は各1回であり、通信やサーバー負荷の変動を除去した統計的な比較ではない。証跡は `n69-real-published/report.json` と `n69-real-balanced/report.json`。

85tokenの実入力も37queryで完了し、独立した63query参照とbit一致した。37queryは72.236秒、参照は112.880秒だった。この参照はMLP生成を8960まで進める検証用の分割であり、以前の公開63query計画との直接比較ではない。最適化したterminal入力で日本語の選択肢2〜7件のbit一致と不正carry拒否も確認した。証跡は `real-balanced-n58-verified.json` と `balanced-n58-terminal-options-report.json`。

## 32query未満の検討

現在の一般query経路を調べた範囲では、32query未満の実行経路はない。`delta_mlp_stream_start_ids` はDeltaとMLP生成を開始するが、down projectionとresidual/normの完了は別queryが必要。`mlp_delta_front` と `attention_mlp_front` は現在のMLPを一つ完了し、次のDelta/AttentionとMLP生成を準備する。既存の部分Delta・Attention融合も次層のMLPまで二つ同時に完了する処理ではない。例外のterminal融合はlayer30と31に固定され、現在の32query経路ですでに使用している。

したがって、開始1回、layer0〜29のMLP完了30回、terminal1回の32回を維持する。単にfrontを大きくするだけではこの境界を省略できない。32回未満には、複数層のMLP完了をまとめる新しいcanister処理が必要であり、今回のフロント変更には含めない。

## 公開と復旧

Candidの6メソッド、匿名署名検証、固定module/model/pack/manifestを維持する。公開対象はCloudflareの `ic-imajev` と既存の `imajev.kinic.xyz`。公開直前のWorker versionを保存し、不具合時はそのversionへ戻す。canisterの復旧操作は必要ない。
