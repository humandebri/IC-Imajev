# 同じ50要求をqueryとupdateで計測

2026-10-05。ローカルの専用実験canister `6eydd-o3777-77775-aaama-cai` に、検証済み主問題の50要求を再送した。既存query exportの `step` と `terminal_step_decision` を、ic-agentの `query().call()` または `update().call_and_wait()` で呼ぶ。canisterのinstall/upgradeや固定重み準備は行わない。

## 結果

| 指標 | query | replicated update ingress |
|---|---:|---:|
| 推論呼び出し数 | 50 | 50 |
| handler命令合計 | 234,521,052,335 | 234,521,052,335 |
| 最大handler命令 | 4,918,963,366 | 4,918,963,366 |
| Candid要求＋返信bytes | 149,353,220 | 149,353,220 |
| 呼び出し待ち時間合計（秒） | 37.519563 | 53.301840 |
| 比較・保存込みloop時間（秒） | 38.883915 | 53.912531 |
| 観測cycles残高減少 | 90,946,826 | 401,269,034,430 |

全100返信のstate bytesが固定参照と完全一致し、各要求のhandler命令数、stable read、Candid byteも一致した。最後の専用readoutのlogits・確率・判断も一致。失敗・再送なし。updateの呼び出し時間はqueryの 1.4206 倍だった。

この結果は現在の最適化済み計算がupdate実行でも同じ命令数で動作することを確認する。最適化前のmoduleをupdate実行した比較ではないため、最適化によるupdate削減率は測っていない。update向けの40B予算を使った再配分・呼び出し融合も行っていない。

## 測定範囲

入力は132 token（prefix45＋suffix87、rotations=1）の既存通し推論から固定した要求で、参照は `artifacts/single-quad/tail-proof-v2/617`。prefix準備・入力構築・クライアントのgraph/codec処理・module検査・cycles status検査の時間は呼び出し時間合計から除外する。要求ごとにcanisterが実計算し、保存済みの返信を返す仕組みはない。下流要求は既存固定chainから読み出しており、今回の返信から新たなgraphを構築したend-to-end実行ではない。

queryを先に50回、次にupdateを50回実行した単発測定。ローカル環境での時間であり、本番subnetの合意・負荷・複製による時間を推定する値ではない。query測定中にはclientの検証ビルドも実行しており、CPU負荷は統制していない。handler counterはCandid decode/encode等の入口外処理を含まない。

cyclesはmanagement `canister_status` の前後残高差で、idle/storage費用とstatus照会の費用を含む。query側の残高減少をquery計算料金として扱わない。update側約0.401269兆cyclesもローカル設定の観測値で、本番価格や純粋な推論料金ではない。

各update返信はreplicated ingressで取得するが、クライアントが入力する中間状態の履歴をcanisterが認証する仕組みは追加していない。この計測だけで任意の最終判断をon-chain actionの根拠として信頼できるとは主張しない。

前後のmodule SHA256は `bdd8ced5ac7afb938168c7d88bafb03a42570a12fafe9470a8723c8e17786087` で一致。固定参照の全hashも前後一致。計測用bridgeは既定queryの動作を維持し、`execution: "update"` を明示した推論要求だけreplicated ingressで呼ぶ。

## 証跡と再現

`artifacts/replicated-inference/full-v1/report.json` に全100呼び出しの命令・時間・hash・残高、query/updateディレクトリに返信とmetricを保存。`measurement-source.zip` はこの計測のRust/Pythonソースを保存する。`smoke-v2` は先頭1要求の事前確認で、full-v1の50回に含めない。

```sh
cargo build --offline --release -p imajev-client
.venv/bin/python scripts/measure_replicated_inference.py \
  --directory artifacts/replicated-inference/full-new
```

`cargo test --offline -p imajev-client` はビルド成功（unit test数0）、Python構文確認成功。実際のquery/update呼び出しと全返信比較が主要な検証。
