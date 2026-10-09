# Canister の本番 API

公開名は「動詞 + 対象」の camelCase。ブラウザの段階 query と課金 update は両方維持し、入口は統合しない。最適化リリースの本番 export は **20件**。専用 Attention・prefix cache ソースを含まない通常 Cargo ビルドは **18件**。旧名と互換 wrapper はない。

## 本番で残す20件

| API | 区分・認可 | 用途 |
| --- | --- | --- |
| `runInferenceStep` | public query | frame を一段進める |
| `runDeltaInferenceStep` | public query | MLP 完了・次 Delta・次 MLP 先頭を実行 |
| `runAttentionInferenceStep` | public query | MLP 完了・次 Attention・次 MLP 先頭を実行 |
| `runFinalInferenceStep` | public query | 最終段・readout・判断 |
| `getModelStatus` | public query | モデル ID、pack hash、受信量、検証量、完了チャンク |
| `getWeightCacheStatus` | public query | 不変重みキャッシュの状況 |
| `getInferenceQuote` | public query | 課金推論の見積もり |
| `runPaidInference` | update | caller の cycles で推論を実行 |
| `getInferenceReceipt` | caller-scoped query | caller 自身の request ID に対応する receipt |
| `retryInferenceRefund` | caller-scoped update | caller 自身の返金再試行 |
| `prepareModelUpload` | owner update | manifest 登録・アップロード準備 |
| `uploadModelChunk` | owner update | 固定チャンク登録・重複確認 |
| `verifyModelUpload` | owner update | 指定バイト数のハッシュ検証・完了判定 |
| `prepareWeightCache` | owner update | 指定した不変重みのキャッシュ準備 |
| `clearWeightCache` | owner update | 重みキャッシュ削除 |
| `prepareFixedPrefixCache` | owner update | 最適化リリースの固定 prefix キャッシュ準備 |
| `installInferencePrefix` | owner update | layer ごとの prefix 登録 |
| `getPaidInferenceConfig` | public query | 課金設定取得 |
| `configurePaidInference` | owner update | 課金設定変更 |
| `runPaidInferenceWorker` | self-only update | 課金ジョブを独立したメッセージで一段実行 |

public query も replicated execution では既存の owner 認可を維持する。blob は独自 frame/carry であり、テキストを直接受け取る API ではない。Rust の内部関数名は snake_case。`init` / `pre_upgrade` / `post_upgrade` は変更しない。

## 診断用ビルドだけに含める8件

`paid-update-diagnostics` feature を指定したビルドだけに、次の owner API を追加する。最適化リリースの診断用ビルドは28件、通常 Cargo の診断用ビルドは26件になる。

| API | 区分 | 用途 |
| --- | --- | --- |
| `profileInferenceStep` | query | ステップ詳細計測。詳細 span には `instruction-profile` も必要 |
| `startOwnerInference` | update | 管理用段階推論の開始 |
| `continueOwnerInference` | update | 管理用段階推論の継続 |
| `getPaidInferenceDebug` | query | hidden/state hash 等の検証結果 |
| `setPaidInferenceFault` | update | worker / refund の障害注入 |
| `setPaidInferenceReference` | update | 推論非実行中に独立scalar Attention参照を切り替える |
| `probePaidInferenceWorker` | update | worker の stale request と self-only 認可の検証 |
| `preparePaidInferenceUpgradeProbe` | update | upgrade guard・receipt 検証用状態の準備 |

CLI の `profile` / `update_infer_start` / `update_infer_continue` は `diagnostics:true` を明示し、診断用 canister にだけ送る。

## 削除した5件

- `getModelUploadProgress`：`getModelStatus` の `received` / `ready` を使う。
- `uploadModelBytes` / `finalizeModelUpload`：`uploadModelChunk` / `verifyModelUpload` を使う。
- `getInferenceDecision` / `getCandidateDecision`：最終 query または課金推論の結果で判断を返す。単独判断 CLI コマンドも削除した。

CLI の `upload` と `upload_parallel` は同じチャンク経路を使う。`upload` は並列数1を強制する。既存チャンクを照会して再開し、全体ハッシュの検証完了まで実行する。検証未完了の pack の変更や不正 hash 後の回復では `reset:true` を明示し、`prepareModelUpload` で再準備する。Python transport では `upload(..., reset=True)` を指定する。既定の `reset:false` では既存データを保持する。検証済みモデルの再準備は、従来どおり `already prepared` で拒否する。全文推論の検証 CLI は `--fuse-terminal-decision` と対応する最終段の融合オプションを指定する。旧単独 readout 検証は削除し、`check_terminal_decision.py` が最終 query の状態・判断・再実行を検証する。stable memory と receipt の保存形式は変更しない。従来どおり、upgrade は未完了アップロードの受信・検証カウンタと heap 上のチャンク受付表をリセットする。未完了ならチャンクを再送して検証し、完了したモデルは保存される。

## ビルドと検証

`scripts/canister_api_names.py` が通常ソースと凍結ソースに共通の削除・診断 feature gate・命名方針を適用する。削除対象は handler 本体を除去し、診断用 API は feature gate で WASM export 自体を外す。Candid の `hidden` 指定だけでは済ませない。

`canisters/inference/paid-inference.did` は default feature、`inference.did` は feature 無効の全サービス定義。feature 無効ビルドの基本 API は9件。`public-query.did` はブラウザに必要な6 query の subset。`optimized-inference.did` は最適化本番の全20メソッド。最適化ビルド時に WASM の `get_candid_pointer` から `service.did` を生成し、method export の名前・件数・query/update 区分との一致を検査する。

- `canisters/inference/tests/api_names.rs`：Candid の名前・件数・feature gate・query/update 区分。
- `scripts/check_canister_api_exports.py`：実際の WASM export の名前・件数・区分。
- `scripts/test_canister_api_names.py`：凍結ソースにも同じ削除と診断分離を適用する。
- `scripts/check_chunk_upload_api.py`：明示した専用 local canister で、不正ハッシュ・順序・重複・再開・逐次 CLI・upgrade を検証する。

本番デプロイは実施しない。公開時は canister upgrade、重みと prefix の heap cache 再準備、frontend の module hash 更新、実推論と計算量・応答サイズの再検証を合わせて行う。既存の測定済みデータは上書きしない。

凍結 scheduler・長文 carry は課金入口の prefix 定義を参照し、ビルド時に共通 prefix の token 数と宣言を照合する。通常 Cargo の登録・readiness・Delta stream・長文 history も同じ定義を参照する。

## 都度払いAPIの引数

`runPaidInference`の引数は推論リクエストと`request_id`の2つ。
料金版は見積もり、料金設定、実行結果から削除した。
受付時に現在の料金を計算し、添付cyclesが不足していれば`InsufficientCycles`で課金前に拒否する。
必要額だけを受領し、受付後はその額を実行記録と返金に使用する。
入力形式の`InferRequest.version`は引き続き使用する。
呼び出し側は新しいCandid定義に合わせて更新する必要がある。
過去の料金版を含むupgrade用JSON記録は読み込める。
