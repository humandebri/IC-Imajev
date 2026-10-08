# Prefix27 query推論のブラウザ直結実装

2026-10-07。公開queryをブラウザから直接呼ぶfrontendを実装した。Cloudflareは静的frontend・固定prefix素材を配信する。推論gatewayや受付queueは設けない。

## 実装

- `frontend/src/tokenizer.worker.ts`: tokenizerの共有ロードと各実行のAbortController。UIスレッドへ送るのはtoken数、進捗、最終結果とエラーだけ。
- `frontend/src/query-runner.ts`: 27-token prefix＋追加1〜57 tokenを検証し、32 queryを順次実行。BF16 carryはopaqueなbyte列で引き継ぎ、モデルの浮動小数演算をJavaScriptへ移植しない。
- `frontend/src/inference-agent.ts`: 匿名IC agent、node署名検証、readiness、実行前後の証明付きmodule hash検証、prefix素材のhash確認。
- `frontend/src/query-codec.ts`: checksum、入力hash、model、pack、step、op、dims等を検証。zero footerのhost-checksum応答は署名検証済みclientからだけ受け入れ、全headerを照合してchecksumを付ける。
- `scripts/prepare_browser_prefix.py`: 検証済みprefix archiveから32層の素材を生成。releaseにmodule/model/packとmanifest hashを固定。素材は15,418,368 bytesで、4.7GBの重みは配布しない。
- `frontend/src/inference-client.ts` と `App.tsx`: 入力をsubmit時に固定し、実行ごとのIDとgenerationで古い返信を破棄。進捗は実際の完了数／32を表示。

同一のWeb Workerでtokenizerを一度だけロードする。carryと中間応答の履歴は保持せず、各実行の現在の状態を引き継ぐ。

## 同時利用と失敗時の挙動

一つの推論内はcarry依存により順次実行する。異なる入力・タブ・利用者を直列化しない。Web Locks、全体同時数制限、サーバーqueue、結果cache、同一入力の共有promiseは導入していない。各入力のbuffer・進捗・結果・キャンセルを分離する。

query timeoutは30秒、tokenizerロード後の全推論deadlineは5分。SDK retryを無効にし、通信失敗・429/503等はエラー表示して利用者の再実行を待つ。checksum、署名、module/model不一致、instruction limitも自動retryしない。Cancelはfetchと次のquery発行を中断し、古い返信を捨てる。すでに開始したcanister計算の停止は保証しない。

ordinary queryの状態変更は保存されず、各入力のcarryはクライアントが持つ。共有update sessionを取り合わない。全利用者のcapacity管理・IP制限・bot対策は今回の範囲に含めない。

## 検証結果

- 実tokenizerのPython出力とのtoken ID一致。
- 32回のCandid送信byte・frame・checksum付与後の応答byte・最終判断が、最新runtimeの本番653入力の証跡と一致。
- checksum破損、model/pack/input/step/dims不一致を拒否。実行前キャンセルはqueryゼロ、1回完了後キャンセルは次のqueryを発行しない。
- Playwrightで実進捗、実行中編集時の結果ラベル保持、Cancelと再実行、古い応答の破棄、85-token入力の拒否、通信失敗後の再実行可否を確認。既存入力・コピー・token数・レイアウトのテストも実行。
- production buildを実Chromiumの独立した2 contextで同時に実行。本番queryで84-token/3選択肢と短文/2選択肢が完了し、それぞれ87.346秒、67.498秒。各32推論query＋2readiness query、update呼び出しゼロ。84-token入力の候補確率は既存の本番結果と完全一致。

本番ブラウザ証跡: `artifacts/browser-query-test-20261007/report.json`。protocol照合元: `artifacts/mainnet-prefix27-upgrade-20261007/anonymous-query-653/`。

32回で任意の入力が命令数上限に収まるという保証はしない。低速回線、実スマートフォン、大人数同時実行、peak heap/GC、P95は未測定。module変更・timeoutの検証コードは実装したが、実際のupgrade中やtimeoutまで待つ障害注入試験は今回実施していない。

ブラウザ直結なのでCloudflareの推論CPUをMiniflareで測る対象はない。公開にはビルド済み静的素材をCloudflareへ反映する必要がある。frontendのデプロイは今回の作業には含まない。

## 根拠

- 現行scheduler/codec: `client/query32_balanced.py`、`client/transport.py`。
- [公式JavaScript agent](https://js.icp.build/core/latest/libs/agent/)。
- [IC interface specification](https://docs.internetcomputer.org/references/ic-interface-spec/): queryの状態変更が保存されない。
