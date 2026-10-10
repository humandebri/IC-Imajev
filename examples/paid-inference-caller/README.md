# 都度払いのImajev呼び出し

`lib.rs`はテスト／統合用のowner管理relayで、指定canisterのAPIへCandidとcyclesを転送する。`forward`はownerしか呼べない。戻り値にはraw response、未受領で返ったcycles、呼び出し前後の実行用残高を含む。前払いcredit口座は作らない。

サービスの型は `canisters/inference/src/paid_types.rs`、Candidは `canisters/inference/paid-inference.did` を参照する。推論用の入力には現在のtokenizerによる全token IDsと選択肢を使い、サーバーは固定common prefixを使う。標準ビルドの全入力の上限は512 tokenで、prefixとsuffixの長さは見積もりで確認できる。94 tokens以下は一括処理、95〜512 tokensは分割処理を使う。[現在のprefix構成](../../docs/PREFIX5_MIGRATION.md)を参照する。任意の長文や画像には対応しない。

POC用の料金設定は [`poc-config.json`](poc-config.json)。基本10B＋suffix tokenごと3B cyclesで、常駐モデルの維持費は運営負担として扱う。[料金測定と設定方法](../../docs/PAID_UPDATE_PRICING_POC.md)を参照する。`configure_paid_poc.py`は明示したlocal canisterへ設定を適用し、設定のreadbackを行う。

別canisterからは次のように呼ぶ。見積もりは同じ入力形状・料金設定の間は再利用でき、毎回quoteを呼ぶ必要はない。

```rust
use ic_cdk::call::Call;

let quote: Result<Quote, InferError> = Call::unbounded_wait(service, "getInferenceQuote")
    .with_arg(request.clone())
    .await?
    .candid()?;
let quote = quote.map_err(|e| format!("{e:?}"))?;
let result: Result<InferenceResult, InferError> = Call::unbounded_wait(service, "runPaidInference")
    .with_args(&(request, request_id))
    .with_cycles(quote.fee)
    .await?
    .candid()?;
```

`runPaidInference`は内部workerを自己呼び出しで順番に実行し、最終結果を元の呼び出しへ返す。外部からstart/continueを繰り返す必要はない。内部workerの実行・メッセージ費用は残る。timerは使用しない。ブラウザの通常updateからはcyclesを直接添付できないので、canister／wallet／relayを経由する。

料金不足、Busy、不正入力ではサービス料金を受領しない。sender自身の通信・実行費用は別。受付時に現在の料金を再計算し、添付額が足りなければ課金前に拒否する。料金不足の場合は見積もりを取り直して再送する。必要料金だけを受領し、余剰はICが自動返却する。受領後のworker失敗ではサービス側が元callerへ全額を明示返金する。返金Pendingなら、同じcallerから`retryInferenceRefund(request_id)`を呼ぶ。返金InFlightを自動再送しない。

`request_id`はcallerごとに一意な64 bytes以下の文字列にする。同じ入力とIDの再送は、保存receiptが残っている間は二重課金せず保存結果を返す。IDを違う入力に流用するとIdConflict。receiptは128件・24時間が上限で、未解決の返金は期限経過でも捨てない。満杯時は受領前にReceiptCapacityを返す。これは小さな取引記録で、ユーザー残高ではない。

管理者は重みとprefixの準備後に`configurePaidInference`で料金・運営用cycles余裕を設定する。`reserve_cycles`は凍結費用を除いたliquid balance上で確保する追加余裕であり、凍結用残高を二重加算しない。ローカルの試験料金を本番料金として扱わず、対象subnetで実行・通信・保存費用を検証してから設定する。料金とモデルの準備が整えば常に受付可能。


完了receipt・失敗receipt・返金状態・料金設定はupgradeで保持する。実行中jobまたは返金InFlight中のupgradeは拒否する。upgrade後に重み・prefix cacheの再準備が必要だが、完了済みIDの再取得はcache再準備前でも可能。

512-token対応版から通常116-token版へ戻した場合も、保存receiptが残っている間は512 tokensまでの同じ入力・IDを再送して保存結果を取得できる。新規推論は適用中の版の受付上限に従う。

Candid入出力ツールは `CARGO_INCREMENTAL=0 cargo build --locked -p imajev-client --example paid_update_args` でビルドする。現在の `PaidTransport` はこのツールを使う。凍結済みの旧版検証では当時のツールを明示指定する。

診断用のfault/probe APIは`paid-update-diagnostics` featureでのみ存在し、通常版には含めない。各ローカルproofはsnapshotを保存して比較canisterを復元し、テストrelayを停止する。build/proveの成果物は `artifacts/paid-update-v1/` に保存する。

27-token専用化前の測定済み通常版は`artifacts/paid-update-v1/build-v3/full.wasm`（SHA256 `c80e77130301636411b6df230f7ce04892898c5ba694e242b4c390c6a8d746ff`）。32-query版の凍結runtime／kernelを使う`build_paid_update_full_candid.py`の出力で、通常のCargo feature指定だけによるビルドと同じ性能とは扱わない。[最終版・返金・upgradeの実測](../../docs/PAID_UPDATE_INFERENCE_MEASURED.md)を参照する。

`icp canister install`は先に停止するため、実行中の自己呼び出しが停止エラーになる場合がある。推論と返金が完了したことを確認してからstop／upgradeする。診断fixtureのAPIは通常版に含めない。

116 tokens以下は30B命令を目安にworkerを区切り、上限8回で受付する。公開済みcanisterへの適用は別操作であり、通常queryのブラウザUIは84-token制限を維持する。[ローカルでの上限検証](../../docs/PAID_TOKEN_LIMIT_LOCAL.md)。

512-token対応は標準featureで有効になる。互換性確認用の116-token版は `--no-default-features --features paid-update-inference` でビルドする。最適化済みWasmの標準ビルドは `build_latest_common_prefix27.py` または `build_paid_token_chunks.py`、116-token版は前者に `--legacy-116` を指定する。長い入力のAttentionだけ必要な区間でheadを分割する。[分割処理とローカル検証](../../docs/PAID_TOKEN_CHUNKS_LOCAL.md)を参照する。

現行 API・5-token prefix の検証手順は [PAID_PREFIX5_VERIFICATION.md](../../docs/PAID_PREFIX5_VERIFICATION.md) を参照する。
