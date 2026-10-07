# 都度払いのImajev呼び出し

`lib.rs`はテスト／統合用のowner管理relayで、指定canisterのAPIへCandidとcyclesを転送する。`forward`はownerしか呼べない。戻り値にはraw response、未受領で返ったcycles、呼び出し前後の実行用残高を含む。前払いcredit口座は作らない。

サービスの型は `canisters/inference/src/paid_types.rs`、Candidは `canisters/inference/paid-inference.did` を参照する。推論用の入力には現在のtokenizerによる全token IDsと選択肢を使い、サーバーが固定27/38-token prefixを選択する。suffixの上限は57 token。任意の長文や画像には対応しない。

POC用の料金設定は [`poc-config.json`](poc-config.json)。基本10B＋suffix tokenごと3B cyclesで、常駐モデルの維持費は運営負担として扱う。料金版は適用先の現在版より増やす。[料金測定と設定方法](../../docs/PAID_UPDATE_PRICING_POC.md)を参照する。`configure_paid_poc.py`は明示したlocal canisterへ設定を適用し、版の更新とreadbackを行う。

別canisterからは次のように呼ぶ。見積もりと料金版は同じ入力形状・料金設定の間は再利用でき、毎回quoteを呼ぶ必要はない。

```rust
use ic_cdk::call::Call;

let quote: Result<Quote, InferError> = Call::unbounded_wait(service, "quote")
    .with_arg(request.clone())
    .await?
    .candid()?;
let quote = quote.map_err(|e| format!("{e:?}"))?;
let result: Result<InferenceResult, InferError> = Call::unbounded_wait(service, "infer")
    .with_args(&(request, request_id, quote.version))
    .with_cycles(quote.fee)
    .await?
    .candid()?;
```

`infer`は内部workerを自己呼び出しで順番に実行し、最終結果を元の呼び出しへ返す。外部からstart/continueを繰り返す必要はない。内部workerの実行・メッセージ費用は残る。timerは使用しない。ブラウザの通常updateからはcyclesを直接添付できないので、canister／wallet／relayを経由する。

料金不足、Busy、Paused、料金版不一致、不正入力ではサービス料金を受領しない。sender自身の通信・実行費用は別。必要料金だけを受領し、余剰はICが自動返却する。受領後のworker失敗ではサービス側が元callerへ全額を明示返金する。返金Pendingなら、同じcallerから`retry_inference_refund(request_id)`を呼ぶ。返金InFlightを自動再送しない。

`request_id`はcallerごとに一意な64 bytes以下の文字列にする。同じ入力とIDの再送は、保存receiptが残っている間は二重課金せず保存結果を返す。IDを違う入力に流用するとIdConflict。receiptは128件・24時間が上限で、未解決の返金は期限経過でも捨てない。満杯時は受領前にReceiptCapacityを返す。これは小さな取引記録で、ユーザー残高ではない。

管理者は重みとprefixの準備後に`configure_paid`で料金・版・受付状態・運営用cycles余裕を設定する。`reserve_cycles`は凍結費用を除いたliquid balance上で確保する追加余裕であり、凍結用残高を二重加算しない。ローカルの試験料金を本番料金として扱わず、対象subnetで実行・通信・保存費用を検証してから設定する。初期設定は受付停止。

推論中でも、料金版を増やして`enabled=false`を設定すれば新規受付を停止できる。実行中jobは受付時の料金・版で継続する。推論中に`enabled=true`の設定を適用することは拒否するため、受付再開はjob完了後に行う。

完了receipt・失敗receipt・返金状態・料金設定はupgradeで保持する。実行中jobまたは返金InFlight中のupgradeは拒否する。upgrade後に重み・prefix cacheの再準備が必要だが、完了済みIDの再取得はcache再準備前でも可能。

診断用のfault/probe APIは`paid-update-diagnostics` featureでのみ存在し、通常版には含めない。各ローカルproofはsnapshotを保存して比較canisterを復元し、テストrelayを停止する。build/proveの成果物は `artifacts/paid-update-v1/` に保存する。

最終通常版は`artifacts/paid-update-v1/build-v3/full.wasm`（SHA256 `c80e77130301636411b6df230f7ce04892898c5ba694e242b4c390c6a8d746ff`）。32-query版の凍結runtime／kernelを使う`build_paid_update_full_candid.py`の出力で、通常のCargo feature指定だけによるビルドと同じ性能とは扱わない。[最終版・返金・upgradeの実測](../../docs/PAID_UPDATE_INFERENCE_MEASURED.md)を参照する。

`icp canister install`は先に停止するため、実行中の自己呼び出しが停止エラーになる場合がある。受付停止後に推論と返金が完了したことを確認してからstop／upgradeする。診断fixtureのAPIは通常版に含めない。
