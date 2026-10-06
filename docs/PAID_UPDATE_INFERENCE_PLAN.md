# 都度払い・自己呼び出しによるupdate推論の実装計画

2026-10-06。計画に従って実装済み。ローカルの実canister間検証と測定結果は[実測記録](PAID_UPDATE_INFERENCE_MEASURED.md)に保存済み。本番デプロイは行っていない。以下は設計時点の計画。

利用者canisterがcyclesを添付して`infer`を1回呼ぶと、Imajevが自分のworkerを順番に呼び、最後に元の呼び出しへ判定と確率を返す。ユーザーごとの前払い残高は持たない。最初の実装はtimerも待ち行列も使わず、1推論だけを実行する。使用中はcycles受領前に`Busy`を返す。

## 根拠と開始条件

- 固定比較対象は `artifacts/update-templates-v1/build/full.wasm`、SHA256 `de7f5f07a045caaee173ab560777ab7f7c2c91fb4a57b672700ca3d08abf169a`。
- 独立検証は `artifacts/update-templates-v1/proof-v2/verified.json`。617/620/653を各3回、5/4/5 updateで完了し、元の出力と保存中間状態にbit一致した。
- 現行の `update_inference.rs` はowner専用・一つのsession・一つの固定prefix bank。stageは0〜64で、約34B命令を超えた処理段の終了時に区切る。
- 最大update handler counterは35,865,130,885。最大heapは4,204,265,472 bytesで、wasm32上限まで約86.5MiBしかない。並行sessionと大きな中間状態の複製を導入しない。
- 依存するRust CDKはworkspaceの `ic-cdk = "=0.20.3"` を維持。ローカルの同版ソースで`Call::unbounded_wait`、`msg_cycles_available`、`msg_cycles_accept`、`canister_self`を確認した。

## 公開API案

|API|用途|
|---|---|
|`quote(request) -> Result<Quote, Error>` query|model・料金版・対応入力を検査し、料金、予測分割回数、最大worker回数を返す|
|`infer(request, request_id, quote_version) -> Result<InferenceResult, Error>` update|添付cyclesで都度払いし、全段の完了後に結果を返す|
|`inference_step(job_id, expected_stage) -> Result<StepResult, Error>` update|呼び出し元が自canisterの場合だけ許可する内部worker|
|`inference_status(request_id)` query|元のcallerだけが自身の処理結果・失敗・返金状況を確認する。正常系では呼ぶ必要はない|

`request`はmodel/入力形式のversion、全token IDs、選択肢を持つ。API境界でサーバーが固定prefixの一致と対応bankを選び、suffixへ切る。クライアントの中間hiddenや任意prefix packetは受け付けない。現在のtokenizerと入力形式を維持し、サーバーで新たなtokenizerを導入する作業は含めない。

`Quote`は固定料金表から算出する。queryの見積もり自体を信頼せず、`infer`側で同じ条件と現在の料金版を再検査する。初期対応は検証済みprefix bankとsuffix 1〜57 tokenに限定し、全token IDがmodelのvocabulary内であること、既存の選択肢制約、入力全体のサイズを計算前に検査する。未測定の長文は受領前に拒否する。初期の料金表は対応suffix長の帯ごとに設定し、内部通信・実行・失敗返金の費用と運営余裕を含める。具体的なcycles額は対象subnetの料金とworker方式の実測後に確定する。

## 1回の呼び出しの流れ

1. `infer`でcaller、入力上限、選択肢、prefix bank、quote version、重複request ID、モデル準備状態、実行中sessionを確認する。
2. 入力形状から64個の演算stageと推奨境界を計画する。既存の34B打ち切りも維持し、予測回数を実行回数として固定しない。対応入力と最大worker回数を制限し、課金時の最大費用を有限にする。
3. 不足cyclesや`Busy`なら受領せずエラーを返す。全ての確認後、必要額だけ`msg_cycles_accept`で受領してjobを記録する。
4. `infer`から`Call::unbounded_wait(canister_self(), "inference_step")`で最初のworkerを呼ぶ。最初のworkerがembedding・initial normと最初のbatchを実行する。
5. workerは検証済みgraphを1batch進め、`job_id / stage / done`だけを返す。中間配列と計測用hashは外側へコピーしない。新規worker updateが命令予算の単位になる。単に関数を分ける、または`await`するだけで重い計算の予算が増えるという前提にしない。
6. 外側は応答のjob ID・stageを再検査し、未完了なら次のworkerを順番に呼ぶ。前段に依存するので並列送信しない。
7. 全stageが完了したら結果を小さなreceiptへ保存し、大きなsessionを解放して、元の`infer`へ結果を返す。

外部からは1 update。内部では計算workerが現在の対象で4〜5 update程度になる見込みで、入口updateと応答callbackの処理・通信費用も別に発生する。自己呼び出しへの移植後に回数を再測定し、現在の5/4/5をそのまま達成済みとして扱わない。外側の処理とcallbackは小さく保つ。

## 支払いと失敗時の扱い

- ユーザーごとのcredit口座は作らない。job単位でcaller、request ID、入力digest、料金版、受領額、進捗、結果、返金状態だけを記録する。
- 未受領の添付cyclesはICが呼び出し元へ返す。受領済みcyclesは、後続workerが失敗しても自動で戻ると仮定しない。
- 初期案は成功時の固定料金。入力拒否、`Busy`、不足cycles、料金版不一致では受領0。運営側の処理失敗では受領した料金を明示的に全額返金する。失敗時の計算・返金通信費用はサービス側が負担するため、運営用cycles余裕を別に維持する。
- 返金は元caller canisterだけを宛先に、managementの`deposit_cycles`へcyclesを添付して実行する。返金状態を`Pending / InFlight / Done`として保持し、unbounded-waitを使い、確認できた失敗のみ再試行する。返金応答後の処理は最小化し、成功を記録するcallbackでtrapしない構成にする。応答処理の中断で結果が不確定な場合は自動で再送せず、receiptと管理用記録で復旧する。返金成功後の二重送信を防ぐ。
- 同じcaller・request ID・入力digestの再送は二重課金しない。完了済みなら保存結果を返し、実行中ならその状態を返す。異なる入力で同じIDを使った場合は拒否する。receiptの保持件数と有効期間を明示し、保持期限後の再利用保証はしない。
- 正常系は1回の`infer`だけで完結する。異常時の返金receiptは前払い残高ではなく、二重課金・二重返金を避けるための最小記録として必要。
- ブラウザの通常ingressにはcyclesを直接添付できないため、初版はcanisterからの利用を対象にする。ブラウザ対応はウォレット／proxy経由の別段階にする。

## 権限と状態

既存のowner権限を全体から外さない。重み、prefix登録、料金変更、運営操作は引き続きowner専用とする。新しい`infer`だけを支払い条件で公開し、`inference_step`は`caller == canister_self()`だけを許可する。外部から内部workerを直接実行できない。

jobは`Accepted -> Running(stage) -> Completed`、または`Failed -> RefundPending -> Refunded`へ遷移する。workerを呼ぶ前に期待stageと呼び出し中フラグを記録し、各`await`後にjob ID・期待stage・応答を照合する。古いstageと別jobへの呼び出しを拒否する。worker trapではそのworkerの途中変更がrollbackされ、外側がrejectを捕捉して失敗を確定する。

大きなsessionは1つだけ保持する。完了receiptと未処理返金は小さなstable metadataとして保持する。現在のupgradeは重み・prefixのheap cacheを失うため、実行中jobまたは返金呼び出し中は通常のupgradeを拒否し、停止受付・処理完了・receipt保存・upgrade・cache再準備・受付再開の順で運用する。

timerは初版で使わない。後で「即座にjob IDを返す非同期API」や、返金・停止jobの自動回復が必要になった場合に追加する。その場合もjob状態と進捗確認を再利用する。

## prefix bankの段階導入

初版の自己呼び出し・課金検証は既存の1-bank構造をそのまま使う。投票38-token構成で617/620、共通27-token構成で653を別々に測り、既存実測と比較する。

全3入力を一つの公開canisterで扱う段階で、2-bank登録とjobごとのbank選択を追加する。共通部分の共有、packet容量、allocatorの最大heapを測り、4GiB未満を確認する。現在のheap余裕は小さいため、測定なしに2-bankの常駐を確約しない。収まらなければbankごとに推論canisterを分け、caller側のルーティングを採用する。APIと都度払い方式は共通に保つ。

## 実装順と合格条件

1. **自己呼び出しだけを追加**：検証済みupdate wrapperからjob状態・内部worker・外側orchestratorを切り出す。モデル、演算順序、34Bの区切りは維持する。新しい実験ビルダーで凍結済みruntimeとkernelを再利用する。
2. **ローカルで1入口の完走を証明**：外部の`infer`1回だけで617/620/653を各3回実行する。内部worker数・callback数・各handler命令・cycles差・通信量・heapを記録し、最終結果と検証用保存中間状態を元モデルとbit比較する。
3. **都度払いを追加**：料金版、必要額だけの受領、request ID、explicit refund、小さなreceiptを実装する。料金はworker方式の実測と対象subnet料金から設定する。
4. **支払う側のテストcanisterで確認**：余剰cycles、不足、`Busy`、重複request ID、外部worker直接呼び出し、古いstage、worker reject/trap、返金、返金再試行、別callerのstatus照会を検証する。成功・失敗の両方で支払う側と受ける側の残高を照合する。
5. **2-bankと通常運用を確認**：一つの公開canisterでのbank選択、最大入力形状、heap、idle reserve、受付停止、完了receiptのupgrade保持、cache再準備、受付再開を確認する。local比較ではsnapshotを保存し、最後に元のmodule・cacheへ復元する。

最終合格は、外部1回で結果が返り、元モデルとbit一致し、全workerが40B命令未満、wasm32 heapが4GiB未満、二重課金・二重返金・外部worker実行が防がれていること。1-bankの実験で合格しても2-bankや本番公開まで完了したとは扱わない。

予定変更箇所：`canisters/inference/src/update_inference.rs`の内部関数化、`canisters/inference/src/paid_inference.rs`の新設、`canisters/inference/src/lib.rs`のAPI・metadata連携、Candidとcaller用の例、診断用build/prove/reportスクリプト。既存の検証済みartifactは書き換えず、新しい `artifacts/paid-update-v1/` 以下へ成果物を保存する。

## 参照

- [最新update実測](UPDATE_TEMPLATES_MEASURED.md)
- [ICPの呼び出しとawait前後の状態](https://docs.internetcomputer.org/guides/canister-calls/inter-canister-calls/)
- [命令・memory上限](https://docs.internetcomputer.org/references/resource-limits/)
- [cyclesとcanisterからの添付](https://docs.internetcomputer.org/concepts/cycles/)
- [timerの自己呼び出し](https://docs.internetcomputer.org/concepts/timers/)
