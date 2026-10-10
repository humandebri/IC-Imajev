# 都度払い・外部1 updateの推論

2026-10-06。[計画](PAID_UPDATE_INFERENCE_PLAN.md)を実装し、ローカルICPの実canister間呼び出しで検証した。利用者canisterは必要cyclesを添付して`infer`を1回呼び、最終判定を同じ呼び出しの応答で受け取る。内部workerは自己呼び出しで順に実行する。timerとユーザーごとの前払い残高は使わない。

## 通常推論の実測

最終通常版module SHA256は`c80e77130301636411b6df230f7ce04892898c5ba694e242b4c390c6a8d746ff`（`artifacts/paid-update-v1/build-v3/full.wasm`）。Candidの宣言順を修正する前の9回測定は`c46f1f8079a79c9a9b0308379419401e0e89f30bc354cacc6e795561c174dbe6`。診断用fault/probe featureを含まない。検証済みquery32 runtimeと6個のkernelを再利用し、約34B命令で区切る従来の演算順序を維持した。

|入力|外部infer update|内部worker update|worker命令合計の中央値|応答時間の中央値|
|---|---:|---:|---:|---:|
|617|1|5|145,511,841,191|47.27秒|
|620|1|4|125,608,781,512|24.60秒|
|653|1|5|146,531,453,775|28.65秒|

上表は各入力3回、計9推論・42 workerの測定。最終通常版も3入力を各1回再確認し、5/4/5 worker・元モデルとのbit一致・同じ最大heapで完走した。最終版の時間は617が115.83秒、620が24.48秒、653が30.59秒。証拠は`artifacts/paid-update-v1/proof-v4/verified.json`で、raw Candid 20応答を再decodeして検証した。通常推論の成功は合わせて12回。

以前の9回測定での最大worker handler命令は35,865,834,799、最大heapは4,234,608,640 bytesで、40B命令・4GiBの上限内だった。counterはCDKの引数codecと計測後の小さな記録処理を除く。実際のreplicated worker呼び出しも全て成功した。内部通信と外側callbackの費用は残るため、外部呼び出し回数の削減を計算費用の削減とは扱わない。

38-token投票prefixと27-token共通prefixを同じcanisterに登録し、入力から自動選択できた。heap余裕は約57.6MiBで、この実測は固定3入力の範囲である。

判定、確率、logits、31個の保存hidden、32個のconv／suffix KV状態、最終norm hiddenを元のBOOMDAO測定とbit比較して一致した。元のreferenceにlayer30 hiddenがないため、そのhiddenの直接比較は含まない。

試験relayへの外部呼び出しのCandid request／replyは617で589／567 bytes、620で557／543 bytes、653で549／567 bytes。中間配列を外部へ返さず、合計約1.1KBで入力と結果を運ぶ。relayの計測fieldsを含み、内部canister通信とHTTP envelopeは含まない。

時間はローカル環境の測定で、初回617は119.08秒、全9回は23.95〜119.08秒。高速化や本番レイテンシを保証する測定ではない。

証拠は`artifacts/paid-update-v1/proof-v2/verified.json`。`report_paid_update.py`が保存raw Candid 35応答を再decodeし、余剰cycles、不足・不正入力・古い料金版・受付停止の受領ゼロ、完了ID再送の二重課金防止、workerのself限定、callerごとのstatus、upgrade後のreceiptと料金設定、cache未準備での保存結果再取得も検証した。二つの拒否ケースの集計reportにはPythonの共有入力辞書の後続変更が混入したため、検証器は変更前に保存された各callの入力とraw応答を参照する。

試験料金はbase 100B＋suffix tokenごと3B cycles（617:268B、620:244B、653:271B）。これは支払い・返却を確認するためのローカル設定で、本番料金ではない。運営reserve 2Tは凍結費用を差し引いたliquid balanceに対する追加余裕。

## 返金・停止・upgradeの検証

診断版ではworkerのエラー、1batch後のtrap、明示返金の失敗を発生させた。全額返金、Pendingからの再試行、Doneの再試行で二重送金しないこと、別callerの返金拒否、Busyの未受領返却、古いjob IDと不正stageの拒否を確認した。`artifacts/paid-update-v1/fault-proof-v2/verified.json`はraw Candid 31応答から16項目を独立に検証している。

`icp canister install`は先にcanisterを停止する。実推論中のこの操作では、自己呼び出しが「is stopping」で失敗し、受領料金を全額返金してactiveを解放してからupgradeが成功した。停止による中断と返金は最終通常版でも実測できた。通常の運用は、受付を止め、実行中の推論と返金を完了させてからstop／upgradeする。

`pre_upgrade`の拒否自体は、重みを読み込まないowner専用の診断fixtureで検証した。active jobと返金InFlightをそれぞれ保持した状態で実ICのupgradeが拒否され、Pending／Doneの失敗receiptはidle upgrade前後で同一だった。`artifacts/paid-update-v1/upgrade-guards-v1/independent-verified.json`が4件のraw receipt応答と拒否メッセージを再検証し、guard／metadataのcoreソースが最終通常版と同じことも確認している。このfixtureは実推論の実行を主張するものではなく、実推論・実返金の測定と分けている。

Rustの12テストも成功した。公開Candidへの新API登録、失敗receiptと返金状態のmetadata往復、返金InFlightのupgrade拒否を含む。最終ログは`artifacts/paid-update-v1/native-tests-final.log`。通常版に診断fixture APIが混入しないことも検証した。

一部の測定スクリプトは、CLIによる停止・upgradeの動作を「実行中pre_upgrade拒否」と誤って期待してassertで終了した。保存応答と復元結果から成功済み項目を再検証し、未実施の項目は別のhook試験で確認した。`proof-v3`はCLIの引数指定を修正する前の失敗試行である。最終推論の検証器は`report_paid_update_final_completed.py`、返金は`report_paid_update_faults_completed.py`、hookは`report_paid_update_upgrade_guards.py`を参照する。

## 実装と利用

- 本体：`canisters/inference/src/paid_inference.rs`、共有型：`paid_types.rs`、Candid：`canisters/inference/paid-inference.did`。
- 利用例：[caller README](../examples/paid-inference-caller/README.md)。quoteは料金版と入力形状が同じなら再利用できる。通常の推論にstatus pollingは不要。
- receiptはcaller＋request ID＋入力hashで管理し、128件・24時間。未解決返金は期限経過でも保持する。入力は既存tokenizerの全token IDs、suffix 1〜57 token。
- 必要料金だけ受領し、余剰はICが返す。worker失敗では`deposit_cycles`で元callerへ受領料金を明示返金する。Pendingはcallerが再試行可能、InFlightとDoneは二重送信しない。
- 料金変更・重み・prefix操作はowner専用。実行中jobと返金InFlight中のupgradeを拒否する。upgrade後にheap cacheは再準備が必要。
- `scripts/build_paid_update_full_candid.py`で凍結runtimeから最終通常版、診断版は`PAID_DIAGNOSTICS=1 scripts/build_paid_update_diagnostic.py`で構築する。既存artifactを上書きしないためbuild/proof出力先は実行ごとに新しいdirectoryを使う。

通常測定終了後、比較canisterはsnapshotから元のmodule・721重みcache・pack状態へ復元し、snapshotを削除、試験callerを停止した。default canisterとmainnetは変更していない。
