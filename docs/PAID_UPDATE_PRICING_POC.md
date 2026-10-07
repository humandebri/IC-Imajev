# POC用の都度払い料金

2026-10-06。料金候補は **10B cycles + suffix token数 × 3B cycles**。料金は事前に確定し、外部`infer`1回で支払って結果を受け取る。前払いcredit管理や成功後の差額精算は追加しない。

最終設定をローカルcanisterへ適用して実推論を検証済み。費用測定6形状と初回の3問、最終設定の短文1件と3問、合計13推論を実行した。本番への料金反映は行っていない。

## 測定から決めた範囲

共通27-tokenと投票38-tokenの両prefixについて、suffix 1・24・57を各1回測定した。人工token列は費用の確認用で、質問回答の精度評価には使わない。固定された通常版module `c80e77130301636411b6df230f7ce04892898c5ba694e242b4c390c6a8d746ff`を使い、モデル・演算順序・worker分割は変更していない。

|prefix|suffix|worker数|worker命令数|ローカル推論側の残高から算出した消費|
|---|---:|---:|---:|---:|
|共通27|1|1|6.87B|6.96B cycles|
|共通27|24|2|64.58B|64.62B cycles|
|共通27|57|5|146.53B|147.28B cycles|
|投票38|1|1|8.29B|8.31B cycles|
|投票38|24|2|66.12B|66.15B cycles|
|投票38|57|5|148.22B|148.29B cycles|

残高による消費は、受領額を補正した前後残高差である。予約cyclesも併せて記録した。測定期間のメモリ維持費や管理照会の影響も含み、jobに厳密に帰属する実費とは主張しない。host経過時間とidle rateによる補正値もartifactに保存したが、近似なので料金算定の中心にはしない。

13-nodeの公式実行単価ではworker命令1Bあたり1B cycles。測定したworker命令数に20%を加え、さらに未計測の入口・callback・通信費用に1Bを見込む。基本10Bとし、全6形状を覆うトークン単価を100M単位に切り上げた結果3Bとなった。料金が測定した残高消費の120%＋1Bも覆うか、検証器で確認する。[公式cycles料金](https://docs.internetcomputer.org/references/cycle-costs/)

基本5Bを固定した初回の算定では、短い投票入力の固定処理費用がトークン単価に乗り、単価6Bになった。この配分は採用せず、基本10Bへ調整した。過去の試行は`artifacts/paid-pricing-poc-v1/`へ保存し、書き換えない。

## 料金の比較

|入力|suffix|以前の試験料金|POC料金|低下率|
|---|---:|---:|---:|---:|
|BOOMDAO 617|56|268B|178B|33.6%|
|BOOMDAO 620|48|244B|154B|36.9%|
|BOOMDAO 653|57|271B|181B|33.2%|

短い入力はsuffix 1で13B、24で82B。呼び出し元自身の実行・通信費用は別である。成功時はquoteの料金を正確に受領し、余剰添付は未受領返却する。運営側の失敗時は受領料金を全額返金する既存仕様を維持する。

POCでは常駐モデルの維持費を運営負担として扱い、利用件数から料金へ厳密に配分しない。料金の余裕が維持費の一部を賄う可能性はあるが、損益分岐や採算を保証する設定ではない。13-nodeの単価を前提にしたPOC料金で、異なるsubnet・実行単価・モデルに無条件で流用しない。

## 設定の利用

設定ファイルは`examples/paid-inference-caller/poc-config.json`。初期状態の料金版は3で、既存canisterへ適用するときは現在の版より大きい値にする。`reserve_cycles=2T`はローカルPOC用の追加運営余裕で、本番運用の残高設計を確定した値ではない。

重みとprefixを準備した通常のpaid版canisterへ、owner identity `imajev-local`で次を実行する。対象principalと新しい出力directoryを指定する。スクリプトはlocal専用で、現在の料金版を読み、版を増やして設定し、設定のreadbackを確認する。受付も有効になるため、モデル準備を終えてから使う。

```sh
python3 scripts/configure_paid_poc.py --canister <paid-canister-id> --directory artifacts/poc-price-apply-01
```

料金変更は`configure_paid`の設定変更だけで済み、Candid・stable metadata・モデルの変更は不要。保持中のreceiptと返金額には受付時の料金が残る。

## 証拠と再実行

- 初回費用測定：`scripts/measure_paid_pricing_poc.py`、`artifacts/paid-pricing-poc-v1/verified.json`。
- 最終料金の検証：`scripts/measure_paid_pricing_poc_final.py`、`artifacts/paid-pricing-poc-v2/`。
- raw Candid再検証：`scripts/report_paid_pricing_poc.py`。最終版は引数`artifacts/paid-pricing-poc-v2`を指定する。

最終設定の4推論は全て完走し、追加の短文は13B、617/620/653は178B/154B/181Bを正確に受領した。各回の余剰12,345,678 cyclesは返却された。既存3問の判定・確率・logitsは元モデルとbit一致し、内部workerは5/4/5回。料金不足、古い料金版、保持中の同一request ID再送でも受領ゼロを確認した。

最終通常版のraw Candid 14応答を再decodeして独立に検証した。初回測定も24応答を検証済み。最終4件のローカル推論側の残高消費は短文8.31B、617が145.84B、620が125.70B、653が146.60B cycles。新料金はこれらと初回6形状で、測定消費の120%＋1Bを覆った。これは有限のPOC測定の結果であり、全入力や常駐費回収の保証ではない。

Pythonスクリプトの構文確認と設定用CLIのhelpを確認した。`configure_paid_poc.py`自体の適用コマンドは今回実行していないが、同じ公開APIによる設定・版更新・readbackは推論測定スクリプトで実行済み。失敗時全額返金の既存実装は変更せず、以前の[返金検証](PAID_UPDATE_INFERENCE_MEASURED.md)を根拠としている。

測定スクリプトは比較canisterをsnapshotで保存し、終了時に元のmodule・重み・prefix cacheへ復元し、snapshotを削除し、試験relayを停止した。復元・停止はstatusでも再確認した。再実行時はスクリプトの出力directoryを新しい名前に変え、既存artifactを上書きしない。mainnetとdefault canisterは変更していない。通常運用へ設定を反映する作業とは分ける。
