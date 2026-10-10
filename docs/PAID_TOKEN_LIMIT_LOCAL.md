# 有料update推論の入力上限をローカルで検証

2026-10-08の初回境界探索の記録。受付制限を緩めた検証版で、共通prefixを含む116 tokensまで完走し、通常query経路とのbit一致を確認した。この時点では84-token受付制限を変更していない。117 tokens以上は一括演算の制約に当たった。その後の116-tokenソース対応と256-token分割版は[別の検証記録](PAID_TOKEN_CHUNKS_LOCAL.md)を参照する。

## 実測結果

各入力1回。成功ケースは利用者canisterからcyclesを添付した外部 `infer` 1回で最終結果を受け取った。全入力に固定27-token prefixを含む。

|全tokens|suffix tokens|結果|内部worker|ローカル所要時間|最大heap bytes|
|---:|---:|---|---:|---:|---:|
|84|57|成功|4|28.11秒|4,177,199,104|
|85|58|成功|4|28.85秒|4,177,199,104|
|95|68|成功|5|33.83秒|4,177,199,104|
|116|89|成功|7|51.58秒|4,188,995,584|
|117|90|MLP入力制限で失敗|—|0.82秒|未計測|
|132|105|Deltaのprefix入力検証で失敗|—|0.49秒|未計測|
|500|473|embeddingの形状制限で失敗|—|0.49秒|未計測|

116 tokensでの最大heapは約3.90 GiB。設定された4 GiBとの差は約101.06 MiBだった。最大worker計測値は33,527,410,990命令。実際のreplicated workerメッセージも成功した。計測はメッセージ入口からのcounter zeroのチェックポイントで、後続の小さな記録処理と返信末尾は含まない。

84 tokensは既存のBOOM 653入力。長い入力は、同じ入力の末尾4 tokensを維持し、その直前へ `[198, 220, 16]` を必要数挿入した長さ試験用の入力である。意味のある長文に対する精度評価や、全入力での性能保証ではない。時間はローカルでの各1回の測定である。

## 検証版で変えたもの

検証専用のコピーを作り、有料APIの全入力上限を500、suffix上限を473、内部worker上限を64に緩めた。schedulerのsuffix上限も473にし、計算を区切る目安を34Bから30B命令へ下げた。owner用の診断APIを有効にして、中間状態を取得した。

演算runtimeと34個の投影kernelは既存の最適化版を維持した。各kernel本体のhash一致とWasm validationを確認している。検証module SHA256は `fe1c7e79d778f6abb404ab833c4d4bf0f266a463a6b3adc853dcc79e82c2177f`。

従って、116-token対応は受付上限だけを変更した結果ではない。今回の30B設定ではworkerが7回になったため、採用する場合はworker回数上限もschedulerと整合させる必要がある。元の34B設定で必要な回数は今回測定していない。

## 数値と支払いの検証

- 成功4入力を通常query経路でも新規計算した。各66 query、合計264 query。各入力の32層のhiddenとconv／suffix KV状態、最終norm hidden、判定・確率・unknown確率・logitsが有料update結果とbit一致した。
- 84-token入力は保存済みの既存参照とも一致した。参照に存在する31層のhiddenと全32層の状態を比較した。参照にないlayer30 hiddenは、新規query経路との比較で検証した。
- 成功した4入力は同じIDで再送し、保存結果の一致と添付料金の全額未受領返却を確認した。
- 失敗3入力では受領した料金の全額返金とcaller残高を確認した。caller自身の実行・通信費用は残る。
- 保存された23件のraw Candid応答を再decodeし、結果記録と照合した。保存済みquery配列からも全4入力・全32層のhash一致を再確認した。

## 環境と復元

対象は `http://localhost:8001/` の検証用canister `4caro-hl777-77775-aaaba-cai`。試験前に停止してスナップショットを作り、検証版へupgradeした。重みcacheの721 update、固定prefix状態の24 update、prefix登録の32 updateは準備として別計上した。

試験後はスナップショットから元のmodule `6052cc94…`、重みcache、pack状態へ復元し、それぞれの一致を確認して再開した。スナップショットを削除し、作成した試験callerを停止した。mainnetは操作していない。

最初の準備試行は検証スクリプトのcaller残高の確認先を修正するため中断した。その試行も元の状態へ復元済み。完了した測定は `proof-v2` に保存した。

## 証跡と再実行

- `artifacts/paid-token-limit-local-v1/build/report.json`: ビルド、変更箇所、kernel照合。
- `artifacts/paid-token-limit-local-v1/proof-v2/report.json`: 全7入力の要求、結果、料金、計測、query比較、復元結果。
- 同directoryの `audit.json`、`existing-reference-84.json`、`calls/`、`query-reference-*/`: raw応答と数値の照合。

別の新しい出力directoryを指定して実行する。`--source-root` はモデル素材、凍結runtime、helper binariesと既存ローカルネットワークがある元checkoutを指す。新しいローカルネットワークの作成やモデルのuploadはこのdriverには含めていない。

```sh
python3 scripts/build_paid_token_limit_probe.py \
  --source-root "$PWD" \
  --directory artifacts/paid-token-limit-local-v2/build

python3 scripts/prove_paid_token_limit_local.py \
  --source-root "$PWD" \
  --build artifacts/paid-token-limit-local-v2/build \
  --directory artifacts/paid-token-limit-local-v2/proof
```
