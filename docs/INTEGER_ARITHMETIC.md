# 整数演算を全モデルへ接続した結果

この後、同じ整数方式の数値を保った追加改善で0.847兆命令・868 queryへ削減した。[最新の計測・改善](BOTTLENECKS.md)。以下は整数方式導入時の履歴。

2026-10-01。固定INT8 baseのI32 dotを全dense射影へ接続した。専用local canisterのみ使用し、Layaのソース・Git・canisterは参照だけ。通信は可逆BF16のままにして、既存のINT8通信誤差と混ぜない。

## 全32層実測

BOOM DAO 617、132 tokens、1質問、rotations=1。embeddingから専用decisionまで通常queryで完走。

| 指標 | 直前のF32演算 | 整数base演算 |
| --- | ---: | ---: |
| handler総命令数 | 1,761,784,928,449 | 1,022,432,919,278 |
| query数 | 1,116 | 932 |
| Candid request＋reply | 1,854,531,015 bytes | 1,670,790,371 bytes |
| 単回local時間 | 213.376秒 | 107.274秒 |
| 最大query handler命令数 | 3,972,106,554 | 3,822,049,569 |
| handler終端heap最大観測 | 169,607,168 bytes | 83,099,648 bytes |

**命令数41.97%追加減。** 最初の全モデル3.682兆命令からは72.24%減、query3,908→932。時間はcacheとホスト負荷を制御していない単回値。counterはCDK decode/encodeを除き、通信はHTTP/CBOR/署名を除く。heapは瞬間peakではない。[全query・精度差](integer-full-results.json)。

## 数値差と判断精度を分ける

整数経路では入力をtokenごと・input列256ごとにmaxabs/127、RNEでINT8へ変換する。これはbase射影の内部に限定する。LoRA A/Bには元のF32入力を渡し、既存F32積和とBF16丸め境界を維持する。F32 recurrent state・gate・readoutをINT8へ変換しない。重みのquantization/pack/hash、adapter、tokenizer、calibrationは同じ。

各blockのI32 dotは正確、F32へ変換してactivation scale→weight scaleを別々に乗算し、block順にF32加算する。元の列ごとのF32加算とは違う。この全層617の候補logit差最大0.18749、候補確率差最大0.01466（約1.47ポイント）。判定はlikely、typed outputは有効。元のF32演算とのbit一致や校正維持は主張しない。

公式の同じprompt・LoRA/readout/calibration・nonlinear graphを使い、固定INT8 packについてF32と整数のホストA/Bを23件取得した。10質問に対する独立した選択肢順変更で、rotations=1を平均していない。**23/23ラベル一致**、公式未量子化参照とも23/23一致。元順序の正解付き7問は両方式6/7。情報不足はunknown、未変更はno、最大ロック期間の重大変更をnoとする既存の見逃しは残る。620の選択肢順依存も残る。

ホスト比較のembeddingもcanisterと同じ固定INT8 packを使い、F32 dequantization後にBF16へ丸めた。採用記録は`host-integer-orders-pinned-embedding.json`と`host-pack-f32-orders-pinned-embedding.json`。旧記録はembeddingだけ公式BF16だったため、同じpackのA/Bから除外し、履歴として保存した。旧条件の確率差10.46ポイントを、全embedding固定後の8.48ポイントへ訂正した。

一方、同じpackのF32ホストと整数ホストで確率差は最大0.08483（約8.48ポイント）。ラベル一致だけで一般精度・校正・確率の同等性を認定しない。ホストとWasmのnonlinear実装差も残る。検証範囲は [integer-judgments.json](integer-judgments.json)。ローカルcanisterでも620の元順序/offset2、maximum、unchanged、insufficientの5 forwardを完走した。5/5公式ラベル一致、全typed output有効。maximumは誤ったnoを維持し、620のlikely/possibleという順序依存も維持、情報不足はunknown。[canister診断](integer-canister-diagnostics.json)。

## 演算・境界検証

整数タイルは最大64 tokens×16出力行。列32個分のdotを整数でまとめ、accumulatorへの更新回数を減らす。token端数は32/16/8、出力行端数は8。blockは256なので最大整数絶対和は256×127×128でI32内。q bufferのpadding、scale数・正値・finite、activation範囲をunsafe load前に検証する。

最初の64×16・8列更新は旧628,986,326→660,743,160と増えたため不採用。32列展開の初期prototypeは568,971,960で9.54%減。融合dispatchを含む最終Wasmでは571,100,907（旧prototype628,986,326比9.20%減）、同じ旧整数出力とbit一致。初期測定のlocal target hashは実行中の再buildで変わっていたため、保存済みinstalled prototypeのhashと区別して訂正した。最終全層実測はsource/build/installを固定した別記録。32-token候補は577,440,267と不利。O4候補は同じ最終ソースの未最適化版571,100,907に対して570,337,116、約0.13%減だった。全graphの再測定をしていないため採用扱いにせず、最終canisterは全層実測済みの未最適化Wasmへ戻した。候補module/source/reportはignored artifactsに保存した。

非ゼロrow offset、実重み、token1/7/8/9/16/31/32/33/63/64/65/96/124/128/132/134、行8/16/24で、raw整数scale出力とF32 LoRA融合出力をnative scalar oracleへ比較した。96ケースすべてbit一致。不正なrow幅・cols・LoRA metadata・row範囲・A形状・work超過の7条件も実canisterで拒否を確認した。[契約検証](integer-contracts.json)。[integer-parity.json](integer-parity.json)。これは旧F32演算への一致ではなく、新整数方式の正しい実装を検証したもの。

新opはowner-onlyの既存`step`内の`lora_integer`／`linear_integer_bf16`。dims `[tokens, rows, cols, row_start]`。LoRAありはaux A/B、scalars `[2.0]`、なしは両方空。n<=512、rowsは8の倍数、colsは256の倍数、出力900k float、重みtile30M bytes、base＋LoRA4B MACまで。既定のrunnerはF32を維持し、整数方式は明示指定する。整数方式のtoken cap既定値は512、F32は従来の132を維持する。入力float/blob、work、weight tileの上限で実際の幅をさらに制限する。失敗queryは行幅を8行単位で半減し、client journalへ別記録する。


### token分割の急増を解消

134-tokenのmaximum問題では従来token cap132で末尾2 tokensの射影やDeltaNet継続が増え、1,356 queryだった。整数方式のtoken capを512へ広げ、各演算のfloat/blob/work上限で制限すると932 queryへ減った。命令数1,101,071,001,651→1,035,678,570,016、通信1,801,300,407→1,694,860,923 bytes。全32層hidden・conv/DeltaNet/KV状態・logit・確率が旧整数方式と全bit一致。選択noという既存の誤りは残る。[全query・bit比較](integer-token-cap-results.json)。

132-tokenの最初の932-query実測はtoken cap132で取得した履歴であり、再現には明示する。512の既定値で過去のjournalを開くと設定不一致を拒否するので、新しいdirectoryを使うか旧cap132を指定する。

## 50 queryまでの残り

**50 queryは未達。** 現在の総命令数1.022兆をそのまま50億の枠に詰めても、計算だけで最低205回相当。932回との差にはstage・blob・メモリによる分割の余地があるが、それをまとめるだけで50回にはならない。50回の2,500億命令予算へは、総命令数をさらに約75.55%削る必要がある。

候補は共有prefixの正しいDeltaNet/KV cache、射影と後処理のquery内融合、量子化block/scale融合の整数カーネル改善。準備・初回cache費用も含めた実測、情報不足・選択肢順・僅差ケースの判断確認が必要。近似や量子化の追加を無条件に精度維持と扱わない。relaxed SIMDの高速dotは公式ICの[validation設定](https://github.com/dfinity/ic/blob/master/rs/embedders/src/wasm_utils/validation.rs)で無効なので、それを前提にした速度は計上しない。

## 再現

```sh
cargo test --workspace --offline
cargo build --release --target wasm32-unknown-unknown -p imajev-inference --offline
cargo build --release -p imajev-runtime --offline
# 自分の専用local canisterをsealed pack維持でupgradeした後
.venv/bin/python scripts/check_integer_parity.py
.venv/bin/python scripts/run_full_canister.py --arithmetic int8 \
  --wire-codec bf16-exact --compact-lossless \
  --delta-head-cap 8 --attention-head-cap 8 --row-cap 8192 \
  --work-cap 2500000000 --token-cap 132 --directory artifacts/new-integer-run
HF_HUB_OFFLINE=1 .venv/bin/python scripts/host_integer_reference.py \
  --arithmetic f32 --orders --output artifacts/new-host-pack-f32.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/host_integer_reference.py \
  --arithmetic int8 --orders --output artifacts/new-host-integer.json
.venv/bin/python scripts/evaluate_integer_canister.py \
  --directory artifacts/new-integer-diagnostics
```

ホスト検証はMetalへのアクセスが必要。各モデル・pack・入力のhashを記録し、既存参照を上書きしない。最初の保存済み全層実測Wasm SHA256は`f4e8f5e4936b33c97869d7117e85e0f64514d044ed6ef9e2c7f2abdd730a9048`。
