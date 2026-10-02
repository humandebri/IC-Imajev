# 命令ボトルネックの計測と追加改善

2026-10-02の追加改善は[MLP_FUSION.md](MLP_FUSION.md)。256列dot・MLP融合・広い分割で708 query、0.739兆命令へ削減した。以下はその前段階の計測。


2026-10-01。対象は固定Imajev-4B INT8 pack、元F32 LoRA・専用readout・calibration、通常query、client保持の中間状態。新しい量子化は追加せず、前回の整数演算の数値を保存する改善を実装した。Layaは読み取りだけ。計測用・採用版とも専用local canisterを使用した。

## 全モデルの測定

BOOM DAO 617、132 tokens、rotations=1。CDKのCandid decode/encodeを除くhandler命令数と、HTTP/CBOR/署名を除くCandid通信量。

| 指標 | 前回整数版 | SIMD＋token整列 | 残差/norm融合も適用 |
| --- | ---: | ---: | ---: |
| 総命令数 | 1,022,432,919,278 | 851,822,790,043 | **846,799,463,736** |
| query数 | 932 | 932 | **868** |
| Candid通信bytes | 1,670,790,371 | 1,670,790,371 | **1,624,778,979** |
| 単回local時間 | 107.274秒 | 102.764秒 | **81.145秒** |
| 最大query命令数 | 3,822,049,569 | 3,055,392,959 | **3,055,392,959** |
| handler終端heap最大観測 | 83,099,648 | 82,968,576 | **82,968,576 bytes** |

前回比17.18%命令減、query6.87%減、通信2.75%減。当初の全モデル3.682兆命令からは77.00%減。時間はcache/ホスト負荷を制御しない単回値で、速度倍率の一般保証には使わない。heapは瞬間peakではない。

追加改善の両段階で全32層hidden・conv/DeltaNet/KV state・最終logit・確率・unknownが前回整数版と全bit一致。これは前回整数方式への追加誤差がゼロという検証であり、公式未量子化や元F32方式へのbit一致ではない。以前の判断ミスや選択肢順依存も修正したとは扱わない。[全query・bit比較](bottleneck-fused-results.json)、[SIMD段階](bottleneck-results.json)。

情報不足125 tokensでも全32層・state・logit・確率は旧整数方式と全bit一致し、unknownを維持した。867 query、802,440,096,369命令、1,542,968,292 bytes、単回85.285秒。[情報不足のbit比較](bottleneck-insufficient-results.json)。重大変更134 tokensも全bit一致、868 query、862,839,843,171命令、1,648,153,211 bytes、単回93.990秒。[重大変更のbit比較](bottleneck-maximum-results.json)。重大変更への誤ったnoという既存の見逃しもそのまま残る。型が有効であることと判断が正しいことを分ける。

検証はRust22件、scheduler4件、wire3件、INT8 wire4件、journal4件、不正整数契約7件、実重み整数射影96ケース、量子化境界16ケース、3入力の全32層bit比較。

## どこに命令を使っていたか

932 queryすべてのcounterを演算別に集計した。LoRA付き射影77.01%、DeltaNet7.10%、SwiGLU4.08%、convolution2.46%。最初に射影を掘り下げ、stable read、activation quantization、INT8→I16 weight展開、整数dot、block scale/sum、LoRA A/BのloadとF32積和、wire encode/decode、SHA-256へspanを分けた。

profilingは独立した`instruction-profile` buildとowner-only通常queryで行う。各spanはinclusiveなので、base_projectとinteger_dotなどを重ねて加算しない。最初の計測buildではcompiler最適化も変わり、同じ入力の通常stepが保存済みbaselineより最大約19%重くなった。この初期spanは原因探索に限定し、削減量には使用しなかった。採用版の全モデルcounterはfeature無効で、span hookをcompile時に除去して測定した。最終profiling版は別Wasm hashとして記録する。[counter・候補・計測影響](bottleneck-profile.json)。

## 採用した改善

1. **block scale/sumのSIMD化。** 4出力行をI32→F32へ変換し、activation scale、weight scale、既存sumへの加算を別命令で実行する。元のF32演算順とblock順を保持。unsafe loadの前に、行数、padding、scale配列、finite、値域を検証する。
2. **integer horizontal sumのSIMD化。** dotの4 laneをshuffle＋I32加算で集約する。256×127×128という上限の内側で、整数の結合法則だけを使用する。
3. **lossless BF16 codec。** bitmapが全ゼロのframeは8値単位でpack/unpackする。混在F32は従来処理へ戻す。frame format、SHA-256、canonical/padding/finite検証は保存する。SHA検証がwire処理の約6〜7割だったため、codecだけの改善には上限がある。
4. **block256の量子化をSIMD化。** maxabs、元の除算、nearest ties-even、clampを同じ順で実行。逆数乗算へ置換しない。ゼロ・signed zero・RNE境界・subnormal・大きなfinite値16ケースでもnative scalarとbit一致。
5. **token分割の8境界整列。** cols9216の入力上限で97 tokensを取ると104 tokens分のdotを計算していた。96＋36へ分け、padding合計144→136へ減らす。末尾の有効tokenはすべて保持。新しいsession identityを記録し、旧journalとの混在を拒否する。
6. **残差加算とRMSNormのquery融合。** BF16丸め後の残差sumをRMSNormへ渡し、sumとnormの両方を返す。post-attention norm、次層input norm、最後のnormを融合して64 queryを削減。層境界のnormは前層の最後のqueryへ計上する。中間状態は引き続きclientが保持する。

8/16/32/64-token整数タイルを同じ入力で比較し、主要shapeは64が最小だった。大タイルで一時配列負荷が増える懸念を実測したが、小タイルへの一律変更は不採用。35-token tailだけ32が微差で有利だったが、整列後は36-tokenとなり、採用版は64/32/16/8 dispatchを維持した。候補は全出力bit一致、各source/Wasm hashを保存した。

## 残る負荷と制約

最終profilingで376本すべての整数射影shapeをカバーし、代表queryのspanをshapeの出現数へ展開した。以下は**標本からの内訳推定**であり、別の全モデルcounterではない。sample通常stepの合計と実測済み全モデル内の対応stepの差は41,382,013命令（総量の約0.0049%）。profiling spanのoverheadは各caseにも記録した。

| 整数射影内のstage | shape展開した命令数 | 全モデル命令数に対する比率 |
| --- | ---: | ---: |
| integer dot | 456,697,010,560 | **53.93%** |
| LoRA A F32積和 | 25,703,800,152 | 3.04% |
| LoRA B F32積和 | 31,378,547,976 | 3.71% |
| INT8 weight展開 | 18,314,813,888 | 2.16% |
| block scale/sum | 11,793,587,840 | 1.39% |
| base stable read | 9,969,647,392 | 1.18% |
| activation量子化 | 6,001,820,784 | 0.71% |
| 射影入力/出力SHA-256 | 48,229,179,392 | 5.70% |

最終profiling Wasm hashは`e0be3fc2661c2565a89ed9f7558f12d71108e9518199531404e225728cb30943`。計測後は実測済み採用Wasmへ戻し、installed hashを確認した。量子化や重みreadのさらなる改善だけでは大きい削減は得にくく、次の主対象はinteger dotと演算境界のactivation再送になる。

採用後もLoRA付き射影が75.89%、DeltaNet8.59%、SwiGLU4.11%。重み読み出しだけを改善しても全体を大幅には減らせない。整数dotと、演算間で再送・SHA検証するactivationが次の対象。専用readoutやJSON型処理は主要負荷ではない。

**50 queryは未達。** 0.847兆命令は50億の枠に計算だけを詰めても170回相当。50回へはさらに約70.48%の総命令削減が必要。projection間のquery融合だけではこの命令予算は解決しない。次の候補はdotのload/accumulator更新回数削減、同じ入力からの複数射影で量子化・LoRA Aの再利用、正しい共有prefix state cache。query/blob/メモリ上限、初回準備費用、fallbackを含めて測定する必要がある。

## 再現

```sh
cargo test --workspace --offline
cargo build --release --target wasm32-unknown-unknown -p imajev-inference --offline
# 自分の専用local canisterへupgradeしてから、新しいdirectoryで実行
.venv/bin/python scripts/run_full_canister.py --arithmetic int8 \
  --wire-codec bf16-exact --compact-lossless --fuse-add-norm \
  --delta-head-cap 8 --attention-head-cap 8 --row-cap 8192 \
  --work-cap 2500000000 --token-cap 132 --directory artifacts/new-bottleneck-run
.venv/bin/python scripts/check_integer_parity.py
.venv/bin/python scripts/check_simd_quantization.py
# 細粒度計測用の別build。全モデル採用値には混ぜない
cargo build --release --target wasm32-unknown-unknown -p imajev-inference \
  --features instruction-profile --offline
# 計測版へupgrade後に実行し、終了後は採用版へ戻す
.venv/bin/python scripts/profile_bottlenecks.py --phase new-profile
```

採用Wasm SHA256: `32cf7c07b143e7af192d1b4d15add5197296d9e369002486b2edea45c688c03e`。adapter/base/tokenizer/readout/calibrationとpack hashは前回と同じ。
