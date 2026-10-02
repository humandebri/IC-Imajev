# Layaと比べてImajevの命令数が大きい理由

2026-10-01。Layaは既存の固定pack・レポート・ソースを読み取り専用で参照した。Layaのソース、Git、network、canisterは変更していない。Imajevの変更は専用のlocal canisterで測定した。

## 比較対象

追加調査でBOOM DAO 617の44-token・5-query実測と後続68-token・5-query実測を確認した。以下の128-token比較は歴史的な別入力として残す。現在の比較条件・追加削減・50-query予算は [FIFTY_QUERY_ANALYSIS.md](FIFTY_QUERY_ANALYSIS.md)。

当時比較したLayaの128-token実験 [INT8_QUERY_OPTIMIZATION_SEARCH.md](../../IC-Laya-Standalone/docs/INT8_QUERY_OPTIMIZATION_SEARCH.md) は128 tokens・3 markersで42,129,893,539 handler命令、17通常query、Candid通信4,230,208 bytes。旧V4の「32ステップ」は28 encoder層＋2 decision層＋前後の2段階という論理段数で、最新query版は2段ずつまとめるため17回である。32という段数をImajevの32層へそのまま当てはめても、query命令予算は同じにならない。

比較するImajevはBOOM DAO 617・132 tokens・rotations=1、head融合版で3,133,392,633,751 handler命令、2,132 query。総命令数はLayaレポートの約74.4倍だが、モデル・入力・graph・計測区間が異なる。これは同じ質問での速度や精度のA/Bではない。

## 原因を分ける

1. **モデル固有の積和数が大きい。** 固定packのshapeから数えると、Imajevの層内base dense行列は1 tokenあたり約35.69億MAC、Layaのencoderは約3.43億MACで約10.4倍。Imajevはhidden2560/MLP9216/32層、Layaはhidden1024/MLP2624/28 encoder層。embeddingはlookupとして除外し、Layaの別decision層やattention/DeltaNetの積和もこの比較には含めない。4B/421Mという総parameter比だけで演算量を決めていない。
2. **INT8の意味が違っていた。** Imajevの全モデル経路は、INT8重みをstableから読む→F32へ展開→F32積和→BF16境界丸めを行う。activation通信もINT8だが、これは整数積和への置換ではない。LayaはINT8 activationを保ち、I32 dotとscale/writebackを融合している。Imajevの `int8_matmul` は別prototypeで、全graphへ接続されていない。前回の命令数差をモデル容量だけで説明するのは不十分だった。
3. **ロード共有が少なかった。** ImajevのF32 SIMDは4 token×1出力行で、各出力行ごとに同じactivationをロードする。Layaは64 token×16出力行の整数タイルとループ展開でロードを共有する。加えてImajevでは内側ループのindex/overflow検査を含むアドレス計算が繰り返されていた。
4. **分割時にLoRA Aが繰り返される。** 保存した実tile計画ではbase約4,706億MAC、A約350億、B約95.5億。Aを同一入力ごとに1回へ減らすと約284.6億MACが不要になり、射影の実MAC総数の約5.52%に相当する。これは命令数・通信量の削減率ではない。AのF32値をclientに返して保持する場合の追加通信も必要になる。

head融合版の各queryを集計すると、**base＋LoRA射影が総命令数の92.97%**。DeltaNet2.20%、Attention中核1.22%。残る主因はheadごとのquery呼出しではなくdense射影である。shape・実query別の集計と参照hashは [laya-cost-analysis.json](laya-cost-analysis.json)。

## 今回採用した変更

Layaのロード共有を参考に、既存のF32積和を4 token×16出力行へ広げた。16個の独立したSIMD accumulatorは、元と同じ列順でF32の乗算→加算をする。FMA、横方向reduce、量子化scale変更、LoRA統合は行わない。行の基点pointerを先に作り、内側のindex乗算・加算を整理した。重み配置やquery入力/出力形式は維持する。

実重み・同じ実入力8件で比較した。8行、16行、16行＋pointer整理の全候補が以前の出力とbit一致。

| 射影の実shape [tokens, rows, cols] | 元の命令数 | 8行タイル | 16行タイル | 16行＋pointer整理 |
| --- | ---: | ---: | ---: | ---: |
| QKV [132,1236,2560] | 2,485,549,936 | 2,085,964,195 | 2,040,995,755 | 1,781,833,819 |
| out [132,756,4096] | 2,497,200,863 | 2,101,092,794 | 2,054,668,658 | 1,795,959,746 |
| down [96,441,9216] | 2,613,697,463 | 2,219,190,206 | 2,178,269,630 | 1,923,048,254 |

最後の候補は、7件のLoRA射影で約26.4〜28.3%削減した。32行の小さいgate射影では約14.5%削減。これは射影8件の実測であり、全モデルの削減率とは分ける。[全候補の実測](linear-tiling-candidates.json)。

BF16出力だけでは小さい誤差を隠し得るため、raw F32の49形状も実Wasmで検証した。token数1/3/4/5/7/36/132、出力行1/7/15/16/17/33/34、非ゼロrow offset、実readout重み、可逆通信を使い、native scalar順のF32出力と全bit一致した。[境界検証](linear-tiling-parity.json)。Rust workspaceの19 testsも通過した。

## 全32層の確認

同じ132-token質問をembeddingからreadoutまで新Wasmの通常queryだけで再実行した。

| 指標 | head融合版 | 今回の射影改善版 |
| --- | ---: | ---: |
| 総handler命令数 | 3,133,392,633,751 | 2,329,467,221,335 |
| query数 | 2,132 | 2,132 |
| Candid request+reply | 1,283,895,563 | 1,283,895,563 |
| 単回local時間 | 461.531秒 | 251.002秒 |

**総命令数25.66%減。** 全32層のhidden・conv/DeltaNet/KV state・raw logits・候補確率・unknownが改善前とbit一致した。typed choiceは有効でlikely。最大queryは2,389,001,610 handler命令、終端heap最大観測86,441,984 bytes。[全queryと比較](linear-tiling-full-results.json)。

queryの行幅・token幅・演算予算は変更せず、入力の量子化block境界を保ったため、query数と通信量は変わらない。測定時間の差にはcache/負荷も含まれる。INT8通信によるBF16通信版との候補確率差約0.121は、この実装改善で解消していない。

## 次の削減候補

- **整数積和を全射影へ接続する。** 最大の未採用候補。Layaの64×16/loop展開/scale融合を比較する。ただしImajevのblock256 scale、ties-to-even、F32 LoRA、公式BF16境界はLayaとは異なる。整数prototypeをそのまま挿すだけでは数値や判断が変わり得る。重みINT8・activation INT8・整数演算の差を別々に測る。
- **LoRA Aを行tile間で再利用する。** 上述の余分なMACを減らす。前計算queryとclient-held Aの通信を含めて評価する。
- **最終層の必要行だけ計算する。** Layaは最後のdecision層でmarker行だけを計算する。Imajevも専用readoutが読む最後のtokenについて、最終AttentionのQ/出力/MLPを省略計算できる候補。K/Vには全tokenが必要で、以前の層も全tokenが必要。現在は未実装で、INT8の通信block境界変更も検証対象。
- **shared prefixをclientに保持する。** 保存した23ホスト入力の共通prefixは45 tokensで、617の残りは87 tokens。87-token部分だけにしても32 queryの達成保証はない。DeltaNet stateとAttention KVを正しい絶対位置で継続する実装と、cache生成の初回費用・追加通信を報告する必要がある。1層のF32 Delta state約2.10 MBは、byte shuffle＋zlibのoffline試験で約1.68 MBへ可逆圧縮できたが、canister codec費用は未測定。

固定scale、近似exp/SiLU、LoRA merge、小型モデルへの変更は今回採用していない。通常query・client-held stateを保つ。32 query目標は引き続き未達。

## 再現

```sh
.venv/bin/python scripts/analyze_laya_cost.py
cargo test --workspace --offline
cargo build --release --target wasm32-unknown-unknown -p imajev-inference --offline
cargo build --release -p imajev-runtime --offline
# 自分で作成したImajev専用canisterをsealed pack維持でupgradeした後
.venv/bin/python scripts/check_linear_tiling.py
.venv/bin/python scripts/run_full_canister.py --wire-codec int8-block256-v1 \
  --delta-head-cap 16 --attention-head-cap 8 --directory artifacts/new-linear-tiling-run
```

候補A/Bは `scripts/benchmark_linear_tiling.py --phase NAME`。実測履歴を再測定するときは新しいphase名を使う。歴史的Wasm・ソース・要求/返信はignoredの `artifacts/linear-tiling/` に保存した。時間はcache/負荷未制御の単回local測定。命令数はhandler内、通信はCandid request/replyでHTTP/CBOR/signatureを含まない。
