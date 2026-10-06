# このリポジトリからLayaの命令削減へ移せる候補

2026-10-05。目的はLayaの命令数削減。共通runtimeへの移植は進めず、Imajevの既存最適化をLayaの数値契約に合わせて個別に検証する。

## 1. 最大値探索の整数bit化：単体で改善を確認

移植元は [QUANTIZE_PEAK_SCAN](../../QUANTIZE_PEAK_SCAN.md) と `crates/imajev-runtime/src/quantize_simd.rs`。Layaはすでに有限値検査とpeak探索を一走査で行うが、4要素ごとにF32比較、lane bitmask、分岐を行う。候補では絶対値のF32 bit patternをunsigned integerとして最大化し、走査後のpeakが `0x7f800000` 以上かを一度判定する。有限・非負のF32ではbit patternと値の大小順が同じなので、有限入力のpeakは変わらない。

Layaのtoken単位scale、最小normalのscale floor、zeroのscale=1、元のF32除算、half-away-from-zero、±127 clampを保持した。ImajevのRNEや最小subnormalの設定はコピーしていない。無限大・NaNは量子化前に拒否し、scalar tailも同じ判定に含める。

### Local ICでの比較

Layaの元SIMD関数と候補を同じ診断moduleへ入れ、同じ入力で交互に測定した。入力は合成F32。表は128 tokenの行単位量子化で、出力確保と診断用byte組立を含む。

| 1 tokenの列数 | 元Laya命令数 | unsigned peak候補 | 削減 |
|---:|---:|---:|---:|
| 1024 | 5,471,318 | 4,753,366 | 13.12% |
| 2624 | 13,818,518 | 11,986,966 | 13.25% |
| 4096 | 21,526,742 | 18,670,678 | 13.27% |
| 5248 | 27,537,526 | 23,879,670 | 13.28% |

1/7/32/65/128 token × 1024/2624/4096/5248列の20条件では6.86〜13.28%削減。全128条件で整数出力とF32 scaleのbyte一致、独立scalar参照との一致、非有限値の拒否を確認した。ICでも36条件・216通常queryで出力digestと拒否状態を照合し、各命令数は3回とも一致した。固有nonceでquery返信cacheを避けた。診断用canisterは終了後stop/delete済み。

[適用候補patch](../../../patches/laya/int8-unsigned-peak.patch) はこのリポジトリに保存し、現Laya checkoutに対する `git apply --check` が通過した。Layaのソース・Git・既存canisterには適用していない。

これは量子化区間の改善であり、全推論の13%削減ではない。Layaの保存済みprofileではLinear積和・書き戻し約301.7億命令が主要負荷で、Linear出力量子化は約11.37億命令。次は専用のLaya全モデル診断で96入力の元INT8 logits一致と通常query合計命令数を確認して採否を決める。

数値・hashは [laya-peak-results.json](laya-peak-results.json)。原記録は `artifacts/runtime-comparison/laya-peak-v1/{build.json,source.zip,node-report.json,diagnostic.wasm}` と `artifacts/runtime-comparison/laya-peak-ic-v1/report.json`。再現手順・測定区間は [harness README](../../../scripts/laya_peak_bench/README.md)。元関数と候補は同一の簡易Result/error shimを使い、非有限値のerror生成費用はCandleそのものではない。早期NaNでは元実装は途中で走査を止め、候補は行末まで走査するので、拒否時間まで改善したとは主張しない。

## 2. RoPE表の事前準備：次に測る候補

Imajevの `rope.rs::prepare_fixed` はsin/cos表を準備updateで計算し、queryで再計算しない。一方Layaの `quantized.rs` のAttentionは層ごとに `pos / theta.powf(2*i/d)` とF64 `sin_cos` を実行し、F32へcastした表を作る。表はhead間ではすでに共有されるが、層やqueryをまたいだ固定表にはなっていない。

固定Laya packはhead幅64、最大128 token、thetaがglobal 160000 / local 10000の2種類。両thetaの128×32個のsin/cosをF32で保存する場合、表payloadは計65,536 bytes。固定モデルのwarmupで同じWasm・同じF64式を用いて作り、各層で借用する候補になる。元の除算を逆数乗算へ変えず、powf・sin_cos・F32 castの順序を保持する。ImajevのF32の式やthetaは流用しない。違うhead幅・theta・position範囲は元経路へ戻す。

Layaの命令削減率は未測定。最初に表生成＋RoPE区間の同一module A/Bでbit一致と費用を測り、その後に全推論で確認する。準備に移した命令と推論命令は別計上する。

## 3. 整数S1とSIMD local共有：大きな負荷を狙うが移植が必要

このリポジトリの後期S1は、元容量INT8 quadrant、入力変換の一回準備、weight変換の最初のtoken pairからの再利用、整数再構成後の水平和、128出力への入力load共有、定数offsetのWAT展開を組み合わせる。[STRASSEN_RAW_REUSE](../../STRASSEN_RAW_REUSE.md)、[WAT_S1_REDUCTION_REUSE](../../WAT_S1_REDUCTION_REUSE.md)、[S1_OUTPUT_TILE_REUSE](../../S1_OUTPUT_TILE_REUSE.md)。

以前比較した `int8_token_kernel` はこの採用済みS1ではない。その候補がLayaより遅かったことを、S1にも効果がないという根拠にはしない。

Layaへ持ち込む場合は、K256ごとにF32へscaleして加算するImajevの処理を使わず、元の全列I32内積を最後まで正確に再構成し、最後にtoken scale→weight scaleを一度適用する必要がある。幅2624の端数と奇数token、bias、output group、再量子化も元と同じ契約で扱う。I16変換・I32再構成の上界をLayaの最大列数で検証する。7/8という積和数の比を命令削減率にはしない。

まずLayaの3実shape（3072×1024、5248×1024、1024×2624）と実activationでLinear区間を比較する。単tokenには直接dot、複数tokenには整数分解という選択も別に測る。大きな出力tileはICのlocals上限も確認する。効果は未確認で、量子化peakやRoPE表のような小さい候補より実装費用が大きい。

## 優先しない案

- **I16重み常駐、単純なtile変更、入力の先行I16化**：Layaで既に悪化して見送っている。条件を変える根拠なしに繰り返さない。
- **Attentionのまとめ内積、SIMD norm、head間scratch共有**：現Layaにすでに入っている。
- **最終decision層のmarker行だけ計算**：現Layaの `forward_markers` / INT8 marker経路で対応済み。
- **Imajevのprefix cacheをそのまま移す**：Laya encoderでは後続tokenが前方tokenのhiddenにも影響する。入力内容が同じというだけでprefix hiddenを再利用できない。
- **W4 LUTや疎補正付き事前Strassen**：このリポジトリの実測では命令が大幅に増えた。量子化誤差も伴う候補を最初には選ばない。
- **schedulerのqueryを詰めるだけ**：query回数と総命令は別。主要演算を省かない限り、総命令の大幅削減として扱わない。

Laya側の既存試行・profileは [INT8_QUERY_OPTIMIZATION_SEARCH](../../../../IC-Laya-Standalone/docs/INT8_QUERY_OPTIMIZATION_SEARCH.md) と [INT8_V4_REJECTED_TRIALS](../../../../IC-Laya-Standalone/docs/INT8_V4_REJECTED_TRIALS.md) を参照した。今回の探索ではLaya checkoutは読み取り専用である。
