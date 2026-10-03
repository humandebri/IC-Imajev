# 分割MLPで入力量子化と2つのLoRA Aを再利用

2026-10-02。prefixなしの132-token経路に残る31組のMLP分割で、同じ入力のblock256量子化とgate/upそれぞれの元F32 LoRA Aを再計算しない実装。元INT8 base、scale、F32 A/B、BF16投影/SwiGLU境界、readout/calibrationは維持する。新たな量子化は行わない。

## 通常queryとクライアント状態

新しい`mlp_gate_up_capture`の最初の出力tileに、量子化済みINT8・raw F32 scale・gate A結果・up A結果を付けて返す。`mlp_gate_up_reuse`は同じ入力に対する次の出力行でこのstateを受け取る。中間状態はclient-held、canisterに質問依存のstateを残さない。固定重み準備だけupdate、推論は通常queryのまま。

`projection-block256-exact-v1`の既存tag0/1/2形式をこの2 opにも接続した。dimsは`[tokens,rows,cols,row_start,rank]`。MLPのtailだけA結果を2組持つ。gate/upのbase・A・B shape/rank・dtype、対応する名前、combined work4.5G、frame2M/header16K/900K valuesの上限を検査する。prepared reuse入力はINT8から直接I16へ復元し、F32 scales/Aは元bitのまま保持する。不正整数-128、nonfinite/zero scale、nonfiniteの両方のA結果、不正tag、長さ違いを拒否する。private stateは復元後のrequest全識別/演算fieldへ結び付ける。

クライアントは`--reuse-projection-inputs`が有効でrow分割が必要、rank>0、add/norm融合を使わず、captureを返すためのtile縮小によってquery数が増えない場合だけ選ぶ。現132-tokenの5704+3512行は4120+5096行になり2queryを維持する。短い87-token等の一括MLPは従来opを使う。instruction failureでは出力幅だけを縮め、未成功の進捗/stateを確定しない。codecはtry/finallyで戻す。sessionのreuse方式を`client-held-int8-f32-a-and-mlp-v2`へ更新し、旧checkpointとの混在を避ける。

## nativeと専用部分canister

Rust feature有効runtime46＋cache3＋compile-fail doctest2、通常feature runtime43＋cache3＋doctest1が通過。Python scheduler13、wire4、block2、INT8 wire4、journal4も通過。syntheticの1/7/8/32/132 token・非ゼロ行で両Aのbit保持と従来MLPとの一致、sealed up A rank違いの重み読出し前拒否を検査した。別途実入力7長で独立旧nativeとbit一致し、Rust/Python frame再encodeもbyte一致した。

元layer0 gate/up9216×2560のINT8と元rank64 F32 A/Bを1つの部分pack（53,288,960 bytes）に固定した。pack hash `2026235523bc842df5590dcb6c746342c2ed9dfe1d7b638d91b7767fa881f189`。新設した専用local canister `4xhad-gd777-77775-aaacq-cai`を使い、QKV検証canisterやLayaを変更していない。

cold/warm/cleared×7入力長×4query=84通常queryで独立旧native出力とbit一致。別の6queryでreserved integer、zero scale、tag、truncated、gate A NaN、up A NaNを送り全て拒否。元の2行境界を両方式で揃えた比較なので、全層で行境界を変えた結果とは分けて評価する。

| tokens | 従来2query命令 | capture/reuse命令 | 削減率 | 従来Candid bytes | reuse Candid bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 145,402,650 | 132,142,781 | 9.1194% | 30,964 | 32,091 |
| 7 | 592,627,553 | 547,462,378 | 7.6212% | 203,040 | 210,771 |
| 8 | 389,384,174 | 381,100,786 | 2.1273% | 231,718 | 240,550 |
| 32 | 1,416,306,255 | 1,390,939,666 | 1.7910% | 920,018 | 955,262 |
| 64 | 2,792,954,418 | 2,743,898,382 | 1.7564% | 1,837,746 | 1,908,206 |
| 87 | 4,075,718,111 | 3,964,671,472 | 2.7246% | 2,497,364 | 2,593,135 |
| 132 | 5,859,473,258 | 5,757,690,040 | 1.7371% | 3,787,923 | 3,933,248 |

132-token warmは101,783,218命令（1.7371%）減だが、通信は145,325 bytes増（約3.8%）。初回captureの返送stateがあるため、計算を省けば通信も減るとは扱わない。単回時間はquery cache未制御で速度向上の根拠としない。

module `752a35b413fc6ec790cd8737a673c374132748ab825483d2073242fa91e3b480`を部分試験の前後に認証hashで確認。生記録は`artifacts/mlp-reuse/{native,probe,upload}`、候補Wasm/native/deploymentも同directoryに保存しgitignore。scopeは1層のgate/upで、全モデルquery数や一般判断精度の証明とは扱わない。

```sh
.venv/bin/python scripts/check_projection_reuse.py --mlp --compact-state \
  --verify-malformed --native artifacts/mlp-reuse/primitive \
  --canister 4xhad-gd777-77775-aaacq-cai \
  --wasm artifacts/mlp-reuse/full.wasm --directory artifacts/new-mlp-probe
```

部分packを選択canisterへ事前upload/sealしておく必要がある。対象は必ず検証用のlocal canisterを指定する。

## 全32層の実測と採用

全層moduleを`experimental-projection-reuse,experimental-full-weight-cache`でbuildし、主local canister `4caro-hl777-77775-aaaba-cai`へupgradeした。元sealed packを保持し、721 tensor・4,065,416,192-byte cacheを準備updateで復元した。準備は15,027,997,708 handler命令、229.9895秒、Candid要求47,473／返信14,915,249 bytesで、命令とbytesは前版と同じ。pack status1、cache status2、認証module read2は別計上。各質問では再準備しない。

新しい45-token prefix、主問題・情報不足・最大変更、prefixなし132-tokenの全5実行を従来`direct-projection-v1`と比較した。prefixは全32層hiddenと72 state arrays、問題は前31層の全hidden＋最終層の必要なlast-token hiddenと48 state arraysでbit一致。final hidden・logits・確率・unknown・型付き判断も一致した。最終層の省略済みtokenを含めた全hiddenの比較とは扱わない。失敗/replayは全て0、各実行前後の認証module、model/pack、全721 cache tensorとbyte数、検証対象ソース32 hashも照合した。

| 実行 | 通常query | handler命令 | Candid bytes | 単回秒 | 最大query命令 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix | 282 | 193,719,805,173 | 318,918,016 | 27.1830 | 2,134,116,392 |
| 617 | 305 | 354,704,765,427 | 506,366,104 | 45.8761 | 4,021,006,086 |
| insufficient | 305 | 310,182,042,285 | 469,941,494 | 41.8721 | 3,489,466,937 |
| maximum | 305 | 365,457,606,400 | 516,788,871 | 50.2477 | 4,157,303,672 |
| normal | 485 | 519,494,180,139 | 821,148,886 | 72.1781 | 3,083,017,050 |

prefixなし132-tokenでは522,650,865,435→519,494,180,139命令、追加3,156,685,296命令（0.6040%）減。31組のMLP capture/reuseを追加し、既存QKVの31組も維持した。485 queryは同じ、Candidは816,643,811→821,148,886 bytes、4,505,075 bytes（0.5517%）増。最大queryは3,612,834,057→3,083,017,050命令に減ったが、これだけでquery統合が完了したとは扱わない。

prefix準備済み主問題にはこのMLP再利用対象が0で、305 query、354,704,765,427命令。前版との差は-13,523命令（0.0000038%）で、実質的な高速化として報告しない。初回prefix込み587、prefixなし485 query、50/32の目標は未達。5B/queryを使い切る理想値でも現在のhandler命令だけで主問題71、初回prefix込み110、prefixなし104 query以上が必要。query統合と整数dot、毎queryの検証/codec費用の削減が引き続き必要である。

終端heap観測最大4,115,005,440 bytesで前版と同じ、瞬間peakは未測定。単回時間はquery cache未制御で速度改善の証拠にしない。handler counterはCDK Candid decode/encodeを、Candid bytesはHTTP/CBOR/signatureを含まない。元モデルの最大変更gold=yes/出力noは残る。bit一致は採用済み整数実装から数値劣化がないことを示し、一般的な判断精度の改善を示さない。

本module `752a35b413fc6ec790cd8737a673c374132748ab825483d2073242fa91e3b480`と721 cacheを主canisterに保持した。Layaのソース/Git/稼働canister、mainnet、remoteは変更していない。生結果は`artifacts/mlp-reuse-v1-*`、要約`docs/mlp-reuse-v1-summary.json`、source/cache照合`artifacts/mlp-reuse-v1-cache-checks/report.json`。生成物はgitignore対象。

```sh
.venv/bin/python scripts/validate_prepared_weights.py \
  --canister 4caro-hl777-77775-aaaba-cai --run-name mlp-reuse-v1 \
  --baseline direct-projection-v1 --wasm artifacts/mlp-reuse/full.wasm \
  --preparation artifacts/mlp-reuse/full-preparation --reuse-projection-inputs
```

既存の固定pack、fixture、比較元artifact、完了済み準備reportが必要。次は入力/出力frameごとに繰り返すchecksum処理の命令数を測る。[BLAKE3公式実装](https://github.com/BLAKE3-team/BLAKE3/blob/master/Cargo.toml)にはWasm SIMD featureがあるが、今回のcodecに実装したり削減として計上したりしていない。model/pack lockのSHA256は維持し、frameの代替案は独立に測定する。
