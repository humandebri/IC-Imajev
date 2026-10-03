# 同じ入力・Q/Kに対する重複計算の除去

2026-10-02。DeltaNet内で同じ値を再計算する箇所を除いた。重み・量子化方式・丸め・演算順・wire formatは変更せず、通常query内で既に計算した値だけを共有する。中間状態は引き続きclient保持で、推論をupdateへ移していない。

- 隣接する2つのvalue headは同じQ/K headを使うため、Q/KのgatherとBF16 RMSNormをpairの先頭で一度だけ計算し、二つ目でそのまま再利用する。pair以外のheadには共有しない。
- QKVとA/B gateの3射影は同じ入力なので、block256のactivation量子化を1回だけ実行して共有する。独立したgate queryでもA/B間で共有する。F32 LoRAには元のF32入力を使う。

固定4B packを持つ実canisterで、5つの以前の全層記録から20種類の実Delta shapeを選び、全出力が前版Wasmとbit一致した。87-tokenのDelta stageは689,770,107→657,462,840命令（4.684%減）、QKV/gateは1,865,777,579→1,848,344,567（0.934%減）、132-token gate単体は130,277,291→117,040,172（10.161%減）。

古いnative実行器とも比較し、整数QKV部分は全bit一致した。ただしF32超越関数等を含む全配列のnative/Wasm bit一致は成立せず、20形状の最大絶対差は1.52587890625e-5。たとえば45-tokenのQKVは一致する一方、F32 gateの122値に最大5.9604645e-8の差があった。これは前版Wasmとの全bit一致と別の比較であり、native横断で精度差0とは報告しない。

新prefixを作り直した全5実行で、全保持hidden/stateと最終hidden/logits/確率/unknown/判断が前版block-codec-v1とbit一致。prefix72状態配列・各問題48状態配列を比較し、terminal layerは使う最後のhiddenを比較した。失敗/replay 0、前後の認証済みmodule hash一致。Rust36/Python42 tests通過。

| 実行 | query | handler命令数 | 前版から減少 | Candid通信bytes | 単回時間s | 最大query命令数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix45 | 298 | 206,608,672,012 | 0.534% | 326,306,416 | 38.982 | 2,284,521,063 |
| 主問題132 / suffix87 | 321 | 369,120,738,555 | 0.589% | 519,228,106 | 50.750 | 4,176,191,974 |
| 情報不足125 / suffix80 | 321 | 324,421,914,068 | 0.616% | 481,771,044 | 47.476 | 3,645,641,866 |
| 最大変更134 / suffix89 | 321 | 380,089,754,527 | 0.584% | 529,945,857 | 54.266 | 4,313,477,166 |
| cacheなし132 | 501 | 543,416,648,084 | 0.549% | 833,732,106 | 86.598 | 3,721,652,273 |

主問題は371,307,078,099→369,120,738,555命令で約21.86億命令・0.589%減。query数321回、通信519,228,106 bytes、heap最大観測46,923,776 bytesは同じ。prefix準備298回を含む初回主問題は619 query。cacheなし132 tokenは501 query・543,416,648,084命令。handler counterはCandid decode/encodeを除き、通信はHTTP等を除く。単回時間から速度改善を断定しない。前版後半の時間には同時ビルドの影響もある。

50/32 queryは未達。準備済み主問題は命令数だけでも5B予算の最低74 query相当で、50回へさらに約32.3%の命令削減と演算統合が必要。初回のprefix準備を除いて達成扱いにしない。cacheなし132 tokenでも50回へ約54.0%の削減が必要。既存のmaximum問題の誤判定は残り、判断精度・校正を改善した結果ではない。

Wasm SHA256 `3a76af563902ea008c0c7581ba530b39c8b9d2f69a22ac5bb654051fd3363d06`。生データ `artifacts/delta-reuse-v1-*`、検証付き集計 `docs/delta-reuse-v1-summary.json`、20形状比較 `artifacts/delta-reuse/probe-v2/report.json`。生成物はgitignore対象。Laya・mainnet・Git remoteは変更していない。

別候補のStrassen補正をquery内で一度復元する実装は、同じmoduleの通常INT8より命令が80.5%増えたため不採用。[STRASSEN_WIDE.md](STRASSEN_WIDE.md)。固定重みの変換を準備時へ移す経路は未実装で、改善には計上しない。

次の重複除去対象は、RoPEの周波数powfと同じpositionのsin/cos、固定A_logのexp。現コードではtoken/headごとに再計算している。まずquery内で共有し、固定パラメータはモデル準備時の一度だけの計算へ移せるか検証する。残差/normと次層投影の融合も続ける。

```sh
.venv/bin/python scripts/validate_terminal_readout.py --canister <専用local ID> --run-name <新しいrun名> --baseline block-codec-v1 --fuse-mlp-norm --wire-codec bf16-block256-exact-v1
.venv/bin/python scripts/summarize_kernel_unroll.py --run-name <同じrun名> --baseline block-codec-v1
```
