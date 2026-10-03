# RoPEの共有とQ/K正規化のquery統合

2026-10-02。同じ値の繰り返し計算を除く方向で、RoPEの周波数powfをrotary次元ごとに1回、sin/cosをpositionごとに1回計算し、複数headで共有するようにした。元のF32引数、乗算・加算・減算順、BF16丸めは保存した。固定A_logのexpもtokenループから外し、headごとにquery内で1回だけ計算する。

`norm_rope_heads_bf16`はQ/KのBF16 RMSNormとpartial RoPEを1通常queryで計算し、正規化済み配列の往復を省く。入力と出力はhead-major。clientは配列を並べ替えるだけで、正規化や回転はcanisterで行う。raw `rope` / `rope_heads`も共有計算へ変更した。既存の単一head RoPEのdimension上限を維持し、fused版はtextモデルのtokens<=512、width<=256、heads<=16、900,000 floats、encoded frame 2,000,000 bytesを厳密に確認する。非finiteな計算結果も拒否する。

実装は `crates/imajev-runtime/src/rope.rs`。`--fuse-norm-rope`でquery統合を有効にし、既存のlossless codecを要求する。raw operationの共有計算とA_logのexp共有は通常版にも適用する。sessionにfusion markerを含め、source/module/encodingに束縛された新prefixとjournalで再検証する。weight pack、adapter、readout、tokenizer、calibration、INT8 block256算術は同じ。

Rust39 / Python42 tests通過。scalarの元式との角度・丸め・tail比較、position-zero norm、異常metadataと非finite回転の拒否を確認した。固定4B packを持つ実canisterで、5つの以前の全層実行から13種類の実Q/K shape・positionを取り出し、旧RoPE・統合norm/RoPEとも保存済み旧出力に全bit一致した。87-token・16 Q headsは365,432,426→187,027,508命令（48.820%減）、4 K headsは91,637,222→47,533,521（48.129%減）。これは当該operationの値であり、全モデルの削減率ではない。

新prefixから全5実行を通し、保持hidden/stateと最終hidden/logits/確率/unknown/判断が前版delta-reuse-v1とbit一致した。prefix72状態配列、各問題48状態配列を比較し、terminal layerは使う最後のhiddenを比較した。失敗/replay 0、前後の認証済みmodule hash一致。

| 実行 | query：前→後 | handler命令数 | Candid通信bytes | 単回時間s | 最大query命令数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix45 | 298→282 | 205,688,262,872 | 318,918,016 | 38.361 | 2,284,521,085 |
| 主問題132 / suffix87 | 321→305 | 367,507,172,885 | 506,366,104 | 48.848 | 4,176,191,996 |
| 情報不足125 / suffix80 | 321→305 | 322,938,399,549 | 469,941,494 | 47.342 | 3,645,641,888 |
| 最大変更134 / suffix89 | 321→305 | 378,439,238,531 | 516,788,871 | 54.271 | 4,313,477,188 |
| cacheなし132 | 501→485 | 540,980,157,172 | 814,232,972 | 89.802 | 3,721,652,273 |

主問題はquery4.984%、命令0.437%、通信2.477%減。初回はprefix準備282＋推論305＝587 query（前619）。主問題のhandler終端heap最大観測46,923,776 bytesは同じ。各queryの命令数はCandid decode/encodeを除き、通信はCandid request＋replyでHTTP等を除く。時間は単回値で、速度向上を一般化しない。

50/32 queryは未達。準備済み主問題でも5B予算の最低74 query相当で、50回へさらに約32.0%の命令削減とquery統合が必要。初回準備を除いて達成扱いにしない。cacheなし132 tokenも485 query・約5410億命令で、50回へ約53.8%の命令削減が必要。既存のmaximum問題の誤判定は残り、判断精度・校正の改善を示す結果ではない。

採用Wasm `fe2762551294c903846410175e5f1f9d6af5200ff09ac4662b13ef4d805d3b10`。実測 `artifacts/rope-reuse-v1-*`、検証付き集計 `docs/rope-reuse-v1-summary.json`、13条件比較 `artifacts/rope-reuse/probe/report.json`。生成物はgitignore対象。Laya・mainnet・Git remoteは変更していない。

追加調査では、前版cacheなし132-tokenのrow分割に同一入力・同一tensorの重複射影62組を確認した。準備済み87-tokenの主問題ではこのrow分割の重複はない。大きい入力のLoRA A再計算をclient-held cacheで省く方式と、固定adapterを準備時に統合する方式を別々に検討する。後者は丸め・weight quantizationが変わるため、判断精度の評価なしに採用しない。

```sh
.venv/bin/python scripts/validate_terminal_readout.py --canister <専用local ID> --run-name <未使用run名> --baseline delta-reuse-v1 --fuse-mlp-norm --fuse-norm-rope --wire-codec bf16-block256-exact-v1
.venv/bin/python scripts/summarize_kernel_unroll.py --run-name <同じrun名> --baseline delta-reuse-v1
```
