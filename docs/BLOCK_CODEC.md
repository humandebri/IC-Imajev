# 数値を保持するblock256通信と最新の命令内訳

2026-10-02。`bf16-block256-exact-v1`をRust/WasmとPython clientへ実装し、固定4B packを持つ専用ローカルcanisterで全層検証した。256要素ごとのbitmapで、全BF16 blockは2 bytes/要素、F32例外を含むblockは元の4 bytes/要素を保存する。新たな丸め・量子化は行わず、F32再帰状態の全bitを保持する。元の可逆形式と、別のlossy INT8通信形式も残した。

値数900,000、encoded frame 2,000,000 bytes、header16,384 bytesの上限は変更していない。bitmapのpadding、完全長、正規形、checksum、finite値を検査する。F32例外が疎なblockは旧形式より大きくなる場合があり、サイズ超過は拒否する。CLIの`--wire-codec bf16-block256-exact-v1`で有効にする。演算融合・prefix cacheは双方の可逆形式を受け付け、再開時にはencodingを含むsession identityを照合する。形式変更時は新しいjournalとprefixを使う。

Rust36 tests、Python42 tests通過。実入力の代表的11演算を同じmodule上で新旧codecにより計算し、保存済み出力との全bit一致と、Rust/Pythonのcanonical payload一致を確認した。さらに新prefixから5条件を実行し、全保持hidden/state、最終hidden/logits/確率/unknown/判断が前版`mlp-norm-v1`とbit一致した。prefix72状態配列、各問題48状態配列を比較し、terminal layerでは使う最後のhiddenを比較した。失敗・replay 0、全実行の前後で認証済みmodule hashが一致した。

| 実行 | query | handler命令数 | Candid通信bytes | 単回時間s | 最大query命令数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix45 | 298 | 207,717,900,700 | 326,306,416 | 38.589 | 2,284,521,063 |
| 主問題132 / suffix87 | 321 | 371,307,078,099 | 519,228,106 | 52.043 | 4,176,191,974 |
| 情報不足125 / suffix80 | 321 | 326,433,820,340 | 481,771,044 | 60.694 | 3,645,641,866 |
| 最大変更134 / suffix89 | 321 | 382,321,344,223 | 529,945,857 | 69.410 | 4,313,477,166 |
| cacheなし132 | 501 | 546,414,888,500 | 833,732,106 | 124.395 | 3,721,652,273 |

主問題の命令数は377,303,015,436→371,307,078,099（1.589%減）、通信は549,890,193→519,228,106 bytes（5.576%減）。query数は変わらない。初回の主問題はprefix準備298＋推論321＝619 queryで、準備を除外して達成扱いにはしない。主問題のhandler終端heap最大観測は46,923,776 bytes。Candid decode/encodeの命令やHTTP等の通信は測定に含まない。

時間は単回測定で、後半は別target-dirでの計測版ビルドを同時実行した。cacheなしの時間増加をcodecの計算速度低下とは断定しない。命令数・通信量の削減と速度向上は別の主張である。既存のmaximum問題の誤判定も残り、判断精度や校正を改善した結果ではない。

通常版Wasm SHA256: `02c4243b9ff0af5419f8b5c07cf7175ab3884ad91b5913a774a7611d53c5b7f4`。model/packは従来固定値のまま。生データ`artifacts/block-codec-v1-*`、検証付き集計`docs/block-codec-v1-summary.json`、代表演算`artifacts/block-codec/probe/report.json`。生成物はgitignore対象。Laya・mainnet・Git remoteは変更していない。

## 現在のボトルネック

通常版の主問題では`lora_integer`35.64%、`mlp_add_norm_integer`34.69%、`delta_input_integer`12.06%で、射影を含む3群が計82.39%を占める。Delta stageは10.19%、残差chainは2.18%。通信の圧縮だけでは50 queryに届かない。

計測feature版を専用canisterに一時適用し、保存した実入力25形状を通常stepとprofile_stepで計算した。全出力は通常版とbit一致、前後module hash一致を確認。計測後に上記通常版へupgradeし、認証済みmodule hashで復帰を確認した。`profile_current_graph.py`はinstallを行わず、明示したmoduleと保存入力を検証する。

87-token MLP gate/upの内訳は以下。spanはinclusiveで、相互に重なるため合算しない。計測版の通常stepにもfeatureの費用があるので、全層の実測には通常版カウンタを使う。

| 計測span | 命令数 |
| --- | ---: |
| MLP profile全体 | 4,320,183,459 |
| base射影 inclusive（2回） | 2,916,684,190 |
| 整数dot（69,120回） | 2,537,326,080 |
| 整数scale/sum（69,120回） | 125,844,480 |
| base stable read（2回） | 108,773,673 |
| LoRA A matmul（2回） | 128,359,060 |
| LoRA B matmul（2回） | 495,122,490 |
| wire decode / encode | 80,954,793 / 144,177,040 |

この形状では整数dotがprofile全体の約58.7%で最大、LoRAの積和も約14.4%を占める。重み読み出しやactivation量子化だけを省いて大幅改善できる状態ではない。計測Wasm `99ab494642250f71582f8a9fb5de569b477572c1105fa91edc0316fdce911632`、実入力・計測費用・重なったspanを`artifacts/block-codec/profile/report.json`に保存した。

50/32 queryは未達。主問題の命令数だけでも5B予算の最低75回に相当し、50回へさらに約32.7%、32回へ約56.9%の命令削減とquery統合が必要。この値は達成可能性の保証ではない。

次は、(1)残差/normと次のQKV射影を同じqueryで計算し、(2)整数積和の演算方式と、事前変換Strassenの補正・復元費用を再検討する。block codecで87-tokenのQKV/gates＋hiddenのpacketが2 MBに収まる試算はあるが、940,992 floatsで現在の900,000上限を超える。この融合演算は未実装であり、専用の厳密な値数検査、2 MB検査、全層照合を導入してから回数削減に計上する。

## 再検証

```sh
.venv/bin/python scripts/validate_terminal_readout.py --canister <専用local ID> --run-name <未使用run名> --baseline mlp-norm-v1 --fuse-mlp-norm --wire-codec bf16-block256-exact-v1
.venv/bin/python scripts/summarize_kernel_unroll.py --run-name <同じrun名> --baseline mlp-norm-v1
```
