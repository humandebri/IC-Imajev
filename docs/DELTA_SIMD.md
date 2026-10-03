# Delta SIMDの係数共有

2026-10-02。DeltaNetの再帰計算を、4要素ごとの処理から16要素をまとめる処理へ変更した。四つのSIMD vectorに対してK/Q係数のloadとsplat、key軸のループを共有する。key順のF32積和、decayの丸め、BF16出力、F32状態、重み、LoRA、tokenizer、readout、calibrationは保存した。16要素に満たない端は従来の4要素経路を使う。主モデルのvalue幅は128。

診断用canisterのsynthetic recurrence 12条件（token 1/45/87、value幅4/12/16/128）で、以前のWasmとnative scalarの両方に出力・状態が全bit一致した。87 token・幅128の命令数は73,671,620→52,611,247（28.59%減）。これは再帰単体の測定であり、全モデルの削減率ではない。幅4の小条件では最大0.40%の命令増があった。

固定4B packを保持した主canister `4caro-hl777-77775-aaaba-cai` で新prefixから全5実行を再検証した。前後の認証済みmodule hashが一致し、失敗・replayは全て0。前版 `kernel-default-v4` と、32層の保持hidden（最終層は使用する最後のtoken）、prefixの72状態配列／各問題の48状態配列、最終hidden・logits・確率・unknown・判断が全bit一致した。既存の最大lock問題の誤判定を改善したとは扱わない。

| 実行 | query | 改善前の命令数 | 改善後の命令数 | 削減 | 通信bytes | 時間s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix45 | 330 | 229,437,549,313 | 221,068,499,713 | 3.65% | 360,695,462 | 34.226 |
| 主問題132 / suffix87 | 352 | 405,322,437,010 | 389,142,274,450 | 3.99% | 579,259,841 | 45.005 |
| 情報不足125 / suffix80 | 352 | 358,162,787,577 | 343,284,477,177 | 4.15% | 537,098,941 | 39.417 |
| 最大変更134 / suffix89 | 352 | 418,451,724,030 | 401,899,603,710 | 3.96% | 591,305,752 | 47.604 |
| cacheなし132 | 501 | 593,293,161,174 | 568,743,949,014 | 4.14% | 885,497,502 | 83.386 |

query数と通信量は前版と同じ。初回の主問題はprefix準備330＋推論352＝682 query。主問題の最大query命令数4,167,348,841、観測heap43,450,368 bytes。handler命令数はCandid decode/encodeを含まない。通信量はCandid request＋replyでHTTP等を含まない。時間は単発の実測で、速度改善とは断定しない。

現在の主問題では整数射影／MLP群が約79.8%、Delta stageが約11.8%を占める。50 queryの理想的な命令予算2500億へさらに約35.8%、32 queryの1600億へ約58.9%の削減が必要で、実際にはquery統合・メッセージサイズ制約もある。この計算は各queryを上限まで詰められると仮定した下限であり、達成予測ではない。

既定Wasm SHA256: `ec3ebc93ae0b3b0cde8faff3e9f40deaec21f0e786fbe05e6e57ac45a4589e08`。Rust33 tests、Python40 tests通過。生成物はgitignore対象、Laya・mainnet・Git remoteは変更していない。既定featureには、採用しなかった[事前変換Strassen](STRASSEN_PREPACKED.md)とtoken単位scaleは含めていない。

再現は未使用run名で以下を実行する。

```sh
.venv/bin/python scripts/validate_terminal_readout.py --canister <専用local ID> --run-name <新しいrun名> --baseline kernel-default-v4
.venv/bin/python scripts/summarize_kernel_unroll.py --run-name <同じrun名> --baseline kernel-default-v4
```

今回の生測定は `artifacts/delta-lanes-v1-*`、検証付き集計は `docs/delta-lanes-v1-summary.json`。単体比較は `scripts/check_delta_lanes.py` と `artifacts/delta-lanes4/{before,after}`。より大きな係数共有の候補を作る `scripts/generate_delta_group.py` も追加したが、32 vector候補は未測定・未採用。
