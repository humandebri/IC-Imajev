# INT8事前変換Strassenの実測

固定済みの実重み（layer 3 Q投影、8192×2560）を使い、従来INT8と事前変換Strassenを同一の診断用ローカルcanisterで比較した。Strassenは隣接するtoken／出力行と、元の256列block内の128列を組にする。七つの変換済み行列をINT8で保持し、INT8範囲を超える部分だけ疎な±256補正を持つ。元のblock scale、F32加算順、BF16境界は保存した。

| token数 | 従来の命令数 | 事前変換Strassenの命令数 |
| --- | ---: | ---: |
| 1 | 72,711,518 | 403,109,345 |
| 7 | 185,363,746 | 418,790,020 |
| 8 | 186,406,712 | 421,400,066 |
| 32 | 592,028,124 | 1,263,770,611 |
| 64 | 1,106,459,786 | 2,263,587,028 |
| 87 | 1,511,474,475 | 3,042,874,526 |

全6形状で従来出力と全bit一致し、nativeの独立したscalar実装とも一致した。しかし87 tokenで命令数は約2.01倍に増えた。stable readは21,004,288→38,278,204 bytes、観測heapは574→814 pages。積和回数だけの削減は、この実装の変換・補正・復元・データ配置費用を相殺できない。この候補は既定経路に採用しない。

測定はbase Q投影のみであり、LoRA・readout・全モデルの精度やquery数の改善を示さない。準備にはupdate、数値計算には通常queryを使った。59,249,724-byte部分packの準備時間は5.411秒（全モデルのupload時間ではない）。この比較の期間中は、主canisterの固定4B packと既定Wasmを変更していない。後のDelta改善に伴う既定Wasmの変更は[別の検証](DELTA_SIMD.md)に記録した。Layaは変更していない。

診断canister: `4fbx2-kt777-77775-aaabq-cai`、測定Wasm SHA256 `9b420b0b54d6234e1bd572f8fd49709d13e8f1591335bf62926bb68788bada14`、部分pack SHA256 `0aa00943520728ed0a610e0120402466e8d9778b4fb7f6d516aec55a1f2ab9b6`。前後の認証済みmodule hashを確認した。元packのSHA256は `364e3aa3f69a61a69e6df62e3e2a71d4f55457f1a8e1c16934aefc6130f04c84`。

実装は `experimental-strassen-prepacked` featureに隔離した。準備コード `scripts/prepare_strassen_pack.py`、測定コード `scripts/check_prepacked_strassen.py`、生データ `artifacts/strassen-prepacked/probe/report.json`。生成した重み・Wasm・測定データはgitignore対象。feature有効のRust testsは33件通過。Wasmが使うSIMD演算とnative scalarを実canisterで比較した結果は上表の6形状であり、native unit testだけをWasmの証拠として扱わない。
