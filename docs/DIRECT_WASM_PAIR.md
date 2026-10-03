# 整数積和をWasmで直接生成する診断

2026-10-03。全tokenで読み込んだ重みを共有し、整数加算の順序を明示した標準SIMDの関数を実装した。全モデルへ未採用。Rustのbalanced/left-deep加算が同じ命令へ最適化されたため、Wasmのoperand stackに積和を保持する経路を直接生成して比較した。

固定重みは元INT8のK4/output2配置をupdateで一度だけ準備する。queryのinput整形は一度だけ。K256内の32出力に必要な重みを読み込みI16へ展開した後、全実tokenで共有する。I32加算の結合順序は変更するが、最終block dotの絶対値は256×127×128以下でoverflowしない。`generate_wat_pair.py`は全32出力のlane mappingと全256積の復元を検査する。元F32 scaleの2回の掛け算とblock加算順序は維持する。

`wasm_patch`は診断moduleの名前付き関数だけを置き換える。9個のI32引数・返値なしの型を検査し、他のsectionとfunction bodyを保存する。Wasm Validatorを通し、標準SIMDのみの関数を実際のlocal canisterへinstallした。元module、WAT、置換body、結果moduleのhashを`patch-report.json`に記録する。通常Cargoで未置換のmoduleは有効な推論結果を返さない。関数のplaceholderは全出力範囲をvolatile writeし、全ABI引数を必要とする。最初のplaceholderは引数と書込範囲が実装と合わず出力不一致となり、その結果を性能改善として扱わない。失敗moduleと結果は`first-patch`に保持した。

修正版は全22 queryで独立native・旧kernel・直接生成kernelの出力digestが一致した。ただし命令数は実入力の全5条件で増えたため採用しない。全queryで前後のmodule/source hashが一致。nativeのshape/extremes/scale test2件も通過したが、native tests単独でWATを検証したとは扱わない。

module `2f7aab88801881ac20d7186b64b6793184633583ea6192e594f0011a3d62ca29`、4,934,583-byte Wasm。専用canister `7hukf-2d777-77775-aaakq-cai`。rawは`artifacts/wat_pair/check-v2/report.json`、ビルドは`cached-build/report.json`、注入は`patch-report.json`、固定sourceは`validated-source.zip` / `validated-source-hashes.json`。生成物はGit ignore対象。

計測counterはdecode・量子化・query内の整形・射影を含み、digest/CDK Candid・LoRAを含まない。診断は元重み、旧配置、候補配置を同時保持する。全モデルのheap配置・判断精度・通信削減・query数についての改善証拠ではない。主canister、Laya、Git remoteは変更していない。

| 条件 | tokens | 旧命令 | 直接生成命令 | 変化 |
|---|---:|---:|---:|---:|
| prefix | 45 | 543,136,819 | 560,022,972 | +3.1090% |
| 617 | 87 | 1,044,316,360 | 1,065,965,905 | +2.0731% |
| insufficient | 80 | 962,099,208 | 981,646,097 | +2.0317% |
| maximum | 89 | 1,067,862,040 | 1,090,112,417 | +2.0836% |
| normal | 132 | 798,455,508 | 812,072,189 | +1.7054% |

固定準備65 updateは合計1,441,971,079命令。全推論の改善は示せておらず、主問題は67 query、50/32回には未達。

再現は`generate_wat_pair.py`を先に実行し、`build_wat_pair_cached.py --runtime <固定rlib>`で未置換Wasmを作る。`imajev-wasm-patch <未置換Wasm> <kernel.wat> <診断Wasm>`で名前と型を検査して置換し、診断canisterへinstallして`check_wat_pair.py --canister <ID>`を実行する。`kernel.wat`自体は生成物としてignoreし、検証時の固定コピーをソースarchiveへ含める。
