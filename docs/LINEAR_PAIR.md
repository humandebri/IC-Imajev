# 全tokenで重み展開を共有する候補の実測

2026-10-03。採用済み87-tokenの44+43分割などで繰り返す固定INT8重みのI16展開を、全実tokenで共有する診断を実装した。入力のI16複製もqueryで一度だけ行い、固定重みのK4/output2配置はupdateで一度だけ準備する。通常queryの動的ループで45/80/87/89/132行を処理し、整数加算をbalanced treeとleft-deep treeの2種類で比較した。INT32のblock256和の絶対値は256×127×128以下で、整数の結合順序変更にoverflowはない。F32 scale、blockの加算順序は変えない。

**主条件では増えたため、全推論へ採用しない。** 重みの準備を減らしても動的ループの命令が上回った。2つの加算treeはLLVM最適化後に同じ実測命令数だった。

| 条件 | tokens | 従来命令 | 候補命令 | 変化 |
|---|---:|---:|---:|---:|
| 共通prefix | 45 | 543,136,818 | 561,373,889 | +3.358% |
| 主問題617 | 87 | 1,044,316,359 | 1,071,080,022 | +2.563% |
| 情報不足 | 80 | 962,099,207 | 986,133,014 | +2.498% |
| 最大変更 | 89 | 1,067,862,039 | 1,095,405,734 | +2.579% |
| prefixなし | 132 | 798,455,507 | 816,645,250 | +2.278% |

全33通常query（5実入力と6境界入力の各3経路）でnativeの独立scalar出力とdigestが一致。境界の7 tokensだけは従来の4+2+1 tileを1ループへまとめる効果で11.08%減り、それ以外の境界は増えた。これは単一層Q baseの診断であり、LoRA、全層精度、通信削減、全推論query数についての改善を示さない。counterは入力decode・量子化・query内の整形・射影を含み、digestとCDKのCandid処理を含まない。

専用canister `7hukf-2d777-77775-aaakq-cai`、module `37f61f2e5886977148e506a3166c9a2520169696b1a64300c5cb4fa47ebda240`。固定モデル重み8192×2560は元のpackから読み、64分割uploadとsealの65 updateを行った。診断は元重み、旧pair配置、候補pair配置を保持するので全モデル4 GiBへの配置証明には使わない。最適化flags・依存rlib・source・実行前後moduleをhashで固定した。

実装は`scripts/linear_pair_bench`、生成は`generate_linear_pair.py`、実行は`check_linear_pair.py`。rawは`artifacts/linear_pair/check/report.json`、ビルド証拠は`cached-build/report.json`。生成物はGit ignore対象。主canisterとLayaは変更していない。

## 87行を静的に展開した再試行

動的ループの費用を避けるため、同じ32出力×block256の演算を87行まで静的展開する候補を追加した。native 2 testsは通過したが、通常opt3/thinLTO/cgu1のWasmコンパイルがSIGKILLとなった。LTOなし/cgu8でもSIGKILLとなり、Wasm実測には到達していない。後者の実行中RSSは約20.7 GBだった。これはコンパイル時の制約であり、canisterの実行時メモリ・命令数・精度の測定値ではない。試行ログは`artifacts/linear_pair/static-build.log`と`static-nolto-build.log`。有効なWasmの性能改善として扱わない。

現在の診断ソースにはmethod3と`--static-87`があるが、成功したビルドがないため未検証である。前の33-query診断の固定ソースは`validated-source.zip`に保存し、現在の拡張ソースと混同しない。

静的87候補はCargo feature `static-87` / builder `--static-87` の明示指定時だけ有効にした。通常の動的診断ビルドは巨大な未成功kernelを含めず、既存の再現経路を保つ。
