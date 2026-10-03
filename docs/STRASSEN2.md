# 二段の整数Strassen診断

2026-10-03。単段のC64候補は主Q投影1.61%減に留まり、一律採用していない。50 queryに必要な全体削減を狙い、4 token×4出力とblock256の4分割を二段で因数分解した。64個のK64整数内積を49個へ減らす。内積数23.4375%減は、全handler命令の削減率ではない。

生成script `scripts/generate_strassen2.py`は全16出力の49積の係数を記号的に展開し、元の4×4行列積との恒等性を確認する。元入力[-127,127]・元重み[-128,127]、導出入力の絶対値<=508・導出重み<=512。64項の積和と各再構成の三角不等式による上界はI32範囲内であることを生成時に検証する。各block256の整数dotを先に復元し、元のF32 activation scale、row scale、block加算順を保つ。追加量子化はない。

native test2件は1/7/8/32/45/64/80/87/88/89/132 token、符号付き極値・0・複数blockの異なるscale、および固定coef strideを保つrow prefixで元INT8 kernelと全出力bit一致。Wasmでは入力と固定重みの係数生成をSIMD化し、導出値は出力tile間で共有する。R<=32のK64 leafは32 row quartetへ入力loadを共有する。再構成とscale/storeは明示展開する。

診断は元INT8・採用配置・係数を同時保持する。係数の容量は元weight byteの6.125倍。今回の実tensor8192×2560では128,450,560 Bで、全モデルのheap常駐配置を証明しない。計算で改善した場合も、固定係数のstable配置とquery内の小さいtile読出し、または小tileの一度展開を別に実測する。中間状態をupdateへ移す案ではない。質問の推論は通常query、準備だけowner updateを使う。

実装`scripts/strassen2_bench`、比較script`scripts/check_strassen2.py`。専用診断canisterで22通常queryを実行し、独立native・既存配置・候補のdigestは全て一致した。主87-token Q投影は1,049,194,171→1,051,365,345命令（0.2069%増）、prefix45-tokenは10.5810%増。80-tokenは0.4510%減、132-token部分出力は0.7585%減で、形状によって悪化するため採用しない。全モデルの判断・query数の検証は行っていない。

生結果は`artifacts/strassen2/check/report.json`。この比較は同じ診断module内で行い、直接rustcビルドの固定flagsと依存hashは`artifacts/strassen2/cached-build/report.json`に保存した。Cargoによる全推論moduleと同じ最適化結果とは扱わない。準備65 updateは合計1,486,667,666命令、固定診断payload170,459,136 B。生成物はGit ignore対象。
