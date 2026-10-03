# 二段のWinogradによる共有積と共有加算

2026-10-03。主問題50 queryへの削減を目指す未採用候補。以前の二段Strassenは64内積を49個にしたが、主Qでは0.2069%命令増だった。今回、積だけでなく再構成の中間加算も共有する二段Winogradを実装した。性能はWasm実測を待つ。

4 token×4出力、元block256のK64分割を維持する。各2×2段で7積を生成し、U2=P1+P6、U3=U2+P7、U4=U2+P5を一度だけ計算して各出力へ使い回す。固定重みの変換はモデル準備update、入力変換は量子化の後に通常queryで一度だけ行い、全出力tileで共有する。元のINT8値、F32 scaleの掛け算とblockの加算順序を変えない。追加量子化はない。

`generate_winograd2.py`は全16出力について49積の係数を展開し、元の行列積との恒等性を確認する。導出入力の絶対値は最大2,032、導出重みは2,048でI16範囲内。K64 leafの最大絶対和は266,338,304でI32範囲内。再構成は元変数の多項式として各加算後の係数を展開し、相殺も含めて上界を検査する。nativeのflat加算とWasmの共有U nodeの両方を確認する。flat中間の最大上界は86,351,872。各元blockのdotをI32として完全に再構成してから元のF32処理へ渡す。

native 2 testsは、1/7/8/32/45/64/80/87/88/89/132 tokens、極値・0・異なるblock scale、非ゼロrow prefixの出力を既存整数kernelとbit比較して通過した。これだけでWasmの性能や全モデルの判断精度を証明したとは扱わない。

固定係数の容量は元weight byteの6.125倍。単一Qでは128,450,560 bytes。全モデルを同じ配置で4 GiB heapへ置ける証拠はない。改善が実測できた場合も、重み配置と固定準備費用を別に検証する必要がある。クライアント保持の推論状態をcanister updateへ移す実装ではない。

実装は`scripts/winograd2_bench`、生成は`generate_winograd2.py`、Wasmビルドは`build_winograd2_cached.py`、通常query測定は`check_winograd2.py`。nativeログは`artifacts/winograd2-native.log`。ビルド成果物・raw結果はGit ignore対象。主canisterとLayaは変更していない。

Wasmビルドは通常opt3/thinLTO/cgu1/overflow-checksありで完了。module `70991a7b8e3599f06c3c980694a75a19f276080f7404bf3c2d2c1e7f4b4cbbc8`、8,558,419 bytes。固定依存と全sourceのビルド前後hash一致を`artifacts/winograd2/cached-build/report.json`に記録した。測定には以前のpair-factor専用canister `5tkpr-7d777-77775-aaaeq-cai` を再利用する。以前の測定は固定Wasmの歴史的証拠であり、現在の同canisterのmoduleとは区別する。

## 通常queryの結果（未採用）

全22 queryでnative・旧kernel・Winogradの出力digestが一致。主87-token Qは1,044,316,347→1,047,741,505命令（0.3280%増）、prefix45は10.7287%増、最大89は2.2271%増。80-tokenは0.3258%減、132-token部分出力は0.6305%減に留まり、形状依存の悪化があるため採用しない。境界1/7/8/32でも増加した。加算を共有しても、増える変換・再構成等の費用を上回れなかった。

固定準備65 updateは合計1,495,515,026命令。rawは`artifacts/winograd2/check/report.json`、固定ソースは`validated-source.zip` / `validated-source-hashes.json`。moduleとsourceの実行前後一致を確認した。counterは量子化・query内の入力変換・射影を含み、digest/CDK Candid・LoRAを含まない。全モデルや判断精度の測定ではない。主問題67 query・50/32 query未達は変わらない。
