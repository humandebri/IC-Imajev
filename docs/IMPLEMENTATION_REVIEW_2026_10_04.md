# 2026-10-04 実装レビューと修正

今回の差分では、通常queryの所有者検査とframe境界、署名と保存checksumの役割、型付き継続入力のrequest identity、BF16/F32丸め順序、SIMD/MaybeUninitの初期化範囲、再利用重みとcodec境界、clientの保存状態、実測と未達事項を確認した。

修正した点：

- compact tailがlayer30 hiddenを返さないとき、以前の `layer-30.npy` が残り、古い生成物を今回の結果として読める。非返却時に当該生成ファイルを除去する。既存ファイルを置いた回帰試験で確認した。
- MLP carryのencodeが、検査のためだけにRust/Python双方で状態全体をdecodeし直してコピーしていた。既存のlength・finite・BF16・integer検査に加え、入力/product scaleの2区間を直接検査し、復元と破棄を省く。入力decodeの検査は維持する。
- Python Transportのframe versionにbool/floatが渡ると、整数との比較でconstructorを通過し、後のencodeで失敗する。厳密にintだけを受け付け、bridge作成前に拒否する。

検証：候補と同じfeatureでRust runtime105/canister9 unit、integration9、compile-fail doc8が通過。通常featureでもruntime48/canister5 unitとdoc2が通過。Pythonはcodec4件・tail/replay7件が通過し、保存済み実canisterのMLP stream payload315個を新encoderで再構築してすべてbyte一致した。ゼロ/負数/非有限scale、範囲外/小数/負ゼロ整数、frame versionのbool/floatも拒否した。変更したPython32ファイルの構文検査、`git diff --check`、候補Wasmのoffline release buildも通過。

ログは `artifacts/f32_k_continue/review-*.log`、ビルドは `artifacts/prefix_codec/full-build-review-v1` に保存。raw Wasm hashは `ebaada23eb5ed5fc65dd1d366b9e86ad0c51d6c3352818ad16a6f33e7e35d027`。fail-closed演算stubを含むraw buildであり、そのままinstallする成果物ではない。

修正前のmodule `7203222a…` の全6条件・INT8列継続32条件・Delta head再利用12条件の実canister測定は保存している。今回のencoder修正後について、新しい実canister命令削減量や全6条件の再測定を行ったとは扱わない。送信byteの一致とRust/Python検証、Wasmビルドまでを今回の証拠とする。既存の最大変更の見逃しは改善していない。

全体50 queryは未達で、直近の完走graphは主62 query。次はMLP後半の完了処理を所有権付き中間状態から直接実行し、Delta前半とのquery融合を実装・実測する。生成物はignoreしたまま、主canisterとLayaに変更はない。mainnet/push/PRは行わない。

コミット後の継続作業ではレビュー修正を含む新候補 `c5ad3b17…` の全6条件も再検証した。詳細とMLP融合の45条件は [MLP_COMPLETE.md](MLP_COMPLETE.md) を参照。上のraw build・未再測定という記録はレビューコミット時点の検証範囲。
