# 検証済みINT8 spanの重複確認を省く候補

2026-10-03。採用版`36c04a57…`は主問題67 query。50 queryには未達。前回の二項Strassen診断で内部アドレス確認の反復を除去すると、主Q投影は1.6108%減となったが、I16係数は3.5倍になり45 tokenも回帰した。今回は採用済みのINT8 K4/2出力配置を保ち、同種の重複確認だけを省く。

変更は`output_pairs_simd.rs`と生成元`scripts/generate_output_pairs.py`の2ファイルだけ。私有unsafe関数のweight pointer、input pointer、scale pointerの添字計算をwrapping演算にした。整数dot・F32 scale順・block加算順・row/token dispatch・外部metadata検証は変更しない。Vec長とpaddingを確立するpublic constructorと、private callerのshape検証は維持する。

`PreparedPairs::from_le_bytes`は元重み要素数30,000,000以下、行数32の倍数、cols256の倍数とscaleを検証する。`PackedView`は偶数start・8行境界と固定tensor内の範囲を保証する。呼び出し側はrowsとscale長、q.rows<=132、cols一致、出力要素上限を検証し、最後の32行未満tileのみ32行へpaddingする。qの不透明型とreal-token dispatchにより、input/scaleのoffsetも各allocation内に収まる。これらの内部offsetは32bit usizeでも実際にはwrapしない。wrapping演算の狙いは、この保証後に各blockで再実行するoverflow分岐を省くことに限る。

候補のsourceは`artifacts/bounded-address/source.zip`、55 core hashは同directoryの`source-hashes.json`。採用版と差のあるcore fileは上記2件だけ。Wasm測定と全モデルの5条件比較を完了するまでは採用を主張しない。生成再現を確認済み。実測結果は完了後に追記する。


## 単体の通常query比較

新診断module `6413b63bc4846dfe3ec22fb4638d904b9ea14c2d9343827b52156d8facb5d4f7`を専用`7hukf-2d777-77775-aaakq-cai`で測定。生成WasmのR44 `accumulate`からoverflow panic call104、BrIf144、I64Mul37を除去し、いずれも0になった。これは静的opcode数で、命令の効果は実queryで確認した。

旧診断V4の採用版INT8投影と新診断の採用配置投影を、同じ保存入力・同じweight・同じ量子化結果で比較。実5条件45/87/80/89/132 tokenはそれぞれ0.4721/0.4768/0.4805/0.4759/0.4727%減。主Qは1,049,769,418→1,044,763,594命令。合成1/7/8/32/64/88 tokenは3.1105/2.2505/0.8768/0.4874/0.4918/0.5243%減。全ケースの量子化counterは旧診断と同じ。22通常queryの出力は独立native oracleと一致。旧・新moduleは同時保持ではなくbookendされた別実行で、input digestと量子化counterの一致を確認した。

証拠`artifacts/bounded-address/diagnostic-check/report.json`と`diagnostic-comparison.json`、`diagnostic-opcodes.json`。native67 unit・6 integration・3 compile-failと生成再現が通過した。単体の差は全モデルの改善率でもquery数の削減でもない。全モデル検証は続行中。

全推論Wasm候補`ff7c039d…`はビルド済みだが未採用。`current-source.zip`には既存のDelta cache形式・返信境界等のレビュー修正も含まれ、最初の2-file snapshotとは区別する。主canisterへのupgradeは自動承認レビューが全体検証前・upgrade hook未確認を理由に拒否し、実行されていない。hookを読み、packはstable memoryで保持し、owner/manifest/readyが復元され、heap cacheは再準備される実装と確認した。主moduleは`36c04a57…`のまま。候補の全体検証は専用canisterで行ってから採用を判断する。
