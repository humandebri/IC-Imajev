# Laya由来実装の置き換えと検証

2026-10-08。基準は`ede51e051f689bf08918843cf34a54563b5afbda`。Laya専用の比較driver、checkpoint読込adapter、比較文書、patchを削除した。Imajevの保存済み検証入力と過去runのhashは保持する。

## 実装

INT8 tileを、block256の整数dotと順序を固定したF32 scaleという仕様から生成する実装に置き換えた。整数集約は4出力をまとめ、重みとscaleはWasm localで共有する。`scripts/generate_int8_tile.py`から`crates/imajev-runtime/src/int8_tile.rs`を再生成できる。古いrow/column macroは使用しない。

符号付きI8重みの`-128`も扱う。activationは`[-127,127]`、blockの絶対和は最大`256*127*128`でI32に収まる。F32では各blockの`(dot*activation_scale)*weight_scale + previous_sum`を維持し、blockを昇順に処理する。whole-column経路は整数を全列にわたって集約してからscaleする。

proposal評価は型検査・rendering解析・証拠gate・exactな単位変換を置き換えた。Imajev driverが使う関数を維持し、過去のprompt sweepとLaya classifier CLIは廃止した。質問文・field説明は入力を保持するためデータとして残す。`prompt_contract.json`と保存済みbenchmark入力の出所を記録し、MIT noticeを維持する。

入力に影響する新しいcontractデータも各driverのsource hashへ追加した。凍結済みrunのmanifestを書き換えず、旧runは当時の保存済みソースで検証する。

## 測定結果と採用範囲

[測定表と証拠hash](comparison/independent-verification.json)に記録する。旧・新は同一のfixtureソース、Cargo.lock、Rust compiler、release設定を使用した。合成投影は32出力、token数1〜132の全整数、入力幅256・512・2560・9216、通常・最大振幅・ゼロ・subnormal・重み-128の正負極値を含む6patternで比較した。

- Wasm/V8で22,176ケースの出力F32 bytesが完全一致した。
- ローカルICで84条件を各3回測定し、V8で確認した出力hashとも一致した。handler内のquantize＋projectionを`performance_counter(0)`で測る。測定84条件のheap page数は旧・新で同じ。診断replyは両版24bytes。
- 161件のsnapshotでfacts・task JSON bytes・処理経路・圧縮結果が一致した。161件用の18種類と600〜660用の6種類のモデル入力token列も完全一致。新規のモデル推論は実行していない。
- Rust 49テスト、Python 35テスト、canisterのWasm target checkが通った。

|互換・実験経路|測定範囲の命令数変化|
|---|---:|
|8-column block256|−25.30%〜+0.31%|
|16-column|−30.01%〜+0.24%|
|16-column/token48|−30.01%〜+0.20%|
|fused dot/scale|−30.01%〜−0.47%|
|balanced44|−30.01%〜+0.20%|
|column32/balanced|−30.01%〜+0.16%|
|whole-column token scale|−1.30%〜+1.46%|

初期の単純な配列/loop案は最大約3.9倍に増えたため破棄した。最終版も一部互換経路で増加が残る。全経路の計算量が減るという結果ではなく、この表を全モデルの命令数やcyclesへ換算しない。V8 timingは別のhost測定であり、ICの処理時間や実cyclesの保証には使わない。

現在の公開Prefix27 runtimeは`experimental-paired-only`と検証済みの投影34個を使う。上記の互換tileを公開経路へ追加採用しない。この変更で公開canisterのupgrade・モデルupload・frontend deployは行っていない。

## Prefix27のビルドと公開条件

`scripts/build_independent_prefix27.py`は検証済みの凍結optimized runtimeを新しいartifactへコピーし、由来kernel部分だけ置き換えてruntimeを再コンパイルする。quantizerの最適化やrank49等の既存変更を維持し、34個の投影bodyは検証済みparentとhash一致を確認して適用する。元artifactは変更しない。source/dependencyのhash検証と、ビルド中のsource変更検出を行う。

ビルドは成功し、34bodyの一致を確認した。Wasm全体のhashは変わる。出力先のsource pathもWasmへ含まれるため、ビルド先を固定して証拠を作る。ビルド成功と投影body一致は、32層の最終hidden・判定・確率や全query命令数の実測一致を証明しない。`full_inference_verified`はfalseのままとする。

公開前には新しいWasmで同一pack/cache/inputを使う全層A/Bを行い、最終hidden・判定・確率のbit一致、各query命令数、通信量、メモリ、経過時間を確認する。module hashに結びつくfrontend実行計画も再検証が必要。この条件を満たすまで公開版は更新しない。

## 再現

基準ソースは`git archive ede51e0`で新しいartifact内に保存する。baseline harnessは現行`scripts/independent_int8_bench`の同じfixtureを使い、依存の`imajev-runtime`パスのみ基準ソースへ向ける。

```sh
python3 scripts/generate_int8_tile.py
cargo build --offline --release --target wasm32-unknown-unknown \
  --manifest-path scripts/independent_int8_bench/Cargo.toml \
  --target-dir artifacts/laya-removal-20261008/candidate-target
node scripts/check_independent_int8.mjs \
  artifacts/laya-removal-20261008/baseline-target/wasm32-unknown-unknown/release/independent_int8_bench.wasm \
  artifacts/laya-removal-20261008/candidate-target/wasm32-unknown-unknown/release/independent_int8_bench.wasm \
  artifacts/laya-removal-20261008/node-final-extremes.json
```

`scripts/measure_independent_int8.py`へそのproof、before/after Wasm、新しいoutput directoryを渡すと、loopbackのローカルnetwork上に専用canisterを2個作成して測定し、その2個だけをstop/deleteする。既存の推論canisterは操作しない。

```sh
python3 -B scripts/build_independent_prefix27.py \
  --directory artifacts/laya-removal-20261008/prefix27-verified
```

このbuild directoryは実測済みのため、再ビルド時は新しい出力先を指定する。
