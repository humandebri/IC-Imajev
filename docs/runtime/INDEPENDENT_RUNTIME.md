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
- 161件のsnapshotでfacts・task JSON bytes・処理経路・圧縮結果が一致した。161件用の18種類と600〜660用の6種類のモデル入力token列も完全一致。このsnapshot比較は入力の一致までを対象とする。
- Rust 49テスト、main統合後のPython 37テスト、canisterのWasm target checkが通った。

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

mainへ統合された512token対応についても、比較元builderが生成したruntimeとcompile設定をそのまま引き継ぐ。typed token carry・adaptive token tilesを失わず、INT8 kernel部分だけ差し替える。古いbuilderのreport形式にも対応する。

ビルドは成功し、34bodyの一致を確認した。Wasm全体のhashは変わる。出力先のsource pathもWasmへ含まれるため、ビルド先を固定して証拠を作る。build reportの`full_inference_verified: false`はビルド時点の記録として保持し、追加の全層比較を別の証拠へ記録する。

## 全32層の追加検証

`artifacts/laya-removal-20261008/prefix27-current`の変更前・変更後を専用ローカルcanisterで比較した。同じ4.7GB pack、重み721個、固定prefix27、同じ入力を使い、各版で84tokenと86tokenを最後の判定まで32queryで新規実行した。queryのreplayは0。最終hidden・保存した途中hidden/state・判定・logits・確率はbit一致した。保存した138個の応答frame・attention hidden・KVのbytesも一致した。

|入力|総命令数の変化|Candid通信量|観測した最大heapの増分|
|---|---:|---|---:|
|84token|−0.0040%|一致|655,360 bytes|
|86token|+0.0157%|一致|851,968 bytes|

各queryの命令数比も測定表へ記録した。最大query命令数は両版とも5B未満で、実際のIC queryも成功した。命令数はhandler counterでCDKのdecode/encodeを含まない。経過時間は各入力・各版1回の観測値で、latencyの改善保証としては扱わない。全層の計算量が完全に同じ、または全経路で減るという結果ではない。

証拠は`artifacts/laya-removal-20261008/full-model-proof-4gib/comparison.json`と`raw-output-comparison.json`。専用canisterはstop/delete済み。共有ローカルcanisterと公開canisterは変更していない。この追加検証は84/86tokenのquery経路を対象とし、512tokenのpaid推論は再実行していない。新しいmoduleへ公開版を切り替える場合はfrontend実行計画のmodule確認と対象経路の再検証を別途行う。

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
  --directory artifacts/laya-removal-20261008/prefix27-current
```

このbuild directoryは実測済みのため、再ビルド時は新しい出力先を指定する。

全層比較用の`scripts/check_independent_full_model.py`は、作成receiptを持つ専用ローカルcanisterでのみ実行する。4.7GB packをupload/hash検証し、比較元moduleをinstallしておく。既存の基準canisterと同じWasm memory limit 4GiBが必要で、開始前に検査する。共有canisterとmainnet targetを拒否し、選択されたlocal endpointとreceiptの一致も確認する。初回の3GiB設定では比較元の重み準備中にIC0539となった。その専用canisterはstop/delete済みで、失敗は`artifacts/laya-removal-20261008/full-model-proof/comparison.json`に保存した。
