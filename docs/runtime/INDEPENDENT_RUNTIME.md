# Laya由来実装の置き換えと検証

2026-10-08。基準は`ede51e051f689bf08918843cf34a54563b5afbda`。Laya専用の比較driver、checkpoint読込adapter、比較文書、patchを削除した。Imajevの保存済み検証入力と過去runのhashは保持する。

## 実装

INT8 tileを、block256の整数dotと順序を固定したF32 scaleという仕様から生成する実装に置き換えた。整数集約は4出力をまとめ、重みとscaleはWasm localで共有する。`scripts/generate_int8_tile.py`から`crates/imajev-runtime/src/int8_tile.rs`を再生成できる。古いrow/column macroは使用しない。16/32列、balanced44、dot-scale、whole-token scaleの旧選択API・Cargo feature・実装を削除し、非prepared投影はこのblock256の1方式へ統一した。ベンチマークの呼び出しも更新した。

符号付きI8重みの`-128`も扱う。activationは`[-127,127]`、blockの絶対和は最大`256*127*128`でI32に収まる。F32では各blockの`(dot*activation_scale)*weight_scale + previous_sum`を維持し、blockを昇順に処理する。native実装は同じblock256仕様のscalar参照で、prepared-pair投影の数値検証にも使う。whole-token scaleの量子化実験は残さない。

proposal評価は型検査・rendering解析・証拠gate・exactな単位変換を置き換えた。Imajev driverが使う関数を維持し、過去のprompt sweepとLaya classifier CLIは廃止した。質問文・field説明は入力を保持するためデータとして残す。`prompt_contract.json`と保存済みbenchmark入力の出所を記録し、MIT noticeを維持する。

入力に影響する新しいcontractデータも各driverのsource hashへ追加した。凍結済みrunのmanifestを書き換えず、旧runは当時の保存済みソースで検証する。

## 測定結果と採用範囲

[測定表と証拠hash](comparison/independent-verification.json)に記録する。旧・新は同一のfixtureソース、Cargo.lock、Rust compiler、release設定を使用した。合成投影は32出力、token数1〜132の全整数、入力幅256・512・2560・9216、通常・最大振幅・ゼロ・subnormal・重み-128の正負極値を含む6patternで比較した。

- 最終整理後のWasm/V8比較3,168ケースすべてで、旧block256経路と出力F32 bytesが完全一致した。
- ローカルICで12条件（token数1/69/87/132、幅256/2560/9216）を各3回測定し、V8で確認した出力hashとも一致した。handler内のquantize＋projectionを`performance_counter(0)`で測る。命令数は全12条件で0.13〜6.01%減少し、heap page数は同じだった。診断replyは両版24bytes。専用canister2個はstop/delete済み。
- 161件のsnapshotでfacts・task JSON bytes・処理経路・圧縮結果が一致した。161件用の18種類と600〜660用の6種類のモデル入力token列も完全一致。このsnapshot比較は入力の一致までを対象とする。
- 整理後のRust 47テスト、Python 33テスト、APIを更新した9ベンチマークの全target check、canisterのWasm target checkが通った。

以前の7方式を残した版の22,176ケース・84条件や全層比較は、測定表の`historical_before_cleanup`へ移した。当時のmodule/source hashを維持し、現在の実装の測定としては使わない。単体カーネルの改善率を全モデルの命令数やcyclesへ換算しない。V8 timingは別のhost測定であり、ICの処理時間や実cyclesの保証には使わない。

現在の公開Prefix27 runtimeは`experimental-paired-only`と検証済みの投影34個を使う。現行prepared-pair投影を維持し、未使用の旧方式だけを削除した。この変更で公開canisterのupgrade・モデルupload・frontend deployは行っていない。

## Prefix27のビルドと公開条件

`scripts/build_independent_prefix27.py`は検証済みの凍結optimized runtimeを新しいartifactへコピーし、由来kernel部分を置き換え、廃止したwhole-token処理・module・compile flagを除いてruntimeを再コンパイルする。quantizerの最適化やrank49等の既存変更を維持し、34個の投影bodyは検証済みparentとhash一致を確認して適用する。元artifactは変更しない。source/dependencyのhash検証と、ビルド中のsource変更検出を行う。

mainへ統合された512token対応についても、比較元builderが生成したruntimeとcompile設定をそのまま引き継ぐ。typed token carry・adaptive token tilesを失わず、INT8 kernel部分だけ差し替える。古いbuilderのreport形式にも対応する。

ビルドは成功し、34bodyの一致を確認した。Wasm全体のhashは変わる。出力先のsource pathもWasmへ含まれるため、ビルド先を固定して証拠を作る。build reportの`full_inference_verified: false`はビルド時点の記録として保持し、追加の全層比較を別の証拠へ記録する。

## 全32層の追加検証

`artifacts/laya-removal-20261008/prefix27-pruned`の変更前・変更後を専用ローカルcanisterで比較した。同じ4.7GB pack、重み721個、固定prefix27、同じ入力を使い、各版で84tokenと86tokenを最後の判定まで32queryで新規実行した。queryのreplayは0。最終hidden・保存した途中hidden/state・判定・logits・確率はbit一致した。保存した138個の応答frame・attention hidden・KVのbytesも一致した。

|入力|総命令数の変化|Candid通信量|観測した最大heapの増分|
|---|---:|---|---:|
|84token|+0.0093%|一致|655,360 bytes|
|86token|+0.0114%|一致|851,968 bytes|

各queryの命令数比も測定表へ記録した。最大query命令数は両版とも5B未満で、実際のIC queryも成功した。命令数はhandler counterでCDKのdecode/encodeを含まない。経過時間は各入力・各版1回の観測値で、latencyの改善保証としては扱わない。全層の計算量の評価はこの2入力に限る。

証拠は`artifacts/laya-removal-20261008/pruned-full-model-proof/comparison.json`と`raw-output-comparison.json`。専用canisterはstop/delete済み。共有ローカルcanisterと公開canisterは変更していない。この追加検証は84/86tokenのquery経路を対象とし、512tokenのpaid推論は下記の追加検証で再実行した。新しいmoduleへ公開版を切り替える場合はfrontend実行計画のmodule確認と対象経路の再検証を別途行う。

## update経路の追加検証

削除前後の同じmoduleを専用ローカルcanisterで比較し、`infer`の課金付きupdateを各版14条件で実行した。合計token数28・84・86・116・117・256・257・512の8点でbulk/分割・Attention分割の境界と上限を検証した。選択肢数2〜7は28tokenの入力で確認した。全条件で32層のhidden/state hash、最終hidden、判定・logits・確率のF32 bitsが一致した。各条件のworker数も一致し、512tokenは34回で完了した。

命令数の観測差は-0.0480%〜+0.0129%。削除後の最大worker checkpointは33,524,762,342命令、最大heapは4,204,855,296bytesで、すべて実際のreplicated updateが成功した。各版・各入力1回の測定で、upgrade後には旧receiptを保持している。小さな差を純粋なkernel性能の差やlatency改善と解釈しない。checkpointはCDKの前処理を含み、記録・返信の末尾を除く。

84・86tokenのupdateは保存済みquery参照とも照合した。query参照が保存するhiddenは31層（融合された30層目を未出力）、stateは32層で、最終hidden・判定・確率も一致した。updateの削除前後の比較は30層目を含む全32層を対象とする。owner用`update_infer_start`/`update_infer_continue`も84tokenで実行し、paid/queryと照合した。完了後の不正continueと空suffixは拒否された。

quoteどおりのcycles受領、余剰返却、同じID/入力の再送での再課金なし、upgrade直後の512token receipt再送を確認した。空suffix・prefix不一致・513token・古いquote・不足cycles・ID競合は課金前に拒否され、添付cyclesがすべて返った。外部からの`inference_step`はself-onlyで拒否された。canisterのdefault-feature Rustテスト22件も通った。公開moduleにないfault注入によるworker trap/返金は、この実行比較では試していない。

証拠は`artifacts/laya-removal-20261008/update-proof-reference-v2/comparison.json`。専用canisterとrelayはstop/delete済みで、共有・公開canisterを変更していない。最初のrelay作成時のローカルcycles不足と、融合queryの未出力hiddenを参照した検証スクリプトの失敗はそれぞれ`update-proof`/`update-proof-funded`に保存した。両方とも専用canisterの後始末済みで、ランタイムの推論失敗ではない。ローカルのテストICPだけでcyclesを補充し、参照範囲を修正して再実行した。

ローカル保存の検証スクリプト（実行時のパスは`scripts/check_independent_update.py`、source hashは測定表に記録）はdedicated receipt、local endpoint、module/tool/source hashを検査する。比較元moduleをinstallし、4GiBのWasm limitとupload済みpackを用意してから、新しいevidence directoryで実行する。query参照は今回の保存済み84/86token fixturesを使う。未測定のtoken数や新しいfrontend query実行計画の合格を、このupdate検証から推定しない。

## 再現

基準ソースは`git archive ede51e0`で新しいartifact内に保存する。baseline harnessは同じ入力生成fixtureを使い、依存の`imajev-runtime`パスを基準ソースへ向ける。現行harnessを基準ソース向けにもビルドし、測定ラッパー・featureを揃えて比較する。

```sh
python3 scripts/generate_int8_tile.py
cargo build --offline --release --target wasm32-unknown-unknown \
  --manifest-path scripts/independent_int8_bench/Cargo.toml \
  --target-dir artifacts/laya-removal-20261008/prune-candidate-target
node scripts/check_independent_int8.mjs \
  artifacts/laya-removal-20261008/prune-baseline-target/wasm32-unknown-unknown/release/independent_int8_bench.wasm \
  artifacts/laya-removal-20261008/prune-candidate-target/wasm32-unknown-unknown/release/independent_int8_bench.wasm \
  artifacts/laya-removal-20261008/pruned-node-fair.json
```

`scripts/measure_independent_int8.py`へそのproof、before/after Wasm、新しいoutput directoryを渡すと、loopbackのローカルnetwork上に専用canisterを2個作成して測定し、その2個だけをstop/deleteする。既存の推論canisterは操作しない。

```sh
python3 -B scripts/build_independent_prefix27.py \
  --directory artifacts/laya-removal-20261008/prefix27-pruned
```

このbuild directoryは実測済みのため、再ビルド時は新しい出力先を指定する。

全層比較用の`scripts/check_independent_full_model.py`は、作成receiptを持つ専用ローカルcanisterでのみ実行する。4.7GB packをupload/hash検証し、比較元moduleをinstallしておく。既存の基準canisterと同じWasm memory limit 4GiBが必要で、開始前に検査する。共有canisterとmainnet targetを拒否し、選択されたlocal endpointとreceiptの一致も確認する。初回の3GiB設定では比較元の重み準備中にIC0539となった。その専用canisterはstop/delete済みで、失敗は`artifacts/laya-removal-20261008/full-model-proof/comparison.json`に保存した。
